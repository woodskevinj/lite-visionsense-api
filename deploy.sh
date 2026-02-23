#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="LiteInfraStack"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
echo "=== Checking prerequisites ==="

MISSING=0
for cmd in aws docker jq; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "  [MISSING] $cmd is not installed or not on PATH"
    MISSING=1
  else
    echo "  [OK]      $cmd"
  fi
done

if ! docker info &>/dev/null; then
  echo "  [MISSING] Docker daemon is not running — start Docker and try again"
  MISSING=1
else
  echo "  [OK]      Docker daemon"
fi

if ! aws sts get-caller-identity &>/dev/null; then
  echo "  [MISSING] AWS credentials are not configured or have expired"
  echo "            Run 'aws configure' or set AWS_PROFILE / AWS_ACCESS_KEY_ID"
  MISSING=1
else
  echo "  [OK]      AWS credentials"
fi

if [ "$MISSING" -eq 1 ]; then
  echo ""
  echo "One or more prerequisites are missing. Fix the issues above and re-run."
  exit 1
fi

echo ""
echo "=== Querying CloudFormation stack: ${STACK_NAME} ==="

# Fetch all stack outputs in one call and extract needed values
STACK_OUTPUTS=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs" \
  --output json)

get_output() {
  echo "$STACK_OUTPUTS" | jq -r --arg key "$1" '.[] | select(.OutputKey == $key) | .OutputValue'
}

CLUSTER_NAME=$(get_output "EcsClusterName")
SERVICE_NAME=$(get_output "VisionSenseEcsServiceName")
APP_ECR_URI=$(get_output "VisionSenseEcrUri")
ALB_DNS=$(get_output "AlbDnsName")

echo "  Cluster:  ${CLUSTER_NAME}"
echo "  Service:  ${SERVICE_NAME}"
echo "  ECR URI:  ${APP_ECR_URI}"
echo "  ALB DNS:  ${ALB_DNS}"

# Derive the ECR registry host from the app URI (strip /repo-name)
ECR_REGISTRY="${APP_ECR_URI%%/*}"

echo ""
echo "=== Logging in to ECR ==="
aws ecr get-login-password \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo ""
echo "=== Building and pushing image ==="
docker build --platform linux/amd64 -t "${APP_ECR_URI}:latest" .
docker push "${APP_ECR_URI}:latest"

echo ""
echo "=== Registering new task definition revision ==="

# Get the current task definition ARN from the running service
CURRENT_TASK_DEF_ARN=$(aws ecs describe-services \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME" \
  --query "services[0].taskDefinition" \
  --output text)

echo "  Current task definition: ${CURRENT_TASK_DEF_ARN}"

# Fetch the current task definition and transform it:
#   - Replace the app container image with the new ECR image
#   - Strip fields that cannot be included when re-registering
TASK_DEF_JSON=$(aws ecs describe-task-definition \
  --task-definition "$CURRENT_TASK_DEF_ARN" \
  --query "taskDefinition")

NEW_TASK_DEF=$(echo "$TASK_DEF_JSON" | jq \
  --arg app_image "${APP_ECR_URI}:latest" \
  '
  .containerDefinitions |= map(
    if .name == "app" then .image = $app_image | del(.command)
    else .
    end
  )
  | del(.taskDefinitionArn, .revision, .status, .requiresAttributes,
        .compatibilities, .registeredAt, .registeredBy)
  ')

NEW_TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json "$NEW_TASK_DEF" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)

echo "  New task definition: ${NEW_TASK_DEF_ARN}"

echo ""
echo "=== Updating ECS service ==="
aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --task-definition "$NEW_TASK_DEF_ARN" \
  --force-new-deployment \
  --query "service.deployments[0].{Status:status,Running:runningCount,Desired:desiredCount}" \
  --output table

echo ""
echo "=== Deployment triggered successfully ==="
echo "Task definition: ${NEW_TASK_DEF_ARN}"
echo "Application URL: http://${ALB_DNS}:8080"
echo "Monitor with:    aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --query 'services[0].deployments'"
