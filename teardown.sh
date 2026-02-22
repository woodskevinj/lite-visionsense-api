#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="LiteInfraStack"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
echo "=== Checking prerequisites ==="

MISSING=0
for cmd in aws jq; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "  [MISSING] $cmd is not installed or not on PATH"
    MISSING=1
  else
    echo "  [OK]      $cmd"
  fi
done

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

STACK_OUTPUTS=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs" \
  --output json)

get_output() {
  echo "$STACK_OUTPUTS" | jq -r --arg key "$1" '.[] | select(.OutputKey == $key) | .OutputValue'
}

CLUSTER_NAME=$(get_output "EcsClusterName")
SERVICE_NAME=$(get_output "EcsServiceName")

echo "  Cluster: ${CLUSTER_NAME}"
echo "  Service: ${SERVICE_NAME}"

echo ""
echo "=== Scaling ECS service to 0 ==="
aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --desired-count 0 \
  --query "service.{Service:serviceName,Desired:desiredCount,Running:runningCount}" \
  --output table

echo ""
echo "=== Waiting for all tasks to drain and stop ==="
echo "(This may take up to 2 minutes...)"
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME"

echo ""
echo "=== ECS service stopped ==="
echo "All tasks have been drained and stopped."
echo ""
echo "Note: AWS infrastructure (VPC, ECR, ALB, ECS) is managed in lite-infra."
echo "To fully tear down all resources, run 'cdk destroy' in the lite-infra repo."