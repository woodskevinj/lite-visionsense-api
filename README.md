# lite-visionsense-api

A lightweight, containerized image-classification microservice built with **FastAPI** and **PyTorch**. Uses a pretrained ResNet-18 (ImageNet) with optional fine-tuned CIFAR-10 weights, served via Uvicorn on port 80. Includes a Tailwind-powered web dashboard for interactive predictions.

Deployed to the shared **LiteInfraStack** AWS ECS Fargate cluster. Infrastructure is managed separately in [lite-infra](https://github.com/woodskevinj/lite-infra).

---

## API Endpoints

| Route        | Method | Description                              |
|--------------|--------|------------------------------------------|
| `/`          | GET    | Welcome message                          |
| `/dashboard` | GET    | Web dashboard for uploading images       |
| `/predict`   | POST   | Upload image → top-5 label + confidence  |
| `/health`    | GET    | Model/API health check                   |
| `/info`      | GET    | Service metadata (version, model, etc.)  |
| `/logs`      | GET    | Recent prediction log entries            |
| `/docs`      | GET    | Swagger UI                               |
| `/redoc`     | GET    | ReDoc UI                                 |

---

## Run Locally

**Prerequisites:** Python 3.10, pip

```bash
# 1. Install dependencies (includes onnxruntime)
pip install -r requirements.txt

# 2. Export the model to ONNX (one-time step — requires torch locally)
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
python src/export_onnx.py
# → produces models/resnet18.onnx and models/labels.json

# 3. Start the server
uvicorn app:app --reload

# 4. Open the dashboard
open http://127.0.0.1:8000/dashboard

# 5. Or test the API directly
curl -X POST -F "file=@test.jpg" http://127.0.0.1:8000/predict
```

The inference runtime is **onnxruntime** — PyTorch is only needed to run the one-time export script (`src/export_onnx.py`). The export defaults to the pretrained ImageNet ResNet-18. To use a CIFAR-10 fine-tuned model, run `python src/train_finetune.py` first, then re-run `python src/export_onnx.py` — it detects `models/resnet18_finetuned.pth` automatically.

---

## Run the Test Suite

```bash
# Install dev dependencies (pytest + httpx + torch for export script)
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v
```

Tests mock the VisionClassifier so no ONNX model file is needed. All 21 tests pass in under 5 seconds.

---

## Docker

```bash
# Build (port 80)
docker build -t lite-visionsense-api .

# Run locally on port 8080 (maps host 8080 → container 80)
docker run -p 8080:80 lite-visionsense-api

# Verify
curl http://localhost:8080/health
```

---

## Deploy to AWS ECS Fargate

### Prerequisites

| Tool | Purpose |
|------|---------|
| [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | Interact with AWS |
| [Docker](https://docs.docker.com/get-docker/) | Build and push images |
| [jq](https://jqlang.github.io/jq/) | Parse CloudFormation output JSON |
| Valid AWS credentials | `aws configure` or `AWS_PROFILE` env var |

The deploy script reads **all** resource names (cluster, service, ECR URI, ALB DNS) from the `LiteInfraStack` CloudFormation outputs at runtime. No hardcoded ARNs or account IDs.

### Deploy

```bash
./deploy.sh
```

What it does:

1. Checks all prerequisites (aws, docker, jq, credentials, Docker daemon)
2. Queries `LiteInfraStack` CloudFormation outputs for cluster name, service name, ECR URI, and ALB DNS
3. Authenticates Docker with ECR
4. Builds the image for `linux/amd64` and pushes to ECR as `:latest`
5. Fetches the current ECS task definition from the running service
6. Replaces the `app` container image in the task definition JSON using `jq`
7. Registers a new task definition revision
8. Updates the ECS service with `--force-new-deployment`
9. Prints the ALB URL and a monitoring command

When complete, the app is accessible at the printed ALB DNS name on port 80.

### Stop (scale to zero)

```bash
./teardown.sh
```

Scales the ECS service desired count to 0 and waits for all tasks to fully drain. This stops all running containers and eliminates compute charges. AWS infrastructure (VPC, ECR, ALB) remains intact and is managed in `lite-infra`.

---

## Infrastructure

All AWS infrastructure is defined in the **lite-infra** repository using AWS CDK. This repo only contains application code and deployment scripts. To fully tear down infrastructure, run `cdk destroy` in `lite-infra`.

### Flags / Dependencies on lite-infra

| Item | Status | Notes |
|------|--------|-------|
| ECR repository output key | `VisionSenseEcrUri` | Confirmed CloudFormation output key in `lite-infra` |
| Health check path | `GET /` on port 80 | App's root route returns `200 OK` — no changes needed in lite-infra |
| Container name in task definition | `app` | deploy.sh targets this name in the jq transform |
| Container port | `80` | Dockerfile now runs Uvicorn on port 80 |

---

## Project Structure

```
lite-visionsense-api/
├── app.py                  # FastAPI application and routes
├── src/
│   ├── classifier.py       # VisionClassifier (ResNet-18 inference)
│   └── train_finetune.py   # CIFAR-10 fine-tuning script
├── templates/
│   └── index.html          # Dashboard UI (Jinja2 + TailwindCSS)
├── static/
│   └── style.css
├── tests/
│   └── test_app.py         # Full route + error-handling test suite
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Test dependencies (pytest, httpx)
├── Dockerfile              # Container definition (port 80)
├── deploy.sh               # Build, push, and deploy to ECS Fargate
└── teardown.sh             # Scale ECS service to zero
```

---

Author: Kevin Woods — Applied ML Engineer | AWS Certified AI Practitioner | AWS ML Engineer Associate
