---
# 🖼️ Vision Sense API & Dashboard

VisionSense is a complete end-to-end image classification system built using **FastAPI**, **PyTorch**, and **TailwindCSS**.

It includes both a REST API **and** a modern web dashboard for uploading images and viewing predictions.

This project demonstrates the applied ML engineer workflow.

- model selection → fine-tuning → API serving → UI → containerization → deployment.
---

## 🧩 Project Overview

VisionSense API provides:

✅ /predict API endpoint for image classification

✅ Tailwind-powered web dashboard at /dashboard

✅ Real-time top-5 predictions with confidence scores

✅ Optional fine-tuned model (CIFAR-10)

✅ Full logging + health checks

✅ Dockerized microservice ready for AWS deployment

---

## ⚙️ Tech Stack

| Component            | Purpose                               |
| -------------------- | ------------------------------------- |
| **FastAPI**          | REST API + HTML template rendering    |
| **PyTorch**          | Model loading + inference             |
| **TorchVision**      | Pretrained ResNet18 + transforms      |
| **TailwindCSS**      | Front-end styling for dashboard UI    |
| **Uvicorn**          | ASGI server for FastAPI               |
| **Docker**           | Containerized deployment              |
| **Python-Multipart** | File upload handling                  |
| **Optional**         | AWS ECS / App Runner deployment later |

---

## 📂 Project Structure

```css
visionsense-api/
├── app.py
├── src/
│   ├── classifier.py
│   └── train_finetune.py
├── templates/
│   └── index.html          # Dashboard UI
├── static/
│   └── (optional CSS, images)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore

```

---

## 🚀 Getting Started

```bash
# 1️⃣ Clone the repository
git clone https://github.com/woodskevinj/visionsense-api.git
cd visionsense-api
```

# 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

# 3️⃣ Run the FastAPI app

```bash
uvicorn api.app:app --reload
```

# 4️⃣ Visit the dashboard

👉 http://127.0.0.1:8000/dashboard

Upload an image and see predictions instantly.

# 5️⃣ Test the API directly

```bash
curl -X POST -F "file=@test.jpg" http://127.0.0.1:8000/predict
```

Expected JSON response:

```json
{
  "success": true,
  "results": [
    { "label": "dog", "confidence": 0.9853 },
    { "label": "cat", "confidence": 0.0094 },
    { "label": "frog", "confidence": 0.0021 },
    { "label": "deer", "confidence": 0.0013 },
    { "label": "horse", "confidence": 0.0011 }
  ]
}
```

---

## 🌐 API Endpoints

| Endpoint   | Method | Description                                            |
| ---------- | ------ | ------------------------------------------------------ |
| /dashboard | GET    | HTML dashboard UI for uploading and classifying images |
| /predict   | POST   | Upload image → get top-5 predictions                   |
| /health    | GET    | Model/API health check (device, model loaded, status)  |
| /info      | GET    | Model + service metadata                               |
| /logs      | GET    | Returns recent prediction logs                         |
| /          | GET    | Welcome message                                        |

Swagger UI: http://127.0.0.1:8000/docs

ReDoc UI: http://127.0.0.1:8000/redoc

---

## 🧠 Model Options

# ✅ Default

ResNet-18 pretrained on ImageNet (torchvision.models).

# ✅ Fine-Tuned Model Support

If models/resnet18_finetuned.pth exists, the classifier automatically switches to CIFAR-10 labels:

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

Fine-tuning script:

```bash
python src/train_finetune.py
```

---

## 🐳 Docker Usage

Build Docker Image

```bash
docker build -t visionsense-api .
```

Run Container

```bash
docker run -p 8000:8000 visionsense-api
```

Verify

```bash
docker ps
```

- Example

```nginx
CONTAINER ID   IMAGE              COMMAND                  STATUS         PORTS                    NAMES
xxxxxx         visionsense-api   "uvicorn app:app --h…"   Up 5 seconds   0.0.0.0:8000->8000/tcp   visionsense

```

Then open:

👉 http://127.0.0.1:8000/dashboard

---

## ☁️ AWS ECR Deployment & Cleanup

<details>
<summary><b>Push Container to AWS ECR</b></summary>

```bash
# 1️⃣ Create repository (only once)
aws ecr create-repository --repository-name visionsense-api --region us-east-1

# 2️⃣ Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

# 3️⃣ Tag image
docker tag visionsense-api:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/visionsense-api:latest

# 4️⃣ Push to ECR
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/visionsense-api:latest
```

</details>

---

<details>
<summary><b>Clean Up Resources to Avoid Charges</b></summary>

```bash
# 🧹 Delete image from ECR
aws ecr batch-delete-image \
  --repository-name visionsense-api \
  --image-ids imageTag=latest \
  --region us-east-1

# 🧼 Remove local image
docker rmi visionsense-api

# 🧾 Optional: Delete repository (only if no longer needed)
aws ecr delete-repository \
  --repository-name visionsense-api \
  --region us-east-1 \
  --force
```

✅ Note: Keeping an empty repository incurs **no cost**.

</details>

---

## 📊 Current Progress

| Phase                              | Status      |
| ---------------------------------- | ----------- |
| **Pretrained ResNet18 Inference**  | ✅ Done     |
| **FastAPI Backend API**            | ✅ Done     |
| **Dashboard UI (Tailwind + HTML)** | ✅ Done     |
| **Logging + Health Endpoints**     | ✅ Done     |
| **CIFAR-10 Fine-Tuning Pipeline**  | ✅ Done     |
| **Dockerization**                  | ✅ Done     |
| **AWS Deployment(ECR/ECS)**        | 🔜 Upcoming |

---

## 🔁 Transfer Learning (Fine-Tuning)

Fine-tuning adapts the pretrained ResNet-18 (ImageNet) to a new, smaller dataset such as CIFAR-10.

**Steps performed in** `src/train_finetune.py`:

1. Freeze pretrained layers and replace the final fully connected layer (`fc`).

2. Train on CIFAR-10 (10 classes).

3. Save weights to `models/resnet18_finetuned.pth`.

4. The API automatically detects and loads the fine-tuned model if present.

---

## ☁️ Deployment Notes

- Image builds ~1 GB with default PyTorch

- Can shrink using CPU-only wheels (torch==x.x.x+cpu)

- Works cleanly on ECS, App Runner, or EC2

---

👨‍💻 Author

# Kevin Woods

Applied ML Engineer

AWS Certified AI Practitioner

AWS Machine Learning Certified Engineer – Associate

- 🔗 [GitHub: woodskevinj](https://github.com/woodskevinj)
