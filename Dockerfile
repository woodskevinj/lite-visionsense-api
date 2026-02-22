# ==============================================================
# Stage 1: Export ResNet-18 to ONNX
#
# PyTorch and TorchVision are only needed here to run the export
# script. They do NOT appear in the final runtime image.
# ==============================================================
FROM python:3.10-slim AS exporter

WORKDIR /export

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/

# Produces models/resnet18.onnx and models/labels.json
RUN python src/export_onnx.py


# ==============================================================
# Stage 2: Slim runtime image (~150-200 MB total)
#
# Only onnxruntime + FastAPI stack — no PyTorch dependency.
# ==============================================================
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# libgl1 is required by Pillow for certain image formats
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pull the exported ONNX model and labels from the exporter stage
COPY --from=exporter /export/models/ ./models/

# Copy application source
COPY . .

# Expose port 80 — required by ALB health check in LiteInfraStack
EXPOSE 80

# Port Uvicorn will bind to
ENV PORT=80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
