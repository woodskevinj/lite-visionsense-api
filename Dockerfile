# ==============================================================
# 🐳  VisionSense API - Optimized Dockerfile (CPU-only, ~700MB)
# ==============================================================

FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS dependencies required by Pillow & Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port 80 — required by ALB health check in LiteInfraStack
EXPOSE 80

# Port Uvicorn will bind to
ENV PORT=80

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
