from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.classifier import VisionClassifier
import shutil
import tempfile
import os
import logging

app = FastAPI(
    title="VisionSense API",
    description="Image classification API using a pretrained ResNet model",
    version="1.0.0",
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template engine
templates = Jinja2Templates(directory="templates")

# ======================================================
# Logging Setup (add near the top of api/app.py)
# ======================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "predictions.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# Initialize classifier once at startup
classifier = VisionClassifier()

    
@app.get("/")
def root():
    return {"message": "Welcome to VisionSense API!  Use POST /predict to classify images."}

@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    device = "cuda" if classifier.device == "cuda" else "cpu"
    return {
        "status": "healthy", 
        "model_loaded": "True",
        "device": device,
        "message": "Model and API are ready for inference."
        }


@app.get("/info")
def model_info():
    """Return model and app metadata"""
    return {
        "service": "VisionSense API",
        "version": "1.1",
        "model": "TorchVision ResNet-50 (CIFAR-10)",
        "framework": "FastAPI",
        "description": "Lightweight containerized image-classification microservice."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file and returns top-5 predicted labels + confidences.
    """

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # ✅ Now returns top-5 list instead of single dict
        results = classifier.predict(temp_path, top_k=5)
        os.remove(temp_path)

        predicted_labels = [r['label'] for r in results]

        logging.info(
            f"File: {file.filename} | predictions: {predicted_labels}"
        )

        return JSONResponse(content={"success": True, "result": results})
    
    
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.get("/logs")
async def get_logs(limit: int = 10):
    """Return the most recent vision logs."""
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            return {"logs": lines[-limit:]}
    except FileNotFoundError:
        return {"logs": [], "message": "No logs found yet."}