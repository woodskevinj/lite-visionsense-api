import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image


class VisionClassifier:
    def __init__(
        self,
        model_path: str = "models/resnet18.onnx",
        labels_path: str = "models/labels.json",
        device: str = None,
    ):
        """
        Initialize VisionSense classifier using an ONNX model.

        Runs on CPU via onnxruntime — no PyTorch required at inference time.
        Generate the ONNX model with:  python src/export_onnx.py
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at '{model_path}'. "
                "Run 'python src/export_onnx.py' to generate it."
            )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels file not found at '{labels_path}'. "
                "Run 'python src/export_onnx.py' to generate it."
            )

        # device attribute kept for compatibility with app.py health check
        self.device = "cpu"

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        with open(labels_path) as f:
            self.labels = json.load(f)

        # ImageNet normalisation constants — match preprocessing used during training
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, C], range 0-1
        arr = arr.transpose(2, 0, 1)                     # [C, H, W]
        arr = (arr - self._mean) / self._std
        return arr[None, ...]                            # [1, C, H, W]

    def predict(self, image_path: str, top_k: int = 5) -> list:
        """Run image classification and return top-K predictions."""
        input_array = self._preprocess(image_path)
        logits = self.session.run(None, {self.input_name: input_array})[0][0]

        # Numerically stable softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        top_k = min(top_k, len(self.labels))
        top_indices = np.argsort(probs)[::-1][:top_k]

        return [
            {"label": self.labels[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
