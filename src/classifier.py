import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import json
import os

class VisionClassifier:
    def __init__(self, model_path: str = "models/resnet18_finetuned.pth", device: str = None):
        """
        Initialize VisionSense classifier.
        Loads a fine-tuned ResNet18 model trained on CIFAR-10.
        Falls back to ImageNet weights if no fine-tuned model is found.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Define CIFAR-10 labels
        self.cifar10_labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Initialize model
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.cifar10_labels))

        # Try loading fine-tuned weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded fine-tuned model from {model_path}")
            self.labels = self.cifar10_labels

        else:
            print("⚠️ Fine-tuned model not found - using pretrained ImageNet weights.")
            self.model = resnet18(weights=weights)
            self.labels = weights.meta["categories"]

        self.model.to(self.device)
        self.model.eval()

        # Use CIFAR-like transforms (matches fine-tuning preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def predict(self, image_path: str, top_k: int = 5):
        """
        Run image classification and return top-K prediction
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top-K predictions
        top_probs, top_classes = torch.topk(probabilities, min(top_k, len(self.labels)))
        results = [
            {
                "label": self.labels[idx.item()],
                "confidence": round(prob.item(), 4)
            }
            for prob, idx in zip(top_probs, top_classes)
        ]
        return results