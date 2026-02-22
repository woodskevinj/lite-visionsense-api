"""
Export ResNet-18 to ONNX format for production inference.

Run directly to generate models/resnet18.onnx and models/labels.json:

    python src/export_onnx.py

This script is also executed automatically during `docker build` via the
multi-stage Dockerfile.  The exported ONNX model is then copied into the
slim runtime image so PyTorch is not a runtime dependency.

Fine-tuned model support
------------------------
If models/resnet18_finetuned.pth exists, the fine-tuned CIFAR-10 weights are
exported.  Otherwise the pretrained ImageNet ResNet-18 weights are used.
"""

import json
import os

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def export(
    pth_path: str = "models/resnet18_finetuned.pth",
    output_dir: str = "models",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    CIFAR10_LABELS = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if os.path.exists(pth_path):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(CIFAR10_LABELS))
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        labels = CIFAR10_LABELS
        print(f"Exporting fine-tuned CIFAR-10 model from {pth_path}")
    else:
        labels = list(weights.meta["categories"])
        print("No fine-tuned model found — exporting pretrained ImageNet ResNet-18.")

    model.eval()

    onnx_path = os.path.join(output_dir, "resnet18.onnx")
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"ONNX model saved → {onnx_path}")

    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    print(f"Labels saved     → {labels_path}")


if __name__ == "__main__":
    export()
