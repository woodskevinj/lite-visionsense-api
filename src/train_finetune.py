"""
Fine-tune a pretrained ResNet18 model on CIFAR-10 or a custom dataset.

Usage:
    python src/train_finetune.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time
import json

# -------------------------
# 1️⃣ Configuration
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"
MODEL_DIR = "./models"
NUM_CLASSES = 10            # CIFAR-10 has 10 classes
EPOCHS = 5                  # adjust for faster/slower training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# -------------------------
# 2️⃣ Data transforms
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # resize to ResNet input size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# 3️⃣ Load CIFAR-10 Dataset
# -------------------------
train_dataset = datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=train_transforms
)
val_dataset = datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 4️⃣ Load Pretrained Model
# -------------------------
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze base layers to retain learned features
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer for our new classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

model = model.to(DEVICE)

# -------------------------
# 5️⃣ Loss, Optimizer, and Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# -------------------------
# 6️⃣ Training Loop
# -------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / total
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # SAve best model
        if val_acc > best_acc:
            best_acc = val_acc
            Path(MODEL_DIR).mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_DIR}/resnet18_finetuned.pth")

    # Save training history
    with open(f"{MODEL_DIR}/training_history.json", "w") as f:
        json.dump(history, f)

    print(f"✅ Training complete. Best accuracy: {best_acc:.2f}%")
    return model

# -------------------------
# 7️⃣ Execute training
# -------------------------
if __name__ == "__main__":
    start = time.time()
    model = train_model(model, train_loader, val_loader,
                        criterion, optimizer, scheduler, EPOCHS)
    end = time.time()
    print(f"Total training time: {(end - start)/60:.2f} minutes")
