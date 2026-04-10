import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter

# =========================
# CONFIG
# =========================
TRAIN_DIR = "processed_rgb/train/no_bg"
VAL_DIR = "processed_rgb/test/no_bg"

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3   # as per your description

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATA
# =========================
train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_dataset.classes)

# =========================
# CLASS WEIGHTS (IMPORTANT)
# =========================
class_counts = Counter(train_dataset.targets)
print("Class Distribution:", class_counts)

total = sum(class_counts.values())
weights = [total / class_counts[i] for i in range(len(class_counts))]
weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

# =========================
# MODEL (RESNET-18)
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Fine-tuning (DO NOT freeze layers)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

# =========================
# EVALUATE
# =========================
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader)
    y_true, y_pred = evaluate(model, val_loader)

    acc = np.mean(y_true == y_pred)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")

# =========================
# FINAL METRICS
# =========================
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "dr_binary_finetuned.pth")

print("\n🎉 Training Complete!")