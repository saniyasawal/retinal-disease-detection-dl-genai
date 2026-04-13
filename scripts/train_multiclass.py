# =========================
# 1. SETUP
# =========================
import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. PATHS
# =========================
CSV_PATH = "/home/student/retinal-disease-detection-dl-genai/Aptos/Aptos_dataset.csv"
IMG_DIR = "/home/student/retinal-disease-detection-dl-genai/Aptos/Images"

# =========================
# 3. LOAD & PREPARE DATA
# =========================
df = pd.read_csv(CSV_PATH)

# ✅ FIX: correct column names
df.columns = ['image_name', 'label', 'types']

# remove spaces if any
df['image_name'] = df['image_name'].astype(str).str.strip()

# Keep only severity classes (1–4)
df = df[df['label'] != 0]

# Map labels → 0–3
label_map = {1:0, 2:1, 3:2, 4:3}
df['label'] = df['label'].map(label_map)

# Train-test split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

print("Train size:", len(train_df))
print("Val size:", len(val_df))

# =========================
# 4. CUSTOM DATASET
# =========================
class RetinalDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ✅ FIX: handle both .png and .jpg safely
        img_name = row['image_name']
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name + ".png")

        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name + ".jpg")

        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        # preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = self.transform(img)

        label = int(row['label'])

        return img, label

# =========================
# 5. DATALOADERS
# =========================
train_loader = DataLoader(
    RetinalDataset(train_df, IMG_DIR),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    RetinalDataset(val_df, IMG_DIR),
    batch_size=32
)

# =========================
# 6. MODEL (RESNET18)
# =========================
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block
for param in model.layer4.parameters():
    param.requires_grad = True

# Final layer (4 classes)
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(device)

# =========================
# 7. LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# =========================
# 8. TRAINING
# =========================
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# =========================
# 9. EVALUATION
# =========================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Class names
class_names = ["Mild", "Moderate", "Severe", "Proliferative"]

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# =========================
# 10. CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 11. MACRO F1 SCORE
# =========================
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print("Macro F1 Score:", macro_f1)

# =========================
# 12. SAVE MODEL
# =========================
torch.save(model.state_dict(), "dr_severity_model.pth")
print("✅ Model saved successfully!")