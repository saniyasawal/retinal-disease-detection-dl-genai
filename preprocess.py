import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "Aptos", "Images")
CSV_PATH = os.path.join(BASE_DIR, "Aptos", "csv", "Aptos_dataset.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

IMG_SIZE = 224

# =========================
# LOAD CSV
# =========================
df = pd.read_csv("./Aptos/Aptos_dataset.csv")

# Clean + standardize column names
df.columns = ['image_name', 'label', 'types']
df['image_name'] = df['image_name'].astype(str).str.strip()

print("✅ CSV Loaded:", df.shape)

# =========================
# TRAIN TEST SPLIT
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

print("Train:", len(train_df), "Test:", len(test_df))

# =========================
# BACKGROUND REMOVAL FUNC
# =========================
def crop_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to separate retina
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    # Largest contour = retina
    c = max(contours, key=cv2.contourArea)

    # Fit enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)

    center = (int(x), int(y))
    radius = int(radius)

    # Create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Crop tight bounding box
    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, img.shape[1])
    y2 = min(center[1] + radius, img.shape[0])

    cropped = result[y1:y2, x1:x2]

    return cropped

# =========================
# PROCESS FUNCTION
# =========================
def process(df, split):
    for _, row in df.iterrows():
        img_name = row['image_name']
        label = str(row['label'])

        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            print("❌ Missing:", img_name)
            continue

        img = cv2.imread(img_path)

        if img is None:
            print("❌ Corrupt:", img_name)
            continue

        # Resize original
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Background removed version
        img_bg_removed = crop_retina(img)
        img_bg_removed = cv2.resize(img_bg_removed, (IMG_SIZE, IMG_SIZE))

        # =========================
        # SAVE PATHS
        # =========================
        bg_dir = os.path.join(OUTPUT_DIR, split, "with_bg", label)
        nobg_dir = os.path.join(OUTPUT_DIR, split, "no_bg", label)

        os.makedirs(bg_dir, exist_ok=True)
        os.makedirs(nobg_dir, exist_ok=True)

        # Save images
        cv2.imwrite(os.path.join(bg_dir, img_name), img_resized)
        cv2.imwrite(os.path.join(nobg_dir, img_name), img_bg_removed)

    print(f"✅ Done processing {split}")

# =========================
# RUN
# =========================
process(train_df, "train")
process(test_df, "test")

print("🎉 PREPROCESSING COMPLETE!")