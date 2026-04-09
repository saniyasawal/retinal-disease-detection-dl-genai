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
def crop_retina(img_input):
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    _, thresh_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_mask = cv2.medianBlur(thresh_mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        center = (int(x), int(y))
        radius = int(radius)

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        masked_img = cv2.bitwise_and(img_input, img_input, mask=mask)

        x1 = max(center[0]-radius, 0)
        y1 = max(center[1]-radius, 0)
        x2 = min(center[0]+radius, img_input.shape[1])
        y2 = min(center[1]+radius, img_input.shape[0])

        cropped_bgr = masked_img[y1:y2, x1:x2]
        cropped_alpha = mask[y1:y2, x1:x2]

        bgra = cv2.merge([cropped_bgr[:,:,0], cropped_bgr[:,:,1], cropped_bgr[:,:,2], cropped_alpha])

        return bgra
    else:
        b,g,r = cv2.split(img_input)
        alpha = np.full(img_input.shape[:2], 255, dtype=np.uint8)
        return cv2.merge([b,g,r,alpha])

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

        # Separate BGR + Alpha
        bgr = img_bg_removed[:, :, :3]
        alpha = img_bg_removed[:, :, 3]

        # Resize separately
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        alpha = cv2.resize(alpha, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Merge back
        img_bg_removed = cv2.merge([bgr[:,:,0], bgr[:,:,1], bgr[:,:,2], alpha])

        # =========================
        # SAVE PATHS
        # =========================
        bg_dir = os.path.join(OUTPUT_DIR, split, "with_bg", label)
        nobg_dir = os.path.join(OUTPUT_DIR, split, "no_bg", label)

        os.makedirs(bg_dir, exist_ok=True)
        os.makedirs(nobg_dir, exist_ok=True)

        # Save images
        cv2.imwrite(os.path.join(bg_dir, img_name), img_resized)
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(nobg_dir, base_name + ".png"), img_bg_removed)
        print(f"✅ Done processing {split}")

# =========================
# RUN
# =========================
process(train_df, "train")
process(test_df, "test")

print("🎉 PREPROCESSING COMPLETE!")