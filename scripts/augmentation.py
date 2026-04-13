import os
import cv2
import random

# =========================
# AUGMENTATION FUNCTIONS
# =========================
def random_flip(img):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    return img

def random_rotation(img):
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def augment_image(img):
    img = random_flip(img)
    img = random_rotation(img)
    return img

# =========================
# PROCESS FUNCTION (MISSING BEFORE)
# =========================
def process_folder(input_folder, output_folder, is_train=True):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                path = os.path.join(root, file)

                img = cv2.imread(path)
                if img is None:
                    continue

                # Maintain folder structure
                new_path = path.replace(INPUT_DIR, OUTPUT_DIR)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                # Save original
                cv2.imwrite(new_path, img)

                # Augment only train
                if is_train:
                    aug_img = augment_image(img)

                    base_name = os.path.splitext(new_path)[0]
                    ext = os.path.splitext(new_path)[1]

                    aug_path = base_name + "_aug" + ext
                    cv2.imwrite(aug_path, aug_img)

    print(f"✅ Done: {input_folder}")

# =========================
# RUN
# =========================

INPUT_DIR = "processed_rgb"
OUTPUT_DIR = "processed_rgb_aug"

train_input = os.path.join(INPUT_DIR, "train")
test_input = os.path.join(INPUT_DIR, "test")

train_output = os.path.join(OUTPUT_DIR, "train")
test_output = os.path.join(OUTPUT_DIR, "test")

process_folder(train_input, train_output, is_train=True)
process_folder(test_input, test_output, is_train=False)

print("🎉 AUGMENTATION COMPLETE!")