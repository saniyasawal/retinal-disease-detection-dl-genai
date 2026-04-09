import os
import cv2

INPUT_DIR = "processed"
OUTPUT_DIR = "processed_rgb"

def convert_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                path = os.path.join(root, file)

                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                if img is None:
                    continue

                # If BGRA → convert to RGB
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # If BGR → convert to RGB
                elif len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Create output path
                new_path = path.replace(INPUT_DIR, OUTPUT_DIR)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                cv2.imwrite(new_path, img)

    print("✅ RGB conversion complete")

convert_folder(INPUT_DIR, OUTPUT_DIR)