import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import os

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD BINARY MODEL
# =========================
binary_model = models.resnet18(pretrained=False)
binary_model.fc = nn.Linear(binary_model.fc.in_features, 2)

binary_model.load_state_dict(torch.load(
    "/home/student/retinal-disease-detection-dl-genai/models/dr_binary_finetuned.pth",
    map_location=device
))
binary_model = binary_model.to(device)
binary_model.eval()

# =========================
# LOAD SEVERITY MODEL
# =========================
severity_model = models.resnet18(pretrained=False)
severity_model.fc = nn.Linear(severity_model.fc.in_features, 4)

severity_model.load_state_dict(torch.load(
    "/home/student/retinal-disease-detection-dl-genai/models/dr_severity_model.pth",
    map_location=device
))
severity_model = severity_model.to(device)
severity_model.eval()

# =========================
# CLASS NAMES
# =========================
binary_classes = ["No_DR", "DR"]
severity_classes = ["Mild", "Moderate", "Severe", "Proliferative"]

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    rgb_img = img.astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    return input_tensor, rgb_img

# =========================
# GRADCAM SETUP (FOR SEVERITY MODEL)
# =========================
target_layer = severity_model.layer4[-1]
cam = GradCAM(model=severity_model, target_layers=[target_layer])

# =========================
# PIPELINE FUNCTION
# =========================
def run_pipeline(img_path):
    input_tensor, rgb_img = preprocess_image(img_path)

    # =========================
    # STEP 1: BINARY PREDICTION
    # =========================
    with torch.no_grad():
        binary_out = binary_model(input_tensor)
        _, binary_pred = torch.max(binary_out, 1)

    binary_label = binary_classes[binary_pred.item()]
    print("Binary Prediction:", binary_label)

    # ❌ If no DR → STOP
    if binary_pred.item() == 0:
        print("🟢 No DR detected → No need for grading")
        return

    print("🔴 DR detected → Running severity model...")

    # =========================
    # STEP 2: SEVERITY PREDICTION
    # =========================
    with torch.no_grad():
        severity_out = severity_model(input_tensor)
        _, severity_pred = torch.max(severity_out, 1)

    severity_label = severity_classes[severity_pred.item()]
    print("Severity:", severity_label)

    # =========================
    # STEP 3: GRAD-CAM
    # =========================
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(f"{severity_label}")
    plt.axis("off")

    os.makedirs("outputs", exist_ok=True)
    save_path = f"outputs/gradcam_{severity_label}.png"
    plt.savefig(save_path, bbox_inches='tight')

    print("🔥 Grad-CAM saved at:", save_path)

# =========================
# TEST RUN
# =========================
if __name__ == "__main__":
    test_image = "Aptos/Images/Aptos_0_1.jpg"  # change path
    run_pipeline(test_image)