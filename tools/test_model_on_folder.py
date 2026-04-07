import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ========= CONFIG =========
MODEL_PATH = "models/hail_path_triage_STABLE_20260317.pth"
IMAGE_DIR = "test_vehicle"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["green_pdr", "red_conventional", "yellow_review"]

# ========= TRANSFORM =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========= LOAD MODEL =========
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ========= RUN =========
image_paths = list(Path(IMAGE_DIR).glob("*"))

print("\nTesting images in:", IMAGE_DIR)
print("----------------------------------")

for img_path in image_paths:
    try:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = probs.argmax()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]

        print(f"{img_path.name}")
        print(f"  → {pred_class} ({confidence:.2%})")

    except Exception as e:
        print(f"{img_path.name} → ERROR: {e}")