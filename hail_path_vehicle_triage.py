import sys
import os
from collections import Counter
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from PIL import Image

MODEL_PATH = "models/hail_path_triage.pth"
IMAGE_SIZE = 224
YELLOW_THRESHOLD = 0.80
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

if len(sys.argv) < 2:
    print("Usage: python hail_path_vehicle_triage.py <folder_path>")
    sys.exit(1)

folder_path = sys.argv[1]

if not os.path.isdir(folder_path):
    print(f"Folder not found: {folder_path}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

route_map = {
    "green_pdr": "GREEN - PDR Candidate",
    "yellow_review": "YELLOW - Manual Review Required",
    "red_conventional": "RED - Likely Conventional Repair"
}

results = []

for name in os.listdir(folder_path):
    if not name.lower().endswith(IMAGE_EXTS):
        continue

    image_path = os.path.join(folder_path, name)

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        if confidence < YELLOW_THRESHOLD:
            final_class = "yellow_review"
        else:
            final_class = predicted_class

        results.append((name, final_class, confidence))

    except Exception as e:
        print(f"Skipped {name}: {e}")

if not results:
    print("No valid images found.")
    sys.exit(1)

counts = Counter([r[1] for r in results])

# Final routing logic
if counts["red_conventional"] >= 2:
    final_vehicle_route = "red_conventional"
elif counts["yellow_review"] >= 2:
    final_vehicle_route = "yellow_review"
elif counts["red_conventional"] >= 1 and counts["green_pdr"] >= 1:
    final_vehicle_route = "yellow_review"
else:
    final_vehicle_route = counts.most_common(1)[0][0]

avg_conf = sum(r[2] for r in results) / len(results)

print("\nHAIL PATH TRIAGE REPORT")
print("-" * 30)
print(f"Photos analyzed: {len(results)}\n")

for name, final_class, confidence in results:
    print(f"{name}: {route_map[final_class]} ({confidence:.2%})")

print("\nFinal vehicle route:", route_map[final_vehicle_route])
print(f"Average confidence: {avg_conf:.2%}")