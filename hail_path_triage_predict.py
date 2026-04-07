import sys
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from PIL import Image
import os

MODEL_PATH = "models/hail_path_triage.pth"
IMAGE_SIZE = 224
YELLOW_THRESHOLD = 0.80

if len(sys.argv) < 2:
    print("Usage: python hail_path_triage_predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
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

route_map = {
    "green_pdr": "GREEN - PDR Candidate",
    "yellow_review": "YELLOW - Manual Review Required",
    "red_conventional": "RED - Likely Conventional Repair"
}

print(f"Prediction: {route_map.get(final_class, final_class)}")
print(f"Confidence: {confidence:.2%}")

print("\nClass probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {probabilities[i].item():.2%}")