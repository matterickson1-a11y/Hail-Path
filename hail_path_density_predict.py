import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from PIL import Image

DENSITY_MODEL_PATH = "models/density_classifier.pth"
IMAGE_SIZE = 224

DENSITY_SCORE_MAP = {
    "low": 0.25,
    "medium": 0.60,
    "high": 0.90,
}


def load_density_model():
    if not os.path.exists(DENSITY_MODEL_PATH):
        raise FileNotFoundError(f"Density model not found: {DENSITY_MODEL_PATH}")

    checkpoint = torch.load(DENSITY_MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def get_density_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])


def predict_density_image(image: Image.Image):
    model, class_names = load_density_model()
    transform = get_density_transform()

    image_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    probability_map = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()
    density_score = DENSITY_SCORE_MAP.get(predicted_class, 0.50)

    return {
        "density_class": predicted_class,
        "density_confidence": confidence,
        "density_score": density_score,
        "density_probabilities": probability_map,
    }