import os
import csv
from datetime import datetime
from collections import Counter

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from PIL import Image
import streamlit as st

from hail_path_density_predict import predict_density_image

MODEL_PATH = "models/hail_path_triage.pth"
IMAGE_SIZE = 224

YELLOW_THRESHOLD = 0.65
RED_CONFIDENCE_THRESHOLD = 0.90
RED_MARGIN_THRESHOLD = 0.20

UPLOAD_ROOT = "session_uploads"
REPORT_ROOT = "reports"

CRITICAL_RED_PANELS = {
    "Left Roof Rail",
    "Right Roof Rail",
    "Left Quarter",
    "Right Quarter",
    "Left Fender",
    "Right Fender",
}

PANEL_SLOTS = [
    "Roof",
    "Left Roof Rail",
    "Right Roof Rail",
    "Hood",
    "Left Fender",
    "Right Fender",
    "Decklid",
    "Left Quarter",
    "Right Quarter",
    "Left Front Door",
    "Right Front Door",
    "Left Rear Door",
    "Right Rear Door",
    "Other / Detail"
]

ROUTE_MAP = {
    "green_pdr": "GREEN - PDR Candidate",
    "yellow_review": "YELLOW - Manual Review Required",
    "red_conventional": "RED - Likely Conventional Repair"
}


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


@st.cache_resource
def get_model():
    return load_model()


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])


def predict_image(image: Image.Image):

    model, class_names = get_model()
    transform = get_transform()

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

    green_prob = probability_map.get("green_pdr", 0.0)
    yellow_prob = probability_map.get("yellow_review", 0.0)
    red_prob = probability_map.get("red_conventional", 0.0)

    red_margin = red_prob - green_prob

    if (
        predicted_class == "red_conventional"
        and confidence >= RED_CONFIDENCE_THRESHOLD
        and red_margin >= RED_MARGIN_THRESHOLD
    ):
        final_class = "red_conventional"

    elif confidence < YELLOW_THRESHOLD:
        final_class = "yellow_review"

    else:
        final_class = predicted_class

    density_result = predict_density_image(image)

    return {
        "raw_class": predicted_class,
        "final_class": final_class,
        "confidence": confidence,
        "probabilities": probability_map,
        "route_text": ROUTE_MAP[final_class],
        "green_prob": green_prob,
        "yellow_prob": yellow_prob,
        "red_prob": red_prob,
        "red_margin": red_margin,
        "density_class": density_result["density_class"],
        "density_score": density_result["density_score"],
        "density_confidence": density_result["density_confidence"],
    }


def vehicle_level_route(results):

    counts = Counter([r["final_class"] for r in results])

    hard_reds = 0

    for r in results:
        if (
            r["final_class"] == "red_conventional"
            and r["slot"] in CRITICAL_RED_PANELS
            and r["confidence"] >= RED_CONFIDENCE_THRESHOLD
        ):
            hard_reds += 1

    if counts["red_conventional"] >= 2:
        final_vehicle_route = "red_conventional"
        reason = "Two or more panels flagged red."

    elif hard_reds >= 1:
        final_vehicle_route = "red_conventional"
        reason = "A critical panel flagged strong red."

    elif counts["yellow_review"] >= 1 or counts["red_conventional"] >= 1:
        final_vehicle_route = "yellow_review"
        reason = "At least one panel requires review."

    else:
        final_vehicle_route = "green_pdr"
        reason = "All panels green."

    avg_conf = sum(r["confidence"] for r in results) / len(results)

    return {
        "final_class": final_vehicle_route,
        "route_text": ROUTE_MAP[final_vehicle_route],
        "average_confidence": avg_conf,
        "counts": counts,
        "reason": reason,
    }


st.set_page_config(page_title="HAIL Path", layout="wide")

st.title("HAIL Path")
st.caption("AI hail triage system with dent density scoring")

try:
    get_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

vehicle_id = st.text_input("Vehicle / Claim ID", value="vehicle_001")

st.subheader("Upload vehicle photos")

uploads = {}

for slot in PANEL_SLOTS:
    uploads[slot] = st.file_uploader(slot, type=["jpg","jpeg","png"])

if st.button("Run HAIL Path Triage"):

    photo_results = []

    for slot, uploaded in uploads.items():

        if uploaded is None:
            continue

        image = Image.open(uploaded).convert("RGB")

        pred = predict_image(image)

        pred["slot"] = slot
        pred["filename"] = uploaded.name

        photo_results.append(pred)

    if not photo_results:
        st.warning("Upload at least one image.")
        st.stop()

    summary = vehicle_level_route(photo_results)

    st.subheader("Final Vehicle Route")
    st.success(summary["route_text"])

    st.write("Average confidence:", f"{summary['average_confidence']:.2%}")

    st.subheader("Panel Results")

    for result in photo_results:

        with st.expander(result["slot"], expanded=True):

            st.write("Route:", result["route_text"])
            st.write("Confidence:", f"{result['confidence']:.2%}")

            st.write("Dent Density:", result["density_class"])
            st.write("Density Score:", result["density_score"])

            st.write({
                "green": f"{result['green_prob']:.2%}",
                "yellow": f"{result['yellow_prob']:.2%}",
                "red": f"{result['red_prob']:.2%}",
            })