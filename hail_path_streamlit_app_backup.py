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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

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

    # Safer triage logic:
    # 1. Red must be top class, very confident, and clearly ahead of green.
    # 2. Weak/uncertain calls become yellow.
    # 3. Otherwise accept the top class.
    if (
        predicted_class == "red_conventional"
        and confidence >= RED_CONFIDENCE_THRESHOLD
        and red_margin >= RED_MARGIN_THRESHOLD
    ):
        final_class = "red_conventional"
        decision_reason = (
            f"Red accepted: top class {confidence:.2%}, "
            f"red-green margin {red_margin:.2%}."
        )
    elif confidence < YELLOW_THRESHOLD:
        final_class = "yellow_review"
        decision_reason = f"Confidence below yellow threshold ({confidence:.2%})."
    else:
        final_class = predicted_class
        decision_reason = f"Top class accepted ({predicted_class}, {confidence:.2%})."

    return {
        "raw_class": predicted_class,
        "final_class": final_class,
        "confidence": confidence,
        "probabilities": probability_map,
        "route_text": ROUTE_MAP[final_class],
        "decision_reason": decision_reason,
        "green_prob": green_prob,
        "yellow_prob": yellow_prob,
        "red_prob": red_prob,
        "red_margin": red_margin,
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
        reason = "No confirmed red route, but at least one panel requires review."
    else:
        final_vehicle_route = "green_pdr"
        reason = "All analyzed panels were green."

    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0.0

    return {
        "final_class": final_vehicle_route,
        "route_text": ROUTE_MAP[final_vehicle_route],
        "average_confidence": avg_conf,
        "counts": counts,
        "reason": reason,
    }


def save_uploaded_file(uploaded_file, folder_path, display_name):
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, display_name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath


def generate_csv(vehicle_id, photo_results, summary, vehicle_meta):
    os.makedirs(REPORT_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(REPORT_ROOT, f"hail_path_report_{vehicle_id}_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["vehicle_id", vehicle_id])
        writer.writerow(["vin", vehicle_meta.get("vin", "")])
        writer.writerow(["year", vehicle_meta.get("year", "")])
        writer.writerow(["make", vehicle_meta.get("make", "")])
        writer.writerow(["model", vehicle_meta.get("model", "")])
        writer.writerow(["color", vehicle_meta.get("color", "")])
        writer.writerow(["customer_or_insured", vehicle_meta.get("owner_name", "")])
        writer.writerow(["notes", vehicle_meta.get("notes", "")])
        writer.writerow(["final_route", summary["route_text"]])
        writer.writerow(["average_confidence", f"{summary['average_confidence']:.2%}"])
        writer.writerow(["reason", summary["reason"]])
        writer.writerow([])
        writer.writerow([
            "slot",
            "filename",
            "image_route",
            "top_confidence",
            "green_pdr",
            "yellow_review",
            "red_conventional",
            "red_minus_green",
            "decision_reason"
        ])

        for r in photo_results:
            probs = r["probabilities"]
            writer.writerow([
                r["slot"],
                r["filename"],
                r["route_text"],
                f"{r['confidence']:.2%}",
                f"{probs.get('green_pdr', 0):.2%}",
                f"{probs.get('yellow_review', 0):.2%}",
                f"{probs.get('red_conventional', 0):.2%}",
                f"{r['red_margin']:.2%}",
                r["decision_reason"],
            ])

    return csv_path


def generate_html_report(vehicle_id, photo_results, summary, vehicle_meta):
    os.makedirs(REPORT_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORT_ROOT, f"hail_path_report_{vehicle_id}_{timestamp}.html")

    rows = ""
    for r in photo_results:
        probs = r["probabilities"]
        rows += f"""
        <tr>
            <td>{r['slot']}</td>
            <td>{r['filename']}</td>
            <td>{r['route_text']}</td>
            <td>{r['confidence']:.2%}</td>
            <td>{probs.get('green_pdr', 0):.2%}</td>
            <td>{probs.get('yellow_review', 0):.2%}</td>
            <td>{probs.get('red_conventional', 0):.2%}</td>
            <td>{r['red_margin']:.2%}</td>
            <td>{r['decision_reason']}</td>
        </tr>
        """

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>HAIL Path Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; color: #111; }}
            h1, h2 {{ margin-bottom: 8px; }}
            .meta, .summary {{ margin-bottom: 22px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: #f3f3f3; }}
            .route {{ font-size: 20px; font-weight: bold; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>HAIL Path Triage Report</h1>
        <div class="meta">
            <p><strong>Vehicle / Claim ID:</strong> {vehicle_id}</p>
            <p><strong>VIN:</strong> {vehicle_meta.get('vin', '')}</p>
            <p><strong>Year:</strong> {vehicle_meta.get('year', '')}</p>
            <p><strong>Make:</strong> {vehicle_meta.get('make', '')}</p>
            <p><strong>Model:</strong> {vehicle_meta.get('model', '')}</p>
            <p><strong>Color:</strong> {vehicle_meta.get('color', '')}</p>
            <p><strong>Customer / Insured:</strong> {vehicle_meta.get('owner_name', '')}</p>
            <p><strong>Notes:</strong> {vehicle_meta.get('notes', '')}</p>
        </div>
        <div class="summary">
            <h2>Final Vehicle Route</h2>
            <div class="route">{summary['route_text']}</div>
            <p><strong>Average Confidence:</strong> {summary['average_confidence']:.2%}</p>
            <p><strong>Reason:</strong> {summary['reason']}</p>
            <p><strong>Bucket Counts:</strong> Green {summary['counts']['green_pdr']} | Yellow {summary['counts']['yellow_review']} | Red {summary['counts']['red_conventional']}</p>
        </div>
        <h2>Photo-by-Photo Triage</h2>
        <table>
            <thead>
                <tr>
                    <th>Slot</th>
                    <th>Filename</th>
                    <th>Image Route</th>
                    <th>Top Confidence</th>
                    <th>Green</th>
                    <th>Yellow</th>
                    <th>Red</th>
                    <th>Red-Green Margin</th>
                    <th>Decision Reason</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


st.set_page_config(page_title="HAIL Path", layout="wide")
st.title("HAIL Path")
st.caption("Damage triage and repair-route decision support")

if "form_version" not in st.session_state:
    st.session_state["form_version"] = 1

with st.sidebar:
    st.header("Settings")
    st.write(f"Yellow confidence threshold: {YELLOW_THRESHOLD:.0%}")
    st.write(f"Red confidence threshold: {RED_CONFIDENCE_THRESHOLD:.0%}")
    st.write(f"Red margin threshold: {RED_MARGIN_THRESHOLD:.0%}")
    st.write("Target workflow: one vehicle at a time")

try:
    get_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

fv = st.session_state["form_version"]

vehicle_id = st.text_input("Vehicle / Claim ID", value="vehicle_001", key=f"vehicle_id_{fv}")
vin = st.text_input("VIN", value="", key=f"vin_{fv}")
year = st.text_input("Year", value="", key=f"year_{fv}")
make = st.text_input("Make", value="", key=f"make_{fv}")
model_name = st.text_input("Model", value="", key=f"model_{fv}")
color = st.text_input("Color", value="", key=f"color_{fv}")
owner_name = st.text_input("Customer / Insured Name", value="", key=f"owner_{fv}")
notes = st.text_area("Notes", value="", key=f"notes_{fv}")

st.subheader("Upload required vehicle photos")
st.write("Assign each photo to a panel slot. HAIL Path will score each image and produce one final route.")

slot_uploads = {}
cols = st.columns(2)
for idx, slot in enumerate(PANEL_SLOTS):
    with cols[idx % 2]:
        slot_uploads[slot] = st.file_uploader(
            f"{slot}",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key=f"upload_{slot}_{fv}"
        )

top_actions = st.columns(3)
with top_actions[0]:
    run_clicked = st.button("Run HAIL Path Triage", type="primary")
with top_actions[1]:
    if st.button("Start Next Vehicle"):
        st.session_state["form_version"] += 1
        st.rerun()

if run_clicked:
    selected = {slot: file for slot, file in slot_uploads.items() if file is not None}

    if not selected:
        st.warning("Upload at least one photo to run triage.")
        st.stop()

    session_folder = os.path.join(UPLOAD_ROOT, f"{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_folder, exist_ok=True)

    photo_results = []

    for slot, uploaded in selected.items():
        clean_name = f"{slot.replace(' ', '_').replace('/', '_')}_{uploaded.name}"
        filepath = save_uploaded_file(uploaded, session_folder, clean_name)

        image = Image.open(filepath).convert("RGB")
        pred = predict_image(image)
        pred["slot"] = slot
        pred["filename"] = os.path.basename(filepath)
        pred["filepath"] = filepath
        photo_results.append(pred)

    summary = vehicle_level_route(photo_results)
    vehicle_meta = {
        "vin": vin,
        "year": year,
        "make": make,
        "model": model_name,
        "color": color,
        "owner_name": owner_name,
        "notes": notes,
    }

    csv_path = generate_csv(vehicle_id, photo_results, summary, vehicle_meta)
    html_path = generate_html_report(vehicle_id, photo_results, summary, vehicle_meta)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Final vehicle route")
        st.success(summary["route_text"])
        st.write(f"Average confidence: {summary['average_confidence']:.2%}")
        st.write(f"Reason: {summary['reason']}")
        st.write("Bucket counts:")
        st.write({
            "green_pdr": summary["counts"]["green_pdr"],
            "yellow_review": summary["counts"]["yellow_review"],
            "red_conventional": summary["counts"]["red_conventional"],
        })

        if vin:
            st.write(f"VIN: {vin}")
        if year or make or model_name or color:
            st.write(f"Vehicle: {year} {make} {model_name} | Color: {color}".strip())
        if owner_name:
            st.write(f"Customer / Insured: {owner_name}")
        if notes:
            st.write(f"Notes: {notes}")

        with open(csv_path, "rb") as f:
            st.download_button(
                label="Download CSV report",
                data=f.read(),
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )

        with open(html_path, "rb") as f:
            st.download_button(
                label="Download printable report",
                data=f.read(),
                file_name=os.path.basename(html_path),
                mime="text/html"
            )

    with right:
        st.subheader("Photo-by-photo triage")
        for result in photo_results:
            title = f"{result['slot']} — {result['route_text']} ({result['confidence']:.2%})"
            with st.expander(title, expanded=True):
                st.image(result["filepath"], use_container_width=True)
                st.write(f"File: {result['filename']}")
                st.write(f"Decision reason: {result['decision_reason']}")
                st.write("Probabilities:")
                st.write({
                    "green_pdr": f"{result['probabilities'].get('green_pdr', 0):.2%}",
                    "yellow_review": f"{result['probabilities'].get('yellow_review', 0):.2%}",
                    "red_conventional": f"{result['probabilities'].get('red_conventional', 0):.2%}",
                    "red_minus_green": f"{result['red_margin']:.2%}",
                })