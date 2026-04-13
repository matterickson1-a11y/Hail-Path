import csv
import html
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

st.set_page_config(page_title="HAIL Path", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True
)

EXPORT_FILE = Path("carrier_triage_results.csv")
FEEDBACK_DIR = Path("retraining_feedback")

ROUTE_MODEL_CANDIDATES = [
    Path("models/hail_path_triage_STABLE_20260320_feedback.pth"),
    Path("models/hail_path_triage_STABLE_20260317.pth"),
    Path("models/hail_path_triage_pilot.pth"),
    Path("models/hail_path_triage.pth"),
]

CLASS_NAMES_FALLBACK = ["green_pdr", "red_conventional", "yellow_review"]

PANEL_WEIGHTS = {
    "roof": 1.50,
    "left_roof_rail": 1.35,
    "right_roof_rail": 1.35,
    "hood": 1.35,
    "decklid": 1.25,
    "left_fender": 0.75,
    "right_fender": 0.75,
    "left_front_door": 0.75,
    "left_rear_door": 0.75,
    "right_front_door": 0.75,
    "right_rear_door": 0.75,
    "left_quarter": 0.85,
    "right_quarter": 0.85,
}

GUIDED_PANELS = [
    ("roof", "Roof"),
    ("hood", "Hood"),
    ("decklid", "Decklid"),
    ("left_roof_rail", "Left Roof Rail"),
    ("right_roof_rail", "Right Roof Rail"),
    ("left_fender", "Left Fender"),
    ("right_fender", "Right Fender"),
    ("left_front_door", "Left Front Door"),
    ("left_rear_door", "Left Rear Door"),
    ("right_front_door", "Right Front Door"),
    ("right_rear_door", "Right Rear Door"),
    ("left_quarter", "Left Quarter"),
    ("right_quarter", "Right Quarter"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_state():
    defaults = {
        "vehicle_claim_id": "",
        "vin": "",
        "year": "",
        "make": "",
        "model": "",
        "color": "",
        "customer_insured_name": "",
        "notes": "",
        "review_notes": "",
        "manual_final_route": "green_pdr",
        "do_reset": False,
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_pending_reset():
    if st.session_state.get("do_reset", False):
        st.session_state["vehicle_claim_id"] = ""
        st.session_state["vin"] = ""
        st.session_state["year"] = ""
        st.session_state["make"] = ""
        st.session_state["model"] = ""
        st.session_state["color"] = ""
        st.session_state["customer_insured_name"] = ""
        st.session_state["notes"] = ""
        st.session_state["review_notes"] = ""
        st.session_state["manual_final_route"] = "green_pdr"
        st.session_state["uploader_key"] += 1
        st.session_state["do_reset"] = False


def queue_reset():
    st.session_state["do_reset"] = True


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def normalize_class_names(value):
    if value is None:
        return list(CLASS_NAMES_FALLBACK)
    if isinstance(value, list):
        return value if value else list(CLASS_NAMES_FALLBACK)
    if isinstance(value, tuple):
        return list(value) if value else list(CLASS_NAMES_FALLBACK)
    try:
        converted = list(value)
        return converted if converted else list(CLASS_NAMES_FALLBACK)
    except Exception:
        return list(CLASS_NAMES_FALLBACK)


def get_route_model_path():
    for p in ROUTE_MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def load_route_model():
    model_path = get_route_model_path()
    if model_path is None:
        return None, list(CLASS_NAMES_FALLBACK), None, "No route model file found in models folder."

    checkpoint = torch.load(model_path, map_location=DEVICE)

    class_names = list(CLASS_NAMES_FALLBACK)
    image_size = 224

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = normalize_class_names(checkpoint.get("class_names", CLASS_NAMES_FALLBACK))
        maybe_image_size = checkpoint.get("image_size", 224)
        if isinstance(maybe_image_size, int) and maybe_image_size > 0:
            image_size = maybe_image_size
        model = build_model(len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = build_model(len(class_names))
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return model, class_names, transform, str(model_path)


def predict_image(image, model, class_names, transform):
    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    prob_map = {}
    for i, class_name in enumerate(class_names):
        prob_map[class_name] = float(probs[i])

    return pred_class, confidence, prob_map


def aggregate_vehicle_prediction(all_results, class_names):
    if not all_results or not class_names:
        return None, None, {}

    totals = {name: 0.0 for name in class_names}
    total_weight = 0.0

    for item in all_results:
        weight = PANEL_WEIGHTS.get(item["panel"], 1.0)
        total_weight += weight
        for class_name in class_names:
            totals[class_name] += item["prob_map"].get(class_name, 0.0) * weight

    if total_weight == 0:
        return None, None, {}

    averages = {}
    for class_name in class_names:
        averages[class_name] = totals[class_name] / total_weight

    best_class = max(averages, key=averages.get)
    best_conf = averages[best_class]

    return best_class, best_conf, averages


def save_result(row):
    file_exists = EXPORT_FILE.exists()
    with open(EXPORT_FILE, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "vehicle_claim_id",
            "vin",
            "year",
            "make",
            "model",
            "color",
            "customer_insured_name",
            "notes",
            "model_file",
            "ai_vehicle_route",
            "ai_vehicle_confidence",
            "manual_final_route",
            "review_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_feedback_image(item, corrected_class):
    FEEDBACK_DIR.mkdir(exist_ok=True)
    target_dir = FEEDBACK_DIR / corrected_class
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = item["panel"] + "__" + item["name"]
    target_path = target_dir / safe_name

    counter = 1
    while target_path.exists():
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        target_path = target_dir / (stem + "_" + str(counter) + suffix)
        counter += 1

    item["image"].save(target_path)
    return str(target_path)


def make_print_summary(data):
    lines = []
    lines.append("HAIL PATH TRIAGE SUMMARY")
    lines.append("")
    lines.append("Timestamp: " + data["timestamp"])
    lines.append("Vehicle / Claim ID: " + data["vehicle_claim_id"])
    lines.append("VIN: " + data["vin"])
    lines.append("Year: " + data["year"])
    lines.append("Make: " + data["make"])
    lines.append("Model: " + data["model"])
    lines.append("Color: " + data["color"])
    lines.append("Customer / Insured Name: " + data["customer_insured_name"])
    lines.append("")
    lines.append("AI Model: " + data["model_file"])
    lines.append("AI Vehicle Route: " + data["ai_vehicle_route"])
    lines.append("AI Vehicle Confidence: " + data["ai_vehicle_confidence"])
    lines.append("Manual Final Route: " + data["manual_final_route"])
    lines.append("")
    lines.append("Reviewer Notes:")
    lines.append(data["review_notes"])
    lines.append("")
    lines.append("Per-Panel Predictions:")
    for row in data["image_rows"]:
        lines.append(
            "  {} | {} | {} | {}".format(
                row["label"],
                row["panel"],
                row["pred_class"],
                row["confidence_text"]
            )
        )
    return "\n".join(lines)


def make_print_html(data):
    rows_html = []
    for row in data["image_rows"]:
        rows_html.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(row["label"]),
                html.escape(row["panel"]),
                html.escape(row["pred_class"]),
                html.escape(row["confidence_text"]),
            )
        )

    return """
    <html>
    <head>
        <title>HAIL Path Triage Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 13px; }}
            th {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>HAIL Path Triage Summary</h1>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Vehicle / Claim ID:</strong> {vehicle_claim_id}</p>
        <p><strong>VIN:</strong> {vin}</p>
        <p><strong>Year:</strong> {year}</p>
        <p><strong>Make:</strong> {make}</p>
        <p><strong>Model:</strong> {model}</p>
        <p><strong>Color:</strong> {color}</p>
        <p><strong>Customer / Insured Name:</strong> {customer_insured_name}</p>
        <p><strong>AI Model:</strong> {model_file}</p>
        <p><strong>AI Vehicle Route:</strong> {ai_vehicle_route}</p>
        <p><strong>AI Vehicle Confidence:</strong> {ai_vehicle_confidence}</p>
        <p><strong>Manual Final Route:</strong> {manual_final_route}</p>
        <p><strong>Reviewer Notes:</strong><br>{review_notes}</p>
        <table>
            <thead>
                <tr>
                    <th>Panel Label</th>
                    <th>Panel Key</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </body>
    </html>
    """.format(
        timestamp=html.escape(data["timestamp"]),
        vehicle_claim_id=html.escape(data["vehicle_claim_id"]),
        vin=html.escape(data["vin"]),
        year=html.escape(data["year"]),
        make=html.escape(data["make"]),
        model=html.escape(data["model"]),
        color=html.escape(data["color"]),
        customer_insured_name=html.escape(data["customer_insured_name"]),
        model_file=html.escape(data["model_file"]),
        ai_vehicle_route=html.escape(data["ai_vehicle_route"]),
        ai_vehicle_confidence=html.escape(data["ai_vehicle_confidence"]),
        manual_final_route=html.escape(data["manual_final_route"]),
        review_notes=html.escape(data["review_notes"]).replace("\n", "<br>"),
        rows_html="".join(rows_html),
    )


init_state()
apply_pending_reset()
route_model, class_names, route_transform, model_info = load_route_model()
class_names = normalize_class_names(class_names)

if Path("logo.png").exists():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("logo.png", width=420)

st.caption("Guided beta upload flow for panel-by-panel hail triage testing")

st.subheader("Vehicle / Claim Intake")

i1, i2 = st.columns(2)
with i1:
    vehicle_claim_id = st.text_input("Vehicle / Claim ID", key="vehicle_claim_id")
with i2:
    vin = st.text_input("VIN", key="vin")

i3, i4, i5 = st.columns(3)
with i3:
    year = st.text_input("Year", key="year")
with i4:
    make = st.text_input("Make", key="make")
with i5:
    model_name = st.text_input("Model", key="model")

i6, i7 = st.columns(2)
with i6:
    color = st.text_input("Color", key="color")
with i7:
    customer_insured_name = st.text_input("Customer / Insured Name", key="customer_insured_name")

notes = st.text_area("Notes", height=100, key="notes")

with st.expander("AI Model Info"):
    st.write(model_info)

st.subheader("Guided Panel Uploads")

uploaded_panel_items = []

for panel_key, panel_label in GUIDED_PANELS:
    uploaded = st.file_uploader(
        "Upload " + panel_label,
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="uploader_" + panel_key + "_" + str(st.session_state["uploader_key"])
    )

    if uploaded is not None:
        image = Image.open(BytesIO(uploaded.getvalue()))
        image = ImageOps.exif_transpose(image).convert("RGB")

        item = {
            "name": uploaded.name,
            "panel": panel_key,
            "label": panel_label,
            "image": image,
            "pred_class": "no_model",
            "confidence": 0.0,
            "prob_map": {},
            "panel_strength": "core panel" if panel_key in {"roof", "hood", "decklid", "left_roof_rail", "right_roof_rail"} else "limited training panel",
        }

        if route_model is not None and route_transform is not None and class_names:
            pred_class, confidence, prob_map = predict_image(image, route_model, class_names, route_transform)
            item["pred_class"] = pred_class
            item["confidence"] = confidence
            item["prob_map"] = prob_map

        uploaded_panel_items.append(item)

if uploaded_panel_items:
    ai_vehicle_route, ai_vehicle_confidence, ai_averages = aggregate_vehicle_prediction(uploaded_panel_items, class_names)

    if st.session_state["manual_final_route"] == "green_pdr" and ai_vehicle_route in ["green_pdr", "yellow_review", "red_conventional"]:
        st.session_state["manual_final_route"] = ai_vehicle_route

    if not st.session_state.get("review_notes") and ai_vehicle_route is not None:
        st.session_state["review_notes"] = "AI suggested " + ai_vehicle_route + " at " + "{:.2%}".format(ai_vehicle_confidence) + ". Human review required."

    left, right = st.columns([2.4, 1.2])

    with right:
        st.subheader("AI Vehicle Triage")

        if ai_vehicle_route is None:
            st.warning("No AI vehicle recommendation available.")
        else:
            st.write("**AI Route Suggestion:**", ai_vehicle_route)
            st.write("**AI Confidence:**", "{:.2%}".format(ai_vehicle_confidence))
            if ai_averages and class_names:
                st.write("**Weighted Class Averages**")
                for class_name in class_names:
                    st.write("- " + class_name + ": " + "{:.2%}".format(ai_averages.get(class_name, 0.0)))

        st.markdown("---")
        st.subheader("Manual Final Triage")

        manual_final_route = st.selectbox(
            "Final Route",
            ["green_pdr", "yellow_review", "red_conventional"],
            index=["green_pdr", "yellow_review", "red_conventional"].index(st.session_state["manual_final_route"]),
            key="manual_final_route"
        )

        review_notes = st.text_area("Reviewer Notes", height=140, key="review_notes")

        st.markdown("---")
        st.subheader("Print / Share Summary")

        image_rows = []
        for item in uploaded_panel_items:
            image_rows.append({
                "label": item["label"],
                "panel": item["panel"],
                "pred_class": item["pred_class"],
                "confidence_text": "{:.2%}".format(item["confidence"]) if item["confidence"] is not None else "",
            })

        export_data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "vehicle_claim_id": vehicle_claim_id,
            "vin": vin,
            "year": year,
            "make": make,
            "model": model_name,
            "color": color,
            "customer_insured_name": customer_insured_name,
            "notes": notes,
            "model_file": model_info,
            "ai_vehicle_route": ai_vehicle_route if ai_vehicle_route else "",
            "ai_vehicle_confidence": "{:.2%}".format(ai_vehicle_confidence) if ai_vehicle_confidence is not None else "",
            "manual_final_route": manual_final_route,
            "review_notes": review_notes,
            "image_rows": image_rows,
        }

        print_text = make_print_summary(export_data)
        print_html = make_print_html(export_data)

        st.download_button(
            "Download Summary (.txt)",
            data=print_text,
            file_name="hail_path_summary.txt",
            mime="text/plain"
        )

        st.download_button(
            "Download Summary (.html)",
            data=print_html,
            file_name="hail_path_summary.html",
            mime="text/html"
        )

        if st.button("Export Triage Result to CSV"):
            row = {
                "timestamp": export_data["timestamp"],
                "vehicle_claim_id": vehicle_claim_id,
                "vin": vin,
                "year": year,
                "make": make,
                "model": model_name,
                "color": color,
                "customer_insured_name": customer_insured_name,
                "notes": notes,
                "model_file": model_info,
                "ai_vehicle_route": export_data["ai_vehicle_route"],
                "ai_vehicle_confidence": "{:.4f}".format(ai_vehicle_confidence) if ai_vehicle_confidence is not None else "",
                "manual_final_route": manual_final_route,
                "review_notes": review_notes,
            }
            save_result(row)
            st.success("Triage result exported to carrier_triage_results.csv")

        if st.button("Start Next Car / Reset"):
            queue_reset()
            st.rerun()

    with left:
        st.subheader("Panel Review")

        for item in uploaded_panel_items:
            st.markdown("### " + item["label"])
            c1, c2 = st.columns([1.6, 1.0])

            with c1:
                st.image(item["image"], caption=item["name"], width="stretch")

            with c2:
                st.write("**Panel Key:**", item["panel"])
                st.write("**Panel Status:**", item["panel_strength"])
                st.write("**AI Route:**", item["pred_class"])
                st.write("**Confidence:**", "{:.2%}".format(item["confidence"]))

                if item["prob_map"] and class_names:
                    for class_name in class_names:
                        st.write("- " + class_name + ": " + "{:.2%}".format(item["prob_map"].get(class_name, 0.0)))

                st.markdown("**Send Wrong Prediction to Retraining Bucket**")
                fb1, fb2, fb3 = st.columns(3)

                with fb1:
                    if st.button("Mark Green", key="g_" + item["panel"] + "_" + item["name"]):
                        saved_to = save_feedback_image(item, "green_pdr")
                        st.success("Saved to " + saved_to)

                with fb2:
                    if st.button("Mark Yellow", key="y_" + item["panel"] + "_" + item["name"]):
                        saved_to = save_feedback_image(item, "yellow_review")
                        st.success("Saved to " + saved_to)

                with fb3:
                    if st.button("Mark Red", key="r_" + item["panel"] + "_" + item["name"]):
                        saved_to = save_feedback_image(item, "red_conventional")
                        st.success("Saved to " + saved_to)

            st.markdown("---")
else:
    st.info("Upload at least one guided panel photo to begin triage.")