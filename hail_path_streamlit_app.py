import csv
import html
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="HAIL Path", layout="wide")
st.image("logo.png", width=300)
EXPORT_FILE = Path("carrier_triage_results.csv")
FEEDBACK_DIR = Path("retraining_feedback")

ROUTE_MODEL_CANDIDATES = [
    Path("models/hail_path_triage_STABLE_20260317.pth"),
    Path("models/hail_path_triage_pilot.pth"),
    Path("models/hail_path_triage.pth"),
]

CLASS_NAMES_FALLBACK = ["green_pdr", "red_conventional", "yellow_review"]

CORE_PANELS = {"roof", "hood", "decklid", "roof_rail"}
ALL_PANEL_ORDER = ["roof", "roof_rail", "hood", "decklid", "quarter", "door", "fender", "other"]

PANEL_WEIGHTS = {
    "roof": 1.50,
    "roof_rail": 1.35,
    "hood": 1.35,
    "decklid": 1.25,
    "quarter": 0.85,
    "door": 0.75,
    "fender": 0.75,
    "other": 0.50,
}

PANEL_KEYWORDS = [
    ("roof_rail", ["left_roof_rail", "right_roof_rail", "roof_rail", "lt rail", "rt rail", "roof rail"]),
    ("decklid", ["decklid", "trunk", "deck lid"]),
    ("hood", ["hood"]),
    ("quarter", ["quarter", "qp"]),
    ("door", ["door"]),
    ("fender", ["fender"]),
    ("roof", ["roof"]),
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
        st.session_state["do_reset"] = False


def queue_reset():
    st.session_state["do_reset"] = True


def detect_panel(filename):
    name = filename.lower()
    for panel, keywords in PANEL_KEYWORDS:
        for keyword in keywords:
            if keyword in name:
                return panel
    return "other"


def get_route_model_path():
    for p in ROUTE_MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def normalize_class_names(value):
    if value is None:
        return list(CLASS_NAMES_FALLBACK)

    if isinstance(value, list):
        if len(value) == 0:
            return list(CLASS_NAMES_FALLBACK)
        return value

    if isinstance(value, tuple):
        if len(value) == 0:
            return list(CLASS_NAMES_FALLBACK)
        return list(value)

    try:
        converted = list(value)
        if len(converted) == 0:
            return list(CLASS_NAMES_FALLBACK)
        return converted
    except Exception:
        return list(CLASS_NAMES_FALLBACK)


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
        class_names = list(CLASS_NAMES_FALLBACK)
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
    if not all_results:
        return None, None, {}

    if not class_names:
        return None, None, {}

    totals = {}
    for name in class_names:
        totals[name] = 0.0

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
    lines.append("Per-Image Predictions:")
    for row in data["image_rows"]:
        lines.append(
            "  {} | {} | {} | {}".format(
                row["filename"],
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
                html.escape(row["filename"]),
                html.escape(row["panel"]),
                html.escape(row["pred_class"]),
                html.escape(row["confidence_text"]),
            )
        )

    html_doc = """
    <html>
    <head>
        <title>HAIL Path Triage Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
            h1, h2 {{ margin-bottom: 8px; }}
            .section {{ margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; font-size: 13px; }}
            th {{ background: #f2f2f2; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
            .box {{ border: 1px solid #ddd; padding: 10px; }}
        </style>
    </head>
    <body>
        <h1>HAIL Path Triage Summary</h1>
        <div class="section grid">
            <div class="box">
                <strong>Timestamp:</strong> {timestamp}<br>
                <strong>Vehicle / Claim ID:</strong> {vehicle_claim_id}<br>
                <strong>VIN:</strong> {vin}<br>
                <strong>Year:</strong> {year}<br>
                <strong>Make:</strong> {make}<br>
                <strong>Model:</strong> {model}<br>
                <strong>Color:</strong> {color}<br>
                <strong>Customer / Insured Name:</strong> {customer_insured_name}<br>
            </div>
            <div class="box">
                <strong>AI Model:</strong> {model_file}<br>
                <strong>AI Vehicle Route:</strong> {ai_vehicle_route}<br>
                <strong>AI Vehicle Confidence:</strong> {ai_vehicle_confidence}<br>
                <strong>Manual Final Route:</strong> {manual_final_route}<br>
            </div>
        </div>
        <div class="section">
            <h2>Reviewer Notes</h2>
            <div class="box">{review_notes}</div>
        </div>
        <div class="section">
            <h2>Per-Image Predictions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Panel</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
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
    return html_doc


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
            "core_panel_count",
            "roof_count",
            "roof_rail_count",
            "hood_count",
            "decklid_count",
            "quarter_count",
            "door_count",
            "fender_count",
            "other_count",
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


init_state()
apply_pending_reset()
route_model, class_names, route_transform, model_info = load_route_model()
class_names = normalize_class_names(class_names)

st.title("HAIL Path")
st.caption("All-panel hail triage with AI suggestions, export, print/share, reset, and retraining feedback buckets")

with st.sidebar:
    st.header("Vehicle / Claim Info")
    vehicle_claim_id = st.text_input("Vehicle / Claim ID", key="vehicle_claim_id")
    vin = st.text_input("VIN", key="vin")
    year = st.text_input("Year", key="year")
    make = st.text_input("Make", key="make")
    model_name = st.text_input("Model", key="model")
    color = st.text_input("Color", key="color")
    customer_insured_name = st.text_input("Customer / Insured Name", key="customer_insured_name")
    notes = st.text_area("Notes", height=120, key="notes")

    st.markdown("---")
    st.subheader("AI Model")
    st.write(model_info)

uploaded_files = st.file_uploader(
    "Upload vehicle hail photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    grouped = defaultdict(list)
    all_predictions = []

    for uploaded in uploaded_files:
        panel = detect_panel(uploaded.name)
        image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")

        item = {
            "name": uploaded.name,
            "panel": panel,
            "image": image,
            "pred_class": "no_model",
            "confidence": 0.0,
            "prob_map": {},
            "panel_strength": "core panel" if panel in CORE_PANELS else "limited training panel",
        }

        if route_model is not None and route_transform is not None and class_names:
            pred_class, confidence, prob_map = predict_image(image, route_model, class_names, route_transform)
            item["pred_class"] = pred_class
            item["confidence"] = confidence
            item["prob_map"] = prob_map

        grouped[panel].append(item)
        all_predictions.append(item)

    ai_vehicle_route, ai_vehicle_confidence, ai_averages = aggregate_vehicle_prediction(all_predictions, class_names)

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

        review_notes = st.text_area(
            "Reviewer Notes",
            height=140,
            key="review_notes"
        )

        st.markdown("---")
        st.subheader("Panel Counts")
        st.write("Roof:", len(grouped.get("roof", [])))
        st.write("Roof Rail:", len(grouped.get("roof_rail", [])))
        st.write("Hood:", len(grouped.get("hood", [])))
        st.write("Decklid:", len(grouped.get("decklid", [])))
        st.write("Quarter:", len(grouped.get("quarter", [])))
        st.write("Door:", len(grouped.get("door", [])))
        st.write("Fender:", len(grouped.get("fender", [])))
        st.write("Other:", len(grouped.get("other", [])))

        st.markdown("---")
        st.subheader("PDR Print / Share Summary")

        image_rows = []
        for item in all_predictions:
            image_rows.append({
                "filename": item["name"],
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
            "core_panel_count": sum(1 for x in all_predictions if x["panel"] in CORE_PANELS),
            "roof_count": len(grouped.get("roof", [])),
            "roof_rail_count": len(grouped.get("roof_rail", [])),
            "hood_count": len(grouped.get("hood", [])),
            "decklid_count": len(grouped.get("decklid", [])),
            "quarter_count": len(grouped.get("quarter", [])),
            "door_count": len(grouped.get("door", [])),
            "fender_count": len(grouped.get("fender", [])),
            "other_count": len(grouped.get("other", [])),
            "review_notes": review_notes,
            "image_rows": image_rows,
        }

        print_text = make_print_summary(export_data)
        print_html = make_print_html(export_data)

        st.download_button(
            "Download PDR Print Summary (.txt)",
            data=print_text,
            file_name="hail_path_summary.txt",
            mime="text/plain"
        )

        st.download_button(
            "Download Print / Share Summary (.html)",
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
                "core_panel_count": export_data["core_panel_count"],
                "roof_count": export_data["roof_count"],
                "roof_rail_count": export_data["roof_rail_count"],
                "hood_count": export_data["hood_count"],
                "decklid_count": export_data["decklid_count"],
                "quarter_count": export_data["quarter_count"],
                "door_count": export_data["door_count"],
                "fender_count": export_data["fender_count"],
                "other_count": export_data["other_count"],
                "review_notes": review_notes,
            }
            save_result(row)
            st.success("Triage result exported to carrier_triage_results.csv")

        if st.button("Start Next Car / Reset"):
            queue_reset()
            st.rerun()

    with left:
        st.subheader("Photo Review")
        st.info("All panels are displayed and scored. Core panels are weighted heavier in the overall vehicle recommendation.")

        for panel_name in ALL_PANEL_ORDER:
            panel_items = grouped.get(panel_name, [])
            if not panel_items:
                continue

            st.markdown("### " + panel_name.replace("_", " ").title())
            cols = st.columns(2)

            for i, item in enumerate(panel_items):
                with cols[i % 2]:
                    st.image(item["image"], caption=item["name"], width="stretch")
                    st.write("**Panel Type:**", item["panel"])
                    st.write("**Panel Status:**", item["panel_strength"])
                    st.write("**AI Route:**", item["pred_class"])
                    st.write("**Confidence:**", "{:.2%}".format(item["confidence"]))

                    if item["prob_map"] and class_names:
                        for class_name in class_names:
                            st.write("- " + class_name + ": " + "{:.2%}".format(item["prob_map"].get(class_name, 0.0)))

                    st.markdown("**Send Wrong Prediction to Retraining Bucket**")
                    fb1, fb2, fb3 = st.columns(3)

                    with fb1:
                        if st.button("Mark Green", key="g_" + panel_name + "_" + str(i) + "_" + item["name"]):
                            saved_to = save_feedback_image(item, "green_pdr")
                            st.success("Saved to " + saved_to)

                    with fb2:
                        if st.button("Mark Yellow", key="y_" + panel_name + "_" + str(i) + "_" + item["name"]):
                            saved_to = save_feedback_image(item, "yellow_review")
                            st.success("Saved to " + saved_to)

                    with fb3:
                        if st.button("Mark Red", key="r_" + panel_name + "_" + str(i) + "_" + item["name"]):
                            saved_to = save_feedback_image(item, "red_conventional")
                            st.success("Saved to " + saved_to)

                    st.markdown("---")
else:
    st.info("Upload a vehicle photo set to begin triage.")