from io import BytesIO
from pathlib import Path
from datetime import datetime
import html

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
    .assessment-box {
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .assessment-green {
        background-color: rgba(40, 167, 69, 0.18);
        border-left: 6px solid #28a745;
    }
    .assessment-yellow {
        background-color: rgba(255, 193, 7, 0.18);
        border-left: 6px solid #ffc107;
    }
    .assessment-red {
        background-color: rgba(220, 53, 69, 0.18);
        border-left: 6px solid #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DISPLAY_NAMES = {
    "green_pdr": "PDR Candidate",
    "yellow_review": "Review Recommended",
    "red_conventional": "Conventional Likely",
    "no_model": "Model Not Loaded",
}

DISPLAY_CLASSES = {
    "green_pdr": "assessment-green",
    "yellow_review": "assessment-yellow",
    "red_conventional": "assessment-red",
    "no_model": "assessment-yellow",
}

ROUTE_MODEL_CANDIDATES = [
    Path("models/hail_path_triage_STABLE_20260320_feedback.pth"),
    Path("models/hail_path_triage_STABLE_20260317.pth"),
    Path("models/hail_path_triage_UPDATED_20260413.pth"),
    Path("models/hail_path_triage_pilot.pth"),
    Path("models/hail_path_triage.pth"),
]

FEEDBACK_DIR = Path("retraining_feedback")

CLASS_NAMES_FALLBACK = ["green_pdr", "red_conventional", "yellow_review"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PANEL_CONFIG = [
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

PANEL_WEIGHTS = {
    "roof": 1.50,
    "hood": 1.35,
    "decklid": 1.25,
    "left_roof_rail": 1.30,
    "right_roof_rail": 1.30,
    "left_fender": 0.75,
    "right_fender": 0.75,
    "left_front_door": 0.75,
    "left_rear_door": 0.75,
    "right_front_door": 0.75,
    "right_rear_door": 0.75,
    "left_quarter": 0.85,
    "right_quarter": 0.85,
}

MAX_UPLOAD_IMAGE_SIZE = 768
JPEG_QUALITY = 80

if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0


def trigger_reset():
    st.session_state["reset_counter"] += 1


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource
def load_model():
    for path in ROUTE_MODEL_CANDIDATES:
        if not path.exists():
            continue

        try:
            checkpoint = torch.load(path, map_location=DEVICE)

            model = build_model(len(CLASS_NAMES_FALLBACK))
            class_names = list(CLASS_NAMES_FALLBACK)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                maybe_names = checkpoint.get("class_names")
                if isinstance(maybe_names, (list, tuple)) and len(maybe_names) > 0:
                    class_names = list(maybe_names)
                    model = build_model(len(class_names))
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            return model, class_names, str(path)

        except Exception as e:
            print("MODEL LOAD ERROR:", str(e))
            continue

    return None, list(CLASS_NAMES_FALLBACK), "Model failed to load"


model, class_names, model_info = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def prepare_uploaded_image(file_obj):
    img = Image.open(BytesIO(file_obj.getvalue()))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_UPLOAD_IMAGE_SIZE, MAX_UPLOAD_IMAGE_SIZE))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buffer.seek(0)

    processed = Image.open(buffer).convert("RGB")
    return processed


def predict(image):
    if model is None:
        return "no_model", 0.0, {}

    try:
        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[0]

        idx = int(probs.argmax())
        if idx >= len(class_names):
            return "no_model", 0.0, {}

        prob_map = {}
        for i, name in enumerate(class_names):
            prob_map[name] = float(probs[i])

        return class_names[idx], float(probs[idx]), prob_map
    except Exception:
        return "no_model", 0.0, {}


def save_feedback_image(item, corrected_class):
    FEEDBACK_DIR.mkdir(exist_ok=True)
    target_dir = FEEDBACK_DIR / corrected_class
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = item["panel"] + "__" + item["filename"]
    target_path = target_dir / safe_name

    counter = 1
    while target_path.exists():
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        target_path = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    item["image"].save(target_path)
    return str(target_path)


def aggregate_results(results):
    usable = [r for r in results if r["prediction"] != "no_model"]
    if not usable:
        return None, 0.0, {}

    totals = {name: 0.0 for name in class_names}
    total_weight = 0.0

    for item in usable:
        weight = PANEL_WEIGHTS.get(item["panel"], 1.0)
        total_weight += weight
        for name in class_names:
            totals[name] += item["prob_map"].get(name, 0.0) * weight

    if total_weight == 0:
        return None, 0.0, {}

    averages = {}
    for name in class_names:
        averages[name] = totals[name] / total_weight

    best = max(averages, key=averages.get)
    return best, averages[best], averages


def make_summary_text(claim_id, vin, year, make, model_name, color, customer, notes, results, model_info, overall_pred, overall_conf):
    lines = []
    lines.append("HAIL PATH TRIAGE SUMMARY")
    lines.append("")
    lines.append("Timestamp: " + datetime.now().isoformat(timespec="seconds"))
    lines.append("Claim ID: " + str(claim_id))
    lines.append("VIN: " + str(vin))
    lines.append("Year: " + str(year))
    lines.append("Make: " + str(make))
    lines.append("Model: " + str(model_name))
    lines.append("Color: " + str(color))
    lines.append("Customer Name: " + str(customer))
    lines.append("Notes: " + str(notes))
    lines.append("AI Model: " + str(model_info))
    lines.append("")
    lines.append("Overall Assessment: " + str(DISPLAY_NAMES.get(overall_pred, overall_pred)))
    lines.append("Overall Confidence: " + "{:.2%}".format(overall_conf))
    lines.append("")
    lines.append("Panel Results:")
    for item in results:
        lines.append(
            "{} | {} | {} | {:.2%} | {}".format(
                item["instance_label"],
                item["panel"],
                DISPLAY_NAMES.get(item["prediction"], item["prediction"]),
                item["confidence"],
                item["filename"]
            )
        )
    return "\n".join(lines)


def make_summary_html(claim_id, vin, year, make, model_name, color, customer, notes, results, model_info, overall_pred, overall_conf):
    row_html = []
    for item in results:
        row_html.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2%}</td><td>{}</td></tr>".format(
                html.escape(item["instance_label"]),
                html.escape(item["panel"]),
                html.escape(DISPLAY_NAMES.get(item["prediction"], item["prediction"])),
                item["confidence"],
                html.escape(item["filename"])
            )
        )

    return """
    <html>
    <head>
        <title>HAIL Path Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>HAIL Path Triage Summary</h1>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Claim ID:</strong> {claim_id}</p>
        <p><strong>VIN:</strong> {vin}</p>
        <p><strong>Year:</strong> {year}</p>
        <p><strong>Make:</strong> {make}</p>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Color:</strong> {color}</p>
        <p><strong>Customer Name:</strong> {customer}</p>
        <p><strong>Notes:</strong> {notes}</p>
        <p><strong>AI Model:</strong> {model_info}</p>
        <p><strong>Overall Assessment:</strong> {overall_pred}</p>
        <p><strong>Overall Confidence:</strong> {overall_conf}</p>
        <table>
            <thead>
                <tr>
                    <th>Panel Label</th>
                    <th>Panel Key</th>
                    <th>Assessment</th>
                    <th>Confidence</th>
                    <th>Filename</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """.format(
        timestamp=html.escape(datetime.now().isoformat(timespec="seconds")),
        claim_id=html.escape(str(claim_id)),
        vin=html.escape(str(vin)),
        year=html.escape(str(year)),
        make=html.escape(str(make)),
        model_name=html.escape(str(model_name)),
        color=html.escape(str(color)),
        customer=html.escape(str(customer)),
        notes=html.escape(str(notes)),
        model_info=html.escape(str(model_info)),
        overall_pred=html.escape(DISPLAY_NAMES.get(overall_pred, overall_pred)),
        overall_conf="{:.2%}".format(overall_conf),
        rows="".join(row_html),
    )


try:
    if Path("logo.png").exists():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("logo.png", width="stretch")
except Exception:
    pass

st.caption("AI-assisted hail triage beta — human review required")

st.subheader("Vehicle / Claim Intake")

col1, col2 = st.columns(2)
vehicle_claim_id = col1.text_input("Claim ID")
vin = col2.text_input("VIN")

col3, col4, col5 = st.columns(3)
year = col3.text_input("Year")
make = col4.text_input("Make")
model_name = col5.text_input("Model")

color = st.text_input("Color")
customer = st.text_input("Customer Name")
notes = st.text_area("Notes", height=100)

with st.expander("AI Model Info"):
    st.write(model_info)

st.subheader("Guided Panel Upload")
st.caption("Each panel has 3 upload slots for phone-friendly beta testing.")

results = []

for panel_key, panel_label in PANEL_CONFIG:
    st.markdown("### " + panel_label)

    slot1, slot2, slot3 = st.columns(3)

    uploader_specs = [
        (slot1, 1),
        (slot2, 2),
        (slot3, 3),
    ]

    for container, slot_number in uploader_specs:
        with container:
            file = st.file_uploader(
                f"{panel_label} Photo {slot_number}",
                key=f"{panel_key}_slot_{slot_number}_{st.session_state['reset_counter']}",
                accept_multiple_files=False,
                type=["jpg", "jpeg", "png", "webp"]
            )

            if file is not None:
                try:
                    img = prepare_uploaded_image(file)
                    pred, conf, prob_map = predict(img)

                    results.append({
                        "panel": panel_key,
                        "label": panel_label,
                        "prediction": pred,
                        "confidence": conf,
                        "prob_map": prob_map,
                        "image": img,
                        "filename": file.name,
                        "instance_label": f"{panel_label} Photo {slot_number}",
                    })
                except Exception:
                    st.warning(f"Could not process {panel_label} Photo {slot_number}")

if results:
    overall_pred, overall_conf, overall_probs = aggregate_results(results)

    st.subheader("Overall Vehicle Assessment")

    if overall_pred is not None:
        css_class = DISPLAY_CLASSES.get(overall_pred, "assessment-yellow")
        st.markdown(
            "<div class='assessment-box {}'>Overall Assessment: {}<br>Confidence: {:.2%}</div>".format(
                css_class,
                DISPLAY_NAMES.get(overall_pred, overall_pred),
                overall_conf
            ),
            unsafe_allow_html=True
        )

        with st.expander("Overall Probability Breakdown"):
            for name in class_names:
                st.write(DISPLAY_NAMES.get(name, name) + ": " + "{:.2%}".format(overall_probs.get(name, 0.0)))
    else:
        st.warning("AI model did not load on this app instance. The UI is still available.")

    st.subheader("AI Results")

    for item in results:
        c1, c2 = st.columns([1.5, 1.0])

        with c1:
            st.image(item["image"], caption=item["filename"], width="stretch")

        with c2:
            css_class = DISPLAY_CLASSES.get(item["prediction"], "assessment-yellow")
            st.write("**Panel:**", item["instance_label"])
            st.markdown(
                "<div class='assessment-box {}'>AI Assessment: {}<br>Confidence: {:.2%}</div>".format(
                    css_class,
                    DISPLAY_NAMES.get(item["prediction"], item["prediction"]),
                    item["confidence"]
                ),
                unsafe_allow_html=True
            )

            with st.expander("Probability Breakdown"):
                for name in class_names:
                    st.write(DISPLAY_NAMES.get(name, name) + ": " + "{:.2%}".format(item["prob_map"].get(name, 0.0)))

            st.markdown("**Correction / Retraining**")
            r1, r2, r3 = st.columns(3)

            with r1:
                if st.button("Mark PDR", key="g_" + item["panel"] + "_" + item["filename"] + "_" + item["instance_label"]):
                    saved_to = save_feedback_image(item, "green_pdr")
                    st.success("Saved to " + saved_to)

            with r2:
                if st.button("Mark Review", key="y_" + item["panel"] + "_" + item["filename"] + "_" + item["instance_label"]):
                    saved_to = save_feedback_image(item, "yellow_review")
                    st.success("Saved to " + saved_to)

            with r3:
                if st.button("Mark Conventional", key="r_" + item["panel"] + "_" + item["filename"] + "_" + item["instance_label"]):
                    saved_to = save_feedback_image(item, "red_conventional")
                    st.success("Saved to " + saved_to)

        st.markdown("---")

    st.subheader("Summary / Export")

    summary_text = make_summary_text(
        vehicle_claim_id, vin, year, make, model_name, color, customer, notes, results, model_info, overall_pred, overall_conf
    )
    summary_html = make_summary_html(
        vehicle_claim_id, vin, year, make, model_name, color, customer, notes, results, model_info, overall_pred, overall_conf
    )

    st.download_button(
        "Download Summary (.txt)",
        data=summary_text,
        file_name="hail_path_summary.txt",
        mime="text/plain",
        width="stretch"
    )

    st.download_button(
        "Download Summary (.html)",
        data=summary_html,
        file_name="hail_path_summary.html",
        mime="text/html",
        width="stretch"
    )

if st.button("Start Next Vehicle", width="stretch"):
    trigger_reset()
    st.rerun()