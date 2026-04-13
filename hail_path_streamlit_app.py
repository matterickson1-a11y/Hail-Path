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

ROUTE_MODEL_CANDIDATES = [
    Path("models/hail_path_triage_STABLE_20260320_feedback.pth"),
    Path("models/hail_path_triage_STABLE_20260317.pth"),
    Path("models/hail_path_triage_pilot.pth"),
    Path("models/hail_path_triage.pth"),
]

CLASS_NAMES_FALLBACK = ["green_pdr", "red_conventional", "yellow_review"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if "reset" not in st.session_state:
    st.session_state.reset = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def do_reset():
    st.session_state.reset = True

if st.session_state.reset:
    st.session_state.uploader_key += 1
    st.session_state.reset = False

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

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
            return None, list(CLASS_NAMES_FALLBACK), "Model load failed: " + str(e)

    return None, list(CLASS_NAMES_FALLBACK), "No model found"

model, class_names, model_info = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    if model is None:
        return "no_model", 0.0

    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    idx = int(probs.argmax())
    if idx >= len(class_names):
        return "no_model", 0.0

    return class_names[idx], float(probs[idx])

def make_summary_text(claim_id, vin, year, make, model_name, color, customer, results, model_info):
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
    lines.append("AI Model: " + str(model_info))
    lines.append("")
    lines.append("Panel Results:")
    for key, label, pred, conf, img, filename in results:
        lines.append(
            "{} | {} | {} | {:.2%} | {}".format(
                label,
                key,
                DISPLAY_NAMES.get(pred, pred),
                conf,
                filename
            )
        )
    return "\n".join(lines)

def make_summary_html(claim_id, vin, year, make, model_name, color, customer, results, model_info):
    row_html = []
    for key, label, pred, conf, img, filename in results:
        row_html.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2%}</td><td>{}</td></tr>".format(
                html.escape(label),
                html.escape(key),
                html.escape(DISPLAY_NAMES.get(pred, pred)),
                conf,
                html.escape(filename)
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
        <p><strong>AI Model:</strong> {model_info}</p>
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
        model_info=html.escape(str(model_info)),
        rows="".join(row_html),
    )

if Path("logo.png").exists():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("logo.png", width=420)

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

with st.expander("AI Model Info"):
    st.write(model_info)

st.subheader("Guided Panel Upload")

results = []

for key, label in GUIDED_PANELS:
    file = st.file_uploader(
        label,
        key=key + "_" + str(st.session_state.uploader_key)
    )

    if file is not None:
        img = Image.open(BytesIO(file.getvalue()))
        img = ImageOps.exif_transpose(img).convert("RGB")
        pred, conf = predict(img)
        results.append((key, label, pred, conf, img, file.name))

if results:
    st.subheader("AI Results")

    for key, label, pred, conf, img, filename in results:
        c1, c2 = st.columns([1.5, 1.0])

        with c1:
            st.image(img, width="stretch", caption=filename)

        with c2:
            st.write("**Panel:**", label)
            st.write("**AI Assessment:**", DISPLAY_NAMES.get(pred, pred))
            st.write("**Confidence:**", f"{conf:.2%}")
            st.markdown("---")

    st.subheader("Summary / Export")

    summary_text = make_summary_text(
        vehicle_claim_id, vin, year, make, model_name, color, customer, results, model_info
    )
    summary_html = make_summary_html(
        vehicle_claim_id, vin, year, make, model_name, color, customer, results, model_info
    )

    st.download_button(
        "Download Summary (.txt)",
        data=summary_text,
        file_name="hail_path_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.download_button(
        "Download Summary (.html)",
        data=summary_html,
        file_name="hail_path_summary.html",
        mime="text/html",
        use_container_width=True
    )

if st.button("Start Next Vehicle", use_container_width=True):
    do_reset()
    st.rerun()