from io import BytesIO
from pathlib import Path
from datetime import datetime
import html
import base64
import csv

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

st.set_page_config(page_title="HAIL Path Beta", layout="wide")

BUILD_VERSION = "HAIL Path Beta Build 2026-04-20"
SESSION_LOG_FILE = Path("hail_path_beta_session_log.csv")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    .beta-box {
        padding: 14px;
        border-radius: 10px;
        background-color: rgba(255, 193, 7, 0.16);
        border-left: 6px solid #ffc107;
        margin: 12px 0;
        font-weight: 600;
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

ROUTE_MODEL_CANDIDATES = [
    Path("models/hail_path_triage_STABLE_20260320_feedback.pth"),
    Path("models/hail_path_triage_STABLE_20260317.pth"),
    Path("models/hail_path_triage_UPDATED_20260413.pth"),
    Path("models/hail_path_triage_pilot.pth"),
    Path("models/hail_path_triage.pth"),
]

FEEDBACK_DIR = Path("retraining_feedback")
LOGO_PATH = Path("logo.png")

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

MAX_UPLOAD_IMAGE_SIZE = 640
JPEG_QUALITY = 76
DISPLAY_IMAGE_WIDTH = 420
LOGO_WIDTH = 380

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0

def get_beta_password():
    try:
        return st.secrets.get("BETA_PASSWORD", "hailpathbeta")
    except Exception:
        return "hailpathbeta"

def login_screen():
    try:
        if LOGO_PATH.exists():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image("logo.png", width=LOGO_WIDTH)
    except Exception:
        pass

    st.subheader("HAIL Path Beta Access")
    st.markdown(
        """
        <div class='beta-box'>
        AI-assisted hail triage beta. Authorized testers only.
        Human review required. This is not a final claim decision tool.
        </div>
        """,
        unsafe_allow_html=True
    )

    entered = st.text_input("Beta Password", type="password")

    if st.button("Enter Beta"):
        if entered == get_beta_password():
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid beta password.")

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

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

    return Image.open(buffer).convert("RGB")

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

def get_logo_base64():
    if not LOGO_PATH.exists():
        return None
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def log_beta_session(row):
    file_exists = SESSION_LOG_FILE.exists()
    fieldnames = [
        "timestamp",
        "tester_name",
        "tester_company",
        "claim_id",
        "vin",
        "year",
        "make",
        "model",
        "color",
        "customer_name",
        "overall_assessment",
        "overall_confidence",
        "photo_count",
        "ai_helpful",
        "tester_notes",
        "build_version",
        "model_info",
    ]

    with open(SESSION_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def make_summary_text(tester_name, tester_company, claim_id, vin, year, make, model_name, color, customer, notes, results, model_info, overall_pred, overall_conf, ai_helpful, tester_notes):
    lines = []
    lines.append("HAIL PATH TRIAGE SUMMARY")
    lines.append("")
    lines.append("Beta Notice: AI-assisted preliminary triage only. Human review required.")
    lines.append("Build: " + BUILD_VERSION)
    lines.append("")
    lines.append("Timestamp: " + datetime.now().isoformat(timespec="seconds"))
    lines.append("Tester Name: " + str(tester_name))
    lines.append("Tester Company: " + str(tester_company))
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
    lines.append("AI Helpful: " + str(ai_helpful))
    lines.append("Tester Notes: " + str(tester_notes))
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

def make_summary_html(tester_name, tester_company, claim_id, vin, year, make, model_name, color, customer, notes, results, model_info, overall_pred, overall_conf, ai_helpful, tester_notes):
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

    logo_html = ""
    logo_b64 = get_logo_base64()
    if logo_b64:
        logo_html = f"<img src='data:image/png;base64,{logo_b64}' style='max-width:240px; height:auto; margin-bottom:14px;'>"

    return """
    <html>
    <head>
        <title>HAIL Path Beta Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .notice {{
                padding: 12px;
                background: #fff3cd;
                border-left: 6px solid #ffc107;
                border-radius: 8px;
                margin-bottom: 16px;
                font-weight: 600;
            }}
            .summary-box {{
                padding: 14px;
                border-radius: 10px;
                margin: 14px 0;
                background: #f4f4f4;
                border-left: 6px solid #444;
                font-weight: 600;
            }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            {logo_html}
            <h1>HAIL Path Beta Triage Summary</h1>
        </div>

        <div class="notice">
            AI-assisted preliminary triage only. Human review required. Not a final claim decision.
        </div>

        <p><strong>Build:</strong> {build_version}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Tester:</strong> {tester_name}</p>
        <p><strong>Company:</strong> {tester_company}</p>
        <p><strong>Claim ID:</strong> {claim_id}</p>
        <p><strong>VIN:</strong> {vin}</p>
        <p><strong>Year:</strong> {year}</p>
        <p><strong>Make:</strong> {make}</p>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Color:</strong> {color}</p>
        <p><strong>Customer Name:</strong> {customer}</p>
        <p><strong>Notes:</strong> {notes}</p>
        <p><strong>AI Model:</strong> {model_info}</p>

        <div class="summary-box">
            Overall Assessment: {overall_pred}<br>
            Overall Confidence: {overall_conf}<br>
            AI Helpful: {ai_helpful}<br>
            Tester Notes: {tester_notes}
        </div>

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
        logo_html=logo_html,
        build_version=html.escape(BUILD_VERSION),
        timestamp=html.escape(datetime.now().isoformat(timespec="seconds")),
        tester_name=html.escape(str(tester_name)),
        tester_company=html.escape(str(tester_company)),
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
        ai_helpful=html.escape(str(ai_helpful)),
        tester_notes=html.escape(str(tester_notes)),
        rows="".join(row_html),
    )

def render_assessment_box(prediction, confidence, label_prefix):
    pretty = DISPLAY_NAMES.get(prediction, prediction)

    if prediction == "green_pdr":
        st.success(f"{label_prefix}: {pretty} | Confidence: {confidence:.2%}")
    elif prediction == "yellow_review":
        st.warning(f"{label_prefix}: {pretty} | Confidence: {confidence:.2%}")
    elif prediction == "red_conventional":
        st.error(f"{label_prefix}: {pretty} | Confidence: {confidence:.2%}")
    else:
        st.info(f"{label_prefix}: {pretty} | Confidence: {confidence:.2%}")

try:
    if LOGO_PATH.exists():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("logo.png", width=LOGO_WIDTH)
except Exception:
    pass

st.caption("AI-assisted hail triage beta — human review required")

st.markdown(
    """
    <div class='beta-box'>
    HAIL Path Beta: AI-assisted preliminary hail triage only. 
    Human review is required. This tool is not a final claim decision or estimate.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("**Build:**", BUILD_VERSION)
st.write("**Model:**", model_info)

st.subheader("Tester Information")

tc1, tc2 = st.columns(2)
tester_name = tc1.text_input("Tester Name")
tester_company = tc2.text_input("Company")

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

st.subheader("Guided Panel Upload")
st.caption("Each panel opens into 3 compact photo slots for cleaner phone testing.")

results = []

for panel_key, panel_label in PANEL_CONFIG:
    with st.expander(panel_label, expanded=False):
        slot1, slot2, slot3 = st.columns(3)
        uploader_specs = [(slot1, 1), (slot2, 2), (slot3, 3)]

        for container, slot_number in uploader_specs:
            with container:
                file = st.file_uploader(
                    f"Photo {slot_number}",
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

                        st.image(img, caption=f"Photo {slot_number}", width=140)
                    except Exception:
                        st.warning(f"Could not process {panel_label} Photo {slot_number}")

if results:
    overall_pred, overall_conf, overall_probs = aggregate_results(results)

    st.subheader("Overall Vehicle Assessment")

    if overall_pred is not None:
        render_assessment_box(overall_pred, overall_conf, "Overall Assessment")
    else:
        st.warning("AI model did not load on this app instance. The UI is still available.")

    st.subheader("AI Results")

    for item in results:
        c1, c2 = st.columns([1.5, 1.0])

        with c1:
            st.image(item["image"], caption=item["filename"], width=DISPLAY_IMAGE_WIDTH)

        with c2:
            st.write("**Panel:**", item["instance_label"])
            render_assessment_box(item["prediction"], item["confidence"], "AI Assessment")

            with st.expander("Probability Breakdown"):
                for name in class_names:
                    st.write(DISPLAY_NAMES.get(name, name) + ": " + "{:.2%}".format(item["prob_map"].get(name, 0.0)))

            st.markdown("**Correction / Retraining**")
            r1, r2, r3 = st.columns(3)

            unique_key = item["panel"] + "_" + item["filename"] + "_" + item["instance_label"]

            with r1:
                if st.button("Mark PDR", key="g_" + unique_key):
                    saved_to = save_feedback_image(item, "green_pdr")
                    st.success("Saved to " + saved_to)

            with r2:
                if st.button("Mark Review", key="y_" + unique_key):
                    saved_to = save_feedback_image(item, "yellow_review")
                    st.success("Saved to " + saved_to)

            with r3:
                if st.button("Mark Conventional", key="r_" + unique_key):
                    saved_to = save_feedback_image(item, "red_conventional")
                    st.success("Saved to " + saved_to)

        st.markdown("---")

    st.subheader("Beta Feedback")

    ai_helpful = st.selectbox(
        "Was the AI result useful?",
        ["Not answered", "Yes", "Somewhat", "No"]
    )

    tester_notes = st.text_area("Tester Feedback / Notes", height=120)

    if st.button("Save Beta Session Log"):
        log_beta_session({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tester_name": tester_name,
            "tester_company": tester_company,
            "claim_id": vehicle_claim_id,
            "vin": vin,
            "year": year,
            "make": make,
            "model": model_name,
            "color": color,
            "customer_name": customer,
            "overall_assessment": DISPLAY_NAMES.get(overall_pred, overall_pred),
            "overall_confidence": "{:.2%}".format(overall_conf),
            "photo_count": len(results),
            "ai_helpful": ai_helpful,
            "tester_notes": tester_notes,
            "build_version": BUILD_VERSION,
            "model_info": model_info,
        })
        st.success("Beta session logged.")

    st.subheader("Summary / Export")

    summary_text = make_summary_text(
        tester_name, tester_company, vehicle_claim_id, vin, year, make, model_name, color, customer,
        notes, results, model_info, overall_pred, overall_conf, ai_helpful, tester_notes
    )

    summary_html = make_summary_html(
        tester_name, tester_company, vehicle_claim_id, vin, year, make, model_name, color, customer,
        notes, results, model_info, overall_pred, overall_conf, ai_helpful, tester_notes
    )

    st.download_button(
        "Download Summary (.txt)",
        data=summary_text,
        file_name="hail_path_beta_summary.txt",
        mime="text/plain"
    )

    st.download_button(
        "Download Branded Summary (.html)",
        data=summary_html,
        file_name="hail_path_beta_summary.html",
        mime="text/html"
    )

if st.button("Start Next Vehicle"):
    trigger_reset()
    st.rerun()