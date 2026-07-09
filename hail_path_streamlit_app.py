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

st.set_page_config(page_title="H.A.I.L. Path Pilot", layout="wide")

BUILD_VERSION = "Enterprise Polish v1 — 2026-07-09"
MODEL_VERSION_LABEL = "HAIL Route Model v0.3"
SESSION_LOG_FILE = Path("hail_path_beta_session_log.csv")
FEEDBACK_DIR = Path("retraining_feedback")
LOGO_PATH = Path("logo.png")

CLASS_NAMES_FALLBACK = ["green_pdr", "red_conventional", "yellow_review"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROUTE_MODEL_CANDIDATES = [
    Path("models/hail_path_triage_STABLE_20260320_feedback.pth"),
    Path("models/hail_path_triage_STABLE_20260317.pth"),
    Path("models/hail_path_triage_UPDATED_20260413.pth"),
    Path("models/hail_path_triage_pilot.pth"),
    Path("models/hail_path_triage.pth"),
]

DISPLAY_NAMES = {
    "green_pdr": "PDR Candidate",
    "yellow_review": "Review Recommended",
    "red_conventional": "Conventional Likely",
    "no_model": "Model Not Loaded",
}

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

CORE_PANEL_KEYS = {
    "roof",
    "hood",
    "decklid",
    "left_roof_rail",
    "right_roof_rail",
}

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
LOGO_WIDTH = 360

LOW_CONFIDENCE_THRESHOLD = 0.60
CLOSE_MARGIN_THRESHOLD = 0.12
MIN_CORE_PANEL_COUNT = 2

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; max-width: 1200px;}
    .enterprise-header {
        padding: 14px 0 8px 0;
        border-bottom: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 18px;
    }
    .notice-box {
        padding: 12px;
        border-radius: 8px;
        background: rgba(255, 193, 7, 0.14);
        border-left: 6px solid #ffc107;
        margin: 12px 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0


def get_beta_password():
    try:
        return st.secrets.get("BETA_PASSWORD", "hailpathbeta")
    except Exception:
        return "hailpathbeta"


def trigger_reset():
    st.session_state["reset_counter"] += 1


def get_logo_base64():
    if not LOGO_PATH.exists():
        return None
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def render_logo():
    try:
        if LOGO_PATH.exists():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image("logo.png", width=LOGO_WIDTH)
    except Exception:
        pass


def login_screen():
    render_logo()
    st.subheader("Pilot Access")

    st.markdown(
        """
        <div class='notice-box'>
        Authorized pilot users only. H.A.I.L. Path provides AI-assisted preliminary hail triage.
        Human review is required. This tool is not a final claim decision or estimate.
        </div>
        """,
        unsafe_allow_html=True
    )

    entered = st.text_input("Access Code", type="password")

    if st.button("Enter Pilot"):
        if entered == get_beta_password():
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid access code.")


if not st.session_state["authenticated"]:
    login_screen()
    st.stop()


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


def confidence_tier(confidence):
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


def top_two_margin(prob_map):
    if not prob_map:
        return 0.0
    values = sorted(prob_map.values(), reverse=True)
    if len(values) < 2:
        return values[0]
    return values[0] - values[1]


def apply_safety_logic(raw_pred, raw_conf, prob_map, core_count):
    reasons = []

    if raw_pred is None:
        return None, raw_conf, reasons

    margin = top_two_margin(prob_map)

    if raw_conf < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("confidence is below the preferred threshold")

    if margin < CLOSE_MARGIN_THRESHOLD:
        reasons.append("assessment probabilities are closely grouped")

    if core_count < MIN_CORE_PANEL_COUNT:
        reasons.append("photo coverage is limited")

    if reasons:
        return "yellow_review", raw_conf, reasons

    return raw_pred, raw_conf, reasons


def aggregate_results(results):
    usable = [r for r in results if r["prediction"] != "no_model"]
    if not usable:
        return None, 0.0, {}, []

    totals = {name: 0.0 for name in class_names}
    total_weight = 0.0

    for item in usable:
        weight = PANEL_WEIGHTS.get(item["panel"], 1.0)
        total_weight += weight
        for name in class_names:
            totals[name] += item["prob_map"].get(name, 0.0) * weight

    if total_weight == 0:
        return None, 0.0, {}, []

    averages = {}
    for name in class_names:
        averages[name] = totals[name] / total_weight

    raw_best = max(averages, key=averages.get)
    raw_conf = averages[raw_best]

    core_uploaded = len({r["panel"] for r in results if r["panel"] in CORE_PANEL_KEYS})
    final_pred, final_conf, reasons = apply_safety_logic(raw_best, raw_conf, averages, core_uploaded)

    return final_pred, final_conf, averages, reasons


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


def log_beta_session(row):
    file_exists = SESSION_LOG_FILE.exists()
    fieldnames = [
        "timestamp", "tester_name", "tester_company", "claim_id", "vin",
        "year", "make", "model", "color", "insured_name", "overall_assessment",
        "confidence_tier", "overall_confidence", "photo_count", "photo_coverage",
        "ai_helpful", "tester_notes", "build_version", "model_version", "model_info"
    ]

    with open(SESSION_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def photo_coverage_label(core_count, photo_count):
    if photo_count >= 8 and core_count >= 4:
        return "Strong"
    if photo_count >= 4 and core_count >= 2:
        return "Standard"
    return "Limited"


def make_summary_text(data):
    lines = []
    lines.append("H.A.I.L. PATH PRELIMINARY TRIAGE SUMMARY")
    lines.append("")
    lines.append("AI-assisted preliminary hail triage only. Human review required.")
    lines.append("Not a final claim decision. Not an estimate.")
    lines.append("")
    lines.append("Build: " + BUILD_VERSION)
    lines.append("Model Version: " + MODEL_VERSION_LABEL)
    lines.append("Timestamp: " + data["timestamp"])
    lines.append("")
    lines.append("Tester: " + str(data["tester_name"]))
    lines.append("Company: " + str(data["tester_company"]))
    lines.append("Claim Number: " + str(data["claim_id"]))
    lines.append("VIN: " + str(data["vin"]))
    lines.append("Vehicle: " + str(data["year"]) + " " + str(data["make"]) + " " + str(data["model_name"]))
    lines.append("Color: " + str(data["color"]))
    lines.append("Insured Name: " + str(data["insured_name"]))
    lines.append("Notes: " + str(data["notes"]))
    lines.append("")
    lines.append("Overall Triage Recommendation: " + data["overall_label"])
    lines.append("Confidence Tier: " + data["confidence_tier"])
    lines.append("Overall Confidence: " + data["overall_confidence"])
    lines.append("Photo Count: " + str(data["photo_count"]))
    lines.append("Photo Coverage: " + data["photo_coverage"])
    lines.append("Assessment Notes: " + data["review_reason"])
    lines.append("")
    lines.append("Panel Results:")
    for item in data["results"]:
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


def make_summary_html(data):
    row_html = []
    for item in data["results"]:
        row_html.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2%}</td><td>{}</td></tr>".format(
                html.escape(item["instance_label"]),
                html.escape(item["panel"]),
                html.escape(DISPLAY_NAMES.get(item["prediction"], item["prediction"])),
                html.escape(confidence_tier(item["confidence"])),
                item["confidence"],
                html.escape(item["filename"]),
            )
        )

    logo_html = ""
    logo_b64 = get_logo_base64()
    if logo_b64:
        logo_html = f"<img src='data:image/png;base64,{logo_b64}' style='max-width:240px; height:auto; margin-bottom:14px;'>"

    return """
    <html>
    <head>
        <title>H.A.I.L. Path Triage Summary</title>
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
                padding: 16px;
                border-radius: 10px;
                margin: 16px 0;
                background: #f4f4f4;
                border-left: 6px solid #333;
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
            <h1>Preliminary Hail Triage Summary</h1>
        </div>

        <div class="notice">
            AI-assisted preliminary hail triage only. Human review required.
            This is not a final claim decision or estimate.
        </div>

        <p><strong>Build:</strong> {build}</p>
        <p><strong>Model Version:</strong> {model_version}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Tester:</strong> {tester_name}</p>
        <p><strong>Company:</strong> {tester_company}</p>
        <p><strong>Claim Number:</strong> {claim_id}</p>
        <p><strong>VIN:</strong> {vin}</p>
        <p><strong>Vehicle:</strong> {vehicle}</p>
        <p><strong>Color:</strong> {color}</p>
        <p><strong>Insured Name:</strong> {insured_name}</p>
        <p><strong>Notes:</strong> {notes}</p>

        <div class="summary-box">
            Overall Triage Recommendation: {overall_label}<br>
            Confidence Tier: {confidence_tier}<br>
            Overall Confidence: {overall_confidence}<br>
            Photo Count: {photo_count}<br>
            Photo Coverage: {photo_coverage}<br>
            Assessment Notes: {review_reason}
        </div>

        <table>
            <thead>
                <tr>
                    <th>Photo</th>
                    <th>Panel</th>
                    <th>Assessment</th>
                    <th>Confidence Tier</th>
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
        build=html.escape(BUILD_VERSION),
        model_version=html.escape(MODEL_VERSION_LABEL),
        timestamp=html.escape(data["timestamp"]),
        tester_name=html.escape(str(data["tester_name"])),
        tester_company=html.escape(str(data["tester_company"])),
        claim_id=html.escape(str(data["claim_id"])),
        vin=html.escape(str(data["vin"])),
        vehicle=html.escape(str(data["year"]) + " " + str(data["make"]) + " " + str(data["model_name"])),
        color=html.escape(str(data["color"])),
        insured_name=html.escape(str(data["insured_name"])),
        notes=html.escape(str(data["notes"])),
        overall_label=html.escape(data["overall_label"]),
        confidence_tier=html.escape(data["confidence_tier"]),
        overall_confidence=html.escape(data["overall_confidence"]),
        photo_count=str(data["photo_count"]),
        photo_coverage=html.escape(data["photo_coverage"]),
        review_reason=html.escape(data["review_reason"]),
        rows="".join(row_html),
    )


def render_assessment(prediction, confidence, prefix):
    label = DISPLAY_NAMES.get(prediction, prediction)
    tier = confidence_tier(confidence)
    message = f"{prefix}: {label} | {tier} Confidence | {confidence:.2%}"

    if prediction == "green_pdr":
        st.success(message)
    elif prediction == "yellow_review":
        st.warning(message)
    elif prediction == "red_conventional":
        st.error(message)
    else:
        st.info(message)


render_logo()

st.markdown("<div class='enterprise-header'>", unsafe_allow_html=True)
st.subheader("H.A.I.L. Path Pilot")
st.caption("Pre-Estimate Hail Triage Platform")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='notice-box'>
    AI-assisted preliminary triage only. Human review required.
    This tool is not a final claim decision or estimate.
    </div>
    """,
    unsafe_allow_html=True
)

status_1, status_2, status_3 = st.columns(3)
status_1.info("Pilot Build")
status_2.info(BUILD_VERSION)
status_3.info(MODEL_VERSION_LABEL)

st.subheader("1. Pilot User")
u1, u2 = st.columns(2)
tester_name = u1.text_input("Tester Name")
tester_company = u2.text_input("Company")

st.subheader("2. Claim Intake")
c1, c2 = st.columns(2)
claim_id = c1.text_input("Claim Number")
vin = c2.text_input("VIN")

v1, v2, v3, v4 = st.columns(4)
year = v1.text_input("Year")
make = v2.text_input("Make")
model_name = v3.text_input("Model")
color = v4.text_input("Color")

insured_name = st.text_input("Insured Name")
notes = st.text_area("Claim / Vehicle Notes", height=90)

with st.expander("Model / Build Details"):
    st.write("Build:", BUILD_VERSION)
    st.write("Model Version:", MODEL_VERSION_LABEL)
    st.write("Loaded Model:", model_info)

st.subheader("3. Guided Photo Upload")
st.caption("Upload clear panel photos. The application will evaluate photo coverage automatically.")

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
    st.subheader("4. Triage Review")

    overall_pred, overall_conf, overall_probs, safety_reasons = aggregate_results(results)
    core_panel_count = len({r["panel"] for r in results if r["panel"] in CORE_PANEL_KEYS})
    photo_count = len(results)
    coverage = photo_coverage_label(core_panel_count, photo_count)

    if safety_reasons:
        review_reason = "Review recommended because " + ", ".join(safety_reasons) + "."
    else:
        review_reason = "No automatic review override applied."

    if overall_pred is not None:
        render_assessment(overall_pred, overall_conf, "Overall Triage Recommendation")
        st.write("**Photo Coverage:**", coverage)
        st.write("**Photo Count:**", photo_count)
        st.write("**Assessment Notes:**", review_reason)

        with st.expander("Overall Probability Details"):
            for name in class_names:
                st.write(DISPLAY_NAMES.get(name, name) + ": " + "{:.2%}".format(overall_probs.get(name, 0.0)))
    else:
        st.warning("No AI assessment available.")

    st.markdown("---")
    st.subheader("Panel-Level Results")

    for item in results:
        left, right = st.columns([1.4, 1.0])

        with left:
            st.image(item["image"], caption=item["filename"], width=DISPLAY_IMAGE_WIDTH)

        with right:
            st.write("**Photo:**", item["instance_label"])
            render_assessment(item["prediction"], item["confidence"], "Panel Assessment")

            with st.expander("Probability Details"):
                for name in class_names:
                    st.write(DISPLAY_NAMES.get(name, name) + ": " + "{:.2%}".format(item["prob_map"].get(name, 0.0)))

            st.markdown("**Correct Assessment**")
            b1, b2, b3 = st.columns(3)
            unique_key = item["panel"] + "_" + item["filename"] + "_" + item["instance_label"]

            with b1:
                if st.button("PDR Candidate", key="pdr_" + unique_key):
                    save_feedback_image(item, "green_pdr")
                    st.success("Correction saved.")

            with b2:
                if st.button("Review", key="rev_" + unique_key):
                    save_feedback_image(item, "yellow_review")
                    st.success("Correction saved.")

            with b3:
                if st.button("Conventional", key="conv_" + unique_key):
                    save_feedback_image(item, "red_conventional")
                    st.success("Correction saved.")

        st.markdown("---")

    st.subheader("5. Pilot Feedback / Export")

    ai_helpful = st.selectbox("Was this AI result useful?", ["Not answered", "Yes", "Somewhat", "No"])
    tester_notes = st.text_area("Tester Feedback", height=100)

    report_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tester_name": tester_name,
        "tester_company": tester_company,
        "claim_id": claim_id,
        "vin": vin,
        "year": year,
        "make": make,
        "model_name": model_name,
        "color": color,
        "insured_name": insured_name,
        "notes": notes,
        "overall_label": DISPLAY_NAMES.get(overall_pred, overall_pred),
        "confidence_tier": confidence_tier(overall_conf),
        "overall_confidence": "{:.2%}".format(overall_conf),
        "photo_count": photo_count,
        "photo_coverage": coverage,
        "review_reason": review_reason,
        "results": results,
    }

    if st.button("Save Pilot Session Log"):
        log_beta_session({
            "timestamp": report_data["timestamp"],
            "tester_name": tester_name,
            "tester_company": tester_company,
            "claim_id": claim_id,
            "vin": vin,
            "year": year,
            "make": make,
            "model": model_name,
            "color": color,
            "insured_name": insured_name,
            "overall_assessment": report_data["overall_label"],
            "confidence_tier": report_data["confidence_tier"],
            "overall_confidence": report_data["overall_confidence"],
            "photo_count": photo_count,
            "photo_coverage": coverage,
            "ai_helpful": ai_helpful,
            "tester_notes": tester_notes,
            "build_version": BUILD_VERSION,
            "model_version": MODEL_VERSION_LABEL,
            "model_info": model_info,
        })
        st.success("Pilot session logged.")

    summary_text = make_summary_text(report_data)
    summary_html = make_summary_html(report_data)

    st.download_button(
        "Download Text Summary",
        data=summary_text,
        file_name="hail_path_preliminary_triage_summary.txt",
        mime="text/plain"
    )

    st.download_button(
        "Download Branded HTML Summary",
        data=summary_html,
        file_name="hail_path_preliminary_triage_summary.html",
        mime="text/html"
    )

if st.button("Start Next Vehicle"):
    trigger_reset()
    st.rerun()