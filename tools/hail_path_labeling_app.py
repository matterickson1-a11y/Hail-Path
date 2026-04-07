import csv
from pathlib import Path

import streamlit as st
from PIL import Image

DEFAULT_DATASET_DIR = "bulk_review_sorted"
LABELS_FILE = Path("labels_expansion.csv")

PANEL_OPTIONS = ["roof", "roof_rail", "hood", "decklid", "quarter", "door", "fender", "other"]
ROUTE_OPTIONS = ["unlabeled", "green_pdr", "yellow_review", "red_conventional"]
DENSITY_OPTIONS = ["unlabeled", "low", "medium", "high"]

st.set_page_config(page_title="HAIL Path Labeling", layout="wide")

def get_dataset_dir():
    dataset_text = st.sidebar.text_input("Dataset folder", value=DEFAULT_DATASET_DIR)
    return Path(dataset_text.strip())

def get_images(dataset_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = []
    if dataset_dir.exists():
        for p in dataset_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)

def load_labels():
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["image_path"]] = row
    return labels

def save_labels(labels):
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "panel", "route", "density"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_path in sorted(labels.keys()):
            writer.writerow(labels[image_path])

def delete_current_image(img_path, labels):
    try:
        img_key = str(img_path).replace("\\", "/")
        if img_key in labels:
            del labels[img_key]
            save_labels(labels)

        if img_path.exists():
            img_path.unlink()

        return True, "Image deleted."
    except Exception as e:
        return False, "Delete failed: " + str(e)

dataset_dir = get_dataset_dir()
images = get_images(dataset_dir)
labels = load_labels()

st.title("HAIL Path Labeling App")
st.caption("Label the harvested/sorted expansion images")

if not images:
    st.warning("No images found in: " + str(dataset_dir))
    st.stop()

if "index" not in st.session_state:
    st.session_state.index = 0

if st.session_state.index >= len(images):
    st.session_state.index = max(0, len(images) - 1)

col_top1, col_top2, col_top3 = st.columns([1, 1, 2])

with col_top1:
    if st.button("Previous"):
        st.session_state.index = max(0, st.session_state.index - 1)
        st.rerun()

with col_top2:
    if st.button("Next"):
        st.session_state.index = min(len(images) - 1, st.session_state.index + 1)
        st.rerun()

with col_top3:
    st.write("Image", str(st.session_state.index + 1), "of", str(len(images)))
    st.write("Dataset:", str(dataset_dir))

img_path = images[st.session_state.index]
img_key = str(img_path).replace("\\", "/")
existing = labels.get(img_key, {})

left, right = st.columns([3, 2])

with left:
    st.subheader(img_path.name)
    image = Image.open(img_path)
    st.image(image, width="stretch")
    st.text(str(img_path))

with right:
    st.subheader("Labels")

    default_panel = existing.get("panel", img_path.parent.name)
    if default_panel not in PANEL_OPTIONS:
        default_panel = "other"

    default_route = existing.get("route", "unlabeled")
    if default_route not in ROUTE_OPTIONS:
        default_route = "unlabeled"

    default_density = existing.get("density", "unlabeled")
    if default_density not in DENSITY_OPTIONS:
        default_density = "unlabeled"

    panel = st.selectbox("Panel", PANEL_OPTIONS, index=PANEL_OPTIONS.index(default_panel))
    route = st.selectbox("Route", ROUTE_OPTIONS, index=ROUTE_OPTIONS.index(default_route))
    density = st.selectbox("Density", DENSITY_OPTIONS, index=DENSITY_OPTIONS.index(default_density))

    if st.button("Save Label"):
        labels[img_key] = {
            "image_path": img_key,
            "panel": panel,
            "route": route,
            "density": density,
        }
        save_labels(labels)
        st.success("Saved.")

    if st.button("Save Label and Next"):
        labels[img_key] = {
            "image_path": img_key,
            "panel": panel,
            "route": route,
            "density": density,
        }
        save_labels(labels)
        st.session_state.index = min(len(images) - 1, st.session_state.index + 1)
        st.rerun()

    st.markdown("---")

    if st.button("Delete Image", type="primary"):
        ok, msg = delete_current_image(img_path, labels)
        if ok:
            st.session_state.index = max(0, st.session_state.index - 1)
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

st.write("Labeled images:", len(labels))