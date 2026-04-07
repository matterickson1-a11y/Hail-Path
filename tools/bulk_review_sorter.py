import shutil
from pathlib import Path

import streamlit as st
from PIL import Image

SOURCE_DIR = Path("bulk_hail_candidates/images")
OUTPUT_DIR = Path("bulk_review_sorted")

CATEGORY_OPTIONS = ["unreviewed", "roof", "hood", "decklid", "roof_rail", "keep_misc", "junk"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(page_title="HAIL Path Bulk Review Sorter", layout="wide")

def get_images():
    if not SOURCE_DIR.exists():
        return []
    files = []
    for p in SOURCE_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)

def move_file(src_path, category):
    dest_dir = OUTPUT_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name

    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / (src_path.stem + "_" + str(counter) + src_path.suffix)
        counter += 1

    shutil.move(str(src_path), str(dest_path))

def count_remaining(images):
    return len(images)

images = get_images()

st.title("HAIL Path Bulk Review Sorter")
st.caption("Review harvested hail images and move them into clean buckets")

if not images:
    st.success("No images left in bulk_hail_candidates/images")
    st.stop()

if "index" not in st.session_state:
    st.session_state.index = 0

if st.session_state.index >= len(images):
    st.session_state.index = max(0, len(images) - 1)

img_path = images[st.session_state.index]

top1, top2, top3 = st.columns([1, 1, 3])

with top1:
    if st.button("Previous"):
        st.session_state.index = max(0, st.session_state.index - 1)
        st.rerun()

with top2:
    if st.button("Next"):
        st.session_state.index = min(len(images) - 1, st.session_state.index + 1)
        st.rerun()

with top3:
    st.write("Image", str(st.session_state.index + 1), "of", str(len(images)))

left, right = st.columns([3, 2])

with left:
    st.subheader(img_path.name)
    image = Image.open(img_path)
    st.image(image, width="stretch")
    st.text(str(img_path))

with right:
    st.subheader("Sort Action")

    if st.button("Move to Roof"):
        move_file(img_path, "roof")
        st.rerun()

    if st.button("Move to Hood"):
        move_file(img_path, "hood")
        st.rerun()

    if st.button("Move to Decklid"):
        move_file(img_path, "decklid")
        st.rerun()

    if st.button("Move to Roof Rail"):
        move_file(img_path, "roof_rail")
        st.rerun()

    if st.button("Move to Keep Misc"):
        move_file(img_path, "keep_misc")
        st.rerun()

    if st.button("Move to Junk"):
        move_file(img_path, "junk")
        st.rerun()

st.write("Remaining unreviewed images:", count_remaining(images))