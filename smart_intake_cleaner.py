import os
import sys
import shutil
import hashlib
from PIL import Image
import cv2

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MIN_WIDTH = 500
MIN_HEIGHT = 400
BLUR_THRESHOLD = 60.0


def safe_move(src, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    name = os.path.basename(src)
    base, ext = os.path.splitext(name)
    dst = os.path.join(dst_folder, name)

    counter = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_folder, f"{base}_{counter}{ext}")
        counter += 1

    shutil.move(src, dst)


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path, size=8):
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((size, size))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            return hex(int(bits, 2))
    except Exception:
        return None


def blur_score(path):
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    if len(sys.argv) < 2:
        print("Usage: python smart_intake_cleaner.py <source_folder>")
        print(r"Example: python smart_intake_cleaner.py raw_intake\density_low")
        sys.exit(1)

    source_dir = sys.argv[1]
    reject_dir = os.path.join("raw_intake", "rejected", os.path.basename(source_dir))

    if not os.path.exists(source_dir):
        print(f"Source folder not found: {source_dir}")
        sys.exit(1)

    seen_exact = set()
    seen_visual = set()

    kept = 0
    removed = 0

    files = sorted(os.listdir(source_dir))

    for name in files:
        path = os.path.join(source_dir, name)

        if not os.path.isfile(path):
            continue

        if not name.lower().endswith(VALID_EXTS):
            safe_move(path, reject_dir)
            removed += 1
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                safe_move(path, reject_dir)
                removed += 1
                continue

            exact = file_hash(path)
            if exact in seen_exact:
                safe_move(path, reject_dir)
                removed += 1
                continue
            seen_exact.add(exact)

            visual = perceptual_hash(path)
            if visual is None:
                safe_move(path, reject_dir)
                removed += 1
                continue

            if visual in seen_visual:
                safe_move(path, reject_dir)
                removed += 1
                continue
            seen_visual.add(visual)

            score = blur_score(path)
            if score < BLUR_THRESHOLD:
                safe_move(path, reject_dir)
                removed += 1
                continue

            kept += 1

        except Exception:
            safe_move(path, reject_dir)
            removed += 1

    print(f"\nSmart clean complete | kept {kept} | removed/moved {removed}")
    print(f"Rejected folder: {reject_dir}")


if __name__ == "__main__":
    main()
