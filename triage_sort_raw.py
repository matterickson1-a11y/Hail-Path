import os
import shutil
import cv2

RAW_FOLDERS = [
    r"raw_crawl\green_candidates",
    r"raw_crawl\yellow_candidates",
    r"raw_crawl\red_candidates"
]

GREEN_DIR = r"dataset\green_pdr"
YELLOW_DIR = r"dataset\yellow_review"
RED_DIR = r"dataset\red_conventional"
REJECT_DIR = r"raw_crawl\rejected\manual_rejects"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

os.makedirs(GREEN_DIR, exist_ok=True)
os.makedirs(YELLOW_DIR, exist_ok=True)
os.makedirs(RED_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)


def safe_move(src, dst_folder):
    name = os.path.basename(src)
    base, ext = os.path.splitext(name)
    dst = os.path.join(dst_folder, name)

    counter = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_folder, f"{base}_{counter}{ext}")
        counter += 1

    shutil.move(src, dst)


def resize_for_screen(img, max_width=1400, max_height=900):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


for folder in RAW_FOLDERS:
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        continue

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    print(f"\nSorting folder: {folder} | {len(files)} files")

    for path in files:
        img = cv2.imread(path)
        if img is None:
            safe_move(path, REJECT_DIR)
            continue

        img = resize_for_screen(img)

        cv2.imshow("1=GREEN  2=YELLOW  3=RED  D=REJECT  Q=QUIT", img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("1"):
            safe_move(path, GREEN_DIR)
        elif key == ord("2"):
            safe_move(path, YELLOW_DIR)
        elif key == ord("3"):
            safe_move(path, RED_DIR)
        elif key == ord("d"):
            safe_move(path, REJECT_DIR)
        elif key == ord("q"):
            print("Stopped by user.")
            raise SystemExit
        else:
            safe_move(path, REJECT_DIR)

print("Sorting complete.")