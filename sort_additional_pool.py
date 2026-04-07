import os
import shutil
import cv2

SOURCE_FOLDERS = [
    r"raw_intake\green_pool",
    r"raw_intake\yellow_pool",
]

GREEN_DIR = r"dataset\green_pdr"
YELLOW_DIR = r"dataset\yellow_review"
RED_DIR = r"dataset\red_conventional"
REJECT_DIR = r"raw_intake\rejected\manual_rejects"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

os.makedirs(GREEN_DIR, exist_ok=True)
os.makedirs(YELLOW_DIR, exist_ok=True)
os.makedirs(RED_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)


def count_files(folder):
    if not os.path.exists(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    )


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
    return dst


def resize_for_screen(img, max_width=1400, max_height=900):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


before_green = count_files(GREEN_DIR)
before_yellow = count_files(YELLOW_DIR)
before_red = count_files(RED_DIR)

print("Dataset counts BEFORE sorting:")
print(f"  green_pdr: {before_green}")
print(f"  yellow_review: {before_yellow}")
print(f"  red_conventional: {before_red}")

added_green = 0
added_yellow = 0
added_red = 0
rejected = 0

for folder in SOURCE_FOLDERS:
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
            rejected += 1
            continue

        img = resize_for_screen(img)

        cv2.imshow("1=GREEN  2=YELLOW  3=RED  D=REJECT  Q=QUIT", img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("1"):
            safe_move(path, GREEN_DIR)
            added_green += 1

        elif key == ord("2"):
            safe_move(path, YELLOW_DIR)
            added_yellow += 1

        elif key == ord("3"):
            safe_move(path, RED_DIR)
            added_red += 1

        elif key == ord("d"):
            safe_move(path, REJECT_DIR)
            rejected += 1

        elif key == ord("q"):
            print("\nStopped by user.")
            after_green = count_files(GREEN_DIR)
            after_yellow = count_files(YELLOW_DIR)
            after_red = count_files(RED_DIR)

            print("\nAdded this run:")
            print(f"  green_pdr: +{added_green}")
            print(f"  yellow_review: +{added_yellow}")
            print(f"  red_conventional: +{added_red}")
            print(f"  rejected: +{rejected}")

            print("\nDataset counts AFTER sorting:")
            print(f"  green_pdr: {after_green}")
            print(f"  yellow_review: {after_yellow}")
            print(f"  red_conventional: {after_red}")
            raise SystemExit

        else:
            safe_move(path, REJECT_DIR)
            rejected += 1

after_green = count_files(GREEN_DIR)
after_yellow = count_files(YELLOW_DIR)
after_red = count_files(RED_DIR)

print("\nSorting complete.")

print("\nAdded this run:")
print(f"  green_pdr: +{added_green}")
print(f"  yellow_review: +{added_yellow}")
print(f"  red_conventional: +{added_red}")
print(f"  rejected: +{rejected}")

print("\nDataset counts AFTER sorting:")
print(f"  green_pdr: {after_green}")
print(f"  yellow_review: {after_yellow}")
print(f"  red_conventional: {after_red}")