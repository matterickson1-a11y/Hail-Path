import os
import shutil
import sys
import cv2

if len(sys.argv) < 2:
    print("Usage: python density_label_tool.py <source_folder>")
    print(r'Example: python density_label_tool.py raw_intake\density_medium')
    sys.exit(1)

SOURCE_DIR = sys.argv[1]

LOW_DIR = r"data\density_dataset\train\low"
MEDIUM_DIR = r"data\density_dataset\train\medium"
HIGH_DIR = r"data\density_dataset\train\high"
REJECT_DIR = os.path.join(r"raw_intake\rejected", os.path.basename(SOURCE_DIR))

folders_to_make = [LOW_DIR, MEDIUM_DIR, HIGH_DIR, REJECT_DIR]
image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

for folder in folders_to_make:
    os.makedirs(folder, exist_ok=True)

def move_image(src, dst):
    name = os.path.basename(src)
    base, ext = os.path.splitext(name)
    dest = os.path.join(dst, name)

    counter = 1
    while os.path.exists(dest):
        dest = os.path.join(dst, f"{base}_{counter}{ext}")
        counter += 1

    shutil.move(src, dest)

files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(image_exts)]

for f in files:
    path = os.path.join(SOURCE_DIR, f)

    if not os.path.isfile(path):
        continue

    img = cv2.imread(path)

    if img is None:
        os.remove(path)
        continue

    cv2.imshow(f"DENSITY SORTER | {os.path.basename(SOURCE_DIR)} | 1=LOW 2=MEDIUM 3=HIGH D=DELETE Q=QUIT", img)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("1"):
        move_image(path, LOW_DIR)

    elif key == ord("2"):
        move_image(path, MEDIUM_DIR)

    elif key == ord("3"):
        move_image(path, HIGH_DIR)

    elif key == ord("d"):
        move_image(path, REJECT_DIR)

    elif key == ord("q"):
        cv2.destroyAllWindows()
        break

    cv2.destroyAllWindows()

print("Density review complete.")
