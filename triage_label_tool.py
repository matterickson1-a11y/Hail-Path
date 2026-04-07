import os
import shutil
import cv2

BASE_DIR = "dataset"

green_dir = os.path.join(BASE_DIR, "green_pdr")
yellow_dir = os.path.join(BASE_DIR, "yellow_review")
red_dir = os.path.join(BASE_DIR, "red_conventional")

folders = [green_dir, yellow_dir, red_dir]
image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def move_image(src, dst):
    name = os.path.basename(src)
    base, ext = os.path.splitext(name)
    dest = os.path.join(dst, name)

    counter = 1
    while os.path.exists(dest):
        dest = os.path.join(dst, f"{base}_{counter}{ext}")
        counter += 1

    shutil.move(src, dest)

def review_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(image_exts)]

    for f in files:
        path = os.path.join(folder, f)

        if not os.path.isfile(path):
            continue

        img = cv2.imread(path)

        if img is None:
            os.remove(path)
            continue

        cv2.imshow("HAIL PATH TRIAGE | 1=GREEN 2=YELLOW 3=RED D=DELETE Q=QUIT", img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("1"):
            move_image(path, green_dir)

        elif key == ord("2"):
            move_image(path, yellow_dir)

        elif key == ord("3"):
            move_image(path, red_dir)

        elif key == ord("d"):
            os.remove(path)

        elif key == ord("q"):
            cv2.destroyAllWindows()
            return

        cv2.destroyAllWindows()

for folder in folders:
    review_folder(folder)

print("Review complete.")