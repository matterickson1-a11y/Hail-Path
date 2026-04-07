import os
import shutil

SOURCE = r"hard_examples\false_red_should_be_green"
DEST = r"dataset\green_pdr"

DUPLICATE_MULTIPLIER = 3   # how many times to duplicate each correction image

VALID_EXTS = (".jpg",".jpeg",".png",".webp",".bmp")


def safe_copy(src, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    name = os.path.basename(src)
    base, ext = os.path.splitext(name)

    dst = os.path.join(dst_folder, name)

    counter = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_folder, f"{base}_{counter}{ext}")
        counter += 1

    shutil.copy(src, dst)
    return dst


def main():

    if not os.path.exists(SOURCE):
        print("Source folder not found.")
        return

    files = [
        f for f in os.listdir(SOURCE)
        if f.lower().endswith(VALID_EXTS)
    ]

    if not files:
        print("No images found in correction folder.")
        return

    added = 0

    for f in files:

        src_path = os.path.join(SOURCE, f)

        for i in range(DUPLICATE_MULTIPLIER):
            safe_copy(src_path, DEST)
            added += 1

    print()
    print("Correction images loaded.")
    print(f"Original files: {len(files)}")
    print(f"Total added to green dataset: {added}")
    print()


if __name__ == "__main__":
    main()