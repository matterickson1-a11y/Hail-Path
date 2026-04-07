import shutil
from pathlib import Path

SOURCE_DIR = Path("session_uploads")
OUTPUT_DIR = Path("dataset_sorted")

# More specific matches must come first
CATEGORIES = [
    ("roof_rail", ["left_roof_rail", "right_roof_rail", "roof_rail"]),
    ("decklid", ["decklid"]),
    ("hood", ["hood"]),
    ("quarter", ["quarter", "qp"]),
    ("door", ["door"]),
    ("fender", ["fender"]),
    ("roof", ["roof"]),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def categorize(filename):
    name = filename.lower()
    for category, keywords in CATEGORIES:
        for keyword in keywords:
            if keyword in name:
                return category
    return "other"

def main():
    if not SOURCE_DIR.exists():
        print("Source folder not found:", SOURCE_DIR)
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    total = 0
    counts = {}

    for session_folder in SOURCE_DIR.iterdir():
        if not session_folder.is_dir():
            continue

        session_name = session_folder.name

        for file_path in session_folder.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in IMAGE_EXTS:
                continue

            total += 1
            category = categorize(file_path.name)
            counts[category] = counts.get(category, 0) + 1

            dest_folder = OUTPUT_DIR / category
            dest_folder.mkdir(parents=True, exist_ok=True)

            safe_name = session_name + "__" + file_path.name
            dest_path = dest_folder / safe_name

            counter = 1
            while dest_path.exists():
                name_only = Path(safe_name).stem
                ext_only = Path(safe_name).suffix
                dest_path = dest_folder / (name_only + "_" + str(counter) + ext_only)
                counter += 1

            shutil.copy2(file_path, dest_path)

    print()
    print("Done.")
    print("Total images copied:", total)
    print("Category counts:")
    for k in sorted(counts.keys()):
        print("  " + k + ": " + str(counts[k]))

    print()
    print("Output folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()