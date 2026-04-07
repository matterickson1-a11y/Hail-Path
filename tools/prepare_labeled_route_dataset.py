import csv
import random
import shutil
from pathlib import Path

LABEL_FILES = [
    Path("labels.csv"),
    Path("labels_expansion.csv"),
]

FEEDBACK_DIR = Path("retraining_feedback")

OUTPUT_DIR = Path("training_data_route")
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

VALID_ROUTE_LABELS = {"green_pdr", "yellow_review", "red_conventional"}
VALID_PANELS = {"roof", "roof_rail", "hood", "decklid", "quarter", "door", "fender", "other"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(42)

def resolve_image_path(raw_path):
    p = Path(raw_path)

    if p.exists():
        return p

    alt1 = Path(".") / p
    if alt1.exists():
        return alt1

    alt2 = Path(str(raw_path).replace("/", "\\"))
    if alt2.exists():
        return alt2

    return None

def safe_copy(src, dst_folder):
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst_path = dst_folder / src.name

    if not dst_path.exists():
        shutil.copy2(src, dst_path)
        return

    stem = src.stem
    suffix = src.suffix
    counter = 1
    while True:
        candidate = dst_folder / (stem + "_" + str(counter) + suffix)
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return
        counter += 1

def add_label_file_rows(rows):
    skipped_invalid_panel = 0
    skipped_invalid_route = 0
    skipped_missing_image = 0
    label_files_used = 0

    for labels_file in LABEL_FILES:
        if not labels_file.exists():
            continue

        label_files_used += 1

        with open(labels_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_image_path = row["image_path"].strip()
                panel = row["panel"].strip().lower()
                route = row["route"].strip().lower()

                if panel not in VALID_PANELS:
                    skipped_invalid_panel += 1
                    continue

                if route not in VALID_ROUTE_LABELS:
                    skipped_invalid_route += 1
                    continue

                image_path = resolve_image_path(raw_image_path)
                if image_path is None:
                    skipped_missing_image += 1
                    continue

                if image_path.suffix.lower() not in IMAGE_EXTS:
                    skipped_missing_image += 1
                    continue

                rows.append({
                    "image_path": image_path,
                    "panel": panel,
                    "route": route,
                    "source": "label_file",
                })

    return label_files_used, skipped_invalid_panel, skipped_invalid_route, skipped_missing_image

def add_feedback_rows(rows):
    feedback_count = 0

    if not FEEDBACK_DIR.exists():
        return feedback_count

    for class_name in VALID_ROUTE_LABELS:
        class_dir = FEEDBACK_DIR / class_name
        if not class_dir.exists():
            continue

        for image_path in class_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            name_lower = image_path.name.lower()

            panel = "other"
            if "roof_rail" in name_lower or "rail" in name_lower:
                panel = "roof_rail"
            elif "decklid" in name_lower or "trunk" in name_lower:
                panel = "decklid"
            elif "hood" in name_lower:
                panel = "hood"
            elif "quarter" in name_lower or "qp" in name_lower:
                panel = "quarter"
            elif "door" in name_lower:
                panel = "door"
            elif "fender" in name_lower:
                panel = "fender"
            elif "roof" in name_lower:
                panel = "roof"

            rows.append({
                "image_path": image_path,
                "panel": panel,
                "route": class_name,
                "source": "feedback",
            })
            feedback_count += 1

    return feedback_count

def main():
    rows = []

    label_files_used, skipped_invalid_panel, skipped_invalid_route, skipped_missing_image = add_label_file_rows(rows)
    feedback_count = add_feedback_rows(rows)

    if label_files_used == 0 and feedback_count == 0:
        print("No label files or feedback images found.")
        return

    if not rows:
        print("No valid labeled rows found.")
        print("Skipped invalid panel:", skipped_invalid_panel)
        print("Skipped invalid route:", skipped_invalid_route)
        print("Skipped missing image:", skipped_missing_image)
        return

    by_class = {}
    for row in rows:
        by_class.setdefault(row["route"], []).append(row)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    summary = {}

    for class_name, items in by_class.items():
        random.shuffle(items)

        val_count = max(1, round(len(items) * 0.2))
        if len(items) == 1:
            val_count = 0

        val_items = items[:val_count]
        train_items = items[val_count:]

        if len(train_items) == 0 and len(val_items) > 0:
            train_items = val_items[:1]
            val_items = val_items[1:]

        for item in train_items:
            safe_copy(item["image_path"], TRAIN_DIR / class_name)

        for item in val_items:
            safe_copy(item["image_path"], VAL_DIR / class_name)

        summary[class_name] = {
            "total": len(items),
            "train": len(train_items),
            "val": len(val_items),
        }

    print()
    print("Done.")
    print("Prepared route training dataset at:", OUTPUT_DIR.resolve())
    print("Label files used:", label_files_used)
    print("Feedback images used:", feedback_count)
    print()
    print("Skipped invalid panel:", skipped_invalid_panel)
    print("Skipped invalid route:", skipped_invalid_route)
    print("Skipped missing image:", skipped_missing_image)
    print()

    total_train = 0
    total_val = 0
    total_all = 0

    for class_name in sorted(summary.keys()):
        s = summary[class_name]
        total_all += s["total"]
        total_train += s["train"]
        total_val += s["val"]
        print(class_name)
        print("  total:", s["total"])
        print("  train:", s["train"])
        print("  val:", s["val"])

    print()
    print("ALL")
    print("  total:", total_all)
    print("  train:", total_train)
    print("  val:", total_val)

if __name__ == "__main__":
    main()