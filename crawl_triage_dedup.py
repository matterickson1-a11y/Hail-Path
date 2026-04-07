from icrawler.builtin import BingImageCrawler
from PIL import Image
import os
import hashlib
import shutil

BASE_DIR = "raw_crawl"

SEARCH_GROUPS = {
    "green_candidates": [
        "light hail damage hood pdr",
        "minor hail dents roof paintless dent repair",
        "repairable hail damage pdr candidate",
        "hail dents no paint damage car",
        "shallow hail dents hood vehicle",
        "light hail damage roof car"
    ],
    "yellow_candidates": [
        "moderate hail damage vehicle",
        "borderline hail damage pdr",
        "hail damage body line moderate",
        "hail dents mixed severity car",
        "hail damage uncertain repair path",
        "hail damage moderate quarter panel"
    ],
    "red_candidates": [
        "severe hail damage roof sharp dents",
        "hail damage broken paint vehicle",
        "heavy hail damage quarter panel",
        "hail damage body line severe",
        "hail damage roof rail severe",
        "hail damage edge dents car",
        "extreme hail damage hood conventional repair",
        "hail damage sharp crowns conventional",
        "hail damage stretched metal car",
        "hail damage severe quarter panel body shop"
    ]
}

MAX_PER_QUERY = 100
MIN_WIDTH = 500
MIN_HEIGHT = 400
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


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


def clean_folder(folder, rejected_folder):
    seen_file_hashes = set()
    seen_image_hashes = set()
    removed = 0
    kept = 0

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        if not name.lower().endswith(VALID_EXTS):
            safe_move(path, rejected_folder)
            removed += 1
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                safe_move(path, rejected_folder)
                removed += 1
                continue

            exact_hash = file_hash(path)
            if exact_hash in seen_file_hashes:
                safe_move(path, rejected_folder)
                removed += 1
                continue

            seen_file_hashes.add(exact_hash)

            p_hash = perceptual_hash(path)
            if p_hash is None:
                safe_move(path, rejected_folder)
                removed += 1
                continue

            if p_hash in seen_image_hashes:
                safe_move(path, rejected_folder)
                removed += 1
                continue

            seen_image_hashes.add(p_hash)
            kept += 1

        except Exception:
            safe_move(path, rejected_folder)
            removed += 1

    print(f"Cleaned {folder} | kept {kept} | removed/moved {removed}")


for bucket, queries in SEARCH_GROUPS.items():
    save_dir = os.path.join(BASE_DIR, bucket)
    rejected_dir = os.path.join(BASE_DIR, "rejected", bucket)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    for query in queries:
        print(f"\n[{bucket}] downloading: {query}")

        crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": save_dir}
        )

        crawler.crawl(
            keyword=query,
            max_num=MAX_PER_QUERY,
            min_size=(MIN_WIDTH, MIN_HEIGHT)
        )

    clean_folder(save_dir, rejected_dir)

print("\nDone. Sort from raw_crawl into dataset folders next.")