from icrawler.builtin import BingImageCrawler
from PIL import Image
import os
import hashlib
import shutil

SAVE_DIR = r"raw_intake\yellow_pool"
REJECT_DIR = r"raw_intake\rejected\yellow_pool"

QUERIES = [
    "moderate hail damage vehicle",
    "borderline hail damage car",
    "hail damage mixed severity",
    "hail damage body line moderate",
    "hail damage uncertain repair path",
    "hail damage moderate quarter panel",
    "hail dents moderate severity car",
    "hail damage questionable pdr",
    "hail damage repair decision vehicle",
    "hail damage moderate roof dents",
    "hail damage moderate hood dents",
    "hail damage review needed car",
    "hail damage borderline conventional repair",
    "hail damage borderline pdr candidate"
]

MAX_PER_QUERY = 100
MIN_WIDTH = 500
MIN_HEIGHT = 400
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


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


def clean_pool():
    seen_exact = set()
    seen_visual = set()
    kept = 0
    removed = 0

    for name in os.listdir(SAVE_DIR):
        path = os.path.join(SAVE_DIR, name)

        if not os.path.isfile(path):
            continue

        if not name.lower().endswith(VALID_EXTS):
            safe_move(path, REJECT_DIR)
            removed += 1
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                safe_move(path, REJECT_DIR)
                removed += 1
                continue

            exact = file_hash(path)
            if exact in seen_exact:
                safe_move(path, REJECT_DIR)
                removed += 1
                continue
            seen_exact.add(exact)

            visual = perceptual_hash(path)
            if visual is None:
                safe_move(path, REJECT_DIR)
                removed += 1
                continue

            if visual in seen_visual:
                safe_move(path, REJECT_DIR)
                removed += 1
                continue
            seen_visual.add(visual)

            kept += 1

        except Exception:
            safe_move(path, REJECT_DIR)
            removed += 1

    print(f"\nYellow clean complete | kept {kept} | removed/moved {removed}")


os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

for query in QUERIES:
    print(f"\nDownloading: {query}")

    crawler = BingImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={"root_dir": SAVE_DIR}
    )

    crawler.crawl(
        keyword=query,
        max_num=MAX_PER_QUERY,
        min_size=(MIN_WIDTH, MIN_HEIGHT)
    )

clean_pool()
print("\nDone.")