from icrawler.builtin import BingImageCrawler
import os
import hashlib
from PIL import Image

SAVE_DIR = r"dataset\yellow_review"

YELLOW_QUERIES = [
    "moderate hail damage car",
    "borderline hail damage pdr",
    "hail damage body line car",
    "hail damage moderate severity hood",
    "hail damage moderate severity roof",
    "hail damage quarter panel borderline",
    "hail dents mixed severity vehicle",
    "hail damage uncertain repair path",
    "hail damage questionable paintless dent repair",
    "hail damage review needed car",
    "hail damage body shop or pdr",
    "moderate hail dents on hood",
    "moderate hail dents on roof",
    "hail damage around body line",
    "hail damage edge dents moderate",
    "hail damage with crowns car",
    "hail damage moderate quarter panel",
    "hail damage repair decision vehicle",
    "hail dents moderate panel damage",
    "hail damage pdr candidate borderline"
]

MAX_PER_QUERY = 150
MIN_WIDTH = 400
MIN_HEIGHT = 300

def hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_images(folder):
    seen = set()
    removed = 0

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                os.remove(path)
                removed += 1
                continue

            digest = hash_file(path)
            if digest in seen:
                os.remove(path)
                removed += 1
                continue

            seen.add(digest)

        except Exception:
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass

    print(f"Cleaned {folder} | removed {removed} files")

os.makedirs(SAVE_DIR, exist_ok=True)

for query in YELLOW_QUERIES:
    print(f"\n[yellow_review] downloading: {query}")

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

clean_images(SAVE_DIR)

print("\nDone populating yellow_review.")