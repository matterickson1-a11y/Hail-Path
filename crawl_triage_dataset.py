from icrawler.builtin import BingImageCrawler
import os
import hashlib
from PIL import Image

BASE_DIR = "dataset"

SEARCH_GROUPS = {
    "green_pdr": [
        "light hail damage car hood pdr",
        "minor hail dents roof paintless dent repair",
        "hail damage shallow dents hood",
        "repairable hail damage pdr candidate",
        "hail dents no paint damage car"
    ],
    "yellow_review": [
        "moderate hail damage car roof",
        "hail damage body line vehicle",
        "hail dents mixed severity car",
        "hail damage quarter panel moderate",
        "hail damage borderline pdr"
    ],
    "red_conventional": [
        "severe hail damage vehicle roof",
        "extreme hail damage car body shop",
        "hail damage broken paint vehicle",
        "sharp hail dents edge damage",
        "heavy hail damage conventional repair"
    ]
}

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

for category, queries in SEARCH_GROUPS.items():
    save_dir = os.path.join(BASE_DIR, category)
    os.makedirs(save_dir, exist_ok=True)

    for query in queries:
        print(f"\n[{category}] downloading: {query}")

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

    clean_images(save_dir)

print("\nDone populating triage dataset folders.")