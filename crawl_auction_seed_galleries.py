import os
import re
import time
import shutil
import hashlib
from io import BytesIO
from urllib.parse import urljoin

import requests
from PIL import Image
from playwright.sync_api import sync_playwright

SEED_FILE = "auction_seed_urls.txt"
SAVE_DIR = r"raw_crawl\auction_seed_galleries"
REJECT_DIR = r"raw_crawl\rejected\auction_seed_galleries"

MIN_WIDTH = 800
MIN_HEIGHT = 600
MAX_IMAGES_PER_LISTING = 100
WAIT_MS = 7000

BAD_URL_HINTS = [
    "logo",
    "icon",
    "banner",
    "thumbnail",
    "thumb",
    "sprite",
    "graphic",
    "vector",
    "promo",
    "stock",
    "avatar",
]


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


def bad_url(url):
    if not url:
        return True

    lower = url.lower()

    if lower.startswith("data:image"):
        return True

    return any(hint in lower for hint in BAD_URL_HINTS)


def make_filename(listing_index, image_index):
    return f"listing_{listing_index:03d}_img_{image_index:03d}.jpg"


def download_image(url, dst_path):
    try:
        if bad_url(url):
            return False, "bad/non-http image url"

        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")
        width, height = img.size

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False, f"too small {width}x{height}"

        img.save(dst_path, format="JPEG", quality=95)
        return True, f"saved {width}x{height}"

    except Exception as e:
        return False, str(e)


def clean_pool():
    seen_exact = set()
    seen_visual = set()
    kept = 0
    removed = 0

    for name in os.listdir(SAVE_DIR):
        path = os.path.join(SAVE_DIR, name)

        if not os.path.isfile(path):
            continue

        try:
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

    print(f"\nSeed gallery clean complete | kept {kept} | removed/moved {removed}")


def read_seed_urls():
    if not os.path.exists(SEED_FILE):
        raise FileNotFoundError(f"Missing seed file: {SEED_FILE}")

    urls = []
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and u.startswith("http"):
                urls.append(u)

    return urls


def extract_image_urls_from_listing(page, base_url):
    urls = set()

    img_nodes = page.locator("img").evaluate_all(
        """nodes => nodes.map(n =>
            n.src || n.getAttribute('src') || n.getAttribute('data-src') || n.getAttribute('data-lazy') || n.getAttribute('data-original')
        ).filter(Boolean)"""
    )

    for src in img_nodes:
        full = urljoin(base_url, src)
        if not bad_url(full):
            urls.add(full)

    html = page.content()

    raw_matches = re.findall(r'https?://[^"\\\']+\.(?:jpg|jpeg|png|webp)', html, flags=re.I)
    for m in raw_matches:
        if not bad_url(m):
            urls.add(m)

    return sorted(urls)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(REJECT_DIR, exist_ok=True)

    seed_urls = read_seed_urls()
    print(f"Seed urls loaded: {len(seed_urls)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        for listing_index, listing_url in enumerate(seed_urls, start=1):
            print(f"\nOpening listing {listing_index}: {listing_url}")

            try:
                page.goto(listing_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(WAIT_MS)
                page.mouse.wheel(0, 7000)
                page.wait_for_timeout(2500)

                image_urls = extract_image_urls_from_listing(page, listing_url)
                print(f"Found {len(image_urls)} candidate image urls")

                saved_count = 0
                for image_index, image_url in enumerate(image_urls, start=1):
                    if saved_count >= MAX_IMAGES_PER_LISTING:
                        break

                    filename = make_filename(listing_index, image_index)
                    dst_path = os.path.join(SAVE_DIR, filename)

                    ok, msg = download_image(image_url, dst_path)
                    print(f"{filename} -> {msg}")

                    if ok:
                        saved_count += 1

                    time.sleep(0.15)

            except Exception as e:
                print(f"Failed listing: {e}")

        browser.close()

    clean_pool()
    print("\nDone.")


if __name__ == "__main__":
    main()