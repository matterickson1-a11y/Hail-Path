import csv
import hashlib
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests
from PIL import Image
from playwright.sync_api import sync_playwright

QUERIES_FILE = Path("queries/hail_queries.txt")
OUTPUT_DIR = Path("bulk_hail_candidates")
IMAGES_DIR = OUTPUT_DIR / "images"
META_DIR = OUTPUT_DIR / "metadata"
CSV_FILE = META_DIR / "results.csv"

MIN_WIDTH = 1000
MIN_HEIGHT = 800
MAX_PER_QUERY = 120
SCROLLS = 8
WAIT_MS = 2500

NEGATIVE_TERMS = [
    "house", "roofing", "shingles", "home", "property",
    "interior", "seat", "dashboard", "steering",
    "engine", "wheel", "tire",
    "logo", "icon", "banner", "stock", "placeholder",
    "diagram", "cartoon",
    "ceiling", "window"
]

POSITIVE_TERMS = [
    "car", "vehicle", "auto",
    "hail", "dent", "damage",
    "roof", "hood", "decklid", "trunk",
    "panel", "pdr", "paintless",
    "reflection", "light"
]

USER_AGENT = "Mozilla/5.0"

def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

def read_queries():
    if not QUERIES_FILE.exists():
        print("Queries file not found:", QUERIES_FILE)
        return []
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def is_relevant(text):
    t = (text or "").lower()
    if any(neg in t for neg in NEGATIVE_TERMS):
        return False
    if not any(pos in t for pos in POSITIVE_TERMS):
        return False
    return True

def clean_name(text):
    return re.sub(r'[^a-zA-Z0-9_-]+', "_", text)[:60] or "img"

def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()

def fetch_image(url):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    return r.content

def save_if_valid(img_bytes, path):
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
            return False
        img.save(path, quality=90)
        return True
    except:
        return False

def harvest_query(page, query):
    url = "https://www.bing.com/images/search?q=" + quote_plus(query)
    page.goto(url)

    for _ in range(SCROLLS):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(WAIT_MS)

    thumbs = page.locator("a.iusc")
    results = []
    seen = set()

    for i in range(thumbs.count()):
        if len(results) >= MAX_PER_QUERY:
            break

        try:
            node = thumbs.nth(i)
            m = node.get_attribute("m") or ""
            text = (m or "").lower()

            if not is_relevant(text):
                continue

            match = re.search(r'"murl":"(.*?)"', m)
            if not match:
                continue

            img_url = match.group(1).replace("\\/", "/")

            if img_url in seen:
                continue
            seen.add(img_url)

            results.append(img_url)

        except:
            continue

    return results

def main():
    ensure_dirs()
    queries = read_queries()

    if not queries:
        print("No queries found.")
        return

    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)

        for query in queries:
            print("\nSearching:", query)

            urls = harvest_query(page, query)
            print("Candidates:", len(urls))

            saved = 0

            for url in urls:
                try:
                    img_bytes = fetch_image(url)
                    name = clean_name(query) + "__" + hash_url(url)[:10] + ".jpg"
                    path = IMAGES_DIR / name

                    if save_if_valid(img_bytes, path):
                        saved += 1
                        rows.append({"query": query, "saved_path": str(path)})
                except:
                    continue

            print("Saved:", saved)

        browser.close()

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "saved_path"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nDONE")
    print("Images:", IMAGES_DIR)

if __name__ == "__main__":
    main()