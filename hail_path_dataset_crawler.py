import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO
import imagehash
import cv2
import numpy as np
import csv

visited = set()
image_hashes = set()

def is_valid_domain(url, allowed_domains):
    return any(domain in url for domain in allowed_domains)

def is_image_url(url):
    return any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"])

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def download_and_filter_image(url, out_dir, min_width, min_height, blur_thresh):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")

        if img.width < min_width or img.height < min_height:
            return None

        if is_blurry(img, blur_thresh):
            return None

        h = str(imagehash.phash(img))
        if h in image_hashes:
            return None
        image_hashes.add(h)

        filename = os.path.join(out_dir, f"{h}.jpg")
        img.save(filename)
        return filename

    except:
        return None

def crawl(url, allowed_domains, out_dir, min_width, min_height, blur_thresh, max_images):
    if url in visited:
        return []
    visited.add(url)

    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
    except:
        return []

    images_saved = []
    count = 0

    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue

        full_url = urljoin(url, src)

        if not is_valid_domain(full_url, allowed_domains):
            continue

        if not is_image_url(full_url):
            continue

        saved = download_and_filter_image(full_url, out_dir, min_width, min_height, blur_thresh)

        if saved:
            images_saved.append((url, full_url, saved))
            count += 1

        if count >= max_images:
            break

    return images_saved

def main():
    seeds_file = "seeds/seeds.txt"
    out_dir = "data/raw_candidates/images"
    os.makedirs(out_dir, exist_ok=True)

    allowed_domains = ["copart.com", "iaai.com"]

    results = []

    with open(seeds_file, "r") as f:
        seeds = [line.strip() for line in f if line.strip()]

    for url in seeds:
        print(f"Crawling: {url}")
        res = crawl(
            url,
            allowed_domains,
            out_dir,
            min_width=1200,
            min_height=900,
            blur_thresh=120,
            max_images=20
        )
        results.extend(res)

    os.makedirs("data/raw_candidates/metadata", exist_ok=True)

    with open("data/raw_candidates/metadata/crawl_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_page", "image_url", "saved_path"])
        writer.writerows(results)

    print(f"Saved {len(results)} images.")

if __name__ == "__main__":
    main()