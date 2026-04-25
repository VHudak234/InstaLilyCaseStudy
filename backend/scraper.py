"""PartSelect scraper.

Two-stage design:
1. Discover part numbers from category listing pages.
2. Scrape each part detail page for structured data.

PartSelect has anti-bot protection, so we send a realistic User-Agent and
throttle requests. If scraping becomes unreliable, the live `get_part_details`
tool can fall back to a curated cache.

Usage:
    # Inspect what a page looks like (saves raw HTML to /tmp for eyeballing).
    python scraper.py --inspect https://www.partselect.com/Refrigerator-Parts.htm

    # Full catalogue build (after selectors are verified).
    python scraper.py --build --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = "https://www.partselect.com"
CATEGORY_PAGES = {
    "refrigerator": f"{BASE}/Refrigerator-Parts.htm",
    "dishwasher": f"{BASE}/Dishwasher-Parts.htm",
}

# PS followed by 6+ digits. Used to extract part numbers from hrefs.
PART_NUMBER_RE = re.compile(r"PS\d{6,}")

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOGUE_PATH = DATA_DIR / "parts_catalogue.json"


def fetch(url: str, page) -> str:
    """Navigate a Playwright page and return the rendered HTML.

    Real Chromium passes Cloudflare's JS challenge automatically. We wait for
    networkidle so any post-load XHRs finish before we read the DOM.
    """
    page.goto(url, wait_until="networkidle", timeout=30000)
    return page.content()


def _new_page(browser):
    """Configure a page to look like a normal user session.

    PartSelect is behind Akamai Bot Manager which fingerprints headless browsers.
    Overriding `navigator.webdriver` defeats the most common check; using headed
    mode (visible window) defeats most of the rest.
    """
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
        locale="en-US",
    )
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    )
    return context.new_page()


def inspect(url: str) -> None:
    """Fetch a URL and save the HTML + a short report so we can write selectors."""
    with sync_playwright() as pw:
        # headless=False → real visible window. Harder for Akamai to detect.
        browser = pw.chromium.launch(headless=False)
        page = _new_page(browser)
        try:
            html = fetch(url, page)
        except Exception as e:
            print(f"Blocked or error: {e}")
            browser.close()
            sys.exit(1)
        browser.close()

    out = Path("/tmp/partselect_sample.html")
    out.write_text(html)
    soup = BeautifulSoup(html, "lxml")

    print(f"Fetched {len(html):,} bytes -> {out}")
    print(f"Title: {soup.title.string.strip() if soup.title else '(no title)'}")
    print(f"Part-number hrefs found on page: {len(PART_NUMBER_RE.findall(html))}")
    # Surface the first 5 anchors that look part-related, for selector hints.
    anchors = [a for a in soup.find_all("a", href=True) if PART_NUMBER_RE.search(a["href"])]
    print(f"\nFirst 5 part anchors:")
    for a in anchors[:5]:
        print(f"  href={a['href']!r}  text={a.get_text(strip=True)[:60]!r}")


def discover_part_numbers(category: str, page) -> list[str]:
    """Pull unique PS-numbers from a category listing page."""
    html = fetch(CATEGORY_PAGES[category], page)
    # dict.fromkeys preserves first-seen order while deduping.
    return list(dict.fromkeys(PART_NUMBER_RE.findall(html)))


def scrape_part_detail(part_number: str, page) -> dict[str, Any]:
    """Scrape one part detail page into a structured dict.

    Selectors target schema.org microdata (itemprop="...") where possible —
    those are more stable than CSS classes, which sites rename often.
    """
    url = f"{BASE}/{part_number}-Part.htm"
    html = fetch(url, page)
    soup = BeautifulSoup(html, "lxml")

    def text(selector: str) -> str | None:
        el = soup.select_one(selector)
        return el.get_text(strip=True) if el else None

    def attr(selector: str, name: str) -> str | None:
        el = soup.select_one(selector)
        return el.get(name) if el else None

    # Price is cleanest from the `content` attribute (always a raw decimal).
    price_str = attr('[itemprop="price"]', "content")
    try:
        price = float(price_str) if price_str else None
    except ValueError:
        price = None

    # Main product image — first full-size itemprop="image" with an http src.
    image_url = None
    for img in soup.select('img[itemprop="image"]'):
        src = img.get("src", "")
        if src.startswith("http"):
            image_url = src
            break

    # Install difficulty + time live inside the repair-rating container as two
    # <p class="bold"> elements. We pull them positionally.
    rating_paras = [
        p.get_text(strip=True)
        for p in soup.select(".pd__repair-rating__container p.bold")
    ]
    install_difficulty = rating_paras[0] if len(rating_paras) > 0 else None
    install_time = rating_paras[1] if len(rating_paras) > 1 else None

    # Compatible brands — dedupe from the model cross-reference table.
    compat_brands: list[str] = []
    seen: set[str] = set()
    for row in soup.select(".pd__crossref__list .row"):
        cols = row.select("div,a")
        if cols:
            brand = cols[0].get_text(strip=True)
            if brand and brand not in seen:
                seen.add(brand)
                compat_brands.append(brand)

    return {
        "part_number": text('[itemprop="productID"]') or part_number,
        "mpn": text('[itemprop="mpn"]'),
        "name": text('h1[itemprop="name"]'),
        "brand": text('[itemprop="brand"] [itemprop="name"]'),
        "price": price,
        "description": text('[itemprop="description"]'),
        "install_difficulty": install_difficulty,
        "install_time": install_time,
        "compatible_brands": compat_brands,
        "image_url": image_url,
        "url": url,
    }


def build_catalogue(limit_per_category: int) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    catalogue: list[dict[str, Any]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        page = _new_page(browser)
        for category in ("refrigerator", "dishwasher"):
            print(f"\n[{category}] discovering part numbers...")
            part_numbers = discover_part_numbers(category, page)[:limit_per_category]
            print(f"[{category}] found {len(part_numbers)} parts")

            for i, pn in enumerate(part_numbers, 1):
                try:
                    part = scrape_part_detail(pn, page)
                    part["category"] = category
                    catalogue.append(part)
                    print(f"  [{i}/{len(part_numbers)}] {pn} — {part.get('name') or '?'}")
                except Exception as e:
                    print(f"  [{i}/{len(part_numbers)}] {pn} — FAILED: {e}")
                time.sleep(1.0)  # be polite.
        browser.close()

    CATALOGUE_PATH.write_text(json.dumps(catalogue, indent=2))
    print(f"\nWrote {len(catalogue)} parts -> {CATALOGUE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--inspect", metavar="URL", help="Fetch one URL and dump HTML for selector writing.")
    g.add_argument("--build", action="store_true", help="Build the full catalogue.")
    parser.add_argument("--limit", type=int, default=50, help="Max parts per category when --build.")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
    else:
        build_catalogue(args.limit)
