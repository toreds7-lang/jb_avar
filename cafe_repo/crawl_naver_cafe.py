"""Crawl all articles in one menu (board) of a Naver Cafe into plain-text files.

Usage:
    python crawl_naver_cafe.py <cafe_menu_url> <menu_name>

Example:
    python crawl_naver_cafe.py https://cafe.naver.com/f-e/cafes/30600956/menus/99 전황

Output:
    cafe_repo/<cafe_id>/<menu_name>/<date>_<title>.txt
"""
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
REQUEST_DELAY_SECONDS = 0.6
OUTPUT_ROOT = Path(__file__).parent


def parse_cafe_menu_url(url: str) -> tuple[str, str]:
    """Pull (cafeId, menuId) out of a https://cafe.naver.com/.../cafes/<id>/menus/<id> URL."""
    match = re.search(r"cafes/(\d+)/menus/(\d+)", url)
    if not match:
        raise ValueError(
            "Expected a URL of the form "
            "https://cafe.naver.com/f-e/cafes/<cafeId>/menus/<menuId>"
        )
    return match.group(1), match.group(2)


def list_articles_in_menu(cafe_id: str, menu_id: str) -> list[dict]:
    """Page through the cafe-boardlist-api articles endpoint until a page comes back empty.

    The endpoint caps each page at 15 articles regardless of the requested perPage,
    and there's no totalCount in the response, so pagination just continues until
    a page returns no articles.
    """
    articles = []
    page = 1
    while True:
        resp = requests.get(
            f"https://apis.naver.com/cafe-web/cafe-boardlist-api/v1/cafes/{cafe_id}/menus/{menu_id}/articles",
            params={"page": page, "perPage": 15, "sortBy": "TIME"},
            headers={
                **HEADERS,
                "Referer": f"https://cafe.naver.com/f-e/cafes/{cafe_id}/menus/{menu_id}",
            },
            timeout=15,
        )
        resp.raise_for_status()
        page_articles = resp.json()["result"]["articleList"]
        if not page_articles:
            break

        for entry in page_articles:
            item = entry["item"]
            articles.append(
                {
                    "articleId": item["articleId"],
                    "subject": item["subject"],
                    "writeDateTimestamp": item["writeDateTimestamp"],
                }
            )

        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    return articles


def fetch_article_text(cafe_id: str, article_id: int) -> str:
    """Fetch an article's full body via the article-detail API and return clean plain text."""
    resp = requests.get(
        f"https://article.cafe.naver.com/gw/v4/cafes/{cafe_id}/articles/{article_id}",
        params={"useCafeId": "true", "requestFrom": "A"},
        headers={
            **HEADERS,
            "Referer": f"https://cafe.naver.com/f-e/cafes/{cafe_id}/articles/{article_id}",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()["result"]
    article = data.get("article")
    if not article or not article.get("contentHtml"):
        raise ValueError(f"Could not fetch article body (errorCode={data.get('errorCode')})")

    soup = BeautifulSoup(article["contentHtml"], "html.parser")
    raw_text = soup.get_text("\n", strip=True)
    lines = [line for line in raw_text.splitlines() if line.strip("​‌ ")]
    return "\n".join(lines)


def sanitize_filename(name: str, max_length: int = 80) -> str:
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip()
    return name[:max_length] if name else "untitled"


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <cafe_menu_url> <menu_name>")
        sys.exit(1)

    cafe_menu_url, menu_name = sys.argv[1], sys.argv[2]
    try:
        cafe_id, menu_id = parse_cafe_menu_url(cafe_menu_url)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Listing articles in menu {menu_id} of cafe {cafe_id}...")
    articles = list_articles_in_menu(cafe_id, menu_id)
    print(f"Found {len(articles)} articles")

    out_dir = OUTPUT_ROOT / cafe_id / sanitize_filename(menu_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    for article in articles:
        date_str = time.strftime(
            "%Y-%m-%d", time.localtime(article["writeDateTimestamp"] / 1000)
        )
        filename = f"{date_str}_{sanitize_filename(article['subject'])}.txt"
        out_path = out_dir / filename

        if out_path.exists():
            skipped += 1
            continue

        url = f"https://cafe.naver.com/f-e/cafes/{cafe_id}/articles/{article['articleId']}"
        try:
            body_text = fetch_article_text(cafe_id, article["articleId"])
        except Exception as exc:
            print(f"  [skip] {article['subject']} ({article['articleId']}): {exc}")
            continue

        header = (
            f"제목: {article['subject']}\n"
            f"날짜: {date_str}\n"
            f"원문: {url}\n"
            f"{'-' * 40}\n\n"
        )
        out_path.write_text(header + body_text, encoding="utf-8")
        downloaded += 1
        print(f"  [saved] {filename}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(
        f"\nDone. Downloaded {downloaded} new article(s), "
        f"skipped {skipped} already-saved article(s)."
    )
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
