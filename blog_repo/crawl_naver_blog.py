"""Crawl all posts in one category of a Naver Blog into plain-text files.

Usage:
    python crawl_naver_blog.py <blog_url_or_id> <category_name>

Example:
    python crawl_naver_blog.py https://blog.naver.com/chamberine3 "전황의 주식철학"

Output:
    blog_repo/<blog_id>/<category_name>/<date>_<title>.txt
"""
import re
import sys
import time
import json
import unicodedata
from pathlib import Path
from urllib.parse import unquote_plus

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


def extract_blog_id(url_or_id: str) -> str:
    """Pull the blogId out of any blog.naver.com URL, or pass through a bare id."""
    match = re.search(r"blog\.naver\.com/([^/?#]+)", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def get_category_no(blog_id: str, category_name: str) -> str:
    """Resolve a category's display name to its categoryNo via the mobile category-list API.

    The PC PostList.naver page's category sidebar is populated client-side by this
    same API, not server-rendered, so it must be queried directly rather than scraped.
    """
    resp = requests.get(
        f"https://m.blog.naver.com/api/blogs/{blog_id}/category-list",
        headers={**HEADERS, "Referer": f"https://m.blog.naver.com/{blog_id}"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    categories = {
        cat["categoryName"]: str(cat["categoryNo"])
        for cat in data["result"]["mylogCategoryList"]
    }

    if category_name in categories:
        return categories[category_name]

    available = ", ".join(sorted(categories)) or "(none found)"
    raise ValueError(
        f"Category '{category_name}' not found on blog '{blog_id}'. "
        f"Available categories: {available}"
    )


def list_posts_in_category(blog_id: str, category_no: str) -> list[dict]:
    """Page through PostTitleListAsync.naver and collect every post's logNo/title/date.

    The API doesn't reliably respect currentPage for trailing pages (it can repeat
    the same results), so pagination is bounded by the response's totalCount and
    posts are deduplicated by logNo as a safety net.
    """
    posts_by_log_no = {}
    page = 1
    total_count = None
    while total_count is None or len(posts_by_log_no) < total_count:
        resp = requests.get(
            "https://blog.naver.com/PostTitleListAsync.naver",
            params={
                "blogId": blog_id,
                "viewdate": "",
                "currentPage": page,
                "categoryNo": category_no,
                "parentCategoryNo": "",
                "countPerPage": 30,
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        # The API embeds HTML containing invalid `\'` escapes, which breaks strict JSON parsing.
        data = json.loads(resp.text.replace("\\'", "'"))
        total_count = int(data.get("totalCount", 0))
        page_posts = data.get("postList", [])
        if not page_posts:
            break

        for post in page_posts:
            posts_by_log_no[post["logNo"]] = {
                "logNo": post["logNo"],
                "title": unquote_plus(post["title"]),
                "addDate": post["addDate"],
            }

        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    return list(posts_by_log_no.values())


def fetch_post_text(blog_id: str, log_no: str) -> str:
    """Fetch a post's mobile view and return its body as clean plain text."""
    resp = requests.get(
        "https://m.blog.naver.com/PostView.naver",
        params={"blogId": blog_id, "logNo": log_no},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    container = soup.find("div", class_="se-main-container") or soup.find(
        "div", id="postViewArea"
    )
    if container is None:
        raise ValueError(f"Could not find post content for logNo={log_no}")

    raw_text = container.get_text("\n", strip=True)
    lines = [line for line in raw_text.splitlines() if line.strip("​‌ ")]
    return "\n".join(lines)


def sanitize_filename(name: str, max_length: int = 80) -> str:
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip()
    return name[:max_length] if name else "untitled"


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <blog_url_or_id> <category_name>")
        sys.exit(1)

    blog_url_or_id, category_name = sys.argv[1], sys.argv[2]
    blog_id = extract_blog_id(blog_url_or_id)

    print(f"Resolving category '{category_name}' on blog '{blog_id}'...")
    try:
        category_no = get_category_no(blog_id, category_name)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Found categoryNo={category_no}")

    print("Listing posts in category...")
    posts = list_posts_in_category(blog_id, category_no)
    print(f"Found {len(posts)} posts")

    out_dir = OUTPUT_ROOT / blog_id / sanitize_filename(category_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    for post in posts:
        date_match = re.match(r"(\d{4})[.\-]\s*(\d{1,2})[.\-]\s*(\d{1,2})", post["addDate"])
        date_prefix = (
            f"{date_match[1]}-{int(date_match[2]):02d}-{int(date_match[3]):02d}"
            if date_match
            else post["addDate"]
        )
        filename = f"{date_prefix}_{sanitize_filename(post['title'])}.txt"
        out_path = out_dir / filename

        if out_path.exists():
            skipped += 1
            continue

        url = f"https://blog.naver.com/{blog_id}/{post['logNo']}"
        try:
            body_text = fetch_post_text(blog_id, post["logNo"])
        except Exception as exc:
            print(f"  [skip] {post['title']} ({post['logNo']}): {exc}")
            continue

        header = (
            f"제목: {post['title']}\n"
            f"날짜: {post['addDate']}\n"
            f"원문: {url}\n"
            f"{'-' * 40}\n\n"
        )
        out_path.write_text(header + body_text, encoding="utf-8")
        downloaded += 1
        print(f"  [saved] {filename}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Downloaded {downloaded} new post(s), skipped {skipped} already-saved post(s).")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
