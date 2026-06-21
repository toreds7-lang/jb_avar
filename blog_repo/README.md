# Naver Blog Category Crawler

Crawls every post in one category of a Naver Blog into clean plain-text files, given just the blog's URL and the category's display name.

## How it works

1. **Resolve the blog ID** — `extract_blog_id()` pulls the blogId out of a `blog.naver.com/<id>` URL, or passes a bare id straight through.
2. **Resolve the category** — `get_category_no()` fetches the blog's `PostList.naver` widget frame (the iframe that the desktop blog page loads) and scans its category navigation links for one whose visible text matches the category name you gave, returning its `categoryNo`. If nothing matches, it exits with an error listing every real category name on the blog.
3. **List posts** — `list_posts_in_category()` pages through the `PostTitleListAsync.naver` JSON endpoint, collecting each post's `logNo`, title, and date.
   - This endpoint doesn't reliably respect `currentPage` near the end of the list (it can repeat the same page), so pagination is bounded by the response's `totalCount` and posts are deduplicated by `logNo`.
   - The JSON it returns also embeds HTML with invalid `\'` escapes, which are sanitized before parsing.
4. **Fetch each post's text** — `fetch_post_text()` fetches the post's *mobile* view (`m.blog.naver.com/PostView.naver`), which renders the article body server-side — unlike the desktop view, which loads content into an iframe and is harder to scrape. Text is extracted from the `se-main-container` div (or `#postViewArea` for old-editor posts).
5. **Save** — `main()` writes one `.txt` file per post to `<blog_id>/<category_name>/`, with a small header (title, date, source URL) followed by the article body. Posts whose output file already exists are skipped, so re-running only fetches newly published posts.

No API key is needed — everything here scrapes Naver's own undocumented internal endpoints (there is no official public API for blog categories/posts).

## Usage

```bash
python crawl_naver_blog.py <blog_url_or_id> <category_name>
```

Example:

```bash
python crawl_naver_blog.py https://blog.naver.com/chamberine3 "전황의 주식철학"
```

Output is saved to `<blog_id>/<category_name>/<date>_<title>.txt`, e.g. `chamberine3/전황의 주식철학/2026-03-24_매서운 봄바람.txt`.

## Requirements

- Python 3.11+
- `requests` and `beautifulsoup4` (installed via `pip install requests beautifulsoup4`)

## Limitations

- Relies on undocumented Naver endpoints (`PostList.naver`, `PostTitleListAsync.naver`, `m.blog.naver.com/PostView.naver`) that could change or break without notice.
- Posts that are member-only, password-protected, or otherwise restricted will fail to parse; the script prints a warning and skips them rather than stopping the whole run.
- A fixed 0.6-second delay is used between requests to stay polite to Naver's servers. Very large categories will take a while to crawl, and unusually heavy use could still get rate-limited.
- Category name matching is an exact string match against the blog's displayed category name (not fuzzy) — if unsure of the exact name, run with any name and the error message will list the real ones.
