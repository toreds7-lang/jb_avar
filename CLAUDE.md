# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

This is not a git repository — it's a working directory containing independent sub-projects, each with its own purpose:

- `youtube_repo/` — a YouTube transcript downloader (see below).
- `blog_repo/` — a Naver Blog category crawler (see below).
- `.venv/` — a single Python 3.11 virtual environment shared at the root, used by both `youtube_repo` and `blog_repo`.

Treat each `*_repo` directory as its own project root when working inside it; there is no shared build system tying them together.

## Environment setup

The venv exists but does not yet have project dependencies installed. From the repo root:

```bash
.venv/Scripts/python.exe -m pip install youtube-transcript-api requests beautifulsoup4
```

(or activate the venv first: `source .venv/Scripts/activate` in bash, `.venv\Scripts\Activate.ps1` in PowerShell)

## youtube_repo

A single-file script (`get_transcript.py`) that downloads a YouTube video's transcript as clean plain text given just the video URL — no API key, no video download. It uses the `youtube-transcript-api` library to read caption tracks directly.

### Running it

```bash
python youtube_repo/get_transcript.py <youtube_url> [language_code ...]
```

- Language args are preference order for caption track selection (e.g. `ko en` prefers Korean, falls back to English). Defaults to `en`.
- If no track matches the requested languages, it falls back to the first available track (manual or auto-generated, any language).
- Output is written to `youtube_repo/transcripts/<video_id>.txt` (created if missing).

### Architecture

The script is three steps in sequence, each a standalone function in `get_transcript.py`:

1. `extract_video_id(url)` — parses any common YouTube URL form (`watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`) into the 11-character video ID via regex/query parsing.
2. `fetch_transcript_text(video_id, languages)` — uses `YouTubeTranscriptApi().list(video_id)` to enumerate caption tracks, picks one via `find_transcript(languages)` with a fallback to the first available track, fetches it, then strips timestamps and collapses whitespace into one plain-text block.
3. `main()` — wires up CLI args, handles `TranscriptsDisabled` / `VideoUnavailable` by printing an error and exiting 1, then writes the result to `transcripts/<video_id>.txt`.

### Known limitations (see `youtube_repo/README.md` for detail)

- Videos without any captions (manual or auto-generated) will fail with `TranscriptsDisabled`.
- No timestamps in output — for `.srt`/`.vtt` or timed output, a different tool (e.g. `yt-dlp --write-auto-sub`) is needed.
- Heavy automated use may get rate-limited/blocked by YouTube; the upstream library's README documents proxy workarounds.

## blog_repo

A single-file script (`crawl_naver_blog.py`) that crawls every post in one category of a Naver Blog into plain-text files, given the blog's URL (or bare blog ID) and the category's display name (e.g. `전황의 주식철학`) — no API key. It scrapes Naver's undocumented internal endpoints directly (no official public API exists for this).

### Running it

```bash
python blog_repo/crawl_naver_blog.py <blog_url_or_id> <category_name>
```

- Output is written to `blog_repo/<blog_id>/<category_name>/<date>_<title>.txt`, one file per post, with a small header (title/date/source URL) followed by the article body.
- Re-running is safe and incremental: posts whose output file already exists are skipped, so only newly published posts get fetched.
- If the category name doesn't match exactly, the script exits with an error listing the blog's actual category names.

### Architecture

Each step is a standalone function in `crawl_naver_blog.py`:

1. `extract_blog_id(url)` — pulls the blogId out of a `blog.naver.com/<id>` URL, or passes a bare id through.
2. `get_category_no(blog_id, category_name)` — fetches the blog's `PostList.naver` widget frame and scans its category nav links for one whose text matches `category_name`, returning its `categoryNo`.
3. `list_posts_in_category(blog_id, category_no)` — pages through the `PostTitleListAsync.naver` JSON endpoint collecting `logNo`/`title`/`addDate` per post. The endpoint's `currentPage` isn't reliably respected near the end of the list (it can repeat the same page), so pagination is bounded by the response's `totalCount` and results are deduplicated by `logNo`. The JSON it returns also embeds HTML with invalid `\'` escapes that must be sanitized before `json.loads`.
4. `fetch_post_text(blog_id, log_no)` — fetches the post's *mobile* view (`m.blog.naver.com/PostView.naver`), which renders the body server-side (unlike the desktop view, which loads content into an iframe) in a `div.se-main-container` (or `#postViewArea` for old-editor posts), and extracts clean text from it.
5. `main()` — wires up CLI args, resolves the category, lists posts, and writes one text file per post, skipping ones already downloaded.

### Known limitations

- Relies on undocumented Naver endpoints (`PostList.naver`, `PostTitleListAsync.naver`, `m.blog.naver.com/PostView.naver`) that could change or break without notice.
- Posts that are member-only, password-protected, or otherwise restricted will fail to parse and are skipped with a printed warning rather than stopping the whole run.
- A fixed `REQUEST_DELAY_SECONDS` delay (0.6s) is used between requests to stay polite; very large categories will take a while and heavy use could still get rate-limited.
