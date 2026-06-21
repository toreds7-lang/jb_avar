# YouTube Transcript Downloader

Downloads the transcript/script of a YouTube video as clean plain text, given just the video URL.

## How it works

1. **Parse the URL** — `extract_video_id()` accepts any common YouTube URL form
   (`youtube.com/watch?v=...`, `youtu.be/...`, `/shorts/...`, `/embed/...`, `/live/...`)
   and pulls out the 11-character video ID.
2. **Fetch captions** — uses the [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api)
   library, which reads YouTube's caption tracks directly (no video download, no API key).
   - `YouTubeTranscriptApi().list(video_id)` lists all available caption tracks (manual and
     auto-generated, in every language YouTube has them).
   - It tries your requested language(s) first via `find_transcript()`; if none match, it
     falls back to the first available track (e.g. auto-generated Korean if no English
     captions exist — this is why the example video came back in Korean).
3. **Clean the text** — the API returns timestamped snippets; the script strips timestamps,
   joins all snippets with spaces, and collapses whitespace/newlines into one readable block
   of plain text.
4. **Save** — writes the result to `transcripts/<video_id>.txt` and prints the language code
   and character count.

## Usage

```bash
python get_transcript.py <youtube_url> [language_code ...]
```

Examples:

```bash
# Default: try English captions, fall back to whatever's available
python get_transcript.py https://www.youtube.com/watch?v=D4n7ytNwfLY

# Prefer Korean, then English
python get_transcript.py https://www.youtube.com/watch?v=D4n7ytNwfLY ko en
```

Output is saved to `transcripts/<video_id>.txt`, e.g. `transcripts/D4n7ytNwfLY.txt`.

## Requirements

- Python 3.11+
- `youtube-transcript-api` (installed via `pip install youtube-transcript-api`)

## Limitations

- Only works for videos that have captions (manual or auto-generated) enabled. If
  captions are disabled, the script exits with an error.
- Some videos may be region-restricted or block automated caption requests; in that
  case YouTube may rate-limit or block the request IP (see the library's README for
  proxy workarounds if this becomes an issue).
- Output is plain text only — no timestamps. If you need timestamps or raw subtitle
  files (`.srt`/`.vtt`), a different tool (e.g. `yt-dlp --write-auto-sub`) would be
  more appropriate.
