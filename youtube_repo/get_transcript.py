"""Download a YouTube video's transcript as clean plain text.

Usage:
    python get_transcript.py <youtube_url> [language_code ...]

Example:
    python get_transcript.py https://www.youtube.com/watch?v=D4n7ytNwfLY
    python get_transcript.py https://www.youtube.com/watch?v=D4n7ytNwfLY ko en
"""
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

OUTPUT_DIR = Path(__file__).parent / "transcripts"


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""

    if "youtu.be" in host:
        return parsed.path.lstrip("/")

    if "youtube.com" in host:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return video_id
        match = re.match(r"/(shorts|embed|live)/([^/?]+)", parsed.path)
        if match:
            return match.group(2)

    raise ValueError(f"Could not extract a video ID from URL: {url}")


def fetch_transcript_text(video_id: str, languages: list[str]) -> tuple[str, str]:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript(languages)
    except NoTranscriptFound:
        transcript = next(iter(transcript_list))

    fetched = transcript.fetch()
    text = " ".join(snippet.text.replace("\n", " ").strip() for snippet in fetched)
    text = re.sub(r"\s+", " ", text).strip()
    return text, transcript.language_code


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_transcript.py <youtube_url> [language_code ...]")
        sys.exit(1)

    url = sys.argv[1]
    languages = sys.argv[2:] or ["en"]

    video_id = extract_video_id(url)

    try:
        text, language_code = fetch_transcript_text(video_id, languages)
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video: {video_id}")
        sys.exit(1)
    except VideoUnavailable:
        print(f"Video unavailable: {video_id}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{video_id}.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"Saved transcript ({language_code}, {len(text)} chars) to {out_path}")


if __name__ == "__main__":
    main()
