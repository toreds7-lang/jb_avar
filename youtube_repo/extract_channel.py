"""Download transcripts for every video in a YouTube channel tab.

Usage:
    python extract_channel.py <channel_tab_url> <output_dir> [language_code ...]

Example:
    python extract_channel.py https://www.youtube.com/@rpnestmaker365/streams ../전황 ko en

Each video is saved as <output_dir>/전황_<YYMMDD>.txt (YYMMDD = upload date).
Videos whose output file already exists are skipped, so re-running only fetches
newly published videos.
"""
import sys
import time
from pathlib import Path

import yt_dlp

from get_transcript import fetch_transcript_text
from youtube_transcript_api._errors import RequestBlocked, TranscriptsDisabled, VideoUnavailable

REQUEST_DELAY_SECONDS = 2


def list_channel_video_ids(channel_url: str) -> list[str]:
    opts = {"extract_flat": "in_playlist", "quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    return [entry["id"] for entry in info.get("entries", []) if entry.get("id")]


def get_upload_date(video_id: str) -> str:
    opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return info["upload_date"]  # YYYYMMDD


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_channel.py <channel_tab_url> <output_dir> [language_code ...]")
        sys.exit(1)

    channel_url = sys.argv[1]
    output_dir = Path(sys.argv[2])
    languages = sys.argv[3:] or ["ko", "en"]

    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids = list_channel_video_ids(channel_url)
    print(f"Found {len(video_ids)} videos in {channel_url}")

    for video_id in video_ids:
        try:
            upload_date = get_upload_date(video_id)
        except Exception as e:
            print(f"Skipping {video_id}: failed to fetch metadata ({e})")
            continue

        out_path = output_dir / f"전황_{upload_date[2:]}.txt"
        if out_path.exists():
            print(f"Already have {out_path.name}, skipping")
            continue

        try:
            text, language_code = fetch_transcript_text(video_id, languages)
        except TranscriptsDisabled:
            print(f"Transcripts disabled for {video_id} ({upload_date}), skipping")
            continue
        except VideoUnavailable:
            print(f"Video unavailable: {video_id}, skipping")
            continue
        except RequestBlocked:
            print(f"YouTube blocked the request for {video_id} ({upload_date}), skipping")
            continue

        out_path.write_text(text, encoding="utf-8")
        print(f"Saved {out_path.name} ({language_code}, {len(text)} chars)")
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()
