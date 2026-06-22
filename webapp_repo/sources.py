"""Source ingestion: YouTube transcript fetch + raw .txt upload, plus a small registry
(sources.json) so the UI can list and reopen sources without re-fetching.

Reuses youtube_repo/get_transcript.py's URL parsing and transcript-fetch logic directly
rather than reimplementing it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from config import DATA_DIR, BASE_DIR

sys.path.insert(0, str(BASE_DIR.parent / "youtube_repo"))
from get_transcript import extract_video_id, fetch_transcript_text  # noqa: E402

_REGISTRY_PATH = DATA_DIR / "sources.json"


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_").lower() or "source"


def _load_registry() -> dict[str, Any]:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_registry(reg: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _REGISTRY_PATH)


def list_sources() -> list[dict[str, Any]]:
    reg = _load_registry()
    return sorted(reg.values(), key=lambda s: s.get("created_at", ""), reverse=True)


def get_source(source_id: str) -> dict[str, Any] | None:
    return _load_registry().get(source_id)


def source_dir(source_id: str) -> Path:
    return DATA_DIR / source_id


def _register(source_id: str, title: str, kind: str, origin: str) -> dict[str, Any]:
    reg = _load_registry()
    entry = {
        "id": source_id,
        "title": title,
        "kind": kind,           # "youtube" | "upload"
        "origin": origin,       # url or original filename
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "has_clean": False,
        "ingested": False,
    }
    reg[source_id] = entry
    _save_registry(reg)
    return entry


def mark_preprocessed(source_id: str) -> None:
    reg = _load_registry()
    if source_id in reg:
        reg[source_id]["has_clean"] = True
        _save_registry(reg)


def mark_ingested(source_id: str) -> None:
    reg = _load_registry()
    if source_id in reg:
        reg[source_id]["ingested"] = True
        _save_registry(reg)


def load_text(source_id: str, clean: bool = False) -> str:
    name = "clean.txt" if clean else "source.txt"
    p = source_dir(source_id) / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def save_clean_text(source_id: str, text: str) -> None:
    p = source_dir(source_id) / "clean.txt"
    p.write_text(text, encoding="utf-8")
    mark_preprocessed(source_id)


def load_from_youtube(url: str, languages: list[str] | None = None) -> dict[str, Any]:
    """Fetch a transcript and register it as a new source. Returns the registry entry."""
    languages = languages or ["en"]
    video_id = extract_video_id(url)
    text, language_code = fetch_transcript_text(video_id, languages)

    source_id = f"yt_{video_id}"
    d = source_dir(source_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.txt").write_text(text, encoding="utf-8")

    entry = _register(source_id, title=f"YouTube {video_id} ({language_code})", kind="youtube", origin=url)
    entry["language"] = language_code
    reg = _load_registry()
    reg[source_id] = entry
    _save_registry(reg)
    return entry


def load_from_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Register an uploaded .txt file as a new source. Returns the registry entry."""
    text = content.decode("utf-8", errors="ignore")
    digest = hashlib.sha1(content).hexdigest()[:8]
    base = _slugify(Path(filename).stem)
    source_id = f"up_{base}_{digest}"

    d = source_dir(source_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.txt").write_text(text, encoding="utf-8")

    return _register(source_id, title=filename, kind="upload", origin=filename)
