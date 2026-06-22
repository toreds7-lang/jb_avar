"""Generalized irrelevant-text removal: classify text segments as keep/drop, then
concatenate the kept segments in original order.

This generalizes the verbatim-quote-extraction pattern in 전황_result/extract.py (which
locks onto 6 stock-investing categories) to any topic: instead of asking the LLM to quote
spans (which it can fail to reproduce verbatim, breaking a source-match check), we split
the text ourselves and ask the LLM only to label EXISTING segments keep/drop. That keeps
the same "never let the LLM rewrite the source" safety property while removing the
risk of paraphrased "quotes" and automatically preserving order.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import llm_client

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SEGMENTS_PER_CALL = 40
_MAX_SEGMENT_CHARS = 1200

# Split on blank lines (real paragraphs) when present; YouTube transcripts have none, so
# fall back to sentence boundaries.
_PARA_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _segment_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if len(paras) > 1:
        segments = paras
    else:
        segments = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    # Re-split any over-long segment so one giant paragraph/sentence doesn't dominate a call.
    out: list[str] = []
    for seg in segments:
        if len(seg) <= _MAX_SEGMENT_CHARS:
            out.append(seg)
        else:
            for i in range(0, len(seg), _MAX_SEGMENT_CHARS):
                out.append(seg[i : i + _MAX_SEGMENT_CHARS])
    return out


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _classify_batch(segments: list[str]) -> dict[int, str]:
    """Return {local_index: 'keep'|'drop'} for one batch. Missing/invalid -> keep (safe default)."""
    numbered = "\n\n".join(f"[{i}] {seg}" for i, seg in enumerate(segments))
    system = _load("preprocess.system.txt")
    user = _load("preprocess.user.txt").format(segments=numbered)
    out = llm_client.chat_json(system, user)
    labels: dict[int, str] = {}
    if isinstance(out, dict) and isinstance(out.get("labels"), list):
        for item in out["labels"]:
            if not isinstance(item, dict):
                continue
            i = item.get("i")
            action = str(item.get("action", "keep")).lower()
            if isinstance(i, int) and 0 <= i < len(segments):
                labels[i] = "drop" if action == "drop" else "keep"
    return labels


def preprocess(text: str) -> Iterator[dict[str, Any]]:
    """Yield NDJSON-style progress dicts; final one carries the cleaned text and the
    per-segment decisions (for a before/after diff view in the UI)."""
    segments = _segment_text(text)
    if not segments:
        yield {"done": True, "ok": True, "clean_text": "", "decisions": []}
        return

    yield {"stage": "classifying", "total_segments": len(segments)}

    decisions: list[dict[str, Any]] = []
    for start in range(0, len(segments), _SEGMENTS_PER_CALL):
        batch = segments[start : start + _SEGMENTS_PER_CALL]
        try:
            labels = _classify_batch(batch)
        except Exception:
            labels = {}
        for local_i, seg in enumerate(batch):
            action = labels.get(local_i, "keep")
            decisions.append({"text": seg, "action": action})
        yield {"stage": "progress", "done_segments": min(start + len(batch), len(segments)), "total_segments": len(segments)}

    kept = [d["text"] for d in decisions if d["action"] == "keep"]
    clean_text = "\n\n".join(kept)
    yield {"done": True, "ok": True, "clean_text": clean_text, "decisions": decisions,
           "kept": len(kept), "dropped": len(decisions) - len(kept)}
