"""Chunking + embedding index for RAG chat. Ported from paper_read_project_pdf/rag.py,
adapted from per-page PDF chunks to per-source-id text chunks (a "source" here is one
ingested transcript/file; the global scope's index spans chunks from many sources)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import llm_client

_CHUNK_CHARS = 2000
_OVERLAP_CHARS = 200


def _chunk_text(text: str, source_id: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    text = text.strip()
    if not text:
        return chunks
    i = 0
    while i < len(text):
        seg = text[i : i + _CHUNK_CHARS]
        chunks.append({"source_id": source_id, "text": seg})
        if i + _CHUNK_CHARS >= len(text):
            break
        i += _CHUNK_CHARS - _OVERLAP_CHARS
    return chunks


def build_index(texts: list[dict[str, Any]], data_dir: Path) -> None:
    """Embed all chunks and save vectors + metadata to data_dir.

    texts: [{"id": source_id, "text": full_text}, ...] — one entry per source folded
    into this index (a single entry for a per-source root, many for the global root).
    """
    all_chunks: list[dict[str, Any]] = []
    for t in texts:
        all_chunks.extend(_chunk_text(t["text"], t["id"]))
    if not all_chunks:
        print("[rag] no chunks to index", file=sys.stderr)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_dir / "rag.npz", vecs=np.zeros((0, 0), dtype=np.float32))
        (data_dir / "rag.json").write_text("[]", encoding="utf-8")
        return

    print(f"[rag] embedding {len(all_chunks)} chunks", file=sys.stderr)
    vecs_parts: list[np.ndarray] = []
    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        batch = [c["text"] for c in all_chunks[i : i + BATCH]]
        vecs_parts.append(llm_client.embed(batch))
    vecs = np.vstack(vecs_parts).astype(np.float32)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "rag.npz", vecs=vecs)
    (data_dir / "rag.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[rag] wrote rag.npz ({vecs.shape}) + rag.json", file=sys.stderr)


class RagIndex:
    """In-memory RAG index, reloaded per request (cheap: numpy load + json parse)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.vecs: np.ndarray | None = None
        self.chunks: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        vpath = self.data_dir / "rag.npz"
        cpath = self.data_dir / "rag.json"
        if not (vpath.exists() and cpath.exists()):
            return
        self.vecs = np.load(vpath)["vecs"]
        self.chunks = json.loads(cpath.read_text(encoding="utf-8"))

    def topk(self, query: str, k: int = 5, source_filter: str | None = None) -> list[dict[str, Any]]:
        if self.vecs is None or not self.chunks or self.vecs.shape[0] == 0:
            return []
        q = llm_client.embed([query])  # (1, D), already normalized
        sims = (self.vecs @ q[0]).astype(np.float32)
        if source_filter is not None:
            mask = np.array([c["source_id"] == source_filter for c in self.chunks])
            sims = np.where(mask, sims, -1.0)
        order = np.argsort(-sims)[:k]
        return [
            {**self.chunks[i], "score": float(sims[i])}
            for i in order
            if sims[i] > -1.0
        ]
