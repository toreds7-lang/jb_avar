"""Hybrid raw-chunk retrieval (vector + keyword) over a built GraphRAG index.

GraphRAG's four search methods each return a *synthesized answer* that can refuse ("the
data does not contain…"). The vector-RAG reference apps never refuse, because they always
return the nearest raw chunks. This module restores that property as an always-on evidence
source ALONGSIDE graph search: it ranks the index's own text-unit chunks by dense vector
similarity AND keyword overlap, fuses the two rankings with Reciprocal Rank Fusion, and
returns the raw chunks for the agent to synthesize from. Graph answers stay primary; these
guarantee real document text is always present.

Design for this app's constraints:
  * Reuses output/text_units.parquet — the exact chunks GraphRAG already built (no re-chunk).
  * Caches the one-time chunk embeddings on disk (output/hybrid_vecs.npz); a whole chat
    then costs a single *batched* query-embedding call.
  * Vector is best-effort: if the (self-hosted, flaky) embedding server is down, every
    embed call degrades to keyword-only retrieval rather than failing the chat.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import llm_client

# Question-shell stopwords so topical terms (including acronyms/numbers) drive ranking.
STOPWORDS = frozenset(
    "the a an of to in on for and or is are was were be been being this that these those "
    "what which who whom whose how why when where does do did can could should would will "
    "with from into about over under as at by it its their your our his her they them we "
    "you i he she summarize summary overview explain describe tell give list paper document "
    "section please main key".split()
)

_RRF_K = 60          # standard Reciprocal Rank Fusion damping constant
_HYBRID_TOP_K = 6    # raw chunks contributed to the evidence pool per question
_VEC_CACHE = "hybrid_vecs.npz"


def content_terms(s: str) -> list[str]:
    """Content words of a string: lowercase alphanumerics, length > 2, minus stopwords."""
    return [w for w in re.findall(r"[a-z0-9]+", s.lower())
            if len(w) > 2 and w not in STOPWORDS]


def _rrf_fuse(ranked_lists: list[list[int]]) -> dict[int, float]:
    """Reciprocal Rank Fusion: each list contributes 1/(K + rank) to an item's score."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for r, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (_RRF_K + r)
    return fused


class HybridRetriever:
    """Per-paper raw-chunk retriever over the built index's text units. Construct one per
    index and cache it. Degrades to keyword-only when embeddings are unavailable, so a flaky
    embedding server never breaks retrieval."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._units = self.root / "output" / "text_units.parquet"
        # Snapshot the source mtime so a later index rebuild invalidates this instance and
        # its vector cache (see stale() / _ensure_chunk_vecs).
        self._mtime = self._units.stat().st_mtime
        df = pd.read_parquet(self._units)
        self.chunks: list[str] = [t for t in (str(x).strip() for x in df["text"].tolist()) if t]
        # Precompute per-chunk term frequencies once for the keyword ranker.
        self._tf: list[dict[str, int]] = []
        for c in self.chunks:
            tf: dict[str, int] = {}
            for w in content_terms(c):
                tf[w] = tf.get(w, 0) + 1
            self._tf.append(tf)
        self._cvecs: np.ndarray | None = None
        self._cvecs_tried = False

    def stale(self) -> bool:
        """True if the source text units changed on disk since construction (index rebuilt),
        so the cache holder should drop this instance and build a fresh one."""
        try:
            return self._units.stat().st_mtime != self._mtime
        except OSError:
            return True

    # --- chunk vectors (best-effort, cached on disk) --------------------------
    def _ensure_chunk_vecs(self) -> np.ndarray | None:
        if self._cvecs is not None or self._cvecs_tried:
            return self._cvecs
        self._cvecs_tried = True
        cache = self.root / "output" / _VEC_CACHE
        if cache.exists():
            try:
                data = np.load(cache)
                vecs = data["vecs"].astype(np.float32)
                # Guard on BOTH row count and source mtime so a rebuild that happens to keep
                # the same chunk count can't serve stale vectors.
                if vecs.shape[0] == len(self.chunks) and float(data["mtime"]) == self._mtime:
                    self._cvecs = vecs
                    return self._cvecs
            except Exception:  # noqa: BLE001 — stale/corrupt cache: recompute below
                pass
        try:
            vecs = llm_client.embed(self.chunks).astype(np.float32)
        except Exception as e:  # noqa: BLE001 — embedding server down: keyword-only
            print(f"[hybrid] chunk embedding unavailable, keyword-only: {e}", file=sys.stderr)
            return None
        if vecs.shape[0] != len(self.chunks):
            return None
        self._cvecs = vecs
        try:
            np.savez_compressed(cache, vecs=vecs, mtime=np.array(self._mtime))
        except Exception:  # noqa: BLE001 — cache is an optimization, not required
            pass
        return self._cvecs

    def _embed_queries(self, queries: list[str]) -> np.ndarray | None:
        """One batched embed for all queries (best-effort)."""
        try:
            qv = llm_client.embed(queries).astype(np.float32)
        except Exception as e:  # noqa: BLE001 — embedding server down: keyword-only
            print(f"[hybrid] query embedding unavailable, keyword-only: {e}", file=sys.stderr)
            return None
        return qv if qv.shape[0] == len(queries) else None

    # --- ranking --------------------------------------------------------------
    def _keyword_rank(self, qterms: set[str]) -> list[int]:
        scored = [
            (len(present) + 0.1 * sum(tf[t] for t in present), i)
            for i, tf in enumerate(self._tf)
            if (present := qterms & tf.keys())
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored]

    def retrieve(self, queries: list[str], k: int = _HYBRID_TOP_K) -> list[dict]:
        """Top-k raw chunks across all queries, fusing per-query vector+keyword rankings with
        RRF. Returns evidence dicts (tagged 'doc') for the agent's synthesis step."""
        queries = [q for q in (str(q).strip() for q in queries) if q]
        if not self.chunks or not queries:
            return []
        # Vector side is best-effort and batched; skip entirely if embeddings are down.
        qvecs = self._embed_queries(queries)
        cvecs = self._ensure_chunk_vecs() if qvecs is not None else None

        ranked_lists: list[list[int]] = []
        for qi, q in enumerate(queries):
            ranked_lists.append(self._keyword_rank(set(content_terms(q))))
            if cvecs is not None and qvecs is not None:
                sims = cvecs @ qvecs[qi]
                ranked_lists.append(list(np.argsort(-sims)))

        fused = _rrf_fuse(ranked_lists)
        if not fused:
            return []
        top = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        used_vector = cvecs is not None
        # Scale RRF (~0.0–0.05) into a small positive band that sits BELOW graph answers, so
        # raw chunks are always present but graph stays primary.
        return [{
            "source": f"hybrid/{idx}",                  # unique → each chunk is its own block
            "method": "doc",                            # a verbatim document passage
            "query": "vector+keyword" if used_vector else "keyword",
            "text": self.chunks[idx],
            "score": 50.0 - rank,
        } for rank, idx in enumerate(top)]
