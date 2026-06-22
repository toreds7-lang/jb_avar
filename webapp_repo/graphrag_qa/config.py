"""Configuration + key loading for graphrag_qa (per-paper GraphRAG roots).

Unlike the original gemini_rag layout (one shared `ragproject/`), this app gives
every paper its own GraphRAG root at `data/<paper_id>/graphrag/`. So the root is
always passed explicitly to `GraphRAGQA(root=...)` / `build_index(root=...)`; the
helpers here only resolve the OpenAI key and surface it to GraphRAG's
`${GRAPHRAG_API_KEY}` settings substitution.

The key is read from this project's `config.OPENAI_API_KEY` (which itself loads
`env.txt`) and is always whitespace/CR-stripped — env.txt uses Windows CRLF, and an
unstripped key yields an illegal Authorization header that surfaces as a confusing
"Connection error".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import config as _app_config  # this project's config (loads env.txt)


def load_api_key() -> str:
    """Return the OpenAI key, CR/whitespace-stripped. Order: env var, app config."""
    key = (
        os.environ.get("GRAPHRAG_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or _app_config.OPENAI_API_KEY
    )
    if not key:
        raise RuntimeError("No OpenAI API key found (GRAPHRAG_API_KEY / OPENAI_API_KEY / env.txt)")
    return key.strip()


@dataclass
class Settings:
    api_key: str
    chat_model: str = _app_config.LLM_MODEL
    embedding_model: str = _app_config.EMBEDDING_MODEL


def get_settings() -> Settings:
    """Ensure GraphRAG can see the key via ${GRAPHRAG_API_KEY}, return resolved Settings."""
    key = load_api_key()
    # GraphRAG's settings.yaml substitutes ${GRAPHRAG_API_KEY}; make sure it's present.
    os.environ.setdefault("GRAPHRAG_API_KEY", key)
    return Settings(api_key=key)
