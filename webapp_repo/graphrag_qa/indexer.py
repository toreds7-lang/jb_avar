"""In-process GraphRAG index build.

The original gemini_rag indexer shelled out to `sys.executable -m graphrag index`,
which cannot work inside a frozen PyInstaller exe (there `sys.executable` is the app,
not a Python interpreter). Here we call the in-process async API
`graphrag.api.build_index(config=load_config(root))` instead, so the same code path
works from source and from the bundled exe.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import graphrag.api as api
from graphrag.config.load_config import load_config

from .config import get_settings


def build_index(root: str | Path) -> None:
    """Build the GraphRAG index for a single per-paper root (in-process, blocking).

    Raises on failure so the caller (graphrag_manager) can record a 'failed' status.
    """
    root = Path(root)
    get_settings()  # ensures GRAPHRAG_API_KEY is present for ${...} substitution
    cfg = load_config(root)
    asyncio.run(api.build_index(config=cfg))
