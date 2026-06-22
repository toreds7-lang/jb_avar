"""Chat dispatch across the three modes: rag, agentic_graphrag, whole_file.

Prompt templates live in ./prompts/*.txt and are reloaded on every call, so edits take
effect without restarting the server (same convention as paper_read_project_pdf/ai.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import llm_client
from config import WHOLE_FILE_MAX_CHARS
from rag import RagIndex

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load(name: str, lang: str = "en") -> str:
    path = _PROMPTS_DIR / name
    if lang == "ko":
        ko_path = path.with_name(f"{path.stem}.ko{path.suffix}")
        if ko_path.exists():
            path = ko_path
    return path.read_text(encoding="utf-8").strip()


def _format_context(snippets: list[dict]) -> str:
    parts = []
    for s in snippets:
        parts.append(f"[{s['source_id']}]\n{s['text'].strip()}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: list[dict], n_pairs: int = 3) -> str:
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None
    pairs = pairs[-n_pairs:]
    if not pairs:
        return "(no prior turns)"
    return "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in pairs)


def rag_chat_stream(
    index: RagIndex, message: str, history: list[dict], source_filter: str | None = None,
    lang: str = "en",
) -> Iterator[str]:
    """Vector top-k RAG over the scope's rag.json (works for both per-source and global scope)."""
    hits = index.topk(message, k=5, source_filter=source_filter)
    context = _format_context(hits) if hits else "(no indexed excerpts found)"
    scope_note = "Use the most relevant excerpts; cite the [source_id] you used."
    user_msg = _load("chat.user.txt", lang).format(
        scope_note=scope_note, context=context, history=_format_history(history), question=message.strip(),
    )
    messages = [
        {"role": "system", "content": _load("chat.system.txt", lang)},
        {"role": "user", "content": user_msg},
    ]
    yield from llm_client.stream_messages(messages)


def agentic_graphrag_chat_stream(
    question: str, engine, use_graph: bool = True, lang: str = "en",
) -> Iterator[str]:
    """Agentic GraphRAG: planner -> fanout search -> sufficiency gate -> synthesis. Caller
    is responsible for checking the engine is ready (graphrag_manager.get_engine)."""
    import agentic_rag  # lazy: pulls in graphrag only when actually used
    yield from agentic_rag.stream_with_trace(question, engine, use_graph=use_graph, lang=lang)


def whole_file_chat_stream(
    message: str, history: list[dict], full_text: str, label: str, lang: str = "en",
) -> Iterator[str]:
    """Whole-file: the entire (raw or cleaned) source text is the only context, no retrieval."""
    text = full_text
    if len(text) > WHOLE_FILE_MAX_CHARS:
        text = text[:WHOLE_FILE_MAX_CHARS] + "\n\n[...truncated...]"
    user_msg = _load("whole_file.user.txt", lang).format(
        label=label, text=text, history=_format_history(history), question=message.strip(),
    )
    messages = [
        {"role": "system", "content": _load("whole_file.system.txt", lang)},
        {"role": "user", "content": user_msg},
    ]
    yield from llm_client.stream_messages(messages)
