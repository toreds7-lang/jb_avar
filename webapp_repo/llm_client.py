"""LangChain-based chat and embeddings client with OpenAI backend. Supports vLLM via LLM_BASE_URL."""
from typing import Iterable, Any
import functools
import json
import re
import time

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
    VISION_MODEL,
    VISION_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    LLM_SLEEP_SECONDS,
)

_sleep_seconds: float = LLM_SLEEP_SECONDS


def set_llm_sleep(seconds: float) -> None:
    global _sleep_seconds
    _sleep_seconds = max(0.0, seconds)


def llm_sleep() -> None:
    """Sleep between consecutive batch LLM calls to avoid overwhelming the server."""
    if _sleep_seconds > 0:
        time.sleep(_sleep_seconds)


@functools.lru_cache(maxsize=8)
def _get_llm(model: str):
    """Get or create a cached LangChain LLM instance."""
    kwargs = {}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
        kwargs["api_key"] = "EMPTY"
    else:
        kwargs["api_key"] = OPENAI_API_KEY
    return init_chat_model(model, model_provider="openai", **kwargs)


@functools.lru_cache(maxsize=8)
def _get_vision_llm(model: str):
    """Get or create a cached LangChain vision LLM instance."""
    kwargs = {}
    if VISION_BASE_URL:
        kwargs["base_url"] = VISION_BASE_URL
        kwargs["api_key"] = "EMPTY"
    else:
        kwargs["api_key"] = OPENAI_API_KEY
    return init_chat_model(model, model_provider="openai", **kwargs)


@functools.lru_cache(maxsize=2)
def _get_embeddings(model: str):
    """Get or create a cached LangChain embeddings instance."""
    kwargs = {"model": model}
    if EMBEDDING_BASE_URL:
        kwargs["base_url"] = EMBEDDING_BASE_URL
        kwargs["openai_api_key"] = "EMPTY"
    else:
        kwargs["openai_api_key"] = OPENAI_API_KEY
    return OpenAIEmbeddings(**kwargs)


def _to_lc_messages(messages: list[dict]):
    """Convert OpenAI-format message dicts to LangChain message objects."""
    mapping = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [mapping[m["role"]](content=m["content"]) for m in messages]


def chat(system: str, user: str, model: str = LLM_MODEL) -> str:
    """Non-streaming chat completion."""
    llm = _get_llm(model)
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return result.content or ""


def stream_messages(messages: list[dict], model: str = LLM_MODEL):
    """Yield content tokens from a streaming chat completion."""
    llm = _get_llm(model)
    for chunk in llm.stream(_to_lc_messages(messages)):
        token = chunk.content or ""
        if token:
            yield token


def vision_chat(
    system: str,
    user_text: str,
    image_b64: str,
    mime: str = "image/png",
    model: str = VISION_MODEL,
) -> str:
    """Non-streaming multimodal completion: one image plus a text instruction.
    Uses the dedicated vision model/endpoint (VISION_MODEL / VISION_BASE_URL)."""
    llm = _get_vision_llm(model)
    human = HumanMessage(content=[
        {"type": "text", "text": user_text},
        {"type": "image_url",
         "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ])
    result = llm.invoke([SystemMessage(content=system), human])
    return result.content or ""


def stream_vision_messages(messages: list[dict], model: str = VISION_MODEL):
    """Multimodal variant: messages may contain content lists with
    {'type': 'image_url', 'image_url': {...}} entries. Uses the dedicated
    vision model/endpoint configured via VISION_MODEL and VISION_BASE_URL."""
    llm = _get_vision_llm(model)
    for chunk in llm.stream(_to_lc_messages(messages)):
        token = chunk.content or ""
        if token:
            yield token


def embed(texts: Iterable[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a list of strings; returns (N, D) float32 array, L2-normalized."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embedder = _get_embeddings(model)
    vecs = np.array(embedder.embed_documents(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# ---------------------------------------------------------------------------
# Structured-output helper used by the agentic RAG layer
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort parse of the first JSON object/array out of an LLM reply.

    Tries, in order: the whole string, a ```json fenced block, then the first
    balanced {...} or [...] span. Returns the parsed value, or None if nothing
    parses — callers degrade gracefully on None rather than crashing the loop."""
    if not text:
        return None
    candidates: list[str] = [text.strip()]
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            candidates.append(text[i : j + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def chat_json(system: str, user: str, model: str = LLM_MODEL) -> Any:
    """Chat completion whose reply is parsed as JSON. Returns the parsed value, or
    None when the model produced nothing JSON-parseable (caller decides the fallback)."""
    system = system.rstrip() + "\n\nRespond with ONLY valid JSON. No prose, no code fences."
    raw = chat(system, user, model=model)
    return _extract_json(raw)
