"""Load env.txt (repo root, shared with the other *_repo projects) and expose typed constants."""
import logging
import os
import sys
import warnings
from pathlib import Path

# Keep the agent trace / server logs clean: the in-process Microsoft GraphRAG backend
# otherwise emits tqdm progress bars, a numpy swapaxes FutureWarning, LiteLLM warnings,
# and asyncio "Task was destroyed" chatter (each GraphRAG search runs in its own
# short-lived event loop in a worker thread, orphaning LiteLLM's background logger).
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", message=".*swapaxes.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*was never awaited.*", category=RuntimeWarning)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

_BASE = Path(__file__).parent
_ENV_PATH = _BASE.parent / "env.txt"  # shared repo-root env.txt (youtube_repo/blog_repo also live here)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env(_ENV_PATH)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
# Unused by this app (no vision pipeline) but kept so llm_client.py (ported as-is) imports cleanly.
VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
VISION_BASE_URL: str = os.getenv("VISION_BASE_URL", "")
LLM_SLEEP_SECONDS: float = float(os.getenv("LLM_SLEEP_SECONDS", "1.0"))

BASE_DIR: Path = _BASE
DATA_DIR: Path = _BASE / "data"
GLOBAL_SCOPE_ID = "_global"

# Bundled-but-editable template (settings.yaml + GraphRAG prompts) copied into each
# scope's root at data/<scope_id>/graphrag/ on first build.
GRAPHRAG_TEMPLATE_DIR: Path = _BASE / "graphrag_template"

# Agentic-RAG orchestrator knobs (mirrors paper_read_project_pdf/config.py defaults).
MAX_ITERS: int = int(os.getenv("MAX_ITERS", "3"))
MAX_SNIPPETS: int = int(os.getenv("MAX_SNIPPETS", "24"))
FANOUT_TOP_K: int = int(os.getenv("FANOUT_TOP_K", "5"))

# Models GraphRAG itself uses for indexing + search (default to the app's models).
GRAPHRAG_CHAT_MODEL: str = os.getenv("GRAPHRAG_CHAT_MODEL", LLM_MODEL)
GRAPHRAG_EMBED_MODEL: str = os.getenv("GRAPHRAG_EMBED_MODEL", EMBEDDING_MODEL)
GRAPHRAG_CONCURRENT_REQUESTS: int = int(os.getenv("GRAPHRAG_CONCURRENT_REQUESTS", "4"))

# Whole-file chat mode: a transcript longer than this (chars) gets truncated rather than
# blowing the context window.
WHOLE_FILE_MAX_CHARS: int = int(os.getenv("WHOLE_FILE_MAX_CHARS", "60000"))
