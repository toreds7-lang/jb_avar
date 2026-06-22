"""GraphRAGQA — reusable query layer over one built Microsoft GraphRAG index.

Loads the output parquets once and answers many questions, so a long-running server
can reuse one engine per paper. Wraps the async `graphrag.api` search functions
(verified against graphrag 3.1.0) behind a sync `ask()`.

    from graphrag_qa import GraphRAGQA
    qa = GraphRAGQA(root="data/<paper_id>/graphrag")
    qa.ask("What is multi-head attention?", method="local").text
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import graphrag.api as api
from graphrag.config.load_config import load_config

from .config import get_settings

METHODS = ("global", "local", "drift", "basic")

_ROUTER_SYSTEM = (
    "You route a question to ONE retrieval method for a GraphRAG system over a "
    "document. Reply with ONLY one word: global, local, drift, or basic.\n"
    "- global: broad, whole-document sensemaking (overall themes, taxonomy, summary, "
    "what are the main/key ...).\n"
    "- local: specific facts about particular entities (what is X, who, define, "
    "attributes or relationships of a named thing, which/list).\n"
    "- drift: complex questions needing both the big picture AND specifics "
    "(compare several things and how each works, multi-hop).\n"
    "- basic: simple verbatim lookup of a passage; no graph reasoning needed."
)


@dataclass
class Answer:
    text: str
    method: str          # the method actually used (resolved if 'auto' was requested)
    routed: bool = False  # True if the method was chosen automatically


class GraphRAGQA:
    def __init__(self, root: str | Path, community_level: int = 2,
                 response_type: str = "multiple paragraphs"):
        self.root = Path(root)
        self.settings = get_settings()  # ensures GRAPHRAG_API_KEY in env (CR-stripped)
        self.config = load_config(self.root)
        self.community_level = community_level
        self.response_type = response_type
        self._load_outputs()

    def _load_outputs(self) -> None:
        out = self.root / "output"
        missing = [n for n in ("entities", "communities", "community_reports",
                               "text_units", "relationships")
                   if not (out / f"{n}.parquet").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing index outputs {missing} in {out}. "
                f"Build the index first (graphrag_manager.build).")
        self.entities = pd.read_parquet(out / "entities.parquet")
        self.communities = pd.read_parquet(out / "communities.parquet")
        self.community_reports = pd.read_parquet(out / "community_reports.parquet")
        self.text_units = pd.read_parquet(out / "text_units.parquet")
        self.relationships = pd.read_parquet(out / "relationships.parquet")

    def ask(self, question: str, method: str = "auto") -> Answer:
        m = method.lower()
        routed = m == "auto"
        if routed:
            m = self.route(question)
        if m not in METHODS:
            raise ValueError(f"method must be one of {METHODS + ('auto',)}, got {method!r}")
        resp, _ctx = asyncio.run(self._search(m, question))
        return Answer(text=str(resp), method=m, routed=routed)

    # --- automatic method selection ------------------------------------------
    def route(self, question: str) -> str:
        """Pick global/local/drift/basic for a question (LLM, heuristic fallback)."""
        try:
            return self._route_llm(question)
        except Exception:
            return self._route_heuristic(question)

    def _route_llm(self, question: str) -> str:
        # Route via this project's llm_client so it honors LLM_BASE_URL (local servers)
        # and the configured model, instead of a hard-coded OpenAI client.
        import llm_client
        out = (llm_client.chat(_ROUTER_SYSTEM, question) or "").strip().lower()
        out = out.split()[0].strip(".,!:") if out else ""
        return out if out in METHODS else self._route_heuristic(question)

    @staticmethod
    def _route_heuristic(question: str) -> str:
        q = question.lower()
        if any(k in q for k in ("compare", "versus", " vs ", "difference between",
                                "differ", "trade-off", "relationship between")):
            return "drift"
        if any(k in q for k in ("verbatim", "exact wording", "quote", "find the sentence")):
            return "basic"
        if any(k in q for k in ("overall", "summary", "summarize", "main theme",
                                "key theme", "taxonomy", "in general", "high-level",
                                "categories", "across the", "what are the main")):
            return "global"
        if any(k in q for k in ("what is", "who is", "define", "definition", "which ",
                                "list ", "examples of", "explain ")):
            return "local"
        return "global"

    async def _search(self, m: str, query: str):
        if m == "global":
            return await api.global_search(
                config=self.config, entities=self.entities,
                communities=self.communities, community_reports=self.community_reports,
                community_level=self.community_level, dynamic_community_selection=False,
                response_type=self.response_type, query=query)
        if m == "local":
            return await api.local_search(
                config=self.config, entities=self.entities,
                communities=self.communities, community_reports=self.community_reports,
                text_units=self.text_units, relationships=self.relationships,
                covariates=None, community_level=self.community_level,
                response_type=self.response_type, query=query)
        if m == "drift":
            return await api.drift_search(
                config=self.config, entities=self.entities,
                communities=self.communities, community_reports=self.community_reports,
                text_units=self.text_units, relationships=self.relationships,
                community_level=self.community_level,
                response_type=self.response_type, query=query)
        # basic: plain vector RAG over text chunks
        return await api.basic_search(
            config=self.config, text_units=self.text_units,
            response_type=self.response_type, query=query)
