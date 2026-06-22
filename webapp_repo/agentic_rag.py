"""Agentic-RAG orchestration over a per-scope Microsoft GraphRAG backend.

The agentic loop is the brain; GraphRAG's four retrieval methods (global / local /
drift / basic) are its tools:

    Planner (sub-question + method each) -> Search Fanout (call GraphRAG per item)
            -> Sufficient-Context gate -> (loop back with follow-up {q, method})
            -> Synthesis (cited by method)

Ported from paper_read_project_pdf/agentic_rag.py (itself ported from a project named
gemini_rag — despite the name, neither project calls Google's Gemini API; this is a
provider-agnostic loop running on this app's OpenAI-compatible llm_client). One change
from the source: _corpus_text() concatenates ALL input/*.txt files in the scope's
GraphRAG root instead of just the first — required for the global scope, whose root
accumulates one file per ingested source.
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import hybrid_retrieval as hybrid
import llm_client
from config import MAX_ITERS, MAX_SNIPPETS
from graphrag_qa import GraphRAGQA, METHODS

_PROMPTS = Path(__file__).parent / "prompts"

_ESCALATE = {"basic": "local", "local": "drift", "drift": "global", "global": "global"}


class GraphRAGUnavailable(RuntimeError):
    pass


_NONANSWER_RE = re.compile(
    r"i\s*(?:'?a?m)?\s+sorry|unable\s+to\s+answer|cannot\s+answer|can'?t\s+answer|"
    r"do(?:es)?\s+not\s+contain\s+(?:any\s+)?(?:information|relevant|details?)|"
    r"does\s+not\s+(?:provide|mention|include|discuss)|no\s+(?:relevant\s+)?information|"
    r"not\s+enough\s+information|do(?:\s+not|n'?t)\s+have\s+(?:enough\s+)?(?:information|details?)|"
    r"(?:provided|retrieved)\s+(?:data|evidence|context|text|information)\s+does\s+not",
    re.IGNORECASE,
)
_NONANSWER_MAX_CHARS = 700


def _is_nonanswer(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= _NONANSWER_MAX_CHARS and _NONANSWER_RE.search(t) is not None


def _prompt(name: str, lang: str = "en") -> str:
    path = _PROMPTS / f"{name}.system.txt"
    if lang == "ko":
        ko_path = path.with_name(f"{path.stem}.ko{path.suffix}")
        if ko_path.exists():
            path = ko_path
    return path.read_text(encoding="utf-8").strip()


def _norm_method(m: Any) -> str:
    m = str(m or "").lower().strip()
    return m if m in METHODS or m == "auto" else "auto"


def _norm_item(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        q = item.strip()
        return {"q": q, "method": "auto"} if q else None
    if isinstance(item, dict):
        q = str(item.get("q") or item.get("question") or "").strip()
        return {"q": q, "method": _norm_method(item.get("method"))} if q else None
    return None


def _ask_graphrag(engine: GraphRAGQA, question: str, method: str = "auto") -> dict:
    m = method.lower().strip()
    if m not in METHODS and m != "auto":
        m = "auto"
    try:
        ans = engine.ask(question, method=m)
    except Exception as e:
        raise GraphRAGUnavailable(f"GraphRAG search failed ({m}): {e}") from e
    return {"text": ans.text, "method": ans.method, "routed": ans.routed}


_SECTION_MAX_CHARS = 6000


def _corpus_text(engine: GraphRAGQA) -> str:
    """Concatenate every document in the scope's input/ dir (best-effort). A per-source
    scope has one file; the global scope has one file per ingested source."""
    try:
        files = sorted((engine.root / "input").glob("*.txt"))
        parts = [f.read_text(encoding="utf-8") for f in files]
        return "\n\n".join(parts)
    except Exception:
        return ""


def list_documents(engine: GraphRAGQA) -> tuple[str, ...]:
    try:
        import pandas as pd
        docs = pd.read_parquet(engine.root / "output" / "documents.parquet")
        col = "title" if "title" in docs.columns else docs.columns[0]
        return tuple(str(t) for t in docs[col].tolist())
    except Exception:
        return ()


_FALLBACK_CHUNK_CHARS = 1600
_FALLBACK_TOP_K = 4


def _corpus_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if len(para) < 60:
            continue
        for i in range(0, len(para), _FALLBACK_CHUNK_CHARS):
            chunks.append(para[i:i + _FALLBACK_CHUNK_CHARS])
    return chunks


def corpus_fallback(engine: GraphRAGQA, question: str, k: int = _FALLBACK_TOP_K) -> list[dict]:
    chunks = _corpus_chunks(_corpus_text(engine))
    if not chunks:
        return []
    qterms = set(hybrid.content_terms(question))
    scored: list[tuple[float, str]] = []
    for ch in chunks:
        cterms = hybrid.content_terms(ch)
        present = qterms.intersection(cterms)
        if not present:
            continue
        scored.append((len(present) + 0.1 * sum(cterms.count(t) for t in present), ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:k] if scored else [(0.0, ch) for ch in chunks[:k]]
    return [{
        "source": "corpus/fallback", "method": "corpus", "query": question,
        "text": ch[:_SECTION_MAX_CHARS], "score": 100.0 - rank,
    } for rank, (_score, ch) in enumerate(picked)]


_retrievers: dict[str, hybrid.HybridRetriever] = {}


def _retriever_for(engine: GraphRAGQA) -> hybrid.HybridRetriever | None:
    key = str(engine.root)
    r = _retrievers.get(key)
    if r is not None and r.stale():
        _retrievers.pop(key, None)
        r = None
    if r is None:
        try:
            r = hybrid.HybridRetriever(engine.root)
        except Exception as e:
            print(f"[hybrid] retriever unavailable for {key}: {e}", file=sys.stderr)
            return None
        _retrievers[key] = r
    return r


def hybrid_evidence(engine: GraphRAGQA, queries: list[str]) -> list[dict]:
    r = _retriever_for(engine)
    if r is None:
        return []
    try:
        return r.retrieve(queries)
    except Exception as e:
        print(f"[hybrid] retrieval failed: {e}", file=sys.stderr)
        return []


def hybrid_fanout(items: list[dict[str, str]], engine: GraphRAGQA) -> list[dict]:
    queries = [it["q"] for it in items if it.get("q")]
    return hybrid_evidence(engine, queries)


def _evidence_key(e: dict[str, Any]) -> tuple[str, str]:
    return (str(e.get("source", "")), str(e.get("query", "")))


def _merge_evidence(existing: list[dict], new: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for e in existing + new:
        k = _evidence_key(e)
        if k not in by_key or e.get("score", 0.0) > by_key[k].get("score", 0.0):
            by_key[k] = e
    return sorted(by_key.values(), key=lambda e: e.get("score", 0.0), reverse=True)[:MAX_SNIPPETS]


def _render_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for i, e in enumerate(evidence, 1):
        method = str(e.get("method") or e.get("source", "")).split("/")[-1] or "?"
        query = str(e.get("query", "")).strip()
        head = f"[{i}] (via {method})" + (f"  query: {query}" if query else "")
        lines.append(f"{head}\n{str(e.get('text', '')).strip()}")
    return "\n\n".join(lines)


def plan(question: str, engine: GraphRAGQA, lang: str = "en") -> dict[str, Any]:
    docs = list_documents(engine)
    corpus = ""
    if docs:
        listing = "\n".join(f"- {d}" for d in docs)
        corpus = (f"The corpus contains exactly {len(docs)} document(s):\n{listing}\n\n"
                  "Treat this list as authoritative — do not invent or assume any other documents.\n\n")
    out = llm_client.chat_json(_prompt("planner", lang), f"{corpus}Question: {question}")
    items: list[dict[str, str]] = []
    if isinstance(out, dict) and isinstance(out.get("subquestions"), list):
        items = [it for it in (_norm_item(x) for x in out["subquestions"]) if it]
    elif isinstance(out, list):
        items = [it for it in (_norm_item(x) for x in out) if it]
    if not items:
        return {"reasoning": "(planner fallback)", "subquestions": [{"q": question, "method": "auto"}]}
    reasoning = str(out.get("reasoning", "")) if isinstance(out, dict) else ""
    return {"reasoning": reasoning, "subquestions": items}


def search_fanout(items: list[dict[str, str]], engine: GraphRAGQA) -> list[dict]:
    def run(item: dict[str, str], rank: int) -> dict | None:
        try:
            res = _ask_graphrag(engine, item["q"], item["method"])
        except GraphRAGUnavailable:
            return None
        if not res.get("text") or _is_nonanswer(res["text"]):
            return None
        return {
            "source": f"graphrag/{res['method']}", "method": res["method"], "query": item["q"],
            "text": res["text"], "score": 500.0 + (len(items) - rank),
        }

    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(4, len(items))) as ex:
        results = list(ex.map(lambda p: run(*p), [(it, i) for i, it in enumerate(items)]))
    return _merge_evidence([], [r for r in results if r])


def assess_sufficiency(question: str, evidence: list[dict], lang: str = "en") -> dict[str, Any]:
    user = f"Question: {question}\n\nRetrieved evidence:\n{_render_evidence(evidence)}"
    out = llm_client.chat_json(_prompt("sufficient_context", lang), user)
    if not isinstance(out, dict):
        return {"sufficient": True, "draft": "", "missing": [], "followup_queries": []}
    followups = [it for it in (_norm_item(x) for x in (out.get("followup_queries") or [])) if it]
    return {
        "sufficient": bool(out.get("sufficient", True)), "draft": str(out.get("draft", "")),
        "missing": [str(m) for m in (out.get("missing") or [])], "followup_queries": followups,
    }


def _escalate(items: list[dict[str, str]], prior: list[dict[str, str]]) -> list[dict[str, str]]:
    prior_methods = {it["method"] for it in prior}
    base = max(prior_methods, key=lambda m: list(_ESCALATE).index(m) if m in _ESCALATE else -1) if prior_methods else "local"
    out = []
    for it in items:
        method = it["method"] if it["method"] != "auto" else _ESCALATE.get(base, "global")
        out.append({"q": it["q"], "method": method})
    return out


def synthesize(question: str, evidence: list[dict], lang: str = "en") -> Iterator[str]:
    user = f"Question: {question}\n\nRetrieved evidence:\n{_render_evidence(evidence)}"
    messages = [
        {"role": "system", "content": _prompt("synthesis", lang)},
        {"role": "user", "content": user},
    ]
    yield from llm_client.stream_messages(messages)


@dataclass
class IterationTrace:
    index: int
    items: list[dict[str, str]]
    n_evidence: int
    sufficient: bool
    missing: list[str] = field(default_factory=list)
    followup_queries: list[dict[str, str]] = field(default_factory=list)


TraceCb = Callable[[str, Any], None]


def stream_with_trace(
    question: str, engine: GraphRAGQA, use_graph: bool = True, lang: str = "en",
) -> Iterator[str]:
    """Run plan -> search -> sufficiency-gate -> synthesis, folding a compact markdown
    trace into the same plain-text stream as the final answer."""
    p = plan(question, engine, lang)
    subs = p["subquestions"]
    yield "**Planner**\n"
    if p.get("reasoning"):
        yield f"> {p['reasoning']}\n"
    for it in subs:
        yield f"- {it['q']}" + (f"  _[{it['method']}]_\n" if use_graph else "\n")
    yield "\n"

    items = subs
    queries = [question] + [it["q"] for it in subs]
    hits = hybrid_evidence(engine, queries)
    if hits:
        how = hits[0].get("query", "keyword")
        tail = "alongside graph search" if use_graph else "(graph search excluded)"
        yield f"**Hybrid retrieval** — added {len(hits)} raw passage(s) ({how}) {tail}\n\n"
    evidence: list[dict] = _merge_evidence([], hits)
    for i in range(MAX_ITERS):
        label = f"methods: {', '.join(it['method'] for it in items)}" if use_graph else "vector+keyword retrieval"
        yield f"**Search + Sufficiency** _(iteration {i + 1})_ — {label}\n"
        fresh = search_fanout(items, engine) if use_graph else hybrid_fanout(items, engine)
        evidence = _merge_evidence(evidence, fresh)
        verdict = assess_sufficiency(question, evidence, lang)
        state = "sufficient" if verdict["sufficient"] else "insufficient, retrying"
        yield f"> {len(evidence)} evidence block(s) — {state}\n"
        if not verdict["sufficient"]:
            for m in verdict["missing"]:
                yield f"> missing: {m}\n"
        yield "\n"
        if verdict["sufficient"] or not verdict["followup_queries"]:
            break
        items = _escalate(verdict["followup_queries"], items) if use_graph else verdict["followup_queries"]

    if not evidence:
        evidence = corpus_fallback(engine, question)
        if evidence:
            yield f"**Corpus fallback** — grounding on {len(evidence)} raw passage(s) pulled directly from the document(s)\n\n"

    yield f"**Synthesis** _(grounding on {len(evidence)} block(s))_\n\n---\n\n"
    yield from synthesize(question, evidence, lang)
