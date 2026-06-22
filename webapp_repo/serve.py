"""FastAPI app: source loading, preprocessing, ingestion, chat, and knowledge-graph endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

import ai
import config
import graph as graphmod
import graphrag_manager
import lang_util
import preprocess as preprocessmod
import rag
import sources
import wiki

app = FastAPI(title="Source Knowledge Graph")

GLOBAL = config.GLOBAL_SCOPE_ID


def _ndjson(records) -> StreamingResponse:
    def gen():
        for rec in records:
            yield json.dumps(rec, ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    html_path = Path(__file__).parent / "viewer.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@app.get("/api/sources")
def api_list_sources():
    return JSONResponse(sources.list_sources())


@app.get("/api/sources/{source_id}")
def api_get_source(source_id: str, clean: bool = False):
    entry = sources.get_source(source_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    text = sources.load_text(source_id, clean=clean)
    return JSONResponse({**entry, "text": text})


@app.post("/api/sources/youtube")
def api_load_youtube(url: str = Form(...), languages: str = Form("en")):
    langs = [s.strip() for s in languages.split(",") if s.strip()] or ["en"]
    try:
        entry = sources.load_from_youtube(url, langs)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    text = sources.load_text(entry["id"], clean=False)
    return JSONResponse({**entry, "text": text})


@app.post("/api/sources/upload")
async def api_upload(file: UploadFile = File(...)):
    content = await file.read()
    entry = sources.load_from_upload(file.filename, content)
    text = sources.load_text(entry["id"], clean=False)
    return JSONResponse({**entry, "text": text})


@app.post("/api/sources/{source_id}/preprocess")
def api_preprocess(source_id: str):
    entry = sources.get_source(source_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    raw_text = sources.load_text(source_id, clean=False)

    def run():
        clean_text = ""
        for rec in preprocessmod.preprocess(raw_text):
            if rec.get("done"):
                clean_text = rec.get("clean_text", "")
            yield rec
        sources.save_clean_text(source_id, clean_text)

    return _ndjson(run())


# ---------------------------------------------------------------------------
# Ingest: build per-source AND global RAG index, concept graph, wiki, GraphRAG.
# ---------------------------------------------------------------------------

@app.post("/api/sources/{source_id}/ingest")
def api_ingest(source_id: str):
    entry = sources.get_source(source_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    clean_text = sources.load_text(source_id, clean=True)
    if not clean_text.strip():
        return JSONResponse({"error": "no cleaned text — run /preprocess first"}, status_code=400)

    def run():
        yield {"stage": "rag_source", "msg": "Embedding this source..."}
        rag.build_index([{"id": source_id, "text": clean_text}], sources.source_dir(source_id))

        yield {"stage": "rag_global", "msg": "Rebuilding cumulative RAG index..."}
        all_texts = [{"id": source_id, "text": clean_text}]
        for other in sources.list_sources():
            if other["id"] != source_id and other.get("ingested"):
                t = sources.load_text(other["id"], clean=True)
                if t.strip():
                    all_texts.append({"id": other["id"], "text": t})
        rag.build_index(all_texts, sources.source_dir(GLOBAL))

        yield {"stage": "graph_source", "msg": "Extracting concepts (this source)..."}
        for rec in graphmod.ingest_into_scope(source_id, source_id, clean_text, entry.get("title")):
            yield rec
            if rec.get("done"):
                g = graphmod.load_graph(source_id)
                wiki.rebuild_index(source_id, g)

        yield {"stage": "graph_global", "msg": "Merging concepts into cumulative graph..."}
        for rec in graphmod.ingest_into_scope(GLOBAL, source_id, clean_text, entry.get("title")):
            yield rec
            if rec.get("done"):
                g = graphmod.load_graph(GLOBAL)
                wiki.rebuild_index(GLOBAL, g)

        # NOTE: the two builds below run sequentially, not concurrently. Running GraphRAG's
        # build_index() for two different roots at the same time in this process was
        # observed to silently produce an empty output/ for one of the two roots (no
        # exception raised) — some shared state inside the graphrag package isn't safe for
        # concurrent indexing runs. A background thread per build is still fine as long as
        # only one is ever in flight at a time, so build_source_then_global runs both in a
        # single background thread, one after the other.
        yield {"stage": "graphrag", "msg": "Starting GraphRAG builds (per-source, then cumulative)..."}
        graphrag_manager.write_source_input(source_id, source_id, clean_text)
        graphrag_manager.write_source_input(GLOBAL, source_id, clean_text)
        graphrag_manager.build_sequential_async([source_id, GLOBAL], force=True)

        sources.mark_ingested(source_id)
        yield {"done": True, "ok": True, "source_id": source_id}

    return _ndjson(run())


@app.get("/api/graphrag/status")
def api_graphrag_status(scope_id: str):
    return JSONResponse(graphrag_manager.read_status(scope_id))


# ---------------------------------------------------------------------------
# Knowledge graph + wiki
# ---------------------------------------------------------------------------

def _resolve_scope_id(scope: str, source_id: str | None) -> str:
    if scope == "global":
        return GLOBAL
    if not source_id:
        raise ValueError("source_id required when scope=video")
    return source_id


@app.get("/api/graph")
def api_graph(scope: str = "video", source_id: str | None = None):
    try:
        scope_id = _resolve_scope_id(scope, source_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(graphmod.get_graph(scope_id))


@app.get("/api/wiki/{scope}/{concept_id}")
def api_wiki_page(scope: str, concept_id: str, source_id: str | None = None):
    try:
        scope_id = _resolve_scope_id(scope, source_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    content = wiki.load_page(scope_id, concept_id)
    if content is None:
        g = graphmod.load_graph(scope_id)
        node = next((n for n in g.get("nodes", []) if n["id"] == concept_id), None)
        if not node:
            return JSONResponse({"error": "concept not found"}, status_code=404)
        idx = rag.RagIndex(sources.source_dir(scope_id))
        content = wiki.generate_page(node, g, idx)
        wiki.write_page(scope_id, concept_id, content)
    return JSONResponse({"concept_id": concept_id, "markdown": wiki.resolve_wiki_links(content)})


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def api_chat(payload: dict[str, Any]):
    message: str = payload.get("message", "")
    history: list[dict] = payload.get("history", [])
    scope: str = payload.get("scope", "video")       # "video" | "global"
    mode: str = payload.get("mode", "rag")           # "rag" | "agentic_graphrag" | "whole_file"
    source_id: str | None = payload.get("source_id")
    use_clean: bool = payload.get("use_clean", True)  # whole_file only
    prompt_lang: str = payload.get("prompt_lang", "auto")  # "auto" | "en" | "ko"

    try:
        scope_id = _resolve_scope_id(scope, source_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    lang = prompt_lang if prompt_lang in ("en", "ko") else lang_util.detect_lang(message)

    def text_stream():
        if mode == "whole_file":
            if scope != "video" or not source_id:
                yield "Whole-file mode is only available for a single source (scope=video)."
                return
            full_text = sources.load_text(source_id, clean=use_clean)
            entry = sources.get_source(source_id) or {}
            yield from ai.whole_file_chat_stream(
                message, history, full_text, entry.get("title", source_id), lang=lang,
            )
            return

        if mode == "agentic_graphrag":
            engine = graphrag_manager.get_engine(scope_id)
            if engine is not None:
                yield from ai.agentic_graphrag_chat_stream(message, engine, lang=lang)
                return
            status = graphrag_manager.read_status(scope_id)
            yield f"> GraphRAG index not ready yet ({status.get('state', 'idle')}) — answering from vector search instead.\n\n"
            # fall through to rag

        idx = rag.RagIndex(sources.source_dir(scope_id))
        source_filter = source_id if scope == "video" else None
        yield from ai.rag_chat_stream(idx, message, history, source_filter=source_filter, lang=lang)

    return StreamingResponse(text_stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
