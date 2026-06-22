# Source Knowledge Graph

A FastAPI app that turns a loaded text source (a YouTube transcript, fetched live or
from a saved `.txt`) into a chat-able, explorable knowledge base: clean the raw text,
chat over it in three different ways, and browse an auto-built knowledge graph with
wiki pages — all in one page with a chat column, a text column, and a graph column.

Modeled directly on `D:\2026_Agent\1_Understanding_fast\paper_read_project_pdf`
(FastAPI + Microsoft GraphRAG + LLM-extracted concept graph) and
`D:\2026_Agent\llm-wiki` (sources accumulating into one cross-document wiki/graph).
Most of the backend modules are ported from the former with the PDF-specific parts
removed and a "scope" concept added so the same code drives both a single source's
graph and a cumulative graph across every source you've loaded.

## Running it

```bash
# from the repo root (D:\2026_Agent\3_주시기)
.venv\Scripts\python.exe -m pip install -r webapp_repo\requirements.txt
cd webapp_repo
..\.venv\Scripts\python.exe -m uvicorn serve:app --reload
```

Open `http://127.0.0.1:8000/`. Needs `OPENAI_API_KEY` (and friends) set in the
repo-root `env.txt` — see [Configuration](#configuration).

## Using the app

1. **Load a source** — paste a YouTube URL (with optional language preference, e.g.
   `ko,en`) and click **Load YouTube**, or pick a `.txt` file and click **Upload**.
   The raw text appears in the text column immediately.
2. **Preprocess** — click **Preprocess**. An LLM labels the text's natural
   paragraphs/sentences as `keep` (substance) or `drop` (intros, sponsor reads,
   "like and subscribe", filler) and the result is shown as a strikethrough diff. It
   never rewrites text, only keeps or drops what's already there, so nothing is
   paraphrased or hallucinated.
3. **Ingest** — click **Ingest**. This builds, for the source you just cleaned:
   - a vector RAG index,
   - an LLM-extracted concept graph + wiki pages,
   - a Microsoft GraphRAG index (entities/communities — takes the longest; runs in
     the background, poll-able via `/api/graphrag/status`),

   and does all of it **twice**: once into that source's own store, and once merged
   into a shared `_global` store that accumulates across every source you've ever
   ingested. That's what makes the scope toggle below work.
4. **Pick a scope** — *This source* (just the currently loaded one) or *All sources*
   (the cumulative `_global` graph/index). Switching doesn't require re-ingesting
   anything; both stores are always kept in sync at ingest time.
5. **Chat**, in one of three modes (dropdown next to the chat box):
   - **RAG** — top-k vector search over the chunked text.
   - **Agentic GraphRAG** — a planner breaks your question into sub-questions, each
     routed to a GraphRAG search method (`global`/`local`/`drift`/`basic`), a
     sufficiency gate decides whether to dig deeper, then a synthesis step writes a
     cited answer. Falls back to RAG if the index isn't built yet.
   - **Whole-file** — the entire source's raw or cleaned text (your choice) is
     stuffed into the prompt with no retrieval at all. Only available when scope is
     *This source*, since concatenating every source could blow the context window.
6. **Explore the graph** — the right column renders the scope's concept graph as a
   force-directed SVG. Click a node to open its wiki page (generated on first click,
   cached on disk after that).

## Architecture

```
webapp_repo/
├── serve.py              # FastAPI app — every HTTP endpoint lives here
├── sources.py             # YouTube fetch (reuses youtube_repo/get_transcript.py) + .txt
│                          # upload + the sources.json registry
├── preprocess.py           # segment text -> LLM keep/drop classification -> clean.txt
├── config.py                 # loads repo-root env.txt, typed constants, DATA_DIR
├── llm_client.py               # OpenAI-compatible chat/embedding wrapper (streaming, JSON mode)
├── rag.py                       # chunk + embed + cosine top-k, per scope
├── graph.py                      # LLM concept/relationship extraction + merge-into-graph
├── wiki.py                        # wiki page generation/read/write/index, per scope
├── graphrag_manager.py             # per-scope Microsoft GraphRAG lifecycle (build/status/engine)
├── graphrag_qa/                     # thin wrapper over graphrag.api (global/local/drift/basic + router)
├── agentic_rag.py                    # planner -> search fanout -> sufficiency gate -> synthesis loop
├── ai.py                              # chat_stream dispatch: rag / agentic_graphrag / whole_file
├── viewer.html                         # single-file frontend (vanilla JS, no build step)
├── graphrag_template/prompts/           # GraphRAG's own index/query prompts (copied per scope root)
├── prompts/                              # this app's prompts (chat, preprocess, planner, etc.)
└── data/
    ├── <source_id>/                       # one folder per ingested source
    │   ├── source.txt / clean.txt
    │   ├── rag.json / rag.npz                  # vector index over just this source
    │   ├── graph.json, wiki/*.md                # concept graph + wiki pages for this source
    │   └── graphrag/                              # this source's own GraphRAG root
    └── _global/                                     # the cumulative scope — same shape as
        ├── rag.json / rag.npz                       # above, but every file folds in every
        ├── graph.json, wiki/*.md                    # ingested source instead of just one
        └── graphrag/input/<source_id>.txt (one per source, never overwritten)
```

### The "scope" abstraction

`graph.py`, `wiki.py`, `rag.py`, and `graphrag_manager.py` don't know about "per-source"
vs. "cumulative" — they just take a `scope_id` string and read/write
`data/<scope_id>/...`. The per-source scope and the global scope
(`config.GLOBAL_SCOPE_ID = "_global"`) are the exact same code path; ingest just calls
each module twice, once with `scope_id=source_id` and once with `scope_id="_global"`.
`graphrag_manager.write_source_input(scope_id, source_id, text)` writes
`data/<scope_id>/graphrag/input/<source_id>.txt` — for a per-source scope that's one
file named after itself; for the global scope it's one file per source, so re-ingesting
a new source adds a file without touching previous ones, and a rebuild folds it into the
shared index.

### Chat dispatch (`ai.py` + `serve.py`'s `/api/chat`)

```
mode=rag              -> rag.RagIndex(scope_dir).topk(question)
mode=agentic_graphrag  -> graphrag_manager.get_engine(scope_id) ready?
                            yes -> agentic_rag.stream_with_trace(...)
                            no  -> fall back to rag mode + a "still building" notice
mode=whole_file         -> scope must be "video"; stuffs source.txt or clean.txt whole
```

### Known gotcha: GraphRAG builds must run sequentially

Running Microsoft GraphRAG's `build_index()` for two different roots concurrently in
two threads of the same process was observed to silently produce an **empty** `output/`
for one of the two roots — no exception, no error in the log, `status.json` still says
`"ready"`. Some state inside the `graphrag` package isn't safe for concurrent indexing
runs in-process. `graphrag_manager.build_sequential_async()` works around this with one
global lock shared across every scope, so ingest's per-source and global builds (and any
future concurrent ingests) always run one at a time, never in parallel. If you ever see a
scope report `"ready"` but `health(scope_id)` is `False` / the graph view comes back
empty, that lock was bypassed somewhere — check for a direct call to
`graphrag_manager.build()` or `build_async()` running alongside another build.

## Configuration

Read from the repo-root `env.txt` (shared with `youtube_repo`/`blog_repo`), via
`config.py`:

| Key | Purpose |
|---|---|
| `OPENAI_API_KEY` | required |
| `LLM_MODEL` | chat model (planner/gate/synthesis/extraction/chat all use this) |
| `LLM_BASE_URL` | optional — point at a local/self-hosted OpenAI-compatible server |
| `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL` | for RAG + hybrid retrieval |
| `GRAPHRAG_CHAT_MODEL`, `GRAPHRAG_EMBED_MODEL` | default to the above if unset |
| `GRAPHRAG_CONCURRENT_REQUESTS` | parallelism *within* one GraphRAG build (not across builds — see gotcha above) |
| `MAX_ITERS`, `MAX_SNIPPETS` | agentic loop iteration cap / evidence pool size |
| `WHOLE_FILE_MAX_CHARS` | truncation limit for whole-file chat mode (default 60000) |

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /api/sources` | list all loaded sources |
| `GET /api/sources/{id}?clean=bool` | fetch one source's raw or clean text |
| `POST /api/sources/youtube` (form: `url`, `languages`) | fetch a transcript, register as a new source |
| `POST /api/sources/upload` (multipart `file`) | register an uploaded `.txt` as a new source |
| `POST /api/sources/{id}/preprocess` | NDJSON stream of keep/drop progress; writes `clean.txt` |
| `POST /api/sources/{id}/ingest` | NDJSON stream; builds RAG + graph + GraphRAG for both scopes |
| `GET /api/graphrag/status?scope_id=...` | `idle` \| `building` \| `ready` \| `failed` |
| `GET /api/graph?scope=video\|global&source_id=...` | `{nodes, edges}` for the graph column |
| `GET /api/wiki/{video\|global}/{concept_id}?source_id=...` | wiki page markdown (generated lazily, cached) |
| `POST /api/chat` (`message, history, scope, mode, source_id, use_clean`) | streamed plain-text/markdown answer |

## Limitations

- GraphRAG builds are slow (many LLM calls) and run one at a time across the whole
  app (see gotcha above) — ingesting several sources back-to-back queues their builds.
- The global RAG index is fully rebuilt from every ingested source's `clean.txt` on
  every ingest (not incremental); fine at personal scale, would need revisiting for a
  large source count.
- Whole-file chat mode truncates anything over `WHOLE_FILE_MAX_CHARS`.
- The graph view's force layout is a small fixed-iteration simulation, not a real
  physics engine — fine for tens of nodes, will get cramped well before hundreds.
