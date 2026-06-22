"""Wiki page creation, update, index maintenance, link resolution.

Ported from paper_read_project_pdf/wiki.py, simplified: one scope (per-source or the
global cumulative scope) has one RagIndex and one graph, instead of juggling many papers'
indices/TOC summaries at once.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import llm_client
from config import DATA_DIR
from rag import RagIndex

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _wiki_dir(scope_id: str) -> Path:
    return DATA_DIR / scope_id / "wiki"


def ensure_wiki_dir(scope_id: str) -> None:
    _wiki_dir(scope_id).mkdir(parents=True, exist_ok=True)


def page_path(scope_id: str, concept_id: str) -> Path:
    return _wiki_dir(scope_id) / f"{concept_id}.md"


def list_pages(scope_id: str) -> list[dict[str, Any]]:
    ensure_wiki_dir(scope_id)
    pages = []
    for md_file in sorted(_wiki_dir(scope_id).glob("*.md")):
        if md_file.name in ("index.md", "log.md"):
            continue
        pages.append({
            "name": md_file.stem,
            "path": md_file.name,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(md_file.stat().st_mtime)),
        })
    return pages


def load_page(scope_id: str, concept_id: str) -> str | None:
    p = page_path(scope_id, concept_id)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def write_page(scope_id: str, concept_id: str, content: str) -> None:
    ensure_wiki_dir(scope_id)
    p = page_path(scope_id, concept_id)
    tmp_p = p.with_suffix(".md.tmp")
    tmp_p.write_text(content, encoding="utf-8")
    os.replace(tmp_p, p)


def rebuild_index(scope_id: str, graph: dict[str, Any]) -> None:
    ensure_wiki_dir(scope_id)
    rows = []
    for node in sorted(graph.get("nodes", []), key=lambda n: n.get("label", "")):
        label = node.get("label", "")
        node_type = node.get("type", "concept")
        sources = ", ".join(node.get("sources", []))
        summary = (node.get("summary", "") or "")[:80]
        wiki_page = node.get("wiki_page", f"{node.get('id')}.md")
        rows.append(f"| [{label}]({wiki_page}) | {node_type} | {sources} | {summary} |")

    content = (
        "# Wiki Index\n"
        + f"_Last updated: {time.strftime('%Y-%m-%dT%H:%M:%S')}_\n\n"
        + "## Concepts\n\n"
        + "| Concept | Type | Sources | Summary |\n"
        + "|---------|------|---------|----------|\n"
        + "\n".join(rows)
    )
    (_wiki_dir(scope_id) / "index.md").write_text(content, encoding="utf-8")


def resolve_wiki_links(markdown: str) -> str:
    def replace_link(match):
        concept_id = match.group(1)
        return f"[{concept_id}]({concept_id}.md)"
    return re.sub(r"\[\[([a-z0-9_]+)\]\]", replace_link, markdown, flags=re.IGNORECASE)


def _build_page_context(node: dict[str, Any], graph: dict[str, Any], rag_index: RagIndex) -> str:
    rag_parts = []
    try:
        results = rag_index.topk(node["label"], k=4)
        for r in results:
            rag_parts.append(f"- {r['text'][:300]}...")
    except Exception:
        pass

    neighbor_parts = []
    for edge in graph.get("edges", []):
        if edge["source"] == node["id"]:
            neighbor_id = edge["target"]
        elif edge["target"] == node["id"]:
            neighbor_id = edge["source"]
        else:
            continue
        neighbor = next((n for n in graph["nodes"] if n["id"] == neighbor_id), None)
        if neighbor:
            relation = edge.get("relation", "related")
            neighbor_parts.append(f"- [[{neighbor_id}]] ({relation}): {neighbor.get('summary', '')}")

    context = ""
    if rag_parts:
        context += "=== Excerpts ===\n" + "\n".join(rag_parts) + "\n\n"
    if neighbor_parts:
        context += "=== Related Concepts ===\n" + "\n".join(neighbor_parts) + "\n"
    return context


def generate_page(node: dict[str, Any], graph: dict[str, Any], rag_index: RagIndex) -> str:
    system_path = _PROMPTS_DIR / "wiki_page.system.txt"
    system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else ""
    user_path = _PROMPTS_DIR / "wiki_page.user.txt"
    user_template = user_path.read_text(encoding="utf-8") if user_path.exists() else ""

    context = _build_page_context(node, graph, rag_index)
    system_prompt = system_prompt.format(
        label=node["label"], type=node["type"],
        sources=", ".join(node["sources"]), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    user_prompt = user_template.format(
        label=node["label"], concept_id=node["id"], type=node["type"],
        sources=", ".join(node["sources"]), context=context,
    )
    try:
        response = llm_client.chat(system_prompt, user_prompt)
        llm_client.llm_sleep()
        return response.strip()
    except Exception as e:
        return f"# {node['label']}\n\n_Error generating page: {e}_"


def update_pages_for_scope(scope_id: str, affected_node_ids: list[str], graph: dict[str, Any], rag_index: RagIndex) -> Iterator[dict[str, Any]]:
    pages_written = 0
    for concept_id in affected_node_ids:
        node = next((n for n in graph.get("nodes", []) if n["id"] == concept_id), None)
        if not node:
            continue
        yield {"stage": "page", "concept_id": concept_id, "msg": f"Generating wiki page for {node['label']}..."}
        content = generate_page(node, graph, rag_index)
        write_page(scope_id, concept_id, content)
        pages_written += 1
        yield {"stage": "page_done", "concept_id": concept_id}
    yield {"done": True, "pages_written": pages_written}


def regenerate_page(scope_id: str, concept_id: str, graph: dict[str, Any], rag_index: RagIndex) -> str:
    node = next((n for n in graph.get("nodes", []) if n["id"] == concept_id), None)
    if not node:
        return ""
    content = generate_page(node, graph, rag_index)
    write_page(scope_id, concept_id, content)
    return content
