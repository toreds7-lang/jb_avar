"""LLM-driven concept/relationship extraction and graph CRUD.

Ported from paper_read_project_pdf/graph.py. Adapted: a "paper" there is a "scope" here
(a per-source id, or config.GLOBAL_SCOPE_ID for the cumulative graph), the node/edge
provenance list is "sources" instead of "papers", and extraction takes plain text
directly instead of a list of {"markdown": ...} pages.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import llm_client
from config import DATA_DIR

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MAX_EXTRACT_CHARS = 20000


def _graph_path(scope_id: str) -> Path:
    return DATA_DIR / scope_id / "graph.json"


def load_graph(scope_id: str) -> dict[str, Any]:
    graph_path = _graph_path(scope_id)
    if graph_path.exists():
        return json.loads(graph_path.read_text(encoding="utf-8"))
    return _empty_graph()


def save_graph(graph: dict[str, Any], scope_id: str) -> None:
    graph_path = _graph_path(scope_id)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = graph_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, graph_path)


def _empty_graph() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nodes": [],
        "edges": [],
    }


def _canonical_id(label: str) -> str:
    s = label.lower().strip()
    s = re.sub(r"['\"]", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def _find_existing_node(graph: dict, candidate_id: str, candidate_aliases: list[str]) -> str | None:
    candidate_aliases = candidate_aliases or []
    for node in graph.get("nodes", []):
        if node["id"] == candidate_id:
            return node["id"]
        node_aliases = [a.lower() for a in (node.get("aliases") or [])]
        candidate_lower = [a.lower() for a in candidate_aliases]
        if set(node_aliases) & set(candidate_lower):
            return node["id"]
    return None


def _build_extract_prompt(source_label: str, text: str) -> tuple[str, str]:
    system_path = _PROMPTS_DIR / "graph_extract.system.txt"
    system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else ""

    user_path = _PROMPTS_DIR / "graph_extract.user.txt"
    user_template = user_path.read_text(encoding="utf-8") if user_path.exists() else ""

    text = text[: int(_MAX_EXTRACT_CHARS * 1.5)]
    user_prompt = user_template.format(source_label=source_label, text=text)
    return system_prompt, user_prompt


def _parse_extraction(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_concepts(source_id: str, text: str, source_label: str | None = None) -> dict[str, Any]:
    """Call the LLM to extract concepts and relationships from a source's cleaned text.
    Returns {"nodes": [...], "edges": [...]}."""
    system_prompt, user_prompt = _build_extract_prompt(source_label or source_id, text)
    try:
        response = llm_client.chat(system_prompt, user_prompt)
        result = _parse_extraction(response)
        if result:
            return result
    except Exception as e:
        print(f"[graph] extraction LLM error: {e}", file=sys.stderr)
    return {"nodes": [], "edges": []}


def merge_extraction_into_graph(graph: dict[str, Any], extraction: dict[str, Any], source_id: str) -> dict[str, str]:
    """Merge extracted nodes/edges into the graph in-place. Returns id_map
    {extracted_label -> resolved_canonical_id}. Calling this with the same source_id
    against both the per-source graph and the global graph is how cumulative scope works."""
    id_map: dict[str, str] = {}
    extracted_nodes = extraction.get("nodes", [])
    extracted_edges = extraction.get("edges", [])

    for extracted_node in extracted_nodes:
        label = (extracted_node.get("label") or "").strip()
        if not label:
            continue
        node_type = extracted_node.get("type", "concept")
        node_summary = extracted_node.get("summary", "")
        node_aliases = extracted_node.get("aliases", []) or []

        canonical_id = _canonical_id(label)
        existing_id = _find_existing_node(graph, canonical_id, node_aliases)

        if existing_id:
            id_map[label] = existing_id
            node = next((n for n in graph["nodes"] if n["id"] == existing_id), None)
            if node:
                if source_id not in node.get("sources", []):
                    node["sources"].append(source_id)
                for alias in node_aliases:
                    if alias not in node.get("aliases", []):
                        node["aliases"].append(alias)
        else:
            graph["nodes"].append({
                "id": canonical_id,
                "label": label,
                "type": node_type,
                "aliases": node_aliases,
                "sources": [source_id],
                "summary": node_summary,
                "wiki_page": f"{canonical_id}.md",
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            id_map[label] = canonical_id

    for extracted_edge in extracted_edges:
        source_label = (extracted_edge.get("source_label") or "").strip()
        target_label = (extracted_edge.get("target_label") or "").strip()
        relation = (extracted_edge.get("relation") or "").strip() or "related"
        if not (source_label and target_label):
            continue
        src_id = id_map.get(source_label)
        tgt_id = id_map.get(target_label)
        if not (src_id and tgt_id):
            continue
        existing_edge = next(
            (e for e in graph["edges"] if e["source"] == src_id and e["target"] == tgt_id and e["relation"] == relation),
            None,
        )
        if existing_edge:
            existing_edge["weight"] = existing_edge.get("weight", 1) + 1
            if source_id not in existing_edge.get("sources", []):
                existing_edge["sources"].append(source_id)
        else:
            graph["edges"].append({
                "source": src_id,
                "target": tgt_id,
                "relation": relation,
                "weight": 1,
                "sources": [source_id],
            })

    return id_map


def ingest_into_scope(scope_id: str, source_id: str, text: str, source_label: str | None = None) -> Iterator[dict[str, Any]]:
    """Full pipeline against one graph scope: extract -> merge -> save. Yields NDJSON progress."""
    yield {"stage": "extracting", "msg": "Calling LLM for concept extraction..."}
    extraction = extract_concepts(source_id, text, source_label)
    yield {"stage": "extracted", "nodes": len(extraction.get("nodes", [])), "edges": len(extraction.get("edges", []))}

    yield {"stage": "merging", "msg": "Merging into graph..."}
    graph = load_graph(scope_id)
    old_count = len(graph["nodes"])
    id_map = merge_extraction_into_graph(graph, extraction, source_id)
    new_count = len(graph["nodes"])

    yield {"stage": "merged", "new_nodes": new_count - old_count, "total_nodes": new_count}

    graph["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_graph(graph, scope_id)
    yield {"done": True, "ok": True, "scope_id": scope_id, "new_nodes": new_count - old_count,
           "total_nodes": new_count, "id_map": id_map}


def get_graph(scope_id: str) -> dict[str, Any]:
    return load_graph(scope_id)
