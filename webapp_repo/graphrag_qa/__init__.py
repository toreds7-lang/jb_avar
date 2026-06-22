"""graphrag_qa — in-process Microsoft GraphRAG Q&A over a per-paper index.

Public API:
    from graphrag_qa import GraphRAGQA, build_index, METHODS
    GraphRAGQA(root="data/<paper_id>/graphrag").ask("q", method="global").text
"""
from .qa import GraphRAGQA, Answer, METHODS
from .indexer import build_index
from .config import get_settings

__all__ = ["GraphRAGQA", "Answer", "METHODS", "build_index", "get_settings"]
__version__ = "0.2.0"
