"""
Search & retrieval module for CodeScope.

Takes a natural-language query, embeds it, searches Endee, and returns
ranked context chunks ready for the LLM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from endee_client import EndeeClient
from ingester import embed_texts

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieved chunk with its similarity score."""
    text: str = ""
    file_path: str = ""
    source_type: str = ""
    repo_name: str = ""
    language: str = ""
    chunk_index: int = 0
    score: float = 0.0

    @property
    def metadata(self) -> dict:
        return {
            "text": self.text,
            "file_path": self.file_path,
            "source_type": self.source_type,
            "repo_name": self.repo_name,
            "language": self.language,
            "chunk_index": self.chunk_index,
        }


def search(
    query: str,
    index_name: str,
    *,
    k: int = 8,
    source_filter: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    endee_url: str = "http://localhost:8080/api/v1",
) -> list[SearchResult]:
    """Embed a query and run kNN search against the Endee index.

    Args:
        query:           Natural-language question.
        index_name:      Endee index to search.
        k:               Number of results to return.
        source_filter:   Optional — "code", "docs", or None (both).
        embedding_model: sentence-transformers model name.
        endee_url:       Endee base API URL.

    Returns:
        Ranked list of SearchResult objects.
    """
    client = EndeeClient(base_url=endee_url)

    # Embed the query
    vec = embed_texts([query], model_name=embedding_model)[0].tolist()

    # Build optional filter
    filters = None
    if source_filter and source_filter in ("code", "docs"):
        filters = [{"source_type": source_filter}]

    # Search Endee
    raw = client.search(index_name, vec, k=k, filters=filters)

    # Parse response — Endee returns a list of hits.
    # Each hit is a list: [score, id, meta_bytes, filter_str, ?, ?]
    results: list[SearchResult] = []
    hits = raw if isinstance(raw, list) else []

    for hit in hits:
        # Support both list-based (msgpack) and dict-based formats
        if isinstance(hit, list) and len(hit) >= 3:
            score = hit[0] if isinstance(hit[0], (int, float)) else 0.0
            meta_raw = hit[2]
        elif isinstance(hit, dict):
            score = hit.get("distance", hit.get("score", 0.0))
            meta_raw = hit.get("meta", hit.get("metadata", b""))
        else:
            continue

        # Parse metadata
        if isinstance(meta_raw, (bytes, bytearray)):
            try:
                meta = json.loads(meta_raw.decode("utf-8"))
            except Exception:
                meta = {}
        elif isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = {}

        results.append(
            SearchResult(
                text=meta.get("text", ""),
                file_path=meta.get("file_path", ""),
                source_type=meta.get("source_type", ""),
                repo_name=meta.get("repo_name", ""),
                language=meta.get("language", ""),
                chunk_index=meta.get("chunk_index", 0),
                score=score,
            )
        )

    return results


def retrieve_context(
    query: str,
    index_name: str,
    *,
    k: int = 8,
    source_filter: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    endee_url: str = "http://localhost:8080/api/v1",
) -> list[dict]:
    """Convenience wrapper that returns context dicts suitable for llm.ask_llm()."""
    results = search(
        query,
        index_name,
        k=k,
        source_filter=source_filter,
        embedding_model=embedding_model,
        endee_url=endee_url,
    )
    return [{"metadata": r.metadata, "score": r.score} for r in results]
