"""
Endee Vector Database Client
Handles all HTTP communication and msgpack deserialization for the Endee REST API.
"""

import json
import logging
from typing import Any

import msgpack
import requests

logger = logging.getLogger(__name__)


class EndeeClient:
    """Thin wrapper around the Endee vector database REST API."""

    def __init__(self, base_url: str = "http://localhost:8080/api/v1", auth_token: str = ""):
        self.base = base_url.rstrip("/")
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> dict:
        r = requests.get(f"{self.base}/health", headers=self.headers)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def create_index(
        self,
        name: str,
        dim: int,
        space: str = "cosine",
        precision: str = "float32",
        m: int = 16,
        ef_con: int = 200,
    ) -> str:
        r = requests.post(
            f"{self.base}/index/create",
            json={
                "index_name": name,
                "dim": dim,
                "space_type": space,
                "precision": precision,
                "M": m,
                "ef_con": ef_con,
            },
            headers=self.headers,
        )
        return r.text

    def delete_index(self, index_name: str) -> str:
        r = requests.delete(
            f"{self.base}/index/{index_name}/delete",
            headers=self.headers,
        )
        return r.text

    def list_indexes(self) -> list[dict]:
        r = requests.get(f"{self.base}/index/list", headers=self.headers)
        r.raise_for_status()
        data = r.json()
        # API returns {"indexes": [...]}
        if isinstance(data, dict) and "indexes" in data:
            return data["indexes"]
        if isinstance(data, list):
            return data
        return []

    def index_exists(self, index_name: str) -> bool:
        try:
            indexes = self.list_indexes()
            return any(index_name == idx.get("name", "") for idx in indexes)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------
    def insert(self, index_name: str, vectors: list[dict]) -> int:
        """Insert vectors into an index.

        Each vector dict should have:
            id:     str
            vector: list[float]
            meta:   str  (JSON-encoded metadata string)
            filter: str  (JSON-encoded filter string, optional)
        """
        r = requests.post(
            f"{self.base}/index/{index_name}/vector/insert",
            json=vectors,
            headers=self.headers,
        )
        return r.status_code

    def search(
        self,
        index_name: str,
        vector: list[float],
        k: int = 5,
        filters: list[dict] | None = None,
        ef: int = 0,
        include_vectors: bool = False,
    ) -> Any:
        payload: dict[str, Any] = {"vector": vector, "k": k}
        if filters:
            payload["filter"] = json.dumps(filters)
        if ef:
            payload["ef"] = ef
        if include_vectors:
            payload["include_vectors"] = True

        r = requests.post(
            f"{self.base}/index/{index_name}/search",
            json=payload,
            headers=self.headers,
        )
        r.raise_for_status()
        return msgpack.unpackb(r.content, raw=False)

    def get_vector(self, index_name: str, vector_id: str) -> Any:
        r = requests.post(
            f"{self.base}/index/{index_name}/vector/get",
            json={"id": vector_id},
            headers=self.headers,
        )
        r.raise_for_status()
        return msgpack.unpackb(r.content, raw=False)
