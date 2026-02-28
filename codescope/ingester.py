"""
Ingestion pipeline for CodeScope.

Handles:
  - Cloning GitHub repos
  - Scraping documentation URLs
  - Reading local files / ZIP uploads
  - Chunking text (code-aware + docs)
  - Embedding with sentence-transformers
  - Inserting vectors + metadata into Endee
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from endee_client import EndeeClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

CHUNK_SIZE = 500  # approx tokens (~4 chars/token → ~2000 chars)
CHUNK_OVERLAP = 50  # token overlap
CHAR_CHUNK = CHUNK_SIZE * 4
CHAR_OVERLAP = CHUNK_OVERLAP * 4

CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".html", ".css", ".scss", ".less", ".vue", ".svelte",
    ".sql", ".graphql", ".proto",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".json", ".xml",
    ".dockerfile",
}

DOC_EXTENSIONS: set[str] = {".md", ".txt", ".rst", ".adoc", ".org"}

SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "vendor", ".tox",
    ".mypy_cache", ".pytest_cache", ".eggs", "egg-info",
    "target", "out", "bin", "obj",
}

SKIP_FILES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Gemfile.lock",
}

MAX_FILE_SIZE = 512_000  # 500 KB — skip huge generated files

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".rb": "ruby", ".c": "c", ".cpp": "cpp",
    ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "shell",
    ".html": "html", ".css": "css", ".sql": "sql",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".xml": "xml", ".md": "markdown",
    ".txt": "text", ".rst": "rst",
}

INSERT_BATCH_SIZE = 64

# ---------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if name not in _model_cache:
        logger.info("Loading embedding model: %s", name)
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = get_embedding_model(model_name)
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ---------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------


def clone_repo(url: str, dest: str | None = None) -> Path:
    """Clone a git repo to a temp dir and return the path."""
    import git  # gitpython

    dest = dest or tempfile.mkdtemp(prefix="codescope_")
    logger.info("Cloning %s → %s", url, dest)
    git.Repo.clone_from(url, dest, depth=1)
    return Path(dest)


def read_local_files(root: Path) -> Generator[tuple[str, str, str], None, None]:
    """Yield (relative_path, content, source_type) for every eligible file."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            continue

        ext = path.suffix.lower()
        if ext == "" and path.name.lower() in ("dockerfile", "makefile", "rakefile", "gemfile"):
            ext = ".dockerfile"

        if ext not in CODE_EXTENSIONS and ext not in DOC_EXTENSIONS:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        source_type = "docs" if ext in DOC_EXTENSIONS else "code"
        rel = str(path.relative_to(root))
        yield rel, content, source_type


def extract_zip(data: bytes, dest: str | None = None) -> Path:
    """Extract a zip archive to a temp dir."""
    dest_path = Path(dest or tempfile.mkdtemp(prefix="codescope_zip_"))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_path)
    return dest_path


def scrape_docs_url(url: str, max_pages: int = 50) -> list[tuple[str, str]]:
    """Crawl a docs site and return list of (page_url, text_content).

    Stays within the same domain/path prefix. Follows <a> links on each page.
    """
    parsed = urlparse(url)
    base_domain = parsed.netloc
    prefix = parsed.path.rstrip("/")

    visited: set[str] = set()
    pages: list[tuple[str, str]] = []
    queue: list[str] = [url]

    while queue and len(pages) < max_pages:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        try:
            resp = requests.get(current, timeout=15, headers={"User-Agent": "CodeScope/1.0"})
            if resp.status_code != 200:
                continue
            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav / header / footer / script / style
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 100:  # skip near-empty pages
            pages.append((current, text))

        # Follow links
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].split("?")[0]
            if not href:
                continue
            full = urljoin(current, href)
            fp = urlparse(full)
            if fp.netloc == base_domain and fp.path.startswith(prefix):
                if full not in visited:
                    queue.append(full)

    return pages


# ---------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------

# Regex for common code boundaries (function / class definitions)
_CODE_BOUNDARY = re.compile(
    r"^(?:def |class |function |const |let |var |export |public |private |"
    r"protected |async |func |fn |impl |struct |enum |interface |type |"
    r"module |package |import |from |#include |using |namespace )",
    re.MULTILINE,
)


def chunk_text(text: str, source_type: str = "code") -> list[str]:
    """Split text into overlapping chunks. Code-aware when source_type == 'code'."""
    if not text.strip():
        return []

    if source_type == "code":
        return _chunk_code(text)
    return _chunk_plain(text)


def _chunk_plain(text: str) -> list[str]:
    """Character-level chunking with overlap for docs / plain text."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHAR_CHUNK
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += CHAR_CHUNK - CHAR_OVERLAP
    return chunks


def _chunk_code(text: str) -> list[str]:
    """Try to split on function/class boundaries, fall back to plain chunking."""
    boundaries = [m.start() for m in _CODE_BOUNDARY.finditer(text)]
    if len(boundaries) < 2:
        return _chunk_plain(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    lines = text.split("\n")
    line_idx = 0

    for line in lines:
        line_idx += 1
        current.append(line)
        current_len += len(line) + 1

        if current_len >= CHAR_CHUNK:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            # Keep overlap
            overlap_chars = 0
            overlap_start = len(current)
            for j in range(len(current) - 1, -1, -1):
                overlap_chars += len(current[j]) + 1
                if overlap_chars >= CHAR_OVERLAP:
                    overlap_start = j
                    break
            current = current[overlap_start:]
            current_len = sum(len(l) + 1 for l in current)

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------
# Unique chunk ID
# ---------------------------------------------------------------


def _chunk_id(repo_name: str, file_path: str, chunk_index: int) -> str:
    raw = f"{repo_name}:{file_path}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------


class IngestionProgress:
    """Mutable progress tracker for UI callbacks."""

    def __init__(self):
        self.stage: str = "idle"
        self.detail: str = ""
        self.current: int = 0
        self.total: int = 0
        self.done: bool = False


def ingest(
    *,
    repo_url: str | None = None,
    docs_url: str | None = None,
    local_path: str | None = None,
    zip_data: bytes | None = None,
    index_name: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    endee_url: str = "http://localhost:8080/api/v1",
    progress: IngestionProgress | None = None,
) -> str:
    """Run the full ingestion pipeline. Returns the Endee index name used."""

    prog = progress or IngestionProgress()
    client = EndeeClient(base_url=endee_url)

    # ---- 1. Gather sources ----
    prog.stage = "sources"
    prog.detail = "Gathering source files..."

    file_items: list[tuple[str, str, str]] = []  # (path, content, source_type)
    repo_name = "local"
    tmp_dirs: list[str] = []

    if repo_url:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        prog.detail = f"Cloning {repo_name}..."
        repo_dir = clone_repo(repo_url)
        tmp_dirs.append(str(repo_dir))
        for item in read_local_files(repo_dir):
            file_items.append(item)

    if local_path:
        local_root = Path(local_path)
        if local_root.is_file():
            try:
                content = local_root.read_text(encoding="utf-8", errors="ignore")
                ext = local_root.suffix.lower()
                st = "docs" if ext in DOC_EXTENSIONS else "code"
                file_items.append((local_root.name, content, st))
            except Exception:
                pass
        elif local_root.is_dir():
            repo_name = local_root.name
            for item in read_local_files(local_root):
                file_items.append(item)

    if zip_data:
        prog.detail = "Extracting ZIP..."
        zip_dir = extract_zip(zip_data)
        tmp_dirs.append(str(zip_dir))
        # Use the top-level folder name if there is exactly one
        children = list(zip_dir.iterdir())
        root = children[0] if len(children) == 1 and children[0].is_dir() else zip_dir
        if repo_name == "local":
            repo_name = root.name
        for item in read_local_files(root):
            file_items.append(item)

    if docs_url:
        prog.detail = f"Scraping docs from {docs_url}..."
        doc_pages = scrape_docs_url(docs_url)
        for page_url, text in doc_pages:
            file_items.append((page_url, text, "docs"))

    if not file_items:
        prog.done = True
        raise ValueError("No files found to ingest.")

    # ---- 2. Chunk ----
    prog.stage = "chunking"
    prog.detail = f"Chunking {len(file_items)} files..."

    chunks_data: list[dict] = []
    for file_path, content, source_type in file_items:
        ext = Path(file_path).suffix.lower() if not file_path.startswith("http") else ""
        language = LANGUAGE_MAP.get(ext, "text")
        pieces = chunk_text(content, source_type)
        for idx, piece in enumerate(pieces):
            chunks_data.append(
                {
                    "text": piece,
                    "source_type": source_type,
                    "file_path": file_path,
                    "repo_name": repo_name,
                    "language": language,
                    "chunk_index": idx,
                }
            )

    prog.total = len(chunks_data)
    logger.info("Total chunks: %d from %d files", len(chunks_data), len(file_items))

    # ---- 3. Embed ----
    prog.stage = "embedding"
    prog.detail = f"Embedding {len(chunks_data)} chunks..."

    texts = [c["text"] for c in chunks_data]
    embeddings = embed_texts(texts, model_name=embedding_model)
    dim = embeddings.shape[1]

    # ---- 4. Create Endee index ----
    prog.stage = "indexing"
    idx_name = index_name or f"codescope-{repo_name}"
    idx_name = re.sub(r"[^a-zA-Z0-9_-]", "-", idx_name).lower()

    if not client.index_exists(idx_name):
        prog.detail = f"Creating index '{idx_name}' (dim={dim})..."
        client.create_index(idx_name, dim=dim, space="cosine", precision="float32")
    else:
        prog.detail = f"Index '{idx_name}' already exists, appending..."

    # ---- 5. Insert in batches ----
    prog.stage = "inserting"
    total = len(chunks_data)

    for batch_start in range(0, total, INSERT_BATCH_SIZE):
        batch_end = min(batch_start + INSERT_BATCH_SIZE, total)
        batch_vectors: list[dict] = []

        for i in range(batch_start, batch_end):
            c = chunks_data[i]
            vec = embeddings[i].tolist()
            cid = _chunk_id(repo_name, c["file_path"], c["chunk_index"])

            # Endee expects meta as a JSON string and filter as a JSON string
            meta = json.dumps({
                "text": c["text"],
                "file_path": c["file_path"],
                "source_type": c["source_type"],
                "repo_name": c["repo_name"],
                "language": c["language"],
                "chunk_index": c["chunk_index"],
            })

            filter_obj = {
                "source_type": c["source_type"],
                "language": c["language"],
            }

            batch_vectors.append(
                {
                    "id": cid,
                    "vector": vec,
                    "meta": meta,
                    "filter": json.dumps(filter_obj),
                }
            )

        status = client.insert(idx_name, batch_vectors)
        prog.current = batch_end
        prog.detail = f"Inserted {batch_end}/{total} chunks..."

        if status not in (200, 201):
            logger.warning("Insert batch returned status %d", status)

    # ---- Cleanup ----
    for d in tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    prog.stage = "done"
    prog.detail = f"Indexed {total} chunks into '{idx_name}'"
    prog.done = True
    logger.info("Ingestion complete: %s (%d chunks)", idx_name, total)
    return idx_name
