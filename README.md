# CodeSource — Chat With Any Codebase

> **Semantic search + RAG over any GitHub repository, powered by [Endee](https://github.com/endee-db/endee) — a high-performance vector database.**

Stop grep-ing. Stop guessing. Ask your codebase a question and get a grounded answer with file-level citations in seconds.

---

## The Problem

Every developer has been here: dropped into an unfamiliar codebase with a deadline. `grep` only works if you already know what to look for. GitHub search is keyword-only. Neither understands *meaning*.

**CodeSource solves this.** It semantically understands what you're asking and surfaces the exact code that answers it — whether that's an auth flow buried in a middleware file, or a config option mentioned only in a comment.

---



## How It Works

### Ingestion Pipeline *(runs once per repo)*

```
  GitHub URL / Local Files / ZIP / Docs URL
                    │
                    ▼
     Code-Aware Chunking  (~500 tokens, with overlap)
                    │
                    ▼
  sentence-transformers  ──►  384-dim embeddings
                    │
                    ▼
     Endee Vector DB  (HNSW + SIMD + RoaringBitmap filters)
```

### Query Pipeline *(every question)*

```
  "How does authentication work?"
                    │
                    ▼
     Embed query  ──►  kNN search in Endee
                    │
                    ▼
  Top-k chunks  +  question  ──►  LLM
                    │
                    ▼
  Grounded answer with file:line references
```

---

## Why Endee?

Endee is the **core search engine** — not a plugin, not optional. It handles all vector storage, retrieval, and filtering.

| Operation | Endee API | Purpose |
|---|---|---|
| Create index | `POST /api/v1/index/create` | One index per repo (`dim=384`, cosine similarity) |
| Insert vectors | `POST /api/v1/index/{name}/vector/insert` | Store embedded chunks with rich metadata |
| kNN search | `POST /api/v1/index/{name}/search` | Retrieve most semantically similar chunks |
| Filtered search | `filter` param on search | Pre-filter by `source_type` (code vs. docs) |
| List indexes | `GET /api/v1/index/list` | Show and switch between previously indexed repos |
| Delete index | `DELETE /api/v1/index/{name}/delete` | Clean up stale indexes |

Each vector stores metadata — `file_path`, `source_type`, `language`, `chunk_text` — and a `filter` object enabling Endee's **RoaringBitmap-backed pre-filtering** for fast, precise retrieval.

Search responses are **MessagePack-encoded** and decoded client-side with `msgpack.unpackb()`, keeping serialization overhead minimal.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | **Endee** (self-hosted · HNSW + SIMD + quantization) |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` · 384 dims |
| LLM | Groq / OpenAI / Gemini / OpenRouter / Ollama *(switchable)* |
| Backend | Python 3.10+ |
| UI | Streamlit |
| Serialization | `requests` + `msgpack` |

---

## Project Structure

```
codesource/
├── app.py              # Streamlit UI — chat interface & sidebar controls
├── endee_client.py     # Endee HTTP/msgpack wrapper
├── ingester.py         # Clone → chunk → embed → insert pipeline
├── searcher.py         # Query embedding + Endee kNN search
├── llm.py              # Multi-provider LLM abstraction (add a provider in 1 line)
├── requirements.txt    # Python dependencies
└── _smoke_test.py      # Module-level smoke tests

docker-compose.yml      # Spin up Endee
src/                    # Endee source (C++)
```

---

## Quickstart

### Prerequisites

- Docker (running)
- Python 3.10+
- API key for your chosen LLM provider — *or* Ollama running locally

---

### 1. Start Endee

```bash
docker-compose up -d
```

Verify it's healthy:

```bash
curl http://localhost:8080/api/v1/health
```

---

### 2. Install Python dependencies

```bash
cd codesource
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

---

### 3. Configure your LLM

Create `codesource/.env`:

```env
# Pick one — or use Ollama locally (no key needed)
GROQ_API_KEY=your-key-here
# OPENAI_API_KEY=your-key-here
# GEMINI_API_KEY=your-key-here
# OPENROUTER_API_KEY=your-key-here
```

---

### 4. Run

```bash
cd codesource
source venv/bin/activate
streamlit run app.py
```

---

### 5. Verify (optional)

```bash
python3 _smoke_test.py
```

Runs a quick sanity check across all modules.

---

## Usage

1. **Paste a GitHub URL** (or upload files / a ZIP) in the sidebar
2. **Optionally add a docs URL** to also index documentation pages
3. **Click Ingest** — CodeSource clones, chunks, embeds, and inserts into Endee
4. **Ask questions** in the chat — answers cite specific files from the repo
5. **Toggle search scope** — code only, docs only, or both
6. **Switch repos** — previously indexed repos are available without re-ingesting

---

## Switching LLM Providers

Select any provider from the sidebar dropdown. All providers use OpenAI-compatible chat completion endpoints. Adding a new provider is a **one-line config change** in `llm.py`.

**Supported out of the box:** Groq · OpenAI · Google Gemini · OpenRouter · Ollama

---

## Architecture Notes

- **Chunking is code-aware**: chunks respect function/class boundaries where possible and use overlapping windows to avoid cutting context mid-thought.
- **Filters are first-class**: Endee's RoaringBitmap filtering lets you scope searches to code or documentation *before* the vector scan — not as a post-filter. This keeps precision high on mixed-content repos.
- **LLM is grounded**: the system prompt explicitly instructs the LLM to answer *only* from retrieved chunks and to cite file paths. Hallucination is structurally discouraged.
- **One index per repo**: indexes are named, persistent, and listable. Re-ingestion is only needed if the repo changes.

---

## License

Built on top of the [Endee](https://github.com/endee-db/endee) vector database. See `LICENSE` for details.
