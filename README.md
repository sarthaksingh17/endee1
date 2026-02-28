# CodeScope — Chat With Any Codebase

> Semantic search + RAG powered by [Endee](https://github.com/endee-io/endee), a high-performance vector database.

CodeScope lets you ingest any GitHub repository (or local codebase) and ask questions about it in plain English. It finds the most relevant code and documentation chunks using vector similarity search, then generates grounded answers with source references.

---

## Problem

Navigating an unfamiliar codebase is slow. `grep` requires knowing what to search for. GitHub search is keyword-only. Neither understands *meaning*.

CodeScope solves this with **semantic search** — it understands what you are asking and retrieves the code that answers it.

---

## How It Works

```
                  INGESTION (once per repo)

  GitHub URL / Local Files / ZIP / Docs URL
                    |
                    v
     Chunking (code-aware, ~500 tokens, overlap)
                    |
                    v
  Sentence-Transformers  -->  384-dim vectors
                    |
                    v
     Endee Vector DB (HNSW + filters + SIMD)


                  QUERY (every question)

  "How does authentication work?"
                    |
                    v
  Embed query  -->  kNN search in Endee
                    |
                    v
  Top-k chunks + question  -->  LLM
                    |
                    v
  Grounded answer with file references
```

---

## How Endee Is Used

Endee is the **core search engine** — not an optional add-on.

| Operation | Endee API | Purpose |
|---|---|---|
| Create index | `POST /api/v1/index/create` | One index per repo (dim=384, cosine) |
| Insert vectors | `POST /api/v1/index/{name}/vector/insert` | Store embedded chunks with metadata |
| kNN search | `POST /api/v1/index/{name}/search` | Retrieve most similar chunks for a query |
| Filter | `filter` param on search | Pre-filter by `source_type` (code vs docs) |
| List indexes | `GET /api/v1/index/list` | Show previously indexed repos |
| Delete index | `DELETE /api/v1/index/{name}/delete` | Clean up |

Each vector carries metadata (file path, source type, language, chunk text) and a filter object for pre-filtered search using Endee's RoaringBitmap-backed filtering engine.

Search responses are **MessagePack-encoded** — the client decodes them with `msgpack.unpackb()`.

---

## Tech Stack

| Component | Tool |
|---|---|
| Vector Database | **Endee** (self-hosted, HNSW + SIMD + quantization) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| LLM | Groq / OpenAI / Gemini / OpenRouter / Ollama (configurable) |
| Backend | Python |
| UI | Streamlit |
| HTTP + Serialization | `requests` + `msgpack` |

---

## Project Structure

```
endee/
  codescope/
    app.py              Streamlit UI
    endee_client.py     Endee HTTP/msgpack wrapper
    ingester.py         Clone, scrape, chunk, embed, insert
    searcher.py         Query embedding + Endee search
    llm.py              Multi-provider LLM abstraction
    requirements.txt    Python dependencies
    _smoke_test.py      Module smoke tests
    README.md           Developer guide
  docker-compose.yml    Run Endee
  src/                  Endee source (C++)
```

---

## Setup and Running

### Prerequisites

- Docker installed and running
- Python 3.10+
- An API key for your chosen LLM provider (or Ollama running locally)

### 1. Start Endee

```bash
docker-compose up -d
```

Verify:

```bash
curl http://localhost:8080/api/v1/health
```

### 2. Create a virtual environment and install dependencies

```bash
cd codescope
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

### 3. Set your LLM API key

Create a `codescope/.env` file (recommended):

```bash
GROQ_API_KEY=your-key-here
```

Or use environment variables:

```bash
# Pick one:
export GROQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
# Or use Ollama locally (no key needed)
```

### 4. Run the app

```bash
cd codescope
source venv/bin/activate   # if not already activated
streamlit run app.py
```

### 5. Verify installation (optional)

```bash
python3 codescope/_smoke_test.py
```

This runs a quick sanity check across all modules.

---

## Usage

1. Paste a GitHub URL (or upload files/ZIP) in the sidebar.
2. Optionally add a documentation URL to also index docs pages.
3. Click **Ingest** — the pipeline clones, chunks, embeds, and inserts into Endee.
4. Ask questions in the chat — answers cite specific files from the codebase.
5. Use the search scope toggle to search code only, docs only, or both.
6. Switch between previously indexed repos without re-ingesting.

---

## Switching LLM Providers

Select any provider from the sidebar dropdown. The system uses OpenAI-compatible chat completion endpoints. Adding a new provider is a one-line config change in `llm.py`.

Supported out of the box: **Groq**, **OpenAI**, **Google Gemini**, **OpenRouter**, **Ollama**.

---

## License

This project is built on top of the Endee vector database. See [LICENSE](LICENSE) for details.
