# CodeScope — Developer Guide

Internal notes for contributors working inside `codescope/`.

## File overview

| File | Purpose |
|---|---|
| `endee_client.py` | HTTP + msgpack wrapper for the Endee vector DB REST API |
| `ingester.py` | Clone repos, scrape docs, chunk text, embed, insert into Endee |
| `searcher.py` | Embed a query, run kNN search on Endee, return ranked context |
| `llm.py` | Multi-provider LLM abstraction (Groq, OpenAI, Gemini, OpenRouter, Ollama) |
| `app.py` | Streamlit UI — ingestion sidebar + chat interface |
| `requirements.txt` | Python dependencies |
| `_smoke_test.py` | Module-level smoke tests (run with `python3 _smoke_test.py`) |

## Quick start

```bash
# From the repo root — start the vector database
docker-compose up -d

# Create venv and install deps
cd codescope
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | If using Groq | Groq API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `GEMINI_API_KEY` | If using Gemini | Google Gemini API key |
| `OPENROUTER_API_KEY` | If using OpenRouter | OpenRouter API key |
| `ENDEE_URL` | No | Override Endee base URL (default: `http://localhost:8080/api/v1`) |

## Adding a new LLM provider

Edit `llm.py` and add an entry to the `PROVIDERS` dict:

```python
PROVIDERS["my_provider"] = {
    "base_url": "https://api.example.com/v1/chat/completions",
    "env_key": "MY_PROVIDER_API_KEY",
    "default_model": "model-name",
}
```

Any provider that exposes an OpenAI-compatible `/chat/completions` endpoint works out of the box.
