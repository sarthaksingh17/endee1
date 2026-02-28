"""
Multi-provider LLM abstraction.

Supports: Groq, OpenAI, Google Gemini, OpenRouter, Ollama (local).
Swap providers by changing the provider name — no code changes needed.
"""

import json
import logging
import os
from dataclasses import dataclass

import requests as httpx  # alias to avoid clash with future async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1/chat/completions",
        "env_key": "",
        "default_model": "llama3",
    },
}

SYSTEM_PROMPT = (
    "You are CodeScope, a precise code assistant. "
    "ALWAYS respond in English, regardless of what language appears in the code or comments. "
    "Answer the user's question using ONLY the provided context chunks. "
    "Each chunk is labelled with a source number in square brackets like [1], [2], etc. "
    "Multiple chunks may share the same source number because they come from the same file. "
    "When you reference information from a chunk, add its SOURCE number as a Unicode superscript (¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹). "
    "ONLY use source numbers that actually appear in the context — do NOT invent higher numbers. "
    "Do NOT write file paths or URLs inline — just the superscript number. "
    "Example: 'FastAPI uses Starlette under the hood¹ and Pydantic for validation³.' "
    "If the context doesn't contain enough information, say so clearly."
)


@dataclass
class LLMConfig:
    provider: str = "groq"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048


def _resolve_config(cfg: LLMConfig) -> tuple[str, dict[str, str], str]:
    """Return (url, headers, model) for the configured provider."""
    prov = PROVIDERS.get(cfg.provider)
    if prov is None:
        raise ValueError(f"Unknown LLM provider: {cfg.provider}. Choose from: {list(PROVIDERS)}")

    model = cfg.model or prov["default_model"]
    api_key = cfg.api_key or os.environ.get(prov["env_key"], "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return prov["base_url"], headers, model


def build_prompt(
    question: str,
    context_chunks: list[dict],
    file_map: dict[str, int] | None = None,
) -> list[dict]:
    """Assemble the chat messages from retrieved context and a user question.

    If *file_map* is provided it maps file_path → 1-based source number so
    that multiple chunks from the same file share the same citation number.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", chunk)
        source = meta.get("file_path", meta.get("source", "unknown"))
        text = meta.get("text", "")
        num = file_map.get(source, i) if file_map else i
        context_parts.append(f"[{num}] {source}\n{text}")

    context_block = "\n\n---\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer (use superscript numbers like ¹ ² ³ to cite sources):"
            ),
        },
    ]


def ask_llm(
    question: str,
    context_chunks: list[dict],
    config: LLMConfig | None = None,
    file_map: dict[str, int] | None = None,
) -> str:
    """Send a RAG query to the configured LLM and return the answer text."""
    cfg = config or LLMConfig()
    url, headers, model = _resolve_config(cfg)
    messages = build_prompt(question, context_chunks, file_map=file_map)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except httpx.exceptions.HTTPError as exc:
        logger.error("LLM request failed: %s — %s", exc, getattr(exc.response, "text", ""))
        return f"Error communicating with LLM ({cfg.provider}): {exc}"
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.error("Unexpected LLM response: %s", exc)
        return f"Unexpected response from LLM: {exc}"
