#!/usr/bin/env python3
"""Smoke test — verifies every module imports and key functionality works."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []

# ---------- endee_client ----------
print("--- endee_client ---")
try:
    from endee_client import EndeeClient
    c = EndeeClient()
    methods = ["health", "create_index", "delete_index", "list_indexes",
               "index_exists", "insert", "search", "get_vector"]
    for m in methods:
        assert hasattr(c, m), f"Missing method: {m}"
    print("  EndeeClient: all 8 methods present")
except Exception as e:
    errors.append(f"endee_client: {e}")
    print(f"  FAIL: {e}")

# ---------- llm ----------
print("--- llm ---")
try:
    from llm import PROVIDERS, LLMConfig, ask_llm, build_prompt, SYSTEM_PROMPT
    assert len(PROVIDERS) == 5, f"Expected 5 providers, got {len(PROVIDERS)}"
    for p in ["groq", "openai", "gemini", "openrouter", "ollama"]:
        assert p in PROVIDERS, f"Missing provider: {p}"
        for k in ["base_url", "env_key", "default_model"]:
            assert k in PROVIDERS[p], f"{p} missing key {k}"
    cfg = LLMConfig()
    assert cfg.provider == "groq"
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 2048

    # Test build_prompt with file_map
    msgs = build_prompt(
        "test?",
        [{"metadata": {"file_path": "a.py", "text": "hello"}, "score": 0.9}],
        file_map={"a.py": 1},
    )
    assert len(msgs) == 2
    assert "[1] a.py" in msgs[1]["content"]

    # Test build_prompt WITHOUT file_map (sequential numbering)
    msgs2 = build_prompt(
        "test?",
        [
            {"metadata": {"file_path": "a.py", "text": "hello"}, "score": 0.9},
            {"metadata": {"file_path": "b.py", "text": "world"}, "score": 0.8},
        ],
    )
    assert "[1] a.py" in msgs2[1]["content"]
    assert "[2] b.py" in msgs2[1]["content"]

    print("  llm: 5 providers OK, LLMConfig OK, build_prompt OK")
except Exception as e:
    errors.append(f"llm: {e}")
    print(f"  FAIL: {e}")

# ---------- ingester ----------
print("--- ingester ---")
try:
    from ingester import (
        embed_texts, chunk_text, clone_repo, read_local_files,
        extract_zip, scrape_docs_url, ingest, IngestionProgress,
        CODE_EXTENSIONS, DOC_EXTENSIONS, SKIP_DIRS, LANGUAGE_MAP,
    )

    # chunk_text code
    chunks_code = chunk_text("def foo():\n    return 42\n\ndef bar():\n    return 99\n", "code")
    assert len(chunks_code) >= 1, "chunk_text (code) returned empty"

    # chunk_text docs
    chunks_docs = chunk_text("This is a documentation page.\n" * 20, "docs")
    assert len(chunks_docs) >= 1, "chunk_text (docs) returned empty"

    # chunk_text empty
    chunks_empty = chunk_text("", "code")
    assert chunks_empty == [], "chunk_text should return [] for empty input"

    # IngestionProgress defaults
    p = IngestionProgress()
    assert p.stage == "idle"
    assert p.done is False
    assert p.current == 0

    # LANGUAGE_MAP coverage
    assert LANGUAGE_MAP[".py"] == "python"
    assert LANGUAGE_MAP[".rs"] == "rust"

    print(f"  ingester: exports OK, chunking OK, IngestionProgress OK")
except Exception as e:
    errors.append(f"ingester: {e}")
    print(f"  FAIL: {e}")

# ---------- searcher ----------
print("--- searcher ---")
try:
    from searcher import SearchResult, search, retrieve_context
    sr = SearchResult(text="x", file_path="f.py", score=0.5)
    assert sr.metadata["text"] == "x"
    assert sr.metadata["file_path"] == "f.py"
    assert sr.score == 0.5
    print("  searcher: SearchResult OK, search/retrieve_context importable")
except Exception as e:
    errors.append(f"searcher: {e}")
    print(f"  FAIL: {e}")

# ---------- app helpers (import without running Streamlit) ----------
print("--- app helpers ---")
try:
    import ast
    with open("app.py") as fh:
        tree = ast.parse(fh.read(), filename="app.py")
    # Verify key functions are defined
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    expected_funcs = {"check_endee", "_github_file_url", "_detect_default_branch",
                      "_render_source_card", "_linkify_superscripts"}
    missing = expected_funcs - func_names
    assert not missing, f"app.py missing functions: {missing}"
    print(f"  app.py: all {len(expected_funcs)} helper functions defined")
except Exception as e:
    errors.append(f"app helpers: {e}")
    print(f"  FAIL: {e}")

# ---------- embedding model ----------
print("--- embedding model ---")
try:
    # Only test if model is already cached (don't download in CI/smoke test)
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    if cached is not None:
        vecs = embed_texts(["hello world", "test query"])
        assert vecs.shape == (2, 384), f"Expected (2,384), got {vecs.shape}"
        import numpy as np
        norms = np.linalg.norm(vecs, axis=1)
        assert all(abs(n - 1.0) < 0.01 for n in norms), f"Vectors not normalized: {norms}"
        print(f"  embed_texts: shape={vecs.shape}, normalized OK")
    else:
        print("  embed_texts: SKIPPED (model not cached, would need download)")
except Exception as e:
    errors.append(f"embedding: {e}")
    print(f"  FAIL: {e}")

# ---------- Summary ----------
print()
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("All smoke tests passed.")
