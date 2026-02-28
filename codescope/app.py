"""
CodeScope — Streamlit UI

A sleek, dark-themed interface for ingesting codebases and chatting with them
via semantic search (Endee) + LLM.
"""

from __future__ import annotations

import os
import sys
import tempfile

from dotenv import load_dotenv
load_dotenv()  # load .env from cwd (codescope/.env)

import streamlit as st

# Ensure the codescope directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from endee_client import EndeeClient
from ingester import IngestionProgress, ingest
from llm import PROVIDERS, LLMConfig, ask_llm
from searcher import retrieve_context

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------

st.set_page_config(
    page_title="CodeScope",
    page_icon="⟨⟩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Custom CSS — Dark cyberpunk theme with gradients & glassmorphism
# ---------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Root variables ---- */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --accent-cyan: #00d4ff;
        --accent-purple: #7b2ff7;
        --accent-pink: #f472b6;
        --accent-green: #34d399;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border-glow: rgba(0, 212, 255, 0.15);
        --gradient-main: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #f472b6 100%);
        --gradient-subtle: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(123,47,247,0.1) 100%);
        --glass-bg: rgba(15, 23, 42, 0.85);
        --glass-border: rgba(0, 212, 255, 0.12);
    }

    /* ---- Global ---- */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stApp > header { background: transparent !important; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #111827 40%, #0f172a 100%) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }

    /* ---- Brand header ---- */
    .brand-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .brand-icon {
        font-size: 2.2rem;
        font-family: 'JetBrains Mono', monospace;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: 2px;
        display: block;
        margin-bottom: 0.25rem;
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .brand-tagline {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---- Status badges ---- */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-ok {
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.08);
    }
    .status-err {
        background: rgba(248, 113, 113, 0.12);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.25);
        box-shadow: 0 0 12px rgba(248, 113, 113, 0.08);
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-ok .status-dot { background: #34d399; box-shadow: 0 0 6px #34d399; }
    .status-err .status-dot { background: #f87171; box-shadow: 0 0 6px #f87171; }

    /* ---- Hero section (no repo selected) ---- */
    .hero-container {
        text-align: center;
        padding: 4rem 2rem 3rem;
        max-width: 680px;
        margin: 0 auto;
    }
    .hero-icon {
        font-size: 4rem;
        font-family: 'JetBrains Mono', monospace;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        display: block;
        margin-bottom: 0.8rem;
        animation: pulse-glow 3s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 2.5rem;
    }
    .hero-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        text-align: left;
    }
    .hero-step {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        transition: border-color 0.3s ease, transform 0.2s ease;
    }
    .hero-step:hover {
        border-color: var(--accent-cyan);
        transform: translateY(-2px);
    }
    .hero-step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .hero-step-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .hero-step-desc {
        font-size: 0.78rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }

    /* ---- Chat messages ---- */
    .stChatMessage {
        border-radius: 14px !important;
        border: 1px solid var(--glass-border) !important;
        background: var(--glass-bg) !important;
        backdrop-filter: blur(8px) !important;
        margin-bottom: 0.8rem !important;
    }

    /* ---- Expander (sources) ---- */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        color: var(--accent-cyan) !important;
        letter-spacing: 0.04em;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: var(--gradient-main) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.02em;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.35) !important;
        transform: translateY(-1px);
    }

    /* ---- Inputs ---- */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider {
        border-color: var(--glass-border) !important;
    }

    /* ---- Dividers ---- */
    hr {
        border-color: var(--glass-border) !important;
        opacity: 0.5;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
        border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------

_defaults = {
    "index_name": None,
    "repo_name": None,
    "repo_url": None,
    "repo_branch": "main",
    "messages": [],
    "ingesting": False,
    "endee_ok": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------
# Helper: check Endee connectivity
# ---------------------------------------------------------------

ENDEE_URL = os.environ.get("ENDEE_URL", "http://localhost:8080/api/v1")


def check_endee() -> bool:
    try:
        EndeeClient(base_url=ENDEE_URL).health()
        return True
    except Exception:
        return False


def _github_file_url(file_path: str) -> str | None:
    """Build a GitHub blob URL for a file path, or return None."""
    url = st.session_state.get("repo_url", "") or ""
    if not url or file_path.startswith("http"):
        return file_path if file_path.startswith("http") else None
    # Normalize: https://github.com/user/repo(.git) → base
    base = url.rstrip("/").removesuffix(".git")
    if "github.com" not in base:
        return None
    branch = st.session_state.get("repo_branch", "main")
    return f"{base}/blob/{branch}/{file_path}"


def _detect_default_branch(repo_url: str) -> str:
    """Query GitHub API for the repo's default branch. Falls back to 'main'."""
    import re as _re, requests as _req
    try:
        cleaned = repo_url.rstrip("/").removesuffix(".git")
        m = _re.search(r"github\.com/([^/]+/[^/]+)", cleaned)
        if not m:
            return "main"
        resp = _req.get(f"https://api.github.com/repos/{m.group(1)}", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("default_branch", "main")
    except Exception:
        pass
    return "main"


# Unicode superscript digits 1-9
_SUPERSCRIPTS = "¹²³⁴⁵⁶⁷⁸⁹"


def _render_source_card(file_path: str, score: float, index: int = 0) -> str:
    """Return a numbered source line with a clickable link."""
    link_url = _github_file_url(file_path)
    sup = _SUPERSCRIPTS[index - 1] if 1 <= index <= 9 else f"{index}."
    prefix = f"**{sup}** "
    if link_url:
        return f"{prefix}[{file_path}]({link_url}) &nbsp; `{score:.3f}`"
    else:
        return f"{prefix}**{file_path}** &nbsp; `{score:.3f}`"


def _linkify_superscripts(text: str, sources: list[dict]) -> str:
    """Replace bare Unicode superscript digits with clickable markdown links.

    Valid citations (those with a matching source) become clickable links.
    Invalid citations (hallucinated numbers beyond the source count) are
    silently removed so users never see orphan superscripts.

    Safe to call multiple times (idempotent) — strips existing markdown links
    before re-linkifying.
    """
    import re as _re

    n = len(sources)

    # Strip any previously-applied citation links so we can re-linkify cleanly.
    # Pattern: [¹](https://...) → ¹
    text = _re.sub(r"\[([" + _re.escape(_SUPERSCRIPTS) + r"])\]\([^)]+\)", r"\1", text)

    # First pass — linkify valid superscripts (¹ … up to n)
    for i, src in enumerate(sources):
        if i >= 9:
            break
        sup = _SUPERSCRIPTS[i]
        url = _github_file_url(src["file"])
        if url:
            text = text.replace(sup, f"[{sup}]({url})")
        # If no GitHub URL, leave the bare superscript as-is (still valid)

    # Second pass — strip any remaining bare superscript digits that have
    # no corresponding source (LLM hallucinated them).
    invalid_sups = _SUPERSCRIPTS[n:]  # e.g. if 4 sources, strip ⁵⁶⁷⁸⁹
    if invalid_sups:
        # Remove them (and optional trailing space/comma/period cleanup)
        pattern = f"[{_re.escape(invalid_sups)}]+"
        text = _re.sub(pattern, "", text)

    return text


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand-header">
            <span class="brand-icon">⟨ ⟩</span>
            <div class="brand-title">CodeScope</div>
            <div class="brand-tagline">Semantic Code Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Endee status
    endee_ok = check_endee()
    st.session_state.endee_ok = endee_ok
    if endee_ok:
        st.markdown(
            '<span class="status-badge status-ok"><span class="status-dot"></span>Endee connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-err"><span class="status-dot"></span>Endee offline</span>',
            unsafe_allow_html=True,
        )
        st.info("Start Endee with `docker-compose up` from the repo root.")

    st.markdown("---")

    # ---- LLM settings ----
    st.markdown("### LLM Provider")

    provider = st.selectbox("Provider", list(PROVIDERS.keys()), index=0)
    prov_info = PROVIDERS[provider]

    default_model = prov_info["default_model"]
    model_name = st.text_input("Model", value=default_model)

    api_key = ""
    if prov_info["env_key"]:
        env_val = os.environ.get(prov_info["env_key"], "")
        api_key = st.text_input(
            "API Key",
            value=env_val,
            type="password",
            help=f"Or set {prov_info['env_key']} env var.",
        )

    llm_cfg = LLMConfig(
        provider=provider,
        model=model_name,
        api_key=api_key,
    )

    st.markdown("---")

    # ---- Ingestion inputs ----
    st.markdown("### Ingest a Repository")

    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")

    # Keep repo_url / branch in sync from sidebar input (so links work without re-ingesting)
    if repo_url and "github.com" in repo_url:
        if st.session_state.get("repo_url") != repo_url:
            st.session_state.repo_url = repo_url
            st.session_state.repo_branch = _detect_default_branch(repo_url)

    docs_url = st.text_input("Docs URL (optional)", placeholder="https://docs.example.com")

    # File / ZIP upload
    uploaded = st.file_uploader(
        "Or upload files / ZIP",
        type=["zip", "py", "js", "ts", "java", "go", "rs", "rb", "c", "cpp", "h",
              "md", "txt", "rst", "yaml", "yml", "toml", "json"],
        accept_multiple_files=True,
    )

    ingest_btn = st.button("Ingest", use_container_width=True, disabled=not endee_ok)

    # ---- Previously indexed ----
    if endee_ok:
        try:
            indexes = EndeeClient(base_url=ENDEE_URL).list_indexes()
            cs_indexes = [idx["name"] for idx in indexes if idx.get("name", "").startswith("codescope-")]
        except Exception:
            cs_indexes = []

        if cs_indexes:
            st.markdown("---")
            st.markdown("### Indexed Repositories")
            selected = st.selectbox(
                "Switch index",
                cs_indexes,
                index=cs_indexes.index(st.session_state.index_name) if st.session_state.index_name in cs_indexes else 0,
            )
            if selected != st.session_state.index_name:
                st.session_state.index_name = selected
                st.session_state.repo_name = selected.replace("codescope-", "", 1)
                st.session_state.messages = []

    # ---- Search filter ----
    st.markdown("---")
    st.markdown("### Search Settings")
    source_options = {"Both code and docs": None, "Code only": "code", "Docs only": "docs"}
    source_label = st.radio("Search scope", list(source_options.keys()), index=0)
    source_filter = source_options[source_label]
    top_k = st.slider("Results to retrieve", 3, 20, 8)

# ---------------------------------------------------------------
# Ingestion handler
# ---------------------------------------------------------------

if ingest_btn:
    if not repo_url and not uploaded:
        st.sidebar.error("Provide a GitHub URL or upload files.")
    else:
        progress_bar = st.progress(0, text="Starting ingestion...")
        status_text = st.empty()

        prog = IngestionProgress()

        # Build kwargs
        kwargs: dict = {
            "endee_url": ENDEE_URL,
            "progress": prog,
        }
        if repo_url:
            kwargs["repo_url"] = repo_url
        if docs_url:
            kwargs["docs_url"] = docs_url

        # Handle uploads
        if uploaded:
            tmp_dir = tempfile.mkdtemp(prefix="codescope_upload_")
            for f in uploaded:
                if f.name.endswith(".zip"):
                    kwargs["zip_data"] = f.read()
                else:
                    path = os.path.join(tmp_dir, f.name)
                    with open(path, "wb") as fh:
                        fh.write(f.getbuffer())
            if "zip_data" not in kwargs and tmp_dir:
                kwargs["local_path"] = tmp_dir

        try:
            idx = ingest(**kwargs)
            st.session_state.index_name = idx
            st.session_state.repo_name = idx.replace("codescope-", "", 1)
            st.session_state.repo_url = repo_url or ""
            if repo_url and "github.com" in repo_url:
                st.session_state.repo_branch = _detect_default_branch(repo_url)
            else:
                st.session_state.repo_branch = "main"
            st.session_state.messages = []
            progress_bar.progress(100, text="Ingestion complete.")
            status_text.success(f"Indexed into **{idx}**")
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"Ingestion failed: {e}")

# ---------------------------------------------------------------
# Main area — Chat
# ---------------------------------------------------------------

if not st.session_state.index_name:
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-icon">⟨ / ⟩</span>
            <div class="hero-title">CodeScope</div>
            <p class="hero-subtitle">
                Ingest any GitHub repository or upload files from the sidebar,
                then ask questions about the codebase in plain English.
            </p>
            <div class="hero-steps">
                <div class="hero-step">
                    <div class="hero-step-num">Step 01</div>
                    <div class="hero-step-title">Chunk & Embed</div>
                    <div class="hero-step-desc">Code and docs are split into intelligent chunks and embedded as vectors.</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">Step 02</div>
                    <div class="hero-step-title">Vector Store</div>
                    <div class="hero-step-desc">Vectors are indexed in Endee, a high-performance HNSW search engine.</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">Step 03</div>
                    <div class="hero-step-title">Semantic Match</div>
                    <div class="hero-step-desc">Your questions are semantically matched against the entire codebase.</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-num">Step 04</div>
                    <div class="hero-step-title">Grounded Answer</div>
                    <div class="hero-step-desc">An LLM generates a precise answer with source file references.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Header
st.markdown(f"## {st.session_state.repo_name}")
st.caption(f"Index: {st.session_state.index_name}")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        # Re-linkify superscripts on every render so citations stay
        # clickable even for history messages or after repo_url changes.
        if msg.get("sources") and st.session_state.get("repo_url"):
            content = _linkify_superscripts(content, msg["sources"])
        st.markdown(content)
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(_render_source_card(src["file"], src["score"], i))

# Chat input
if prompt := st.chat_input("Ask about the codebase..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve & answer
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase..."):
            try:
                context = retrieve_context(
                    prompt,
                    st.session_state.index_name,
                    k=top_k,
                    source_filter=source_filter,
                    endee_url=ENDEE_URL,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                context = []

        if not context:
            answer = "No relevant results found. Try rephrasing or broadening your question."
            sources_display = []
        else:
            # Build a file_map: unique file_path → 1-based index.
            # This ensures the LLM's [N] labels match the Sources list.
            file_map: dict[str, int] = {}
            sources_display: list[dict] = []
            for c in context:
                fp = c["metadata"]["file_path"]
                if fp not in file_map:
                    file_map[fp] = len(file_map) + 1
                    sources_display.append({"file": fp, "score": c.get("score", 0)})

            with st.spinner("Generating answer..."):
                answer = ask_llm(prompt, context, config=llm_cfg, file_map=file_map)

            # Make superscript citations clickable
            answer = _linkify_superscripts(answer, sources_display)

        st.markdown(answer)
        if sources_display:
            with st.expander("Sources", expanded=False):
                for i, src in enumerate(sources_display, 1):
                    st.markdown(_render_source_card(src["file"], src["score"], i))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources_display}
        )
