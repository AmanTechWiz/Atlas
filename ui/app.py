"""Streamlit UI for the Enterprise Knowledge Ops Agent.

Dark-mode chat interface (ChatGPT/Claude-inspired):

  - Top bar: app title + backend model + Reset Knowledge Base button
  - Sidebar: file upload + corpus stats (collapsible)
  - Main: when empty, centered hero with inline upload dropzone
         when chat exists, scrollable message history using `st.chat_message`
  - Each assistant turn shows: answer, confidence badge, source chips,
    and a collapsible Details panel (plan, retrieved chunks, eval log path)
  - Sticky chat input at the bottom using `st.chat_input`

Wired to real `graph.workflow.run_query` (US 1+2+3+4+5+6).

Single application-level corpus (`chroma_db/atlas_corpus`). There are
no per-session collections; one MemoryAgent is shared across re-runs.
The "Reset Knowledge Base" button clears the corpus, the conversation
history, the memory, and the LangGraph state in one action.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

load_dotenv(
    dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"),
    override=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "atlas_corpus"
LOGS_DIR = PROJECT_ROOT / "logs"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from graph.workflow import run_query as _run_query
    from agents.memory import MemoryAgent, get_default_memory
    from agents.retriever import get_corpus_size, get_indexed_doc_types
    from vector_store.ingest import ingest_files, reset_collection
except Exception as _e:  # pragma: no cover
    _run_query = None
    MemoryAgent = None
    get_default_memory = None
    get_corpus_size = None
    get_indexed_doc_types = None
    ingest_files = None
    reset_collection = None
    _IMPORT_ERROR = repr(_e)
else:
    _IMPORT_ERROR = None

UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]


# --------------------------------------------------------------------------- #
# Dark theme CSS
# --------------------------------------------------------------------------- #

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #0f1014;
            --bg-1: #161821;
            --bg-2: #1d1f2b;
            --bg-3: #252836;
            --line: #2a2d3a;
            --text-0: #e7e9ee;
            --text-1: #a0a4b0;
            --text-2: #6b6f7c;
            --accent: #8b5cf6;
            --accent-soft: rgba(139, 92, 246, 0.15);
            --high: #22c55e;
            --med:  #eab308;
            --low:  #ef4444;
            --danger: #ef4444;
        }

        .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: var(--bg-0) !important;
            color: var(--text-0) !important;
        }
        [data-testid="stToolbar"] { background: transparent !important; }

        #MainMenu, footer, .viewerBadge_link__qRIco, [data-testid="stDecoration"] {
            display: none !important;
        }

        .atlas-top {
            position: sticky; top: 0; z-index: 100;
            display: flex; align-items: center; justify-content: space-between;
            padding: 12px 24px;
            background: rgba(15, 16, 20, 0.85);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--line);
        }
        .atlas-brand { display: flex; align-items: center; gap: 10px; }
        .atlas-logo {
            width: 30px; height: 30px; border-radius: 8px;
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; box-shadow: 0 2px 10px rgba(139, 92, 246, 0.3);
        }
        .atlas-title { font-size: 1.05rem; font-weight: 600; color: var(--text-0); line-height: 1.2; }
        .atlas-sub { font-size: 0.72rem; color: var(--text-1); margin-top: 1px; }

        .atlas-meta { display: flex; align-items: center; gap: 16px; color: var(--text-1); font-size: 0.78rem; }
        .atlas-meta .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--high); display: inline-block; margin-right: 6px; }

        .main .block-container { max-width: 820px; padding-top: 1rem; padding-bottom: 160px; }

        .hero {
            text-align: center; padding: 64px 16px 24px;
        }
        .hero .big {
            font-size: 2.4rem; font-weight: 700; color: var(--text-0);
            margin: 0 0 8px 0; letter-spacing: -0.02em;
        }
        .hero .sub { color: var(--text-1); font-size: 0.95rem; margin-bottom: 32px; }
        .hero .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; max-width: 560px; margin: 0 auto; }
        .hero .chip {
            background: var(--bg-1); border: 1px solid var(--line);
            border-radius: 12px; padding: 14px 16px; text-align: left;
            color: var(--text-0); font-size: 0.86rem; cursor: pointer;
            transition: background 0.15s, border-color 0.15s, transform 0.15s;
        }
        .hero .chip:hover { background: var(--bg-2); border-color: var(--accent); }
        .hero .chip .lbl { color: var(--text-1); font-size: 0.72rem; margin-top: 4px; display: block; }
        .hero .upload-cta {
            margin-top: 24px; padding: 22px; border: 1.5px dashed var(--line);
            border-radius: 14px; background: var(--bg-1);
        }
        .hero .upload-cta .ic { font-size: 28px; margin-bottom: 6px; }
        .hero .upload-cta .tx { color: var(--text-0); font-weight: 500; }
        .hero .upload-cta .sm { color: var(--text-1); font-size: 0.78rem; margin-top: 4px; }

        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: 0 !important;
            padding: 14px 0 !important;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-0) !important; line-height: 1.6; font-size: 0.95rem;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
            background: var(--bg-2) !important; color: #f0abfc !important; border: 0 !important;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
            background: var(--bg-1) !important; border: 1px solid var(--line) !important;
        }
        [data-testid="stChatMessageContent"] {
            background: transparent !important;
        }

        .badge {
            display: inline-block; padding: 2px 9px; border-radius: 999px;
            font-size: 0.7rem; font-weight: 600; letter-spacing: 0.01em;
        }
        .badge-high { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
        .badge-med  { background: rgba(234, 179, 8, 0.15); color: #facc15; }
        .badge-low  { background: rgba(239, 68, 68, 0.15); color: #f87171; }
        .badge-na   { background: var(--bg-3); color: var(--text-1); }

        .src-wrap { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
        .src-lbl { color: var(--text-2); font-size: 0.72rem; }
        .src-chip {
            display: inline-flex; align-items: center; gap: 4px;
            background: var(--bg-2); border: 1px solid var(--line);
            color: var(--text-0); padding: 2px 9px; border-radius: 999px;
            font-size: 0.72rem;
        }
        .src-chip .p { color: var(--text-2); font-size: 0.7rem; }
        .src-chip.empty { color: var(--text-2); }

        details.atlas-details {
            margin-top: 10px; background: var(--bg-1);
            border: 1px solid var(--line); border-radius: 10px; padding: 0;
        }
        details.atlas-details > summary {
            list-style: none; cursor: pointer; padding: 9px 14px;
            color: var(--text-1); font-size: 0.8rem; user-select: none;
        }
        details.atlas-details > summary::-webkit-details-marker { display: none; }
        details.atlas-details[open] > summary { color: var(--text-0); border-bottom: 1px solid var(--line); }
        details.atlas-details .body { padding: 12px 14px; }
        details.atlas-details .body h5 { color: var(--text-1); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; margin: 12px 0 6px; }
        details.atlas-details .body .chunk { padding: 8px 10px; background: var(--bg-2); border-radius: 8px; margin: 6px 0; font-size: 0.82rem; }
        details.atlas-details .body .chunk .meta { color: var(--text-2); font-size: 0.72rem; margin-bottom: 4px; }
        details.atlas-details .body .chunk .txt { color: var(--text-0); }

        [data-testid="stSidebar"] { background: var(--bg-1) !important; border-right: 1px solid var(--line) !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--text-0) !important; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: var(--text-1) !important; }
        [data-testid="stSidebarNav"] { display: none; }

        [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] > div {
            background: var(--bg-2) !important;
            color: var(--text-0) !important;
            border: 1px solid var(--line) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color: var(--text-2) !important; }

        .stButton > button {
            background: var(--bg-2) !important; color: var(--text-0) !important;
            border: 1px solid var(--line) !important; border-radius: 8px !important;
            transition: all 0.15s;
        }
        .stButton > button:hover { background: var(--bg-3) !important; border-color: var(--accent) !important; }
        .stButton > button[kind="primary"] {
            background: var(--accent) !important; border-color: var(--accent) !important; color: white !important;
        }
        .stButton > button[kind="primary"]:hover { background: #7c3aed !important; }
        .stButton > button.reset-kb {
            background: rgba(239, 68, 68, 0.10) !important;
            color: #f87171 !important;
            border: 1px solid rgba(239, 68, 68, 0.4) !important;
            font-weight: 600 !important;
        }
        .stButton > button.reset-kb:hover {
            background: rgba(239, 68, 68, 0.20) !important;
            border-color: var(--danger) !important;
        }

        .side-info { display: flex; flex-direction: column; gap: 8px; }
        .side-info-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 10px; background: var(--bg-2);
            border: 1px solid var(--line); border-radius: 8px;
        }
        .side-info-lbl { color: var(--text-1); font-size: 12px; }
        .side-info-val { color: var(--text-0); font-size: 13px; font-weight: 600; }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-2) !important; border: 1.5px dashed var(--line) !important; border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
        [data-testid="stFileUploaderDropzone"] * { color: var(--text-1) !important; }

        [data-testid="stMetric"] { background: var(--bg-2); padding: 8px 12px; border-radius: 8px; border: 1px solid var(--line); }
        [data-testid="stMetricLabel"] { color: var(--text-1) !important; }
        [data-testid="stMetricValue"] { color: var(--text-0) !important; }

        .atlas-code {
            background: var(--bg-2); border: 1px solid var(--line);
            border-radius: 8px; padding: 8px 12px;
            color: var(--text-0); font-size: 0.82rem;
            font-family: ui-monospace, SFMono-Regular, monospace;
        }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--bg-0); }
        ::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def _ensure_session_state() -> None:
    if "memory" not in st.session_state or st.session_state.memory is None:
        st.session_state.memory = (
            get_default_memory() if get_default_memory is not None else None
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "confirming_reset" not in st.session_state:
        st.session_state.confirming_reset = False
    if "_bootstrapped" not in st.session_state:
        st.session_state._bootstrapped = False


def _corpus_size() -> int:
    if get_corpus_size is None:
        return 0
    try:
        return int(get_corpus_size() or 0)
    except Exception:
        return 0


def _indexed_doc_types() -> List[str]:
    if get_indexed_doc_types is None:
        return []
    try:
        return list(get_indexed_doc_types() or [])
    except Exception:
        return []


def reset_knowledge_base() -> None:
    """Wipe the entire knowledge base: corpus + memory + chat history.

    Called by the prominent "Reset Knowledge Base" button in the
    sidebar. The user is asked to confirm before the wipe happens.
    """
    if reset_collection is not None:
        try:
            reset_collection(PERSIST_DIR, COLLECTION_NAME)
        except Exception:
            pass
    from agents.retriever import invalidate_cache
    try:
        invalidate_cache()
    except Exception:
        pass
    memory = st.session_state.get("memory")
    if memory is not None:
        memory.reset()
    st.session_state.messages = []
    st.session_state.uploaded_filenames = []
    st.session_state.uploader_key += 1
    st.session_state.confirming_reset = False


def _auto_reset_on_startup() -> None:
    """Wipe the knowledge base on every fresh Streamlit session.

    Streamlit's session state is reset on every browser refresh /
    server restart, but the underlying ChromaDB collection persists
    on disk. The user wants a clean slate on every session start, so
    we call `reset_collection` here exactly once per Streamlit
    session. Re-uploads are required after the wipe.
    """
    if reset_collection is not None:
        try:
            reset_collection(PERSIST_DIR, COLLECTION_NAME)
        except Exception:
            pass
    from agents.retriever import invalidate_cache
    try:
        invalidate_cache()
    except Exception:
        pass
    memory = st.session_state.get("memory")
    if memory is not None:
        memory.reset()
    st.session_state.messages = []
    st.session_state.uploaded_filenames = []
    st.session_state.uploader_key += 1


def _ingest_files(uploaded_files) -> tuple[bool, str]:
    if not uploaded_files or ingest_files is None:
        return False, "No files uploaded or ingest module unavailable."

    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_upload_"))
    saved_paths: List[Path] = []
    try:
        for uploaded in uploaded_files:
            target = tmp_dir / uploaded.name
            with open(target, "wb") as f:
                f.write(uploaded.getbuffer())
            saved_paths.append(target)

        backend = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
        n = ingest_files(saved_paths, PERSIST_DIR, COLLECTION_NAME, backend)

        from agents.retriever import invalidate_cache
        try:
            invalidate_cache()
        except Exception:
            pass

        existing = set(st.session_state.uploaded_filenames or [])
        new = {p.name for p in saved_paths}
        st.session_state.uploaded_filenames = sorted(existing | new)

        total = _corpus_size()
        return True, f"Indexed {n} new chunk(s) from {len(saved_paths)} file(s). Corpus now {total}."
    except FileNotFoundError as e:
        return False, f"Unsupported file type: {e}"
    except Exception as e:
        return False, f"Ingest failed: {e}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def _confidence_badge_html(confidence) -> str:
    if confidence is None:
        return '<span class="badge badge-na">no score</span>'
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return '<span class="badge badge-na">no score</span>'
    if c >= 0.7:
        cls, label = "badge-high", "high"
    elif c >= 0.5:
        cls, label = "badge-med", "medium"
    else:
        cls, label = "badge-low", "low"
    return f'<span class="badge {cls}">confidence {c:.2f} · {label}</span>'


def _source_chips(sources: List[str]) -> str:
    if not sources:
        return '<span class="src-chip empty">no sources</span>'
    chips = []
    for s in sources:
        if "::page::" in s:
            src, page = s.split("::page::", 1)
            chips.append(
                f'<span class="src-chip">{src} <span class="p">· p.{page}</span></span>'
            )
        else:
            chips.append(f'<span class="src-chip">{s}</span>')
    return "".join(chips)


def _details_html(msg: Dict[str, Any]) -> str:
    parts: List[str] = ['<details class="atlas-details"><summary>View details</summary><div class="body">']
    intent = msg.get("intent")
    targets = msg.get("target_doc_types") or []
    if intent:
        target_str = ", ".join(targets) if targets else "(none)"
        parts.append(
            f'<h5>Intent</h5><div class="atlas-code">{_escape(intent)} · targets=[{_escape(target_str)}]</div>'
        )
    original = msg.get("original_query")
    rewritten = msg.get("search_query")
    if rewritten and original and rewritten != original:
        parts.append('<h5>Rewritten search query</h5>')
        parts.append(f'<div class="atlas-code">{_escape(rewritten)}</div>')
    plan = msg.get("plan") or []
    if plan:
        parts.append('<h5>Plan</h5><ol style="margin:0;padding-left:18px;color:var(--text-0);">')
        for step in plan:
            parts.append(f"<li>{_escape(step)}</li>")
        parts.append("</ol>")
    chunks = msg.get("retrieved_chunks") or []
    if chunks:
        parts.append(f'<h5>Retrieved chunks ({len(chunks)})</h5>')
        for c in chunks:
            src = _escape(str(c.get("source", "?")))
            page = c.get("page", 0)
            score = c.get("relevance_score", 0.0)
            strategy = c.get("retrieval_strategy", "?")
            doc_type = c.get("doc_type", "?")
            try:
                score_str = f"{float(score):.3f}"
            except (TypeError, ValueError):
                score_str = "0.000"
            text = _escape((c.get("text") or "")[:280])
            more = "..." if len(c.get("text") or "") > 280 else ""
            parts.append(
                f'<div class="chunk"><div class="meta">{src} · p.{page} · {doc_type} · '
                f'score {score_str} · strategy {_escape(str(strategy))}</div>'
                f'<div class="txt">{text}{more}</div></div>'
            )
    flags = (msg.get("verification") or {}).get("flags") or []
    if flags:
        parts.append('<h5>Flags</h5>')
        for f in flags:
            parts.append(f'<div class="atlas-code">{_escape(str(f))}</div>')
    log_path = msg.get("log_path")
    if log_path:
        parts.append('<h5>Evaluation log</h5>')
        parts.append(f'<div class="atlas-code">{_escape(log_path)}</div>')
    parts.append("</div></details>")
    return "".join(parts)


def _escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title="Atlas — Knowledge Ops",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _ensure_session_state()

    if not st.session_state._bootstrapped:
        _auto_reset_on_startup()
        st.session_state._bootstrapped = True

    backend = os.getenv("EMBEDDING_BACKEND", "ollama")
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    corpus = _corpus_size()
    mem_count = len(st.session_state.memory) if st.session_state.memory else 0
    doc_types = _indexed_doc_types()

    st.markdown(
        f"""
        <div class="atlas-top">
            <div class="atlas-brand">
                <div class="atlas-logo">🧠</div>
                <div>
                    <div class="atlas-title">Atlas</div>
                    <div class="atlas-sub">Enterprise Knowledge Ops Agent</div>
                </div>
            </div>
            <div class="atlas-meta">
                <span><span class="dot"></span>{backend} · {model}</span>
                <span>corpus: <b style="color:var(--text-0)">{corpus}</b> chunks</span>
                <span>memory: <b style="color:var(--text-0)">{mem_count}</b> turns</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        if not st.session_state.confirming_reset:
            st.button(
                "Reset Knowledge Base",
                key="reset_kb",
                on_click=lambda: setattr(st.session_state, "confirming_reset", True),
                use_container_width=True,
            )
        else:
            st.warning(
                "This will delete every indexed chunk, clear the conversation "
                "history, and reset the memory. This cannot be undone."
            )
            c1, c2 = st.columns(2)
            if c1.button("Yes, reset", type="primary", key="confirm_reset", use_container_width=True):
                with st.spinner("Resetting..."):
                    reset_knowledge_base()
                st.success("Knowledge base reset.")
                st.rerun()
            if c2.button("Cancel", key="cancel_reset", use_container_width=True):
                st.session_state.confirming_reset = False
                st.rerun()

        st.markdown("---")

        st.markdown(
            f"""
            <div class="side-info">
                <div class="side-info-row">
                    <span class="side-info-lbl">Chunks embedded</span>
                    <span class="side-info-val">{corpus}</span>
                </div>
                <div class="side-info-row">
                    <span class="side-info-lbl">Embedding model</span>
                    <span class="side-info-val">{_escape(os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))}</span>
                </div>
                <div class="side-info-row">
                    <span class="side-info-lbl">LLM</span>
                    <span class="side-info-val">{_escape(model)}</span>
                </div>
                <div class="side-info-row">
                    <span class="side-info-lbl">Backend</span>
                    <span class="side-info-val">{_escape(backend)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if _IMPORT_ERROR:
            st.error(f"Import error: {_IMPORT_ERROR}")

        st.markdown("---")

        with st.expander("Upload documents", expanded=False):
            uploaded = st.file_uploader(
                "Drop PDFs / DOCX / TXT / MD",
                type=UPLOAD_TYPES,
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed",
            )
            if uploaded and st.button("Index uploaded files", type="primary", use_container_width=True):
                with st.spinner("Embedding and indexing..."):
                    ok, msg = _ingest_files(uploaded)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()

        if st.session_state.uploaded_filenames:
            st.markdown("**Indexed files**")
            for name in st.session_state.uploaded_filenames:
                st.markdown(f"`{name}`")
        else:
            st.caption("No files indexed yet.")

    if not st.session_state.messages:
        _render_welcome(corpus, uploaded_filenames=st.session_state.uploaded_filenames)
    else:
        for msg in st.session_state.messages:
            _render_message(msg)

    placeholder = (
        "Ask anything about your documents…"
        if corpus > 0
        else "Upload a document in the sidebar to begin."
    )
    user_input = st.chat_input(placeholder)
    if user_input and user_input.strip():
        _run_turn(user_input.strip())


def _render_welcome(corpus: int, uploaded_filenames: List[str]) -> None:
    """Show centered hero when chat is empty."""
    if corpus == 0 and not uploaded_filenames:
        sub = "Upload a PDF, DOCX, or TXT in the sidebar to get started."
    elif corpus == 0:
        sub = "Files uploaded but not yet indexed — click 'Index uploaded files' in the sidebar."
    else:
        sub = f"Ready · {corpus} chunks indexed across {len(uploaded_filenames)} file(s)."

    st.markdown(
        f"""
        <div class="hero">
            <div class="big">How can I help you today?</div>
            <div class="sub">{_escape(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_message(msg: Dict[str, Any]) -> None:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    avatar = "🧑" if role == "user" else "🧠"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if role == "assistant":
            confidence = msg.get("confidence")
            sources = msg.get("sources") or []
            st.markdown(
                f'<div style="margin-top:6px">{_confidence_badge_html(confidence)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="src-wrap"><span class="src-lbl">sources:</span>{_source_chips(sources)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_details_html(msg), unsafe_allow_html=True)


def _run_turn(user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})

    if _run_query is None or st.session_state.memory is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Backend unavailable.** {_IMPORT_ERROR or 'Unknown error'}",
        })
        st.rerun()
        return

    mem: MemoryAgent = st.session_state.memory
    with st.spinner("Thinking…"):
        result = _run_query(user_text, memory=mem)

    plan = result.get("plan") or []
    chunks = result.get("retrieved_chunks") or []
    final = result.get("final_answer", "")
    verification = result.get("verification_result") or {}
    search_query = result.get("search_query")
    log_path = result.get("log_path")
    intent = result.get("intent")
    target_doc_types = result.get("target_doc_types") or []

    sources: List[str] = []
    for c in chunks:
        src = c.get("source", "")
        page = c.get("page", 0)
        if src:
            sources.append(f"{src}::page::{page}" if page else src)
    sources = sorted(set(sources))

    st.session_state.messages.append({
        "role": "assistant",
        "content": final,
        "confidence": verification.get("confidence"),
        "sources": sources,
        "plan": plan,
        "intent": intent,
        "target_doc_types": target_doc_types,
        "search_query": search_query,
        "original_query": user_text,
        "retrieved_chunks": chunks,
        "verification": verification,
        "log_path": log_path,
    })
    st.rerun()


if __name__ == "__main__":
    main()
