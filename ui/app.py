"""Streamlit UI for the Enterprise Knowledge Ops Agent.

Wired to the real `graph.workflow.run_query` (Official US 1+2+3+4+5+6).

Per-session behavior:
- Each browser session gets a fresh `session_id` (UUID4).
- Each session has its OWN ChromaDB collection (named `session_<id>`,
  persisted to `chroma_db_sessions/<id>/`).
- Each session has its OWN MemoryAgent (multi-turn conversations are
  isolated across sessions).
- "Reset Session" deletes the session's collection, regenerates the
  session_id, and clears all in-memory state.

File upload:
- `st.file_uploader` accepts multiple PDF / DOCX / TXT / MD files.
- Uploaded files are written to a per-session temp dir, ingested into
  the session's collection, then the temp dir is cleaned up.
- The user can re-upload to replace the session's corpus.

Corpus-agnostic:
- No hardcoded sample docs. The system starts with an empty corpus.
- The input guardrail blocks queries until at least one document is
  uploaded (per US 5 governance).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "chroma_db_sessions"
LOGS_DIR = PROJECT_ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ui")

STUB_ENABLED = False

try:
    from graph.workflow import run_query as _run_query
    from agents.memory import MemoryAgent
    from agents.retriever import (
        get_active_collection_name,
        get_active_persist_dir,
        get_corpus_size,
        set_active_collection,
    )
    from vector_store.ingest import safe_ingest_files
except Exception as _e:  # pragma: no cover
    _run_query = None
    MemoryAgent = None
    get_active_collection_name = None
    get_active_persist_dir = None
    get_corpus_size = None
    set_active_collection = None
    safe_ingest_files = None
    _IMPORT_ERROR = repr(_e)
else:
    _IMPORT_ERROR = None

UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]
MAX_UPLOAD_MB = 25

SAMPLE_QUERIES = [
    "Summarize the key points across the uploaded documents.",
    "What is the most important policy or procedure described here?",
    "Compare and contrast the requirements described in the documents.",
]


def stub_run_query(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "plan": ["[RETRIEVE] (stub)", "[ANALYZE] (stub)", "[VERIFY] (stub)"],
        "retrieved_chunks": [],
        "draft_answer": "(stub answer)",
        "verification_result": {"confidence": 0.0, "grounded": False, "flags": ["STUB"]},
        "final_answer": "Stub mode is enabled. Set STUB_ENABLED=False to use the real backend.",
        "decision_trace": ["STUB: stub_run_query called"],
        "session_history": [],
        "error": "stub_mode",
        "api_error": None,
        "needs_disclaimer": False,
    }


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _session_persist_dir(session_id: str) -> Path:
    return SESSIONS_DIR / f"session_{session_id}"


def _session_collection_name(session_id: str) -> str:
    return f"session_{session_id}"


def _ensure_session_active() -> None:
    """Make sure `st.session_state.session_id` is set AND the retriever
    is pointed at this session's collection. Called on every render."""
    if not st.session_state.get("session_id"):
        st.session_state.session_id = _new_session_id()
        st.session_state.memory = MemoryAgent(st.session_state.session_id) if MemoryAgent else None
        st.session_state.uploaded_filenames = []
        st.session_state.corpus_size_at_start = 0
        st.session_state.ingest_message = ""

    if set_active_collection is not None:
        sid = st.session_state.session_id
        target_dir = _session_persist_dir(sid)
        target_coll = _session_collection_name(sid)
        if get_active_collection_name() != target_coll:
            set_active_collection(target_coll, persist_dir=target_dir)
            log.info("UI bound to session=%s collection=%s", sid, target_coll)


def reset_session() -> None:
    """Delete the session's collection, generate a new session_id, clear state."""
    if (
        set_active_collection is not None
        and get_active_collection_name is not None
        and get_active_persist_dir is not None
    ):
        try:
            from vector_store.ingest import delete_collection
            delete_collection(
                get_active_persist_dir(),
                get_active_collection_name(),
            )
        except Exception as e:
            log.warning("Reset: could not delete collection cleanly: %s", e)
    st.session_state.session_id = _new_session_id()
    st.session_state.memory = MemoryAgent(st.session_state.session_id) if MemoryAgent else None
    st.session_state.uploaded_filenames = []
    st.session_state.ingest_message = ""
    st.session_state.corpus_size_at_start = 0
    st.session_state.messages = []
    st.session_state.session_history = []
    st.session_state.last_result = None
    if set_active_collection is not None:
        set_active_collection(
            _session_collection_name(st.session_state.session_id),
            persist_dir=_session_persist_dir(st.session_state.session_id),
        )


def _ingest_files_into_session(uploaded_files) -> tuple[bool, str]:
    """Write uploaded files to a per-session temp dir, ingest them into
    the session's collection, then clean up the temp dir. Returns
    (success, message)."""
    if not uploaded_files or safe_ingest_files is None:
        return False, "No files uploaded or ingest module unavailable."

    sid = st.session_state.session_id
    persist_dir = _session_persist_dir(sid)
    collection_name = _session_collection_name(sid)
    persist_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"atlas_upload_{sid}_"))
    saved_paths: list[Path] = []
    try:
        for uf in uploaded_files:
            safe_name = Path(uf.name).name
            target = tmp_dir / safe_name
            with open(target, "wb") as f:
                f.write(uf.getbuffer().tobytes())
            saved_paths.append(target)

        backend = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
        n = safe_ingest_files(saved_paths, persist_dir, collection_name, backend)
        st.session_state.uploaded_filenames = sorted({p.name for p in saved_paths})
        return True, f"Indexed {n} chunk(s) from {len(saved_paths)} file(s)."
    except FileNotFoundError as e:
        return False, f"Ingest failed — unsupported file type: {e}"
    except Exception as e:
        log.exception("Ingest failed")
        return False, f"Ingest failed: {e}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_answer_tab(result: dict[str, Any]) -> None:
    v = result.get("verification_result") or {}
    confidence = float(v.get("confidence", 0.0) or 0.0)
    grounded = bool(v.get("grounded", False))
    flags = v.get("flags", []) or []
    api_error = result.get("api_error")
    error = result.get("error") or ""
    guardrail_rejected = error.startswith("guardrail_rejected")

    if guardrail_rejected:
        reason = error.split("guardrail_rejected: ", 1)[-1]
        st.error(f"**Input guardrail rejected this query** — {reason}")
        st.caption("The pipeline did not run. Rephrase the query and try again.")
    elif api_error:
        st.error(f"**Service unavailable** — {api_error}")
        st.caption("The Agent Trace tab shows which stage failed. The low-confidence "
                   "banner is suppressed because the cause is an upstream API failure, "
                   "not a weak retrieval.")
    elif confidence < 0.6:
        st.error(
            f"**Low confidence** ({confidence:.2f}) — the answer may not be fully supported "
            f"by the retrieved source documents. Treat it as provisional and verify against "
            f"the cited sources before acting on it."
        )
    elif not grounded:
        st.warning(
            f"**Not grounded** ({confidence:.2f}) — the Verifier flagged issues with the answer. "
            f"See the Agent Trace tab for details."
        )

    if not api_error and not guardrail_rejected:
        st.markdown(confidence_badge(confidence), unsafe_allow_html=True)
    st.markdown("### Final Answer")
    st.markdown(result["final_answer"])

    if result["retrieved_chunks"]:
        unique_sources = sorted({c["source"] for c in result["retrieved_chunks"]})
        st.markdown(
            "**Sources:** " + " · ".join(f"`{s}`" for s in unique_sources)
        )

    if flags:
        with st.expander(f"Verifier flags ({len(flags)})"):
            for f in flags:
                st.markdown(f"- `{f}`")


def _score_badge_html(label: str, value: float) -> str:
    if value >= 0.7:
        color, band = "#1f9d55", "HIGH"
    elif value >= 0.5:
        color, band = "#b08800", "MEDIUM"
    else:
        color, band = "#c0392b", "LOW"
    return (
        f'<span style="background:{color};color:white;padding:3px 9px;'
        f'border-radius:10px;font-weight:600;font-size:0.8em;margin-right:6px;">'
        f"{label}: {value:.2f} ({band})</span>"
    )


def render_agent_trace_tab(result: dict[str, Any]) -> None:
    st.markdown("### Orchestrator Plan")
    for i, step in enumerate(result["plan"], 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("### Decision Trace")
    for line in result["decision_trace"]:
        st.code(line, language="text")

    st.markdown("### Retriever Detail")
    st.markdown(f"**{len(result['retrieved_chunks'])}** chunk(s) retrieved.")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        with st.expander(
            f"Chunk {i} — {c['source']} (page {c['page']}) — relevance {c['relevance_score']:.2f}"
        ):
            st.write(c["text"])

    st.markdown("### Analyst Draft")
    with st.expander("Show draft answer (pre-verification)"):
        st.text(result["draft_answer"])

    st.markdown("### Verifier Result")
    v = result["verification_result"]

    if "grounding_confidence" in v:
        cols = st.columns(3)
        cols[0].markdown(
            _score_badge_html("Grounding", float(v.get("grounding_confidence") or 0)),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            _score_badge_html("Answer Q.", float(v.get("answer_quality") or 0)),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            _score_badge_html("Retrieval", float(v.get("retrieval_confidence") or 0)),
            unsafe_allow_html=True,
        )

    st.markdown(f"- **confidence:** {v.get('confidence')}")
    st.markdown(f"- **grounded:** {v.get('grounded')}")
    if v.get("flags"):
        st.markdown(f"- **flags:** {', '.join(v['flags'])}")
    claims = v.get("claims") or []
    if claims:
        with st.expander(f"Claims ({len(claims)})"):
            for c in claims:
                tag = c.get("support", "?")
                st.markdown(f"- **[{tag}]** {c.get('claim', '')}")
    aspects = v.get("question_aspects") or []
    if aspects:
        with st.expander(f"Question aspects ({len(aspects)})"):
            for a in aspects:
                st.markdown(f"- **[{a.get('status', '?')}]** {a.get('aspect', '')}")
    with st.expander("Raw verification JSON"):
        st.json(v)


def render_sources_tab(result: dict[str, Any]) -> None:
    chunks = result["retrieved_chunks"]
    if not chunks:
        st.info("No sources were retrieved for this query.")
        return
    unique_sources = sorted({c["source"] for c in chunks})
    st.markdown(f"**{len(unique_sources)}** source file(s) cited, **{len(chunks)}** chunk(s) total.")
    for src in unique_sources:
        with st.expander(f"`{src}`"):
            for i, c in enumerate([c for c in chunks if c["source"] == src], 1):
                st.markdown(
                    f"**Chunk {i}** (page {c['page']}, relevance {c['relevance_score']:.2f})"
                )
                st.write(c["text"])


def render_eval_log_tab(result: dict[str, Any]) -> None:
    log_path = result.get("log_path")
    if not log_path or not Path(log_path).exists():
        st.info("No evaluation log file was written for this query.")
        return
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        st.markdown(f"**Log file:** `{log_path}`")
        st.markdown(f"**Entries:** {len(entries)}")
        for e in entries:
            with st.expander(f"[{e['stage']}] {e['event']}  ·  {e['timestamp']}"):
                st.json(e.get("data") or {})
    except Exception as e:
        st.error(f"Could not read log file: {e}")


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.7:
        color, label = "#1f9d55", "HIGH"
    elif confidence >= 0.5:
        color, label = "#b08800", "MEDIUM"
    else:
        color, label = "#c0392b", "LOW"
    return (
        f'<span style="background:{color};color:white;padding:6px 14px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">'
        f"Confidence: {confidence:.2f} ({label})</span>"
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Atlas")
        st.caption("Enterprise Knowledge Ops Agent")

        sid = st.session_state.get("session_id", "?")
        st.markdown(f"**Session ID:** `{sid}`")
        memory_obj = st.session_state.get("memory")
        if memory_obj is not None:
            st.markdown(f"**Conversation turns in memory:** {len(memory_obj)}")
        corpus_size = 0
        if get_corpus_size is not None:
            try:
                corpus_size = get_corpus_size()
            except Exception:
                corpus_size = 0
        st.markdown(f"**Chunks in corpus:** {corpus_size}")

        st.markdown("### Model Info")
        llm = os.getenv("GEMINI_MODEL", "not set")
        backend = os.getenv("EMBEDDING_BACKEND", "not set").lower()
        if backend == "ollama":
            emb_display = f"Ollama (`{os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}`)"
        elif backend == "gemini":
            emb_display = f"Gemini (`{os.getenv('GEMINI_EMBEDDING_MODEL', 'models/gemini-embedding-001')}`)"
        else:
            emb_display = f"unknown backend `{backend}`"
        st.markdown(f"- **LLM:** `{llm}`")
        st.markdown(f"- **Embeddings:** {emb_display}")

        st.markdown("### Upload Documents")
        st.caption(
            "Upload PDF, DOCX, TXT, or MD files. They are indexed into "
            f"this session's private collection (max {MAX_UPLOAD_MB} MB per file)."
        )
        uploaded = st.file_uploader(
            "Choose files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{sid}",
        )
        if uploaded and st.button("Index uploaded files", use_container_width=True):
            with st.spinner(f"Embedding {len(uploaded)} file(s) into session collection..."):
                ok, msg = _ingest_files_into_session(uploaded)
            st.session_state.ingest_message = msg
            st.toast(msg, icon="✅" if ok else "❌")
            st.rerun()

        if st.session_state.get("ingest_message"):
            (st.success if corpus_size > 0 else st.error)(st.session_state.ingest_message)

        if st.session_state.get("uploaded_filenames"):
            st.markdown("**Indexed in this session:**")
            for name in st.session_state.uploaded_filenames:
                st.markdown(f"- `{name}`")

        st.markdown("### Actions")
        if st.button("Reset Session", use_container_width=True):
            reset_session()
            st.toast("Session reset — fresh session_id and empty corpus.", icon="🔄")
            st.rerun()

        st.markdown("### Session History")
        if not st.session_state.session_history:
            st.caption("No queries yet.")
        else:
            for i, entry in enumerate(reversed(st.session_state.session_history[-10:]), 1):
                with st.expander(f"{i}. {entry['query'][:60]}{'…' if len(entry['query']) > 60 else ''}"):
                    st.markdown(f"**Q:** {entry['query']}")
                    st.markdown(f"**A:** {entry['answer'][:200]}{'…' if len(entry['answer']) > 200 else ''}")
                    st.caption(f"Sources: {', '.join(entry.get('sources', []))}  ·  {entry['timestamp']}")


def render_main() -> None:
    st.title("Atlas — Enterprise Knowledge Ops Agent")
    st.markdown(
        "Upload your enterprise documents in the sidebar, then ask complex "
        "questions across them. The answer is grounded in your uploaded "
        "corpus and verified before being shown."
    )

    if STUB_ENABLED or _IMPORT_ERROR:
        msg = "UI shell is wired to **stubbed data**"
        if STUB_ENABLED:
            msg += " (STUB_ENABLED=True)"
        if _IMPORT_ERROR:
            msg += f" — backend import failed: `{_IMPORT_ERROR}`"
        msg += ". The real backend will replace the stub when imports succeed."
        st.warning(msg)

    if not st.session_state.get("uploaded_filenames"):
        st.info(
            "**Getting started:** Use the **Upload Documents** panel in the sidebar "
            "to add at least one PDF, DOCX, or TXT file. Once indexed, the guardrail "
            "will allow queries and you can ask questions across the uploaded corpus."
        )

    st.markdown("### Try a sample query")
    cols = st.columns(len(SAMPLE_QUERIES))
    for col, sample in zip(cols, SAMPLE_QUERIES):
        if col.button(
            sample[:55] + ("…" if len(sample) > 55 else ""),
            key=f"sample_{sample[:10]}_{st.session_state.session_id}",
            use_container_width=True,
        ):
            st.session_state.query_input = sample

    st.markdown("### Ask your own question")
    st.text_input(
        "Your query:",
        key="query_input",
        placeholder="e.g., Summarize the key points in the uploaded documents.",
        label_visibility="collapsed",
    )
    ask = st.button("Ask", type="primary", use_container_width=False)

    if ask and st.session_state.query_input.strip():
        query = st.session_state.query_input
        memory_obj = st.session_state.get("memory")
        with st.spinner(
            "Running agent pipeline (orchestrate → retrieve → analyze → verify → "
            "finalize → memory)..."
        ):
            if STUB_ENABLED or _run_query is None:
                result = stub_run_query(query)
            else:
                result = _run_query(query, memory=memory_obj)
        st.session_state.last_result = result
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.session_history.append(
            {
                "query": query,
                "answer": result["final_answer"],
                "sources": sorted({c["source"] for c in result["retrieved_chunks"]}),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

    if st.session_state.last_result:
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📝 Answer", "🔍 Agent Trace", "📚 Sources", "📊 Evaluation Log"]
        )
        with tab1:
            render_answer_tab(st.session_state.last_result)
        with tab2:
            render_agent_trace_tab(st.session_state.last_result)
        with tab3:
            render_sources_tab(st.session_state.last_result)
        with tab4:
            render_eval_log_tab(st.session_state.last_result)


def main() -> None:
    st.set_page_config(
        page_title="Atlas — Enterprise Knowledge Ops",
        page_icon="🧭",
        layout="wide",
    )

    defaults = {
        "messages": [],
        "last_result": None,
        "session_history": [],
        "query_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _ensure_session_active()

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
