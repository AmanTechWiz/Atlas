"""Streamlit UI for the Enterprise Knowledge Ops Agent.

This is the UI shell for Official US 4 (Explainability & Transparency).
The data shown in the 4 response tabs is currently a STUB
(`stub_run_query`) so the user can validate the UX shape before the real
backend is wired in. The stub is replaced by the real
`graph.workflow.run_query` once US 1 lands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INGEST_SCRIPT = PROJECT_ROOT / "vector_store" / "ingest.py"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
LOGS_DIR = PROJECT_ROOT / "logs"

STUB_ENABLED = False

try:
    from graph.workflow import run_query
except Exception as _e:  # pragma: no cover
    run_query = None
    _IMPORT_ERROR = repr(_e)
else:
    _IMPORT_ERROR = None

SAMPLE_QUERIES = [
    "What is the parental leave policy and how does it interact with the onboarding timeline for new parents in their first 30 days?",
    "What are the MFA requirements and how quickly must a security incident be reported?",
    "Compare the PTO policy with the sick leave policy. Which one carries over and which one does not?",
]


def stub_run_query(query: str) -> dict[str, Any]:
    chunks = [
        {
            "text": (
                "Primary caregivers are entitled to 16 weeks of paid parental leave. "
                "Secondary caregivers are entitled to 8 weeks of paid parental leave. "
                "Parental leave must be taken within the first 12 months of the "
                "child's birth or adoption."
            ),
            "source": "policy_hr.txt",
            "page": 1,
            "relevance_score": 0.89,
        },
        {
            "text": (
                "By the end of week 2, the new hire should be assigned a first "
                "discrete project with clear success criteria and a target "
                "completion date within the 30-day window."
            ),
            "source": "sop_onboarding.txt",
            "page": 1,
            "relevance_score": 0.81,
        },
        {
            "text": (
                "A 30-day check-in is held with the new hire, their manager, "
                "and a People Ops representative. Feedback is captured in the HRIS."
            ),
            "source": "sop_onboarding.txt",
            "page": 1,
            "relevance_score": 0.74,
        },
        {
            "text": (
                "All full-time employees are expected to work a standard 40-hour "
                "work week, typically scheduled as 8 hours per day, Monday through Friday."
            ),
            "source": "policy_hr.txt",
            "page": 1,
            "relevance_score": 0.42,
        },
    ]
    return {
        "query": query,
        "plan": [
            "[RETRIEVE] find parental leave policy in HR handbook",
            "[RETRIEVE] find onboarding timeline and 30-day milestones",
            "[ANALYZE] synthesize parental-leave entitlements with the 30-day onboarding flow",
            "[VERIFY] check that every claim is grounded in a retrieved chunk",
        ],
        "retrieved_chunks": chunks,
        "draft_answer": (
            "Reasoning: The user is asking how ACME's parental leave policy interacts with the "
            "onboarding timeline. The HR handbook (policy_hr.txt) gives the entitlements; the "
            "onboarding SOP (sop_onboarding.txt) gives the 30-day milestones. Combining the two "
            "produces a clear answer.\n\n"
            "Answer: New parents at ACME are entitled to 16 weeks of paid parental leave "
            "(primary caregivers) or 8 weeks (secondary caregivers). That leave can be taken any "
            "time within the first 12 months of the child's birth or adoption. While on leave, "
            "the parent is exempt from the standard 30-day onboarding milestones, but ACME's "
            "onboarding SOP requires a 30-day check-in with the manager and People Ops after "
            "their return.\n\n"
            "Sources Used: policy_hr.txt (page 1), sop_onboarding.txt (page 1)"
        ),
        "verification_result": {
            "confidence": 0.85,
            "grounded": True,
            "flags": [],
        },
        "final_answer": (
            "New parents at ACME are entitled to **16 weeks of paid parental leave** "
            "(primary caregivers) or **8 weeks** (secondary caregivers). That leave may be "
            "taken at any point in the **first 12 months** after birth or adoption.\n\n"
            "When a new parent returns from leave, ACME's onboarding SOP requires a "
            "**30-day check-in** with the manager and People Ops. The standard 30-day onboarding "
            "milestones (first project by end of week 2, 30-day check-in) are scheduled from the "
            "employee's start date, not from their return-from-leave date — so HR and the manager "
            "need to coordinate a tailored ramp-up plan."
        ),
        "decision_trace": [
            "ORCHESTRATOR: parsed query, produced 4-step plan",
            "RETRIEVER:   queried ChromaDB, returned 4 chunks from 2 sources (avg relevance 0.72)",
            "ANALYST:     synthesized answer citing 2 of 4 chunks (policy_hr.txt, sop_onboarding.txt)",
            "VERIFIER:    confidence=0.85, grounded=True, no flags raised",
            "FINALIZE:    appended sources footer to draft answer",
            "MEMORY:      stored Q&A in session history",
        ],
        "session_history": st.session_state.get("session_history", []),
        "error": None,
    }


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.7:
        color, label = "#1f9d55", "HIGH"
    elif confidence >= 0.5:
        color, label = "#b08800", "MEDIUM"
    else:
        color, label = "#c0392b", "LOW"
    return (
        f'<span style="background:{color};color:white;padding:4px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">'
        f"Confidence: {confidence:.2f} ({label})</span>"
    )


def render_answer_tab(result: dict[str, Any]) -> None:
    v = result.get("verification_result") or {}
    confidence = float(v.get("confidence", 0.0) or 0.0)
    grounded = bool(v.get("grounded", False))
    flags = v.get("flags", []) or []
    api_error = result.get("api_error")

    if api_error:
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

    if not api_error:
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
    st.json(v)
    if v.get("flags"):
        for f in v["flags"]:
            st.warning(f"Flag: {f}")


def render_sources_tab(result: dict[str, Any]) -> None:
    st.markdown("### Sources Used")
    if not result["retrieved_chunks"]:
        st.info("No sources retrieved.")
        return
    for i, c in enumerate(result["retrieved_chunks"], 1):
        st.markdown(
            f"**{i}.** `{c['source']}` — page {c['page']} — "
            f"relevance **{c['relevance_score']:.2f}**"
        )
        st.caption(c["text"][:300] + ("…" if len(c["text"]) > 300 else ""))


def render_eval_log_tab(result: dict[str, Any]) -> None:
    log_path = result.get("log_path")
    st.markdown("### On-Disk Evaluation Log")
    if log_path:
        st.caption(f"`{log_path}`")
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            st.success(f"Loaded {len(entries)} stage entries from disk.")

            stages = [e.get("stage", "?") for e in entries]
            st.markdown("**Timeline:** " + " → ".join(stages))

            summary_idx = next(
                (i for i, e in enumerate(entries) if e.get("stage") == "SUMMARY"), None
            )
            if summary_idx is not None:
                with st.expander("Summary entry (executive overview)"):
                    st.json(entries[summary_idx])

            with st.expander("Full on-disk log (all stages, raw JSON)"):
                st.json(entries)
        except FileNotFoundError:
            st.warning(f"Log file not found at `{log_path}`. Was the log directory moved?")
        except json.JSONDecodeError as e:
            st.error(f"Log file is not valid JSON: {e}")
    else:
        st.info("No `log_path` in result — this query was run outside `graph.workflow.run_query`.")

    st.markdown("---")
    st.markdown("### In-Memory State (live result)")
    payload = {k: v for k, v in result.items() if k not in ("session_history", "eval_logger", "query_start_mono")}
    with st.expander("In-memory state (raw Python dict)"):
        st.json(payload)


def run_ingest() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, str(INGEST_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            n = 0
            for line in result.stdout.splitlines():
                if "total chunk" in line.lower():
                    try:
                        n = int(line.split()[0])
                    except (ValueError, IndexError):
                        n = 0
            return True, f"Ingested {n} chunks." if n else "Ingestion completed."
        return False, f"Ingestion failed (exit {result.returncode}).\n{result.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return False, "Ingestion timed out after 180s."
    except Exception as e:
        return False, f"Ingestion error: {e}"


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "last_result": None,
        "session_history": [],
        "ingested": PERSIST_DIR.exists(),
        "ingest_status": "",
        "query_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Atlas")
        st.caption("Enterprise Knowledge Ops Agent")

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

        st.markdown("### Actions")
        if st.button("Ingest Documents", use_container_width=True):
            with st.spinner("Running vector_store/ingest.py ..."):
                ok, msg = run_ingest()
            st.session_state.ingested = ok
            st.session_state.ingest_status = msg
            st.toast(msg, icon="✅" if ok else "❌")

        if st.button("Reset Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.session_history = []
            st.toast("Session reset.", icon="🔄")

        if st.session_state.get("ingest_status"):
            status = st.session_state.ingest_status
            (st.success if st.session_state.ingested else st.error)(status)

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
        "Ask complex questions across your enterprise documents. "
        "The answer is grounded in the source corpus and verified before being shown."
    )

    if STUB_ENABLED or _IMPORT_ERROR:
        msg = "UI shell is wired to **stubbed data**"
        if STUB_ENABLED:
            msg += " (STUB_ENABLED=True)"
        if _IMPORT_ERROR:
            msg += f" — backend import failed: `{_IMPORT_ERROR}`"
        msg += ". The real US 1 backend will replace the stub in a future step."
        st.warning(msg)

    st.markdown("### Try a sample query")
    cols = st.columns(len(SAMPLE_QUERIES))
    for col, sample in zip(cols, SAMPLE_QUERIES):
        if col.button(
            sample[:55] + ("…" if len(sample) > 55 else ""),
            key=f"sample_{sample[:10]}",
            use_container_width=True,
        ):
            st.session_state.query_input = sample

    st.markdown("### Ask your own question")
    st.text_input(
        "Your query:",
        key="query_input",
        placeholder="e.g., What is the parental leave policy?",
        label_visibility="collapsed",
    )
    ask = st.button("Ask", type="primary", use_container_width=False)

    if ask and st.session_state.query_input.strip():
        query = st.session_state.query_input
        with st.spinner("Running agent pipeline (orchestrate → retrieve → analyze → verify → finalize)..."):
            if STUB_ENABLED or run_query is None:
                result = stub_run_query(query)
            else:
                result = run_query(query)
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
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
