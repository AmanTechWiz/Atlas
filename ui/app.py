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

STUB_ENABLED = True

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
    st.markdown(confidence_badge(result["verification_result"]["confidence"]), unsafe_allow_html=True)
    st.markdown("### Final Answer")
    st.markdown(result["final_answer"])

    if result["retrieved_chunks"]:
        unique_sources = sorted({c["source"] for c in result["retrieved_chunks"]})
        st.markdown(
            "**Sources:** " + " · ".join(f"`{s}`" for s in unique_sources)
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
    st.markdown("### Full Evaluation Log (raw JSON)")
    payload = {k: v for k, v in result.items() if k != "session_history"}
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
        emb = os.getenv("GEMINI_EMBEDDING_MODEL", "not set")
        st.markdown(f"- **LLM:** `{llm}`")
        st.markdown(f"- **Embeddings:** `{emb}`")

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

    if STUB_ENABLED:
        st.warning(
            "UI shell is wired to **stubbed data** (STUB_ENABLED=True). "
            "The real US 1 backend (Retriever + Analyst + Orchestrator + LangGraph) "
            "will replace the stub in the next step. Click 'Run a sample query' below "
            "to preview the full UX."
        )

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
        with st.spinner("Running agent pipeline..."):
            result = stub_run_query(query)
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
        st.session_state.query_input = ""

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
