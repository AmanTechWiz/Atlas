"""LangGraph workflow wiring all agents together.

Official US 1+2 flow:
  START -> orchestrate -> retrieve -> analyze -> finalize -> END

US 1: real RAG pipeline (orchestrate, retrieve, analyze, finalize).
US 2: every node calls EvalLogger to write a structured JSON entry
      to logs/eval_<session_id>.json. run_query() writes a SUMMARY
      entry at the end with the final state.

US 3 will insert a Verifier node after `analyze` (with a conditional
edge for low-confidence answers). US 5 will insert Guardrails before
`orchestrate`. The current graph is the US 1+2 vertical slice.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.analyst import analyze
from agents.orchestrator import plan
from agents.retriever import retrieve
from evaluation.logger import EvalLogger

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

log = logging.getLogger("workflow")


class AgentState(TypedDict, total=False):
    query: str
    plan: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str
    verification_result: Dict[str, Any]
    final_answer: str
    decision_trace: List[str]
    session_history: List[Dict[str, Any]]
    error: Optional[str]
    eval_logger: EvalLogger
    query_start_mono: float


def _trace(state: AgentState, line: str) -> List[str]:
    return list(state.get("decision_trace") or []) + [line]


def _ok_node(state: AgentState, node_name: str, result: Dict[str, Any], detail: str) -> Dict[str, Any]:
    result = dict(result)
    result["decision_trace"] = _trace(state, f"{node_name}: {detail}")
    return result


def _get_logger(state: AgentState) -> Optional[EvalLogger]:
    return state.get("eval_logger")


def orchestrate_node(state: AgentState) -> Dict[str, Any]:
    elog = _get_logger(state)
    try:
        plan_steps = plan(state["query"], session_context="")
        if elog is not None:
            elog.log_plan(plan_steps)
        return _ok_node(
            state, "ORCHESTRATOR",
            {"plan": plan_steps},
            f"produced {len(plan_steps)}-step plan",
        )
    except Exception as e:
        log.exception("Orchestrator failed")
        if elog is not None:
            elog.log_failure(str(e), "ORCHESTRATOR")
        return _ok_node(
            state, "ORCHESTRATOR",
            {"plan": [], "error": f"orchestrator_failed: {e}"},
            f"FAILED — {e}",
        )


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    elog = _get_logger(state)
    try:
        chunks = retrieve(state["query"], k=5)
        sources = sorted({c["source"] for c in chunks})
        detail = (
            f"{len(chunks)} chunk(s) from {len(sources)} source(s): "
            f"{', '.join(sources) if sources else '(none)'}"
        )
        if elog is not None:
            elog.log_retrieval(chunks)
        return _ok_node(state, "RETRIEVER", {"retrieved_chunks": chunks}, detail)
    except Exception as e:
        log.exception("Retriever failed")
        if elog is not None:
            elog.log_failure(str(e), "RETRIEVER")
        return _ok_node(
            state, "RETRIEVER",
            {"retrieved_chunks": [], "error": f"retriever_failed: {e}"},
            f"FAILED — {e}",
        )


def analyze_node(state: AgentState) -> Dict[str, Any]:
    elog = _get_logger(state)
    try:
        draft = analyze(state["query"], state.get("retrieved_chunks") or [])
        if elog is not None:
            elog.log_analysis(draft)
        return _ok_node(
            state, "ANALYST",
            {"draft_answer": draft},
            f"synthesized draft ({len(draft)} chars)",
        )
    except Exception as e:
        log.exception("Analyst failed")
        if elog is not None:
            elog.log_failure(str(e), "ANALYST")
        return _ok_node(
            state, "ANALYST",
            {"draft_answer": "", "error": f"analyst_failed: {e}"},
            f"FAILED — {e}",
        )


def _extract_answer_section(draft: str) -> str:
    if "[Answer]" not in draft:
        return draft.strip()
    start = draft.find("[Answer]") + len("[Answer]")
    end = draft.find("[Sources Used]", start)
    if end == -1:
        end = len(draft)
    return draft[start:end].strip()


def finalize_node(state: AgentState) -> Dict[str, Any]:
    elog = _get_logger(state)
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks") or []
    answer_body = _extract_answer_section(draft)

    if chunks:
        unique_sources = sorted({c["source"] for c in chunks})
        footer = "\n\n---\n**Sources:** " + " · ".join(f"`{s}`" for s in unique_sources)
    else:
        footer = "\n\n---\n**Sources:** (none — no relevant chunks were retrieved)"

    verification = {
        "confidence": 1.0,
        "grounded": True,
        "flags": [
            "VERIFIER_NOT_IMPLEMENTED_YET — Official US 3 will add the real grounding check"
        ],
    }
    if elog is not None:
        elog.log_verification(verification)

    return _ok_node(
        state, "FINALIZE",
        {
            "final_answer": answer_body + footer,
            "verification_result": verification,
        },
        "assembled final answer with sources footer",
    )


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("orchestrate", orchestrate_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("analyze", analyze_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "orchestrate")
    g.add_edge("orchestrate", "retrieve")
    g.add_edge("retrieve", "analyze")
    g.add_edge("analyze", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


app = build_graph()


def run_query(query: str) -> AgentState:
    """Run the full US 1+2 pipeline. Returns the populated AgentState.

    Side effect: writes a structured JSON log to logs/eval_<session_id>.json
    with one entry per agent stage plus a final SUMMARY entry.
    """
    eval_logger = EvalLogger()
    eval_logger.log_query_start(query)

    initial: AgentState = {
        "query": query,
        "plan": [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "verification_result": {},
        "final_answer": "",
        "decision_trace": [],
        "session_history": [],
        "error": None,
        "eval_logger": eval_logger,
        "query_start_mono": time.monotonic(),
    }
    result = app.invoke(initial)

    elapsed_ms = (time.monotonic() - result.get("query_start_mono", time.monotonic())) * 1000.0
    final_answer = result.get("final_answer", "") or ""
    plan_steps = result.get("plan") or []
    chunks = result.get("retrieved_chunks") or []
    verification = result.get("verification_result") or {}

    eval_logger.log_final(final_answer, elapsed_ms)
    eval_logger.log_summary(
        query=query,
        plan=plan_steps,
        retrieved_chunks=chunks,
        verification_result=verification,
        final_answer=final_answer,
        total_time_ms=elapsed_ms,
    )

    log.info(
        "run_query complete: %d plan step(s), %d chunk(s), final_answer %d chars, %d ms, error=%s, log=%s",
        len(plan_steps),
        len(chunks),
        len(final_answer),
        int(elapsed_ms),
        result.get("error"),
        eval_logger.log_path,
    )
    return result
