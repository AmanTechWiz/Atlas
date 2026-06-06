"""EvalLogger — structured JSON evaluation log per query.

Each query produces a single JSON file at logs/eval_<timestamp>.json
containing an array of timestamped entries. Each entry has:
    - timestamp (ISO 8601 UTC)
    - stage (QUERY_START | ORCHESTRATION | RETRIEVAL | ANALYSIS |
            VERIFICATION | FINAL | FAILURE | GUARDRAIL | SUMMARY)
    - event (one-line human-readable description)
    - data (stage-specific payload)

The logger is fire-and-forget: if writing the log file fails, the
workflow still returns its answer. Failures are emitted to the Python
logging module under the "eval_logger" logger.

The single-corpus design dropped `session_id` from the log filename —
there is one corpus, not many. A high-resolution UTC timestamp keeps
log filenames unique per query, and the inner `query` field of the
SUMMARY entry remains the human-meaningful handle.

Story 9 (agents.md) acceptance criteria:
- Every query produces a log file
- Log contains entries for every agent stage
- Failures are logged with stage name and error message
- Logs are valid JSON (parseable)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("eval_logger")

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_log_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")


class EvalLogger:
    """Per-query structured JSON logger.

    Usage:
        elog = EvalLogger()
        elog.log_query_start(query)
        ...
        elog.log_summary(...)
    """

    def __init__(self) -> None:
        self.log_id = _now_log_id()
        self.started_at_iso = _now_iso()
        self.log_path = LOGS_DIR / f"eval_{self.log_id}.json"
        self.entries: List[Dict[str, Any]] = []
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("Could not create logs dir %s: %s", LOGS_DIR, e)

    def _append(self, stage: str, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp": _now_iso(),
            "stage": stage,
            "event": event,
            "data": data or {},
        }
        self.entries.append(entry)
        self._flush()

    def _flush(self) -> None:
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
        except OSError as e:
            log.warning("Could not write log file %s: %s", self.log_path, e)

    def log_query_start(self, query: str) -> None:
        self._append(
            "QUERY_START",
            f"query received ({len(query)} chars)",
            {"query": query},
        )

    def log_plan(self, plan: List[str]) -> None:
        self._append(
            "ORCHESTRATION",
            f"plan with {len(plan)} step(s)",
            {"plan": plan},
        )

    def log_retrieval(
        self,
        chunks: List[Dict[str, Any]],
        query_used: Optional[str] = None,
    ) -> None:
        sources = sorted({c.get("source", "<unknown>") for c in chunks})
        scores = [round(float(c.get("relevance_score", 0.0) or 0.0), 4) for c in chunks]
        data: Dict[str, Any] = {
            "chunk_count": len(chunks),
            "sources": sources,
            "scores": scores,
            "chunks": [
                {
                    "text_preview": (c.get("text") or "")[:200],
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "relevance_score": c.get("relevance_score"),
                }
                for c in chunks
            ],
        }
        if query_used is not None:
            data["query_used"] = query_used
        self._append(
            "RETRIEVAL",
            f"retrieved {len(chunks)} chunk(s) from {len(sources)} source(s)",
            data,
        )

    def log_analysis(self, draft_answer: str) -> None:
        self._append(
            "ANALYSIS",
            f"draft answer ({len(draft_answer)} chars)",
            {"draft_answer": draft_answer},
        )

    def log_verification(self, result: Dict[str, Any]) -> None:
        self._append(
            "VERIFICATION",
            f"confidence={result.get('confidence')}, grounded={result.get('grounded')}",
            {
                "confidence": result.get("confidence"),
                "grounded": result.get("grounded"),
                "flags": result.get("flags", []),
                "grounding_confidence": result.get("grounding_confidence"),
                "answer_quality": result.get("answer_quality"),
                "retrieval_confidence": result.get("retrieval_confidence"),
                "claims": result.get("claims", []),
                "question_aspects": result.get("question_aspects", []),
                "conflicts": result.get("conflicts", []),
            },
        )

    def log_final(self, final_answer: str, total_time_ms: float) -> None:
        self._append(
            "FINAL",
            f"final answer ({len(final_answer)} chars, {total_time_ms:.0f}ms)",
            {
                "final_answer": final_answer,
                "total_time_ms": total_time_ms,
            },
        )

    def log_failure(self, error: str, stage: str) -> None:
        self._append(
            "FAILURE",
            f"stage={stage} — {error}",
            {"error": error, "stage": stage},
        )

    def log_guardrail_rejection(self, query: str, reason: str) -> None:
        self._append(
            "GUARDRAIL",
            f"query rejected by input validation: {reason}",
            {"query": query, "reason": reason, "rejected": True},
        )

    def log_rewrite(
        self,
        original: str,
        rewritten: str,
        had_memory: bool,
    ) -> None:
        self._append(
            "QUERY_REWRITE",
            "rewrote" if rewritten != original else "kept original",
            {
                "original": original,
                "rewritten": rewritten,
                "rewritten_changed": rewritten != original,
                "had_memory": had_memory,
            },
        )

    def log_summary(
        self,
        query: str,
        plan: List[str],
        retrieved_chunks: List[Dict[str, Any]],
        verification_result: Dict[str, Any],
        final_answer: str,
        total_time_ms: float,
    ) -> None:
        self._append(
            "SUMMARY",
            "query complete",
            {
                "query": query,
                "plan": plan,
                "retrieval_count": len(retrieved_chunks),
                "sources": sorted({c.get("source", "<unknown>") for c in retrieved_chunks}),
                "confidence": verification_result.get("confidence"),
                "grounded": verification_result.get("grounded"),
                "grounding_confidence": verification_result.get("grounding_confidence"),
                "answer_quality": verification_result.get("answer_quality"),
                "retrieval_confidence": verification_result.get("retrieval_confidence"),
                "flags": verification_result.get("flags", []),
                "claim_count": len(verification_result.get("claims", [])),
                "aspect_count": len(verification_result.get("question_aspects", [])),
                "conflict_count": len(verification_result.get("conflicts", [])),
                "conflicts": verification_result.get("conflicts", []),
                "final_answer": final_answer,
                "total_time_ms": total_time_ms,
            },
        )
