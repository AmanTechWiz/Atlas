"""MemoryAgent — conversation context for multi-turn Q&A.

Stores a history of queries, answers, and the sources cited in each
turn. `get_context()` formats the most recent N turns as a
human-readable block that the OrchestratorAgent injects into its
planning prompt so multi-turn conversations can reference prior
context ("how does that compare to the policy you mentioned
earlier?").

The agent is a single in-process instance; on a fresh start call
`reset()`. The class deliberately does NOT persist to disk —
it is conversation-scoped, not document-scoped.

Public API:
    MemoryAgent(session_id="default")
        .add(query, answer, sources, timestamp=None) -> None
        .get_context(last_n=3) -> str
        .reset() -> None
        .history -> List[dict]  (read-only view)
        session_id : str        (kept for backward compat + log trace)

The in-memory dict is stored on the instance. To share across
LangGraph nodes, pass the instance through `AgentState["memory"]`.

`session_id` is optional in the new single-corpus design (no
per-session collections anymore); it is preserved as an attribute
on the instance so existing tests and audit logs keep working.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("memory")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_DEFAULT_MEMORY: Optional["MemoryAgent"] = None


def get_default_memory() -> "MemoryAgent":
    """Return a process-wide singleton MemoryAgent.

    In the single-corpus design there are no per-session ChromaDB
    collections, so a single memory instance is shared across all UI
    tabs / re-runs. `reset_knowledge_base()` in the UI calls
    `reset()` on this instance.
    """
    global _DEFAULT_MEMORY
    if _DEFAULT_MEMORY is None:
        _DEFAULT_MEMORY = MemoryAgent()
    return _DEFAULT_MEMORY


class MemoryAgent:
    """Conversation memory for multi-turn Q&A.

    A single instance holds the history. The class is intentionally
    minimal — it does not perform any LLM calls, summarization, or
    persistence. The Orchestrator consumes `get_context()` directly
    to inform planning; downstream agents (retriever, analyst, verifier)
    read prior context from `AgentState` if needed.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or "default"
        self._history: List[Dict[str, Any]] = []

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def add(
        self,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a (query, answer, sources) tuple to the session history.

        `answer` is the FINAL answer shown to the user (with the disclaimer
        and sources footer stripped — the agent cares about semantic content,
        not formatting). `sources` is the list of source filenames that
        supported the answer. `timestamp` is ISO 8601 UTC; defaults to now.
        """
        entry = {
            "query": (query or "").strip(),
            "answer": (answer or "").strip(),
            "sources": list(sources or []),
            "timestamp": timestamp or _now_iso(),
        }
        self._history.append(entry)
        log.info("Memory added turn %d (session=%s, %d source(s))",
                 len(self._history), self.session_id, len(entry["sources"]))

    def get_context(self, last_n: int = 3, max_answer_chars: int = 400) -> str:
        """Return the last `last_n` Q&A turns formatted for prompt injection.

        Each turn is rendered as:
            Q{n}: <query>
            A{n}: <answer, truncated to max_answer_chars>
            S{n}: <comma-separated source filenames>

        The whole block is returned as a single string with a leading
        `Prior session context:` header. Returns an empty string when
        there is no history yet.
        """
        if not self._history:
            return ""
        recent = self._history[-last_n:]
        lines: List[str] = ["Prior session context (most recent turns last):"]
        start_idx = max(1, len(self._history) - len(recent) + 1)
        for i, entry in enumerate(recent, start=start_idx):
            q = entry.get("query", "")
            a = entry.get("answer", "")
            if len(a) > max_answer_chars:
                a = a[:max_answer_chars].rstrip() + "..."
            sources = ", ".join(entry.get("sources") or []) or "(no sources)"
            lines.append(f"  Q{i}: {q}")
            lines.append(f"  A{i}: {a}")
            lines.append(f"  S{i}: {sources}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear the history. Called at the start of a new session."""
        log.info("Memory reset (session=%s, discarded %d turn(s))",
                 self.session_id, len(self._history))
        self._history = []

    def __len__(self) -> int:
        return len(self._history)
