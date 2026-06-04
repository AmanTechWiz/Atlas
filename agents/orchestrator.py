"""OrchestratorAgent — decomposes a query into a numbered plan of agent steps.

Acceptance criteria (agents.md Story 6):
- Plan always includes at least RETRIEVE -> ANALYZE -> VERIFY sequence
- Plan is logged and visible in decision trace
- Planning step is distinct from execution (separation of concerns)

The orchestrator does NOT call any agents directly. It returns a plan;
graph/workflow.py is responsible for actually invoking the agents.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"), override=True)

log = logging.getLogger("orchestrator")

DEFAULT_MODEL = "gemini-flash-latest"

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
        )
    return _llm


ORCHESTRATOR_PROMPT = """You are the OrchestratorAgent in an enterprise Knowledge Operations system.

Given a user query (and optional prior session context), produce a numbered plan
of steps that another system will execute. Each step must start with exactly one
of these tags:

  [RETRIEVE]  — query the vector store for relevant chunks
  [ANALYZE]   — reason across retrieved chunks to form an answer
  [VERIFY]    — check that the answer is grounded in the chunks
  [MEMORY]    — update or read session memory

Rules:
- Always include at least one [RETRIEVE], one [ANALYZE], and one [VERIFY] in that order.
- If the query clearly has distinct subtopics that need separate retrievals
  (e.g. "compare the X policy with the Y procedure"), add multiple [RETRIEVE] steps.
- Output ONLY the numbered list, one step per line. No preamble, no explanation.
"""


DEFAULT_PLAN = [
    "[RETRIEVE] find chunks relevant to the query",
    "[ANALYZE] synthesize an answer from the retrieved chunks",
    "[VERIFY] check that the answer is grounded in the chunks",
]


def _parse_numbered_list(raw: str) -> List[str]:
    steps: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if line:
            steps.append(line)
    return steps


def _has_tag(steps: List[str], tag: str) -> bool:
    return any(tag in s for s in steps)


def plan(query: str, session_context: str = "") -> List[str]:
    """Decompose the query into a plan of agent steps."""
    user_msg = f"User query: {query}"
    if session_context.strip():
        user_msg += f"\n\nPrior session context:\n{session_context}"
    user_msg += "\n\nProduce the numbered plan:"

    try:
        llm = _get_llm()
        msg = llm.invoke(f"{ORCHESTRATOR_PROMPT}\n\n{user_msg}")
        steps = _parse_numbered_list(msg.text)
    except Exception as e:
        log.exception("Orchestrator LLM call failed; falling back to default plan")
        return DEFAULT_PLAN

    if not _has_tag(steps, "[RETRIEVE]") or not _has_tag(steps, "[ANALYZE]") or not _has_tag(steps, "[VERIFY]"):
        log.warning(
            "Orchestrator plan missing required tags; appending defaults. Got: %s",
            steps,
        )
        for required in ("[RETRIEVE]", "[ANALYZE]", "[VERIFY]"):
            if not _has_tag(steps, required):
                steps.append(f"{required} (default — added by orchestrator safety check)")

    log.info("Orchestrator plan (%d step(s)):", len(steps))
    for i, s in enumerate(steps, 1):
        log.info("  %d. %s", i, s)
    return steps
