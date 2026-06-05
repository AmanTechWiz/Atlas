"""Input validation and output guardrails for the knowledge-ops pipeline.

Satisfies Official US 5 (Governance & Guardrails) and Story 8 of agents.md.

The guardrails are CORPUS-AGNOSTIC — they do not assume the user has uploaded
HR/compliance/onboarding documents. They run basic safety checks (length,
injection detection, coherence, spam) that apply to any enterprise document
set (legal, finance, IT, HR, operations, etc.).

Two public functions:

- `validate_input(query, corpus_size=0)` - run BEFORE the graph to reject:
    * empty / None queries
    * queries shorter than `MIN_QUERY_LENGTH` characters
    * queries longer than `MAX_QUERY_LENGTH` characters
    * queries with > `MAX_SPECIAL_CHAR_RATIO` non-alphanumeric characters
    * queries with > `MAX_REPETITION_RATIO` repeated tokens
    * queries containing common prompt-injection patterns
    * queries against an empty corpus (corpus_size=0)

  Returns `{"valid": bool, "reason": str}`. When `valid=False`, the graph
  should not run; `run_query()` returns early with a friendly rejection in
  `final_answer` and a `GUARDRAIL_REJECTED` log entry.

- `apply_confidence_guardrail(verification_result, answer, chunks)` - run
  INSIDE `finalize_node` to wrap the answer with:
    * a "Low confidence" disclaimer when `verification_result.confidence < 0.6`
    * a "Sources" footer citing every source file actually used

  Returns the augmented answer string. Replaces the inline disclaimer+footer
  logic that was previously in `graph/workflow.py`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


MIN_QUERY_LENGTH = 5
MAX_QUERY_LENGTH = 2000
MAX_SPECIAL_CHAR_RATIO = 0.50
MAX_REPETITION_RATIO = 0.60

INJECTION_PATTERNS: List[str] = [
    r"ignore (?:all )?(?:previous|prior|above) instructions",
    r"disregard (?:all )?(?:previous|prior|above)",
    r"forget (?:all )?(?:previous|prior|above)",
    r"you are now",
    r"act as (?:a|an)",
    r"pretend (?:to be|you are)",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"###\s*(?:system|assistant|instruction)",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"\bdeveloper mode\b",
    r"bypass (?:[^.\n]{0,30})?(?:safety|guardrail|filter|restriction)",
    r"override (?:[^.\n]{0,30})?(?:safety|guardrail|filter|restriction)",
    r"reveal (?:your|the) (?:system|hidden) prompt",
    r"show (?:your|the) (?:system|hidden) prompt",
    r"\bprompt\s+injection\b",
    r"execute (?:the following|this) (?:code|script|command)",
    r"\bcurl\s+[^\s]+\s*\|\s*sh",
]

_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_WS_RE = re.compile(r"\s+")


def _special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = len(_ALNUM_RE.findall(text))
    ws = len(_WS_RE.findall(text))
    meaningful = alnum + ws
    if meaningful == 0:
        return 1.0
    return 1.0 - (meaningful / len(text))


def _repetition_ratio(text: str) -> float:
    words = _WORD_RE.findall(text.lower())
    if len(words) < 3:
        return 0.0
    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    most_common_count = max(counts.values())
    return most_common_count / len(words)


DISCLAIMER = (
    "**Low confidence — answer may not be fully supported by the source "
    "documents.** Treat the information above as provisional and verify "
    "against the cited sources before acting on it."
)


def validate_input(query: Any, corpus_size: int = 0) -> Dict[str, str]:
    """Validate a user query before it enters the graph.

    Returns `{"valid": bool, "reason": str}`. `reason` is a short
    human-readable explanation of why the query was rejected (empty
    string when valid).

    `corpus_size` is the number of chunks currently in the user's vector
    store. A value of 0 means no documents have been uploaded yet, in
    which case the query is rejected with a request to upload documents.
    """
    if query is None:
        return {"valid": False, "reason": "Query is empty."}
    if not isinstance(query, str):
        return {"valid": False, "reason": "Query must be a string."}
    cleaned = query.strip()
    if not cleaned:
        return {"valid": False, "reason": "Query is empty."}

    if len(cleaned) < MIN_QUERY_LENGTH:
        return {
            "valid": False,
            "reason": f"Query is too short (need at least {MIN_QUERY_LENGTH} characters).",
        }
    if len(cleaned) > MAX_QUERY_LENGTH:
        return {
            "valid": False,
            "reason": f"Query is too long (max {MAX_QUERY_LENGTH} characters).",
        }

    special_ratio = _special_char_ratio(cleaned)
    if special_ratio > MAX_SPECIAL_CHAR_RATIO:
        return {
            "valid": False,
            "reason": (
                "Query contains too many special characters (likely spam or "
                "garbled text). Please rephrase as a plain business question."
            ),
        }

    rep_ratio = _repetition_ratio(cleaned)
    if rep_ratio > MAX_REPETITION_RATIO:
        return {
            "valid": False,
            "reason": (
                "Query contains excessive repetition. Please rephrase as a "
                "clear, distinct question."
            ),
        }

    if _INJECTION_RE.search(cleaned):
        return {
            "valid": False,
            "reason": (
                "Query contains a prompt-injection pattern and was rejected. "
                "Please rephrase as a plain business question."
            ),
        }

    if corpus_size <= 0:
        return {
            "valid": False,
            "reason": (
                "Your knowledge base is empty. Please upload at least one "
                "document (PDF, DOCX, or TXT) using the sidebar before "
                "asking a question."
            ),
        }

    return {"valid": True, "reason": ""}


def apply_confidence_guardrail(
    verification_result: Dict[str, Any],
    answer: str,
    chunks: List[Dict[str, Any]],
    confidence_threshold: float = 0.6,
) -> str:
    """Wrap the answer with a low-confidence disclaimer and a sources footer.

    - Prepends `DISCLAIMER` to the answer when `verification_result.confidence
      < confidence_threshold` (default 0.6 per Story 4 of agents.md).
    - Appends a `**Sources:**` footer listing the unique source files used
      (or "(none — no relevant chunks were retrieved)" when chunks is empty).

    Returns the augmented answer string. The disclaimer is a no-op when
    confidence >= threshold, and the footer is always appended.
    """
    answer = (answer or "").rstrip()
    confidence = float((verification_result or {}).get("confidence", 1.0) or 0.0)

    if chunks:
        unique_sources = sorted({c.get("source", "unknown") for c in chunks})
        sources_line = " · ".join(f"`{s}`" for s in unique_sources)
    else:
        sources_line = "(none — no relevant chunks were retrieved)"
    footer = "\n\n---\n**Sources:** " + sources_line

    if confidence < confidence_threshold:
        return DISCLAIMER + "\n\n" + answer + footer
    return answer + footer
