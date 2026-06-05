"""Input validation and output guardrails for the knowledge-ops pipeline.

Satisfies Official US 5 (Governance & Guardrails) and Story 8 of agents.md.

Two public functions:

- `validate_input(query)` - run BEFORE the graph to reject:
    * empty / None queries
    * queries shorter than 5 characters
    * queries containing common prompt-injection patterns
    * queries that have no enterprise-document topic keywords (out of scope)

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
    r"bypass (?:safety|guardrail|filter|restriction)",
    r"override (?:safety|guardrail|filter|restriction)",
    r"reveal (?:your|the) (?:system|hidden) prompt",
    r"show (?:your|the) (?:system|hidden) prompt",
]

ENTERPRISE_KEYWORDS: List[str] = [
    "policy", "policies", "handbook", "manual", "procedure", "sop",
    "compliance", "audit", "regulatory", "governance",
    "security", "secure", "mfa", "2fa", "two-factor", "password", "encryption",
    "access", "permission", "authorization", "authentication",
    "data", "retention", "privacy", "confidential", "pii",
    "training", "vendor", "third-party", "third party",
    "employee", "employees", "staff", "worker", "personnel", "hr",
    "human resources", "manager", "supervisor", "team lead",
    "leave", "pto", "vacation", "holiday", "sick", "bereavement",
    "parental", "maternity", "paternity", "fmla",
    "attendance", "conduct", "dress code", "dress", "benefit", "benefits",
    "compensation", "salary", "wage", "payroll", "bonus",
    "onboarding", "onboard", "hire", "hired", "newhire", "new hire",
    "orientation", "first day", "first week", "first 30 days",
    "equipment", "laptop", "workstation", "workspace",
    "mentor", "mentorship", "buddy",
    "company", "organization", "organisation", "enterprise", "corporate",
    "deadline", "requirement", "mandatory", "must", "shall",
    "approval", "approve", "approved", "sign-off", "signoff",
    "incident", "breach", "violation",
]

_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)
_KEYWORDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ENTERPRISE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


DISCLAIMER = (
    "**Low confidence — answer may not be fully supported by the source "
    "documents.** Treat the information above as provisional and verify "
    "against the cited sources before acting on it."
)


def validate_input(query: Any) -> Dict[str, str]:
    """Validate a user query before it enters the graph.

    Returns `{"valid": bool, "reason": str}`. `reason` is a short
    human-readable explanation of why the query was rejected (empty
    string when valid).
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
    if _INJECTION_RE.search(cleaned):
        return {
            "valid": False,
            "reason": (
                "Query contains a prompt-injection pattern and was rejected. "
                "Please rephrase as a plain business question."
            ),
        }
    if not _KEYWORDS_RE.search(cleaned):
        return {
            "valid": False,
            "reason": (
                "Query does not appear to be related to enterprise documents "
                "(HR, compliance, onboarding, or policy topics). Please ask a "
                "question about company policies, procedures, or compliance."
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
