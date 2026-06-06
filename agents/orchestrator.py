"""OrchestratorAgent — classifies query intent and produces a plan.

The orchestrator does TWO things, in order:
  1. Classify the user query into a QueryIntent
       (single_document | cross_document | comparison | corpus_summary).
     The intent drives which retrieval strategy the RetrieverAgent uses.
  2. Produce a numbered plan of agent steps. Always includes
     [RETRIEVE] -> [ANALYZE] -> [VERIFY] in that order; the orchestrator
     can add multiple [RETRIEVE] steps when the query spans multiple
     sub-topics or doc types.

The orchestrator does NOT call any other agent. It returns a dict:
    {
      "intent": str,
      "target_doc_types": List[str],   # e.g. ["resume"] for single_document
      "plan": List[str],
    }

`graph/workflow.py` reads the dict and dispatches the actual nodes.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from lib.query_rewriter import _extract_text

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"), override=True)

log = logging.getLogger("orchestrator")

DEFAULT_MODEL = "gemini-flash-latest"

VALID_INTENTS = ("single_document", "cross_document", "comparison", "corpus_summary")

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


# --------------------------------------------------------------------------- #
# Heuristic intent detection (no LLM cost for the common cases)
# --------------------------------------------------------------------------- #

_SUMMARY_TERMS = re.compile(
    r"\b("
    r"summari[sz]e|summary|describe|overview|recap|outline|"
    r"highlights?|main points?|give (?:me )?an overview|"
    r"what (?:are|is in)(?: these| those)?|what's in"
    r")\b",
    re.IGNORECASE,
)

_CORPUS_OVERVIEW_TERMS = re.compile(
    r"\b("
    r"all (?:the )?(?:uploaded|loaded|indexed|docs?|documents?|files?)|"
    r"the (?:whole|entire|full) (?:corpus|docs?|documents?)|"
    r"give (?:me )?(?:a )?(?:high[_\- ]?level|whole[_\- ]?corpus|overall) (?:overview|summary)|"
    r"corpus[_\- ]?summary|"
    r"everything (?:uploaded|indexed)"
    r")\b",
    re.IGNORECASE,
)

_COMPARE_TERMS = re.compile(
    r"\b("
    r"compar(?:e|ison|ing)|contrast|differen[ct]e?|versus|vs\.?|"
    r"how (?:does|do) .* differ|between .* and|"
    r"which (?:is|are) (?:better|more|less)"
    r")\b",
    re.IGNORECASE,
)

_CROSS_DOC_TERMS = re.compile(
    r"\b("
    r"both|across|between|together|alongside|"
    r"relate[sd]? to|how .* (?:relate|connect|align) (?:to|with)"
    r")\b",
    re.IGNORECASE,
)


_DOC_TYPE_KEYWORDS = {
    "resume": re.compile(r"\b(resum[ei]|profile|background)\b", re.IGNORECASE),
    "cv": re.compile(r"\b(cv|curriculum vitae)\b", re.IGNORECASE),
    "job_requirements": re.compile(r"\b(job (?:desc|description|posting|requirements?)|jd|role|position|hiring|qualifications?|requirements?|company requirements)\b", re.IGNORECASE),
    "policy": re.compile(r"\b(polic(?:y|ies)|rule|regulation)\b", re.IGNORECASE),
    "compliance": re.compile(r"\b(compliance|gdpr|hipaa|sox|regulatory)\b", re.IGNORECASE),
    "sop": re.compile(r"\b(sop|standard operating procedure)\b", re.IGNORECASE),
    "procedure": re.compile(r"\b(procedure|process|work instruction|runbook)\b", re.IGNORECASE),
    "contract": re.compile(r"\b(contract|msa|nda|terms of service)\b", re.IGNORECASE),
    "agreement": re.compile(r"\b(agreement|terms)\b", re.IGNORECASE),
    "manual": re.compile(r"\b(manual|handbook|documentation)\b", re.IGNORECASE),
    "guide": re.compile(r"\b(guide|how-to|tutorial|user guide)\b", re.IGNORECASE),
    "report": re.compile(r"\b(report|analysis|white ?paper|study)\b", re.IGNORECASE),
    "research_paper": re.compile(r"\b(research|paper|publication|arxiv|study)\b", re.IGNORECASE),
}


def _classify_intent_heuristic(query: str, available_doc_types: List[str] = None) -> str:
    """Fast rule-based intent classification.

    Doc-type-aware: if the user names one or more document types in the
    query, those drive the intent. If no doc type is named, fall back to
    verb-based heuristics (compare / cross-doc / summary terms).

    Rules:
      - 2+ doc types + compare term  -> comparison
      - 2+ doc types (no compare)    -> cross_document
      - 1 doc type (any verb)        -> single_document
      - 0 doc types + corpus overview -> corpus_summary
      - 0 doc types + compare term    -> comparison
      - 0 doc types + cross term      -> cross_document
      - 0 doc types + summary verb    -> corpus_summary
      - 0 doc types (default)         -> single_document
    """
    q = (query or "").strip()
    if not q:
        return "corpus_summary"

    targets = _extract_target_doc_types(q, available_doc_types or [])

    if len(targets) >= 2:
        if _COMPARE_TERMS.search(q):
            return "comparison"
        return "cross_document"
    if len(targets) == 1:
        return "single_document"

    if _CORPUS_OVERVIEW_TERMS.search(q):
        return "corpus_summary"
    if _COMPARE_TERMS.search(q):
        return "comparison"
    if _CROSS_DOC_TERMS.search(q):
        return "cross_document"
    if _SUMMARY_TERMS.search(q):
        return "corpus_summary"
    return "single_document"


def _extract_target_doc_types(query: str, available_doc_types: List[str]) -> List[str]:
    """Find which doc types the query is asking about. Empty list = no filter."""
    q = (query or "")
    found: List[str] = []
    for doc_type, pattern in _DOC_TYPE_KEYWORDS.items():
        if doc_type in available_doc_types and pattern.search(q):
            if doc_type not in found:
                found.append(doc_type)
    return found


# --------------------------------------------------------------------------- #
# LLM-based intent classification (fallback for tricky queries)
# --------------------------------------------------------------------------- #

INTENT_PROMPT = """You are a query-intent classifier for an enterprise RAG system.

Given a user query, classify it into EXACTLY one of these intents:

  - single_document: the query is about ONE specific document or topic
                     (e.g. "tell me about the resume", "what does the policy say about X?")
  - cross_document:  the query asks for information from multiple documents,
                     not necessarily comparing them
                     (e.g. "what do the resume and the contract say about X?")
  - comparison:      the query asks to compare, contrast, or relate two
                     or more documents
                     (e.g. "compare the resume with the company requirements")
  - corpus_summary:  the query asks for a high-level summary of the
                     whole uploaded corpus
                     (e.g. "summarize the docs", "give me an overview")

Reply with ONLY one of these four words, nothing else:
single_document
cross_document
comparison
corpus_summary
"""


def _classify_intent_llm(query: str) -> str:
    try:
        msg = _get_llm().invoke(f"{INTENT_PROMPT}\n\nUser query: {query}")
        text = _extract_text(msg).strip().lower()
        for intent in VALID_INTENTS:
            if intent in text:
                return intent
    except Exception as e:
        log.warning("LLM intent classification failed: %s", e)
    return "single_document"


def classify_intent(query: str, available_doc_types: List[str] = None, use_llm_fallback: bool = True) -> str:
    """Classify the query into one of VALID_INTENTS.

    Strategy: doc-type-aware heuristic first. The LLM is only consulted
    when the heuristic has low confidence (no doc type was named AND the
    query is ambiguous), so the common case ("tell me about the resume",
    "summarize the policy") never pays the LLM cost.

    When the heuristic already pinned intent via doc-type detection
    (single_document with one named type, or comparison / cross_document
    with two named types), we trust it and skip the LLM entirely.
    """
    heuristic = _classify_intent_heuristic(query, available_doc_types)
    if not use_llm_fallback:
        return heuristic

    targets = _extract_target_doc_types(query, available_doc_types or [])
    if targets:
        return heuristic

    try:
        llm_intent = _classify_intent_llm(query)
        if llm_intent in VALID_INTENTS:
            return llm_intent
    except Exception:
        pass
    return heuristic


# --------------------------------------------------------------------------- #
# Plan generation
# --------------------------------------------------------------------------- #

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


def _make_plan(intent: str, target_doc_types: List[str]) -> List[str]:
    """Build a deterministic plan from the intent. Used as the
    canonical baseline; the LLM may add [MEMORY] or expand
    [RETRIEVE] steps for multi-topic queries."""
    if intent == "comparison":
        return [
            "[RETRIEVE] chunks from the first named document/topic",
            "[RETRIEVE] chunks from the second named document/topic",
            "[ANALYZE] synthesize a comparison answering the query",
            "[VERIFY] confirm both sides of the comparison are grounded",
        ]
    if intent == "cross_document":
        return [
            "[RETRIEVE] chunks from the relevant documents",
            "[ANALYZE] synthesize a cross-document answer",
            "[VERIFY] confirm every cited claim is grounded in at least one source",
        ]
    if intent == "corpus_summary":
        return [
            "[RETRIEVE] a representative chunk from each uploaded document",
            "[ANALYZE] synthesize a corpus-wide summary",
            "[VERIFY] confirm the summary reflects every uploaded document",
        ]
    # single_document
    return [
        "[RETRIEVE] chunks from the targeted document",
        "[ANALYZE] synthesize an answer focused on the targeted document",
        "[VERIFY] confirm the answer is grounded in the retrieved chunks",
    ]


def plan(
    query: str,
    session_context: str = "",
    available_doc_types: List[str] = None,
) -> Dict[str, Any]:
    """Classify the intent, extract target doc types, and produce a plan.

    Returns a dict with keys: intent, target_doc_types, plan.
    """
    available_doc_types = available_doc_types or []

    intent = classify_intent(query, available_doc_types=available_doc_types)
    target_doc_types = _extract_target_doc_types(query, available_doc_types)

    # Try to refine the plan with the LLM (it can add [MEMORY] or
    # extra [RETRIEVE] steps for multi-topic queries). Fall back to
    # the deterministic intent-based plan on any error.
    user_msg = f"User query: {query}"
    if session_context.strip():
        user_msg += f"\n\nPrior session context:\n{session_context}"
    user_msg += (
        f"\n\nDetected intent: {intent}"
        f"\nTarget document types: {', '.join(target_doc_types) or '(none)'}"
        "\n\nProduce the numbered plan:"
    )
    steps: List[str] = []
    try:
        msg = _get_llm().invoke(f"{ORCHESTRATOR_PROMPT}\n\n{user_msg}")
        steps = _parse_numbered_list(_extract_text(msg))
    except Exception:
        log.exception("Orchestrator LLM call failed; using deterministic plan")

    if not steps or not all(
        _has_tag(steps, t) for t in ("[RETRIEVE]", "[ANALYZE]", "[VERIFY]")
    ):
        steps = _make_plan(intent, target_doc_types)
        if not _has_tag(steps, "[RETRIEVE]"):
            steps.insert(0, "[RETRIEVE] find chunks relevant to the query")
        if not _has_tag(steps, "[ANALYZE]"):
            steps.append("[ANALYZE] synthesize an answer from the retrieved chunks")
        if not _has_tag(steps, "[VERIFY]"):
            steps.append("[VERIFY] check that the answer is grounded in the chunks")

    log.info(
        "Orchestrator: intent=%s, target_doc_types=%s, plan=%d step(s)",
        intent, target_doc_types, len(steps),
    )
    for i, s in enumerate(steps, 1):
        log.info("  %d. %s", i, s)

    return {
        "intent": intent,
        "target_doc_types": target_doc_types,
        "plan": steps,
    }
