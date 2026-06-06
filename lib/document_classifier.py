"""Document type classification.

Each uploaded file gets a `doc_type` label (resume, policy, contract,
etc.) stored as chunk metadata. The retriever uses this label to
boost matching documents when the user's query is about a specific
document type.

Strategy: heuristic filename + first-chunk keyword match first, then
LLM fallback for ambiguous cases. The heuristic handles the common
cases without an API call; the LLM is only consulted when neither
signal is conclusive.

Public types:
    DocType  — a string label, one of the values in `KNOWN_DOC_TYPES`
               plus `"general"` as the catch-all.

Public functions:
    classify_by_filename(filename)          -> DocType (heuristic, no LLM)
    classify_by_content(first_chunk_text)   -> DocType (heuristic, no LLM)
    classify_document(filename, first_chunk) -> DocType (heuristic,
                                                       then LLM fallback
                                                       if ambiguous)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"), override=True)

log = logging.getLogger("doc_classifier")


# --------------------------------------------------------------------------- #
# Known document types
# --------------------------------------------------------------------------- #

KNOWN_DOC_TYPES = (
    "resume",
    "cv",
    "job_requirements",
    "policy",
    "compliance",
    "sop",
    "procedure",
    "contract",
    "agreement",
    "manual",
    "guide",
    "report",
    "research_paper",
    "general",
)

# Heuristic filename keyword -> doc_type. Order matters (first match wins).
_FILENAME_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bresum[ei]\b", re.IGNORECASE), "resume"),
    (re.compile(r"\b(cv|curriculum)\b", re.IGNORECASE), "cv"),
    (re.compile(r"\b(job[_\- ]?(desc|description|requirement|posting)|jd[_\- ]?doc)\b", re.IGNORECASE), "job_requirements"),
    (re.compile(r"\b(policy|policies)\b", re.IGNORECASE), "policy"),
    (re.compile(r"\b(compliance|regulation|regulatory|gdpr|hipaa|sox)\b", re.IGNORECASE), "compliance"),
    (re.compile(r"\b(sop|standard[_\- ]?operating[_\- ]?procedure)\b", re.IGNORECASE), "sop"),
    (re.compile(r"\b(procedure|process[_\- ]?doc|work[_\- ]?instruction)\b", re.IGNORECASE), "procedure"),
    (re.compile(r"\b(contract|agreement|msa|nda|terms[_\- ]?of[_\- ]?service)\b", re.IGNORECASE), "contract"),
    (re.compile(r"\b(manual|handbook|guide|user[_\- ]?guide)\b", re.IGNORECASE), "manual"),
    (re.compile(r"\b(report|analysis|white[_\- ]?paper|study[_\- ]?doc)\b", re.IGNORECASE), "report"),
    (re.compile(r"\b(research|paper|publication|arxiv)\b", re.IGNORECASE), "research_paper"),
]

# Heuristic content keyword -> doc_type. Matched against first ~1000 chars.
_CONTENT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(experience|skills|education|projects)\b.*\b(resume|cv|amandeep|profile)\b", re.IGNORECASE | re.DOTALL), "resume"),
    (re.compile(r"\b(job description|responsibilities|qualifications|requirements|we are hiring|role:)\b", re.IGNORECASE), "job_requirements"),
    (re.compile(r"\b(must (?:comply|adhere) to|policy|employees are required|prohibited from)\b", re.IGNORECASE), "policy"),
    (re.compile(r"\b(compliance|gdpr|hipaa|sox|iso ?\d+)\b", re.IGNORECASE), "compliance"),
    (re.compile(r"\b(standard operating procedure|step ?1|step ?2|procedure:)\b", re.IGNORECASE), "sop"),
    (re.compile(r"\b(this agreement|between .* and .*|terms and conditions|whereas)\b", re.IGNORECASE), "contract"),
    (re.compile(r"\b(chapter|section|table of contents|getting started|introduction)\b.*\b(manual|guide)\b", re.IGNORECASE), "manual"),
    (re.compile(r"\b(abstract|methodology|results|conclusion|references)\b", re.IGNORECASE), "research_paper"),
]


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

def classify_by_filename(filename: str) -> Optional[str]:
    """Return a doc_type if the filename matches a known pattern, else None."""
    if not filename:
        return None
    name = Path(filename).stem
    name = name.replace("_", " ").replace("-", " ")
    for pattern, doc_type in _FILENAME_RULES:
        if pattern.search(name):
            log.info("Filename match: %r -> %s", filename, doc_type)
            return doc_type
    return None


def classify_by_content(text: str) -> Optional[str]:
    """Return a doc_type if the first chunk of text matches a known pattern."""
    if not text:
        return None
    sample = text[:1500]
    for pattern, doc_type in _CONTENT_RULES:
        if pattern.search(sample):
            log.info("Content match: %s...", sample[:60].replace("\n", " "))
            return doc_type
    return None


def classify_with_llm(text_sample: str) -> Optional[str]:
    """LLM fallback for ambiguous documents. Returns a doc_type or None.

    Uses a small, fast Gemini call. Returns None on any error.
    """
    if not text_sample or not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
        )
        prompt = (
            "Classify this document into ONE of these types: "
            "resume, cv, job_requirements, policy, compliance, sop, "
            "procedure, contract, agreement, manual, guide, report, "
            "research_paper, general.\n"
            "Reply with ONLY the single word/phrase, no explanation.\n\n"
            f"Document sample (first 1500 chars):\n{text_sample[:1500]}"
        )
        from lib.query_rewriter import _extract_text
        resp = llm.invoke(prompt)
        label = _extract_text(resp).strip().lower().split()[0].strip(".,;:")
        if label in KNOWN_DOC_TYPES:
            log.info("LLM match: %s", label)
            return label
    except Exception as e:
        log.warning("LLM classifier failed: %s", e)
    return None


def classify_document(filename: str, first_chunk_text: str = "") -> str:
    """Classify a document by filename + first chunk. Returns one of
    KNOWN_DOC_TYPES. Order: filename -> content heuristic -> LLM -> "general"."""
    by_name = classify_by_filename(filename)
    if by_name:
        return by_name
    by_content = classify_by_content(first_chunk_text)
    if by_content:
        return by_content
    by_llm = classify_with_llm(first_chunk_text)
    if by_llm:
        return by_llm
    return "general"
