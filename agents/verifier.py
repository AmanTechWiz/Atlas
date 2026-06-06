"""VerifierAgent — checks whether a draft answer is grounded in the source chunks.

This is a RAG-Triad style verifier: it asks the LLM to decompose the draft
answer into individual *claims* (each tagged direct / reasonable_inference /
absence_supported / unsupported / contradicted) and the user's question into
*question_aspects* (each tagged answered / partially_answered / not_answered).
Then three deterministic scores are computed in Python:

    grounding_confidence = avg of support weights over claims
    answer_quality      = avg of aspect weights over question_aspects
    retrieval_confidence = top-3 average of retrieved chunk relevance scores

The final confidence is min(grounding, answer_quality), with safety caps
for contradicted / unsupported claims and weak retrieval. The shape of
the returned dict is backward-compatible: it still has `confidence`,
`grounded`, and `flags`, plus the new `grounding_confidence`,
`answer_quality`, `retrieval_confidence`, `claims`, and `question_aspects`.

Acceptance criteria (agents.md Story 4):
- Returns structured dict with confidence score
- Flags low-confidence responses
- Never suppresses a flag to make the answer look better
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*The class `Chroma` was deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*The class `OllamaEmbeddings` was deprecated.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"), override=True)

log = logging.getLogger("verifier")

DEFAULT_MODEL = "gemini-flash-latest"

SYSTEM_PREFIXES = (
    "INSUFFICIENT_RETRIEVAL",
    "LOW_CONFIDENCE",
    "EMPTY_ANSWER",
    "LLM_ERROR",
    "PARSE_ERROR",
    "NO_ANSWER_FROM_CORPUS",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM",
    "PARTIAL_ANSWER",
    "CONFLICTING_SOURCES",
    "SINGLE_SOURCE_RETRIEVAL",
)

SUPPORT_WEIGHTS: Dict[str, float] = {
    "direct": 1.0,
    "reasonable_inference": 0.75,
    "absence_supported": 0.70,
    "unsupported": 0.20,
    "contradicted": 0.0,
}

ASPECT_WEIGHTS: Dict[str, float] = {
    "answered": 1.0,
    "partially_answered": 0.60,
    "not_answered": 0.25,
}

_VALID_SUPPORT = set(SUPPORT_WEIGHTS.keys())
_VALID_ASPECT_STATUS = set(ASPECT_WEIGHTS.keys())
_USEFUL_SUPPORT = {"direct", "reasonable_inference", "absence_supported"}

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


VERIFIER_SYSTEM_PROMPT = """You are the VerifierAgent in an enterprise Knowledge Operations system.

You will receive:
1. The original user query.
2. A set of retrieved document chunks (each labeled with source and page).
3. A draft answer that was generated from those chunks.

Your job: decompose the draft answer into individual factual *claims* and
classify each one by how well it is supported by the chunks. Then identify
which *aspects* of the user's question were actually answered. Finally,
detect any *conflicts* between the retrieved sources themselves.

Return ONLY a JSON object in this exact format (no other text, no markdown fences):

{
  "claims": [
    {
      "claim": "<the specific factual claim from the answer>",
      "support": "<direct | reasonable_inference | absence_supported | unsupported | contradicted>",
      "source": "<filename>" or null,
      "reason": "<short explanation>"
    }
  ],
  "question_aspects": [
    {
      "aspect": "<a sub-question or topic the user asked about>",
      "status": "<answered | partially_answered | not_answered>",
      "reason": "<short explanation>"
    }
  ],
  "conflicts": [
    {
      "topic": "<the factual topic both chunks address>",
      "source_a": "<filename>",
      "claim_a": "<what source A says about the topic>",
      "source_b": "<filename>",
      "claim_b": "<what source B says about the topic>"
    }
  ],
  "flags": ["<list of short issue strings>"]
}

SUPPORT TAGS (use exactly one per claim):
- direct:               the claim is explicitly stated in the retrieved chunks
- reasonable_inference: the claim follows logically from chunks but is not directly stated (e.g., combining two facts)
- absence_supported:    the answer correctly says the documents do not specify X, AND the retrieved chunks confirm the documents were searched and X is genuinely absent (not just missing from the top results)
- unsupported:          the claim is not supported by any retrieved chunk
- contradicted:         the claim conflicts with information in the retrieved chunks

ASPECT STATUS TAGS (use exactly one per aspect):
- answered:           this part of the question is fully addressed
- partially_answered: some relevant info is provided but a key part is missing
- not_answered:       no useful information was provided for this aspect

CONFLICTS (cross-source disagreement):
- A conflict exists when two or more retrieved chunks from DIFFERENT sources
  make factually opposite or materially different claims about the SAME topic
  (e.g. policy A says parental leave is 12 weeks, policy B says 8 weeks).
- Do NOT flag stylistic differences, paraphrases of the same fact, or
  chunks that discuss different sub-topics. Only flag genuine factual
  disagreement on a single topic across distinct sources.
- If no conflicts are present, return an empty list.

RULES:
- Break the answer into 2-8 specific factual claims. Each distinct claim should be its own entry.
- Identify 1-3 question_aspects based on what the user is actually asking.
- Use "absence_supported" only when the chunks genuinely don't cover the topic. Do NOT default to it.
- If the answer says "I cannot answer" or "the documents do not contain X", add a question_aspect with status "not_answered" and reason explaining what was missing.
- `conflicts`: list every genuine cross-source disagreement you find across the retrieved chunks. Empty list if there are none.
- `flags`: list specific issues that don't fit in claims (e.g., "answer too long", "uses external knowledge", "missing source citations", "answer contradicts itself").
- Do NOT include explanations or markdown fences. Return ONLY the JSON object.
"""


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "(no chunks were retrieved)"
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Chunk {i}] source={c['source']}, page={c['page']}, "
            f"relevance={c.get('relevance_score', '?')}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Try hard to extract a JSON object from the model's response."""
    if not raw:
        return None
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    if not text.startswith("{"):
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Verifier JSON parse failed: %s. Raw (first 300 chars): %r", e, raw[:300])
        return None


def _normalize_claim(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {"claim": "", "support": "unsupported", "source": None, "reason": ""}
    support = str(c.get("support", "unsupported")).lower().strip()
    if support not in _VALID_SUPPORT:
        support = "unsupported"
    return {
        "claim": str(c.get("claim", "")).strip(),
        "support": support,
        "source": c.get("source"),
        "reason": str(c.get("reason", "")).strip(),
    }


def _normalize_aspect(a: Any) -> Dict[str, Any]:
    if not isinstance(a, dict):
        return {"aspect": "", "status": "not_answered", "reason": ""}
    status = str(a.get("status", "not_answered")).lower().strip()
    if status not in _VALID_ASPECT_STATUS:
        status = "not_answered"
    return {
        "aspect": str(a.get("aspect", "")).strip(),
        "status": status,
        "reason": str(a.get("reason", "")).strip(),
    }


def _normalize_conflict(c: Any) -> Optional[Dict[str, str]]:
    """Normalize a verifier-emitted conflict record. Returns None when the
    record is malformed or has no real content (so the caller can drop it).

    A conflict must have a topic and two distinct sources making distinct
    claims. Self-conflicts (same source on both sides) and empty fields
    are dropped.
    """
    if not isinstance(c, dict):
        return None
    topic = str(c.get("topic", "")).strip()
    src_a = str(c.get("source_a", "")).strip()
    src_b = str(c.get("source_b", "")).strip()
    claim_a = str(c.get("claim_a", "")).strip()
    claim_b = str(c.get("claim_b", "")).strip()
    if not topic or not src_a or not src_b or not claim_a or not claim_b:
        return None
    if src_a == src_b and claim_a == claim_b:
        return None
    return {
        "topic": topic,
        "source_a": src_a,
        "claim_a": claim_a,
        "source_b": src_b,
        "claim_b": claim_b,
    }


def _is_useful_claim(claim: Dict[str, Any]) -> bool:
    return claim.get("support") in _USEFUL_SUPPORT


def grounding_confidence_from_claims(claims: List[Dict[str, Any]]) -> float:
    """Average support weight over claims. Empty list -> 0.0."""
    if not claims:
        return 0.0
    weights = [SUPPORT_WEIGHTS.get(c.get("support", "unsupported"), 0.20) for c in claims]
    return sum(weights) / len(weights)


def answer_quality_from_aspects(aspects: List[Dict[str, Any]]) -> float:
    """Average aspect weight. Empty list -> 0.0."""
    if not aspects:
        return 0.0
    weights = [ASPECT_WEIGHTS.get(a.get("status", "not_answered"), 0.25) for a in aspects]
    return sum(weights) / len(aspects)


def retrieval_confidence_from_chunks(chunks: List[Dict[str, Any]]) -> float:
    """Top-3 average of chunk relevance scores.

    0 chunks  -> 0.0
    1 chunk   -> capped at 0.55 (single chunk = no triangulation)
    2+ chunks -> top-3 average, clamped to [0, 1]
    """
    if not chunks:
        return 0.0
    scores = sorted(
        (float(c.get("relevance_score", 0.0) or 0.0) for c in chunks),
        reverse=True,
    )
    if len(scores) == 1:
        return min(0.55, max(0.0, scores[0]))
    top_n = scores[:3]
    return min(1.0, max(0.0, sum(top_n) / len(top_n)))


def _has_flag(flags: List[str], substring: str) -> bool:
    return any(substring in f for f in flags)


def _add_flag(flags: List[str], new_flag: str) -> None:
    if not _has_flag(flags, new_flag.split(" — ")[0]):
        flags.append(new_flag)


def verify(draft_answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check the draft answer against the source chunks.

    Returns a dict with the RAG-Triad breakdown:
        {
          "confidence": float in [0.0, 1.0],
          "grounded": bool,
          "flags": list[str],
          "grounding_confidence": float in [0.0, 1.0],
          "answer_quality": float in [0.0, 1.0],
          "retrieval_confidence": float in [0.0, 1.0],
          "claims": list[dict],
          "question_aspects": list[dict],
          "conflicts": list[dict],   # cross-source disagreements (US 6)
        }

    On any internal failure (no chunks, empty draft, LLM error, JSON parse
    error), returns a safe default so the workflow always has a
    verification_result to act on.
    """
    if not chunks:
        return {
            "confidence": 0.0,
            "grounded": False,
            "flags": [
                "INSUFFICIENT_RETRIEVAL — no chunks were retrieved, so the answer cannot be grounded",
                "LOW_CONFIDENCE — answer may not be fully supported by documents",
            ],
            "grounding_confidence": 0.0,
            "answer_quality": 0.0,
            "retrieval_confidence": 0.0,
            "claims": [],
            "question_aspects": [],
            "conflicts": [],
        }

    if not draft_answer or not draft_answer.strip():
        return {
            "confidence": 0.0,
            "grounded": False,
            "flags": [
                "EMPTY_ANSWER — the analyst produced no draft to verify",
                "LOW_CONFIDENCE — answer may not be fully supported by documents",
            ],
            "grounding_confidence": 0.0,
            "answer_quality": 0.0,
            "retrieval_confidence": retrieval_confidence_from_chunks(chunks),
            "claims": [],
            "question_aspects": [],
            "conflicts": [],
        }

    context = _format_chunks(chunks)
    user_prompt = (
        f"Retrieved chunks:\n\n{context}\n\n"
        f"---\n\nDraft answer to verify:\n\n{draft_answer}\n\n"
        f"Return ONLY the JSON object."
    )

    try:
        llm = _get_llm()
        msg = llm.invoke(f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_prompt}")
        raw = msg.text
    except Exception as e:
        log.exception("Verifier LLM call failed")
        return {
            "confidence": 0.5,
            "grounded": False,
            "flags": [
                f"LLM_ERROR — verifier LLM call failed: {e}",
                "LOW_CONFIDENCE — answer may not be fully supported by documents",
            ],
            "grounding_confidence": 0.5,
            "answer_quality": 0.5,
            "retrieval_confidence": retrieval_confidence_from_chunks(chunks),
            "claims": [],
            "question_aspects": [],
            "conflicts": [],
        }

    parsed = _parse_json_response(raw)
    if parsed is None:
        return {
            "confidence": 0.5,
            "grounded": False,
            "flags": [
                "PARSE_ERROR — verifier response was not valid JSON",
                "LOW_CONFIDENCE — answer may not be fully supported by documents",
            ],
            "grounding_confidence": 0.5,
            "answer_quality": 0.5,
            "retrieval_confidence": retrieval_confidence_from_chunks(chunks),
            "claims": [],
            "question_aspects": [],
            "conflicts": [],
        }

    claims: List[Dict[str, Any]] = [_normalize_claim(c) for c in (parsed.get("claims") or [])]
    aspects: List[Dict[str, Any]] = [_normalize_aspect(a) for a in (parsed.get("question_aspects") or [])]
    conflicts: List[Dict[str, str]] = []
    for c in (parsed.get("conflicts") or []):
        normalized = _normalize_conflict(c)
        if normalized is not None:
            conflicts.append(normalized)
    flags: List[str] = []
    for f in (parsed.get("flags") or []):
        if isinstance(f, str) and f.strip():
            flags.append(f.strip())

    gc = grounding_confidence_from_claims(claims)
    aq = answer_quality_from_aspects(aspects)
    rc = retrieval_confidence_from_chunks(chunks)

    confidence = min(gc, aq)

    has_contradicted = any(c.get("support") == "contradicted" for c in claims)
    has_unsupported = any(c.get("support") == "unsupported" for c in claims)

    if has_contradicted:
        confidence = min(confidence, 0.25)
    if has_unsupported:
        confidence = min(confidence, 0.65)
    if rc < 0.5:
        confidence = min(confidence, 0.45)

    grounded = (gc >= 0.6) and (not has_contradicted)

    if confidence < 0.6:
        _add_flag(flags, "LOW_CONFIDENCE — answer may not be fully supported by documents")
    if len(chunks) < 2:
        _add_flag(flags, f"INSUFFICIENT_RETRIEVAL — only {len(chunks)} chunk(s) retrieved (< 2)")

    # Source-diversity failure: when the corpus has 2+ sources but the
    # retriever only returned chunks from one of them, the answer is built
    # on a single document even though the user uploaded more. The retriever
    # marks these chunks with `_diversity_flag="SINGLE_SOURCE_RETRIEVAL"`.
    unique_sources_in_result = {c.get("source") for c in chunks if c.get("source")}
    has_diversity_flag = any(c.get("_diversity_flag") == "SINGLE_SOURCE_RETRIEVAL" for c in chunks)
    if has_diversity_flag:
        only_src = next(iter(unique_sources_in_result), "unknown")
        _add_flag(
            flags,
            f"SINGLE_SOURCE_RETRIEVAL — all {len(chunks)} chunk(s) came from "
            f"'{only_src}'; corpus has multiple sources that were not consulted",
        )
        # Cap confidence: even if the answer reads well, the retrieval
        # missed the other documents the user uploaded.
        confidence = min(confidence, 0.5)
        grounded = False

    if has_unsupported:
        n_unsup = sum(1 for c in claims if c.get("support") == "unsupported")
        _add_flag(flags, f"UNSUPPORTED_CLAIM — {n_unsup} claim(s) lack source support")
    if has_contradicted:
        n_contra = sum(1 for c in claims if c.get("support") == "contradicted")
        _add_flag(flags, f"CONTRADICTED_CLAIM — {n_contra} claim(s) conflict with sources")
    if 0.4 <= aq < 0.7:
        _add_flag(flags, "PARTIAL_ANSWER — some aspects of the question were not fully addressed")
    if aq < 0.4 and not any(_is_useful_claim(c) for c in claims):
        _add_flag(
            flags,
            "NO_ANSWER_FROM_CORPUS — the analyst found no relevant information in the retrieved documents",
        )

    # US 6 — Conflicting agent outputs: when two or more retrieved sources
    # disagree on the same factual topic, the analyst's draft cannot be
    # fully grounded in any single source. Flag the conflict, cap the
    # confidence below the LOW_CONFIDENCE threshold, and mark the answer
    # as not grounded.
    if conflicts:
        n_conf = len(conflicts)
        if n_conf == 1:
            _add_flag(
                flags,
                f"CONFLICTING_SOURCES — 1 cross-source disagreement found: "
                f"'{conflicts[0]['topic']}' between "
                f"'{conflicts[0]['source_a']}' and '{conflicts[0]['source_b']}'",
            )
        else:
            topics = ", ".join(f"'{c['topic']}'" for c in conflicts)
            _add_flag(
                flags,
                f"CONFLICTING_SOURCES — {n_conf} cross-source disagreements: {topics}",
            )
        # 1 conflict -> 0.55 cap (forces LOW_CONFIDENCE disclaimer),
        # 2+   -> 0.45 cap (multiple disagreements = even less trust).
        conflict_cap = 0.55 if n_conf == 1 else 0.45
        confidence = min(confidence, conflict_cap)
        grounded = False

    log.info(
        "Verifier result: confidence=%.2f (gc=%.2f, aq=%.2f, rc=%.2f), "
        "grounded=%s, %d flag(s), %d claim(s), %d aspect(s), %d conflict(s)",
        confidence, gc, aq, rc, grounded, len(flags), len(claims), len(aspects), len(conflicts),
    )

    return {
        "confidence": round(confidence, 2),
        "grounded": grounded,
        "flags": flags,
        "grounding_confidence": round(gc, 2),
        "answer_quality": round(aq, 2),
        "retrieval_confidence": round(rc, 2),
        "claims": claims,
        "question_aspects": aspects,
        "conflicts": conflicts,
    }
