"""VerifierAgent — checks whether a draft answer is grounded in the source chunks.

Acceptance criteria (agents.md Story 4):
- Returns structured dict with confidence score
- Flags low-confidence responses
- Never suppresses a flag to make the answer look better

Flow:
  1. If chunks is empty, return a low-confidence result with
     INSUFFICIENT_RETRIEVAL — no point asking the LLM.
  2. If draft_answer is empty, return a low-confidence result with
     EMPTY_ANSWER.
  3. Otherwise, make a second Gemini call with a strict "return JSON
     only" prompt. Parse the JSON; if it fails, return a safe default
     with confidence=0.5 and a PARSE_ERROR flag.
  4. Apply the threshold logic from agents.md:
     - confidence < 0.6 -> grounded=False, add LOW_CONFIDENCE flag
     - len(chunks) < 2  -> add INSUFFICIENT_RETRIEVAL flag
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
)

FLAG_PENALTY = 0.20

# Phrases the Analyst uses to signal "I cannot answer from these chunks".
# If the draft contains any of these, the Verifier should NOT report high
# confidence, because the user did not actually receive an answer — even
# though the refusal itself is technically grounded in the absence of
# relevant chunks. We cap confidence at 0.3 and add a flag so the UI
# shows a clear low-confidence banner.
_REFUSAL_PATTERNS = (
    "i cannot answer",
    "i can't answer",
    "unable to answer",
    "cannot be answered",
    "do not contain",
    "does not contain",
    "no information",
    "not mentioned",
    "not specified",
    "not addressed",
    "is not mentioned",
    "is not specified",
    "is not addressed",
    "is unclear",
    "not available in the",
    "based on the available information",
    "the provided documents do not",
    "no relevant information",
)


def _is_refusal(draft_answer: str) -> bool:
    if not draft_answer:
        return False
    text = draft_answer.lower()
    return any(p in text for p in _REFUSAL_PATTERNS)

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
1. A set of retrieved document chunks (each labeled with source and page).
2. A draft answer that was generated from those chunks.

Your job: rate how well the draft answer is GROUNDED in the sources — i.e.,
whether every claim in the answer is supported by the chunks.

Scoring guide (return ONE float in [0.0, 1.0]):
- 1.0    : every claim is directly supported by the sources
- 0.7-0.9: most claims supported; minor reasonable inferences
- 0.4-0.6: mixed; some claims supported, some not
- 0.1-0.3: most claims are NOT supported by the sources
- 0.0    : answer contradicts the sources or is entirely fabricated

Return ONLY a JSON object in this exact format (no other text, no markdown fences):
{"confidence": <float>, "grounded": <bool>, "flags": [<list of short strings>]}

`flags` should list specific issues, e.g. "unsupported claim about X",
"contradicts source Y", "claim about Z not in any source".
If the answer is well-grounded, return an empty flags list.
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
    """Try hard to extract a JSON object from the model's response.

    Some models wrap JSON in markdown fences (```json ... ```) or add
    preamble. We strip fences, find the first {...} block, then try
    json.loads. Returns None on failure.
    """
    if not raw:
        return None
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
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


def verify(draft_answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check the draft answer against the source chunks.

    Returns:
        {
            "confidence": float in [0.0, 1.0],
            "grounded": bool,
            "flags": list[str],
        }

    On any internal failure (LLM error, JSON parse error, missing
    chunks/draft), returns a safe default with confidence=0.5 and
    a flag describing the failure, so the workflow always has a
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
        }

    if not draft_answer or not draft_answer.strip():
        return {
            "confidence": 0.0,
            "grounded": False,
            "flags": [
                "EMPTY_ANSWER — the analyst produced no draft to verify",
                "LOW_CONFIDENCE — answer may not be fully supported by documents",
            ],
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
        }

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    flags: List[str] = []
    for f in (parsed.get("flags") or []):
        if isinstance(f, str) and f.strip():
            flags.append(f.strip())

    grounded = bool(parsed.get("grounded", confidence >= 0.6))

    llm_detected_flags = [f for f in flags if not f.startswith(SYSTEM_PREFIXES)]
    if llm_detected_flags:
        confidence = max(0.10, confidence - len(llm_detected_flags) * FLAG_PENALTY)

    if confidence < 0.6:
        grounded = False
        if not any("LOW_CONFIDENCE" in f for f in flags):
            flags.append("LOW_CONFIDENCE — answer may not be fully supported by documents")

    if len(chunks) < 2:
        if not any("INSUFFICIENT_RETRIEVAL" in f for f in flags):
            flags.append(f"INSUFFICIENT_RETRIEVAL — only {len(chunks)} chunk(s) retrieved (< 2)")

    if _is_refusal(draft_answer):
        confidence = min(confidence, 0.30)
        grounded = False
        if not any("NO_ANSWER_FROM_CORPUS" in f for f in flags):
            flags.append(
                "NO_ANSWER_FROM_CORPUS — the analyst found no relevant information in the retrieved documents"
            )
        if not any("LOW_CONFIDENCE" in f for f in flags):
            flags.append("LOW_CONFIDENCE — answer may not be fully supported by documents")

    log.info(
        "Verifier result: confidence=%.2f, grounded=%s, %d flag(s) (%d LLM-detected)",
        confidence, grounded, len(flags), len(llm_detected_flags),
    )

    return {
        "confidence": round(confidence, 2),
        "grounded": grounded,
        "flags": flags,
    }
