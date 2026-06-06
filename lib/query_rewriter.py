"""Query rewriter — memory-aware query reformulation for RAG retrieval.

Sits between the Orchestrator and the Retriever in the LangGraph
flow. When the user asks a follow-up question that references prior
context (e.g. "any more info about him?"), the raw query has no
embeddable signal for the new entity ("him"). A direct similarity
search against the raw query will return whatever happens to be
nearest in vector space — which is usually a chunk from a long,
generic document, not the document the user is actually asking
about.

This module reformulates the query using the prior session context
so the embedding model can find the right chunks.

The rewriter is a single, small Gemini call. It receives:
  - the raw user query
  - the prior session context (formatted by MemoryAgent.get_context)

It returns either:
  - a self-contained search query (preferred), or
  - the original query verbatim (fallback if no context or call fails)

Public API:
    rewrite_query(query, memory_context) -> str
        Returns a self-contained search query. If `memory_context` is
        empty or the LLM call fails, returns `query` unchanged.

    _extract_text(response) -> str
        Helper that pulls a plain string out of a LangChain 1.x
        `AIMessage` response (handles list-of-blocks and plain string
        content for backward compat).
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings(
    "ignore",
    message=".*The class `.*` was deprecated.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(
    dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"),
    override=True,
)

log = logging.getLogger("query_rewriter")


REWRITE_PROMPT = """\
You are a search-query rewriter for a RAG (retrieval-augmented) system. \
Given a conversation history and the user's latest question, produce a \
single, self-contained search query that captures the user's actual \
information need. The search query will be embedded against a corpus of \
document chunks; the embedding model can only find things that are \
mentioned in the query, so pronouns ("he", "she", "it", "him", "her", \
"that", "this"), demonstratives, and deictic references must be \
resolved into the specific entity, attribute, or topic they refer to.

Rules:
- Output ONLY the rewritten query. No explanation, no preamble, no quotes.
- If the latest question is already self-contained (no unresolved \
references, no missing context), return it verbatim.
- Keep named entities, file names, and technical terms intact.
- Preserve the user's apparent intent (a "more details" follow-up \
should stay scoped to the same topic as the prior turn).
- Never invent entities that are not present in the history.
- Never add information that the user did not ask for.

Conversation history (most recent turn last):
{memory_context}

Latest user question:
{query}

Rewritten search query:"""


_NO_CONTEXT_FALLBACK_RE = re.compile(r"\b(it|that|this|he|she|him|her|his|hers|they|them|their|the (guy|person|policy|doc|document|file))\b", re.IGNORECASE)


def _looks_like_followup(query: str) -> bool:
    """Heuristic: does the query contain a pronoun that needs resolving?

    A query like "what about her projects?" needs context. A query like
    "what is the attention mechanism?" is already self-contained. We use
    a tiny regex so we can avoid the LLM call when context is empty.
    """
    if not query or len(query.split()) > 12:
        return False
    return bool(_NO_CONTEXT_FALLBACK_RE.search(query))


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Set it in .env to use the query rewriter."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.0,
    )


def rewrite_query(query: str, memory_context: str = "") -> str:
    """Rewrite a user query into a self-contained search query.

    Args:
        query: the user's latest question (raw, may contain unresolved
            pronouns or references to prior context).
        memory_context: formatted prior session history, as returned by
            `MemoryAgent.get_context()`. Empty string if no history.

    Returns:
        A self-contained search query suitable for embedding against
        the document corpus. If `memory_context` is empty OR the query
        is already self-contained (heuristic: short with no pronouns),
        OR the LLM call fails for any reason, returns `query` unchanged.

    The function never raises — failure modes return `query` so the
    retriever still gets a usable input.
    """
    query = (query or "").strip()
    if not query:
        return query

    if not memory_context.strip():
        return query

    if not _looks_like_followup(query):
        return query

    try:
        llm = _get_llm()
        prompt = REWRITE_PROMPT.format(memory_context=memory_context, query=query)
        response = llm.invoke(prompt)
        rewritten = _extract_text(response)
        if not rewritten:
            log.warning("Query rewriter returned empty; using original query")
            return query
        rewritten = rewritten.strip('"\'`').strip()
        if len(rewritten) > 500:
            rewritten = rewritten[:500].rstrip()
        log.info(
            "Query rewritten: %r -> %r",
            query[:80],
            rewritten[:80],
        )
        return rewritten
    except Exception as e:
        log.warning("Query rewriter failed (%s); using original query", e)
        return query


def _extract_text(response: Any) -> str:
    """Pull a plain string out of a LangChain 1.x `AIMessage` response.

    In langchain 1.x, `response.content` is a list of content blocks
    (e.g. `[{"type": "text", "text": "..."}]` for plain text, or a
    list of typed parts for multimodal). Older versions returned a
    plain string. This helper handles both, plus the `text` shortcut
    property that some integrations expose.
    """
    if response is None:
        return ""
    content = getattr(response, "content", None)
    if content is None:
        text_attr = getattr(response, "text", None)
        return text_attr if isinstance(text_attr, str) else ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()

    return str(content)
