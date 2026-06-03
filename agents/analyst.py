"""AnalystAgent — reasons across retrieved chunks to form a grounded answer.

Acceptance criteria (agents.md Story 3):
- Answer references multiple chunks when available
- Output format includes Reasoning, Answer, Sources Used sections
- Model does not hallucinate beyond provided context (enforced by prompt)
"""

from __future__ import annotations

import logging
import os
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

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

log = logging.getLogger("analyst")

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


ANALYST_SYSTEM_PROMPT = """You are the AnalystAgent in an enterprise Knowledge Operations system.

You will receive a user query and a set of retrieved document chunks. Each chunk
is labeled with its source document and page number.

Your job:
1. Reason across the chunks — do NOT just extract the first relevant sentence.
2. Use ONLY information that is present in the chunks. Do not use outside
   knowledge, and do not infer facts that are not directly supported.
3. If the chunks do not contain enough information to answer, say so explicitly
   in the [Answer] section.
4. Cite every source you actually used in the [Sources Used] section.

Output format — use these exact section headers, in this order:

[Reasoning]
A short description of how you arrived at the answer by combining the chunks.

[Answer]
The synthesized answer to the user's query, in clear prose.

[Sources Used]
- <filename> (page <n>)   ← one line per source actually cited
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


def analyze(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Synthesize an answer from the retrieved chunks.

    Returns a string with the three required sections.
    """
    if not chunks:
        return (
            "[Reasoning]\nNo chunks were retrieved, so no answer can be "
            "grounded in the documents.\n\n"
            "[Answer]\nI could not find relevant information in the available "
            "enterprise documents to answer this question.\n\n"
            "[Sources Used]\n(none)"
        )

    context = _format_chunks(chunks)
    user_prompt = f"User query: {query}\n\nRetrieved chunks:\n\n{context}"

    llm = _get_llm()
    msg = llm.invoke(f"{ANALYST_SYSTEM_PROMPT}\n\n{user_prompt}")
    log.info("Analyst produced draft answer (%d chars)", len(msg.text))
    return msg.text
