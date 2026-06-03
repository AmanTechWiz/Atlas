"""RetrieverAgent — queries ChromaDB and returns relevant chunks.

Acceptance criteria (agents.md Story 2):
- Returns 3-5 relevant chunks
- Each chunk has source attribution
- Low-relevance chunks are filtered

The relevance score is a backend-aware normalized value in [0, 1] where
1.0 = perfect match. For Ollama (L2-style distance, unbounded), we use
1 / (1 + distance). For Gemini (cosine, distance in [0, 2]), we use
1 - distance. The minimum relevance threshold is applied per-backend.
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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

log = logging.getLogger("retriever")

PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "enterprise_docs"
DEFAULT_K = 5

MIN_RELEVANCE_OLLAMA = 0.001
MIN_RELEVANCE_GEMINI = 0.3

_vectorstore = None


def _get_embeddings():
    backend = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
    if backend == "ollama":
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            google_api_key=api_key,
        )
    raise ValueError(f"Unknown EMBEDDING_BACKEND={backend!r}")


def _is_ollama() -> bool:
    return os.getenv("EMBEDDING_BACKEND", "ollama").lower() == "ollama"


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not PERSIST_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {PERSIST_DIR}. Run vector_store/ingest.py first."
            )
        _vectorstore = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=_get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def _distance_to_relevance(distance: float) -> float:
    if _is_ollama():
        return 1.0 / (1.0 + distance)
    return max(0.0, 1.0 - distance)


def retrieve(query: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """Return top-k chunks for the query as a list of dicts.

    Each dict has: text, source, page, relevance_score, _distance.
    Chunks below the backend-specific minimum relevance are filtered out.
    """
    vs = _get_vectorstore()
    raw = vs.similarity_search_with_score(query, k=k)

    min_rel = MIN_RELEVANCE_OLLAMA if _is_ollama() else MIN_RELEVANCE_GEMINI

    chunks: List[Dict[str, Any]] = []
    for doc, distance in raw:
        relevance = _distance_to_relevance(distance)
        if relevance < min_rel:
            continue
        chunks.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "relevance_score": round(relevance, 4),
                "_distance": round(distance, 4),
            }
        )

    log.info("Retrieved %d chunk(s) for query: %r", len(chunks), query)
    for c in chunks:
        log.info(
            "  - %s (page %d) score=%.4f distance=%.4f",
            c["source"], c["page"], c["relevance_score"], c["_distance"],
        )
    return chunks
