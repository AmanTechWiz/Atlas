"""RetrieverAgent — queries ChromaDB and returns relevant chunks.

Acceptance criteria (agents.md Story 2):
- Returns 3-5 relevant chunks
- Each chunk has source attribution
- Low-relevance chunks are filtered

The relevance score is a backend-aware normalized value in [0, 1] where
1.0 = perfect match.

For Gemini embeddings ChromaDB stores them normalized and uses cosine
distance natively, so relevance = 1 - distance (distance in [0, 2]).

For Ollama embeddings ChromaDB uses L2 (squared euclidean) distance and
the vectors are unnormalized (nomic-embed-text returns vectors of norm
~22.8, so n^2 ~ 520). We convert L2^2 to a pseudo-cosine using the
identity cos = 1 - L2^2 / (2 * n^2), then clamp to [0, 1]. This gives
scores in the [0, 1] range with real dynamic range (0.5 - 0.8 for the
top-5 of a typical query), so the UI can show meaningful badges and
the min-relevance threshold filters out truly unrelated chunks.
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

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"), override=True)

log = logging.getLogger("retriever")

PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "enterprise_docs"
DEFAULT_K = 5

MIN_RELEVANCE_OLLAMA = 0.50
MIN_RELEVANCE_GEMINI = 0.30

# Ollama's nomic-embed-text returns 768-dim vectors with L2 norm ~22.8
# (measured empirically). Used to convert ChromaDB's L2^2 distance into
# a pseudo-cosine similarity. If you swap to a different embed model
# (e.g. mxbai-embed-large, all-minilm), update this constant.
_OLLAMA_EMBED_NORM_SQ = 520.0

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
        pseudo_cos = 1.0 - distance / (2.0 * _OLLAMA_EMBED_NORM_SQ)
        return max(0.0, min(1.0, pseudo_cos))
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


def get_corpus_size() -> int:
    """Return the number of chunks in the active ChromaDB collection.

    Returns 0 when the collection does not exist yet (no documents ingested)
    or when the read fails for any reason. Used by the input guardrail to
    block queries when the corpus is empty.
    """
    try:
        vs = _get_vectorstore()
        return int(vs._collection.count() or 0)
    except FileNotFoundError:
        return 0
    except Exception as e:
        log.warning("Could not read corpus size: %s", e)
        return 0
