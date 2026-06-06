"""RetrieverAgent — intent-aware retrieval over ChromaDB.

The retriever is query-aware: the orchestrator classifies each user
query into one of four intents and passes the intent + target doc
types to `retrieve()`. The retriever then chooses the right strategy:

  - single_document: 70-90% chunks from the target document, 10-30%
    supporting chunks from other documents. Filter by `doc_type`
    metadata for the target. Example for k=4: `resume, resume, resume,
    requirements`.
  - comparison: balanced — top chunks from EACH target doc type so
    the synthesizer can compare. Example for k=4: `resume, resume,
    requirements, requirements`.
  - cross_document: balanced — top chunks from each named doc type,
    or one per source if no doc type was named.
  - corpus_summary: one representative chunk from EVERY uploaded
    document (source-balanced), then fill the rest with global top-k.

Source-name mention detection is also applied: a query like
"tell me about the resume" gets a mention boost on the file
`College resume.pdf` (resolved via doc_type lookup of the
mentioned source).

Each returned chunk has: text, source, page, relevance_score,
doc_type, retrieval_strategy.

A `SINGLE_SOURCE_RETRIEVAL` flag is attached to every chunk when
the corpus has multiple sources but the result came from only one.
The verifier reads this flag and caps confidence at 0.5 with a
clear `SINGLE_SOURCE_RETRIEVAL` flag.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "atlas_corpus"
DEFAULT_K = 5

MIN_RELEVANCE_OLLAMA = 0.50
MIN_RELEVANCE_GEMINI = 0.30

# Ollama's nomic-embed-text returns 768-dim vectors with L2 norm ~22.8
_OLLAMA_EMBED_NORM_SQ = 520.0

# Single application-level vector store. No per-session collections.
_vectorstore = None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_persist_dir() -> Path:
    return PERSIST_DIR


def get_default_collection_name() -> str:
    return DEFAULT_COLLECTION_NAME


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
    """Lazy-init the single Chroma handle. Re-opens on every call to
    pick up writes that happened since the last retrieval."""
    global _vectorstore
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {PERSIST_DIR}. "
            f"Upload documents via the UI to create it."
        )
    _vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=_get_embeddings(),
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    return _vectorstore


def invalidate_cache() -> None:
    """Drop the cached Chroma handle. Call after writes to ensure the
    next read sees the latest data."""
    global _vectorstore
    _vectorstore = None


def _distance_to_relevance(distance: float) -> float:
    if _is_ollama():
        pseudo_cos = 1.0 - distance / (2.0 * _OLLAMA_EMBED_NORM_SQ)
        return max(0.0, min(1.0, pseudo_cos))
    return max(0.0, 1.0 - distance)


def _min_relevance() -> float:
    return MIN_RELEVANCE_OLLAMA if _is_ollama() else MIN_RELEVANCE_GEMINI


def _chunk_identity(chunk: Dict[str, Any]) -> Tuple[str, Any, str]:
    return (
        str(chunk.get("source", "unknown")),
        chunk.get("page", 0),
        chunk.get("text", ""),
    )


def _chunk_from_doc(doc: Any, distance: float) -> Dict[str, Any]:
    relevance = _distance_to_relevance(float(distance))
    return {
        "text": doc.page_content,
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page", 0),
        "doc_type": doc.metadata.get("doc_type", "general"),
        "relevance_score": round(relevance, 4),
        "_distance": round(float(distance), 4),
    }


def _chunks_from_raw(
    raw: Sequence[Tuple[Any, float]],
    *,
    min_relevance: Optional[float],
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for doc, distance in raw:
        chunk = _chunk_from_doc(doc, distance)
        if min_relevance is not None and chunk["relevance_score"] < min_relevance:
            continue
        chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Corpus introspection
# --------------------------------------------------------------------------- #

def _get_indexed_sources(vs: Any) -> List[str]:
    try:
        data = vs._collection.get(include=["metadatas"])
    except Exception as e:
        log.warning("Could not inspect collection sources: %s", e)
        return []

    sources = {
        str(meta.get("source", "")).strip()
        for meta in data.get("metadatas", []) or []
        if meta and str(meta.get("source", "")).strip()
    }
    return sorted(sources)


def _get_indexed_doc_types(vs: Any) -> List[str]:
    try:
        data = vs._collection.get(include=["metadatas"])
    except Exception as e:
        log.warning("Could not inspect doc types: %s", e)
        return []

    types = {
        str(meta.get("doc_type", "")).strip()
        for meta in data.get("metadatas", []) or []
        if meta and str(meta.get("doc_type", "")).strip()
    }
    return sorted(types)


# --------------------------------------------------------------------------- #
# Mention detection (used as a tie-breaker inside single_document)
# --------------------------------------------------------------------------- #

def _source_aliases(source: str) -> List[str]:
    path = Path(source)
    stem = path.stem.lower()
    aliases = {source.lower(), path.name.lower()}
    if len(stem) >= 3:
        aliases.update({
            stem,
            stem.replace("_", " "),
            stem.replace("-", " "),
        })
        words = stem.replace("_", " ").replace("-", " ").split()
        if len(words) > 1:
            for w in words:
                if len(w) >= 4 and w not in {"file", "text", "document", "upload"}:
                    aliases.add(w)
    return sorted(a for a in aliases if a)


def _query_mentions_alias(query: str, alias: str) -> bool:
    if "." in alias or "_" in alias or "-" in alias or " " in alias:
        return alias in query
    return bool(re.search(rf"\b{re.escape(alias)}\b", query))


def _mentioned_sources(query: str, sources: Sequence[str]) -> List[str]:
    q = (query or "").lower()
    mentioned = []
    for source in sources:
        if any(_query_mentions_alias(q, alias) for alias in _source_aliases(source)):
            mentioned.append(source)
    return mentioned


# --------------------------------------------------------------------------- #
# Search helpers
# --------------------------------------------------------------------------- #

def _similarity_search_by_vector(
    vs: Any,
    query_embedding: Optional[List[float]],
    query: str,
    *,
    k: int,
    doc_type: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Tuple[Any, float]]:
    metadata_filter: Dict[str, Any] = {}
    if doc_type:
        metadata_filter["doc_type"] = doc_type
    if source:
        metadata_filter["source"] = source
    if not metadata_filter:
        metadata_filter = None  # type: ignore
    if query_embedding is not None:
        return vs.similarity_search_by_vector_with_relevance_scores(
            query_embedding, k=k, filter=metadata_filter
        )
    return vs.similarity_search_with_score(query, k=k, filter=metadata_filter)


def _embed_query_once(vs: Any, query: str) -> Optional[List[float]]:
    embeddings = getattr(vs, "_embedding_function", None)
    if embeddings is None:
        return None
    try:
        return embeddings.embed_query(query)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Intent-aware retrieval strategies
# --------------------------------------------------------------------------- #

def _retrieve_targeted(
    vs: Any,
    query: str,
    *,
    k: int,
    target_doc_types: Sequence[str],
    sources: Sequence[str],
    min_relevance: float,
) -> List[Dict[str, Any]]:
    """single_document intent: 70-90% target + 10-30% supporting.

    The target document is the primary focus, but a small slice of
    supporting chunks from OTHER documents is added so the analyst
    has cross-context if the user's question straddles documents
    (e.g. "tell me about the resume" might want a single supporting
    chunk from another doc for context).

    Target behavior for k=4, single target: ~3 target + ~1 supporting.
    """
    query_embedding = _embed_query_once(vs, query)
    selected: List[Dict[str, Any]] = []
    seen = set()

    targets = list(target_doc_types) if target_doc_types else []

    # If the user didn't name a doc_type, try to detect a file mention.
    if not targets:
        mentioned = _mentioned_sources(query, sources)
        if mentioned:
            try:
                data = vs._collection.get(include=["metadatas"])
                for meta in data.get("metadatas", []) or []:
                    if meta and meta.get("source") == mentioned[0] and meta.get("doc_type"):
                        targets = [str(meta["doc_type"])]
                        break
            except Exception:
                pass
        if not targets and mentioned:
            for src in mentioned[:1]:
                raw = _similarity_search_by_vector(
                    vs, query_embedding, query, k=k, source=src
                )
                for chunk in _chunks_from_raw(raw, min_relevance=min_relevance):
                    key = _chunk_identity(chunk)
                    if key in seen:
                        continue
                    chunk["retrieval_strategy"] = "targeted_source"
                    selected.append(chunk)
                    seen.add(key)
            selected.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)
            return selected[:k]

    if not targets:
        raw = _similarity_search_by_vector(vs, query_embedding, query, k=k)
        chunks = _chunks_from_raw(raw, min_relevance=min_relevance)
        for c in chunks:
            c["retrieval_strategy"] = "fallback_global"
        return chunks

    target_dt = targets[0]
    n_target = max(1, int(round(k * 0.8)))
    n_supporting = max(0, k - n_target)

    raw = _similarity_search_by_vector(
        vs, query_embedding, query, k=n_target + 2, doc_type=target_dt
    )
    for chunk in _chunks_from_raw(raw, min_relevance=min_relevance)[:n_target]:
        key = _chunk_identity(chunk)
        if key in seen:
            continue
        chunk["retrieval_strategy"] = "targeted_doc_type"
        selected.append(chunk)
        seen.add(key)

    if n_supporting > 0 and len(selected) < k:
        raw = _similarity_search_by_vector(vs, query_embedding, query, k=k * 3)
        supporting_added = 0
        for chunk in _chunks_from_raw(raw, min_relevance=min_relevance):
            if chunk.get("doc_type") == target_dt:
                continue
            key = _chunk_identity(chunk)
            if key in seen:
                continue
            chunk["retrieval_strategy"] = "targeted_supporting"
            selected.append(chunk)
            seen.add(key)
            supporting_added += 1
            if supporting_added >= n_supporting:
                break

    selected.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)
    return selected[:k]


def _retrieve_balanced(
    vs: Any,
    query: str,
    *,
    k: int,
    target_doc_types: Sequence[str],
    sources: Sequence[str],
    min_relevance: float,
) -> List[Dict[str, Any]]:
    """cross_document / comparison intent: top chunks from each side.

    Reserves an even quota per target doc type (or per source if no
    target_doc_types), then fills with global top-k.
    """
    query_embedding = _embed_query_once(vs, query)
    selected: List[Dict[str, Any]] = []
    seen = set()

    units: List[Dict[str, str]] = []  # [{"kind": "doc_type"|"source", "value": ...}]
    if target_doc_types:
        for dt in target_doc_types:
            units.append({"kind": "doc_type", "value": dt})
    else:
        for s in sources:
            units.append({"kind": "source", "value": s})

    if not units:
        raw = _similarity_search_by_vector(vs, query_embedding, query, k=k)
        return _chunks_from_raw(raw, min_relevance=min_relevance)

    per_unit = max(1, k // len(units))
    for unit in units:
        kw = {"doc_type": unit["value"]} if unit["kind"] == "doc_type" else {"source": unit["value"]}
        # Use a direct filter (we can't pass kwargs to similarity_search by vector)
        if unit["kind"] == "doc_type":
            raw = _similarity_search_by_vector(
                vs, query_embedding, query, k=per_unit + 1, doc_type=unit["value"]
            )
        else:
            raw = _similarity_search_by_vector(
                vs, query_embedding, query, k=per_unit + 1, source=unit["value"]
            )
        for chunk in _chunks_from_raw(raw, min_relevance=None)[:per_unit]:
            key = _chunk_identity(chunk)
            if key in seen:
                continue
            chunk["retrieval_strategy"] = f"balanced_{unit['kind']}"
            selected.append(chunk)
            seen.add(key)

    selected.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)

    # Fill remaining slots from global top-k
    target_total = k
    raw_global = _similarity_search_by_vector(vs, query_embedding, query, k=target_total * 3)
    for chunk in _chunks_from_raw(raw_global, min_relevance=min_relevance):
        key = _chunk_identity(chunk)
        if key in seen:
            continue
        chunk["retrieval_strategy"] = "balanced_fill"
        selected.append(chunk)
        seen.add(key)
        if len(selected) >= target_total:
            break

    return selected[:k]


def _retrieve_corpus_overview(
    vs: Any,
    query: str,
    *,
    k: int,
    sources: Sequence[str],
    min_relevance: float,
) -> List[Dict[str, Any]]:
    """corpus_summary intent: one chunk from each source, then global fill."""
    query_embedding = _embed_query_once(vs, query)
    selected: List[Dict[str, Any]] = []
    seen = set()

    for source in sources:
        raw = _similarity_search_by_vector(
            vs, query_embedding, query, k=3, source=source
        )
        for chunk in _chunks_from_raw(raw, min_relevance=None)[:1]:
            key = _chunk_identity(chunk)
            if key in seen:
                continue
            chunk["retrieval_strategy"] = "corpus_overview_per_source"
            selected.append(chunk)
            seen.add(key)

    selected.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)

    target_total = max(k, len(sources))
    raw_global = _similarity_search_by_vector(vs, query_embedding, query, k=target_total * 3)
    for chunk in _chunks_from_raw(raw_global, min_relevance=min_relevance):
        key = _chunk_identity(chunk)
        if key in seen:
            continue
        chunk["retrieval_strategy"] = "corpus_overview_fill"
        selected.append(chunk)
        seen.add(key)
        if len(selected) >= target_total:
            break

    return selected[:max(k, len(sources))]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def retrieve(
    query: str,
    k: int = DEFAULT_K,
    intent: str = "single_document",
    target_doc_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return chunks for the query using an intent-aware strategy.

    `intent` is one of: single_document, cross_document, comparison,
    corpus_summary. `target_doc_types` is the list of doc types the
    query is asking about (e.g. ["resume"], ["resume", "policy"]).
    Empty list = no specific target — let the strategy fall back to
    global similarity.
    """
    vs = _get_vectorstore()
    min_rel = _min_relevance()
    sources = _get_indexed_sources(vs)
    target_doc_types = list(target_doc_types or [])

    if not sources:
        return []

    if intent == "corpus_summary":
        chunks = _retrieve_corpus_overview(
            vs, query, k=k, sources=sources, min_relevance=min_rel
        )
    elif intent in ("cross_document", "comparison"):
        chunks = _retrieve_balanced(
            vs, query, k=k, target_doc_types=target_doc_types,
            sources=sources, min_relevance=min_rel,
        )
    else:  # single_document
        chunks = _retrieve_targeted(
            vs, query, k=k, target_doc_types=target_doc_types,
            sources=sources, min_relevance=min_rel,
        )

    # Single-source failure flagging
    if len(sources) > 1:
        result_sources = {c.get("source") for c in chunks}
        if len(result_sources) == 1 and chunks:
            log.warning(
                "Source diversity failed: %d chunks all from %s (out of %d sources).",
                len(chunks), next(iter(result_sources), "?"), len(sources),
            )
            for c in chunks:
                c["_diversity_flag"] = "SINGLE_SOURCE_RETRIEVAL"

    log.info(
        "Retrieved %d chunk(s) for intent=%s targets=%s query=%r",
        len(chunks), intent, target_doc_types, query,
    )
    for c in chunks:
        log.info(
            "  - %s [%s] p.%d score=%.4f strategy=%s",
            c["source"], c.get("doc_type", "?"), c["page"],
            c["relevance_score"], c.get("retrieval_strategy", "?"),
        )
    return chunks


def get_corpus_size() -> int:
    try:
        vs = _get_vectorstore()
        return int(vs._collection.count() or 0)
    except FileNotFoundError:
        return 0
    except Exception as e:
        log.warning("Could not read corpus size: %s", e)
        return 0


def get_indexed_doc_types() -> List[str]:
    """Public helper for the UI / orchestrator."""
    try:
        vs = _get_vectorstore()
        return _get_indexed_doc_types(vs)
    except Exception:
        return []
