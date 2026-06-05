"""Document ingestion into ChromaDB.

Loads supported files (PDF, DOCX, TXT, MD), chunks them, embeds them, and
writes them to a persistent ChromaDB collection. Supports two embedding
backends, switched via `EMBEDDING_BACKEND` in .env:

  - ollama   (default) — local, uses Ollama + nomic-embed-text
  - gemini             — Google Gemini embedding API

Three public entry points:

1. `safe_ingest_dir(docs_dir, persist_dir, collection_name, backend)` —
   ingest every supported file in a directory. Used by the CLI and by
   dev-mode ingestion. Full replace of the collection.

2. `safe_ingest_files(file_paths, persist_dir, collection_name, backend)` —
   ingest a specific list of file paths. Full replace of the collection.
   Used by the Streamlit upload UI when the user explicitly clicks
   "Reset Session" + re-uploads.

3. `add_files_to_collection(file_paths, persist_dir, collection_name, backend)` —
   append the given files' chunks to an existing (or new) collection.
   Existing data is preserved. Used by the Streamlit upload UI for
   the per-session "add another file" flow.

Why no safe-swap: chromadb 1.5.x + langchain-chroma has a bug where
`Chroma.from_documents` to a temp dir, followed by a rename into the
final persist dir, leaves the HNSW index and SQLite metadata out of
sync. Subsequent reads via `Chroma(persist_directory=...)` return 0
even though the HNSW file is full. We avoid this by writing directly
to the target persist dir via `Chroma.add_documents` (which both
creates and appends safely).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, List

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
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ingest")


DEFAULT_DOCS_DIR = "docs"
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION = "enterprise_docs"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}


def _load_one(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        log.warning("Skipping unsupported file: %s", path.name)
        return []

    docs = loader.load()
    for d in docs:
        meta = dict(d.metadata or {})
        meta["source"] = path.name
        meta.setdefault("page", 0)
        if isinstance(meta["page"], str):
            try:
                meta["page"] = int(meta["page"])
            except ValueError:
                meta["page"] = 0
        d.metadata = meta
    return docs


def load_documents_from_files(file_paths: Iterable[Path]) -> List[Document]:
    """Load a specific list of files. Used by both the CLI and the UI upload path."""
    all_docs: List[Document] = []
    for p in file_paths:
        if not p.exists():
            log.warning("File not found, skipping: %s", p)
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            log.warning("Unsupported file type, skipping: %s", p.name)
            continue
        loaded = _load_one(p)
        log.info("Loaded %d page(s) from %s", len(loaded), p.name)
        all_docs.extend(loaded)
    if not all_docs:
        raise FileNotFoundError("no supported files were loaded")
    return all_docs


def load_documents(docs_dir: Path) -> List[Document]:
    """Load every supported file in a directory. Used by the CLI."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"docs directory not found: {docs_dir}")

    files = sorted(
        p for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"no supported files ({sorted(SUPPORTED_SUFFIXES)}) in {docs_dir}"
        )
    return load_documents_from_files(files)


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    log.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(docs), len(chunks), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return chunks


def _get_embeddings(backend: str):
    if backend == "ollama":
        model = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        log.info("Embedding backend: ollama (model=%s, base_url=%s)", model, base_url)
        return OllamaEmbeddings(model=model, base_url=base_url)

    if backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise EnvironmentError(
                "GEMINI_API_KEY is missing or unset. "
                "Copy .env.example to .env and set a real key, "
                "or set EMBEDDING_BACKEND=ollama to use local embeddings."
            )
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        log.info("Embedding backend: gemini (model=%s)", model)
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    raise ValueError(
        f"Unknown EMBEDDING_BACKEND={backend!r}. Use 'ollama' or 'gemini'."
    )


def _count_in_dir(persist_dir: Path, embeddings, collection_name: str) -> int:
    """Return the chunk count of `collection_name` at `persist_dir`, or 0.

    Opens a fresh `Chroma` handle each call (no module-level caching)
    so callers see the latest on-disk state. The previous safe-swap
    flow was broken in chromadb 1.5.x — the SQLite+HNSW state would
    fall out of sync after a rename. Writing directly to the target
    persist dir (no swap) makes this call reliable.
    """
    try:
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        return vs._collection.count()
    except Exception:
        return 0


def _open_collection(persist_dir: Path, embeddings, collection_name: str):
    """Open (or create) a Chroma collection at `persist_dir` and return
    the LangChain `Chroma` handle. Direct `add_documents` is the safe
    write path that avoids the safe-swap rename bug."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def _report_chunk_counts(chunks: List[Document]) -> None:
    counts: dict = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    log.info(
        "Ingestion complete: %d total chunk(s) across %d source file(s).",
        len(chunks), len(counts),
    )
    for src, n in sorted(counts.items()):
        log.info("  - %s: %d chunk(s)", src, n)


def safe_ingest_files(
    file_paths: List[Path],
    persist_dir: Path,
    collection_name: str,
    backend: str,
) -> int:
    """Ingest a specific list of files into a ChromaDB collection.

    Full replace of any existing collection at `persist_dir`. Writes
    directly to the target dir via `Chroma.add_documents` (no safe-swap
    rename) to avoid a chromadb 1.5.x bug where renamed HNSW indices
    fall out of sync with SQLite metadata.

    WARNING: this is a full replace. For the "add another file to the
    existing corpus" use case (e.g. Streamlit upload), use
    `add_files_to_collection` instead — it preserves existing data.
    """
    docs = load_documents_from_files(file_paths)
    chunks = split_documents(docs)
    if not chunks:
        log.warning("No chunks to ingest.")
        return 0

    embeddings = _get_embeddings(backend)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    log.info("Replacing collection %s at %s ...", collection_name, persist_dir)
    # Full replace: drop the existing collection (if any), then create
    # it fresh and add the new chunks.
    vs = _open_collection(persist_dir, embeddings, collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    vs = _open_collection(persist_dir, embeddings, collection_name)
    vs.add_documents(chunks)
    _report_chunk_counts(chunks)
    return len(chunks)


def add_files_to_collection(
    file_paths: List[Path],
    persist_dir: Path,
    collection_name: str,
    backend: str,
) -> int:
    """Add the given files' chunks to an existing (or new) collection.

    Appends new chunks via `Chroma.add_documents` if the collection
    already exists; creates the collection fresh otherwise. Existing
    data is preserved. Writes directly to `persist_dir` (no safe-swap
    rename).

    Returns the number of NEW chunks added (not the total in the
    collection).

    Used by the Streamlit upload UI for per-session ingestion. The
    caller is responsible for invalidating any cached Chroma handle
    after this function returns (e.g. by calling
    `agents.retriever.set_active_collection`).
    """
    docs = load_documents_from_files(file_paths)
    chunks = split_documents(docs)
    if not chunks:
        log.warning("No chunks to ingest.")
        return 0

    embeddings = _get_embeddings(backend)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    existing_count = _count_in_dir(persist_dir, embeddings, collection_name)
    if existing_count > 0:
        log.info(
            "Collection %s already has %d chunk(s); appending %d new chunk(s).",
            collection_name, existing_count, len(chunks),
        )
        vs = _open_collection(persist_dir, embeddings, collection_name)
        vs.add_documents(chunks)
        _report_chunk_counts(chunks)
        return len(chunks)

    log.info(
        "Collection %s does not exist yet; creating it with %d chunk(s).",
        collection_name, len(chunks),
    )
    vs = _open_collection(persist_dir, embeddings, collection_name)
    vs.add_documents(chunks)
    _report_chunk_counts(chunks)
    return len(chunks)


def safe_ingest_dir(
    docs_dir: Path,
    persist_dir: Path,
    collection_name: str,
    backend: str,
) -> int:
    """Ingest every supported file in `docs_dir` into a ChromaDB collection.

    Full replace of any existing collection at `persist_dir`. Used by
    the CLI and by dev-mode ingestion. Returns the number of chunks
    ingested. Writes directly to `persist_dir` (no safe-swap rename).
    """
    docs = load_documents(docs_dir)
    chunks = split_documents(docs)
    if not chunks:
        log.warning("No chunks to ingest.")
        return 0

    embeddings = _get_embeddings(backend)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    log.info("Replacing collection %s at %s ...", collection_name, persist_dir)
    vs = _open_collection(persist_dir, embeddings, collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    vs = _open_collection(persist_dir, embeddings, collection_name)
    vs.add_documents(chunks)
    _report_chunk_counts(chunks)
    return len(chunks)


def delete_collection(persist_dir: Path, collection_name: str) -> bool:
    """Delete a ChromaDB collection from disk. Returns True if a collection
    was actually removed. Used by the UI to clear a per-session collection
    on Reset."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        return False
    try:
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=_get_embeddings(os.getenv("EMBEDDING_BACKEND", "ollama")),
            collection_name=collection_name,
        )
        vs.delete_collection()
        log.info("Deleted collection %s from %s", collection_name, persist_dir)
        return True
    except Exception as e:
        log.warning("Failed to delete collection %s: %s", collection_name, e)
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest enterprise documents into ChromaDB"
    )
    p.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR,
                   help=f"directory containing documents (default: {DEFAULT_DOCS_DIR})")
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR,
                   help=f"ChromaDB persistence path (default: {DEFAULT_PERSIST_DIR})")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})")
    p.add_argument(
        "--embedding-backend",
        choices=["ollama", "gemini"],
        default=os.getenv("EMBEDDING_BACKEND", "ollama"),
        help="embedding backend (default: ollama, or env EMBEDDING_BACKEND)",
    )
    return p.parse_args()


def main() -> int:
    load_dotenv(override=True)
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)
    try:
        n = safe_ingest_dir(docs_dir, persist_dir, args.collection, args.embedding_backend)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    except EnvironmentError as e:
        log.error("%s", e)
        return 3
    except RuntimeError as e:
        log.error("%s", e)
        return 4
    except Exception:
        log.exception("Ingestion failed.")
        return 1
    log.info("Done. %d chunk(s) persisted.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
