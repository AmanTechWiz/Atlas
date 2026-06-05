"""Document ingestion into ChromaDB.

Loads supported files (PDF, DOCX, TXT, MD), chunks them, embeds them, and
writes them to a persistent ChromaDB collection. Supports two embedding
backends, switched via `EMBEDDING_BACKEND` in .env:

  - ollama   (default) — local, uses Ollama + nomic-embed-text
  - gemini             — Google Gemini embedding API

Two entry points:

1. `safe_ingest_dir(docs_dir, persist_dir, collection_name, backend)` —
   ingest every supported file in a directory. Used by the CLI and by
   dev-mode ingestion.

2. `safe_ingest_files(file_paths, persist_dir, collection_name, backend)` —
   ingest a specific list of file paths. Used by the Streamlit upload
   UI to push user-uploaded files into a per-session collection without
   writing them to disk in the `docs/` directory.

Both use the safe-swap pattern: build the new collection in a temp
directory, verify the chunk count, then swap into place. The existing
collection is left untouched if the embed step fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
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
    try:
        vs = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        return vs._collection.count()
    except Exception:
        return 0


def _swap_into_place(temp_dir: Path, persist_dir: Path) -> None:
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    temp_dir.rename(persist_dir)
    log.info("Swapped %s into place.", persist_dir)


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

    Used by the Streamlit upload UI for per-session ingestion. Uses the
    safe-swap pattern (build in `persist_dir_new`, verify, swap).
    Returns the number of chunks ingested.
    """
    docs = load_documents_from_files(file_paths)
    chunks = split_documents(docs)
    if not chunks:
        log.warning("No chunks to ingest.")
        return 0

    embeddings = _get_embeddings(backend)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = persist_dir.parent / f"{persist_dir.name}_new"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building new collection in %s ...", temp_dir)
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(temp_dir),
            collection_name=collection_name,
        )
    except Exception:
        log.exception("Ingest failed during embedding. Existing data preserved.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    actual = _count_in_dir(temp_dir, embeddings, collection_name)
    expected = len(chunks)
    if actual < expected:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Ingest produced {actual}/{expected} chunks — aborting, "
            f"existing data at {persist_dir} preserved."
        )

    _swap_into_place(temp_dir, persist_dir)
    _report_chunk_counts(chunks)
    return len(chunks)


def safe_ingest_dir(
    docs_dir: Path,
    persist_dir: Path,
    collection_name: str,
    backend: str,
) -> int:
    """Ingest every supported file in `docs_dir` into a ChromaDB collection.

    Used by the CLI and by dev-mode ingestion. Returns the number of
    chunks ingested. Uses the safe-swap pattern.
    """
    docs = load_documents(docs_dir)
    chunks = split_documents(docs)
    if not chunks:
        log.warning("No chunks to ingest.")
        return 0

    embeddings = _get_embeddings(backend)
    persist_dir = Path(persist_dir)

    temp_dir = persist_dir.parent / f"{persist_dir.name}_new"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building new collection in %s ...", temp_dir)
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(temp_dir),
            collection_name=collection_name,
        )
    except Exception:
        log.exception("Ingest failed during embedding. Existing data preserved.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    actual = _count_in_dir(temp_dir, embeddings, collection_name)
    expected = len(chunks)
    if actual < expected:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Ingest produced {actual}/{expected} chunks — aborting, "
            f"existing data at {persist_dir} preserved."
        )

    _swap_into_place(temp_dir, persist_dir)
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
