"""Document ingestion into ChromaDB.

Loads every supported file (PDF, .txt, .md) from the docs/ directory,
chunks them, embeds with Gemini, and writes them to a persistent
ChromaDB collection. Re-running clears and re-populates the collection
so we never end up with duplicate chunks.

Run as a script:
    python vector_store/ingest.py
    python vector_store/ingest.py --docs-dir my_docs --persist-dir my_db
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*manual persistence method is no longer supported.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _load_one(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
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


def load_documents(docs_dir: Path) -> List[Document]:
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

    all_docs: List[Document] = []
    for p in files:
        loaded = _load_one(p)
        log.info("Loaded %d page(s) from %s", len(loaded), p.name)
        all_docs.extend(loaded)
    return all_docs


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


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY is missing or unset. "
            "Copy .env.example to .env and set a real key."
        )
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def reset_persistent_dir(persist_dir: Path) -> None:
    if persist_dir.exists():
        log.info("Clearing existing ChromaDB at %s", persist_dir)
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)


def ingest(docs_dir: Path, persist_dir: Path, collection_name: str) -> int:
    log.info("Loading documents from %s", docs_dir)
    docs = load_documents(docs_dir)
    chunks = split_documents(docs)

    if not chunks:
        log.warning("No chunks to ingest.")
        return 0

    embeddings = _get_embeddings()
    log.info("Embedding and writing to %s (collection=%s)",
             persist_dir, collection_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )

    counts: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

    log.info(
        "Ingestion complete: %d total chunk(s) across %d source file(s).",
        len(chunks), len(counts),
    )
    for src, n in sorted(counts.items()):
        log.info("  - %s: %d chunk(s)", src, n)

    return len(chunks)


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
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)
    try:
        reset_persistent_dir(persist_dir)
        n = ingest(docs_dir, persist_dir, args.collection)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    except EnvironmentError as e:
        log.error("%s", e)
        return 3
    except Exception:
        log.exception("Ingestion failed.")
        return 1
    log.info("Done. %d chunk(s) persisted.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
