"""Tests for the intent-aware RetrieverAgent.

The retriever is query-aware: it receives `intent` and `target_doc_types`
from the orchestrator and chooses the right strategy:

  - single_document: 70-90% chunks from the target doc_type,
    10-30% supporting chunks from other doc_types.
  - comparison:      balanced chunks from each target doc_type.
  - cross_document:  balanced chunks from each source.
  - corpus_summary:  one chunk per source + global fill.

These tests use a fake vector store to verify the selection behavior
deterministically. They do not open ChromaDB or call an embedding
service.

Coverage (per the 2026-06-06 refactor spec):
  - single_document retrieval
  - comparison retrieval
  - cross_document retrieval
  - corpus_summary retrieval
  - target document prioritization (target chunks come first)
  - 70/30 split for single_document
  - source-mention boost (bare words like "resume" still match)
  - SINGLE_SOURCE_RETRIEVAL flag is set on chunks when result has
    only one source despite corpus having multiple.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from agents import retriever


# ---- Fakes ------------------------------------------------------------- #

def _row(source: str, page: int, distance: float, doc_type: str) -> Tuple[Document, float]:
    return (
        Document(
            page_content=f"{source} chunk {page}",
            metadata={"source": source, "page": page, "doc_type": doc_type},
        ),
        distance,
    )


class FakeEmbeddings:
    def embed_query(self, query: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self, rows: List[Document]) -> None:
        self._rows = rows

    def get(self, include=None) -> Dict[str, Any]:
        return {
            "ids": [f"id-{i}" for i in range(len(self._rows))],
            "metadatas": [dict(r.metadata) for r in self._rows],
        }


class FakeVectorStore:
    """3-doc corpus with mixed doc_types.

    Files and types:
      - resume.pdf                -> "resume"
      - requirements.md           -> "job_requirements"
      - policy.pdf                -> "policy"

    Global top-k ranks resume highest (because generic "experience"
    phrases), policy middle, requirements last. The intent-aware
    retriever must STILL heavily pull from the target doc_type
    when intent=single_document and target=resume.
    """

    def __init__(self) -> None:
        self._embedding_function = FakeEmbeddings()
        rows = [
            _row("resume.pdf", 1, 100.0, "resume"),
            _row("resume.pdf", 2, 110.0, "resume"),
            _row("resume.pdf", 3, 120.0, "resume"),
            _row("resume.pdf", 4, 130.0, "resume"),
            _row("resume.pdf", 5, 140.0, "resume"),
            _row("policy.pdf", 1, 200.0, "policy"),
            _row("policy.pdf", 2, 210.0, "policy"),
            _row("policy.pdf", 3, 220.0, "policy"),
            _row("requirements.md", 1, 300.0, "job_requirements"),
            _row("requirements.md", 2, 310.0, "job_requirements"),
            _row("requirements.md", 3, 320.0, "job_requirements"),
            _row("requirements.md", 4, 330.0, "job_requirements"),
        ]
        docs = [r[0] for r in rows]
        self._collection = FakeCollection(docs)
        self.global_rows = rows
        self.by_doc_type: Dict[str, List[Tuple[Document, float]]] = {}
        self.by_source: Dict[str, List[Tuple[Document, float]]] = {}
        for r in rows:
            self.by_doc_type.setdefault(r[0].metadata["doc_type"], []).append(r)
            self.by_source.setdefault(r[0].metadata["source"], []).append(r)
        self.filter_calls: List[Dict[str, Any]] = []

    def similarity_search_with_score(self, query: str, k: int, filter=None):
        self.filter_calls.append(dict(filter or {}))
        if filter and "doc_type" in filter:
            return self.by_doc_type.get(filter["doc_type"], [])[:k]
        if filter and "source" in filter:
            return self.by_source.get(filter["source"], [])[:k]
        return self.global_rows[:k]

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding, k: int, filter=None
    ):
        return self.similarity_search_with_score("", k, filter)


def _use_fake(monkeypatch, fake: FakeVectorStore) -> None:
    monkeypatch.setattr(retriever, "_get_vectorstore", lambda: fake)
    monkeypatch.setattr(retriever, "_is_ollama", lambda: True)


# ---- Aliases + mention detection ------------------------------------- #

def test_source_aliases_include_bare_words():
    aliases = retriever._source_aliases("College resume.pdf")
    assert "college resume" in aliases
    assert "resume" in aliases
    assert "college" in aliases


def test_source_aliases_for_single_word_filenames():
    aliases = retriever._source_aliases("contract.pdf")
    assert "contract" in aliases


def test_mentioned_sources_finds_bare_word_match():
    sources = ["College resume.pdf", "attention.pdf", "codebase_llmHelp.pdf"]
    mentioned = retriever._mentioned_sources(
        "tell me something about the resume", sources
    )
    assert "College resume.pdf" in mentioned


def test_mentioned_sources_empty_when_no_match():
    sources = ["College resume.pdf", "codebase_llmHelp.pdf"]
    assert retriever._mentioned_sources(
        "what is the capital of France?", sources
    ) == []


# ---- single_document intent ------------------------------------------ #

def test_single_document_returns_70_to_90_percent_from_target(monkeypatch):
    """The core spec: ~80% of returned chunks come from the target doc_type."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "tell me about the resume",
        k=5,
        intent="single_document",
        target_doc_types=["resume"],
    )
    assert len(chunks) == 5
    target_count = sum(1 for c in chunks if c.get("doc_type") == "resume")
    supporting_count = len(chunks) - target_count
    # 80% target, 20% supporting for k=5
    assert target_count == 4, f"expected 4 target chunks, got {target_count}"
    assert supporting_count == 1, f"expected 1 supporting chunk, got {supporting_count}"
    # Range check: 70-90% target
    assert 0.7 * 5 <= target_count <= 0.9 * 5, (
        f"target ratio {target_count}/{len(chunks)} = {target_count/len(chunks):.0%} "
        f"is outside 70-90% range"
    )


def test_single_document_marks_targeted_strategy(monkeypatch):
    """Target chunks get `targeted_doc_type`, supporting get `targeted_supporting`."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "anything",
        k=5,
        intent="single_document",
        target_doc_types=["resume"],
    )
    target_chunks = [c for c in chunks if c.get("doc_type") == "resume"]
    supporting_chunks = [c for c in chunks if c.get("doc_type") != "resume"]
    for c in target_chunks:
        assert c.get("retrieval_strategy") == "targeted_doc_type", (
            f"target chunk has strategy {c.get('retrieval_strategy')!r}"
        )
    for c in supporting_chunks:
        assert c.get("retrieval_strategy") == "targeted_supporting", (
            f"supporting chunk has strategy {c.get('retrieval_strategy')!r}"
        )


def test_single_document_prioritizes_target_when_global_differs(monkeypatch):
    """The target doc_type wins even when global top-k would pick others."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "anything",
        k=5,
        intent="single_document",
        target_doc_types=["job_requirements"],
    )
    # 80% from job_requirements, 20% supporting
    job_reqs = [c for c in chunks if c.get("doc_type") == "job_requirements"]
    assert len(job_reqs) >= 3, (
        f"expected 3+ job_requirements chunks, got {len(job_reqs)} "
        f"(result: {[c.get('doc_type') for c in chunks]})"
    )


def test_single_document_no_target_falls_back_to_global(monkeypatch):
    """Without a target_doc_type, the retriever returns a plain top-k."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "anything",
        k=5,
        intent="single_document",
        target_doc_types=[],
    )
    assert len(chunks) == 5


# ---- comparison intent ----------------------------------------------- #

def test_comparison_balances_between_target_doc_types(monkeypatch):
    """Comparison: top chunks from each target doc_type, roughly balanced."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "compare",
        k=4,
        intent="comparison",
        target_doc_types=["resume", "job_requirements"],
    )
    assert len(chunks) == 4
    by_type: Dict[str, int] = {}
    for c in chunks:
        by_type[c.get("doc_type", "?")] = by_type.get(c.get("doc_type", "?"), 0) + 1
    # Both target doc_types should be represented
    assert "resume" in by_type, f"no resume chunks in {by_type}"
    assert "job_requirements" in by_type, f"no job_requirements chunks in {by_type}"
    # Roughly balanced (each side should have at least 1 chunk)
    assert by_type["resume"] >= 1
    assert by_type["job_requirements"] >= 1


# ---- corpus_summary intent ------------------------------------------- #

def test_corpus_summary_returns_one_per_source_then_fill(monkeypatch):
    """Corpus summary: one chunk per source, then global fill."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "summarize all uploaded documents",
        k=5,
        intent="corpus_summary",
    )
    sources_in_result = {c.get("source") for c in chunks}
    # All 3 sources should be represented
    assert sources_in_result == {"resume.pdf", "policy.pdf", "requirements.md"}, (
        f"expected all 3 sources, got {sources_in_result}"
    )
    # First chunk per source uses the per_source strategy
    per_source = [c for c in chunks if c.get("retrieval_strategy") == "corpus_overview_per_source"]
    assert len(per_source) >= 3, (
        f"expected >= 3 per_source chunks (one per uploaded doc), got {len(per_source)}"
    )


# ---- cross_document intent ------------------------------------------- #

def test_cross_document_balances_across_sources(monkeypatch):
    """Cross-doc: per-source balanced, no specific doc_type targeted."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "anything",
        k=6,
        intent="cross_document",
        target_doc_types=[],
    )
    sources_in_result = {c.get("source") for c in chunks}
    # All 3 sources should be represented
    assert len(sources_in_result) >= 2, (
        f"cross_document should hit multiple sources, got {sources_in_result}"
    )


# ---- SINGLE_SOURCE_RETRIEVAL flag ------------------------------------ #

def test_single_source_failure_is_flagged(monkeypatch):
    """When the corpus has multiple sources but only one returns, flag it."""

    class SingleSourceStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            # Clear the other sources' data — only resume.pdf returns chunks.
            # The collection's `get()` still reports all 3 sources (because the
            # metadata is still there), but the similarity search returns 0
            # for them. This simulates a real-world data gap.
            self.by_source = {
                "resume.pdf": self.by_source["resume.pdf"],
                "policy.pdf": [],
                "requirements.md": [],
            }

    fake = SingleSourceStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "anything",
        k=5,
        intent="cross_document",
    )
    assert all(c.get("_diversity_flag") == "SINGLE_SOURCE_RETRIEVAL" for c in chunks)


# ---- chunk schema ----------------------------------------------------- #

def test_returned_chunks_have_required_fields(monkeypatch):
    """Every returned chunk has: text, source, page, relevance_score, doc_type."""
    fake = FakeVectorStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "anything",
        k=5,
        intent="cross_document",
    )
    for c in chunks:
        assert "text" in c
        assert "source" in c
        assert "page" in c
        assert "relevance_score" in c
        assert "doc_type" in c
        assert "retrieval_strategy" in c


# ---- Corpus-agnostic behavior (no hardcoding for any specific doc) --- #

def test_arbitrary_filename_works_via_source_mention(monkeypatch):
    """Corpus-agnostic proof: a file called 'Amandeep_final_v2.pdf' (random
    name, doc_type='general') is still found when the user mentions it.

    The doc_type classifier can't tag this as 'resume' (filename has no
    resume keyword), so the retriever falls through to the source-mention
    path: detects the filename in the query, then filters by source.
    """

    class RandomFilenameStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            row = (
                Document(page_content="My Kalinga Institute degree",
                         metadata={"source": "Amandeep_final_v2.pdf",
                                   "page": 1, "doc_type": "general"}),
                100.0,
            )
            self._collection = FakeCollection([row[0]])
            self.global_rows = [row]
            self.by_doc_type = {"general": [row]}
            self.by_source = {"Amandeep_final_v2.pdf": [row]}

    fake = RandomFilenameStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "tell me about Amandeep_final_v2",
        k=5,
        intent="single_document",
        target_doc_types=[],
    )
    assert len(chunks) >= 1
    assert chunks[0]["source"] == "Amandeep_final_v2.pdf"
    assert chunks[0]["doc_type"] == "general"


def test_arbitrary_filename_works_via_filename_alias(monkeypatch):
    """A bare word from a multi-word filename acts as a mention alias."""

    class BareWordStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            row = (
                Document(page_content="Q4 revenue was 12M",
                         metadata={"source": "Q4_financial_summary.txt",
                                   "page": 1, "doc_type": "general"}),
                100.0,
            )
            self._collection = FakeCollection([row[0]])
            self.global_rows = [row]
            self.by_doc_type = {"general": [row]}
            self.by_source = {"Q4_financial_summary.txt": [row]}

    fake = BareWordStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "what's in the financial summary?",
        k=5,
        intent="single_document",
        target_doc_types=[],
    )
    assert len(chunks) >= 1
    assert chunks[0]["source"] == "Q4_financial_summary.txt"


def test_unknown_doc_type_is_searchable_as_general(monkeypatch):
    """Documents with doc_type='general' are still part of the search corpus."""

    class GeneralOnlyStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            row = (
                Document(page_content="Project update notes",
                         metadata={"source": "notes.md",
                                   "page": 1, "doc_type": "general"}),
                100.0,
            )
            self._collection = FakeCollection([row[0]])
            self.global_rows = [row]
            self.by_doc_type = {"general": [row]}
            self.by_source = {"notes.md": [row]}

    fake = GeneralOnlyStore()
    _use_fake(monkeypatch, fake)

    chunks = retriever.retrieve(
        "project update",
        k=3,
        intent="cross_document",
    )
    assert len(chunks) >= 1
    assert chunks[0]["source"] == "notes.md"
    assert chunks[0]["doc_type"] == "general"
