"""Library modules: shared helpers used by multiple agents.

This package holds the cross-cutting helpers that are NOT themselves
agents (no LLM, no retrieval, no verification). Each module here is
imported by 1+ modules in `agents/`, `vector_store/`, or `ui/`.

Modules:
    query_rewriter       — memory-aware query reformulation (LLM)
    document_classifier  — doc_type classification at ingest (heuristic + LLM)
    api_errors           — friendly messages for raw Gemini API exceptions
"""
