# Atlas

A local-first multi-agent system that answers complex business questions by reasoning across your enterprise documents.

Atlas is not a chatbot. It is a pipeline of five specialized agents, wired together with LangGraph, that plans, retrieves, reasons, verifies, and remembers. Every step is logged. Every claim is traceable back to a source document. The vector store, the documents, the logs, and the UI all run on your machine. Only the LLM and embedding calls leave.

## What it does

Upload your PDFs, DOCX files, or text documents. Ask a question. The orchestrator produces a plan. The retriever pulls the right chunks from the vector store, biased toward whichever document you asked about. The analyst reasons across those chunks. The verifier checks that every claim is grounded in the source. If confidence is low, the answer gets a visible disclaimer. The full reasoning chain is one click away in the UI.

## The five agents

| Agent | Role |
|---|---|
| Orchestrator | Reads the query, decides which agents to call and in order, produces a numbered plan |
| Retriever | Embeds the query, searches ChromaDB, returns the most relevant chunks with source attribution |
| Analyst | Reasons across the retrieved chunks, synthesizes a grounded answer, refuses to hallucinate |
| Verifier | Independently checks that every claim in the answer is supported by source documents, assigns a confidence score |
| Memory | Holds the last few turns of conversation so follow-up questions work naturally |

The full state machine runs through LangGraph. Every node writes to a shared state object that is logged to JSON for later inspection.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 or newer |
| Agent framework | LangGraph |
| LLM | Google Gemini (`gemini-3.1-flash-lite` by default) |
| Embeddings | Ollama `nomic-embed-text` (local, no rate limit) or Gemini |
| Vector database | ChromaDB, persistent on disk |
| Document parsing | LangChain PDF, text, and DOCX loaders |
| Chunking | Recursive character splitter, 500 chars with 50 overlap |
| Guardrails | Custom input validation (length, prompt injection, empty corpus) and output disclaimer |
| Evaluation | Structured JSON logs, one file per query, readable from the UI |
| UI | Streamlit, dark theme |
| Package manager | `uv`, lockfile committed |

## Repository layout

```
agents/             The five official agents
lib/               Cross-cutting helpers (query rewriter, doc classifier, API error translator)
graph/             LangGraph wiring and the run_query entry point
vector_store/      Document ingestion
guardrails/        Input and output safety checks
evaluation/        Observability
ui/                Streamlit frontend
tests/             148 deterministic tests, no LLM calls
chroma_db/         Local ChromaDB store
logs/              Evaluation log output
```

## Prerequisites

Python 3.11 or newer and the `uv` package manager. A Google Gemini API key, free at [aistudio.google.com](https://aistudio.google.com/apikey). For local embeddings, install [Ollama](https://ollama.com/) and run `ollama pull nomic-embed-text`. To use the Gemini API for embeddings instead, skip Ollama and set `EMBEDDING_BACKEND=gemini` in your `.env`.

## Installation

Clone the repo and enter the directory.

```bash
git clone <your-repo-url>
cd Atlas
```

Install dependencies. The lockfile is committed, so you get the exact tested versions.

```bash
uv sync
```

Copy the environment template and add your real Gemini API key.

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder. The other defaults are fine for most users. Your `.env` is already in `.gitignore` and will never be committed.

```bash
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
EMBEDDING_BACKEND=ollama
```

## Running the app

```bash
unset VIRTUAL_ENV
uv run python -m streamlit run ui/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The first launch wipes the local database to give you a clean slate, then you can upload documents from the sidebar and start asking questions.

> Always use `python -m streamlit` and not the bare `streamlit` command. The `streamlit` shim that `uv` places in `.venv/bin/` has a stale shebang and will fail to launch.

## Resetting the knowledge base

The Reset Knowledge Base button sits at the top of the sidebar. Click it to wipe every embedded chunk, clear the conversation history, and reset the memory in one step. The knowledge base also resets automatically every time the Streamlit session restarts, so you can simply refresh the browser tab if you want a fresh slate.

## Documentation

The architecture, design decisions, and LangGraph state schema live in `ARCHITECTURE.md`. The guardrails, grounding scoring, confidence threshold rationale, and failure modes are in `EVALUATION.md`. Every test, what it covers, and how to run the suite are in `UNIT_TESTS.md`.

## Testing

The full suite runs in under a second. No LLM calls, no network, no flake.

```bash
unset VIRTUAL_ENV
uv run python -m pytest tests/
```

## License

Internal use.
