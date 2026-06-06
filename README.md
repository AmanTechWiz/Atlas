# Atlas

A local-first multi-agent system that answers complex business questions by reasoning across enterprise documents. Policies, SOPs, contracts, compliance manuals, anything you upload.

Atlas is not a chatbot. It is a coordinated pipeline of five specialized agents, each with a narrow role, wired together with LangGraph so every decision is inspectable and explainable. The vector store, the documents, the logs, and the UI all run on your machine. Only the LLM and embedding calls go to a remote API (Google Gemini, by default).

## What it does

Drop your enterprise documents into the system, then ask questions that span multiple files. The orchestrator reads your question, decides which agents to invoke and in what order, and produces a numbered plan. The retriever finds the relevant chunks in the vector store, biased toward whichever document you asked about. The analyst reasons across those chunks and writes a synthesized answer. The verifier checks that every claim in the answer is actually supported by the source documents and assigns a confidence score. If the score is low, the answer is wrapped in a disclaimer before you see it.

Every step is logged to a structured JSON file. The Streamlit UI shows the final answer, the plan the orchestrator made, every chunk that was retrieved, the verifier's reasoning, and the raw log. Nothing is hidden.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 or newer |
| Agent framework | LangGraph (explicit state machine, full traceability) |
| LLM | Google Gemini (`gemini-3.1-flash-lite` by default) |
| Embeddings | Ollama `nomic-embed-text` (local, no rate limit) or Gemini |
| Vector database | ChromaDB, persistent on disk |
| Document parsing | LangChain `PyPDFLoader`, `TextLoader`, `Docx2txtLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) |
| Guardrails | Custom Python validation layer (length, injection, empty corpus) |
| Evaluation | Structured JSON logs, one file per query |
| UI | Streamlit, dark theme |
| Package manager | `uv` (lockfile committed) |

## Architecture at a glance

The system is a LangGraph state machine. Each node is one agent, and the state object carries the query, plan, retrieved chunks, draft answer, verification result, and a running decision trace through the whole pipeline.

```
START
  ↓
orchestrate_node       Orchestrator plans the work
  ↓
retrieve_node          Retriever fetches chunks from ChromaDB
  ↓
analyze_node           Analyst reasons across the chunks
  ↓
verify_node            Verifier scores grounding
  ↓
  if confidence below 0.6 → low_confidence_node (adds disclaimer)
  else → finalize_node directly
  ↓
finalize_node          Builds the final answer
  ↓
memory_node            Records the turn for next time
  ↓
END
```

## Repository layout

```
agents/             The five official agents
  orchestrator.py     Planning and routing
  retriever.py        ChromaDB RAG with intent-aware retrieval
  analyst.py          Reasoning and synthesis over chunks
  verifier.py         Grounding check and confidence score
  memory.py           Session conversation buffer
graph/             LangGraph wiring
  workflow.py         StateGraph definition and run_query entry point
vector_store/      Document ingestion
  ingest.py           PDF, DOCX, TXT, MD loader + chunker + embedder
lib/               Cross-cutting helpers
  query_rewriter.py   Optional LLM-based query normalization
  document_classifier.py  Filename and content based doc type detection
  api_errors.py       Translates Gemini errors into friendly messages
guardrails/        Input and output safety checks
  checks.py           validate_input and apply_confidence_guardrail
evaluation/        Observability
  logger.py           EvalLogger writes logs/eval_<UTC>.json per query
ui/                Frontend
  app.py              Streamlit app, dark theme
tests/             148 deterministic tests, no LLM calls
chroma_db/         Local ChromaDB persistent store (gitignored)
logs/              Evaluation log output (gitignored)
```

## Prerequisites

You need Python 3.11 or newer and the `uv` package manager. If you do not have `uv`, install it from [astral.sh](https://docs.astral.sh/uv/). You will also need a Google Gemini API key, which you can get for free at [aistudio.google.com](https://aistudio.google.com/apikey).

For local embeddings you will need Ollama running on your machine. Install it from [ollama.com](https://ollama.com/) and pull the embedding model with `ollama pull nomic-embed-text`. If you prefer to use the Gemini API for embeddings as well, you can skip this step and set `EMBEDDING_BACKEND=gemini` in your `.env` file.

## Installation

Clone the repository and enter the project directory.

```bash
git clone <your-repo-url>
cd Atlas
```

Install all dependencies with `uv`. The lockfile is committed, so you will get the exact versions the project was tested with.

```bash
uv sync
```

Copy the environment template and add your real Gemini API key.

```bash
cp .env.example .env
```

Open `.env` in any editor and replace the placeholder with your actual key. The other defaults are fine for most users. Your `.env` file is already in `.gitignore` and will never be committed.

```bash
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
EMBEDDING_BACKEND=ollama
```

## First run

Start the Streamlit UI. Atlas will create the local database, the logs directory, and the memory state on the first launch.

```bash
unset VIRTUAL_ENV
uv run python -m streamlit run ui/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) in your browser. Use the sidebar to upload your PDFs, DOCX files, or text documents. Atlas will classify each one automatically (resume, policy, contract, manual, and so on) and embed them into the local ChromaDB store.

After the upload completes, type a question in the chat input. The orchestrator will produce a plan, the retriever will find the relevant chunks, the analyst will synthesize an answer, and the verifier will score it. You will see the result in the Answer tab, the full agent trace in the Agent Trace tab, the source documents in the Sources tab, and the raw JSON log in the Evaluation Log tab.

> Always use `python -m streamlit` and not the bare `streamlit` command. The `streamlit` shim that `uv` places in `.venv/bin/` has a stale shebang from a previous install and will fail to launch.

## Resetting the knowledge base

If you want to start over, click the red Reset Knowledge Base button in the sidebar. This deletes every embedded chunk from the local store and clears the conversation history. Your original files on disk are not touched.

## Documentation

The full design notes, agent role descriptions, and the step by step implementation plan live in `agents.md`. The live state of the project, including blockers, deviations from the original plan, and the file change log, lives in `progress.md`. For deeper reading:

`ARCHITECTURE.md` walks through the design decisions, the LangGraph state schema, and the trade offs the implementation makes.

`EVALUATION.md` covers the guardrails in detail, the RAG Triad grounding scoring, the confidence threshold rationale, and every known failure mode.

`UNIT_TESTS.md` lists every test, what it covers, and how to run the suite.

## Testing

The full test suite runs in under a second and makes no LLM calls. Every agent has deterministic tests for its public surface.

```bash
unset VIRTUAL_ENV
uv run python -m pytest tests/
```

## License

Internal use.
