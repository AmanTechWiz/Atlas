# Enterprise Knowledge Ops Agent

A **local-first, multi-agent AI system** that answers complex business questions by reasoning across multiple enterprise documents — policies, SOPs, contracts, compliance manuals. Built as part of the Cognizant Skillspring Agentic AI Developer case study.

The system is **not a single chatbot**. It is a coordinated pipeline of specialized agents — each with a clearly defined role — wired together with [LangGraph](https://langchain-ai.github.io/langgraph/) for full traceability, explainability, and governance.

---

## What it does

- **Plans** how to answer a complex question (not just keyword match)
- **Delegates** subtasks to specialized agents (Retriever, Analyst, Verifier, Memory, Orchestrator)
- **Retrieves** relevant chunks from a local ChromaDB vector store using Gemini embeddings
- **Reasons and synthesizes** across multiple documents using Gemini
- **Verifies** that the final answer is grounded in source documents and assigns a confidence score
- **Flags** low-confidence or hallucinated responses with disclaimers
- **Logs** every decision trace as structured JSON for explainability
- **Exposes** all of this through a Streamlit UI

---

## Tech Stack

| Layer            | Technology                                 |
| ---------------- | ------------------------------------------ |
| Language         | Python 3.11+                               |
| Agent framework  | LangGraph                                  |
| LLM              | Gemini API (`gemini-1.5-flash`)            |
| Embeddings       | Gemini (`models/text-embedding-004`)       |
| Vector database  | ChromaDB (persistent, local)               |
| Document parsing | LangChain `PyPDFLoader`, `TextLoader`      |
| Chunking         | LangChain `RecursiveCharacterTextSplitter` |
| Guardrails       | Custom Python validation layer             |
| Evaluation logs  | Structured JSON via `logs/eval_*.json`     |
| UI               | Streamlit                                  |
| Package manager  | `uv` (lockfile: `uv.lock`)                 |

---

## Project Structure

```
.
├── agents/                # Specialized agents
│   ├── orchestrator.py    # Planning + routing
│   ├── retriever.py       # ChromaDB RAG
│   ├── analyst.py         # Reasoning + synthesis
│   ├── verifier.py        # Grounding + confidence
│   └── memory.py          # Session context
├── graph/workflow.py      # LangGraph StateGraph wiring
├── vector_store/ingest.py # Document ingestion → ChromaDB
├── guardrails/checks.py   # Input validation + disclaimers
├── evaluation/logger.py   # Structured JSON evaluation logs
├── ui/app.py              # Streamlit frontend
├── tests/                 # Unit tests
├── docs/                  # Sample enterprise documents
├── logs/                  # JSON evaluation logs (runtime)
├── agents.md              # Master plan (immutable)
├── progress.md            # Live progress tracker
├── pyproject.toml         # Dependencies
└── README.md              # This file
```

See [`agents.md`](./agents.md) for the full agent design and the step-by-step user stories.

---

## Prerequisites

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) package manager (recommended) **or** `pip`
- A **Gemini API key** — get one at <https://aistudio.google.com/apikey>

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd Atlas
```

### 2. Install dependencies

With `uv` (recommended):

```bash
uv sync
```

With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -e .
```

### 3. Configure environment

```bash
cp .env.example .env
```

Then edit `.env` and set your real `GEMINI_API_KEY`:

```bash
GEMINI_API_KEY=your_real_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

> Never commit `.env`. It is already covered by `.gitignore`.

### 4. Add sample documents

Drop at least 2–3 enterprise-style documents (PDFs and/or `.txt`) into `docs/`. Example set:

- `policy_hr.pdf`
- `sop_onboarding.pdf`
- `compliance_manual.txt`

### 5. Ingest documents into ChromaDB

```bash
uv run python vector_store/ingest.py
```

This loads every file in `docs/`, chunks it, embeds it with Gemini, and writes it to a persistent `chroma_db/` directory. Re-running does not duplicate chunks.

### 6. Launch the Streamlit UI

```bash
uv run streamlit run ui/app.py
```

The UI exposes:

- **Ingest Documents** button (sidebar)
- **Session Reset** button (sidebar)
- **Model info** (sidebar)
- Query input + **Ask** button (main area)
- 4 tabs in the response: **Answer**, **Agent Trace**, **Sources**, **Evaluation Log**
- Confidence badge (green / yellow / red)
- Warning banner if any guardrail was triggered

---

## How the agents work together

```
START
  ↓
orchestrate_node  → OrchestratorAgent builds a plan
  ↓
retrieve_node     → RetrieverAgent queries ChromaDB
  ↓
analyze_node      → AnalystAgent reasons over chunks
  ↓
verify_node       → VerifierAgent scores grounding
  ↓                  (confidence < 0.6 ?)
  ├── yes → low_confidence_node (append disclaimer)
  └── no  ──────────────┐
                        ↓
                  finalize_node
                        ↓
                  memory_node
                        ↓
                       END
```

Every node writes to the `decision_trace` field of `AgentState` so the full reasoning path is inspectable.

---

## Documentation

| File                                  | Purpose                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------ |
| [`agents.md`](./agents.md)            | Master plan, agent roles, LangGraph schema, all 12 implementation stories |
| [`progress.md`](./progress.md)        | Live progress tracker, blockers, deviations, file change log              |
| `ARCHITECTURE.md` (Story 12)          | Architecture diagram, design decisions, trade-offs                        |
| `EVALUATION.md` (Story 12)            | Guardrails, grounding, confidence threshold, failure modes               |
| `UNIT_TESTS.md` (Story 12)            | What each test covers, how to run them, expected outcomes                 |

---

## Testing

```bash
uv run pytest tests/
```

---

## License

Internal case-study project.
