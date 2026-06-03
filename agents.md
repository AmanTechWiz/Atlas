# Enterprise Knowledge Ops Agent
### Cognizant Skillspring — Agentic AI Developer Case Study
### Implementation Plan & Context File (for AI Agent / Developer Reference)

---

## 0. What This File Is

This is a complete context + step-by-step implementation plan for building the
Enterprise Knowledge Operations Agent. Any AI coding agent (or developer) reading
this file should be able to implement the full system from scratch by following
the user stories in order. Each user story maps directly to the evaluation rubric.

---

## 0.5. How an Incoming Agent Should Start

If you are a coding agent picking up this project, do the following before
writing a single line of code:

1. Read `agents.md` fully — understand the full plan and tech stack.
2. Read `progress.md` fully — understand exactly what's done and what isn't.
3. Check the "Currently In Progress" section — start from that exact step.
4. Check "Deviations" — the actual code may differ from `agents.md` in places.
5. Check "Blockers" — do not retry a blocked approach without reading why it failed.
6. Verify the environment state matches what's documented.
7. Only then start writing code.

**Handoff protocol reminder:** When you complete an official US (Section 12),
end your message with the literal line `US(x) completed` and **do not** update
`progress.md` until the user replies with `ok let's move to next us` (or close).
See rule #0a in Section 11 for full details.

---

## 1. Project Summary (Plain English)

Build a **local-first, multi-agent AI system** that answers complex business
questions by reasoning across multiple enterprise documents (policies, SOPs,
contracts, compliance manuals). The system must NOT be a single chatbot — it
must be a coordinated pipeline of specialized agents, each with a clearly
defined role. The application, document store, vector database, logs, and UI run
locally; for now, LLM and embedding calls use the Gemini API via a user-provided
`GEMINI_API_KEY`.

The system must be able to:
- Plan how to answer a complex question (not just keyword match)
- Delegate subtasks to the right agent
- Retrieve relevant document chunks from a vector store
- Reason and synthesize across multiple documents
- Verify that the final answer is grounded in source documents
- Flag low-confidence or hallucinated responses
- Log every decision trace for explainability
- Expose all of this through a Streamlit UI

---

## 2. Final Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Language | Python 3.11+ | Standard for ML/AI tooling |
| Agent Framework | LangGraph | Explicit state machine, multi-agent support, traceable |
| LLM | Gemini API (`gemini-1.5-flash` or current approved Gemini model) | Current user-approved default; strong reasoning with simple API setup |
| Embeddings | Gemini embeddings (`models/text-embedding-004`) | Uses the same provider as the LLM and works with ChromaDB |
| Vector Database | ChromaDB | Local persistent storage, simple setup |
| Document Parsing | LangChain document loaders (PyPDFLoader, TextLoader) | Standard, well-documented |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Reliable chunking |
| Memory / Session | LangGraph state + in-memory dict (ConversationBufferMemory) | Sufficient for local demo |
| Guardrails | Custom Python validation layer | Input sanitization + confidence thresholding |
| Evaluation Logging | Python logging module → structured JSON logs | Meets observability criteria |
| UI | Streamlit | Fast to build, shows agent traces visually |
| Dependency Management | pip + requirements.txt | Simple |

**Tech stack decision rule:** Before implementing each major story, the coding
agent should confirm with the user if a technology choice is still acceptable.
The current approved LLM/embedding provider is Gemini API. Do not switch to
Ollama, Azure OpenAI, OpenAI API, or another provider unless the user explicitly
approves the change.

---

## 3. Project Folder Structure

```
enterprise-knowledge-ops-agent/
│
├── docs/                          # Sample enterprise documents to ingest
│   ├── policy_hr.pdf
│   ├── sop_onboarding.pdf
│   └── compliance_manual.txt
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py            # OrchestratorAgent — planning + routing
│   ├── retriever.py               # RetrieverAgent — ChromaDB RAG
│   ├── analyst.py                 # AnalystAgent — reasoning + synthesis
│   ├── verifier.py                # VerifierAgent — grounding + validation
│   └── memory.py                  # MemoryAgent — session context management
│
├── graph/
│   ├── __init__.py
│   └── workflow.py                # LangGraph StateGraph wiring all agents
│
├── vector_store/
│   ├── __init__.py
│   └── ingest.py                  # Document ingestion + ChromaDB population
│
├── guardrails/
│   ├── __init__.py
│   └── checks.py                  # Input validation, hallucination controls
│
├── evaluation/
│   ├── __init__.py
│   └── logger.py                  # Structured JSON evaluation logger
│
├── ui/
│   └── app.py                     # Streamlit frontend
│
├── tests/
│   ├── test_retriever.py
│   ├── test_verifier.py
│   └── test_orchestrator.py
│
├── logs/                          # Auto-generated evaluation logs (JSON)
├── requirements.txt
├── README.md
├── agents.md                      # This file — master plan
└── progress.md                    # Live progress tracker (see Section 11)
```

---

## 4. Agent Roles (Detailed)

### OrchestratorAgent
- Entry point for every user query
- Analyzes the query and produces a structured execution plan
- Decides which agents to invoke and in what order
- Routes subtasks: retrieval first, then analysis, then verification
- Logs its planning steps explicitly

### RetrieverAgent
- Receives a query or subquery from the Orchestrator
- Queries ChromaDB using Gemini embeddings
- Returns top-k document chunks with metadata (source, page, relevance score)
- Preserves source attribution throughout

### AnalystAgent
- Receives retrieved chunks from the Retriever
- Uses Gemini API to reason across multiple chunks
- Synthesizes a coherent answer with cross-document reasoning
- Does NOT invent — it only reasons over what was retrieved

### VerifierAgent
- Receives the Analyst's draft answer + the source chunks
- Checks whether every claim in the answer is grounded in the sources
- Assigns a confidence score (0.0 to 1.0)
- If confidence < 0.6, flags the response and attaches a disclaimer
- Detects and flags conflicting agent outputs

### MemoryAgent
- Maintains a session-level conversation buffer
- Stores prior queries and answers within a session
- Injects relevant prior context into the Orchestrator's planning step
- Resets between sessions

---

## 5. LangGraph State Schema

```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str                        # Original user query
    plan: List[str]                   # Orchestrator's execution plan steps
    retrieved_chunks: List[dict]      # [{text, source, page, score}]
    draft_answer: str                 # Analyst's synthesized answer
    verification_result: dict         # {grounded: bool, confidence: float, flags: list}
    final_answer: str                 # Answer shown to user
    decision_trace: List[str]         # Full log of agent decisions
    session_history: List[dict]       # Memory agent's context
    error: Optional[str]              # Any failure state
```

---

## 6. User Stories (Implementation Order)

Follow these exactly in order. Each story is self-contained and builds on the previous.

---

### USER STORY 0 — Environment Setup
**Goal:** Get the full stack running locally before writing any agent code.

**Steps:**
1. Create the project folder structure as defined in Section 3.
2. Create `requirements.txt` with the following:
   ```
   langchain
   langchain-community
   langchain-google-genai
   langgraph
   chromadb
   streamlit
   pypdf
   python-dotenv
   ```
3. Run `pip install -r requirements.txt`
4. Create a `.env` file at project root:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   GEMINI_EMBEDDING_MODEL=models/text-embedding-004
   ```
5. Ensure all LLM/embedding modules load environment variables with
   `python-dotenv` and fail clearly if `GEMINI_API_KEY` is missing.
   Do not commit `.env`; document `.env.example` instead if needed.
6. Create a `docs/` folder and add at least 3 sample enterprise documents
   (PDFs or .txt files — can be dummy policy/SOP content for now).
7. Create empty `__init__.py` in all package folders.

**Done when:** `python -c "import langchain; import chromadb; import langgraph; import langchain_google_genai"` runs without errors.

---

### USER STORY 1 — Document Ingestion & Vector Store
**Goal:** Ingest enterprise documents into ChromaDB so they can be retrieved.

**File:** `vector_store/ingest.py`

**Steps:**
1. Use `PyPDFLoader` and `TextLoader` from `langchain_community.document_loaders`
   to load all files in the `docs/` folder.
2. Split documents using `RecursiveCharacterTextSplitter` with:
   - `chunk_size=500`
   - `chunk_overlap=50`
3. Generate embeddings using
   `GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))`.
4. Store chunks in ChromaDB with a persistent path (`./chroma_db`).
5. Preserve metadata: `source` (filename), `page` number.
6. Print confirmation: number of chunks ingested per document.
7. Make this runnable as a standalone script: `python vector_store/ingest.py`

**Acceptance criteria:**
- ChromaDB persists to disk after running the script
- Re-running does not duplicate chunks (use collection `get_or_create`)
- Each chunk has `source` and `page` metadata

---

### USER STORY 2 — RetrieverAgent
**Goal:** Build the agent that queries ChromaDB and returns relevant chunks.

**File:** `agents/retriever.py`

**Steps:**
1. Load the persisted ChromaDB collection using the same embedding model.
2. Implement a `retrieve(query: str, k: int = 5) -> List[dict]` function.
3. Each returned dict must contain: `{text, source, page, relevance_score}`.
4. Relevance score: ChromaDB returns distances — convert to a 0–1 score
   (`score = 1 - distance` for cosine distance).
5. Filter out chunks with relevance score < 0.3 (poor retrieval).
6. Log: "Retrieved N chunks for query: [query]" with sources listed.

**Acceptance criteria:**
- Given a test query, returns 3–5 relevant chunks
- Each chunk has source attribution
- Low-relevance chunks are filtered

---

### USER STORY 3 — AnalystAgent
**Goal:** Build the agent that reasons across retrieved chunks to form an answer.

**File:** `agents/analyst.py`

**Steps:**
1. Use `ChatGoogleGenerativeAI` from `langchain_google_genai`, with
   `model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash")`.
2. Build a prompt template that:
   - Injects the retrieved chunks as context (with source labels)
   - Instructs the model to reason across documents, not just extract
   - Explicitly tells the model: "Do not use any knowledge outside the provided context"
   - Asks for a structured answer: [Reasoning], [Answer], [Sources Used]
3. Implement `analyze(query: str, chunks: List[dict]) -> str`.
4. The output must always include which sources were used.

**Acceptance criteria:**
- Answer references multiple chunks when available
- Output format always includes Reasoning, Answer, Sources Used sections
- Model does not hallucinate beyond provided context (enforced by prompt)

---

### USER STORY 4 — VerifierAgent
**Goal:** Build the agent that checks if the answer is grounded in source chunks.

**File:** `agents/verifier.py`

**Steps:**
1. Implement `verify(draft_answer: str, chunks: List[dict]) -> dict`.
2. Use a second Gemini LLM call with a verification prompt:
   - "Given the following source documents and a generated answer,
     rate how well the answer is grounded in the sources on a scale of 0.0 to 1.0.
     Return JSON: {confidence: float, grounded: bool, flags: list[str]}"
3. Parse the JSON response.
4. If `confidence < 0.6`:
   - Set `grounded = False`
   - Add flag: "LOW_CONFIDENCE — answer may not be fully supported by documents"
5. If retrieval was poor (< 2 chunks), add flag: "INSUFFICIENT_RETRIEVAL"
6. Return the full verification dict.

**Acceptance criteria:**
- Returns structured dict with confidence score
- Flags low-confidence responses
- Never suppresses a flag to make the answer look better

---

### USER STORY 5 — MemoryAgent
**Goal:** Build session-level memory so multi-turn conversations have context.

**File:** `agents/memory.py`

**Steps:**
1. Implement a simple in-memory store: `session_history: List[dict]`
   where each entry is `{query, answer, sources, timestamp}`.
2. Implement `add_to_memory(query, answer, sources)`.
3. Implement `get_context(last_n=3) -> str` — returns last N Q&A pairs
   formatted as a string for injection into the Orchestrator's planning step.
4. Implement `reset()` — clears session (called on new session start).

**Acceptance criteria:**
- Multi-turn: second query can reference context from the first
- Memory resets cleanly between sessions
- Memory context is injected into planning, not directly into the LLM prompt

---

### USER STORY 6 — OrchestratorAgent
**Goal:** Build the planning + routing brain that coordinates all other agents.

**File:** `agents/orchestrator.py`

**Steps:**
1. Implement `plan(query: str, session_context: str) -> List[str]`.
2. Use a Gemini LLM call to decompose the query into subtasks. Prompt:
   - "You are an orchestrator. Given this query and session context,
     produce a numbered list of steps to answer it. Steps must specify
     which agent handles each step: [RETRIEVE], [ANALYZE], [VERIFY], [MEMORY]."
3. Parse the plan into a list of strings.
4. Log the plan explicitly: "Orchestrator plan: step 1... step 2..."
5. The orchestrator does NOT call agents directly — it returns the plan.
   The LangGraph workflow (User Story 7) handles actual invocation.

**Acceptance criteria:**
- Plan always includes at least RETRIEVE → ANALYZE → VERIFY sequence
- Plan is logged and visible in decision trace
- Planning step is distinct from execution (separation of concerns)

---

### USER STORY 7 — LangGraph Workflow (Wiring Everything Together)
**Goal:** Wire all agents into a LangGraph StateGraph with explicit transitions.

**File:** `graph/workflow.py`

**Steps:**
1. Import `StateGraph` from `langgraph.graph`.
2. Use `AgentState` (defined in Section 5) as the state schema.
3. Define nodes — one per agent:
   - `orchestrate_node`: calls OrchestratorAgent, updates `plan` + `decision_trace`
   - `retrieve_node`: calls RetrieverAgent, updates `retrieved_chunks`
   - `analyze_node`: calls AnalystAgent, updates `draft_answer`
   - `verify_node`: calls VerifierAgent, updates `verification_result`
   - `memory_node`: calls MemoryAgent, updates `session_history`
   - `finalize_node`: assembles `final_answer` from draft + verification result
4. Define edges (explicit flow):
   ```
   START → orchestrate_node
   orchestrate_node → retrieve_node
   retrieve_node → analyze_node
   analyze_node → verify_node
   verify_node → finalize_node
   finalize_node → memory_node
   memory_node → END
   ```
5. Add a conditional edge after `verify_node`:
   - If `confidence < 0.6`: route to a `low_confidence_node` that appends
     a disclaimer to the answer before finalization
   - Else: route directly to `finalize_node`
6. Compile the graph: `app = workflow.compile()`
7. Expose a `run_query(query: str) -> AgentState` function that invokes the graph.

**Acceptance criteria:**
- Full query runs end-to-end through all 5 agents
- Decision trace in state captures every node's action
- Conditional routing for low-confidence responses works
- State is fully populated at END

---

### USER STORY 8 — Guardrails Layer
**Goal:** Add input validation and safety checks before queries hit the graph.

**File:** `guardrails/checks.py`

**Steps:**
1. Implement `validate_input(query: str) -> dict`:
   - Returns `{valid: bool, reason: str}`
   - Reject empty queries
   - Reject queries under 5 characters
   - Reject queries containing prompt injection patterns
     (e.g., "ignore previous instructions", "you are now", "jailbreak")
   - Reject queries entirely unrelated to enterprise document topics
     (basic keyword heuristic — if no business/document keywords present,
     flag as out-of-scope)
2. Implement `apply_confidence_guardrail(verification_result: dict, answer: str) -> str`:
   - Wraps answer with disclaimer if confidence < 0.6
   - Adds source attribution footer to every answer
3. These are called in `graph/workflow.py` before the graph runs
   and inside `finalize_node` respectively.

**Acceptance criteria:**
- Prompt injection attempts are blocked with a clear message
- Every final answer has source attribution
- Low confidence answers are visibly flagged, not silently returned

---

### USER STORY 9 — Evaluation & Observability Logger
**Goal:** Log every agent decision in structured JSON for inspection.

**File:** `evaluation/logger.py`

**Steps:**
1. Implement `EvalLogger` class with:
   - `log_query_start(query)` — logs query + timestamp
   - `log_plan(plan)` — logs orchestrator's plan
   - `log_retrieval(chunks)` — logs chunk count, sources, scores
   - `log_analysis(draft_answer)` — logs analyst output
   - `log_verification(result)` — logs confidence, grounded status, flags
   - `log_final(final_answer, total_time_ms)` — logs final answer + latency
   - `log_failure(error, stage)` — logs what failed and where
2. All logs written to `logs/eval_{timestamp}.json`
3. Log format per entry:
   ```json
   {
     "timestamp": "ISO8601",
     "stage": "RETRIEVAL",
     "event": "retrieved 4 chunks",
     "data": {...}
   }
   ```
4. At the end of each query, write a summary log with:
   - Query, plan, retrieval count, confidence score, grounded bool, final answer

**Acceptance criteria:**
- Every query produces a log file
- Log contains entries for every agent stage
- Failures are logged with stage name and error message
- Logs are valid JSON (parseable)

---

### USER STORY 10 — Streamlit UI
**Goal:** Build a Streamlit app that exposes the system and shows agent traces.

**File:** `ui/app.py`

**Steps:**
1. Build a sidebar with:
   - "Ingest Documents" button (triggers `vector_store/ingest.py`)
   - Session reset button (clears MemoryAgent)
   - Model info display (which Gemini model is active)
2. Main area:
   - Text input for user query
   - "Ask" button
   - Spinner while agents run
3. After response, display in tabs:
   - **Tab 1 — Answer:** Final answer with confidence badge
     (green ≥ 0.7, yellow 0.5–0.7, red < 0.5)
   - **Tab 2 — Agent Trace:** Expandable sections per agent showing
     what each agent did (plan, chunks retrieved, reasoning, verification)
   - **Tab 3 — Sources:** List of source documents used with page numbers
   - **Tab 4 — Evaluation Log:** Raw JSON log for the current query
4. Display a warning banner if any guardrail was triggered.
5. Display session history in a collapsible sidebar section.

**Acceptance criteria:**
- Full query runs and displays in UI
- Agent trace tab shows all 5 agents' actions
- Confidence badge is color-coded correctly
- Sources are always displayed
- Evaluation log tab shows the raw JSON

---

### USER STORY 11 — Unit Tests
**Goal:** Basic test coverage for the 3 most critical components.

**Files:** `tests/test_retriever.py`, `tests/test_verifier.py`, `tests/test_orchestrator.py`

**Steps:**
1. `test_retriever.py`:
   - Test that `retrieve()` returns results for a known query
   - Test that results have required keys: text, source, page, relevance_score
   - Test that score < 0.3 results are filtered out
2. `test_verifier.py`:
   - Test that `verify()` returns required keys: confidence, grounded, flags
   - Test that confidence < 0.6 sets grounded = False
   - Test that insufficient chunks triggers INSUFFICIENT_RETRIEVAL flag
3. `test_orchestrator.py`:
   - Test that `plan()` returns a non-empty list
   - Test that plan always contains at least RETRIEVE, ANALYZE, VERIFY steps

**Acceptance criteria:**
- All tests pass with `pytest tests/`
- Tests are independent (no shared state)

---

### USER STORY 12 — Documentation
**Goal:** Produce the required deliverables for the evaluation rubric.

**Files to create:**

1. `README.md` — Setup instructions, how to run ingestion, how to run UI,
   what each agent does, system requirements, and required Gemini `.env` keys

2. `ARCHITECTURE.md` — Agent flow diagram (ASCII or Mermaid), explanation of
   LangGraph state transitions, design decisions and trade-offs

3. `EVALUATION.md` — What guardrails are implemented, how grounding works,
   confidence threshold rationale, known failure modes and how they're handled

4. `UNIT_TESTS.md` — What each test covers, how to run them, expected outcomes

---

## 7. Evaluation Rubric Mapping

| Rubric Category | Points | Covered By |
|---|---|---|
| Agentic Architecture & Design | 20 | User Stories 2–6, Section 4 |
| Query Planning & Orchestration | 15 | User Story 6 (Orchestrator) |
| Retrieval & RAG Effectiveness | 15 | User Story 1–2 (Ingest + Retriever) |
| Reasoning & Synthesis Quality | 15 | User Story 3 (Analyst) |
| Validation, Grounding & Guardrails | 15 | User Stories 4, 8 (Verifier + Guardrails) |
| Evaluation & Observability | 10 | User Story 9 (Logger) |
| Documentation & Explainability | 10 | User Story 12 |
| **Total** | **100** | |

---

## 8. Key Design Decisions (Explain These in Your Presentation)

1. **LangGraph over LangChain AgentExecutor** — LangGraph gives explicit,
   inspectable state transitions. Every node's input/output is logged in state.
   This directly satisfies the "traceable agent interactions" criterion.

2. **Separate VerifierAgent** — Most RAG systems skip this. Having a dedicated
   verification step with a confidence score is what separates a "Good" (11–15)
   from an "Excellent" (16–20) architecture score.

3. **ChromaDB with persistent storage** — Documents are ingested once,
   ChromaDB persists to disk. The system doesn't re-embed on every run.

4. **Local-first with Gemini API** — The app, vector database, documents, logs,
   evaluation, and UI run locally. The current user-approved exception is Gemini
   API for LLM and embedding calls via `GEMINI_API_KEY`. Any switch to Ollama,
   Azure OpenAI, OpenAI API, or another provider must be confirmed with the user.

5. **Structured JSON evaluation logs** — Every query leaves an audit trail.
   This is required for the Evaluation & Observability criterion (10 points).

---

## 9. Common Failure Modes to Handle

| Failure | Stage | Handling |
|---|---|---|
| No relevant documents found | Retriever | Return 0 chunks, flag INSUFFICIENT_RETRIEVAL, do not hallucinate |
| LLM returns non-JSON for verification | Verifier | Try/except, default to confidence=0.5 with flag PARSE_ERROR |
| Gemini API key missing or invalid | Any LLM/embedding call | Validate `GEMINI_API_KEY`, catch provider errors, log failure, return user-friendly error |
| Gemini rate limit or network failure | Any LLM/embedding call | Retry once if safe, then log failure and ask user to retry later |
| Query is out of scope | Guardrails | Block before graph runs, return "Query not related to enterprise documents" |
| ChromaDB collection empty | Retriever | Check collection count before querying, prompt user to ingest first |

---

## 10. Implementation Order Summary

```
Story 0  → Environment + folder structure
Story 1  → Document ingestion (ChromaDB)
Story 2  → RetrieverAgent
Story 3  → AnalystAgent
Story 4  → VerifierAgent
Story 5  → MemoryAgent
Story 6  → OrchestratorAgent
Story 7  → LangGraph workflow (wire everything)
Story 8  → Guardrails
Story 9  → Evaluation logger
Story 10 → Streamlit UI
Story 11 → Unit tests
Story 12 → Documentation
```

Total estimated time for a focused developer: **4–6 days**
Each story is independently testable before moving to the next.

---

## 11. Progress Tracking Protocol (MANDATORY FOR ALL CODING AGENTS)

### What This Is

`progress.md` is a live file that the currently active coding agent MUST
maintain at all times. Its purpose is to enable seamless handoff between
coding agents or sessions — any new agent can read `agents.md` + `progress.md`
and continue exactly where the previous agent left off without re-doing work
or breaking existing functionality.

### Rules for the Active Coding Agent

0. **Before starting to build any new module, file, or non-trivial change, ask
   the user for explicit go-ahead.** Use the standard prompt:
   *"Should I start building `<name of thing>`?"* Wait for a clear yes/no
   before writing any code. This applies to every new file, every new agent,
   every new tab in the UI, every refactor, and every push to a new branch.
   (User-added rule, 2026-06-04.)

0a. **Handoff protocol for official US completion.** When you complete one of
   the **6 official Cognizant user stories** (Section 12), do the following:
   - At the very end of your completion message, write the literal line
     `US(x) completed` where `x` is the official US number.
   - **Do NOT update `progress.md` yet.** Wait for the user to reply with
     the literal phrase `ok let's move to next us` (or close to it).
   - On that reply: first update `progress.md` (checkboxes, summary, file
     log, deviations, environment state), commit, push; then ask the user
     which official US to build next.
   - This rule is more specific than rule #1 — it says the user explicitly
   acknowledges the US is done before `progress.md` is touched. (User-added
   rule, 2026-06-04.)

1. **Update `progress.md` after completing every User Story** — not at the end
   of the session, after each story.

2. **Update `progress.md` before ending any session** — even mid-story.
   Record exactly what was done, what was NOT done, and what the next step is.

3. **Never modify `agents.md`** — that is the immutable master plan.
   All live state goes in `progress.md`.

4. **If you encounter a blocker**, log it in `progress.md` under BLOCKERS
   with full context so the next agent doesn't waste time rediscovering it.

5. **If you deviate from the plan in `agents.md`** (e.g., used a different
   library, changed a function signature), document the deviation in
   `progress.md` under DEVIATIONS. The next agent must know the actual
   state of the code, not just the plan.

### `progress.md` Template (create this file at project root on Story 0)

```markdown
# Project Progress — Enterprise Knowledge Ops Agent

## Last Updated
[ISO timestamp] by [Agent name / session ID]

## Overall Status
[ ] Story 0  — Environment Setup
[ ] Story 1  — Document Ingestion
[ ] Story 2  — RetrieverAgent
[ ] Story 3  — AnalystAgent
[ ] Story 4  — VerifierAgent
[ ] Story 5  — MemoryAgent
[ ] Story 6  — OrchestratorAgent
[ ] Story 7  — LangGraph Workflow
[ ] Story 8  — Guardrails
[ ] Story 9  — Evaluation Logger
[ ] Story 10 — Streamlit UI
[ ] Story 11 — Unit Tests
[ ] Story 12 — Documentation

Mark as [x] when fully complete and tested.

## Currently In Progress
Story N — [Name]
Step currently at: [exact step number from agents.md]
What has been done in this story so far:
- ...

## What To Do Next (for incoming agent)
[Specific next action — file to create, function to write, command to run]

## Completed Stories Summary
### Story 0 ✅
- Completed: [date]
- Notes: [anything relevant]

## Blockers
[Any issue that stopped progress — library errors, model issues, etc.]
Format: BLOCKER [date]: [description] — [what was tried] — [status: open/resolved]

## Deviations from agents.md
[Any place where the implementation differs from the plan]
Format: DEVIATION [story N]: [what agents.md says] → [what was actually done] — [reason]

## Environment State
- OS: [Ubuntu 22.04 / macOS / Windows WSL2]
- Python version: [e.g., 3.11.4]
- Gemini API key configured: [yes/no]
- Gemini model: [e.g., gemini-1.5-flash]
- Gemini embedding model: [e.g., models/text-embedding-004]
- ChromaDB populated: [yes/no — N chunks from M documents]

## File Change Log
[List every file created or modified, most recent first]
- [timestamp] CREATED agents/retriever.py
- [timestamp] MODIFIED requirements.txt — added chromadb==0.5.0
```

(See Section 0.5 at the top of this file for the full "How an Incoming Agent
Should Start" checklist.)

---

## 12. Official Cognizant User Stories (from Case Study Document)

These are the **6 user stories as defined by Cognizant** in the case study PDF.
These are what the evaluator will read. Every acceptance criterion listed here
must be demonstrably satisfied in the final submission.

The implementation stories (0–12) in this file are the *how* — these 6 are the *what*.

---

### Official User Story 1 — Complex Query Handling

> As a business user, I want to ask complex questions across multiple enterprise
> documents to receive accurate, well-reasoned, and explainable answers rather
> than single-pass responses.

**Acceptance Criteria:**
- The system accepts queries that require reasoning across more than one document
- The query is decomposed into logical subtasks by an orchestrator agent
- Relevant documents are retrieved and used to synthesize a final answer
- The final response is coherent and clearly answers the original question

**Satisfied by implementation stories:** 1 (Ingestion), 2 (Retriever), 3 (Analyst), 6 (Orchestrator), 7 (LangGraph Workflow)

---

### Official User Story 2 — Agent Planning and Orchestration

> As a developer, I want the system to plan and route tasks to specialized agents
> so that each step of reasoning is handled by the most appropriate agent.

**Acceptance Criteria:**
- A planning or orchestrator agent determines the sequence of actions needed
- Distinct agents are invoked for retrieval, reasoning, validation, and memory
- Agent interactions are traceable and logged for inspection

**Satisfied by implementation stories:** 6 (Orchestrator), 7 (LangGraph Workflow), 9 (Evaluation Logger)

---

### Official User Story 3 — Grounded and Validated Responses

> As a business user, I want confidence that responses are grounded in source
> documents so that I can trust the system's output.

**Acceptance Criteria:**
- Retrieved sources are explicitly linked to the generated answer
- A verifier or validation agent checks grounding before final output
- If grounding confidence is low, the system flags or limits the response

**Satisfied by implementation stories:** 2 (Retriever — source attribution), 4 (Verifier), 8 (Guardrails)

---

### Official User Story 4 — Explainability and Transparency

> As a learner or reviewer, I want visibility into how the system arrived at an
> answer so that I can understand and evaluate agentic decision-making.

**Acceptance Criteria:**
- The system exposes agent decision traces or reasoning summaries
- Retrieval results and validation outcomes are logged and viewable
- The explanation aligns with the final response and source documents

**Satisfied by implementation stories:** 9 (Evaluation Logger), 10 (Streamlit UI — Agent Trace tab), 7 (LangGraph decision_trace state field)

---

### Official User Story 5 — Governance and Guardrails

> As an enterprise stakeholder, I want guardrails in place so that the system
> minimizes hallucinations and handles uncertainty responsibly.

**Acceptance Criteria:**
- Input validation and basic safety checks are implemented
- The system applies hallucination controls or confidence thresholds
- Clear source attribution or disclaimers are provided in responses

**Satisfied by implementation stories:** 8 (Guardrails), 4 (Verifier — confidence thresholding), 2 (Retriever — source attribution)

---

### Official User Story 6 — Evaluation, Observability, and Failure Detection

> As a learner or system evaluator, I want the system to implement explicit
> evaluation mechanisms — such as grounding checks, retrieval relevance scoring,
> agent decision traces, and failure detection — so that I can assess the
> reliability, transparency, and correctness of agentic responses.

**Acceptance Criteria:**
- The system performs grounding checks to verify that generated responses are
  supported by retrieved source documents
- Retrieval relevance scores or signals are captured for the documents used
  in answering a query
- Agent decision traces (planning steps, agent invocation order, key decisions)
  are logged and accessible
- The system can detect and flag failures: insufficient retrieval, low grounding
  confidence, conflicting agent outputs
- Evaluation outputs are logged in a structured and inspectable format
  (console logs, files, or UI view)
- The system provides a clear explanation of evaluation results alongside the
  final response

**Satisfied by implementation stories:** 9 (Evaluation Logger), 4 (Verifier), 2 (Retriever — relevance scores), 7 (LangGraph — decision_trace), 10 (Streamlit — Evaluation Log tab)

---

### Official User Story → Implementation Story Cross-Reference

| Official US | Title | Key Implementation Stories |
|---|---|---|
| US 1 | Complex Query Handling | Stories 1, 2, 3, 6, 7 |
| US 2 | Agent Planning & Orchestration | Stories 6, 7, 9 |
| US 3 | Grounded & Validated Responses | Stories 2, 4, 8 |
| US 4 | Explainability & Transparency | Stories 7, 9, 10 |
| US 5 | Governance & Guardrails | Stories 4, 8 |
| US 6 | Evaluation, Observability & Failure Detection | Stories 2, 4, 7, 9, 10 |

---

## 13. Evaluation Rubric Mapping (Updated)

| Rubric Category | Points | Covered By |
|---|---|---|
| Agentic Architecture & Design | 20 | Impl. Stories 2–6, Section 4 |
| Query Planning & Orchestration | 15 | Impl. Story 6 (Orchestrator) |
| Retrieval & RAG Effectiveness | 15 | Impl. Stories 1–2 (Ingest + Retriever) |
| Reasoning & Synthesis Quality | 15 | Impl. Story 3 (Analyst) |
| Validation, Grounding & Guardrails | 15 | Impl. Stories 4, 8 (Verifier + Guardrails) |
| Evaluation & Observability | 10 | Impl. Story 9 (Logger) |
| Documentation & Explainability | 10 | Impl. Story 12 |
| **Total** | **100** | |

---

*This file is the immutable master plan for the Cognizant Skillspring
Agentic AI Developer Case Study. Do not modify it during implementation.
All live progress state belongs in `progress.md`.*
