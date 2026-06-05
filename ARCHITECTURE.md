# Architecture

This document describes the design of the multi-agent pipeline, the
runtime state, the agent roles, and the trade-offs that shaped them.
It is meant to be read by reviewers who want to understand *how* the
system works end-to-end, not just *what* it produces.

---

## 1. High-level flow

```
                        +-------------------+
                        |  user query +     |
                        |  session memory   |
                        +---------+---------+
                                  |
                                  v
                       +----------+----------+
                       |   ORCHESTRATOR      |
                       |   (plan steps)      |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |    RETRIEVER        |
                       |  (ChromaDB lookup)  |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |     ANALYST         |
                       |  (synthesize draft) |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |     VERIFIER        |
                       |  (3-axis score)     |
                       +----------+----------+
                                  |
                       confidence < 0.6 ?
                       /               \
                     yes                 no
                     /                    \
        +------------+------+    +---------+----------+
        | LOW_CONFIDENCE    |    |     FINALIZE       |
        | (set disclaimer)  |    |  (wrap + sources)  |
        +------------+------+    +---------+----------+
                     \                   /
                      `------+---------'
                             v
                  +----------+----------+
                  |       MEMORY         |
                  |  (record this turn)  |
                  +----------+----------+
                             v
                            END
```

The pipeline is implemented as a LangGraph `StateGraph` with one node
per agent plus a `low_confidence` node for the conditional disclaimer
path. Every node writes a `decision_trace` entry and an `EvalLogger`
JSON entry, so the runtime behaviour is fully reconstructable from
the audit log.

---

## 2. Runtime state schema

`graph/workflow.py:AgentState` is a `TypedDict` (with `total=False`)
that flows through every node. The schema is intentionally small —
only the fields a node actually reads or writes.

| Field | Type | Owner (writer) | Notes |
|---|---|---|---|
| `query` | `str` | `run_query` | The original user query. |
| `plan` | `List[str]` | `orchestrate_node` | Numbered steps with `[RETRIEVE]`/`[ANALYZE]`/`[VERIFY]`/`[MEMORY]` tags. |
| `retrieved_chunks` | `List[dict]` | `retrieve_node` | `{text, source, page, relevance_score, _distance}`. |
| `draft_answer` | `str` | `analyze_node` | The analyst's `[Reasoning]/[Answer]/[Sources Used]` blob. |
| `verification_result` | `dict` | `verify_node` | `{confidence, grounded, flags, grounding_confidence, answer_quality, retrieval_confidence, claims, question_aspects}`. |
| `final_answer` | `str` | `finalize_node` | The user-facing answer (with disclaimer + sources footer). |
| `decision_trace` | `List[str]` | every node | One-line summary of what the node did. |
| `session_history` | `List[dict]` | `memory_node` | The MemoryAgent's full history. |
| `error` | `Optional[str]` | any node on failure | Internal error code, e.g. `analyst_failed: ...`. |
| `api_error` | `Optional[str]` | any node on LLM failure | User-friendly error message. |
| `needs_disclaimer` | `bool` | `low_confidence_node` | Set when `verification_result.confidence < 0.6`. |
| `eval_logger` | `EvalLogger` | `run_query` | Carried by reference; the same instance writes to one log file for the whole query. |
| `query_start_mono` | `float` | `run_query` | `time.monotonic()` at the start of the run, used for `total_time_ms`. |
| `memory` | `MemoryAgent` | caller | Per-session memory instance. |

A `log_path` field is added to the result after `app.invoke()` returns,
giving the caller the on-disk path of the per-query JSON log.

---

## 3. Agent roles

### 3.1 OrchestratorAgent — `agents/orchestrator.py`

**Role:** Decompose the user query into a numbered plan of agent
steps. Does NOT execute the steps — that is the workflow's job.

**API:** `plan(query, session_context="") -> List[str]`

**Design notes:**

- Uses a single LLM call with temperature=0.0. The prompt is short
  and forces the model to output one tag per line (`[RETRIEVE]`,
  `[ANALYZE]`, `[VERIFY]`, `[MEMORY]`).
- Safety fallback: if the LLM omits any of the required tags
  (`[RETRIEVE]`, `[ANALYZE]`, `[VERIFY]`), the missing tag is
  appended with a `"(default — added by orchestrator safety check)"`
  suffix. The pipeline is therefore guaranteed to have the minimal
  retrieval → analysis → verification sequence.
- If the LLM call itself fails, a hard-coded `DEFAULT_PLAN` is used
  so the graph can still run.
- `session_context` (injected from the MemoryAgent) lets the
  orchestrator resolve follow-up questions like "how does that compare
  to the policy you mentioned earlier?".

### 3.2 RetrieverAgent — `agents/retriever.py`

**Role:** Embed the user query, query ChromaDB, return top-k chunks
with source attribution and a normalized relevance score.

**API:** `retrieve(query, k=5) -> List[dict]`, plus
`get_corpus_size()` and `set_active_collection()` for per-session
isolation.

**Design notes:**

- Backend-aware relevance conversion. ChromaDB returns L2²
  (squared-euclidean) for Ollama embeddings and cosine distance for
  Gemini embeddings. For Ollama we convert to a pseudo-cosine
  (`cos = 1 - L2²/(2n²)`, where `n²=520` for `nomic-embed-text`)
  so the score lives in `[0, 1]` with real dynamic range. For
  Gemini we use `1 - distance` (already in `[0, 1]` for cosine).
- Backend-specific minimum-relevance threshold filters junk
  hits before they reach the Analyst. Ollama 0.50, Gemini 0.30.
- Per-session state: `_active_collection_name` and
  `_active_persist_dir` are module-level mutables. The vectorstore
  singleton is invalidated on `set_active_collection()`, so the
  Streamlit UI can hot-swap to a fresh per-session collection on
  Reset.
- The retriever does NOT perform any LLM work — it is a pure
  vector search wrapper.

### 3.3 AnalystAgent — `agents/analyst.py`

**Role:** Read the retrieved chunks and synthesize a coherent answer
that uses ONLY information present in the chunks.

**API:** `analyze(query, chunks) -> str`

**Design notes:**

- Strict grounding prompt. The system prompt explicitly forbids
  outside knowledge and requires citations.
- Output is forced into three sections: `[Reasoning]`, `[Answer]`,
  `[Sources Used]`. The `finalize_node` strips the `[Reasoning]`
  and `[Sources Used]` sections before showing the answer; the
  `[Sources Used]` list is regenerated by the guardrail's
  `apply_confidence_guardrail()` from the actual chunk metadata.
- Empty-chunk path returns a fixed "no information" answer — never
  hallucinates, even when the retriever returns nothing.

### 3.4 VerifierAgent — `agents/verifier.py`

**Role:** Decompose the draft answer into individual claims and
question aspects, classify each, and compute three deterministic
confidence scores. Output: RAG-Triad verification result.

**API:** `verify(draft_answer, chunks) -> dict`

**Design notes:**

- Single LLM call asks for `claims[]` (support tag), `question_aspects[]`
  (status tag), and `flags[]`. Python computes three scores:
  - `grounding_confidence = mean(SUPPORT_WEIGHTS[c] for c in claims)`
  - `answer_quality = mean(ASPECT_WEIGHTS[a] for a in aspects)`
  - `retrieval_confidence = top-3 avg of chunk relevance scores`
    (1 chunk → capped at 0.55)
- Final `confidence = min(grounding, answer_quality)` with safety
  caps: contradicted → ≤0.25, unsupported → ≤0.65, weak retrieval
  (<0.5) → ≤0.45.
- Early-return paths (no chunks, empty draft, LLM error, JSON parse
  error) return safe defaults with the right system-level flag, so
  the workflow always has a `verification_result` to act on.
- Public helper functions (`grounding_confidence_from_claims`,
  `answer_quality_from_aspects`, `retrieval_confidence_from_chunks`)
  are exposed for direct unit testing without an LLM call.

### 3.5 MemoryAgent — `agents/memory.py`

**Role:** Maintain session-level conversation context so multi-turn
dialogues can reference prior Q&A.

**API:** `MemoryAgent(session_id).add/get_context/reset/history/__len__`

**Design notes:**

- Pure-Python class — no LLM call, no persistence, no summarization.
- `get_context(last_n=3, max_answer_chars=400)` returns a string
  formatted for prompt injection (Q/A/S blocks). Truncation guards
  against prompt-injection via a long prior answer.
- One instance per session. The Streamlit UI creates a fresh
  MemoryAgent when the session is reset.
- Wired into the LangGraph flow as the `memory_node` (final step
  before END) so the just-completed turn is recorded for the next
  query in the same session.

---

## 4. Why LangGraph (and not LangChain AgentExecutor)

LangGraph was chosen over LangChain's higher-level `AgentExecutor`
for three reasons that are visible in the implementation:

1. **Explicit state.** Every node reads and writes a typed
   `AgentState`. A reviewer can see exactly what data crosses each
   edge. With `AgentExecutor`, intermediate reasoning is hidden
   inside the executor's loop.
2. **Traceability.** The `decision_trace` field accumulates a
   one-line summary per node. The UI's Agent Trace tab renders
   this directly. No need to parse free-form agent logs.
3. **Conditional routing.** The `low_confidence → finalize` edge
   is a first-class routing decision, not a post-hoc patch in
   the final answer string. New failure modes (e.g. "conflicting
   agent outputs") can be added as new conditional edges without
   changing the existing nodes.

The trade-off is verbosity: each node is a standalone Python
function with explicit state I/O, and adding a new agent means
adding a new node, an import, an edge, and a logger call. For a
5-agent system this is a small price for the audit clarity it
buys.

---

## 5. Per-session isolation

Each browser session in the Streamlit UI gets:

- a UUID4-derived `session_id` (12 hex chars, stored in
  `st.session_state.session_id`)
- a fresh `MemoryAgent` instance
- its own ChromaDB collection (`session_<id>`) and persist
  directory (`chroma_db_sessions/session_<id>/`)
- its own log file (`logs/eval_<timestamp>_<microsec>.json`)

When the user clicks **Reset Session**, the collection is deleted
from disk, the persist directory is removed, and a new
`session_id` + `MemoryAgent` are generated. Documents uploaded by
session A are invisible to session B.

This was a deliberate design decision. The default behaviour in
the agents.md spec was a single global `chroma_db/` shared across
all users. Per-session isolation is closer to the spec's "user
uploads their own documents" framing and is what the
`safe_ingest_files` / `delete_collection` API in
`vector_store/ingest.py` was built for.

---

## 6. Trade-offs and known limitations

| Decision | Pro | Con |
|---|---|---|
| **Ollama default embedder, Gemini fallback** | No rate limit, no API key, fully local | Slightly different relevance distribution; pseudo-cosine formula is hardcoded for `nomic-embed-text`'s `n²=520` |
| **RAG-Triad verifier with 3-axis confidence** | Calibrated, self-consistent, deterministic score | 1 extra LLM call per query (so 3 total: orchestrator + analyst + verifier) |
| **Per-session ChromaDB collection** | Strong isolation; reset = hard delete | Ingesting the same docs in two sessions re-embeds twice |
| **EvalLogger rewrites the whole file per entry** | Atomic on disk, easy to inspect, easier to parse | Slightly more I/O than append-only (negligible at <1 KB per query) |
| **In-memory MemoryAgent (not persisted)** | Simple, fast, no PII on disk | Session restart loses history. The Reset button is explicit about this. |
| **Corpus-agnostic guardrails** | Works on any enterprise corpus (HR, legal, finance, IT) | Does not catch "out-of-scope" queries per se; relies on the Verifier's RAG-Triad flags |
| **Safe-swap ingest pattern** | Existing data is never destroyed by a failed embed | Doubles disk usage during ingest (build in `<dir>_new/`, swap) |
| **Streamlit `text_input` + sidebar uploader** | Minimal JS, no API key needed client-side | Streamlit session_state quirks (e.g. can't clear widget keys after instantiation) |

---

## 7. Failure handling matrix

| Failure | Stage | Detection | Outcome |
|---|---|---|---|
| Empty corpus | `run_query` (guardrail) | `get_corpus_size() == 0` | Friendly rejection in `final_answer`; `GUARDRAIL` log entry; no graph run |
| Empty / too-short query | `run_query` (guardrail) | `len(query) < 5` | Friendly rejection; `GUARDRAIL` log entry |
| Prompt-injection pattern | `run_query` (guardrail) | 19 regex patterns | Friendly rejection; `GUARDRAIL` log entry |
| LLM quota exceeded (429) | any LLM call | `agents/_api_errors.is_quota_error()` | User-friendly message in `api_error`; "Service unavailable" banner in UI; `FAILURE` log entry |
| LLM model 404 | any LLM call | `agents/_api_errors.is_model_not_found()` | User-friendly message; `api_error` populated; `FAILURE` log entry |
| No chunks retrieved | `verify_node` | `len(chunks) == 0` | `confidence: 0.0`, `INSUFFICIENT_RETRIEVAL` flag, `LOW_CONFIDENCE` |
| Empty draft | `verify_node` | `not draft.strip()` | `confidence: 0.0`, `EMPTY_ANSWER` flag |
| Verifier LLM error | `verify_node` | exception in `_get_llm().invoke()` | `confidence: 0.5`, `LLM_ERROR` flag, `LOW_CONFIDENCE` |
| Verifier JSON parse error | `verify_node` | `_parse_json_response()` returns `None` | `confidence: 0.5`, `PARSE_ERROR` flag, `LOW_CONFIDENCE` |
| Low confidence answer | `finalize_node` | `confidence < 0.6` | `low_confidence_node` sets `needs_disclaimer=True`; `apply_confidence_guardrail` prepends `DISCLAIMER` |
| Unsupported claim in answer | `verify_node` (RAG-Triad) | claim tagged `unsupported` | `UNSUPPORTED_CLAIM` flag added; `confidence` capped at 0.65 |
| Contradicted claim in answer | `verify_node` (RAG-Triad) | claim tagged `contradicted` | `CONTRADICTED_CLAIM` flag added; `confidence` capped at 0.25 |
| No answer in corpus | `verify_node` (RAG-Triad) | `answer_quality < 0.4` and no useful claims | `NO_ANSWER_FROM_CORPUS` flag; low confidence; disclaimer shown |

---

## 8. File map (for the reviewer)

```
agents/
  orchestrator.py   — plan(query, session_context) -> List[str]
  retriever.py      — retrieve(query, k) -> List[dict] + per-session state
  analyst.py        — analyze(query, chunks) -> "[Reasoning]/[Answer]/[Sources Used]"
  verifier.py       — verify(draft, chunks) -> RAG-Triad result dict
  memory.py         — MemoryAgent class (per-session, in-memory)
  _api_errors.py    — friendly error translation for quota / 404 / generic
graph/
  workflow.py       — LangGraph StateGraph, run_query() entry point
guardrails/
  checks.py         — validate_input() + apply_confidence_guardrail()
vector_store/
  ingest.py         — safe_ingest_dir / safe_ingest_files / delete_collection
evaluation/
  logger.py         — EvalLogger class (per-query JSON file)
ui/
  app.py            — Streamlit UI, 4 tabs, per-session
tests/
  test_verifier_helpers.py  — 29 tests for RAG-Triad scoring math
  test_memory.py            — 24 tests for MemoryAgent
  test_guardrails.py        — 74 tests for validate_input + apply_confidence_guardrail
```

See `EVALUATION.md` for how guardrails and confidence thresholds
work, and `UNIT_TESTS.md` for what each test covers.
