# Architecture

This document describes the design of the **Enterprise Knowledge Ops Agent (EKO)** — the multi-agent pipeline, the runtime state, the agent roles, and the trade-offs that shaped them. It is meant to be read by reviewers who want to understand *how* the system works end-to-end, not just *what* it produces.

For guardrails, confidence scoring, and failure handling, see `EVALUATION.md`. For test coverage, see `UNIT_TESTS.md`.

## 1. High-level flow

```
                        +-------------------+
                        |  user query +     |
                        |  session memory   |
                        +---------+---------+
                                  |
                                  v
                       +----------+----------+
                       |  INPUT GUARDRAIL    |
                       |  (length, injection,|
                       |   empty corpus)     |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |   ORCHESTRATOR      |
                       |  classify intent +  |
                       |  build plan         |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |    RETRIEVER        |
                       |  intent-aware       |
                       |  ChromaDB lookup    |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |     ANALYST         |
                       |  synthesize draft   |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |     VERIFIER        |
                       |  RAG-Triad score    |
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

The pipeline is implemented as a LangGraph `StateGraph` with one node per agent plus a `low_confidence` node for the conditional disclaimer path. Every node writes a `decision_trace` entry and an `EvalLogger` JSON entry, so the runtime behaviour is fully reconstructable from the audit log.

## 2. Intent classification (orchestrator's first job)

The Orchestrator classifies every query into one of four intents **before** planning. The intent drives both the plan and the retriever's strategy.

| Intent | When | Retriever strategy | k split |
|---|---|---|---|
| `single_document` | User asks about a specific document (e.g. "the resume", "that HR policy") | `targeted_doc_type` + `targeted_supporting` | 70–90% primary, 10–30% supporting |
| `cross_document` | User asks about a topic that spans multiple docs, no comparison | `cross_document_balanced` | even mix across sources |
| `comparison` | User compares two or more things ("X vs Y", "compare PTO with sick leave") | `cross_document_balanced` | even mix, with one chunk per side preferred |
| `corpus_summary` | User asks for a roll-up of everything ("summarize all", "what documents do you have") | `corpus_overview_per_source` + `corpus_overview_fill` | 1 chunk per source, then global fill |

A query that doesn't match any pattern falls through to `fallback_global` (plain top-k).

The intent + the candidate `target_doc_types` are written to `AgentState` so the retriever can read them and route accordingly. The Orchestrator also has a heuristic fast path that classifies obvious single-document queries without an LLM call (saves a Gemini round-trip on the common case).

## 3. Doc-type classification (at ingest)

When a document is uploaded, `lib/document_classifier.classify_document()` assigns it one of **14 doc types** based on filename and the first chunk's content. The label is stored as `meta["doc_type"]` on every chunk.

```python
KNOWN_DOC_TYPES = (
    "resume", "cv", "job_requirements",
    "policy", "compliance", "sop", "procedure",
    "contract", "agreement",
    "manual", "guide", "report", "research_paper",
    "general",   # catch-all
)
```

Unknown filenames or content that doesn't match any pattern land in `"general"` — they are still retrievable via filename/source-mention, just not preferentially targeted.

This is the lever that makes the system **corpus-agnostic**: the same retriever serves an HR team (policies, contracts, manuals), a research team (papers, reports, guides), and a recruiting pipeline (resumes, job specs) without any code changes. The user's domain is just another `doc_type` value.

## 4. Retrieval strategies (the 7 modes)

`agents/retriever.py` exposes a single public function `retrieve(query, k=5)` and selects the right strategy internally based on the orchestrator's intent and the available `target_doc_types`.

| Strategy | When | Behaviour |
|---|---|---|
| `targeted_source` | Query mentions a specific filename or alias | Pull top chunks from that source only |
| `targeted_doc_type` | `single_document` intent with a doc-type hint | 70–90% of `k` from chunks matching the target doc_type |
| `targeted_supporting` | Same as above | 10–30% of `k` from related doc_types (cross-check) |
| `cross_document_balanced` | `comparison` or `cross_document` intent | Even mix across the 2+ sources named in the query |
| `corpus_overview_per_source` | `corpus_summary` intent | 1 representative chunk per indexed source |
| `corpus_overview_fill` | Same | After per-source: fill remaining `k` with highest-relevance global chunks |
| `fallback_global` | Nothing else matched | Plain top-k global similarity search |

Each chunk returned includes a `retrieval_strategy` field so the eval log and the UI can show *why* each chunk was pulled.

The retriever also handles **source-mention fallback**: if the query mentions a file by alias ("the resume", "that policy") but no chunks match that source, it falls back to a global search so the user still gets an answer.

## 5. Runtime state schema

`graph/workflow.py:AgentState` is a `TypedDict` (with `total=False`) that flows through every node. The schema is intentionally small — only the fields a node actually reads or writes.

| Field | Type | Owner (writer) | Notes |
|---|---|---|---|
| `query` | `str` | `run_query` | The original user query. |
| `intent` | `str` | `orchestrate_node` | One of `single_document` / `cross_document` / `comparison` / `corpus_summary`. |
| `target_doc_types` | `List[str]` | `orchestrate_node` | Candidate doc types the retriever should prefer. |
| `plan` | `List[str]` | `orchestrate_node` | Numbered steps with `[RETRIEVE]` / `[ANALYZE]` / `[VERIFY]` / `[MEMORY]` tags. |
| `retrieved_chunks` | `List[dict]` | `retrieve_node` | `{text, source, page, relevance_score, doc_type, retrieval_strategy, _distance}`. |
| `draft_answer` | `str` | `analyze_node` | The analyst's `[Reasoning] / [Answer] / [Sources Used]` blob. |
| `verification_result` | `dict` | `verify_node` | `{confidence, grounded, flags, grounding_confidence, answer_quality, retrieval_confidence, claims, question_aspects, conflicts}`. |
| `final_answer` | `str` | `finalize_node` | The user-facing answer (with disclaimer + sources footer). |
| `decision_trace` | `List[str]` | every node | One-line summary of what the node did. |
| `session_history` | `List[dict]` | `memory_node` | The MemoryAgent's full history. |
| `error` | `Optional[str]` | any node on failure | Internal error code, e.g. `analyst_failed: ...`. |
| `api_error` | `Optional[str]` | any node on LLM failure | User-friendly error message. |
| `needs_disclaimer` | `bool` | `low_confidence_node` | Set when `verification_result.confidence < 0.6`. |
| `eval_logger` | `EvalLogger` | `run_query` | Carried by reference; the same instance writes to one log file for the whole query. |
| `query_start_mono` | `float` | `run_query` | `time.monotonic()` at the start of the run, used for `total_time_ms`. |
| `memory` | `MemoryAgent` | caller | Per-session memory instance. |

A `log_path` field is added to the result after `app.invoke()` returns, giving the caller the on-disk path of the per-query JSON log.

## 6. Agent roles

### 6.1 OrchestratorAgent — `agents/orchestrator.py`

**Role:** Classify the query's intent, decide which doc types to target, and produce a numbered plan of agent steps. Does NOT execute the steps — that is the workflow's job.

**API:** `plan(query, session_context="") -> List[str]` (plus internal `classify_intent()` and `classify_doc_types()`)

**Design notes:**

- Uses a single LLM call with temperature 0.0. The prompt forces the model to output a JSON object with `{intent, target_doc_types, plan: [...]}` so the downstream nodes can parse it deterministically.
- Heuristic fast path: obvious single-document queries ("what does the resume say about X", "summarize the policy") skip the LLM and are classified in pure Python.
- Safety fallback: if the LLM omits any of the required tags (`[RETRIEVE]`, `[ANALYZE]`, `[VERIFY]`), the missing tag is appended with a `"(default — added by orchestrator safety check)"` suffix. The pipeline is therefore guaranteed to have the minimal retrieval → analysis → verification sequence.
- If the LLM call itself fails, a hard-coded `DEFAULT_PLAN` is used so the graph can still run.
- `session_context` (injected from the MemoryAgent) lets the orchestrator resolve follow-up questions like "how does that compare to the policy you mentioned earlier?".

### 6.2 RetrieverAgent — `agents/retriever.py`

**Role:** Embed the user query, query ChromaDB, return top-k chunks with source attribution, doc-type metadata, and the strategy used to find them.

**API:** `retrieve(query, k=5, intent=None, target_doc_types=None) -> List[dict]`, plus `get_corpus_size()`, `get_indexed_sources()`, `get_indexed_doc_types()`, and `invalidate_cache()` for collection hot-swap.

**Design notes:**

- Backend-aware relevance conversion. ChromaDB returns L2² (squared-euclidean) for Ollama embeddings and cosine distance for Gemini embeddings. For Ollama we convert to a pseudo-cosine (`cos = 1 - L2² / (2n²)`, where `n² = 520` for `nomic-embed-text`) so the score lives in `[0, 1]` with real dynamic range. For Gemini we use `1 - distance` (already in `[0, 1]` for cosine).
- Backend-specific minimum-relevance threshold filters junk hits before they reach the Analyst. Ollama 0.50, Gemini 0.30.
- Intent-aware routing: reads `intent` and `target_doc_types` from `AgentState` and dispatches to the matching strategy (see §4).
- Source-mention fallback: when the query names a file the corpus doesn't have, the retriever falls back to a global search and tags those chunks with `retrieval_strategy="fallback_global"`.
- The retriever does NOT perform any LLM work — it is a pure vector search wrapper.

### 6.3 AnalystAgent — `agents/analyst.py`

**Role:** Read the retrieved chunks and synthesize a coherent answer that uses ONLY information present in the chunks.

**API:** `analyze(query, chunks) -> str`

**Design notes:**

- Strict grounding prompt. The system prompt explicitly forbids outside knowledge and requires citations.
- Output is forced into three sections: `[Reasoning]`, `[Answer]`, `[Sources Used]`. The `finalize_node` strips the `[Reasoning]` and `[Sources Used]` sections before showing the answer; the `[Sources Used]` list is regenerated by the guardrail's `apply_confidence_guardrail()` from the actual chunk metadata.
- Empty-chunk path returns a fixed "no information" answer — never hallucinates, even when the retriever returns nothing.

### 6.4 VerifierAgent — `agents/verifier.py`

**Role:** Decompose the draft answer into individual claims and question aspects, classify each, and compute three deterministic confidence scores. Output: RAG-Triad verification result with conflict detection.

**API:** `verify(draft_answer, chunks) -> dict`

**Design notes:**

- Single LLM call asks for `claims[]` (support tag), `question_aspects[]` (status tag), and `flags[]`. Python computes three scores:
  - `grounding_confidence = mean(SUPPORT_WEIGHTS[c] for c in claims)`
  - `answer_quality = mean(ASPECT_WEIGHTS[a] for a in aspects)`
  - `retrieval_confidence = top-3 avg of chunk relevance scores` (1 chunk → capped at 0.55)
- Final `confidence = min(grounding, answer_quality)` with safety caps: contradicted → ≤0.25, unsupported → ≤0.65, weak retrieval (<0.5) → ≤0.45.
- Conflict detection: when 2+ sources disagree on the same aspect, each disagreement is recorded as a `conflict` object and a cap is applied (1 conflict → `≤0.55`, 2+ conflicts → `≤0.45`).
- Early-return paths (no chunks, empty draft, LLM error, JSON parse error) return safe defaults with the right system-level flag, so the workflow always has a `verification_result` to act on.
- Public helper functions (`grounding_confidence_from_claims`, `answer_quality_from_aspects`, `retrieval_confidence_from_chunks`) are exposed for direct unit testing without an LLM call.

### 6.5 MemoryAgent — `agents/memory.py`

**Role:** Maintain session-level conversation context so multi-turn dialogues can reference prior Q&A.

**API:** `MemoryAgent(session_id=None).add() / get_context() / reset() / history / __len__`

**Design notes:**

- Pure-Python class — no LLM call, no persistence, no summarization.
- `get_context(last_n=3, max_answer_chars=400)` returns a string formatted for prompt injection (Q/A/S blocks). Truncation guards against prompt-injection via a long prior answer.
- Singleton via `get_default_memory()` — a shared default instance is reused across runs in the same process. The Streamlit UI calls `reset()` (not a new instance) on Reset so the singleton pattern works.
- Wired into the LangGraph flow as the `memory_node` (final step before END) so the just-completed turn is recorded for the next query in the same session.

## 7. Helper modules (`lib/`)

Three small modules contain the cross-cutting logic that is not a "real agent" but is shared across the system.

### 7.1 `lib/query_rewriter.py`

Optional LLM-based query normalization. Used by the orchestrator to expand abbreviations, resolve pronouns against session context, and produce a cleaner embedding target. Has a deterministic fallback that just returns the original query if the LLM call fails.

### 7.2 `lib/document_classifier.py`

Filename-and-content based doc-type detection. Two cheap heuristics (`classify_by_filename`, `classify_by_content`) that return a `DocType` from the 14-element `KNOWN_DOC_TYPES` tuple, plus `classify_document()` which composes them. Called at ingest time; result is stored as `meta["doc_type"]` on every chunk.

### 7.3 `lib/api_errors.py`

Translates Gemini exceptions into friendly user-facing messages. `is_quota_error()`, `is_model_not_found()`, and a generic `friendly_error()` keep the quota / 404 / unknown error paths consistent across all LLM call sites.

## 8. Why LangGraph (and not LangChain AgentExecutor)

LangGraph was chosen over LangChain's higher-level `AgentExecutor` for three reasons that are visible in the implementation:

1. **Explicit state.** Every node reads and writes a typed `AgentState`. A reviewer can see exactly what data crosses each edge. With `AgentExecutor`, intermediate reasoning is hidden inside the executor's loop.
2. **Traceability.** The `decision_trace` field accumulates a one-line summary per node. The UI's Details panel renders this directly. No need to parse free-form agent logs.
3. **Conditional routing.** The `low_confidence → finalize` edge is a first-class routing decision, not a post-hoc patch in the final answer string. New failure modes (e.g. "conflicting agent outputs") can be added as new conditional edges without changing the existing nodes.

The trade-off is verbosity: each node is a standalone Python function with explicit state I/O, and adding a new agent means adding a new node, an import, an edge, and a logger call. For a 5-agent system this is a small price for the audit clarity it buys.

## 9. UI design

The Streamlit UI (`ui/app.py`) is a dark-mode chat interface in the ChatGPT/Claude style. Three regions:

- **Top bar** — EKO brand, backend model, and a **Reset Knowledge Base** button at the top right. Single click wipes the corpus, conversation, and memory.
- **Sidebar** — collapsible file uploader, indexed-files list, corpus stats (chunk count, embedding model, LLM, backend).
- **Main area** — empty state shows a "How can I help you today?" hero; chat state shows scrollable message history with `st.chat_message`. Each assistant turn shows: answer, confidence badge, source chips, and a collapsible Details panel (intent, retrieved chunks, eval log path).

**Auto-reset on startup:** every fresh Streamlit session calls `reset_collection()` once (gated by `st.session_state._bootstrapped`). The user must re-upload documents after every browser refresh — this is intentional for the local-first, single-session deployment.

## 10. Single-corpus architecture

One application-level corpus at `chroma_db/eko_corpus`. There is no per-session collection; the same collection serves every query. This is the opposite of the per-session pattern common in multi-tenant SaaS and is the right call for a local-first desktop tool where one person uses one knowledge base at a time.

The **Reset Knowledge Base** button in the sidebar deletes the collection, clears the conversation history, and resets the MemoryAgent in one action. The **auto-reset on startup** does the same thing automatically when Streamlit boots.

## 11. Trade-offs and known limitations

| Decision | Pro | Con |
|---|---|---|
| **Ollama default embedder, Gemini fallback** | No rate limit, no API key, fully local | Slightly different relevance distribution; pseudo-cosine formula is hardcoded for `nomic-embed-text`'s `n² = 520` |
| **Single corpus, auto-reset on startup** | Predictable state, no leftover chunks from previous session | User must re-upload after every browser refresh |
| **Intent classification with LLM** | Routes retrieval to the right strategy automatically | 1 extra LLM call on the first query (heuristic fast path skips it for obvious single-doc queries) |
| **Doc-type classification at ingest** | Corpus-agnostic — works for HR, research, recruiting, etc. with no code changes | Misclassified docs (e.g. unusual filename) land in `general` and are not preferentially targeted |
| **RAG-Triad verifier with 3-axis confidence** | Calibrated, self-consistent, deterministic score | 1 extra LLM call per query (so 3 total: orchestrator + analyst + verifier) |
| **Corpus-agnostic guardrails** | Works on any enterprise corpus (HR, legal, finance, IT) | Does not catch "out-of-scope" queries per se; relies on the Verifier's RAG-Triad flags |
| **In-memory MemoryAgent (not persisted)** | Simple, fast, no PII on disk | Session restart loses history. The Reset button is explicit about this. |
| **EvalLogger rewrites the whole file per entry** | Atomic on disk, easy to inspect, easier to parse | Slightly more I/O than append-only (negligible at <1 KB per query) |

## 12. File map (for the reviewer)

```
agents/
  orchestrator.py   — classify_intent() + plan(query, session_context) -> List[str]
  retriever.py      — retrieve(query, k, intent, target_doc_types) -> List[dict]
                      + 7 retrieval strategies
  analyst.py        — analyze(query, chunks) -> "[Reasoning]/[Answer]/[Sources Used]"
  verifier.py       — verify(draft, chunks) -> RAG-Triad result dict + conflicts
  memory.py         — MemoryAgent class (per-session, in-memory) + get_default_memory()
lib/
  query_rewriter.py      — optional LLM-based query normalization
  document_classifier.py — 14 doc types, filename+content heuristics
  api_errors.py          — friendly error translation for quota / 404 / generic
graph/
  workflow.py       — LangGraph StateGraph, run_query() entry point
guardrails/
  checks.py         — validate_input() + apply_confidence_guardrail()
vector_store/
  ingest.py         — ingest_files / ingest_dir / reset_collection (PDF, DOCX, TXT, MD)
evaluation/
  logger.py         — EvalLogger class (per-query JSON file)
ui/
  app.py            — Streamlit dark-mode UI, auto-reset on startup
tests/
  test_retriever.py          — 16 tests (13 base + 3 corpus-agnostic)
  test_verifier_helpers.py   — 37 tests for RAG-Triad scoring math + conflict caps
  test_memory.py             — 24 tests for MemoryAgent
  test_guardrails.py         — 66 tests for validate_input + apply_confidence_guardrail
  test_query_rewriter.py     — 12 tests for query normalization
```

See `EVALUATION.md` for how guardrails and confidence thresholds work, and `UNIT_TESTS.md` for what each test covers.
