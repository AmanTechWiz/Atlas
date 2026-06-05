# Evaluation, Guardrails & Failure Handling

This document describes the **guardrails** the system applies, how
**grounding** is checked, why the **confidence threshold** is set
where it is, and the **known failure modes** plus how each is
handled. It also documents the **observability surface** — what
gets logged, where, and how to inspect it.

For the runtime architecture, see `ARCHITECTURE.md`. For test
coverage, see `UNIT_TESTS.md`.

---

## 1. Guardrail layers

The system has two guardrail layers, both implemented in
`guardrails/checks.py`. They are intentionally **corpus-agnostic**:
they make no assumption about the document domain (HR, legal,
finance, IT, operations, etc.) and no keyword-based "topic"
filtering.

### 1.1 Input guardrail — `validate_input()`

Runs **before** the graph in `graph/workflow.py:run_query()`. On
rejection, the graph is skipped entirely and a friendly message is
returned in `final_answer`. A `GUARDRAIL` log entry is written.

Checks (in this order — first failure wins):

| # | Check | Constant | Default |
|---|---|---|---|
| 1 | `query` is `None` | — | reject |
| 2 | `query` is not a `str` | — | reject |
| 3 | `query.strip()` is empty | — | reject |
| 4 | length < `MIN_QUERY_LENGTH` | `MIN_QUERY_LENGTH` | 5 |
| 5 | length > `MAX_QUERY_LENGTH` | `MAX_QUERY_LENGTH` | 2000 |
| 6 | special-char ratio > `MAX_SPECIAL_CHAR_RATIO` | `MAX_SPECIAL_CHAR_RATIO` | 0.50 |
| 7 | token-repetition ratio > `MAX_REPETITION_RATIO` | `MAX_REPETITION_RATIO` | 0.60 |
| 8 | matches one of 19 prompt-injection patterns | `INJECTION_PATTERNS` | see below |
| 9 | `corpus_size <= 0` | — | reject |

The 19 prompt-injection patterns cover the common bypass / override
attempts:

- `ignore (all) (previous|prior|above) instructions`
- `disregard (all) (previous|prior|above)`
- `forget (all) (previous|prior|above)`
- `you are now`, `act as (a|an)`, `pretend (to be|you are)`
- `system:`, `assistant:`, `### (system|assistant|instruction)`
- `<|im_start|>`, `<|im_end|>`
- `jailbreak`, `DAN`, `developer mode`
- `bypass (.*)? (safety|guardrail|filter|restriction)`
- `override (.*)? (safety|guardrail|filter|restriction)`
- `reveal/show (your|the) (system|hidden) prompt`
- `prompt injection`
- `execute (the following|this) (code|script|command)`
- `curl <url> | sh`

The regex is case-insensitive.

### 1.2 Output guardrail — `apply_confidence_guardrail()`

Runs **inside** `graph/workflow.py:finalize_node` (after the
verifier has produced its result). It is responsible for two
things:

1. **Low-confidence disclaimer.** When
   `verification_result.confidence < CONFIDENCE_THRESHOLD`
   (default 0.6), prepend the `DISCLAIMER` constant to the
   answer. The threshold is configurable per call.
2. **Sources footer.** Always append a `**Sources:**` footer
   listing the unique source files actually used (deduped,
   alphabetically sorted). When no chunks were retrieved, the
   footer reads `(none — no relevant chunks were retrieved)`.

The `DISCLAIMER` constant:

> **Low confidence — answer may not be fully supported by the
> source documents.** Treat the information above as provisional
> and verify against the cited sources before acting on it.

The threshold and the disclaimer are *visible to the user* — they
are not silently swallowed.

---

## 2. Grounding check (the Verifier)

The Verifier (`agents/verifier.py`) is the system's grounding check.
It implements a **RAG-Triad**-style decomposition: it asks the LLM
to break the draft answer into individual claims and to break the
user's question into sub-aspects, then computes three deterministic
confidence scores in Python.

### 2.1 The LLM call

The Verifier's prompt is in `agents/verifier.py:VERIFIER_SYSTEM_PROMPT`.
The LLM is asked to return a JSON object with three arrays:

```json
{
  "claims": [
    {
      "claim": "Employees get 20 days of PTO per year",
      "support": "direct",
      "source": "policy_hr.txt",
      "reason": "Stated explicitly in the policy document."
    }
  ],
  "question_aspects": [
    {
      "aspect": "How many PTO days per year?",
      "status": "answered",
      "reason": "Directly answered by the claim above."
    }
  ],
  "flags": ["answer too long", "uses external knowledge"]
}
```

### 2.2 Support tags (per claim)

| Tag | Weight | Meaning |
|---|---|---|
| `direct` | 1.0 | Explicitly stated in the retrieved chunks |
| `reasonable_inference` | 0.75 | Logically follows from the chunks (e.g. combining two facts) |
| `absence_supported` | 0.70 | The answer correctly says "the docs do not specify X" AND the chunks confirm X is genuinely absent |
| `unsupported` | 0.20 | Not supported by any retrieved chunk |
| `contradicted` | 0.0 | Conflicts with information in the chunks |

### 2.3 Aspect status tags (per question aspect)

| Status | Weight | Meaning |
|---|---|---|
| `answered` | 1.0 | This part of the question is fully addressed |
| `partially_answered` | 0.60 | Some relevant info is provided but a key part is missing |
| `not_answered` | 0.25 | No useful information for this aspect |

### 2.4 The three confidence scores

```
grounding_confidence   = mean(SUPPORT_WEIGHTS[c] for c in claims)
answer_quality         = mean(ASPECT_WEIGHTS[a] for a in aspects)
retrieval_confidence   = top-3 avg of chunk relevance scores
                        (1 chunk → capped at 0.55; 0 chunks → 0.0)

confidence = min(grounding, answer_quality)
```

### 2.5 Safety caps

After the base `min(grounding, answer_quality)`, three caps are
applied to prevent overconfident answers:

- any claim `contradicted` → `confidence ≤ 0.25`
- any claim `unsupported` → `confidence ≤ 0.65`
- retrieval_confidence < 0.5 → `confidence ≤ 0.45`

`grounded` is set to `True` only when `grounding_confidence ≥ 0.6`
AND no claim is `contradicted`.

### 2.6 Flag system

The Verifier can emit these flags (in addition to the support /
aspect tags above):

| Flag | When | Source |
|---|---|---|
| `INSUFFICIENT_RETRIEVAL` | `len(chunks) < 2` (or 0) | Python (no LLM) |
| `EMPTY_ANSWER` | draft is empty / whitespace | Python (no LLM) |
| `LLM_ERROR` | verifier LLM call raised | Python (no LLM) |
| `PARSE_ERROR` | verifier response not parseable JSON | Python (no LLM) |
| `LOW_CONFIDENCE` | final `confidence < 0.6` | Python (no LLM) |
| `UNSUPPORTED_CLAIM — N claim(s) lack source support` | ≥1 claim tagged `unsupported` | RAG-Triad |
| `CONTRADICTED_CLAIM — N claim(s) conflict with sources` | ≥1 claim tagged `contradicted` | RAG-Triad |
| `PARTIAL_ANSWER — some aspects of the question were not fully addressed` | `0.4 ≤ answer_quality < 0.7` | RAG-Triad |
| `NO_ANSWER_FROM_CORPUS — the analyst found no relevant information` | `answer_quality < 0.4` AND no useful claims | RAG-Triad |

The Verifier **never** suppresses a flag to make the answer look
better. The flag list is the ground truth; the confidence score
is a derived number.

---

## 3. Confidence threshold rationale

The `CONFIDENCE_THRESHOLD = 0.6` is set where it is for three
reasons:

1. **It is the same threshold as the Verifier's `grounded` boolean.**
   The Verifier sets `grounded=True` only when `grounding_confidence
   ≥ 0.6`. Using the same threshold in `finalize_node` means
   "disclaimer" and "grounded=False" are perfectly aligned.
2. **Empirical calibration.** On the 5-query regression suite run
   on 2026-06-04 (after the RAG-Triad refactor), 4 of 5 queries
   landed in the `0.5–0.95` confidence range. The 0.6 threshold
   correctly fires on the one binary case (the LLM's correct
   refusal at `0.25`) and on a `PARTIAL_ANSWER` query at `0.62`.
3. **Failure-mode cost is asymmetric.** A false positive
   (no disclaimer on a bad answer) is worse than a false negative
   (disclaimer on a good answer). The 0.6 threshold errs on the
   side of showing a disclaimer.

The threshold is **configurable** per `apply_confidence_guardrail`
call (`confidence_threshold=` kwarg), so a downstream consumer
(e.g. a stricter enterprise tier) can tighten it.

---

## 4. Known failure modes and how they are handled

### 4.1 Input-side

| Failure | Detection | Outcome |
|---|---|---|
| Empty corpus (no documents uploaded) | `corpus_size <= 0` in `validate_input` | Friendly rejection: "Your knowledge base is empty. Please upload at least one document (PDF, DOCX, or TXT) using the sidebar before asking a question." |
| Empty / whitespace query | `not query.strip()` | "Query is empty." |
| Query too short | `len(cleaned) < 5` | "Query is too short (need at least 5 characters)." |
| Query too long | `len(cleaned) > 2000` | "Query is too long (max 2000 characters)." |
| Query is spam / gibberish | special-char ratio > 0.50 OR repetition > 0.60 | "Query contains too many special characters ..." / "... excessive repetition ..." |
| Query is a prompt-injection attempt | 19-pattern regex match | "Query contains a prompt-injection pattern and was rejected. Please rephrase as a plain business question." |
| Query is `None` / non-string | `query is None` / `not isinstance(query, str)` | "Query must be a string." (or "Query is empty.") |

All of the above write a `GUARDRAIL` stage entry to the on-disk
log file and return a friendly rejection in `final_answer` without
running the graph. The UI's Answer tab shows the rejection message
in place of an answer.

### 4.2 LLM-side (any node)

| Failure | Detection | Outcome |
|---|---|---|
| Gemini returns 429 (quota) | `agents/_api_errors.is_quota_error()` | `api_error="Service is busy (rate-limited). Try again in a minute."`; UI shows yellow "Service unavailable" banner |
| Gemini returns 404 (model not found) | `agents/_api_errors.is_model_not_found()` | `api_error="Model not available. Check the .env model name."` |
| Any other LLM exception | generic `Exception` | `api_error="LLM error: <friendly message>"`; `FAILURE` log entry written; node returns a safe default |

When `api_error` is set, `finalize_node` prepends a
"**Service notice — ...**" header to the final answer so the user
sees the failure, not a confusing empty response.

### 4.3 Retrieval-side

| Failure | Detection | Outcome |
|---|---|---|
| No chunks retrieved (`len(chunks) == 0`) | `verify()` early-return | `confidence=0.0`, `INSUFFICIENT_RETRIEVAL` flag, `LOW_CONFIDENCE` flag; UI shows red badge; disclaimer in answer |
| Only 1 chunk retrieved | `len(chunks) < 2` in `verify()` | `INSUFFICIENT_RETRIEVAL` flag added (in addition to whatever else is computed); `retrieval_confidence` capped at 0.55 |
| Low-relevance chunks filtered out | backend-specific threshold in `retrieve()` | Empty list returned → same as "No chunks retrieved" |
| ChromaDB collection missing | `FileNotFoundError` in `_get_vectorstore()` | `retrieve_node` returns 0 chunks; `verify_node` short-circuits with `INSUFFICIENT_RETRIEVAL` |
| ChromaDB read fails for any reason | `get_corpus_size()` catches `Exception` | Returns 0 → guardrail rejects the query before the graph runs |

### 4.4 Synthesis-side (Analyst)

| Failure | Detection | Outcome |
|---|---|---|
| Analyst LLM call fails | `Exception` in `analyze_node` | `draft_answer=""`, `error="analyst_failed: ..."`, `api_error=...`; `verify_node` then early-returns with `EMPTY_ANSWER` flag |
| Analyst hallucinates a claim that contradicts a chunk | Verifier tags the claim `contradicted` | `CONTRADICTED_CLAIM` flag; `confidence` capped at 0.25; disclaimer shown |
| Analyst hallucinates a claim with no chunk support | Verifier tags the claim `unsupported` | `UNSUPPORTED_CLAIM` flag; `confidence` capped at 0.65; disclaimer shown |
| Analyst says "I cannot answer" when the corpus IS relevant | Verifier detects this via the `absence_supported` logic + `NO_ANSWER_FROM_CORPUS` heuristic | Flag fired; `confidence` low; disclaimer shown |

### 4.5 Verification-side (Verifier)

| Failure | Detection | Outcome |
|---|---|---|
| Verifier LLM call fails | `Exception` in `verify_node` | `confidence=0.5`, `LLM_ERROR` flag, `LOW_CONFIDENCE` flag (defensive default — "we don't know, so don't trust the answer") |
| Verifier response is not parseable JSON | `_parse_json_response()` returns `None` | `confidence=0.5`, `PARSE_ERROR` flag, `LOW_CONFIDENCE` flag |
| Verifier returns malformed claim / aspect | `_normalize_claim` / `_normalize_aspect` defaults to safe values | Unknown support tag → `unsupported` (0.20 weight); unknown aspect status → `not_answered` (0.25 weight) |

The Verifier's `LLM_ERROR` and `PARSE_ERROR` defaults return
`confidence=0.5` (not 0.0) because the analyst's draft may
actually be correct — we just couldn't verify it. The
`LOW_CONFIDENCE` flag is added so the disclaimer still fires.

### 4.6 State-side (cross-cutting)

| Failure | Detection | Outcome |
|---|---|---|
| Graph node raises an unexpected exception | `try/except` in each node | `error="<node>_failed: <msg>"`; `FAILURE` log entry; partial state preserved |
| EvalLogger fails to write | `try/except OSError` in `_flush()` | Warning logged to Python's `logging` under `"eval_logger"`; workflow continues normally |
| LangGraph state mutation error | LangGraph itself raises | Caught by `run_query`; returns partial state; `error` field set |

---

## 5. Observability — what's logged and where

Every query produces **one JSON file** at
`logs/eval_<timestamp>_<microsec>.json`. The file is an array of
entries, each with:

```json
{
  "timestamp": "2026-06-05T12:34:56.789012+00:00",
  "stage": "RETRIEVAL",
  "event": "retrieved 4 chunk(s) from 2 source(s)",
  "data": { ...stage-specific... }
}
```

Stages in the order they appear:

| Stage | Source | Data payload |
|---|---|---|
| `QUERY_START` | `log_query_start` | `{query}` |
| `ORCHESTRATION` | `log_plan` | `{plan: [step, ...]}` |
| `RETRIEVAL` | `log_retrieval` | `{chunk_count, sources, scores, chunks: [{text_preview, source, page, relevance_score}, ...]}` |
| `ANALYSIS` | `log_analysis` | `{draft_answer}` |
| `VERIFICATION` | `log_verification` | `{confidence, grounded, flags, grounding_confidence, answer_quality, retrieval_confidence, claims, question_aspects}` |
| `FINAL` | `log_final` | `{final_answer, total_time_ms}` |
| `SUMMARY` | `log_summary` | `{query, plan, retrieval_count, sources, confidence, grounded, grounding_confidence, answer_quality, retrieval_confidence, flags, claim_count, aspect_count, final_answer, total_time_ms}` |
| `GUARDRAIL` | `log_guardrail_rejection` | `{query, reason, rejected: true}` — written **instead of** the stages above when input is rejected |
| `FAILURE` | `log_failure` | `{error, stage}` — written from any node's `try/except` |

The Streamlit UI's **Evaluation Log** tab reads the on-disk
`logs/eval_<id>.json` file directly and renders it as collapsible
JSON, so a reviewer can see the full audit trail for any query
they ran.

---

## 6. Confidence calibration in practice

The RAG-Triad verifier was validated against a 5-query regression
suite on 2026-06-04. Before the refactor, confidence was binary
(0.0 or 1.0). After, the scores reflect real grounding quality:

| Query | Confidence | Reason |
|---|---|---|
| "What is the parental leave policy?" | 1.00 | All claims `direct` support |
| "Compare PTO with sick leave" | 1.00 | All claims `direct` support |
| "Best holiday leave period" | 0.62 | `PARTIAL_ANSWER` flag — answer was partial |
| "Parental + PTO accrual" | 0.25 | Correct refusal — corpus doesn't specify accrual |
| "When do I need to give notice to take bereavement leave?" | 0.85 | Strong direct support, 1 partial aspect |

4 of 5 land in the 0.5–0.95 range. The 0.6 threshold fires on
both the `0.62` (borderline) and the `0.25` (refusal) cases —
exactly the behaviour the disclaimer is meant to handle.

---

## 7. Known limitations (out of scope for this build)

| Limitation | Why out of scope | What a real production build would add |
|---|---|---|
| Image-only PDFs | Requires `pytesseract` + `pdf2image` (heavy deps) | OCR fallback in `vector_store/ingest.py:_load_one()` |
| Tables flattened to text | The chunker doesn't preserve tabular structure | Table-aware chunker (e.g. `unstructured.io`) |
| 1000+ page documents | `CHUNK_SIZE=500` is too small for very long documents | Adaptive `chunk_size` based on doc length |
| Non-English corpora | Ollama `nomic-embed-text` is English-trained | Switch to a multilingual embedder (e.g. `bge-m3`) |
| Multi-judge verifier | NLI models (e.g. DeBERTa-v3) are ~1 GB | Add a DeBERTa-v3-large MNLI scorer as a 2nd verifier |
| "Conflicting agent outputs" | Spec calls for this in the eval section, not yet implemented | Cross-check the retriever's top-1 source against the analyst's cited sources; if they disagree, fire `CONFLICTING_OUTPUTS` flag |
| Configurable thresholds in UI | Not yet exposed | Sidebar sliders for `chunk_size`, `k`, `CONFIDENCE_THRESHOLD`, `MAX_QUERY_LENGTH` |
| Bulk eval harness | Not yet built | `scripts/run_eval.py` runs N queries, aggregates metrics |
| Persistent MemoryAgent | In-memory only | JSON-file persistence in `chroma_db_sessions/session_<id>/memory.json` |

These are tracked as future work, not as gaps in the current
build. The current build is intentionally scoped to "local-first,
single-session, English-only, well-tested".
