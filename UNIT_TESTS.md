# Unit Tests

This document describes the **unit test suite** for the
multi-agent pipeline. It covers what each test exercises, how to
run the suite, and what "passing" looks like.

For the system architecture, see `ARCHITECTURE.md`. For
guardrails, confidence scoring, and failure handling, see
`EVALUATION.md`.

---

## 1. Quick start

```bash
# All tests
uv run pytest tests/ -v

# Just one file
uv run pytest tests/test_guardrails.py -v

# Just one test
uv run pytest tests/test_verifier_helpers.py::TestVerifyIntegration::test_no_chunks_returns_insufficient_retrieval -v
```

**Expected outcome:** 127 tests pass in **<1 second**, with **no
LLM calls** and **no network access**. The suite is fully
deterministic.

Current counts:

| File | Tests | What it covers |
|---|---|---|
| `tests/test_verifier_helpers.py` | 29 | Verifier RAG-Triad scoring math, cap logic, end-to-end `verify()` with mocked LLM |
| `tests/test_memory.py` | 24 | MemoryAgent public API (add, get_context, reset, history, __len__) |
| `tests/test_guardrails.py` | 74 | `validate_input()` rejection branches + `apply_confidence_guardrail()` output wrapping |
| **Total** | **127** | |

The tests are **independent** (no shared state, no fixtures that
mutate each other) and runnable in any order.

---

## 2. Test design principles

1. **No LLM in tests.** The Verifier tests mock the LLM call with
   `unittest.mock.patch.object(verifier, "_get_llm")` and feed
   pre-canned JSON payloads. This keeps the suite fast, free, and
   deterministic.
2. **No network in tests.** The Retriever and ingest tests are not
   in this suite — they require a live ChromaDB and an embedder
   (Ollama or Gemini). They are covered by the manual E2E checks
   in `progress.md` instead.
3. **Test the math, not the LLM.** Where the agent has a
   deterministic component (the scoring weights, the cap logic,
   the regex patterns), the test exercises the math directly via
   the public helper functions (`grounding_confidence_from_claims`,
   `answer_quality_from_aspects`, `retrieval_confidence_from_chunks`,
   `validate_input`, `apply_confidence_guardrail`, `MemoryAgent.*`).
4. **Test the boundary conditions.** Empty inputs, None inputs,
   very long inputs, special characters, unusual tags — all
   covered. This is where bugs hide.
5. **Test the contract, not the implementation.** The Verifier
   tests check the *shape* of the returned dict (`confidence`,
   `grounded`, `flags`, `grounding_confidence`, `answer_quality`,
   `retrieval_confidence`, `claims`, `question_aspects`), not the
   internal function calls. Refactors that keep the contract green
   don't need to update the tests.

---

## 3. `tests/test_verifier_helpers.py` (29 tests)

The Verifier's RAG-Triad scoring math is the most complex
deterministic logic in the project. These tests pin it down.

### 3.1 `TestGroundingFromClaims` (8 tests)

`grounding_confidence_from_claims(claims)` returns the mean of the
support-tag weights over all claims.

| Test | What it pins |
|---|---|
| `test_all_direct_claims_yield_perfect_score` | Two `direct` claims → 1.0 |
| `test_reasonable_inference_scores_075` | One `reasonable_inference` claim → 0.75 |
| `test_absence_supported_scores_070` | One `absence_supported` claim → 0.70 |
| `test_unsupported_scores_020` | One `unsupported` claim → 0.20 |
| `test_contradicted_scores_zero` | One `contradicted` claim → 0.0 |
| `test_mixed_claims_take_average` | One `direct` + one `reasonable_inference` → 0.875 |
| `test_empty_claims_yield_zero` | Empty list → 0.0 |
| `test_unknown_support_tag_falls_back_to_unsupported` | Unknown tag → treated as `unsupported` (0.20) |

### 3.2 `TestAnswerQualityFromAspects` (5 tests)

`answer_quality_from_aspects(aspects)` returns the mean of the
aspect-status weights.

| Test | What it pins |
|---|---|
| `test_all_answered` | Two `answered` → 1.0 |
| `test_all_partially_answered` | Two `partially_answered` → 0.60 |
| `test_all_not_answered` | One `not_answered` → 0.25 |
| `test_mix_yields_average` | One `answered` + one `not_answered` → 0.625 |
| `test_empty_aspects_yield_zero` | Empty list → 0.0 |

### 3.3 `TestRetrievalConfidenceFromChunks` (5 tests)

`retrieval_confidence_from_chunks(chunks)` returns the top-3 average
of chunk relevance scores, with a single-chunk cap at 0.55.

| Test | What it pins |
|---|---|
| `test_no_chunks_is_zero` | Empty list → 0.0 |
| `test_single_chunk_caps_at_0_55` | One chunk with 0.90 → 0.55 (cap kicks in) |
| `test_single_chunk_low_relevance_preserved` | One chunk with 0.30 → 0.30 (below cap, preserved) |
| `test_top3_average` | 5 chunks → top-3 average, ignoring the 4th and 5th |
| `test_top3_average_with_only_2_chunks` | 2 chunks → mean of both (no top-3 truncation) |

### 3.4 `TestVerifyIntegration` (11 tests)

End-to-end `verify(draft, chunks)` calls with a mocked LLM. The
mock returns pre-canned JSON payloads so the LLM's behaviour is
fully controlled.

| Test | What it pins |
|---|---|
| `test_no_chunks_returns_insufficient_retrieval` | Early return: `confidence=0.0`, `INSUFFICIENT_RETRIEVAL` flag, all axes 0.0 |
| `test_empty_draft_returns_empty_answer` | Early return: `confidence=0.0`, `EMPTY_ANSWER` flag |
| `test_all_direct_all_answered_high_confidence` | Two `direct` claims + one `answered` aspect → `confidence ≥ 0.9`, `grounded=True`, no `LOW_CONFIDENCE` flag |
| `test_absence_supported_partial_answer_medium_confidence` | One `absence_supported` claim + one `direct` claim, two aspects (`not_answered` + `answered`) → `confidence ∈ [0.4, 0.75]`, `PARTIAL_ANSWER` flag fires |
| `test_unsupported_claim_caps_confidence_at_0_65` | One `direct` + one `unsupported` → `confidence ≤ 0.65`, `UNSUPPORTED_CLAIM` flag fires |
| `test_contradicted_claim_caps_confidence_at_0_25` | One `direct` + one `contradicted` → `confidence ≤ 0.25`, `grounded=False`, `CONTRADICTED_CLAIM` flag fires |
| `test_true_refusal_with_no_useful_claims_gets_no_answer_flag` | No claims + one `not_answered` aspect + 2 chunks → `answer_quality=0.25`, `NO_ANSWER_FROM_CORPUS` flag fires |
| `test_partial_answer_with_some_useful_info_does_not_get_no_answer_flag` | Two `direct` claims + one `answered` + one `not_answered` aspect → no `NO_ANSWER_FROM_CORPUS` flag, `answer_quality ≥ 0.5` |
| `test_llm_error_falls_back_to_safe_default` | Mocked `invoke` raises `Exception` → `confidence=0.5`, `LLM_ERROR` flag, empty claims/aspects |
| `test_parse_error_falls_back_to_safe_default` | Mocked `invoke` returns `"not json at all"` → `confidence=0.5`, `PARSE_ERROR` flag, empty claims/aspects |
| `test_result_has_all_three_axes_and_backward_compat_fields` | Result dict has all 8 expected keys: `confidence`, `grounded`, `flags`, `grounding_confidence`, `answer_quality`, `retrieval_confidence`, `claims`, `question_aspects` |

---

## 4. `tests/test_memory.py` (24 tests)

`MemoryAgent` is a pure-Python class. All tests exercise the
public API.

### 4.1 Construction and `add` (10 tests)

| Test | What it pins |
|---|---|
| `test_new_memory_is_empty` | Fresh instance: `len() == 0`, `history == []`, `get_context() == ""` |
| `test_add_appends_entry` | `add()` appends to `history`, with `query`, `answer`, `sources`, `timestamp` keys |
| `test_add_increments_len` | 5 `add()` calls → `len() == 5` |
| `test_add_strips_query_and_answer` | Whitespace trimmed from `query` and `answer` |
| `test_add_normalizes_sources_to_list` | Tuple sources → list |
| `test_add_handles_none_sources` | `None` sources → empty list |
| `test_add_handles_empty_answer` | Empty answer is stored as `""` (not `None`) |
| `test_add_handles_none_query_and_answer` | `None` query and answer → stored as `""` |
| `test_add_uses_provided_timestamp` | `timestamp` kwarg is used verbatim |
| `test_add_auto_timestamp_when_omitted` | `timestamp=None` → auto-generated ISO 8601 string |

### 4.2 `get_context` (8 tests)

| Test | What it pins |
|---|---|
| `test_get_context_empty_history` | `get_context()` on empty memory → `""` |
| `test_get_context_single_turn` | One turn → `Q1`, `A1`, `S1` blocks present, "Prior session context" header |
| `test_get_context_multiple_turns` | Two turns → `Q1`/`A1`/`S1` and `Q2`/`A2`/`S2` blocks |
| `test_get_context_truncates_long_answers` | 1000-char answer with `max_answer_chars=100` → truncated to 100 + `"..."` |
| `test_get_context_last_n_truncates` | 10 turns, `last_n=3` → only the last 3 are returned, in correct order |
| `test_get_context_last_n_one` | 2 turns, `last_n=1` → only turn 2 |
| `test_get_context_uses_correct_numbering_when_truncated` | 5 turns, `last_n=2` → numbered `Q4` and `Q5` (not `Q1` and `Q2`) |
| `test_long_history_does_not_crash` | 50 turns, `last_n=10` → renders without crashing |

### 4.3 `reset` and `history` (4 tests)

| Test | What it pins |
|---|---|
| `test_reset_clears_history` | `reset()` empties `history`, `len() == 0`, `get_context() == ""` |
| `test_reset_then_add_starts_fresh` | `reset()` + `add()` → only the new turn is in `history` |
| `test_history_returns_copy` | Mutating the returned list does NOT mutate the internal state |
| `test_session_id_is_recorded` | `session_id` kwarg is preserved on the instance |

### 4.4 Isolation and edge cases (2 tests)

| Test | What it pins |
|---|---|
| `test_separate_instances_have_separate_history` | Two `MemoryAgent` instances with different `session_id`s don't share state |
| `test_no_sources_renders_correctly` | Turn with no sources → `S1: (no sources)` in the context block |

---

## 5. `tests/test_guardrails.py` (74 tests)

The input/output guardrails are corpus-agnostic, so the tests use
a mix of generic and HR-specific positive queries (HR-specific
queries are not "required" by the guardrail — they happen to be
the most realistic — but the guardrail must accept any coherent
business question).

### 5.1 Positive cases — `validate_input` (3 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_accepts_coherent_queries` (parametrized ×20) | 20 realistic business questions across HR, finance, IT, legal, operations all pass |
| `test_validate_input_accepts_minimum_length` | 10-char query passes |
| `test_validate_input_is_case_insensitive_for_injection` | `IGNORE PREVIOUS INSTRUCTIONS` is still rejected (regex is `re.IGNORECASE`) |
| `test_validate_input_works_with_arbitrary_corpus` | Corpus size 5 still accepts valid queries |

### 5.2 Empty corpus (3 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_when_corpus_is_empty` | `corpus_size=0` → rejection with "knowledge base" or "upload" in the reason |
| `test_validate_input_accepts_when_corpus_has_chunks` | `corpus_size=1` → accept |
| `test_validate_input_defaults_to_empty_corpus` | No `corpus_size` arg → defaults to 0 → reject |

### 5.3 Length boundaries (4 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_short_or_empty` (parametrized ×4) | `""`, `"   "`, `"hi"`, `"a"` all rejected |
| `test_validate_input_rejects_none` | `query=None` rejected |
| `test_validate_input_rejects_non_string` | `query=123`, `query=[]`, `query={"q":"hi"}` all rejected |
| `test_validate_input_rejects_too_long` | `MAX_QUERY_LENGTH + 1` chars → reject with "long" in reason |
| `test_validate_input_accepts_at_max_length` | Exactly `MAX_QUERY_LENGTH` chars → accept |

### 5.4 Prompt-injection (2 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_injection` (parametrized ×17) | 17 different prompt-injection patterns all rejected, including case variations |
| `test_injection_rejection_message_mentions_rephrasing` | The rejection reason mentions "rejected" or "injection" |

### 5.5 Spam / repetition (2 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_spam` (parametrized ×7) | Special-char-heavy strings and repeated tokens all rejected |
| `test_repetition_rejection_message_mentions_repetition` | The rejection reason mentions "repetition" or "rephrase" |

### 5.6 Precedence (3 tests)

| Test | What it pins |
|---|---|
| `test_short_query_is_rejected_before_injection_check` | "hi" → "short" reason, not "injection" reason (length check runs first) |
| `test_injection_check_runs_before_corpus_check` | Injection query with `corpus_size=0` → "injection" reason (injection beats corpus) |
| `test_empty_corpus_blocks_even_valid_query` | Valid query with `corpus_size=0` → reject |

### 5.7 `apply_confidence_guardrail` (10 tests)

| Test | What it pins |
|---|---|
| `test_guardrail_high_confidence_adds_only_footer` | `confidence=0.95` → no disclaimer, `**Sources:**` footer with both source files |
| `test_guardrail_low_confidence_prepends_disclaimer` | `confidence=0.4` → `DISCLAIMER` prepended, sources footer present |
| `test_guardrail_threshold_default_is_0_6` | `confidence=0.59` → disclaimer; `confidence=0.60` → no disclaimer |
| `test_guardrail_custom_threshold` | `confidence=0.8` with `confidence_threshold=0.9` → disclaimer |
| `test_guardrail_no_chunks_uses_none_footer` | Empty chunks → `(none ...)` in footer |
| `test_guardrail_empty_answer_still_has_footer` | Answer is `""` → footer still appended |
| `test_guardrail_none_answer_still_has_footer` | Answer is `None` → footer still appended |
| `test_guardrail_missing_verification_defaults_to_high_confidence` | `verification_result={}` → no disclaimer (defaults to 1.0) |
| `test_guardrail_none_verification_defaults_to_high_confidence` | `verification_result=None` → no disclaimer |
| `test_guardrail_dedupes_sources` | Same source repeated → appears once in footer |
| `test_guardrail_sorts_sources_alphabetically` | Sources appear in alphabetical order in footer |

---

## 6. Running the suite

```bash
# All tests, verbose
uv run pytest tests/ -v

# Just the verifier math
uv run pytest tests/test_verifier_helpers.py -v

# Just the memory agent
uv run pytest tests/test_memory.py -v

# Just the guardrails
uv run pytest tests/test_guardrails.py -v

# Quiet summary
uv run pytest tests/

# With coverage (optional)
uv run pytest tests/ --cov=agents --cov=guardrails --cov-report=term-missing
```

**Expected output (last line):** `127 passed in 0.8s` (timing may
vary by machine; should be under 1.5s on any modern laptop).

If any test fails:

1. Check that `uv sync` was run (dependencies are up to date).
2. Check that no test is making an LLM call — the suite is
   designed to be free and offline. If a test is calling Gemini
   or Ollama, something has changed in the test setup.
3. Check the failure message. Each test's assertion failure
   includes the actual vs expected value, so the bug is usually
   obvious from the message.

---

## 7. Tests NOT in the suite (and why)

The following are deliberately not unit-tested:

| Component | Why not unit-tested | What covers it instead |
|---|---|---|
| `RetrieverAgent.retrieve()` | Requires a live ChromaDB + embedder (Ollama or Gemini) | Manual E2E: `uv run python -c "from agents.retriever import retrieve; print(retrieve('test query'))"` |
| `AnalystAgent.analyze()` | Requires a live Gemini call | Manual E2E: `uv run python -c "from graph.workflow import run_query; print(run_query('test query')['final_answer'])"` |
| `OrchestratorAgent.plan()` | Requires a live Gemini call | Manual E2E + EvalLogger log inspection |
| `graph/workflow.py:app` (the LangGraph) | End-to-end integration; expensive to mock | Manual E2E + 5-query regression suite in `progress.md` (4/5 land in 0.5-0.95 confidence range) |
| `vector_store/ingest.py` | Requires a live embedder | Manual E2E: `uv run python vector_store/ingest.py` |
| `ui/app.py` | Streamlit UI; not amenable to `pytest` | Manual smoke test in browser |

The `test_retriever.py` and `test_orchestrator.py` files listed in
the project's Story 11 spec are **still pending** — they will mock
the embedder / LLM and exercise the parsing logic, but they have
not been built yet. The 127 tests that do exist cover all of the
deterministic logic in the system.

---

## 8. Adding new tests

When adding a new test, follow these rules:

1. **No LLM calls.** If you need an LLM response, mock it with
   `unittest.mock.patch` and feed a hand-crafted JSON payload.
2. **No network access.** The suite must be offline-runnable.
3. **No shared state.** Use function-level fixtures, not
   module-level ones. Each test should pass on its own.
4. **Use the public API.** Don't poke at private helpers
   (`_parse_json_response`, `_normalize_claim`, etc.) — they can
   change without warning. Use the public functions.
5. **Test the contract, not the implementation.** The test should
   not care HOW the function works, only WHAT it returns.
6. **Name tests after the behaviour, not the implementation.**
   `test_unsupported_claim_caps_confidence_at_0_65` is good.
   `test_calculate_cap_loop_correct` is bad.
