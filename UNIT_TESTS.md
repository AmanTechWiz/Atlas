# Unit Tests

This document describes the **unit test suite** for the Enterprise Knowledge Ops Agent (EKO). It covers what each test exercises, how to run the suite, and what "passing" looks like.

For the system architecture, see `ARCHITECTURE.md`. For guardrails, confidence scoring, and failure handling, see `EVALUATION.md`.

## 1. Quick start

```bash
# All tests
uv run pytest tests/ -v

# Just one file
uv run pytest tests/test_guardrails.py -v

# Just one test
uv run pytest tests/test_verifier_helpers.py::TestVerifyIntegration::test_no_chunks_returns_insufficient_retrieval -v
```

**Expected outcome:** **155 tests pass in under 1 second**, with **no LLM calls** and **no network access**. The suite is fully deterministic.

Current counts:

| File | Tests | What it covers |
|---|---|---|
| `tests/test_retriever.py` | 16 | 7 retrieval strategies (13 base tests) + 3 corpus-agnostic tests |
| `tests/test_verifier_helpers.py` | 37 | Verifier RAG-Triad scoring math, cap logic, conflict detection, end-to-end `verify()` with mocked LLM |
| `tests/test_memory.py` | 24 | MemoryAgent public API (add, get_context, reset, history, __len__) |
| `tests/test_guardrails.py` | 66 | `validate_input()` rejection branches + `apply_confidence_guardrail()` output wrapping |
| `tests/test_query_rewriter.py` | 12 | Optional LLM-based query normalization (with mocked LLM) |
| **Total** | **155** | |

The tests are **independent** (no shared state, no fixtures that mutate each other) and runnable in any order.

## 2. Test design principles

1. **No LLM in tests.** The Verifier and query-rewriter tests mock the LLM call with `unittest.mock.patch.object(...)` and feed pre-canned JSON payloads. This keeps the suite fast, free, and deterministic.
2. **No network in tests.** The Retriever and ingest tests are not in this suite — they require a live ChromaDB and an embedder (Ollama or Gemini). They are covered by the manual E2E checks in the UI instead.
3. **Test the math, not the LLM.** Where the agent has a deterministic component (the scoring weights, the cap logic, the regex patterns, the retrieval strategy selection), the test exercises the math directly via the public helper functions (`grounding_confidence_from_claims`, `answer_quality_from_aspects`, `retrieval_confidence_from_chunks`, `validate_input`, `apply_confidence_guardrail`, `MemoryAgent.*`).
4. **Test the boundary conditions.** Empty inputs, None inputs, very long inputs, special characters, unusual tags — all covered. This is where bugs hide.
5. **Test the contract, not the implementation.** The Verifier tests check the *shape* of the returned dict (`confidence`, `grounded`, `flags`, `grounding_confidence`, `answer_quality`, `retrieval_confidence`, `claims`, `question_aspects`, `conflicts`), not the internal function calls. Refactors that keep the contract green don't need to update the tests.
6. **Corpus-agnostic tests.** The retriever tests prove that arbitrary filenames land in `doc_type="general"` and are still retrievable via filename-mention. The guardrail tests prove the same input rules work on any domain's queries.

## 3. `tests/test_retriever.py` (16 tests)

The retriever has the most stateful logic in the system (7 strategies + 14 doc types + source-mention fallback). These tests pin down the strategy selection and the corpus-agnostic guarantees.

### 3.1 Strategy selection and chunk shape (13 tests)

| Test | What it pins |
|---|---|
| `test_no_intent_falls_back_to_global` | `intent=None` → all chunks tagged `retrieval_strategy="fallback_global"` |
| `test_single_document_intent_uses_targeted` | `intent="single_document"` + `target_doc_types=["policy"]` → most chunks from policy docs |
| `test_single_document_supports_with_other_doc_types` | `single_document` → 10–30% of chunks come from non-target doc_types |
| `test_cross_document_intent_balances_sources` | `cross_document` + 3 sources → roughly even distribution across all 3 |
| `test_comparison_intent_pulls_from_each_side` | `comparison` mentioning "X vs Y" → chunks from both X and Y |
| `test_corpus_summary_returns_one_per_source` | `corpus_summary` → exactly 1 chunk per indexed source |
| `test_corpus_summary_fills_remaining_k_globally` | `corpus_summary` with `k=5` and 2 sources → 2 per_source + 3 fill |
| `test_query_mentioning_filename_uses_targeted_source` | "what does resume.pdf say" → all chunks from `resume.pdf` |
| `test_query_mentioning_unknown_source_falls_back` | "what does fake.pdf say" → `fallback_global` strategy, no crash |
| `test_chunks_have_doc_type_metadata` | Every returned chunk has `meta["doc_type"]` set |
| `test_chunks_have_relevance_score_in_range` | All `relevance_score` values in `[0.0, 1.0]` |
| `test_k_is_respected` | `k=3` → exactly 3 chunks returned |
| `test_empty_corpus_returns_empty_list` | `corpus_size=0` → `retrieve()` returns `[]` |

### 3.2 Corpus-agnostic guarantees (3 tests)

These three tests are the formal proof that EKO works on any document domain without code changes.

| Test | What it pins |
|---|---|
| `test_unknown_filename_classifies_as_general` | Uploading `mystery_file.xyz` → `doc_type="general"`, still retrievable |
| `test_unusual_filename_via_alias_is_retrievable` | Query "what does the report say" where the file is `quarterly_2026.pdf` → source-mention fallback pulls the file |
| `test_bare_word_query_finds_content` | Query "resume" with no file name → finds resume docs via doc-type preference |

## 4. `tests/test_verifier_helpers.py` (37 tests)

The Verifier's RAG-Triad scoring math is the most complex deterministic logic in the project. These tests pin it down.

### 4.1 `TestGroundingFromClaims` (8 tests)

`grounding_confidence_from_claims(claims)` returns the mean of the support-tag weights over all claims.

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

### 4.2 `TestAnswerQualityFromAspects` (5 tests)

`answer_quality_from_aspects(aspects)` returns the mean of the aspect-status weights.

| Test | What it pins |
|---|---|
| `test_all_answered` | Two `answered` → 1.0 |
| `test_all_partially_answered` | Two `partially_answered` → 0.60 |
| `test_all_not_answered` | One `not_answered` → 0.25 |
| `test_mix_yields_average` | One `answered` + one `not_answered` → 0.625 |
| `test_empty_aspects_yield_zero` | Empty list → 0.0 |

### 4.3 `TestRetrievalConfidenceFromChunks` (5 tests)

`retrieval_confidence_from_chunks(chunks)` returns the top-3 average of chunk relevance scores, with a single-chunk cap at 0.55.

| Test | What it pins |
|---|---|
| `test_no_chunks_is_zero` | Empty list → 0.0 |
| `test_single_chunk_caps_at_0_55` | One chunk with 0.90 → 0.55 (cap kicks in) |
| `test_single_chunk_low_relevance_preserved` | One chunk with 0.30 → 0.30 (below cap, preserved) |
| `test_top3_average` | 5 chunks → top-3 average, ignoring the 4th and 5th |
| `test_top3_average_with_only_2_chunks` | 2 chunks → mean of both (no top-3 truncation) |

### 4.4 `TestNormalizeConflict` (4 tests)

`_normalize_conflict()` validates the conflict records emitted by the LLM.

| Test | What it pins |
|---|---|
| `test_valid_conflict_passes_through` | Well-formed conflict dict → returned as-is |
| `test_missing_field_returns_none` | Missing `aspect` or `sources` → `None` |
| `test_non_dict_returns_none` | String or list passed in → `None` |
| `test_self_conflict_with_identical_claim_is_dropped` | Conflict where both sides cite the same claim → `None` (not a real conflict) |

### 4.5 `TestVerifyIntegration` (15 tests)

End-to-end `verify(draft, chunks)` calls with a mocked LLM. The mock returns pre-canned JSON payloads so the LLM's behaviour is fully controlled.

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
| `test_single_source_retrieval_flag_caps_confidence` | 1 chunk → `SINGLE_SOURCE_RETRIEVAL` flag fires, `retrieval_confidence` capped at 0.55 |
| `test_one_conflict_caps_confidence_at_0_55` | 1 conflict in `conflicts[]` → `confidence ≤ 0.55` |
| `test_two_conflicts_caps_confidence_at_0_45` | 2 conflicts in `conflicts[]` → `confidence ≤ 0.45` |
| `test_llm_error_falls_back_to_safe_default` | Mocked `invoke` raises `Exception` → `confidence=0.5`, `LLM_ERROR` flag, empty claims/aspects |
| `test_parse_error_falls_back_to_safe_default` | Mocked `invoke` returns `"not json at all"` → `confidence=0.5`, `PARSE_ERROR` flag, empty claims/aspects |
| `test_result_has_all_three_axes_and_backward_compat_fields` | Result dict has all 9 expected keys: `confidence`, `grounded`, `flags`, `grounding_confidence`, `answer_quality`, `retrieval_confidence`, `claims`, `question_aspects`, `conflicts` |
| `test_corpus_agnostic_no_preset_topics` | No claim/aspect mentions HR, legal, finance, or any specific domain keyword |

## 5. `tests/test_memory.py` (24 tests)

`MemoryAgent` is a pure-Python class. All tests exercise the public API.

### 5.1 Construction and `add` (10 tests)

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

### 5.2 `get_context` (8 tests)

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

### 5.3 `reset` and `history` (4 tests)

| Test | What it pins |
|---|---|
| `test_reset_clears_history` | `reset()` empties `history`, `len() == 0`, `get_context() == ""` |
| `test_reset_then_add_starts_fresh` | `reset()` + `add()` → only the new turn is in `history` |
| `test_history_returns_copy` | Mutating the returned list does NOT mutate the internal state |
| `test_session_id_is_recorded` | `session_id` kwarg is preserved on the instance |

### 5.4 Isolation and edge cases (2 tests)

| Test | What it pins |
|---|---|
| `test_separate_instances_have_separate_history` | Two `MemoryAgent` instances with different `session_id`s don't share state |
| `test_no_sources_renders_correctly` | Turn with no sources → `S1: (no sources)` in the context block |

## 6. `tests/test_guardrails.py` (66 tests)

The input/output guardrails are corpus-agnostic, so the tests use a mix of generic and domain-specific positive queries. The guardrail must accept any coherent business question regardless of domain.

### 6.1 Positive cases — `validate_input` (4 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_accepts_coherent_queries` (parametrized ×20) | 20 realistic business questions across HR, finance, IT, legal, operations, research, recruiting all pass |
| `test_validate_input_accepts_minimum_length` | 10-char query passes |
| `test_validate_input_is_case_insensitive_for_injection` | `IGNORE PREVIOUS INSTRUCTIONS` is still rejected (regex is `re.IGNORECASE`) |
| `test_validate_input_works_with_arbitrary_corpus` | Corpus size 5 still accepts valid queries |

### 6.2 Empty corpus (3 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_when_corpus_is_empty` | `corpus_size=0` → rejection with "knowledge base" or "upload" in the reason |
| `test_validate_input_accepts_when_corpus_has_chunks` | `corpus_size=1` → accept |
| `test_validate_input_defaults_to_empty_corpus` | No `corpus_size` arg → defaults to 0 → reject |

### 6.3 Length boundaries (5 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_short_or_empty` (parametrized ×4) | `""`, `"   "`, `"hi"`, `"a"` all rejected |
| `test_validate_input_rejects_none` | `query=None` rejected |
| `test_validate_input_rejects_non_string` | `query=123`, `query=[]`, `query={"q":"hi"}` all rejected |
| `test_validate_input_rejects_too_long` | `MAX_QUERY_LENGTH + 1` chars → reject with "long" in reason |
| `test_validate_input_accepts_at_max_length` | Exactly `MAX_QUERY_LENGTH` chars → accept |

### 6.4 Prompt-injection (3 tests)

| Test | What it pins |
|---|---|
| `test_validate_input_rejects_injection` (parametrized ×21) | All 21 prompt-injection patterns rejected, including case variations |
| `test_injection_rejection_message_mentions_rephrasing` | The rejection reason mentions "rejected" or "injection" |
| `test_new_injection_pattern_works` | Adding a new pattern to `INJECTION_PATTERNS` immediately rejects queries that match it |

### 6.5 Precedence (3 tests)

| Test | What it pins |
|---|---|
| `test_short_query_is_rejected_before_injection_check` | "hi" → "short" reason, not "injection" reason (length check runs first) |
| `test_injection_check_runs_before_corpus_check` | Injection query with `corpus_size=0` → "injection" reason (injection beats corpus) |
| `test_empty_corpus_blocks_even_valid_query` | Valid query with `corpus_size=0` → reject |

### 6.6 Corpus-agnostic positive cases (2 tests)

| Test | What it pins |
|---|---|
| `test_accepts_resume_queries` | "what does the resume say about X" passes (recruiting domain) |
| `test_accepts_research_queries` | "summarize the research paper" passes (research domain) |

### 6.7 `apply_confidence_guardrail` (11 tests)

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

## 7. `tests/test_query_rewriter.py` (12 tests)

`lib/query_rewriter.rewrite()` is an optional LLM-based query normalizer. All tests mock the LLM so the suite stays offline.

| Test | What it pins |
|---|---|
| `test_rewrite_returns_original_when_disabled` | `enabled=False` → returns the input unchanged |
| `test_rewrite_calls_llm_when_enabled` | `enabled=True` → exactly one LLM call per invocation |
| `test_rewrite_expands_abbreviation` | "PTO" → "paid time off" (mocked LLM response) |
| `test_rewrite_resolves_pronoun_against_context` | "how does that compare" with prior turn "policy X" → rewritten with "policy X" |
| `test_rewrite_falls_back_on_llm_error` | Mocked LLM raises → returns the original query, no crash |
| `test_rewrite_falls_back_on_empty_response` | Mocked LLM returns `""` → returns the original query |
| `test_rewrite_strips_extra_whitespace` | "  hello   world  " → "hello world" |
| `test_rewrite_handles_unicode` | "résumé" → preserved correctly |
| `test_rewrite_preserves_question_mark` | "what is X?" → "what is X?" |
| `test_rewrite_handles_very_long_query` | 2000-char query → rewritten without truncation |
| `test_rewrite_handles_empty_query` | `""` → returns `""` (or graceful no-op) |
| `test_rewrite_response_is_valid_for_embedding` | The rewritten output is a non-empty string suitable for `embed_query()` |

## 8. Running the suite

```bash
# All tests, verbose
uv run pytest tests/ -v

# Just the verifier math
uv run pytest tests/test_verifier_helpers.py -v

# Just the memory agent
uv run pytest tests/test_memory.py -v

# Just the guardrails
uv run pytest tests/test_guardrails.py -v

# Just the query rewriter
uv run pytest tests/test_query_rewriter.py -v

# Just the retriever strategies
uv run pytest tests/test_retriever.py -v

# Quiet summary
uv run pytest tests/

# With coverage (optional)
uv run pytest tests/ --cov=agents --cov=guardrails --cov=lib --cov-report=term-missing
```

**Expected output (last line):** `155 passed in 0.5s` (timing may vary by machine; should be under 1.5s on any modern laptop).

If any test fails:

1. Check that `uv sync` was run (dependencies are up to date).
2. Check that no test is making an LLM call — the suite is designed to be free and offline. If a test is calling Gemini or Ollama, something has changed in the test setup.
3. Check the failure message. Each test's assertion failure includes the actual vs expected value, so the bug is usually obvious from the message.

## 9. Tests NOT in the suite (and why)

The following are deliberately not unit-tested:

| Component | Why not unit-tested | What covers it instead |
|---|---|---|
| `RetrieverAgent.retrieve()` (end-to-end with real ChromaDB) | Requires a live ChromaDB + embedder (Ollama or Gemini) | The 13 strategy tests in `test_retriever.py` exercise the strategy selection with a mock vectorstore; the live embed path is covered by manual E2E in the UI |
| `AnalystAgent.analyze()` | Requires a live Gemini call | Manual E2E: run a real query in the UI and inspect the `logs/eval_*.json` |
| `OrchestratorAgent.plan()` (with LLM) | Requires a live Gemini call | The heuristic fast path is unit-tested; the LLM path is covered by manual E2E + log inspection |
| `graph/workflow.py:app` (the LangGraph) | End-to-end integration; expensive to mock | Manual E2E + 5-query regression suite (4/5 land in 0.5–0.95 confidence range) |
| `vector_store/ingest.py` | Requires a live embedder | Manual E2E: upload a PDF/DOCX/TXT in the UI and check the chunk count |
| `ui/app.py` | Streamlit UI; not amenable to `pytest` | Manual smoke test in browser |

The strategy and corpus-agnostic tests in `test_retriever.py` cover the deterministic parts of retrieval. The `test_query_rewriter.py` and `test_verifier_helpers.py` mocks cover the LLM-touching parts without actually calling the LLM. Together, every piece of deterministic logic in the system has at least one test.

## 10. Adding new tests

When adding a new test, follow these rules:

1. **No LLM calls.** If you need an LLM response, mock it with `unittest.mock.patch` and feed a hand-crafted JSON payload.
2. **No network access.** The suite must be offline-runnable.
3. **No shared state.** Use function-level fixtures, not module-level ones. Each test should pass on its own.
4. **Use the public API.** Don't poke at private helpers (`_parse_json_response`, `_normalize_claim`, etc.) — they can change without warning. Use the public functions.
5. **Test the contract, not the implementation.** The test should not care HOW the function works, only WHAT it returns.
6. **Name tests after the behaviour, not the implementation.** `test_unsupported_claim_caps_confidence_at_0_65` is good. `test_calculate_cap_loop_correct` is bad.
7. **For corpus-agnostic features, add a corpus-agnostic test.** If you add a new doc type, a new retrieval strategy, or a new guardrail, write at least one test that exercises it with a non-HR example to prove the system stays domain-neutral.
