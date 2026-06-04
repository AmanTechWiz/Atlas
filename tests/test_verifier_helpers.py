"""Deterministic tests for the VerifierAgent RAG-Triad scoring helpers.

These tests do NOT call any LLM. They exercise the pure-Python scoring
math (grounding / answer quality / retrieval) and the post-processing
cap logic by mocking the LLM call.

Acceptance criteria (agents.md Story 11):
- Tests are independent (no shared state)
- All tests pass with `pytest tests/`
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from agents import verifier


def _claim(claim: str, support: str, source: str = None, reason: str = "") -> Dict[str, Any]:
    return {"claim": claim, "support": support, "source": source, "reason": reason}


def _aspect(aspect: str, status: str, reason: str = "") -> Dict[str, Any]:
    return {"aspect": aspect, "status": status, "reason": reason}


def _chunk(source: str, relevance: float) -> Dict[str, Any]:
    return {
        "text": f"chunk from {source}",
        "source": source,
        "page": 0,
        "relevance_score": relevance,
    }


class TestGroundingFromClaims(unittest.TestCase):
    def test_all_direct_claims_yield_perfect_score(self):
        claims = [
            _claim("X is 16 weeks", "direct"),
            _claim("Y is 8 weeks", "direct"),
        ]
        self.assertAlmostEqual(verifier.grounding_confidence_from_claims(claims), 1.0, places=4)

    def test_reasonable_inference_scores_075(self):
        claims = [_claim("inferred X", "reasonable_inference")]
        self.assertAlmostEqual(verifier.grounding_confidence_from_claims(claims), 0.75, places=4)

    def test_absence_supported_scores_070(self):
        claims = [_claim("not specified", "absence_supported")]
        self.assertAlmostEqual(verifier.grounding_confidence_from_claims(claims), 0.70, places=4)

    def test_unsupported_scores_020(self):
        claims = [_claim("made up", "unsupported")]
        self.assertAlmostEqual(verifier.grounding_confidence_from_claims(claims), 0.20, places=4)

    def test_contradicted_scores_zero(self):
        claims = [_claim("wrong", "contradicted")]
        self.assertEqual(verifier.grounding_confidence_from_claims(claims), 0.0)

    def test_mixed_claims_take_average(self):
        claims = [
            _claim("a", "direct"),               # 1.0
            _claim("b", "reasonable_inference"),  # 0.75
        ]
        self.assertAlmostEqual(verifier.grounding_confidence_from_claims(claims), 0.875, places=4)

    def test_empty_claims_yield_zero(self):
        self.assertEqual(verifier.grounding_confidence_from_claims([]), 0.0)

    def test_unknown_support_tag_falls_back_to_unsupported(self):
        claims = [_claim("x", "garbage_tag")]
        self.assertAlmostEqual(verifier.grounding_confidence_from_claims(claims), 0.20, places=4)


class TestAnswerQualityFromAspects(unittest.TestCase):
    def test_all_answered(self):
        aspects = [_aspect("a", "answered"), _aspect("b", "answered")]
        self.assertEqual(verifier.answer_quality_from_aspects(aspects), 1.0)

    def test_all_partially_answered(self):
        aspects = [_aspect("a", "partially_answered"), _aspect("b", "partially_answered")]
        self.assertAlmostEqual(verifier.answer_quality_from_aspects(aspects), 0.60, places=4)

    def test_all_not_answered(self):
        aspects = [_aspect("a", "not_answered")]
        self.assertAlmostEqual(verifier.answer_quality_from_aspects(aspects), 0.25, places=4)

    def test_mix_yields_average(self):
        aspects = [
            _aspect("a", "answered"),
            _aspect("b", "not_answered"),
        ]
        self.assertAlmostEqual(verifier.answer_quality_from_aspects(aspects), 0.625, places=4)

    def test_empty_aspects_yield_zero(self):
        self.assertEqual(verifier.answer_quality_from_aspects([]), 0.0)


class TestRetrievalConfidenceFromChunks(unittest.TestCase):
    def test_no_chunks_is_zero(self):
        self.assertEqual(verifier.retrieval_confidence_from_chunks([]), 0.0)

    def test_single_chunk_caps_at_0_55(self):
        chunks = [_chunk("a", 0.90)]
        rc = verifier.retrieval_confidence_from_chunks(chunks)
        self.assertEqual(rc, 0.55)

    def test_single_chunk_low_relevance_preserved(self):
        chunks = [_chunk("a", 0.30)]
        rc = verifier.retrieval_confidence_from_chunks(chunks)
        self.assertEqual(rc, 0.30)

    def test_top3_average(self):
        chunks = [
            _chunk("a", 0.80),
            _chunk("b", 0.70),
            _chunk("c", 0.60),
            _chunk("d", 0.30),
            _chunk("e", 0.20),
        ]
        rc = verifier.retrieval_confidence_from_chunks(chunks)
        self.assertAlmostEqual(rc, (0.80 + 0.70 + 0.60) / 3, places=4)

    def test_top3_average_with_only_2_chunks(self):
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70)]
        rc = verifier.retrieval_confidence_from_chunks(chunks)
        self.assertAlmostEqual(rc, (0.80 + 0.70) / 2, places=4)


def _mock_llm_response(claims: List[Dict], aspects: List[Dict], flags: List[str] = None):
    import json
    payload = {
        "claims": claims,
        "question_aspects": aspects,
        "flags": flags or [],
    }
    return type("M", (), {"text": json.dumps(payload)})()


def _llm_mock_with_payload(payload_text: str):
    return type("M", (), {"text": payload_text})()


class TestVerifyIntegration(unittest.TestCase):
    def _run_verify(self, draft: str, chunks: List[Dict], llm_text: str) -> Dict[str, Any]:
        with patch.object(verifier, "_get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_mock_with_payload(llm_text)
            return verifier.verify(draft, chunks)

    def test_no_chunks_returns_insufficient_retrieval(self):
        result = verifier.verify("any answer", [])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("INSUFFICIENT_RETRIEVAL", " ".join(result["flags"]))
        self.assertEqual(result["grounding_confidence"], 0.0)
        self.assertEqual(result["answer_quality"], 0.0)
        self.assertEqual(result["retrieval_confidence"], 0.0)

    def test_empty_draft_returns_empty_answer(self):
        chunks = [_chunk("a", 0.80)]
        result = verifier.verify("", chunks)
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("EMPTY_ANSWER", " ".join(result["flags"]))

    def test_all_direct_all_answered_high_confidence(self):
        import json
        claims = [
            _claim("a", "direct"),
            _claim("b", "direct"),
        ]
        aspects = [_aspect("q1", "answered")]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70), _chunk("c", 0.60)]
        result = self._run_verify("a and b", chunks, payload)
        self.assertGreaterEqual(result["confidence"], 0.9)
        self.assertTrue(result["grounded"])
        self.assertNotIn("LOW_CONFIDENCE", " ".join(result["flags"]))

    def test_absence_supported_partial_answer_medium_confidence(self):
        import json
        claims = [
            _claim("bereavement notice period is not specified", "absence_supported"),
            _claim("flexible work requires 14 days notice", "direct"),
        ]
        aspects = [
            _aspect("bereavement notice", "not_answered"),
            _aspect("flexible work notice", "answered"),
        ]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70), _chunk("c", 0.60)]
        result = self._run_verify("draft", chunks, payload)
        self.assertGreaterEqual(result["confidence"], 0.4)
        self.assertLessEqual(result["confidence"], 0.75)
        self.assertIn("PARTIAL_ANSWER", " ".join(result["flags"]))

    def test_unsupported_claim_caps_confidence_at_0_65(self):
        import json
        claims = [
            _claim("a", "direct"),
            _claim("made up", "unsupported"),
        ]
        aspects = [_aspect("q", "answered")]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70)]
        result = self._run_verify("draft", chunks, payload)
        self.assertLessEqual(result["confidence"], 0.65)
        self.assertIn("UNSUPPORTED_CLAIM", " ".join(result["flags"]))

    def test_contradicted_claim_caps_confidence_at_0_25(self):
        import json
        claims = [
            _claim("a", "direct"),
            _claim("wrong", "contradicted"),
        ]
        aspects = [_aspect("q", "answered")]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70)]
        result = self._run_verify("draft", chunks, payload)
        self.assertLessEqual(result["confidence"], 0.25)
        self.assertFalse(result["grounded"])
        self.assertIn("CONTRADICTED_CLAIM", " ".join(result["flags"]))

    def test_true_refusal_with_no_useful_claims_gets_no_answer_flag(self):
        import json
        claims = []
        aspects = [_aspect("q", "not_answered")]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.50), _chunk("b", 0.40)]
        result = self._run_verify("I cannot answer", chunks, payload)
        self.assertEqual(result["answer_quality"], 0.25)
        self.assertIn("NO_ANSWER_FROM_CORPUS", " ".join(result["flags"]))

    def test_partial_answer_with_some_useful_info_does_not_get_no_answer_flag(self):
        import json
        claims = [
            _claim("PTO is 20 days", "direct"),
            _claim("bereavement is 5 days", "direct"),
        ]
        aspects = [
            _aspect("PTO info", "answered"),
            _aspect("holiday leave category", "not_answered"),
        ]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70), _chunk("c", 0.60)]
        result = self._run_verify("useful partial answer", chunks, payload)
        self.assertNotIn("NO_ANSWER_FROM_CORPUS", " ".join(result["flags"]))
        self.assertGreaterEqual(result["answer_quality"], 0.5)

    def test_llm_error_falls_back_to_safe_default(self):
        with patch.object(verifier, "_get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.side_effect = Exception("network down")
            chunks = [_chunk("a", 0.80), _chunk("b", 0.70)]
            result = verifier.verify("draft", chunks)
        self.assertEqual(result["confidence"], 0.5)
        self.assertIn("LLM_ERROR", " ".join(result["flags"]))
        self.assertEqual(len(result["claims"]), 0)
        self.assertEqual(len(result["question_aspects"]), 0)

    def test_parse_error_falls_back_to_safe_default(self):
        with patch.object(verifier, "_get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_mock_with_payload("not json at all")
            chunks = [_chunk("a", 0.80), _chunk("b", 0.70)]
            result = verifier.verify("draft", chunks)
        self.assertEqual(result["confidence"], 0.5)
        self.assertIn("PARSE_ERROR", " ".join(result["flags"]))

    def test_result_has_all_three_axes_and_backward_compat_fields(self):
        import json
        claims = [_claim("a", "direct")]
        aspects = [_aspect("q", "answered")]
        payload = json.dumps({"claims": claims, "question_aspects": aspects, "flags": []})
        chunks = [_chunk("a", 0.80), _chunk("b", 0.70)]
        result = self._run_verify("draft", chunks, payload)
        for key in ("confidence", "grounded", "flags",
                    "grounding_confidence", "answer_quality", "retrieval_confidence",
                    "claims", "question_aspects"):
            self.assertIn(key, result, f"missing key: {key}")


if __name__ == "__main__":
    unittest.main()
