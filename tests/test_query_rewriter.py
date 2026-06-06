"""Deterministic tests for the query rewriter.

We mock the LLM call (via `agents.query_rewriter._get_llm`) so the
tests run with no API key, no network, and are fully deterministic.

Coverage:
  - empty query -> empty string (no LLM call)
  - empty memory -> original query (no LLM call)
  - long self-contained query -> original query (no LLM call)
  - short query with pronoun + memory -> LLM called, rewritten
  - LLM returns empty -> original query (graceful fallback)
  - LLM raises -> original query (graceful fallback)
  - LLM returns list of content blocks (langchain 1.x format) -> unwrapped
  - LLM returns string (older format) -> unwrapped
  - LLM response wrapped in quotes -> stripped
  - LLM response > 500 chars -> truncated
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

from lib import query_rewriter


def test_empty_query_returns_empty():
    assert query_rewriter.rewrite_query("", "some memory") == ""
    assert query_rewriter.rewrite_query("   ", "some memory") == ""


def test_no_memory_returns_original():
    assert query_rewriter.rewrite_query("any more info about him?", "") == "any more info about him?"
    assert query_rewriter.rewrite_query("any more info about him?", "   ") == "any more info about him?"


def test_long_self_contained_query_skips_rewrite():
    q = "what is the transformer attention mechanism as described in the paper"
    assert query_rewriter.rewrite_query(q, "Some memory") == q


def test_looks_like_followup_heuristic():
    # short, has a pronoun -> True
    assert query_rewriter._looks_like_followup("any more info about him?") is True
    assert query_rewriter._looks_like_followup("what about her?") is True
    assert query_rewriter._looks_like_followup("that policy") is True
    # no pronoun -> False
    assert query_rewriter._looks_like_followup("summarize resume") is False
    # long enough to not be a followup -> False
    long_q = " ".join(["word"] * 20)
    assert query_rewriter._looks_like_followup(long_q) is False


def _make_llm_mock(content: Any) -> MagicMock:
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def test_rewrite_calls_llm_with_memory_context():
    with patch.object(query_rewriter, "_get_llm", return_value=_make_llm_mock("Amandeep's resume details")):
        result = query_rewriter.rewrite_query("any more info about him?", "Q1: tell me about the guy's resume\nA1: it's about Amandeep")
    assert result == "Amandeep's resume details"


def test_rewrite_preserves_original_when_llm_returns_empty():
    with patch.object(query_rewriter, "_get_llm", return_value=_make_llm_mock("")):
        result = query_rewriter.rewrite_query("any more info about him?", "some memory")
    assert result == "any more info about him?"


def test_rewrite_falls_back_on_llm_exception():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("rate limit")
    with patch.object(query_rewriter, "_get_llm", return_value=mock_llm):
        result = query_rewriter.rewrite_query("any more info about him?", "some memory")
    assert result == "any more info about him?"


def test_rewrite_handles_list_of_content_blocks_langchain_1x():
    content_blocks: List[Any] = [
        {"type": "text", "text": "Amandeep's "},
        {"type": "text", "text": "resume details"},
    ]
    with patch.object(query_rewriter, "_get_llm", return_value=_make_llm_mock(content_blocks)):
        result = query_rewriter.rewrite_query("any more info about him?", "memory")
    assert "Amandeep" in result
    assert "resume" in result


def test_rewrite_handles_list_of_string_blocks():
    with patch.object(query_rewriter, "_get_llm", return_value=_make_llm_mock(["More about ", "Amandeep"])):
        result = query_rewriter.rewrite_query("any more info about him?", "memory")
    assert "Amandeep" in result


def test_rewrite_strips_quotes_from_response():
    with patch.object(query_rewriter, "_get_llm", return_value=_make_llm_mock('"Amandeep resume"')):
        result = query_rewriter.rewrite_query("any more info about him?", "memory")
    assert result == "Amandeep resume"


def test_rewrite_truncates_very_long_response():
    long_response = "x" * 1000
    with patch.object(query_rewriter, "_get_llm", return_value=_make_llm_mock(long_response)):
        result = query_rewriter.rewrite_query("any more info about him?", "memory")
    assert len(result) <= 500


def test_rewrite_handles_missing_gemini_key():
    with patch.object(query_rewriter, "_get_llm", side_effect=EnvironmentError("no key")):
        result = query_rewriter.rewrite_query("any more info about him?", "memory")
    assert result == "any more info about him?"
