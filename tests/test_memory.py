"""Tests for agents/memory.py (Story 5, Official US 2 Memory Agent).

Covers the public API of MemoryAgent: add(), get_context(), reset(),
history, and __len__. Pure-Python, no LLM calls. Runs in <1s.
"""

from __future__ import annotations

import pytest

from agents.memory import MemoryAgent


def _add(memory, q, a, sources=None, ts=None):
    memory.add(q, a, sources=sources, timestamp=ts)


def test_new_memory_is_empty():
    m = MemoryAgent()
    assert len(m) == 0
    assert m.history == []
    assert m.get_context() == ""


def test_add_appends_entry():
    m = MemoryAgent()
    _add(m, "Q1", "A1", sources=["a.txt"])
    assert len(m) == 1
    assert m.history[0]["query"] == "Q1"
    assert m.history[0]["answer"] == "A1"
    assert m.history[0]["sources"] == ["a.txt"]
    assert "timestamp" in m.history[0]


def test_add_increments_len():
    m = MemoryAgent()
    for i in range(5):
        _add(m, f"Q{i}", f"A{i}")
    assert len(m) == 5


def test_add_strips_query_and_answer():
    m = MemoryAgent()
    _add(m, "  Q with spaces  ", "  A with spaces  ")
    assert m.history[0]["query"] == "Q with spaces"
    assert m.history[0]["answer"] == "A with spaces"


def test_add_normalizes_sources_to_list():
    m = MemoryAgent()
    _add(m, "Q", "A", sources=("a.txt", "b.txt"))
    assert m.history[0]["sources"] == ["a.txt", "b.txt"]


def test_add_handles_none_sources():
    m = MemoryAgent()
    _add(m, "Q", "A", sources=None)
    assert m.history[0]["sources"] == []


def test_add_handles_empty_answer():
    m = MemoryAgent()
    _add(m, "Q", "")
    assert m.history[0]["answer"] == ""


def test_add_handles_none_query_and_answer():
    m = MemoryAgent()
    _add(m, None, None)
    assert m.history[0]["query"] == ""
    assert m.history[0]["answer"] == ""


def test_add_uses_provided_timestamp():
    m = MemoryAgent()
    _add(m, "Q", "A", ts="2026-01-01T00:00:00+00:00")
    assert m.history[0]["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_add_auto_timestamp_when_omitted():
    m = MemoryAgent()
    _add(m, "Q", "A")
    assert "T" in m.history[0]["timestamp"]


def test_get_context_empty_history():
    m = MemoryAgent()
    assert m.get_context() == ""


def test_get_context_single_turn():
    m = MemoryAgent()
    _add(m, "What is the policy?", "Policy is X.", sources=["a.txt"])
    ctx = m.get_context()
    assert "Prior session context" in ctx
    assert "Q1: What is the policy?" in ctx
    assert "A1: Policy is X." in ctx
    assert "S1: a.txt" in ctx


def test_get_context_multiple_turns():
    m = MemoryAgent()
    _add(m, "Q1", "A1", sources=["a.txt"])
    _add(m, "Q2", "A2", sources=["b.txt", "c.txt"])
    ctx = m.get_context()
    assert "Q1: Q1" in ctx
    assert "Q2: Q2" in ctx
    assert "S1: a.txt" in ctx
    assert "S2: b.txt, c.txt" in ctx


def test_get_context_truncates_long_answers():
    m = MemoryAgent()
    long_answer = "x" * 1000
    _add(m, "Q", long_answer)
    ctx = m.get_context(max_answer_chars=100)
    assert "x" * 100 in ctx
    assert "..." in ctx
    assert "x" * 200 not in ctx


def test_get_context_last_n_truncates():
    m = MemoryAgent()
    for i in range(10):
        _add(m, f"Q{i}", f"A{i}")
    ctx = m.get_context(last_n=3)
    assert "Q1:" not in ctx
    assert "Q2:" not in ctx
    assert "Q3:" not in ctx
    assert "Q8: Q7" in ctx
    assert "Q9: Q8" in ctx
    assert "Q10: Q9" in ctx
    lines = [ln for ln in ctx.splitlines() if ln.startswith("  Q") and ":" in ln]
    assert len(lines) == 3


def test_get_context_last_n_one():
    m = MemoryAgent()
    _add(m, "Q1", "A1")
    _add(m, "Q2", "A2")
    ctx = m.get_context(last_n=1)
    assert "Q2: Q2" in ctx
    assert "Q1: Q1" not in ctx


def test_get_context_uses_correct_numbering_when_truncated():
    m = MemoryAgent()
    for i in range(5):
        _add(m, f"Q{i}", f"A{i}")
    ctx = m.get_context(last_n=2)
    assert "Q4: Q3" in ctx
    assert "Q5: Q4" in ctx


def test_reset_clears_history():
    m = MemoryAgent()
    _add(m, "Q", "A")
    _add(m, "Q2", "A2")
    assert len(m) == 2
    m.reset()
    assert len(m) == 0
    assert m.history == []
    assert m.get_context() == ""


def test_reset_then_add_starts_fresh():
    m = MemoryAgent()
    _add(m, "old Q", "old A")
    m.reset()
    _add(m, "new Q", "new A")
    assert len(m) == 1
    assert m.history[0]["query"] == "new Q"


def test_history_returns_copy():
    m = MemoryAgent()
    _add(m, "Q", "A")
    h = m.history
    h.append({"query": "forged", "answer": "forged", "sources": [], "timestamp": "x"})
    assert len(m) == 1


def test_session_id_is_recorded():
    m = MemoryAgent(session_id="user-abc")
    assert m.session_id == "user-abc"


def test_separate_instances_have_separate_history():
    m1 = MemoryAgent(session_id="a")
    m2 = MemoryAgent(session_id="b")
    _add(m1, "Q1", "A1")
    _add(m2, "Q2", "A2")
    assert len(m1) == 1
    assert len(m2) == 1
    assert m1.history[0]["query"] == "Q1"
    assert m2.history[0]["query"] == "Q2"


def test_long_history_does_not_crash():
    m = MemoryAgent()
    for i in range(50):
        _add(m, f"Q{i}", f"A{i}", sources=[f"doc{i}.txt"])
    ctx = m.get_context(last_n=10)
    assert "Q50: Q49" in ctx
    assert "S50: doc49.txt" in ctx


def test_no_sources_renders_correctly():
    m = MemoryAgent()
    _add(m, "Q", "A", sources=[])
    ctx = m.get_context()
    assert "S1: (no sources)" in ctx
