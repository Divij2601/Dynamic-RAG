"""Unit tests for retrieval metric functions (pure, no services)."""

from evaluation.utils.metrics import (
    recall_at_k,
    hit_rate,
    reciprocal_rank,
    ndcg_at_k,
    context_precision,
    context_recall,
)


def test_recall_perfect():
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x"], ["a", "b"]) == 0.5


def test_recall_empty_relevant_returns_one():
    # No gold chunks -> recall is defined as 1.0
    assert recall_at_k(["a"], []) == 1.0


def test_hit_rate():
    assert hit_rate(["x", "a"], ["a"]) == 1.0
    assert hit_rate(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_first_position_is_one():
    assert ndcg_at_k(["a", "x"], ["a"]) == 1.0


def test_ndcg_no_hit_is_zero():
    assert ndcg_at_k(["x", "y"], ["a"]) == 0.0


def test_context_precision():
    assert context_precision(["a", "x"], ["a"]) == 0.5
    assert context_precision([], ["a"]) == 0.0


def test_context_recall_matches_recall():
    assert context_recall(["a", "x"], ["a", "b"]) == 0.5
