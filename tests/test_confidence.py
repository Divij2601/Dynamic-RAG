"""Unit tests for confidence calibration (regression test for the
negative-confidence bug with unbounded cross-encoder scores)."""

from src.evaluation.confidence import confidence_calibrator
from src.graph.state import EvidenceItem


def _ev(score):
    return EvidenceItem(
        source_type="document",
        text="x",
        score=score
    )


def test_empty_evidence_low_confidence():
    assert confidence_calibrator.calculate([], 0.9) == 0.10


def test_negative_rerank_scores_stay_in_range():
    # Cross-encoder logits can be large negatives;
    # confidence must never go below 0.
    items = [_ev(-9.0), _ev(-8.0), _ev(-12.0)]
    c = confidence_calibrator.calculate(items, 1.0)
    assert 0.0 <= c <= 1.0


def test_strong_evidence_high_confidence():
    items = [_ev(8.0), _ev(7.0), _ev(6.0), _ev(5.0), _ev(5.0)]
    c = confidence_calibrator.calculate(items, 1.0)
    assert c > 0.8


def test_always_bounded():
    items = [_ev(100.0), _ev(100.0)]
    c = confidence_calibrator.calculate(items, 1.0)
    assert 0.0 <= c <= 1.0
