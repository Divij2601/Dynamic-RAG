"""Unit tests for the verifier verdict parser and planner JSON parsing."""

from src.generation.verifier import faithfulness_verifier
from src.planner.planner import query_planner


# ---- Verifier verdict parsing ----

def test_verdict_clean_json():
    v = faithfulness_verifier._parse_verdict(
        '{"faithfulness_score": 0.9, "grounded": true, '
        '"unsupported_claims": [], "reasoning": "ok"}'
    )
    assert v["grounded"] is True
    assert v["faithful"] is True
    assert v["faithfulness_score"] == 0.9


def test_verdict_fenced_json():
    v = faithfulness_verifier._parse_verdict(
        '```json\n{"faithfulness_score": 0.1, '
        '"grounded": false}\n```'
    )
    assert v["grounded"] is False
    assert v["faithful"] is False


def test_verdict_below_threshold_not_faithful():
    v = faithfulness_verifier._parse_verdict(
        '{"faithfulness_score": 0.5, "grounded": true}'
    )
    # grounded but below FAITHFULNESS_THRESHOLD (0.7)
    assert v["faithful"] is False


def test_verdict_clamps_score():
    v = faithfulness_verifier._parse_verdict(
        '{"faithfulness_score": 5, "grounded": true}'
    )
    assert v["faithfulness_score"] == 1.0


# ---- Planner parsing helpers ----

def test_planner_parse_extracts_object():
    d = query_planner._parse(
        'noise {"route": "internal_rag", '
        '"confidence": 0.9} trailing'
    )
    assert d["route"] == "internal_rag"


def test_planner_coerce_confidence():
    assert query_planner._coerce_confidence("1.5") == 1.0
    assert query_planner._coerce_confidence(-1) == 0.0
    assert query_planner._coerce_confidence(None) == 0.7


def test_planner_coerce_list():
    assert query_planner._coerce_list(["a", 1]) == ["a", "1"]
    assert query_planner._coerce_list("not a list") == []
