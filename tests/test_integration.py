"""
Integration tests requiring live services (Qdrant, Mongo)
and/or the Groq API. Run explicitly with:

    pytest -m integration

They are skipped by the default fast suite
(`pytest -m "not integration"`).
"""

import pytest


@pytest.mark.integration
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/health")

    assert r.status_code == 200
    assert "status" in r.json()


@pytest.mark.integration
def test_hybrid_retrieval_returns_results():
    from src.retrieval.hybrid import hybrid_retriever

    out = hybrid_retriever.retrieve(
        "When did the Soviet Union dissolve?",
        top_k=10
    )
    assert len(out["results"]) > 0
    assert "chunk_id" in out["results"][0]


@pytest.mark.integration
def test_retrieval_eval_plane1():
    # Local (no Groq): retrieval + rerank over the corpus.
    from evaluation.retrieval_eval import retrieval_evaluator

    metrics = retrieval_evaluator.evaluate(
        "evaluation/data/test_set.json"
    )
    assert metrics["Hit Rate"] >= 0.9


@pytest.mark.integration
def test_end_to_end_query():
    # Needs Groq quota.
    from src.graph.graph_builder import run_query

    fr = run_query("When did World War II begin?")
    assert fr.status in ("success", "abstained")
    assert fr.route
