from typing import List, Dict, Any
from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """
    Single benchmark sample
    """

    query: str

    ground_truth_answer: str

    relevant_chunk_ids: List[str]

    answerable: bool = True

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


class RetrievalMetrics(BaseModel):
    """
    Plane 1 metrics
    """

    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    context_precision: float
    context_recall: float


class GenerationMetrics(BaseModel):
    """
    Plane 2 metrics
    """

    faithfulness: float
    groundedness: float
    answer_relevance: float
    completeness: float
    citation_accuracy: float
    noise_robustness: float = 0.0


class SystemMetrics(BaseModel):
    """
    Plane 3 metrics
    """

    latency_ms: float
    p95_latency_ms: float
    cost_per_query: float
    rejection_rate: float
    end_to_end_accuracy: float