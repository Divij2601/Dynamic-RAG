from typing import (
    List,
    Optional,
    Dict,
    Any
)

from pydantic import BaseModel, Field


# ==========================================
# Evidence Object
# ==========================================

class EvidenceItem(BaseModel):
    """
    One evidence unit
    """

    source_type: str
    source_id: Optional[str] = None
    chunk_id: Optional[str] = None

    page: Optional[int] = None
    section: Optional[str] = None

    text: str
    score: float = 0.0

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


# ==========================================
# Planner Output
# ==========================================

class PlannerOutput(BaseModel):
    """
    Planner decision object
    """

    intent: Optional[str] = None
    complexity: Optional[str] = None

    route: str = "unknown"

    confidence: float = 0.0

    needs_retrieval: bool = False
    needs_web: bool = False
    needs_decomposition: bool = False

    subqueries: List[str] = Field(
        default_factory=list
    )

    budget: Optional[str] = None


# ==========================================
# Retrieval Metrics
# ==========================================

class RetrievalMetrics(BaseModel):
    """
    Retrieval evaluation metrics
    """

    retrieval_latency_ms: float = 0.0

    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    hit_rate: Optional[float] = None

    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


# ==========================================
# Generation Metrics
# ==========================================

class GenerationMetrics(BaseModel):
    """
    Generation evaluation metrics
    """

    faithfulness: Optional[float] = None
    groundedness: Optional[float] = None
    answer_relevance: Optional[float] = None
    completeness: Optional[float] = None

    citation_accuracy: Optional[float] = None

    generation_latency_ms: float = 0.0


# ==========================================
# Verification Result
# ==========================================

class VerificationResult(BaseModel):
    """
    Critic result
    """

    faithful: bool = False
    supported: bool = False
    complete: bool = False

    retry_required: bool = False

    severity: str = "low"

    issues: List[str] = Field(
        default_factory=list
    )


# ==========================================
# Final Response
# ==========================================

class FinalResponse(BaseModel):
    """
    Final user-facing response
    """

    answer: str = ""

    sources: List[EvidenceItem] = Field(
        default_factory=list
    )

    route: str = "unknown"

    confidence: float = 0.0
    status: str = "pending"


# ==========================================
# Main Query State
# ==========================================

class QueryState(BaseModel):
    """
    Master state object for Dynamic-RAG
    """

    # Request Metadata
    request_id: str
    session_id: str

    # User Input
    query_text: str

    # Memory
    chat_history: List[Dict] = Field(
        default_factory=list
    )

    memory_context: List[str] = Field(
        default_factory=list
    )

    # Planner
    planner_output: Optional[
        PlannerOutput
    ] = None

    selected_route: Optional[
        str
    ] = None

    # Evidence
    internal_evidence: List[
        EvidenceItem
    ] = Field(default_factory=list)

    web_evidence: List[
        EvidenceItem
    ] = Field(default_factory=list)

    # Generation
    candidate_answer: Optional[
        str
    ] = None

    # Evaluation
    retrieval_metrics: RetrievalMetrics = Field(
        default_factory=RetrievalMetrics
    )

    generation_metrics: GenerationMetrics = Field(
        default_factory=GenerationMetrics
    )

    verification_result: Optional[
        VerificationResult
    ] = None

    # Final Output
    final_response: Optional[
        FinalResponse
    ] = None

    # Runtime
    retry_count: int = 0

    error: Optional[str] = None