import math
from typing import List

from src.graph.state import (
    EvidenceItem
)


def _sigmoid(x: float) -> float:
    """
    Map an unbounded score into (0, 1).
    Cross-encoder rerank scores are raw logits
    (can be large negative for irrelevant chunks),
    so they must be squashed before averaging.
    """

    # Guard against overflow for very negative x.
    if x < -60:
        return 0.0

    return 1.0 / (1.0 + math.exp(-x))


class ConfidenceCalibrator:
    """
    Confidence calibration
    for Dynamic-RAG
    """

    def calculate(
        self,
        evidence_items:
        List[EvidenceItem],

        faithfulness_score:
        float
    ) -> float:
        """
        Calculate confidence in [0, 1].
        """

        if not evidence_items:
            return 0.10

        # Normalize each evidence score into (0, 1)
        # before averaging so unbounded cross-encoder
        # logits cannot produce negative confidence.
        normalized_scores = [
            _sigmoid(evidence.score)
            for evidence in evidence_items
        ]

        rerank_confidence = (
            sum(normalized_scores)
            / len(normalized_scores)
        )

        evidence_coverage = min(
            len(evidence_items)
            / 5,
            1.0
        )

        # faithfulness_score is already in [0, 1]
        # (callers substitute a neutral value when
        # it is unknown).
        confidence = (
            0.4
            * rerank_confidence
            +
            0.5
            * faithfulness_score
            +
            0.1
            * evidence_coverage
        )

        # Final safety clamp.
        confidence = max(
            0.0,
            min(confidence, 1.0)
        )

        return round(confidence, 3)


confidence_calibrator = (
    ConfidenceCalibrator()
)