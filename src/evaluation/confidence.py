from typing import List

from src.graph.state import (
    EvidenceItem
)


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
        Calculate confidence
        """

        if not evidence_items:
            return 0.10

        rerank_scores = [
            evidence.score
            for evidence
            in evidence_items
        ]

        rerank_confidence = (
            sum(rerank_scores)
            / len(rerank_scores)
        )

        rerank_confidence = min(
            rerank_confidence,
            1.0
        )

        evidence_coverage = min(
            len(evidence_items)
            / 5,
            1.0
        )

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

        confidence = round(
            confidence,
            3
        )

        return confidence


confidence_calibrator = (
    ConfidenceCalibrator()
)