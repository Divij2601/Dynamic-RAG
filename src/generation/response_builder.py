from typing import List, Dict, Any, Optional

from src.graph.state import (
    EvidenceItem,
    FinalResponse
)

from src.evaluation.confidence import (
    confidence_calibrator
)

from src.observability.logger import (
    app_logger
)


class ResponseBuilder:
    """
    Final response builder.

    Combines the answer, evidence and the
    verifier verdict into a structured
    FinalResponse, applying confidence
    calibration.
    """

    def build(
        self,
        answer: str,

        route: str,

        evidence_items:
        List[EvidenceItem],

        verification: Dict[str, Any],

        status: str = "success",

        query_id: Optional[str] = None,

        session_id: Optional[str] = None
    ) -> FinalResponse:
        """
        Build final response.
        """

        # faithfulness_score may legitimately be
        # None (e.g. verifier degraded or a route
        # with no evidence). Use a neutral 0.5 only
        # for the confidence calculation, but keep
        # the true value on the response.
        raw_score = verification.get(
            "faithfulness_score"
        )

        calc_score = (
            0.5 if raw_score is None
            else raw_score
        )

        confidence = (
            confidence_calibrator
            .calculate(
                evidence_items=evidence_items,
                faithfulness_score=calc_score
            )
        )

        response = FinalResponse(
            answer=answer,

            route=route,

            confidence=confidence,

            faithfulness_score=raw_score,

            grounded=verification.get(
                "grounded"
            ),

            unsupported_claims=verification.get(
                "unsupported_claims",
                []
            ),

            reasoning=verification.get(
                "reasoning",
                ""
            ),

            sources=evidence_items,

            status=status,

            query_id=query_id,

            session_id=session_id
        )

        app_logger.success(
            f"Final response built "
            f"(status={status}, "
            f"confidence={confidence})"
        )

        return response


response_builder = ResponseBuilder()
