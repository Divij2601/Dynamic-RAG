from typing import List

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
    Final response builder
    """

    def build(
        self,
        answer: str,

        route: str,

        evidence_items:
        List[EvidenceItem],

        verification:
        dict
    ) -> FinalResponse:
        """
        Build final response
        """

        faithfulness_score = (
            verification.get(
                "faithfulness_score",
                0.5
            )
        )

        confidence = (
            confidence_calibrator
            .calculate(
                evidence_items=(
                    evidence_items
                ),

                faithfulness_score=(
                    faithfulness_score
                )
            )
        )

        response = (
            FinalResponse(
                answer=answer,

                route=route,

                confidence=(
                    confidence
                ),

                faithfulness_score=(
                    faithfulness_score
                ),

                grounded=(
                    verification.get(
                        "grounded",
                        False
                    )
                ),

                unsupported_claims=(
                    verification.get(
                        "unsupported_claims",
                        []
                    )
                ),

                reasoning=(
                    verification.get(
                        "reasoning",
                        ""
                    )
                ),

                sources=(
                    evidence_items
                ),

                status="success"
            )
        )

        app_logger.success(
            "Final response built"
        )

        return response


response_builder = (
    ResponseBuilder()
)