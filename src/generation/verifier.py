from typing import List

from src.graph.state import (
    EvidenceItem,
    FinalResponse
)

from src.observability.logger import (
    app_logger
)


class ResponseBuilder:
    """
    Build final structured
    Dynamic-RAG response
    """

    def build(
        self,
        answer: str,

        route: str,

        evidence_items:
        List[EvidenceItem],

        confidence: float = 0.85
    ) -> FinalResponse:
        """
        Build final response
        """

        response = (
            FinalResponse(
                answer=answer,

                sources=(
                    evidence_items
                ),

                route=route,

                confidence=(
                    confidence
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