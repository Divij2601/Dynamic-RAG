from typing import List

from src.graph.state import (
    EvidenceItem
)

from src.observability.logger import (
    app_logger
)


class EvidenceBuilder:
    """
    Convert retrieved chunks
    into EvidenceItem objects
    """

    def build(
        self,
        retrieved_chunks: list
    ) -> List[EvidenceItem]:
        """
        Build evidence objects
        """

        evidence_items = []

        for chunk in (
            retrieved_chunks
        ):

            evidence = (
                EvidenceItem(
                    source_type="document",

                    source_id=(
                        chunk.get(
                            "document_id"
                        )
                    ),

                    chunk_id=(
                        chunk.get(
                            "chunk_id"
                        )
                    ),

                    page=(
                        chunk.get(
                            "page"
                        )
                    ),

                    text=(
                        chunk.get(
                            "text",
                            ""
                        )
                    ),

                    score=float(
                        chunk.get(
                            "rerank_score",

                            chunk.get(
                                "hybrid_score",

                                chunk.get(
                                    "score",
                                    0.0
                                )
                            )
                        )
                    ),

                    metadata=(
                        chunk.get(
                            "metadata",
                            {}
                        )
                    )
                )
            )

            evidence_items.append(
                evidence
            )

        app_logger.success(
            f"Built "
            f"{len(evidence_items)} "
            f"evidence items"
        )

        return evidence_items


evidence_builder = (
    EvidenceBuilder()
)