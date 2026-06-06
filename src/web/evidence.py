from typing import List

from src.graph.state import (
    EvidenceItem
)

from src.observability.logger import (
    app_logger
)


class WebEvidenceBuilder:
    """
    Convert web results
    into EvidenceItem objects
    """

    def build(
        self,
        web_results: list
    ) -> List[EvidenceItem]:
        """
        Build web evidence
        """

        evidence_items = []

        for item in web_results:

            evidence = (
                EvidenceItem(
                    source_type="web",

                    source_id=(
                        item.get(
                            "url"
                        )
                    ),

                    chunk_id=None,

                    page=None,

                    text=(
                        item.get(
                            "content",
                            ""
                        )
                    ),

                    score=float(
                        item.get(
                            "score",
                            0.0
                        )
                    ),

                    metadata={
                        "title":
                        item.get(
                            "title",
                            ""
                        ),

                        "url":
                        item.get(
                            "url",
                            ""
                        )
                    }
                )
            )

            evidence_items.append(
                evidence
            )

        app_logger.success(
            f"Built "
            f"{len(evidence_items)} "
            f"web evidence items"
        )

        return evidence_items


web_evidence_builder = (
    WebEvidenceBuilder()
)