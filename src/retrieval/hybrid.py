from typing import Dict, List

from src.retrieval.dense import (
    dense_retriever
)

from src.retrieval.sparse import (
    sparse_retriever
)

from src.observability.logger import (
    app_logger
)


class HybridRetriever:
    """
    Hybrid retrieval
    combining dense + sparse
    """

    def __init__(self):

        self.dense_weight = 0.7
        self.sparse_weight = 0.3

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Hybrid retrieval
        """

        dense_results = (
            dense_retriever
            .retrieve(
                query=query,
                top_k=top_k * 2
            )
        )

        sparse_results = (
            sparse_retriever
            .retrieve(
                query=query,
                top_k=top_k * 2
            )
        )

        fused_results = (
            self._fuse_scores(
                dense_results[
                    "results"
                ],

                sparse_results[
                    "results"
                ]
            )
        )

        final_results = sorted(
            fused_results.values(),
            key=lambda x:
            x["hybrid_score"],
            reverse=True
        )[:top_k]

        app_logger.success(
            f"Hybrid retrieval "
            f"returned "
            f"{len(final_results)} "
            f"chunks"
        )

        return {
            "query": query,

            "retrieval_type":
            "hybrid",

            "results":
            final_results
        }

    def _fuse_scores(
        self,
        dense_results:
        List[Dict],

        sparse_results:
        List[Dict]
    ) -> Dict:
        """
        Fuse dense and sparse scores
        """

        combined = {}

        max_sparse_score = max(
            (
                r["score"]
                for r in sparse_results
            ),
            default=0
        )
        
        # Prevent divide-by-zero
        if max_sparse_score == 0:
            max_sparse_score = 1

        # Dense scores
        for result in dense_results:

            chunk_id = (
                result["chunk_id"]
            )

            combined[
                chunk_id
            ] = {
                **result,

                "dense_score":
                result["score"],

                "sparse_score":
                0.0,

                "hybrid_score":
                (
                    result["score"]
                    *
                    self
                    .dense_weight
                )
            }

        # Sparse scores
        for result in sparse_results:

            chunk_id = (
                result["chunk_id"]
            )

            normalized_sparse = (
                result["score"]
                / max_sparse_score
            )

            if (
                chunk_id
                in combined
            ):

                combined[
                    chunk_id
                ][
                    "sparse_score"
                ] = (
                    normalized_sparse
                )

                combined[
                    chunk_id
                ][
                    "hybrid_score"
                ] += (
                    normalized_sparse
                    *
                    self
                    .sparse_weight
                )

            else:

                combined[
                    chunk_id
                ] = {
                    **result,

                    "dense_score":
                    0.0,

                    "sparse_score":
                    normalized_sparse,

                    "hybrid_score":
                    (
                        normalized_sparse
                        *
                        self
                        .sparse_weight
                    )
                }

        return combined


hybrid_retriever = (
    HybridRetriever()
)