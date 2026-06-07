from typing import Dict, List

from src.config import settings
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
    Hybrid retrieval combining dense + sparse.

    Supports two fusion strategies (settings.FUSION_MODE):

      - "weighted": score-weighted sum. Dense cosine
        scores (already 0..1) and min-max normalized
        sparse (BM25) scores are combined with
        DENSE_WEIGHT / SPARSE_WEIGHT. Favors dense
        semantic similarity.

      - "rrf": Reciprocal Rank Fusion. Rank-based and
        robust to score-scale differences.
    """

    def __init__(self):

        self.fusion_mode = settings.FUSION_MODE
        self.dense_weight = settings.DENSE_WEIGHT
        self.sparse_weight = settings.SPARSE_WEIGHT
        self.rrf_k = settings.RRF_K

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Hybrid retrieval. Fetches a wider pool from
        each retriever, fuses, and returns the top_k
        fused candidates.
        """

        dense_results = dense_retriever.retrieve(
            query=query,
            top_k=top_k * 2
        )

        sparse_results = sparse_retriever.retrieve(
            query=query,
            top_k=top_k * 2
        )

        fused_results = self._fuse_scores(
            dense_results["results"],
            sparse_results["results"]
        )

        final_results = sorted(
            fused_results.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:top_k]

        app_logger.success(
            f"Hybrid retrieval ({self.fusion_mode}) "
            f"returned {len(final_results)} chunks"
        )

        return {
            "query": query,
            "retrieval_type": "hybrid",
            "fusion_mode": self.fusion_mode,
            "results": final_results
        }

    def _fuse_scores(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> Dict:

        if self.fusion_mode == "rrf":
            return self._fuse_rrf(
                dense_results,
                sparse_results
            )

        return self._fuse_weighted(
            dense_results,
            sparse_results
        )

    # ------------------------------------------------

    def _fuse_weighted(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> Dict:
        """
        Score-weighted fusion (dense-favored).
        """

        combined: Dict[str, Dict] = {}

        max_sparse = max(
            (r["score"] for r in sparse_results),
            default=0.0
        ) or 1.0

        for result in dense_results:

            chunk_id = result["chunk_id"]

            combined[chunk_id] = {
                **result,
                "dense_score": result["score"],
                "sparse_score": 0.0,
                "hybrid_score":
                result["score"] * self.dense_weight
            }

        for result in sparse_results:

            chunk_id = result["chunk_id"]

            normalized = result["score"] / max_sparse

            if chunk_id in combined:
                combined[chunk_id][
                    "sparse_score"
                ] = normalized
                combined[chunk_id][
                    "hybrid_score"
                ] += normalized * self.sparse_weight
            else:
                combined[chunk_id] = {
                    **result,
                    "dense_score": 0.0,
                    "sparse_score": normalized,
                    "hybrid_score":
                    normalized * self.sparse_weight
                }

        return combined

    def _fuse_rrf(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> Dict:
        """
        Reciprocal Rank Fusion. Each list contributes
        weight / (k + rank) per chunk.
        """

        combined: Dict[str, Dict] = {}

        def _add_list(results, weight, field):
            for rank, result in enumerate(
                results, start=1
            ):
                chunk_id = result["chunk_id"]
                contribution = weight * (
                    1.0 / (self.rrf_k + rank)
                )
                if chunk_id not in combined:
                    combined[chunk_id] = {
                        **result,
                        "dense_score": 0.0,
                        "sparse_score": 0.0,
                        "hybrid_score": 0.0
                    }
                combined[chunk_id][
                    "hybrid_score"
                ] += contribution
                combined[chunk_id][field] = (
                    result["score"]
                )

        _add_list(
            dense_results,
            self.dense_weight,
            "dense_score"
        )
        _add_list(
            sparse_results,
            self.sparse_weight,
            "sparse_score"
        )

        return combined


hybrid_retriever = HybridRetriever()
