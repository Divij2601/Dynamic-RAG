import time
from typing import Dict, List

from sentence_transformers import (
    CrossEncoder
)

from src.config import settings
from src.observability.logger import (
    app_logger
)


class Reranker:
    """
    Cross-encoder reranker
    """

    _model = None

    def __init__(self):

        self.model_name = (
            settings.RERANK_MODEL
        )

        self.final_top_k = (
            settings.FINAL_TOP_K
        )

    def _load_model(self):
        """
        Singleton load
        """

        if self._model is None:

            app_logger.info(
                f"Loading reranker: "
                f"{self.model_name}"
            )

            self._model = (
                CrossEncoder(
                    self.model_name
                )
            )

            app_logger.success(
                "Reranker loaded"
            )

        return self._model

    def rerank(
        self,
        query: str,
        retrieved_chunks:
        List[Dict]
    ) -> Dict:
        """
        Rerank retrieved chunks
        """

        start_time = (
            time.perf_counter()
        )

        model = (
            self._load_model()
        )

        pairs = []

        for chunk in (
            retrieved_chunks
        ):

            pairs.append(
                (
                    query,
                    chunk["text"]
                )
            )

        scores = (
            model.predict(
                pairs
            )
        )

        reranked = []

        for chunk, score in zip(
            retrieved_chunks,
            scores
        ):

            reranked.append({
                **chunk,
                "rerank_score":
                float(score)
            })

        reranked = sorted(
            reranked,
            key=lambda x:
            x["rerank_score"],
            reverse=True
        )[
            :self.final_top_k
        ]

        rerank_time = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        app_logger.success(
            f"Reranked "
            f"{len(reranked)} "
            f"chunks in "
            f"{rerank_time} ms"
        )

        return {
            "query": query,
            "latency_ms":
            rerank_time,
            "results":
            reranked
        }


reranker = Reranker()