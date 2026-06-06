import re
import time
from typing import Dict, List

from rank_bm25 import BM25Okapi

from src.config import settings
from src.database.qdrant_client import (
    qdrant_client
)
from src.observability.logger import (
    app_logger
)


class SparseRetriever:
    """
    BM25 sparse retrieval
    """

    def __init__(self):

        self.client = (
            qdrant_client
            .get_client()
        )

        self.collection_name = (
            settings
            .QDRANT_COLLECTION_NAME
        )

        self.top_k = (
            settings.TOP_K
        )

    def _tokenize(
        self,
        text: str
    ) -> List[str]:
        """
        Simple tokenizer
        """

        text = text.lower()

        return re.findall(
            r"\w+",
            text
        )

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict:
        """
        Sparse retrieval
        using BM25
        """

        start_time = (
            time.perf_counter()
        )

        if top_k is None:
            top_k = self.top_k

        scroll_result = (
            self.client.scroll(
                collection_name=(
                    self.collection_name
                ),

                limit=10000,

                with_payload=True
            )
        )

        points = (
            scroll_result[0]
        )

        corpus = []

        payloads = []

        for point in points:

            payload = (
                point.payload
            )

            text = (
                payload.get(
                    "text",
                    ""
                )
            )

            corpus.append(
                self._tokenize(
                    text
                )
            )

            payloads.append(
                payload
            )

        bm25 = (
            BM25Okapi(corpus)
        )

        tokenized_query = (
            self._tokenize(
                query
            )
        )

        scores = (
            bm25.get_scores(
                tokenized_query
            )
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i:
            scores[i],
            reverse=True
        )[:top_k]

        retrieval_time = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        results = []

        for idx in ranked_indices:

            payload = (
                payloads[idx]
            )

            results.append({
                "score":
                float(scores[idx]),

                "document_id":
                payload.get(
                    "document_id"
                ),

                "chunk_id":
                payload.get(
                    "chunk_id"
                ),

                "page":
                payload.get(
                    "page"
                ),

                "text":
                payload.get(
                    "text"
                ),

                "metadata":
                payload.get(
                    "metadata",
                    {}
                )
            })

        app_logger.success(
            f"Sparse retrieval "
            f"returned "
            f"{len(results)} "
            f"chunks in "
            f"{retrieval_time} ms"
        )

        return {
            "query": query,

            "retrieval_type":
            "sparse",

            "latency_ms":
            retrieval_time,

            "results":
            results
        }


sparse_retriever = (
    SparseRetriever()
)