"""
BM25 sparse retriever with startup cache.

The corpus is loaded from Qdrant once (at first
retrieval call) and rebuilt only when explicitly
invalidated — e.g. after a new document is ingested.
This avoids the O(N) Qdrant scroll on every query.
"""

import re
import time
from typing import Dict, List, Optional

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
    BM25 sparse retrieval over the Qdrant corpus.
    The BM25 index is built once and cached in memory.
    Call invalidate_cache() after ingesting new docs.
    """

    def __init__(self):

        self.collection_name = (
            settings.QDRANT_COLLECTION_NAME
        )

        self.top_k = settings.TOP_K

        # Cache state
        self._corpus: Optional[List[List[str]]] = None
        self._payloads: Optional[List[Dict]] = None
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:

        return re.findall(r"\w+", text.lower())

    def _ensure_cache(self):
        """
        Build the BM25 index if it has not been
        built yet or has been invalidated.
        """

        if self._bm25 is not None:
            return

        app_logger.info(
            "Building BM25 index from Qdrant corpus..."
        )

        client = qdrant_client.get_client()

        points = []
        offset = None

        while True:

            result = client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                offset=offset
            )

            batch, next_offset = result
            points.extend(batch)

            if not next_offset:
                break

            offset = next_offset

        self._payloads = [
            p.payload for p in points
        ]

        self._corpus = [
            self._tokenize(
                p.get("text", "")
            )
            for p in self._payloads
        ]

        self._bm25 = BM25Okapi(self._corpus)

        app_logger.success(
            f"BM25 index built: "
            f"{len(self._corpus)} documents"
        )

    def invalidate_cache(self):
        """
        Force a rebuild on the next retrieval call.
        Should be called after new documents are indexed.
        """

        self._corpus = None
        self._payloads = None
        self._bm25 = None

        app_logger.info(
            "BM25 cache invalidated — "
            "will rebuild on next query"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict:

        start_time = time.perf_counter()

        if top_k is None:
            top_k = self.top_k

        self._ensure_cache()

        tokenized_query = self._tokenize(query)

        scores = self._bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        retrieval_time = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        results = []

        for idx in ranked_indices:

            payload = self._payloads[idx]

            results.append({
                "score": float(scores[idx]),
                "document_id":
                payload.get("document_id"),
                "chunk_id":
                payload.get("chunk_id"),
                "page": payload.get("page"),
                "text": payload.get("text"),
                "metadata":
                payload.get("metadata", {})
            })

        app_logger.success(
            f"Sparse retrieval returned "
            f"{len(results)} chunks in "
            f"{retrieval_time} ms"
        )

        return {
            "query": query,
            "retrieval_type": "sparse",
            "latency_ms": retrieval_time,
            "results": results
        }


sparse_retriever = SparseRetriever()
