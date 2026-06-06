import time
from typing import List, Dict

from src.config import settings
from src.database.qdrant_client import (
    qdrant_client
)
from src.ingestion.embedder import (
    embedding_generator
)
from src.observability.logger import (
    app_logger
)


class DenseRetriever:
    """
    Dense vector retrieval
    using Qdrant
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

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict:
        """
        Dense retrieval
        """

        start_time = (
            time.perf_counter()
        )

        if top_k is None:
            top_k = self.top_k

        query_embedding = (
            embedding_generator
            .generate_query_embedding(
                query
            )
        )

        results = (
            self.client.query_points(
                collection_name=(
                    self.collection_name
                ),

                query=query_embedding,

                limit=top_k,

                with_payload=True
            )
        )

        retrieval_time = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        retrieved_chunks = []

        for result in results.points:

            payload = (
                result.payload
            )

            retrieved_chunks.append({
                "score":
                float(result.score),

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
            f"Dense retrieval "
            f"returned "
            f"{len(retrieved_chunks)} "
            f"chunks in "
            f"{retrieval_time} ms"
        )

        return {
            "query": query,

            "retrieval_type":
            "dense",

            "latency_ms":
            retrieval_time,

            "results":
            retrieved_chunks
        }


dense_retriever = (
    DenseRetriever()
)