from uuid import uuid4
from typing import List

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

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


class SemanticMemory:
    """
    Semantic memory system
    """

    def __init__(self):

        self.client = (
            qdrant_client
            .get_client()
        )

        self.collection_name = (
            settings
            .MEMORY_COLLECTION_NAME
        )

        self._ensure_collection()

    def _ensure_collection(
        self
    ):
        """
        Create memory collection
        """

        collections = [
            c.name
            for c in (
                self.client
                .get_collections()
                .collections
            )
        ]

        if (
            self.collection_name
            not in collections
        ):

            self.client.create_collection(
                collection_name=(
                    self.collection_name
                ),

                vectors_config=(
                    VectorParams(
                        size=(
                            settings
                            .VECTOR_DIMENSION
                        ),

                        distance=(
                            Distance.COSINE
                        )
                    )
                )
            )

            app_logger.success(
                "Semantic memory "
                "collection created"
            )

    def store_memory(
        self,
        session_id: str,

        query: str,

        answer: str
    ):
        """
        Store semantic memory
        """

        combined_text = f"""
User:
{query}

Assistant:
{answer}
"""

        embedding = (
            embedding_generator
            .generate_query_embedding(
                combined_text
            )
        )

        self.client.upsert(
            collection_name=(
                self.collection_name
            ),

            points=[
                PointStruct(
                    id=str(
                        uuid4()
                    ),

                    vector=embedding,

                    payload={
                        "session_id":
                        session_id,

                        "query":
                        query,

                        "answer":
                        answer,

                        "text":
                        combined_text
                    }
                )
            ]
        )

        app_logger.success(
            "Semantic memory stored"
        )

    def retrieve_memory(
        self,
        session_id: str,

        query: str,

        top_k: int = 3
    ) -> str:
        """
        Retrieve semantic memory
        """

        embedding = (
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

                query=embedding,

                limit=top_k,

                with_payload=True
            )
        )

        memories = []

        for point in (
            results.points
        ):

            payload = (
                point.payload
            )

            if (
                payload.get(
                    "session_id"
                )
                ==
                session_id
            ):

                memories.append(
                    payload.get(
                        "text",
                        ""
                    )
                )

        context = "\n\n".join(
            memories
        )

        app_logger.success(
            f"Retrieved "
            f"{len(memories)} "
            f"semantic memories"
        )

        return context


semantic_memory = (
    SemanticMemory()
)