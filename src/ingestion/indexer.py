from typing import List, Dict
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from src.config import settings
from src.database.qdrant_client import (
    qdrant_client
)
from src.observability.logger import (
    app_logger
)


class QdrantIndexer:
    """
    Index embeddings into Qdrant
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

    def create_collection(self):
        """
        Create collection
        if it does not exist
        """

        existing = [
            collection.name
            for collection in (
                self.client
                .get_collections()
                .collections
            )
        ]

        if (
            self.collection_name
            not in existing
        ):

            self.client.create_collection(
                collection_name=(
                    self.collection_name
                ),

                vectors_config=(
                    VectorParams(
                        size=settings
                        .VECTOR_DIMENSION,

                        distance=(
                            Distance.COSINE
                        )
                    )
                )
            )

            app_logger.success(
                f"Collection created: "
                f"{self.collection_name}"
            )

        else:

            app_logger.info(
                f"Collection exists: "
                f"{self.collection_name}"
            )

    def index_chunks(
        self,
        chunks: List[Dict]
    ):
        """
        Insert chunks into Qdrant
        """

        self.create_collection()

        points = []

        for chunk in chunks:

            payload = {
                "document_id":
                chunk[
                    "document_id"
                ],

                "chunk_id":
                chunk[
                    "chunk_id"
                ],

                "page":
                chunk.get(
                    "page"
                ),

                "text":
                chunk[
                    "text"
                ],

                "metadata":
                chunk[
                    "metadata"
                ]
            }

            # Deterministic point id from the
            # chunk_id keeps re-ingestion
            # idempotent (same chunk upserts
            # in place instead of duplicating).
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    chunk["chunk_id"]
                )
            )

            points.append(
                PointStruct(
                    id=point_id,

                    vector=chunk[
                        "embedding"
                    ],

                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=(
                self.collection_name
            ),

            points=points
        )

        app_logger.success(
            f"Indexed "
            f"{len(points)} "
            f"chunks into Qdrant"
        )


qdrant_indexer = (
    QdrantIndexer()
)