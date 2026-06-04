import hashlib
from datetime import datetime
from typing import Dict, List

from src.observability.logger import (
    app_logger
)


PIPELINE_VERSION = "v1.0"


class MetadataBuilder:
    """
    Build chunk metadata
    for accountability and tracking.
    """

    def enrich_chunks(
        self,
        chunks: List[Dict],
        filename: str,
        document_version: str = "1.0"
    ) -> List[Dict]:
        """
        Add metadata to chunks
        """

        enriched_chunks = []

        timestamp = (
            datetime.utcnow()
            .isoformat()
        )

        for index, chunk in enumerate(
            chunks
        ):

            content_hash = (
                self._generate_hash(
                    chunk["text"]
                )
            )

            enriched_chunk = {
                **chunk,

                "metadata": {
                    **chunk.get(
                        "metadata",
                        {}
                    ),

                    "filename":
                    filename,

                    "document_version":
                    document_version,

                    "pipeline_version":
                    PIPELINE_VERSION,

                    "chunk_index":
                    index,

                    "upload_timestamp":
                    timestamp,

                    "content_hash":
                    content_hash
                }
            }

            enriched_chunks.append(
                enriched_chunk
            )

        app_logger.success(
            f"Metadata enriched "
            f"for "
            f"{len(chunks)} chunks"
        )

        return enriched_chunks

    def _generate_hash(
        self,
        text: str
    ) -> str:
        """
        Generate deterministic hash
        for chunk content
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()


metadata_builder = (
    MetadataBuilder()
)