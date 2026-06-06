"""
Ingestion pipeline orchestrator.

Chains the ingestion stages into one
deterministic flow:

    load -> parse -> chunk -> enrich
         -> embed -> index

Usable programmatically:

    from src.ingestion.pipeline import (
        ingestion_pipeline
    )
    ingestion_pipeline.ingest_file("a.txt")
    ingestion_pipeline.ingest_directory(
        "data/raw/primary"
    )

or from the command line:

    python -m src.ingestion.pipeline data/raw/primary
"""

import sys
from pathlib import Path
from typing import Dict, List

from src.ingestion.loader import (
    document_loader,
    SUPPORTED_EXTENSIONS
)
from src.ingestion.parser import (
    document_parser
)
from src.ingestion.chunker import (
    document_chunker
)
from src.ingestion.metadata import (
    metadata_builder
)
from src.ingestion.embedder import (
    embedding_generator
)
from src.ingestion.indexer import (
    qdrant_indexer
)
from src.observability.logger import (
    app_logger
)


class IngestionPipeline:
    """
    End-to-end document ingestion.
    """

    def ingest_file(
        self,
        file_path: str,
        document_version: str = "1.0"
    ) -> Dict:
        """
        Ingest a single file and return a
        summary of what was indexed.
        """

        app_logger.info(
            f"Ingesting: {file_path}"
        )

        # 1. Save / register the raw file
        saved = document_loader.save_document(
            file_path
        )

        document_id = saved["document_id"]

        # 2. Parse into pages of text
        parsed = document_parser.parse_document(
            saved["file_path"]
        )

        # 3. Chunk
        chunks = document_chunker.chunk_document(
            parsed_document=parsed,
            document_id=document_id
        )

        if not chunks:
            app_logger.warning(
                f"No chunks produced for "
                f"{saved['filename']}"
            )
            return {
                "document_id": document_id,
                "filename": saved["filename"],
                "chunks": 0,
                "indexed": False
            }

        # 4. Enrich metadata
        enriched = metadata_builder.enrich_chunks(
            chunks=chunks,
            filename=saved["filename"],
            document_version=document_version
        )

        # 5. Embed
        embedded = (
            embedding_generator
            .generate_embeddings(enriched)
        )

        # 6. Index into Qdrant
        qdrant_indexer.index_chunks(embedded)

        app_logger.success(
            f"Ingested {saved['filename']} "
            f"-> {len(embedded)} chunks "
            f"(doc_id={document_id})"
        )

        return {
            "document_id": document_id,
            "filename": saved["filename"],
            "chunks": len(embedded),
            "indexed": True
        }

    def ingest_directory(
        self,
        directory: str,
        document_version: str = "1.0"
    ) -> List[Dict]:
        """
        Ingest every supported file under a
        directory (recursively).
        """

        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        files = sorted(
            p for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )

        if not files:
            app_logger.warning(
                f"No supported files under "
                f"{directory}"
            )
            return []

        app_logger.info(
            f"Found {len(files)} files to "
            f"ingest under {directory}"
        )

        results = []

        for file_path in files:
            try:
                results.append(
                    self.ingest_file(
                        str(file_path),
                        document_version
                    )
                )
            except Exception as exc:
                app_logger.error(
                    f"Failed to ingest "
                    f"{file_path}: {exc!r}"
                )
                results.append({
                    "document_id": None,
                    "filename": file_path.name,
                    "chunks": 0,
                    "indexed": False,
                    "error": repr(exc)
                })

        total_chunks = sum(
            r["chunks"] for r in results
        )

        app_logger.success(
            f"Ingestion complete: "
            f"{len(results)} files, "
            f"{total_chunks} chunks"
        )

        return results


ingestion_pipeline = IngestionPipeline()


if __name__ == "__main__":

    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/primary"
    )

    path = Path(target)

    if path.is_file():
        summary = ingestion_pipeline.ingest_file(
            target
        )
        print(summary)
    else:
        summaries = (
            ingestion_pipeline
            .ingest_directory(target)
        )
        for s in summaries:
            print(s)
