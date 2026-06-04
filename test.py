from src.ingestion.loader import (
    document_loader
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


document = (
    document_loader
    .save_document(
        "Divij_Dudeja_resume_26.pdf"
    )
)

parsed = (
    document_parser
    .parse_document(
        document["file_path"]
    )
)

chunks = (
    document_chunker
    .chunk_document(
        parsed_document=parsed,
        document_id=(
            document[
                "document_id"
            ]
        )
    )
)

metadata_chunks = (
    metadata_builder
    .enrich_chunks(
        chunks=chunks,
        filename=(
            document[
                "filename"
            ]
        )
    )
)

embedded_chunks = (
    embedding_generator
    .generate_embeddings(
        metadata_chunks
    )
)

qdrant_indexer.index_chunks(
    embedded_chunks
)

print(
    "Pipeline complete"
)