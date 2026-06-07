"""Unit tests for chunk metadata enrichment."""

from src.ingestion.metadata import metadata_builder


def test_enrich_adds_fields_and_index():
    chunks = [
        {"text": "hello", "metadata": {}},
        {"text": "world", "metadata": {}},
    ]

    out = metadata_builder.enrich_chunks(
        chunks, "file.txt", document_version="1.0"
    )

    assert out[0]["metadata"]["filename"] == "file.txt"
    assert out[0]["metadata"]["document_version"] == "1.0"
    assert out[0]["metadata"]["chunk_index"] == 0
    assert out[1]["metadata"]["chunk_index"] == 1


def test_content_hash_is_deterministic_sha256():
    chunks = [{"text": "hello", "metadata": {}}]
    out = metadata_builder.enrich_chunks(chunks, "f.txt")

    h = out[0]["metadata"]["content_hash"]
    assert len(h) == 64
    assert h == metadata_builder._generate_hash("hello")
