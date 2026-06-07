"""Unit tests for the sentence-aware chunker."""

from src.ingestion.chunker import document_chunker


def _long_doc():
    text = " ".join(
        f"Sentence number {i} discusses an aspect of "
        f"global geopolitics and world history."
        for i in range(400)
    )
    return {"pages": [{"page_number": 1, "text": text}]}


def test_respects_chunk_size():
    doc = _long_doc()
    chunks = document_chunker.chunk_document(doc, "doc_test")

    assert len(chunks) > 1

    # Each chunk stays within (about) the configured size.
    for c in chunks:
        assert len(c["text"]) <= document_chunker.chunk_size + 50


def test_deterministic_chunk_ids():
    doc = _long_doc()
    a = document_chunker.chunk_document(doc, "doc_test")
    b = document_chunker.chunk_document(doc, "doc_test")

    assert [c["chunk_id"] for c in a] == [
        c["chunk_id"] for c in b
    ]


def test_chunk_ids_depend_on_document_id():
    doc = _long_doc()
    a = document_chunker.chunk_document(doc, "doc_one")
    b = document_chunker.chunk_document(doc, "doc_two")

    assert a[0]["chunk_id"] != b[0]["chunk_id"]


def test_empty_page_produces_no_chunks():
    doc = {"pages": [{"page_number": 1, "text": "   "}]}
    assert document_chunker.chunk_document(doc, "d") == []
