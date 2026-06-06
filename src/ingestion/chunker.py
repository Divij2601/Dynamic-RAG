import re
import hashlib
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize

from src.config import settings
from src.observability.logger import (
    app_logger
)


def _ensure_punkt():
    """
    Make sure the NLTK sentence
    tokenizer data is available.
    """

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
            )
        except LookupError:
            try:
                nltk.download(
                    resource,
                    quiet=True
                )
            except Exception:
                # punkt_tab does not exist on
                # older nltk; punkt is enough
                pass


_ensure_punkt()


class DocumentChunker:
    """
    Sentence-aware document chunker.

    Packs sentences greedily up to
    CHUNK_SIZE characters, carries a
    CHUNK_OVERLAP-sized tail of trailing
    sentences into the next chunk, and
    assigns deterministic chunk IDs so the
    same document always yields the same
    chunk IDs (stable gold-chunk mappings
    for evaluation).
    """

    def __init__(self):

        self.chunk_size = (
            settings.CHUNK_SIZE
        )

        self.chunk_overlap = (
            settings.CHUNK_OVERLAP
        )

    def chunk_document(
        self,
        parsed_document: Dict,
        document_id: str
    ) -> List[Dict]:
        """
        Chunk a full parsed document
        page by page.
        """

        chunks = []

        for page in (
            parsed_document["pages"]
        ):

            page_chunks = (
                self._chunk_page(
                    text=page["text"],
                    page_number=(
                        page["page_number"]
                    ),
                    document_id=(
                        document_id
                    )
                )
            )

            chunks.extend(
                page_chunks
            )

        app_logger.success(
            f"Created "
            f"{len(chunks)} chunks"
        )

        return chunks

    def _chunk_page(
        self,
        text: str,
        page_number: int,
        document_id: str
    ) -> List[Dict]:
        """
        Chunk one page into sentence-packed
        chunks that respect chunk_size.
        """

        sentences = (
            self._split_sentences(text)
        )

        chunks = []
        position = 0

        current: List[str] = []
        current_len = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # +1 accounts for the joining space
            addition = len(sentence) + 1

            # If adding this sentence would
            # overflow and we already have
            # content, flush the current chunk.
            if (
                current
                and current_len + addition
                > self.chunk_size
            ):

                chunk_text = (
                    " ".join(current).strip()
                )

                chunks.append(
                    self._build_chunk(
                        text=chunk_text,
                        page_number=(
                            page_number
                        ),
                        document_id=(
                            document_id
                        ),
                        position=position
                    )
                )

                position += 1

                # Seed the next chunk with a
                # trailing-sentence overlap.
                overlap = (
                    self._overlap_sentences(
                        current
                    )
                )

                current = list(overlap)
                current_len = sum(
                    len(s) + 1
                    for s in current
                )

            current.append(sentence)
            current_len += addition

        if current:

            chunk_text = (
                " ".join(current).strip()
            )

            if chunk_text:

                chunks.append(
                    self._build_chunk(
                        text=chunk_text,
                        page_number=(
                            page_number
                        ),
                        document_id=(
                            document_id
                        ),
                        position=position
                    )
                )

        return chunks

    def _split_sentences(
        self,
        text: str
    ) -> List[str]:
        """
        Split text into sentences, first
        normalizing whitespace so PDF line
        breaks do not fragment sentences.
        Falls back to newline splitting if
        the NLTK tokenizer is unavailable.
        """

        normalized = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if not normalized:
            return []

        try:
            return sent_tokenize(normalized)
        except Exception:
            # Defensive fallback: split on
            # sentence-ending punctuation.
            return [
                s.strip()
                for s in re.split(
                    r"(?<=[.!?])\s+",
                    normalized
                )
                if s.strip()
            ]

    def _overlap_sentences(
        self,
        sentences: List[str]
    ) -> List[str]:
        """
        Return the trailing sentences whose
        combined length is about
        chunk_overlap characters.
        """

        overlap: List[str] = []
        total = 0

        for sentence in reversed(sentences):

            if total >= self.chunk_overlap:
                break

            overlap.insert(0, sentence)
            total += len(sentence) + 1

        return overlap

    def _build_chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        position: int
    ) -> Dict:
        """
        Build a chunk object with a
        deterministic chunk_id derived from
        document_id, page, position and text.
        """

        fingerprint = (
            f"{document_id}|"
            f"{page_number}|"
            f"{position}|"
            f"{text}"
        )

        chunk_hash = hashlib.sha1(
            fingerprint.encode("utf-8")
        ).hexdigest()[:12]

        return {
            "chunk_id":
            f"chunk_{chunk_hash}",

            "document_id":
            document_id,

            "page":
            page_number,

            "position":
            position,

            "text":
            text.strip(),

            "metadata": {
                "length":
                len(text),

                "source_type":
                "document"
            }
        }


document_chunker = (
    DocumentChunker()
)
