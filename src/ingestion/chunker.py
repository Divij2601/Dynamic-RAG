import re
from uuid import uuid4
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize

from src.config import settings
from src.observability.logger import (
    app_logger
)


class DocumentChunker:
    """
    Semantic document chunker
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
        Chunk full document
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
        Chunk one page
        """

        chunks = []

        paragraphs = (
            self._split_paragraphs(
                text
            )
        )

        current_chunk = ""

        for paragraph in (
            paragraphs
        ):

            paragraph = (
                paragraph.strip()
            )

            if not paragraph:
                continue

            proposed = (
                current_chunk
                + "\n\n"
                + paragraph
            )

            if len(proposed) < (
                self.chunk_size
            ):

                current_chunk = (
                    proposed
                )

            else:

                if current_chunk:

                    chunks.append(
                        self._build_chunk(
                            text=current_chunk,
                            page_number=(
                                page_number
                            ),
                            document_id=(
                                document_id
                            )
                        )
                    )

                current_chunk = (
                    self._apply_overlap(
                        current_chunk
                    )
                    + paragraph
                )

        if current_chunk:

            chunks.append(
                self._build_chunk(
                    text=current_chunk,
                    page_number=(
                        page_number
                    ),
                    document_id=(
                        document_id
                    )
                )
            )

        return chunks

    def _split_paragraphs(
        self,
        text: str
    ) -> List[str]:
        """
        Semantic paragraph split
        """

        paragraphs = re.split(
            r"\n\s*\n",
            text
        )

        return [
            p.strip()
            for p in paragraphs
            if p.strip()
        ]

    def _apply_overlap(
        self,
        text: str
    ) -> str:
        """
        Sentence overlap
        """

        sentences = (
            sent_tokenize(text)
        )

        overlap_text = ""

        while (
            len(overlap_text)
            < self.chunk_overlap
            and sentences
        ):

            overlap_text = (
                sentences[-1]
                + " "
                + overlap_text
            )

            sentences.pop()

        return overlap_text

    def _build_chunk(
        self,
        text: str,
        page_number: int,
        document_id: str
    ) -> Dict:
        """
        Build chunk object
        """

        return {
            "chunk_id":
            f"chunk_{uuid4().hex[:12]}",

            "document_id":
            document_id,

            "page":
            page_number,

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