"""
Dynamic corpus description builder.

Reads the actual documents indexed in Qdrant and
builds a concise natural-language description of what
the knowledge base contains. This description is fed
to the LLM planner on every query so it always knows
exactly what is (and is not) in the corpus — with
zero manual updates required.

The description is cached in memory and invalidated
automatically whenever a new document is ingested
(called from pipeline.py alongside BM25 cache
invalidation).
"""

from typing import Optional

from src.config import settings
from src.observability.logger import app_logger


# Constant part appended to every description —
# tells the planner what the corpus never contains.
_ALWAYS_EXCLUDE = (
    "The knowledge base does NOT contain: "
    "live/real-time data, current stock or cryptocurrency prices, "
    "live sports results or scores, current population statistics, "
    "weather forecasts, private or internal company documents, "
    "classified or secret information, fictional entities, "
    "or events that occurred after the document upload dates."
)


class CorpusDescriptionBuilder:
    """
    Builds and caches a natural-language description
    of the corpus for the LLM planner.
    """

    _cached_description: Optional[str] = None

    def get_description(self) -> str:
        """
        Return the cached description, building it
        from Qdrant if not yet cached or invalidated.
        """

        if self._cached_description is not None:
            return self._cached_description

        self._cached_description = self._build()
        return self._cached_description

    def invalidate(self):
        """
        Force a rebuild on the next call.
        Called automatically after every ingestion.
        """

        self._cached_description = None

        app_logger.info(
            "Corpus description cache invalidated — "
            "will rebuild on next planner call"
        )

    def _build(self) -> str:
        """
        Query Qdrant for all unique documents and
        build a concise description.
        """

        try:
            return self._build_from_qdrant()

        except Exception as exc:

            app_logger.error(
                f"Corpus description build failed, "
                f"falling back to config: {exc!r}"
            )

            # Graceful fallback: use the config string
            # so the planner still works even if Qdrant
            # is temporarily unavailable.
            return settings.KNOWLEDGE_BASE_DESCRIPTION

    def _build_from_qdrant(self) -> str:

        from src.database.qdrant_client import (
            qdrant_client
        )

        client = qdrant_client.get_client()
        collection = settings.QDRANT_COLLECTION_NAME

        # Scroll all points, keeping only the first
        # chunk (position=0) per document so we get
        # one representative text snippet each.
        doc_info: dict = {}
        offset = None

        while True:

            result = client.scroll(
                collection_name=collection,
                limit=500,
                with_payload=["document_id",
                               "metadata",
                               "text"],
                offset=offset
            )

            batch, next_offset = result

            for point in batch:
                pl = point.payload or {}
                doc_id = pl.get("document_id")
                meta = pl.get("metadata") or {}
                chunk_index = meta.get("chunk_index", 999)

                if doc_id and (
                    doc_id not in doc_info
                    or chunk_index
                    < doc_info[doc_id]["chunk_index"]
                ):
                    doc_info[doc_id] = {
                        "filename": meta.get(
                            "filename", doc_id
                        ),
                        "chunk_index": chunk_index,
                        "snippet": (
                            pl.get("text", "")[:180]
                            .replace("\n", " ")
                            .strip()
                        )
                    }

            if not next_offset:
                break
            offset = next_offset

        if not doc_info:
            return settings.KNOWLEDGE_BASE_DESCRIPTION

        doc_count = len(doc_info)

        # Build the document list, sorted by filename.
        lines = []

        for info in sorted(
            doc_info.values(),
            key=lambda x: x["filename"].lower()
        ):
            fname = info["filename"]
            snippet = info["snippet"]

            # Truncate long snippets cleanly at a word.
            if len(snippet) > 160:
                snippet = snippet[:160].rsplit(" ", 1)[0]
                snippet += "..."

            lines.append(f"- {fname}: {snippet}")

        doc_list = "\n".join(lines)

        description = (
            f"The knowledge base currently contains "
            f"{doc_count} document(s).\n\n"
            f"Documents and their content:\n"
            f"{doc_list}\n\n"
            f"{_ALWAYS_EXCLUDE}"
        )

        app_logger.success(
            f"Corpus description built: "
            f"{doc_count} documents"
        )

        return description


corpus_description_builder = CorpusDescriptionBuilder()
