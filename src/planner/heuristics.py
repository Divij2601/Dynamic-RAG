import re


class QueryHeuristics:
    """
    Rule-based route classifier
    """

    MEMORY_PATTERNS = [
        r"\bprevious\b",
        r"\bearlier\b",
        r"\bcontinue\b",
        r"\bwe discussed\b",
        r"\bsame assumptions\b"
    ]

    WEB_PATTERNS = [
        r"\blatest\b",
        r"\brecent\b",
        r"\bnews\b",
        r"\btoday\b",
        r"\b2026\b",
        r"\bcurrent\b"
    ]

    INTERNAL_PATTERNS = [
        r"\bdocument\b",
        r"\bmanual\b",
        r"\bpdf\b",
        r"\bconfig\b",
        r"\bsetting\b",
        r"\brelay\b",
        r"\bsection\b"
    ]

    DIRECT_PATTERNS = [
        r"\brewrite\b",
        r"\bsummarize\b",
        r"\bexplain\b",
        r"\btranslate\b",
        r"\bimprove\b"
    ]

    @classmethod
    def classify(
        cls,
        query: str
    ):

        query_lower = (
            query.lower()
        )

        memory = any(
            re.search(
                p,
                query_lower
            )
            for p in (
                cls
                .MEMORY_PATTERNS
            )
        )

        web = any(
            re.search(
                p,
                query_lower
            )
            for p in (
                cls
                .WEB_PATTERNS
            )
        )

        internal = any(
            re.search(
                p,
                query_lower
            )
            for p in (
                cls
                .INTERNAL_PATTERNS
            )
        )

        direct = any(
            re.search(
                p,
                query_lower
            )
            for p in (
                cls
                .DIRECT_PATTERNS
            )
        )

        if (
            internal
            and web
        ):
            return "hybrid"

        if internal:
            return "internal_rag"

        if memory:
            return "memory"

        if web:
            return "web_research"

        if direct:
            return "direct_generation"

        return "internal_rag"