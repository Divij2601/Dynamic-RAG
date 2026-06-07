from typing import List, Optional

from src.graph.state import (
    EvidenceItem
)

from src.observability.logger import (
    app_logger
)


class PromptBuilder:
    """
    Unified grounded prompt builder
    supporting:

    - Internal RAG
    - Web Research
    - Memory
    - Hybrid routes
    """

    SYSTEM_PROMPT = """
You are Dynamic-RAG, a grounded retrieval-augmented AI assistant.

STRICT RULES:

1. ONLY answer using provided evidence.

2. Memory is ONLY for conversational continuity.

3. NEVER invent facts.

4. NEVER use outside knowledge.

5. If evidence is insufficient,
respond exactly:

"I could not find sufficient evidence to answer this."

6. Prefer factual accuracy
over completeness.

7. If multiple evidence sources
agree, synthesize carefully.

8. If evidence conflicts,
explicitly mention uncertainty.

9. Do NOT cite memory as evidence.
"""

    def build_prompt(
        self,
        query: str,

        internal_evidence:
        Optional[
            List[EvidenceItem]
        ] = None,

        web_evidence:
        Optional[
            List[EvidenceItem]
        ] = None,

        memory_context:
        Optional[str] = ""
    ) -> str:
        """
        Build unified prompt
        """

        sections = []

        # ------------------
        # Memory
        # ------------------

        if memory_context:

            sections.append(
                f"""
-------------------
CONVERSATION MEMORY
-------------------

{memory_context}
"""
            )

        # ------------------
        # Internal Evidence
        # ------------------

        internal_count = 0

        if internal_evidence:

            internal_text = (
                self._format_evidence(
                    internal_evidence,
                    source_name="INTERNAL DOCUMENTS",
                    index_offset=0
                )
            )

            sections.append(internal_text)
            internal_count = len(internal_evidence)

        # ------------------
        # Web Evidence
        # ------------------

        if web_evidence:

            web_text = (
                self._format_evidence(
                    web_evidence,
                    source_name="WEB RESEARCH",
                    # Offset so source numbers continue
                    # from where internal left off.
                    index_offset=internal_count
                )
            )

            sections.append(web_text)

        combined_context = (
            "\n\n".join(
                sections
            )
        )

        prompt = f"""
{self.SYSTEM_PROMPT}

-------------------
USER QUESTION
-------------------

{query}

{combined_context}

-------------------
TASK
-------------------

Answer ONLY using the evidence sources above.

When citing, use the source label exactly as shown,
e.g. "[Source 1 — india_profile.txt, page 2]".
You may shorten it to "[Source 1]" inline.

Conversation history is ONLY for resolving
references like "it", "that", or "same as before".
Never treat history as factual evidence.

If the evidence is insufficient, respond:
"I could not find sufficient evidence to answer this."
"""

        app_logger.success(
            "Unified grounded "
            "prompt built"
        )

        return prompt

    def _format_evidence(
        self,
        evidence_items: List[EvidenceItem],
        source_name: str,
        index_offset: int = 0
    ) -> str:
        """
        Format evidence block with self-describing
        citations (filename + page) so the LLM
        naturally cites by source name rather than
        a generic number — making answers readable
        without needing to cross-reference a panel.
        """

        formatted = [
            f"\n-------------------\n"
            f"{source_name}\n"
            f"-------------------"
        ]

        for i, evidence in enumerate(evidence_items):

            n = i + 1 + index_offset

            stype = evidence.source_type

            if stype == "web":
                meta = evidence.metadata or {}
                title = meta.get("title", "")
                url = meta.get("url", "")
                citation = (
                    f"[Source {n} — {title}]"
                    if title
                    else f"[Source {n} — {url}]"
                )
            else:
                meta = evidence.metadata or {}
                filename = meta.get("filename", "")
                page = evidence.page
                if filename and page is not None:
                    citation = (
                        f"[Source {n} — "
                        f"{filename}, page {page}]"
                    )
                elif filename:
                    citation = (
                        f"[Source {n} — {filename}]"
                    )
                else:
                    citation = f"[Source {n}]"

            entry = (
                f"\n{citation}\n"
                f"{evidence.text}"
            )

            formatted.append(entry)

        return "\n".join(formatted)

    def _count_evidence(
        self,
        evidence_items: Optional[List[EvidenceItem]]
    ) -> int:
        return len(evidence_items) if evidence_items else 0


prompt_builder = (
    PromptBuilder()
)