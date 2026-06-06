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

        if internal_evidence:

            internal_text = (
                self._format_evidence(
                    internal_evidence,
                    source_name=(
                        "INTERNAL DOCUMENTS"
                    )
                )
            )

            sections.append(
                internal_text
            )

        # ------------------
        # Web Evidence
        # ------------------

        if web_evidence:

            web_text = (
                self._format_evidence(
                    web_evidence,
                    source_name=(
                        "WEB RESEARCH"
                    )
                )
            )

            sections.append(
                web_text
            )

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

Answer the question
ONLY using evidence.

Memory exists ONLY to
resolve conversational
references like:

- "it"
- "that"
- "previous discussion"

Never use memory
as factual evidence.

If evidence is missing,
abstain.
"""

        app_logger.success(
            "Unified grounded "
            "prompt built"
        )

        return prompt

    def _format_evidence(
        self,
        evidence_items:
        List[EvidenceItem],

        source_name: str
    ) -> str:
        """
        Format evidence block
        """

        formatted = [
            f"""
-------------------
{source_name}
-------------------
"""
        ]

        for i, evidence in enumerate(
            evidence_items
        ):

            title = (
                evidence.metadata.get(
                    "title",
                    ""
                )
                if evidence.metadata
                else ""
            )

            entry = f"""
[EVIDENCE {i+1}]

Source Type:
{evidence.source_type}

Title:
{title}

Page:
{evidence.page}

Content:
{evidence.text}
"""

            formatted.append(
                entry
            )

        return "\n".join(
            formatted
        )


prompt_builder = (
    PromptBuilder()
)