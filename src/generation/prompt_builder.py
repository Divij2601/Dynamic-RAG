from typing import List, Optional

from src.graph.state import (
    EvidenceItem
)

from src.observability.logger import (
    app_logger
)


class PromptBuilder:
    """
    Grounded prompt builder
    with conversational memory
    """

    SYSTEM_PROMPT = """
You are Dynamic-RAG, a grounded retrieval-augmented AI assistant.

You MUST follow these rules strictly:

1. ONLY answer using provided evidence.

2. Use memory ONLY for conversational continuity.

3. NEVER invent facts.

4. If evidence is insufficient,
say:
"I could not find sufficient evidence to answer this."

5. Be concise and technically accurate.

6. Never override evidence
with memory.

7. Prioritize factual accuracy
over completeness.
"""

    def build_prompt(
        self,
        query: str,

        evidence_items:
        List[EvidenceItem],

        memory_context:
        Optional[str] = ""
    ) -> str:
        """
        Build grounded prompt
        """

        evidence_text = (
            self._format_evidence(
                evidence_items
            )
        )

        memory_section = ""

        if memory_context:

            memory_section = f"""
-------------------
CONVERSATION MEMORY
-------------------

{memory_context}
"""

        prompt = f"""
{self.SYSTEM_PROMPT}

{memory_section}

-------------------
USER QUESTION
-------------------

{query}

-------------------
EVIDENCE
-------------------

{evidence_text}

-------------------
TASK
-------------------

Answer the question ONLY
using the evidence.

Use memory ONLY to
understand conversational
references such as:
"it", "that", "previously discussed".

Never invent facts not
present in evidence.
"""

        app_logger.success(
            "Grounded memory-aware "
            "prompt built"
        )

        return prompt

    def _format_evidence(
        self,
        evidence_items:
        List[EvidenceItem]
    ) -> str:
        """
        Format evidence
        """

        formatted = []

        for i, evidence in enumerate(
            evidence_items
        ):

            entry = f"""
[EVIDENCE {i+1}]
Page: {evidence.page}

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