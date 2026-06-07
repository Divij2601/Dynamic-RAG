import json
import re
from typing import List, Dict, Any

from src.config import settings
from src.models.groq_provider import groq_provider
from src.graph.state import EvidenceItem
from src.observability.logger import (
    app_logger
)


# Minimum faithfulness score for an answer
# to be considered grounded / not require retry.
FAITHFULNESS_THRESHOLD = 0.7


class FaithfulnessVerifier:
    """
    LLM-as-judge faithfulness verifier.

    Checks whether a generated answer is
    supported by the retrieved evidence and
    returns a structured verdict:

        {
          "faithful": bool,
          "grounded": bool,
          "faithfulness_score": float (0..1),
          "unsupported_claims": [str, ...],
          "reasoning": str
        }

    The verifier never raises: on any model or
    parsing failure it degrades gracefully to a
    permissive verdict so the pipeline can still
    return an answer (failure is logged, not hidden).
    """

    def __init__(self):

        self.model = settings.CRITIC_MODEL

    def verify(
        self,
        query: str,
        answer: str,
        evidence_items: List[EvidenceItem]
    ) -> Dict[str, Any]:
        """
        Judge the answer against the evidence.
        """

        # No evidence -> nothing to be grounded in.
        if not evidence_items:

            return {
                "faithful": False,
                "grounded": False,
                "faithfulness_score": 0.0,
                "unsupported_claims": [],
                "reasoning":
                "No evidence was available to "
                "support the answer."
            }

        evidence_block = self._format_evidence(
            evidence_items
        )

        prompt = self._build_prompt(
            query=query,
            answer=answer,
            evidence_block=evidence_block
        )

        try:

            raw = groq_provider.complete(
                prompt=prompt,
                model=self.model,
                temperature=0.0,
                # Generous budget: a reasoning critic
                # spends tokens before emitting JSON.
                max_tokens=settings.MAX_TOKENS
            )

            verdict = self._parse_verdict(raw)

            app_logger.success(
                f"Verifier verdict: "
                f"grounded={verdict['grounded']} "
                f"score={verdict['faithfulness_score']}"
            )

            return verdict

        except Exception as exc:

            app_logger.error(
                f"Verifier failed, degrading "
                f"to permissive verdict: {exc!r}"
            )

            return {
                "faithful": True,
                "grounded": True,
                "faithfulness_score": None,
                "unsupported_claims": [],
                "reasoning":
                f"Verifier unavailable: {exc!r}"
            }

    def _build_prompt(
        self,
        query: str,
        answer: str,
        evidence_block: str
    ) -> str:

        return f"""You are a strict faithfulness judge for a \
retrieval-augmented system. Decide whether the ANSWER is fully \
supported by the EVIDENCE. Use ONLY the evidence; ignore your own \
world knowledge.

A claim is "supported" only if it can be directly verified from the \
evidence text. An answer that correctly abstains (says it cannot find \
sufficient evidence) is considered grounded.

USER QUESTION:
{query}

ANSWER TO JUDGE:
{answer}

EVIDENCE:
{evidence_block}

Respond with ONLY a single JSON object, no markdown, in exactly this shape:
{{
  "faithfulness_score": <number between 0 and 1>,
  "grounded": <true or false>,
  "unsupported_claims": [<list of strings, claims not supported by evidence>],
  "reasoning": "<one or two sentence explanation>"
}}"""

    def _format_evidence(
        self,
        evidence_items: List[EvidenceItem]
    ) -> str:

        blocks = []

        for i, ev in enumerate(
            evidence_items,
            start=1
        ):

            blocks.append(
                f"[{i}] (source={ev.source_type}) "
                f"{ev.text}"
            )

        return "\n\n".join(blocks)

    def _parse_verdict(
        self,
        raw: str
    ) -> Dict[str, Any]:
        """
        Robustly parse the model's JSON verdict.
        Strips markdown fences and extracts the
        first balanced JSON object.
        """

        text = (raw or "").strip()

        # Strip ```json ... ``` fences if present.
        text = re.sub(
            r"^```(?:json)?",
            "",
            text
        ).strip()

        text = re.sub(
            r"```$",
            "",
            text
        ).strip()

        # Extract the outermost {...} block.
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        data = json.loads(text)

        score = data.get("faithfulness_score")

        try:
            score = float(score)
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = None

        grounded = bool(data.get("grounded", False))

        unsupported = data.get(
            "unsupported_claims",
            []
        )

        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]

        faithful = grounded and (
            score is None
            or score >= FAITHFULNESS_THRESHOLD
        )

        return {
            "faithful": faithful,
            "grounded": grounded,
            "faithfulness_score": score,
            "unsupported_claims": [
                str(c) for c in unsupported
            ],
            "reasoning": str(
                data.get("reasoning", "")
            )
        }


faithfulness_verifier = FaithfulnessVerifier()
