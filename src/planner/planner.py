import json
import re
from typing import Dict, Any

from groq import Groq

from src.config import settings

from src.graph.state import (
    PlannerOutput
)

from src.planner.heuristics import (
    QueryHeuristics
)

from src.observability.logger import (
    app_logger
)


VALID_ROUTES = {
    "internal_rag",
    "web_research",
    "hybrid",
    "memory",
    "direct_generation"
}


class QueryPlanner:
    """
    LLM-based adaptive query planner.

    Uses a fast model to classify each query into
    one of the supported routes, informed by a
    description of what the knowledge base contains.
    Falls back to keyword heuristics if the model
    call or parsing fails, so planning never blocks
    the pipeline.
    """

    _client = None

    def __init__(self):

        self.model = settings.FAST_MODEL

    def _get_client(self):

        if self._client is None:

            self._client = Groq(
                api_key=settings.GROQ_API_KEY
            )

            app_logger.success(
                "Planner LLM client initialized"
            )

        return self._client

    def plan(
        self,
        query: str
    ) -> PlannerOutput:
        """
        Plan execution route for a query.
        """

        try:
            return self._llm_plan(query)

        except Exception as exc:

            app_logger.error(
                f"LLM planner failed, falling back "
                f"to heuristics: {exc!r}"
            )

            route = QueryHeuristics.classify(query)

            return self._build_output(
                route=route,
                intent="heuristic_fallback",
                complexity="medium",
                confidence=0.5,
                needs_decomposition=False,
                subqueries=[],
                budget="medium"
            )

    # ------------------------------------------------

    def _llm_plan(
        self,
        query: str
    ) -> PlannerOutput:

        prompt = self._build_prompt(query)

        response = (
            self._get_client()
            .chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=400
            )
        )

        raw = response.choices[0].message.content

        data = self._parse(raw)

        route = data.get("route")

        if route not in VALID_ROUTES:
            # Model returned an unknown route; fall
            # back to heuristics for the route only.
            route = QueryHeuristics.classify(query)

        output = self._build_output(
            route=route,
            intent=str(
                data.get("intent", "") or ""
            ) or None,
            complexity=str(
                data.get("complexity", "") or ""
            ) or None,
            confidence=self._coerce_confidence(
                data.get("confidence")
            ),
            needs_decomposition=bool(
                data.get("needs_decomposition", False)
            ),
            subqueries=self._coerce_list(
                data.get("subqueries")
            ),
            budget=str(
                data.get("budget", "") or ""
            ) or None
        )

        app_logger.success(
            f"Planner (LLM) selected route: "
            f"{output.route}"
        )

        return output

    def _build_output(
        self,
        route: str,
        intent=None,
        complexity=None,
        confidence: float = 0.7,
        needs_decomposition: bool = False,
        subqueries=None,
        budget=None
    ) -> PlannerOutput:
        """
        Construct PlannerOutput, deriving the
        needs_* flags deterministically from the
        chosen route (more reliable than trusting
        the model's booleans).
        """

        return PlannerOutput(
            intent=intent,
            complexity=complexity,
            route=route,
            confidence=confidence,
            needs_retrieval=route in (
                "internal_rag", "hybrid"
            ),
            needs_memory=route == "memory",
            needs_web=route in (
                "web_research", "hybrid"
            ),
            needs_decomposition=needs_decomposition,
            subqueries=subqueries or [],
            budget=budget
        )

    def _build_prompt(
        self,
        query: str
    ) -> str:

        return f"""You are the query planner for Dynamic-RAG, an \
adaptive retrieval-augmented system. Choose the single best route \
for the user's query.

KNOWLEDGE BASE CONTENTS:
{settings.KNOWLEDGE_BASE_DESCRIPTION}

ROUTES (choose exactly one):
- "internal_rag": answerable from the knowledge base above (facts, \
history, concepts, definitions within that domain), even if the query \
mentions recent years.
- "web_research": needs current/real-time/breaking information (live \
prices, today's news, very recent events) OR is a factual question \
clearly outside the knowledge base domain that the web can answer.
- "hybrid": answering well needs BOTH the knowledge base AND fresh \
web information.
- "memory": refers to the earlier conversation (e.g. "what did we \
discuss", "you said", "earlier", "previously", or pronouns referring \
to prior turns).
- "direct_generation": a general language task needing no retrieval \
(rewrite, summarize provided text, translate, explain a general \
concept, casual chat).

Respond with ONLY a JSON object, no markdown:
{{
  "route": "<one of: internal_rag, web_research, hybrid, memory, direct_generation>",
  "intent": "<short label>",
  "complexity": "<low|medium|high>",
  "confidence": <number 0..1>,
  "needs_decomposition": <true|false>,
  "subqueries": [],
  "budget": "<low|medium|high>"
}}

USER QUERY:
{query}"""

    # ------------------------------------------------

    def _parse(
        self,
        raw: str
    ) -> Dict[str, Any]:

        text = (raw or "").strip()

        text = re.sub(
            r"^```(?:json)?",
            "",
            text
        ).strip()

        text = re.sub(r"```$", "", text).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return json.loads(text)

    def _coerce_confidence(
        self,
        value
    ) -> float:

        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.7

    def _coerce_list(
        self,
        value
    ):

        if isinstance(value, list):
            return [str(v) for v in value]
        return []


query_planner = QueryPlanner()
