import json
import re
from typing import Dict, Any, List, Optional

from src.config import settings
from src.models.groq_provider import groq_provider
from src.knowledge.corpus_description import (
    corpus_description_builder
)

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

# Max turns of history shown to the planner.
# Enough for context without bloating the prompt.
MAX_HISTORY_IN_PROMPT = 5


class QueryPlanner:
    """
    LLM-based adaptive query planner.

    Accepts optional chat_history so it can detect
    follow-up queries ("same as before", "tell me
    more", "search the internet for the same") and
    rewrite them into standalone queries before
    routing to retrieval or web search.
    """

    def __init__(self):

        self.model = settings.FAST_MODEL

    def plan(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> PlannerOutput:
        """
        Plan execution route for a query.
        chat_history: list of {query, answer} dicts,
        most recent last.
        """

        try:
            return self._llm_plan(
                query,
                chat_history or []
            )

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
                budget="medium",
                rewritten_query=None,
                is_followup=False
            )

    # ------------------------------------------------

    def _llm_plan(
        self,
        query: str,
        chat_history: List[Dict]
    ) -> PlannerOutput:

        prompt = self._build_prompt(query, chat_history)

        raw = groq_provider.complete(
            prompt=prompt,
            model=self.model,
            temperature=0.0,
            max_tokens=500
        )

        data = self._parse(raw)

        route = data.get("route")

        if route not in VALID_ROUTES:
            route = QueryHeuristics.classify(query)

        rewritten = (
            str(data.get("rewritten_query") or "")
            .strip()
        ) or None

        is_followup = bool(data.get("is_followup", False))

        # If the model signalled a followup but gave no
        # rewrite, use the original query.
        if is_followup and not rewritten:
            rewritten = None

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
            ) or None,
            rewritten_query=rewritten,
            is_followup=is_followup
        )

        app_logger.success(
            f"Planner selected route: {output.route}"
            + (
                f" (followup, rewrite: {rewritten!r})"
                if is_followup else ""
            )
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
        budget=None,
        rewritten_query=None,
        is_followup: bool = False
    ) -> PlannerOutput:

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
            budget=budget,
            rewritten_query=rewritten_query,
            is_followup=is_followup
        )

    def _format_history(
        self,
        chat_history: List[Dict]
    ) -> str:
        """
        Format the last N turns for the planner prompt.
        """

        if not chat_history:
            return "(No prior conversation)"

        recent = chat_history[-MAX_HISTORY_IN_PROMPT:]

        lines = []
        for i, turn in enumerate(recent, start=1):
            q = turn.get("query", "").strip()
            a = (turn.get("answer", "") or "").strip()
            # Truncate long answers in the planner prompt
            if len(a) > 200:
                a = a[:200] + "…"
            lines.append(
                f"[Turn {i}]\n"
                f"User: {q}\n"
                f"Assistant: {a}"
            )

        return "\n\n".join(lines)

    def _build_prompt(
        self,
        query: str,
        chat_history: List[Dict]
    ) -> str:

        kb_description = (
            corpus_description_builder.get_description()
        )

        history_section = self._format_history(
            chat_history
        )

        has_history = bool(chat_history)

        followup_instruction = (
            """
FOLLOW-UP DETECTION:
If the current query references the prior conversation \
(e.g. "same as before", "search the web for that", \
"tell me more", "what about X", implicit pronouns \
like "it"/"the same"/"that topic") then:
- Set is_followup=true
- Set rewritten_query to a COMPLETE standalone version \
of what the user actually wants — replacing all \
references with their real content from the history.

Example:
  History: User asked about "India's geopolitical stance"
  Current: "search the internet for the same"
  rewritten_query: "India geopolitical stance in world current 2025"
  route: "web_research"

If the query is independent, set is_followup=false and \
rewritten_query=null.
"""
            if has_history
            else ""
        )

        return f"""You are the query planner for Dynamic-RAG, \
an adaptive retrieval-augmented system.

KNOWLEDGE BASE CONTENTS:
{kb_description}

RECENT CONVERSATION HISTORY:
{history_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING DECISION — apply these rules IN ORDER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — MEMORY
If the query explicitly references the current \
conversation ("what did you say", "earlier you \
mentioned", "what did we discuss", "you said") \
→ route = "memory"

STEP 2 — DIRECT GENERATION
If the task is a pure LANGUAGE OPERATION that \
requires NO external facts — rewrite, translate, \
summarise a provided text, fix grammar, explain \
a general abstract concept, perform arithmetic, \
write code, or casual chitchat:
→ route = "direct_generation"
IMPORTANT: A factual question about the real world \
is NEVER direct_generation, even if it sounds simple.

STEP 3 — INTERNAL RAG
If the query is a factual, analytical, or conceptual \
question whose answer is COVERED IN THE KNOWLEDGE \
BASE above (geopolitics, country profiles, \
international organizations, treaties, major wars, \
Cold War, decolonization, BRICS, NATO, UN, NPT, \
Paris Agreement, sovereignty, soft power, deterrence, \
global affairs up to 2026):
→ route = "internal_rag"

STEP 4 — WEB RESEARCH
If the query asks for information that is LIVE, \
REAL-TIME, or OUTSIDE THE CORPUS — use this route \
even when the answer sounds simple:

  • Cryptocurrency / stock / commodity prices \
("price of Bitcoin", "gold price today")
  • Sports results, scores, standings, winners \
("who won the World Cup", "current league table")
  • Current population, census, demographic stats \
("population of Tokyo", "GDP of Germany 2025")
  • Company or organisation leadership that \
changes over time ("CEO of Tesla", "chairman of \
Apple", "president of FIFA")
  • Medical / scientific facts outside the corpus \
("side effects of ibuprofen", "COVID vaccine schedule")
  • Current weather, traffic, flight status
  • Any event, result, or data NOT in the \
knowledge base description above

→ route = "web_research"

STEP 5 — HYBRID
If the query genuinely needs BOTH corpus knowledge \
AND current web data (e.g. "How did Russia's 2024 \
domestic politics affect its BRICS strategy?"):
→ route = "hybrid"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES:
• "direct_generation" is ONLY for language tasks, \
  NEVER for real-world factual questions.
• If a factual question is NOT covered by the \
  knowledge base, use "web_research" — not \
  "direct_generation" or "internal_rag".
• When unsure between internal_rag and web_research, \
  ask: "Is this in the knowledge base?" If no → web.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{followup_instruction}

Respond with ONLY a JSON object, no markdown:
{{
  "route": "<one of the five routes>",
  "intent": "<short label>",
  "complexity": "<low|medium|high>",
  "confidence": <0..1>,
  "is_followup": <true|false>,
  "rewritten_query": "<standalone rewritten query or null>",
  "needs_decomposition": <true if multi-part question needing multiple distinct facts>,
  "subqueries": ["<sub-question 1>", ...],
  "budget": "<low|medium|high>"
}}

CURRENT USER QUERY:
{query}"""

    # ------------------------------------------------

    def _parse(
        self,
        raw: str
    ) -> Dict[str, Any]:

        text = (raw or "").strip()

        text = re.sub(
            r"^```(?:json)?", "", text
        ).strip()

        text = re.sub(r"```$", "", text).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return json.loads(text)

    def _coerce_confidence(self, value) -> float:

        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.7

    def _coerce_list(self, value):

        if isinstance(value, list):
            return [str(v) for v in value]
        return []


query_planner = QueryPlanner()
