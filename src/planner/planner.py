from src.graph.state import (
    PlannerOutput
)

from src.planner.heuristics import (
    QueryHeuristics
)

from src.observability.logger import (
    app_logger
)


class QueryPlanner:
    """
    Planner agent
    deciding execution route
    """

    def plan(
        self,
        query: str
    ) -> PlannerOutput:
        """
        Plan query execution
        """

        route = (
            QueryHeuristics
            .classify(query)
        )

        planner_output = (
            self._build_plan(
                route=route,
                query=query
            )
        )

        app_logger.success(
            f"Planner selected "
            f"route: "
            f"{planner_output.route}"
        )

        return planner_output

    def _build_plan(
        self,
        route: str,
        query: str
    ) -> PlannerOutput:
        """
        Build PlannerOutput
        """

        if route == "internal_rag":

            return PlannerOutput(
                intent="document_qa",

                complexity="medium",

                route="internal_rag",

                confidence=0.90,

                needs_retrieval=True,

                needs_memory=False,

                needs_web=False,

                needs_decomposition=False,

                budget="medium"
            )

        elif route == "memory":

            return PlannerOutput(
                intent="memory_lookup",

                complexity="low",

                route="memory",

                confidence=0.90,

                needs_retrieval=False,

                needs_memory=True,

                needs_web=False,

                budget="low"
            )

        elif route == (
            "web_research"
        ):

            return PlannerOutput(
                intent="external_research",

                complexity="medium",

                route="web_research",

                confidence=0.88,

                needs_retrieval=False,

                needs_memory=False,

                needs_web=True,

                budget="medium"
            )

        elif route == "hybrid":

            return PlannerOutput(
                intent="hybrid_reasoning",

                complexity="high",

                route="hybrid",

                confidence=0.85,

                needs_retrieval=True,

                needs_memory=True,

                needs_web=True,

                needs_decomposition=True,

                budget="high"
            )

        elif route == (
            "direct_generation"
        ):

            return PlannerOutput(
                intent="general_reasoning",

                complexity="low",

                route="direct_generation",

                confidence=0.92,

                needs_retrieval=False,

                needs_memory=False,

                needs_web=False,

                budget="low"
            )

        return PlannerOutput(
            route="internal_rag",

            confidence=0.50,

            needs_retrieval=True
        )


query_planner = (
    QueryPlanner()
)