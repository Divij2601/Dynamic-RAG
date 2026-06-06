from src.graph.state import (
    QueryState
)

from src.retrieval.hybrid import (
    hybrid_retriever
)

from src.retrieval.reranker import (
    reranker
)

from src.retrieval.evidence import (
    evidence_builder
)

from src.observability.logger import (
    app_logger
)


class QueryRouter:
    """
    Route execution engine
    """

    def execute(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Execute selected route
        """

        route = (
            state
            .planner_output
            .route
        )

        app_logger.info(
            f"Executing route: "
            f"{route}"
        )

        if route == (
            "internal_rag"
        ):
            return (
                self
                ._internal_rag_route(
                    state
                )
            )

        elif route == (
            "memory"
        ):
            return (
                self
                ._memory_route(
                    state
                )
            )

        elif route == (
            "web_research"
        ):
            return (
                self
                ._web_route(
                    state
                )
            )

        elif route == (
            "hybrid"
        ):
            return (
                self
                ._hybrid_route(
                    state
                )
            )

        elif route == (
            "direct_generation"
        ):
            return (
                self
                ._direct_route(
                    state
                )
            )

        return state

    def _internal_rag_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Internal retrieval route
        """

        retrieval = (
            hybrid_retriever
            .retrieve(
                state.query_text
            )
        )

        reranked = (
            reranker
            .rerank(
                query=(
                    state.query_text
                ),

                retrieved_chunks=(
                    retrieval[
                        "results"
                    ]
                )
            )
        )

        evidence = (
            evidence_builder
            .build(
                reranked[
                    "results"
                ]
            )
        )

        state.internal_evidence = (
            evidence
        )

        state.selected_route = (
            "internal_rag"
        )

        return state

    def _memory_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Memory route
        """

        app_logger.info(
            "Memory route "
            "placeholder"
        )

        state.selected_route = (
            "memory"
        )

        return state

    def _web_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Web route
        """

        app_logger.info(
            "Web route "
            "placeholder"
        )

        state.selected_route = (
            "web_research"
        )

        return state

    def _hybrid_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Hybrid route
        """

        app_logger.info(
            "Hybrid route "
            "placeholder"
        )

        state.selected_route = (
            "hybrid"
        )

        return state

    def _direct_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Direct generation route
        """

        state.selected_route = (
            "direct_generation"
        )

        return state


query_router = (
    QueryRouter()
)