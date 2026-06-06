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

from src.memory.retriever import (
    memory_retriever
)

from src.web.search import (
    web_search_agent
)

from src.web.evidence import (
    web_evidence_builder
)

from src.observability.logger import (
    app_logger
)


class QueryRouter:
    """
    Dynamic-RAG execution router
    """

    def execute(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Execute planned route
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
        Internal document RAG
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

        app_logger.success(
            "Internal RAG route complete"
        )

        return state

    def _memory_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Memory retrieval route
        """

        memory_context = (
            memory_retriever
            .retrieve_context(
                session_id=(
                    state.session_id
                ),

                query=(
                    state.query_text
                )
            )
        )

        state.memory_context = (
            memory_context
        )

        state.selected_route = (
            "memory"
        )

        app_logger.success(
            "Memory route complete"
        )

        return state

    def _web_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Web research route
        """

        web_results = (
            web_search_agent
            .search(
                state.query_text
            )
        )

        evidence = (
            web_evidence_builder
            .build(
                web_results
            )
        )

        state.web_evidence = (
            evidence
        )

        state.selected_route = (
            "web_research"
        )

        app_logger.success(
            "Web route complete"
        )

        return state

    def _hybrid_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Hybrid route:
        internal RAG
        + memory
        + web
        """

        # -------------------
        # Internal Retrieval
        # -------------------

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

        internal_evidence = (
            evidence_builder
            .build(
                reranked[
                    "results"
                ]
            )
        )

        state.internal_evidence = (
            internal_evidence
        )

        # -------------------
        # Memory Retrieval
        # -------------------

        memory_context = (
            memory_retriever
            .retrieve_context(
                session_id=(
                    state.session_id
                ),

                query=(
                    state.query_text
                )
            )
        )

        state.memory_context = (
            memory_context
        )

        # -------------------
        # Web Research
        # -------------------

        web_results = (
            web_search_agent
            .search(
                state.query_text
            )
        )

        web_evidence = (
            web_evidence_builder
            .build(
                web_results
            )
        )

        state.web_evidence = (
            web_evidence
        )

        state.selected_route = (
            "hybrid"
        )

        app_logger.success(
            "Hybrid route complete"
        )

        return state

    def _direct_route(
        self,
        state: QueryState
    ) -> QueryState:
        """
        Direct generation
        without retrieval
        """

        state.selected_route = (
            "direct_generation"
        )

        app_logger.success(
            "Direct generation route complete"
        )

        return state


query_router = (
    QueryRouter()
)