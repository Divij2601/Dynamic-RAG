from src.memory.store import (
    conversation_store
)

from src.memory.semantic import (
    semantic_memory
)

from src.observability.logger import (
    app_logger
)


class MemoryRetriever:
    """
    Unified memory retrieval
    """

    def retrieve_context(
        self,
        session_id: str,

        query: str,

        recent_limit: int = 5
    ) -> str:
        """
        Build unified memory
        context
        """

        recent_context = (
            self._recent_memory(
                session_id,
                recent_limit
            )
        )

        semantic_context = (
            semantic_memory
            .retrieve_memory(
                session_id=(
                    session_id
                ),

                query=query
            )
        )

        sections = []

        if recent_context:

            sections.append(
                f"""
RECENT MEMORY
--------------
{recent_context}
"""
            )

        if semantic_context:

            sections.append(
                f"""
SEMANTIC MEMORY
----------------
{semantic_context}
"""
            )

        final_context = (
            "\n\n".join(
                sections
            )
        )

        app_logger.success(
            "Unified memory "
            "retrieved"
        )

        return final_context

    def _recent_memory(
        self,
        session_id: str,
        limit: int
    ) -> str:
        """
        Get recent memory
        """

        history = (
            conversation_store
            .get_recent_context(
                session_id=(
                    session_id
                ),

                limit=limit
            )
        )

        if not history:
            return ""

        blocks = []

        for item in history:

            blocks.append(
                f"""
User:
{item.get('query', '')}

Assistant:
{item.get('answer', '')}
"""
            )

        return "\n".join(
            blocks
        )


memory_retriever = (
    MemoryRetriever()
)