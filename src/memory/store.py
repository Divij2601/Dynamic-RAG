from datetime import datetime

from src.database.mongo_client import (
    mongo_client
)

from src.observability.logger import (
    app_logger
)


class ConversationStore:
    """
    Mongo conversation store
    """

    def __init__(self):

        self.db = (
            mongo_client
            .get_database()
        )

        self.collection = (
            self.db[
                "conversation_memory"
            ]
        )

    def save_interaction(
        self,
        session_id: str,

        query: str,

        answer: str,

        route: str,

        confidence: float
    ):
        """
        Save interaction
        """

        document = {
            "session_id":
            session_id,

            "query":
            query,

            "answer":
            answer,

            "route":
            route,

            "confidence":
            confidence,

            "timestamp":
            datetime.utcnow()
        }

        self.collection.insert_one(
            document
        )

        app_logger.success(
            "Interaction saved"
        )

    def get_recent_context(
        self,
        session_id: str,
        limit: int = 5
    ):
        """
        Get recent conversation
        """

        results = list(
            self.collection
            .find({
                "session_id":
                session_id
            })
            .sort(
                "timestamp",
                -1
            )
            .limit(limit)
        )

        results.reverse()

        return results


conversation_store = (
    ConversationStore()
)