from typing import Dict, Any, Optional

from src.database.mongo_client import mongo_client
from src.observability.logger import app_logger


class BaseRepository:
    """
    Base repository for Mongo collections
    """

    def __init__(self, collection_name: str):

        self.db = mongo_client.get_database()
        self.collection = self.db[collection_name]

        app_logger.info(
            f"Repository initialized: {collection_name}"
        )


class SessionRepository(BaseRepository):
    """
    Session storage repository
    """

    def __init__(self):
        super().__init__("sessions")

    def create_session(
        self,
        session_data: Dict[str, Any]
    ):

        result = self.collection.insert_one(
            session_data
        )

        return str(result.inserted_id)

    def get_session(
        self,
        session_id: str
    ) -> Optional[Dict]:

        return self.collection.find_one(
            {"session_id": session_id}
        )


class TraceRepository(BaseRepository):
    """
    Trace persistence repository
    """

    def __init__(self):
        super().__init__("traces")

    def save_trace(
        self,
        trace_data: Dict[str, Any]
    ):

        result = self.collection.insert_one(
            trace_data
        )

        return str(result.inserted_id)


class DocumentRepository(BaseRepository):
    """
    Document metadata repository
    """

    def __init__(self):
        super().__init__("documents")

    def create_document(
        self,
        document_data: Dict[str, Any]
    ):

        result = self.collection.insert_one(
            document_data
        )

        return str(result.inserted_id)


class MemoryRepository(BaseRepository):
    """
    Semantic memory repository
    """

    def __init__(self):
        super().__init__("memory")


# Singleton repositories

session_repo = SessionRepository()
trace_repo = TraceRepository()
document_repo = DocumentRepository()
memory_repo = MemoryRepository()