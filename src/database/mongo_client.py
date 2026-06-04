from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional

from src.config import settings
from src.observability.logger import app_logger


class MongoDBClient:
    """
    MongoDB singleton client
    """

    _client: Optional[MongoClient] = None

    @classmethod
    def connect(cls) -> MongoClient:
        """
        Create MongoDB connection
        """

        if cls._client is None:

            try:
                cls._client = MongoClient(
                    settings.MONGO_URI,
                    serverSelectionTimeoutMS=5000
                )

                # Ping database
                cls._client.admin.command("ping")

                app_logger.success(
                    "MongoDB connected successfully"
                )

            except ConnectionFailure as e:

                app_logger.error(
                    f"MongoDB connection failed: {str(e)}"
                )

                raise e

        return cls._client

    @classmethod
    def get_database(cls):

        client = cls.connect()

        return client["dynamic_rag_db"]

    @classmethod
    def ping(cls) -> bool:
        """
        Health check for MongoDB
        """

        try:
            cls.connect().admin.command("ping")
            return True

        except Exception as e:

            app_logger.error(
                f"MongoDB ping failed: {str(e)}"
            )

            return False


mongo_client = MongoDBClient()