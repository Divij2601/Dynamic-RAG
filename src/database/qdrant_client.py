from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.observability.logger import app_logger


class QdrantDBClient:
    """
    Qdrant singleton client
    """

    _client: Optional[QdrantClient] = None

    @classmethod
    def connect(cls) -> QdrantClient:
        """
        Create Qdrant connection
        """

        if cls._client is None:

            try:
                cls._client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY or None,
                    timeout=settings.REQUEST_TIMEOUT
                )

                # Test connection
                cls._client.get_collections()

                app_logger.success(
                    "Qdrant connected successfully"
                )

            except Exception as e:

                app_logger.error(
                    f"Qdrant connection failed: {str(e)}"
                )

                raise e

        return cls._client

    @classmethod
    def get_client(cls) -> QdrantClient:
        """
        Return Qdrant client
        """

        return cls.connect()

    @classmethod
    def ping(cls) -> bool:
        """
        Health check for Qdrant
        """

        try:
            cls.connect().get_collections()
            return True

        except Exception as e:

            app_logger.error(
                f"Qdrant ping failed: {str(e)}"
            )

            return False


qdrant_client = QdrantDBClient()