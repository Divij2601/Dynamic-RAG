from fastapi import APIRouter
from datetime import datetime

from src.config import settings
from src.database.mongo_client import mongo_client
from src.database.qdrant_client import qdrant_client


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """

    mongo_status = (
        "connected"
        if mongo_client.ping()
        else "disconnected"
    )

    qdrant_status = (
        "connected"
        if qdrant_client.ping()
        else "disconnected"
    )

    overall_status = "healthy"

    if (
        mongo_status == "disconnected"
        or qdrant_status == "disconnected"
    ):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),

        "services": {
            "mongodb": mongo_status,
            "qdrant": qdrant_status
        }
    }