from fastapi import APIRouter
from datetime import datetime

from src.config import settings


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }