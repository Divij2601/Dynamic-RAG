from fastapi import FastAPI

from src.config import settings
from src.observability.logger import app_logger

from src.api.routes.health import router as health_router
from src.api.routes.metrics import router as metrics_router


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Dynamic-RAG API"
)


@app.on_event("startup")
async def startup_event():
    """
    Startup lifecycle event
    """

    app_logger.info(
        f"{settings.APP_NAME} server starting..."
    )


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown lifecycle event
    """

    app_logger.info(
        f"{settings.APP_NAME} shutting down..."
    )


# Register routes
app.include_router(health_router)
app.include_router(metrics_router)