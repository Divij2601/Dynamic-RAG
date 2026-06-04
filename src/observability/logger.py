import sys
from pathlib import Path

from loguru import logger

from src.config import settings


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger():
    """
    Configure application logger
    """

    logger.remove()

    # Console logging
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> "
            "- <level>{message}</level>"
        )
    )

    # File logging
    logger.add(
        "logs/dynamic_rag.log",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level=settings.LOG_LEVEL,
        serialize=True
    )

    return logger


app_logger = setup_logger()