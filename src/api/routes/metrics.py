from fastapi import APIRouter

from src.observability.metrics import (
    get_system_metrics
)


router = APIRouter()


@router.get("/system/metrics")
def get_metrics():
    """
    Operational metrics aggregated from persisted
    request traces.
    """

    return get_system_metrics()
