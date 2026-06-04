from fastapi import APIRouter


router = APIRouter()


@router.get("/system/metrics")
async def get_metrics():
    """
    Placeholder metrics endpoint
    """

    return {
        "status": "active",
        "message": "Metrics collection initialized"
    }