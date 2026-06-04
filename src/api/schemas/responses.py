from typing import Optional
from pydantic import BaseModel


class APIResponse(BaseModel):
    """
    Generic API response
    """

    status: str = "success"

    message: str

    data: Optional[dict] = None