from typing import Optional
from pydantic import BaseModel


class SourceReference(BaseModel):
    """
    Source metadata
    """

    source_id: Optional[str] = None
    source_type: str

    page: Optional[int] = None
    section: Optional[str] = None

    score: Optional[float] = None


class ErrorResponse(BaseModel):
    """
    Standard error response
    """

    status: str = "error"

    message: str
    error_code: Optional[str] = None