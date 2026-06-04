from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """
    Document upload response
    """

    document_id: str

    filename: str

    status: str = "uploaded"

    message: str