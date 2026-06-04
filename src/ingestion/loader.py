from pathlib import Path
from uuid import uuid4
import shutil

from src.observability.logger import app_logger


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt"
}


class DocumentLoader:
    """
    Handles document loading
    and persistence.
    """

    def __init__(self):

        self.upload_dir = Path(
            "data/uploads"
        )

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_document(
        self,
        file_path: str
    ) -> dict:
        """
        Save uploaded document
        """

        source_path = Path(file_path)

        extension = (
            source_path.suffix.lower()
        )

        if extension not in (
            SUPPORTED_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        document_id = (
            f"doc_{uuid4().hex[:12]}"
        )

        target_filename = (
            f"{document_id}{extension}"
        )

        target_path = (
            self.upload_dir /
            target_filename
        )

        shutil.copy2(
            source_path,
            target_path
        )

        app_logger.success(
            f"Document saved: {target_filename}"
        )

        return {
            "document_id": document_id,
            "file_path": str(
                target_path
            ),
            "filename": (
                source_path.name
            ),
            "extension": extension
        }


document_loader = (
    DocumentLoader()
)