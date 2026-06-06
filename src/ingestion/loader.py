from pathlib import Path
import hashlib
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

        # Deterministic document_id derived
        # from file content, so re-ingesting
        # the same file is idempotent and
        # gold-chunk mappings stay stable.
        file_hash = hashlib.sha1(
            source_path.read_bytes()
        ).hexdigest()[:12]

        document_id = (
            f"doc_{file_hash}"
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