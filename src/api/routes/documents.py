import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from src.api.schemas.documents import (
    DocumentUploadResponse
)

from src.ingestion.pipeline import (
    ingestion_pipeline
)
from src.ingestion.loader import (
    SUPPORTED_EXTENSIONS
)
from src.observability.logger import app_logger


router = APIRouter()


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse
)
def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload a document and ingest it into the
    knowledge base (parse -> chunk -> embed ->
    index). Sync endpoint so the blocking ingestion
    runs in a threadpool.
    """

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type "
                f"'{extension}'. Supported: "
                f"{sorted(SUPPORTED_EXTENSIONS)}"
            )
        )

    # Persist the upload to a temporary path; the
    # ingestion loader copies it to data/uploads
    # under a deterministic, content-derived id.
    tmp_dir = Path("data/tmp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_dir / file.filename

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        summary = ingestion_pipeline.ingest_file(
            str(tmp_path)
        )

    except Exception as exc:
        app_logger.error(
            f"Ingestion failed: {exc!r}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc!r}"
        )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    indexed = summary.get("indexed", False)

    return DocumentUploadResponse(
        document_id=summary.get("document_id") or "",
        filename=summary.get("filename") or file.filename,
        status="indexed" if indexed else "failed",
        message=(
            f"{summary.get('chunks', 0)} chunks "
            f"indexed into the knowledge base."
        )
    )
