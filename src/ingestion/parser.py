"""
Document parser supporting:
  - Digital PDFs  (PyMuPDF text extraction)
  - Scanned PDFs  (EasyOCR fallback, lazy-loaded)
  - Plain text    (.txt)

The OCR path activates automatically when a PDF page
yields fewer than MIN_TEXT_CHARS characters of
selectable text. EasyOCR is only imported and its
models downloaded on first use (large download, ~200 MB
for the English model — happens once then is cached).
"""

from pathlib import Path
import fitz

from src.observability.logger import (
    app_logger
)


# Pages with fewer chars than this are treated as
# image-only and sent to OCR.
MIN_TEXT_CHARS = 50

# OCR is capped at this many pages per document to
# avoid extremely long processing on huge scanned books.
# Set to None for no cap.
OCR_PAGE_CAP = 200


class DocumentParser:
    """
    Parse uploaded documents.
    Falls back to EasyOCR for scanned PDF pages.
    """

    _ocr_reader = None

    def _get_ocr_reader(self):
        """
        Lazy-load EasyOCR on first use.
        This triggers a ~200 MB model download
        the very first time it is called.
        """

        if self._ocr_reader is None:

            app_logger.info(
                "Loading EasyOCR (first use — "
                "model download may take a moment)"
            )

            try:
                import easyocr
                # gpu=False keeps it CPU-only;
                # no extra CUDA setup needed.
                self._ocr_reader = (
                    easyocr.Reader(
                        ["en"],
                        gpu=False,
                        verbose=False
                    )
                )
                app_logger.success(
                    "EasyOCR loaded"
                )

            except ImportError:
                app_logger.error(
                    "easyocr not installed. "
                    "Run: pip install easyocr"
                )
                raise

        return self._ocr_reader

    def parse_document(
        self,
        file_path: str
    ) -> dict:

        extension = (
            Path(file_path)
            .suffix
            .lower()
        )

        if extension == ".pdf":
            return self._parse_pdf(file_path)

        elif extension == ".txt":
            return self._parse_txt(file_path)

        else:
            raise ValueError(
                f"Unsupported extension: {extension}"
            )

    def _parse_pdf(
        self,
        file_path: str
    ) -> dict:

        document = fitz.open(file_path)

        pages = []
        ocr_pages = 0
        total_pages = len(document)

        page_limit = (
            min(total_pages, OCR_PAGE_CAP)
            if OCR_PAGE_CAP
            else total_pages
        )

        for page_num in range(page_limit):

            page = document[page_num]
            text = page.get_text("text").strip()

            if len(text) >= MIN_TEXT_CHARS:
                # Good digital text — use as-is.
                pages.append({
                    "page_number": page_num + 1,
                    "text": text
                })

            else:
                # Scanned page — render to image and OCR.
                ocr_text = self._ocr_page(
                    page,
                    page_num + 1
                )

                if ocr_text:
                    pages.append({
                        "page_number": page_num + 1,
                        "text": ocr_text
                    })
                    ocr_pages += 1

        document.close()

        if page_limit < total_pages:
            app_logger.warning(
                f"OCR page cap ({OCR_PAGE_CAP}) "
                f"reached; {total_pages - page_limit} "
                f"pages skipped."
            )

        app_logger.success(
            f"PDF parsed: {len(pages)} pages "
            f"({ocr_pages} via OCR) "
            f"from {Path(file_path).name}"
        )

        return {
            "document_type": "pdf",
            "pages": pages,
            "total_pages": len(pages),
            "ocr_pages": ocr_pages
        }

    def _ocr_page(
        self,
        page,
        page_number: int
    ) -> str:
        """
        Render a fitz page to a pixel image and
        run EasyOCR on it. Returns extracted text
        or empty string on failure.
        """

        try:
            reader = self._get_ocr_reader()

            # Render at 2x scale for better OCR accuracy.
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            results = reader.readtext(
                img_bytes,
                detail=0,          # text only, no boxes
                paragraph=True     # merge lines into paragraphs
            )

            text = "\n".join(results).strip()

            if text:
                app_logger.info(
                    f"OCR page {page_number}: "
                    f"{len(text)} chars extracted"
                )

            return text

        except Exception as exc:
            app_logger.error(
                f"OCR failed on page "
                f"{page_number}: {exc!r}"
            )
            return ""

    def _parse_txt(
        self,
        file_path: str
    ) -> dict:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read().strip()

        app_logger.success("TXT parsed")

        return {
            "document_type": "txt",
            "pages": [{"page_number": 1, "text": text}],
            "total_pages": 1
        }


document_parser = DocumentParser()
