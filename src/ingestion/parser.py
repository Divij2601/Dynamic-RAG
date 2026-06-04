from pathlib import Path
import fitz

from src.observability.logger import (
    app_logger
)


class DocumentParser:
    """
    Parse uploaded documents
    """

    def parse_document(
        self,
        file_path: str
    ) -> dict:
        """
        Route parser
        """

        extension = (
            Path(file_path)
            .suffix
            .lower()
        )

        if extension == ".pdf":
            return self._parse_pdf(
                file_path
            )

        elif extension == ".txt":
            return self._parse_txt(
                file_path
            )

        else:
            raise ValueError(
                f"Unsupported extension: {extension}"
            )

    def _parse_pdf(
        self,
        file_path: str
    ) -> dict:
        """
        Parse PDF document
        """

        document = fitz.open(
            file_path
        )

        pages = []

        for page_num in range(
            len(document)
        ):

            page = (
                document[page_num]
            )

            text = page.get_text(
                "text"
            ).strip()

            if text:

                pages.append({
                    "page_number":
                    page_num + 1,

                    "text":
                    text
                })

        document.close()

        app_logger.success(
            f"PDF parsed: "
            f"{len(pages)} pages"
        )

        return {
            "document_type":
            "pdf",

            "pages": pages,

            "total_pages":
            len(pages)
        }

    def _parse_txt(
        self,
        file_path: str
    ) -> dict:
        """
        Parse TXT document
        """

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = (
                file.read()
                .strip()
            )

        app_logger.success(
            "TXT parsed"
        )

        return {
            "document_type":
            "txt",

            "pages": [{
                "page_number": 1,
                "text": text
            }],

            "total_pages": 1
        }


document_parser = (
    DocumentParser()
)