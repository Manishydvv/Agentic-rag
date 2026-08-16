"""
PDF Loader
==========
Two-stage parser:
  Stage 1: PyMuPDF — fast, handles digital/searchable PDFs
  Stage 2: Tesseract OCR — fallback for scanned/image PDFs

Returns: List[Document] with page-level metadata.
"""

import sys
import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document
from app.utils.logger import logger

# On Windows, set the Tesseract executable path
if sys.platform == "win32":
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    except ImportError:
        pass

# Minimum avg chars per page to consider text valid (else → OCR fallback)
MIN_CHARS_PER_PAGE = 100


def parse_pdf(file_path: str) -> List[Document]:
    """
    Parses a PDF and returns one Document per page.
    Metadata per document: source, page, total_pages, file_type.
    """
    logger.info(f"PDF Loader: Parsing '{file_path}'")
    documents = []

    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_text = []

        # Stage 1: PyMuPDF text extraction
        for page in doc:
            pages_text.append(page.get_text("text").strip())

        avg_chars = sum(len(t) for t in pages_text) / max(total_pages, 1)

        if avg_chars < MIN_CHARS_PER_PAGE:
            logger.warning(
                f"PDF Loader: Scanned PDF detected "
                f"(avg {avg_chars:.0f} chars/page). Falling back to OCR..."
            )
            pages_text = _ocr_pdf(doc, file_path)

        # Build one Document per page
        for page_num, text in enumerate(pages_text):
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "file_type": "pdf",
                    }
                ))

        doc.close()
        logger.info(
            f"PDF Loader: Extracted {len(documents)} pages "
            f"from '{file_path}'"
        )

    except Exception as e:
        logger.error(f"PDF Loader: Failed on '{file_path}': {e}")

    return documents


def _ocr_pdf(doc: fitz.Document, file_path: str) -> List[str]:
    """OCR fallback using Tesseract. Returns list of per-page text strings."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        logger.error(
            "OCR fallback requires 'pytesseract' and 'Pillow'. "
            "Install: uv pip install pytesseract pillow\n"
            "Also install Tesseract: winget install UB-Mannheim.TesseractOCR"
        )
        return [""] * len(doc)

    pages_text = []
    for page_num, page in enumerate(doc):
        try:
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="eng")
            pages_text.append(text.strip())
            logger.debug(f"  OCR page {page_num + 1}: {len(text)} chars")
        except Exception as e:
            logger.warning(f"  OCR failed for page {page_num + 1}: {e}")
            pages_text.append("")

    return pages_text
