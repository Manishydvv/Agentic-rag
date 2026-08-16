from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.docx import parse_docx, parse_xlsx, parse_pptx, parse_csv
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text

__all__ = ["parse_pdf", "parse_docx", "parse_xlsx", "parse_pptx", "parse_csv", "parse_html", "parse_text"]
