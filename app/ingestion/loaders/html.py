from typing import List
from langchain_core.documents import Document
from app.utils.logger import logger


def parse_html(file_path: str) -> List[Document]:
    """
    Parses an HTML file, stripping all tags, scripts, and styles
    to return only the visible, readable text content.

    Returns one Document for the whole file with source metadata.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error(
            "HTML Loader requires 'beautifulsoup4' and 'lxml'. "
            "Install with: uv pip install beautifulsoup4 lxml"
        )
        return []

    logger.info(f"HTML Loader: Parsing '{file_path}'")

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "lxml")

        # Extract page title if available
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove invisible elements
        for tag in soup(["script", "style", "meta", "head", "noscript", "nav", "footer"]):
            tag.decompose()

        # Get clean text
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
        clean_text = "\n".join(line for line in lines if line)

        if not clean_text:
            logger.warning(f"HTML Loader: No text extracted from '{file_path}'")
            return []

        logger.info(f"HTML Loader: Extracted {len(clean_text)} chars")
        return [Document(
            page_content=clean_text,
            metadata={
                "source": file_path,
                "title": page_title,
                "file_type": "html",
            }
        )]

    except Exception as e:
        logger.error(f"HTML Loader: Failed on '{file_path}': {e}")
        return []
