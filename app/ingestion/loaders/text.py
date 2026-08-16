from typing import List
from langchain_core.documents import Document
from app.utils.logger import logger


def parse_text(file_path: str) -> List[Document]:
    """
    Reads a plain text or Markdown file with automatic encoding detection.
    Tries UTF-8 first, then falls back to Latin-1.

    Returns one Document for the whole file with source metadata.
    """
    logger.info(f"Text Loader: Parsing '{file_path}'")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()

            logger.info(
                f"Text Loader: Extracted {len(text)} chars "
                f"(encoding: {encoding})"
            )
            return [Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "encoding": encoding,
                    "file_type": file_path.rsplit(".", 1)[-1].lower(),
                }
            )]

        except (UnicodeDecodeError, LookupError):
            continue
        except Exception as e:
            logger.error(f"Text Loader: Failed to read '{file_path}': {e}")
            return []

    logger.error(f"Text Loader: Could not decode '{file_path}' with any known encoding")
    return []
