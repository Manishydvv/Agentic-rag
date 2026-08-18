from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import logger
from app.config import settings

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# chunk_size=1000 chars (~200-250 tokens): balances context vs. precision
# chunk_overlap=150 chars (~15%): preserves context at chunk boundaries
# ──────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP

# Separator priority: paragraph → sentence → word → character
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=SEPARATORS,
)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.
    All original metadata (source, page, section, etc.) is preserved
    in every chunk via LangChain's split_documents().

    Args:
        documents: List of Documents from any loader.

    Returns:
        List of chunked Documents with metadata intact.
    """
    if not documents:
        logger.warning("Chunker: Received empty document list.")
        return []

    chunks = _splitter.split_documents(documents)

    # Filter out empty chunks
    clean_chunks = [c for c in chunks if c.page_content.strip()]

    total_chars = sum(len(d.page_content) for d in documents)
    logger.info(
        f"Chunker: {len(documents)} document(s) ({total_chars} chars) -> "
        f"{len(clean_chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return clean_chunks
