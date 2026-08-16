from typing import List
from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.utils.logger import logger


def get_embeddings_model() -> OpenAIEmbeddings:
    """
    Returns the OpenAI Embeddings model instance.
    Used by LangChain integrations (e.g., Qdrant vector store).
    """
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set in environment.")
        raise ValueError("OPENAI_API_KEY is required for embeddings.")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of text strings and returns their vector embeddings.
    Uses OpenAI text-embedding-3-small (1536 dimensions).

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = get_embeddings_model()
    logger.debug(f"Embedding {len(texts)} texts via OpenAI...")
    embeddings = model.embed_documents(texts)
    return embeddings
