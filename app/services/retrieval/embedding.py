from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.utils.logger import logger

def get_embeddings():
    """
    Returns the OpenAI Embeddings model.
    Using text-embedding-3-small as default for fast, cheap, and high quality embeddings.
    """
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set in environment.")
        raise ValueError("OPENAI_API_KEY is required for embeddings.")
        
    logger.debug("Initializing OpenAI Embeddings model")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY
    )
