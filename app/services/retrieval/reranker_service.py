from app.utils.logger import logger
from app.config import settings

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# Initialize model once to keep it in memory
_cross_encoder_model = None

def get_reranker():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        logger.info(f"Loading Cross-Encoder model: {settings.RERANKER_MODEL}")
        _cross_encoder_model = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
    return CrossEncoderReranker(model=_cross_encoder_model, top_n=settings.TOP_K_RERANK)
