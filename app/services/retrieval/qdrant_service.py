import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore
from app.services.retrieval.embedding import get_embeddings_model
from app.utils.logger import logger
from app.config import settings

# ──────────────────────────────────────────────────────────────────────────────
# Single Source of Truth for Qdrant Configuration
# Both the ingestion pipeline (processor.py) and retriever use these values.
# ──────────────────────────────────────────────────────────────────────────────
QDRANT_PATH = settings.QDRANT_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_DIM = settings.EMBEDDING_DIM

# Ensure the directory exists if we are using local path
if not settings.QDRANT_URL:
    os.makedirs(QDRANT_PATH, exist_ok=True)

# Initialize the client (URL takes precedence over local path)
if settings.QDRANT_URL:
    client = QdrantClient(url=settings.QDRANT_URL)
else:
    client = QdrantClient(path=QDRANT_PATH)


def get_qdrant_client() -> QdrantClient:
    """Returns the shared Qdrant client instance."""
    return client


def ensure_collection(wipe: bool = False):
    """Creates the collection if it doesn't exist. Optionally wipes it first."""
    if wipe and client.collection_exists(COLLECTION_NAME):
        logger.warning(f"Wiping collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' deleted.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created.")


def get_vector_store():
    """
    Returns the LangChain QdrantVectorStore instance for similarity search.
    Used by the retriever node in the LangGraph agent.
    """
    embeddings = get_embeddings_model()

    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text",
    )
    return vector_store

def delete_document_by_id(doc_id: str) -> bool:
    """
    Deletes all vector chunks associated with a specific document ID.
    Uses the Qdrant payload filter to target the precise chunks.
    """
    try:
        if not client.collection_exists(COLLECTION_NAME):
            return False
            
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Qdrant: Successfully deleted vectors for document_id={doc_id}")
        return True
    except Exception as e:
        logger.error(f"Qdrant: Failed to delete vectors for document_id={doc_id}: {e}")
        return False
