import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
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

# Ensure the directory exists
os.makedirs(QDRANT_PATH, exist_ok=True)

# Initialize the local client
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
    )
    return vector_store
