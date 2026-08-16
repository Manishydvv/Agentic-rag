import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from app.services.retrieval.embedding import get_embeddings_model
from app.utils.logger import logger

# Use a local folder for Qdrant storage (super fast for development)
LOCAL_QDRANT_PATH = ".qdrant"
COLLECTION_NAME = "agentic_rag_docs"

# Ensure the directory exists
os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)

# Initialize the local client
client = QdrantClient(path=LOCAL_QDRANT_PATH)

def get_vector_store():
    """
    Returns the LangChain QdrantVectorStore instance connected to our local Qdrant.
    """
    embeddings = get_embeddings_model()
    
    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store
