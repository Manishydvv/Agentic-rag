import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    PORTKEY_API_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    
    # Ingestion & Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    EMBED_BATCH_SIZE: int = 100
    
    # Reranking Configuration
    TOP_K_RETRIEVE: int = 15
    TOP_K_RERANK: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Qdrant Configuration
    QDRANT_URL: str = ""
    QDRANT_PATH: str = "./qdrant_storage"
    COLLECTION_NAME: str = "documents"
    EMBEDDING_DIM: int = 1536
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
