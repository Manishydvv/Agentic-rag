import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    PORTKEY_API_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
