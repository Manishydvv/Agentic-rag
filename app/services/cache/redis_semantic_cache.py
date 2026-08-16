import json
import numpy as np
import redis
from app.services.retrieval.embedding import get_embeddings_model
from app.config import settings
from app.utils.logger import logger

# Connect to Redis
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# Redis key prefix for our cache
CACHE_PREFIX = "rag_cache:"


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_cache(query: str) -> str | None:
    """
    Check if a semantically similar query exists in the Redis cache.
    Returns the cached response if found (HIT), or None (MISS).
    """
    try:
        embeddings = get_embeddings_model()
        query_embedding = embeddings.embed_query(query)
        
        # Get all cached keys
        cached_keys = redis_client.keys(f"{CACHE_PREFIX}*")
        
        if not cached_keys:
            logger.info("CACHE: No entries in cache yet (MISS)")
            return None
        
        best_score = 0.0
        best_response = None
        
        for key in cached_keys:
            cached_data = redis_client.get(key)
            if not cached_data:
                continue
                
            entry = json.loads(cached_data)
            cached_embedding = entry["embedding"]
            similarity = _cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_response = entry["response"]
        
        if best_score >= settings.CACHE_SIMILARITY_THRESHOLD:
            logger.info(f"CACHE: HIT! Similarity={best_score:.4f} (threshold={settings.CACHE_SIMILARITY_THRESHOLD})")
            return best_response
        else:
            logger.info(f"CACHE: MISS. Best similarity={best_score:.4f} (below threshold={settings.CACHE_SIMILARITY_THRESHOLD})")
            return None
            
    except redis.ConnectionError:
        logger.warning("CACHE: Redis is not available. Skipping cache.")
        return None
    except Exception as e:
        logger.error(f"CACHE: Error checking cache: {e}")
        return None


def save_to_cache(query: str, response: str):
    """
    Save a query-response pair to the Redis cache along with the query embedding.
    """
    try:
        embeddings = get_embeddings_model()
        query_embedding = embeddings.embed_query(query)
        
        cache_entry = {
            "query": query,
            "response": response,
            "embedding": query_embedding,
        }
        
        # Use a hash of the query as the key
        cache_key = f"{CACHE_PREFIX}{hash(query)}"
        redis_client.set(cache_key, json.dumps(cache_entry))
        
        logger.info(f"CACHE: Saved response for query: '{query[:50]}...'")
        
    except redis.ConnectionError:
        logger.warning("CACHE: Redis is not available. Skipping save.")
    except Exception as e:
        logger.error(f"CACHE: Error saving to cache: {e}")
