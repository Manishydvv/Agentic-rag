from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import get_vector_store
from app.utils.logger import logger

from langchain_classic.retrievers import ContextualCompressionRetriever
from app.config import settings
from app.services.retrieval.reranker_service import get_reranker

def retriever_node(state: AgentState):
    """
    Retrieves documents from Qdrant Vector DB, then reranks them using a Cross-Encoder.
    """
    query = state.get("current_query", "")
    plan = state.get("plan", [])
    
    logger.info(f"RETRIEVER: Searching Qdrant for: '{query}'")
    
    try:
        vector_store = get_vector_store()
        base_retriever = vector_store.as_retriever(search_kwargs={"k": settings.TOP_K_RETRIEVE})
        
        # Wrap in compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=get_reranker(),
            base_retriever=base_retriever
        )
        
        # Retrieve and Rerank
        results = compression_retriever.invoke(query)
        
        # Format the documents clearly for the LLM
        documents = [f"CONTENT: {doc.page_content}" for doc in results]
        
        logger.info(f"RETRIEVER: Found {settings.TOP_K_RETRIEVE} chunks -> Reranked to {len(documents)} chunks")
        return {
            "documents": documents,
            "status": "Found and reranked context.",
            "plan": plan + [f"Context Retrieved & Reranked (Top {len(documents)})"]
        }
        
    except Exception as e:
        logger.error(f"RETRIEVER: Failed to search Qdrant: {e}")
        return {
            "documents": [],
            "status": "Failed to retrieve context.",
            "plan": plan + [f"Retrieval Error: {str(e)}"]
        }
