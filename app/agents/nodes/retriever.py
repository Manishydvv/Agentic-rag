from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import get_vector_store
from app.utils.logger import logger

def retriever_node(state: AgentState):
    """
    Retrieves documents from Qdrant Vector DB based on the rewritten query.
    """
    query = state.get("current_query", "")
    plan = state.get("plan", [])
    
    logger.info(f"RETRIEVER: Searching Qdrant for: '{query}'")
    
    try:
        vector_store = get_vector_store()
        
        # Retrieve top 5 chunks
        results = vector_store.similarity_search(query, k=5)
        
        # Format the documents clearly for the LLM
        documents = [f"CONTENT: {doc.page_content}" for doc in results]
        
        logger.info(f"RETRIEVER: Found {len(documents)} relevant chunks")
        return {
            "documents": documents,
            "status": "Found technical context.",
            "plan": plan + ["Context Retrieved"]
        }
        
    except Exception as e:
        logger.error(f"RETRIEVER: Failed to search Qdrant: {e}")
        return {
            "documents": [],
            "status": "Failed to retrieve context.",
            "plan": plan + [f"Retrieval Error: {str(e)}"]
        }
