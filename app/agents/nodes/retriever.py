from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import get_vector_store
from app.utils.logger import logger

def retriever_node(state: AgentState):
    """
    Retrieves documents from Qdrant Vector DB based on the query.
    """
    messages = state.get("messages", [])
    user_query = messages[-1].content if messages else ""
    
    logger.info(f"RETRIEVER: Searching Qdrant for: '{user_query}'")
    
    try:
        vector_store = get_vector_store()
        
        # Retrieve top 3 chunks
        results = vector_store.similarity_search(user_query, k=3)
        documents = [doc.page_content for doc in results]
        
        logger.info(f"RETRIEVER: Found {len(documents)} relevant chunks")
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"RETRIEVER: Failed to search Qdrant: {e}")
        return {"documents": []}
