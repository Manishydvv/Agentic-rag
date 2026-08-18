from app.agents.state import AgentState
from langchain_core.messages import AIMessage
from app.utils.logger import logger

def cache_responder_node(state: AgentState):
    """
    Returns the cached response from L1 Redis without hitting the LLM.
    """
    logger.info("CACHE RESPONDER: Returning L1 Cache Hit.")
    cached_text = state.get("cached_response", "Error: No cached response found.")
    
    return {
        "messages": [AIMessage(content=cached_text)],
        "status": "L1 Cache Hit"
    }
