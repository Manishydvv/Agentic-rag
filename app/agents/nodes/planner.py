from app.agents.state import AgentState
from app.gateway.client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from app.utils.logger import logger

def planner_node(state: AgentState):
    """
    Decides if the agent needs to retrieve documents or can answer directly.
    """
    messages = state.get("messages", [])
    
    # Simple heuristic for now: always retrieve.
    # In a real app, you'd use the LLM to decide or check if the query requires context.
    next_step = "retrieve"
    
    logger.info("DECISION: Need to retrieve documents")
    return {"next_step": next_step}
