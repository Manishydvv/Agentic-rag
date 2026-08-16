from app.agents.state import AgentState
from app.gateway.client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.utils.logger import logger

def responder_node(state: AgentState):
    """
    Generates the final response using the LLM and retrieved documents.
    """
    logger.info("RESPONDER: Generating final answer via Portkey")
    messages = state.get("messages", [])
    documents = state.get("documents", [])
    
    # Format context
    context = "\n".join([f"- {doc}" for doc in documents])
    
    # Get the latest user query
    user_query = messages[-1].content if messages else ""
    
    system_prompt = f"""You are a helpful AI assistant. Answer the user's question using the provided context.
    
Context:
{context}
"""
    
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])
    
    logger.info("RESPONDER: Answer generated successfully")
    return {"messages": [response]}
