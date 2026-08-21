from app.agents.state import AgentState
from app.gateway.client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.utils.logger import logger

def responder_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    """
    query = state.get("current_query", "")
    messages = state.get("messages", [])
    documents = state.get("documents", [])
    plan = state.get("plan", [])
    
    # Get conversation history for prompt injection
    history_str = ""
    if len(messages) > 1:
        for msg in messages[:-1]:
            role = "User" if msg.type == "human" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
            
    user_msg = messages[-1].content if messages else ""
    
    if query == "CONVERSATIONAL":
        logger.info("RESPONDER: Generating conversational response using memory.")
        system_prompt = f"""You are a friendly and helpful Enterprise AI Assistant.
Answer the user's latest message using the CONVERSATION HISTORY below.

CONVERSATION HISTORY:
{history_str}
"""
    else:
        logger.info("RESPONDER: Generating technical RAG response.")
        # Format context
        context = "\n".join([f"- {doc}" for doc in documents]) if documents else "No context found."
        
        system_prompt = f"""You are a Senior Technical Architect.
Answer the user's question using ONLY the information provided in the TECHNICAL CONTEXT below. 

CRITICAL RULES:
1. If the answer cannot be found in the TECHNICAL CONTEXT, you MUST explicitly say: "I cannot answer this question based on the provided documents."
2. DO NOT use your own outside knowledge. DO NOT hallucinate.
3. If the TECHNICAL CONTEXT is empty or says "No context found", you MUST decline to answer.

TECHNICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history_str}
"""
    
    llm = get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])
        
        logger.info("RESPONDER: Answer generated successfully")
        
        # Save to L1 Cache if this was a brand new stateless question
        if len(messages) == 1:
            from app.services.cache.redis_semantic_cache import save_to_cache
            save_to_cache(user_msg, response.content)
            logger.info("RESPONDER: Saved response to L1 Redis Cache.")
            
        return {
            "messages": [response],
            "status": "Response generated.",
            "plan": plan + ["Response Synthesized"]
        }
    except Exception as e:
        logger.error(f"RESPONDER: LLM Generation failed: {e}")
        error_msg = AIMessage(content="I'm sorry, I encountered an error generating the response.")
        return {
            "messages": [error_msg],
            "status": "Generation failed.",
            "plan": plan + [f"Error: {str(e)}"]
        }
