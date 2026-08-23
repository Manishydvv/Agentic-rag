from app.agents.state import AgentState
from app.gateway.client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from app.utils.logger import logger
from app.services.cache.redis_semantic_cache import check_cache

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    It also acts as the L1 Semantic Cache for stateless queries.
    """
    messages = state.get("messages", [])
    user_message = messages[-1].content if messages else ""
    
    # ---------------------------------------------------------
    # L1 CACHE (Redis) - Only run on the very first message
    # ---------------------------------------------------------
    if len(messages) == 1:
        logger.info(f"PLANNER (L1): Checking Redis Semantic Cache for '{user_message}'")
        cached_response = check_cache(user_message)
        if cached_response:
            logger.info("PLANNER (L1): Cache HIT! Bypassing LLM pipeline.")
            return {
                "next_step": "cache",
                "cached_response": cached_response,
                "status": "Cache Hit",
                "plan": ["Intent: L1 Cache Hit", "Action: Return cached response"]
            }
    
    # ---------------------------------------------------------
    # INTENT ANALYSIS (If L1 Miss or Follow-up)
    # ---------------------------------------------------------
    llm = get_llm()
    
    # Get the conversation history (excluding the latest message)
    history = ""
    if len(messages) > 1:
        for msg in messages[:-1]:
            role = "User" if msg.type == "human" else "Assistant"
            history += f"{role}: {msg.content}\n"
    
    prompt = f"""
    You are a strictly logical Assistant Planner. 
    Your ONLY job is to classify the user's latest message and decide if we need to search our documentation database.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Classification Rules:
    1. If the message is a pure greeting ("hi", "hello"), a polite closing ("thanks"), or a direct reference to the immediate conversation history ("can you repeat that"), output EXACTLY: CONVERSATIONAL
    2. If the message is a technical question, a request for facts, a coding question, or ANY query that would benefit from reading external documentation, you MUST output a highly specific search query optimized for a vector database.
    
    CRITICAL: Never try to answer technical questions from your own memory. If it is a technical question, ALWAYS output a search query.
    
    Output ONLY 'CONVERSATIONAL' or the refined search query. No other text whatsoever.
    """
    
    logger.info("PLANNER: Analyzing intent...")
    try:
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])
        decision = response.content.strip()
    except Exception as e:
        logger.error(f"Planner LLM failed: {e}")
        # Fallback to technical search using raw query
        decision = user_message

    logger.info(f"PLANNER: Decision = {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "next_step": "respond",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "next_step": "retrieve",
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
