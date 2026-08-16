import os
from nemoguardrails import RailsConfig, LLMRails
from app.config import settings
from app.utils.logger import logger

# Path to our guardrails config directory
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

# Set the OpenAI key for NeMo to use
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Load config and initialize rails (done once at startup)
config = RailsConfig.from_path(CONFIG_DIR)
rails = LLMRails(config)

# Our unique marker that only appears when self_check_input actually blocks
BLOCK_MARKER = "<<GUARDRAIL_BLOCKED>>"


async def check_guardrails(query: str) -> dict:
    """
    Run the user query through NeMo Guardrails.
    
    Returns:
        dict with keys:
            - "allowed": True if the query passes, False if blocked
            - "message": The guardrail's response message (only if blocked)
    """
    try:
        logger.info(f"GATE 1: Running NeMo Guardrails check on: '{query}'")
        
        response = await rails.generate_async(
            messages=[{"role": "user", "content": query}]
        )
        
        bot_message = response.get("content", "")
        
        # Only block if our unique marker is present
        if BLOCK_MARKER in bot_message:
            logger.warning(f"GATE 1: BLOCKED query: '{query}'")
            return {
                "allowed": False,
                "message": "I'm sorry, I can't respond to that request."
            }
        else:
            logger.info("GATE 1: Query PASSED guardrails check")
            return {"allowed": True, "message": ""}
            
    except Exception as e:
        logger.error(f"GATE 1: Guardrails error: {e}. Allowing query to proceed.")
        # Fail-open: if guardrails crash, let the query through
        return {"allowed": True, "message": ""}
