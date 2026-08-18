import json
from langchain_openai import ChatOpenAI
from app.config import settings

def get_llm():
    """
    Returns a Portkey-configured Langchain LLM.
    Uses OpenAI as primary, falls back to Groq.
    """
    if not settings.PORTKEY_API_KEY:
        # Fallback to direct OpenAI if Portkey is not configured yet for local testing
        return ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

    portkey_headers = {
        "x-portkey-api-key": settings.PORTKEY_API_KEY,
        "x-portkey-config": "pc-portke-36acb4",
        "x-portkey-cache": "semantic" # Enable L2 Semantic Edge Cache
    }
    
    return ChatOpenAI(
        model="gpt-4o-mini", # The model name here doesn't matter as much when using Portkey routing, but required by Langchain
        api_key="dummy", # Portkey handles the real auth
        base_url="https://api.portkey.ai/v1",
        default_headers=portkey_headers
    )
