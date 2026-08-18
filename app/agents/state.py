from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    The state of the agent.
    messages: The chat history and current query.
    documents: Retrieved context documents.
    next_step: Planner decision ("retrieve" or "respond").
    current_query: The rewritten query for Qdrant, or 'CONVERSATIONAL'.
    plan: A list of reasoning steps to trace the agent's logic.
    status: User-facing status message.
    cached_response: The response hit from the L1 Redis cache (if any).
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    documents: list[str]
    next_step: str
    current_query: str
    plan: list[str]
    status: str
    cached_response: str
