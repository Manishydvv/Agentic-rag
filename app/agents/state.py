from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    The state of the agent.
    messages: The chat history and current query.
    documents: Retrieved context documents.
    next_step: Planner decision ("retrieve" or "respond").
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    documents: list[str]
    next_step: str
