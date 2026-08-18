from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.responder import responder_node
from app.agents.nodes.cache_responder import cache_responder_node

# Initialize the graph with our state schema
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("responder", responder_node)
workflow.add_node("cache_responder", cache_responder_node)

# Add edges
# Start at planner
workflow.set_entry_point("planner")

# Conditional routing from planner
def route_from_planner(state: AgentState):
    next_step = state.get("next_step")
    if next_step == "cache":
        return "cache_responder"
    elif next_step == "respond":
        return "responder"
    return "retriever"

workflow.add_conditional_edges("planner", route_from_planner)

# Retriever always goes to responder
workflow.add_edge("retriever", "responder")

# Responder goes to end
workflow.add_edge("responder", END)
workflow.add_edge("cache_responder", END)

# Checkpointer for Conversational Memory
import sqlite3
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile the graph
app_graph = workflow.compile(checkpointer=memory)
