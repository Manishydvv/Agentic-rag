from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.responder import responder_node

# Initialize the graph with our state schema
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("responder", responder_node)

# Add edges
# Start at planner
workflow.set_entry_point("planner")

# Conditional routing from planner
def route_from_planner(state: AgentState):
    if state.get("next_step") == "retrieve":
        return "retriever"
    return "responder"

workflow.add_conditional_edges("planner", route_from_planner)

# Retriever always goes to responder
workflow.add_edge("retriever", "responder")

# Responder goes to end
workflow.add_edge("responder", END)

# Compile the graph
# Note: We are omitting the checkpointer (PostgresSaver) for Phase 1 local testing
app_graph = workflow.compile()
