from langgraph.graph import StateGraph, END
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
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from app.config import settings
from app.utils.logger import logger

try:
    # Use a ConnectionPool with aggressive idle timeouts to survive Neon Serverless dropping idle connections
    pool = ConnectionPool(
        conninfo=settings.DATABASE_URL,
        max_size=10,
        max_idle=120,      # Proactively close connections after 2 mins of inactivity
        max_lifetime=300,  # Recycle connections every 5 mins
        kwargs={"autocommit": True}
    )
    memory = PostgresSaver(pool)
    memory.setup() # Ensures checkpoints tables exist in PostgreSQL
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL for LangGraph Checkpoints: {e}")
    memory = None

# Compile the graph
app_graph = workflow.compile(checkpointer=memory)
