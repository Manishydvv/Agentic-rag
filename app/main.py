import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agents.graph import app_graph
from app.services.cache.redis_semantic_cache import check_cache, save_to_cache
from app.guardrails.rails import check_guardrails
from app.utils.logger import logger
from app.config import settings

app = FastAPI(
    title="Agentic RAG API",
    description="FastAPI Backend for Agentic RAG System",
    version="0.1.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("ui/index.html")

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

class QueryResponse(BaseModel):
    response: str
    source: str  # "cache", "agent", or "guardrail"
    plan: list[str] = []
    status: str = ""

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    start_time = time.time()
    
    # ---- GATE 1: NeMo Guardrails Check ----
    guardrail_result = await check_guardrails(request.query)
    
    if not guardrail_result["allowed"]:
        elapsed = (time.time() - start_time) * 1000
        logger.warning(f"Query BLOCKED by guardrails in {elapsed:.0f}ms")
        return {"response": guardrail_result["message"], "source": "guardrail"}
    
    # ---- GATE 2: Invoke Agentic RAG Pipeline ----
    logger.info(f"Running LangGraph Agent")
    inputs = {
        "messages": [HumanMessage(content=request.query)],
        "documents": [],
        "next_step": "",
        "current_query": "",
        "plan": [],
        "status": ""
    }
    
    # Configure checkpointer thread
    config = {"configurable": {"thread_id": request.session_id or "default"}}
    
    result = app_graph.invoke(inputs, config=config)
    final_message = result["messages"][-1].content
    plan = result.get("plan", [])
    status = result.get("status", "Done")
    
    # Determine source for UI
    source = "cache" if result.get("next_step") == "cache" else "agent"
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Agent response generated in {elapsed:.0f}ms")
    
    return {
        "response": final_message, 
        "source": source,
        "plan": plan,
        "status": status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
