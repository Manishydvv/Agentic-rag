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

class QueryResponse(BaseModel):
    response: str
    source: str  # "cache", "agent", or "guardrail"

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
    
    # ---- GATE 2: Semantic Cache Check ----
    logger.info(f"GATE 2: Checking semantic cache for: '{request.query}'")
    cached_response = check_cache(request.query)
    
    if cached_response:
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Returning cached response in {elapsed:.0f}ms")
        return {"response": cached_response, "source": "cache"}
    
    # ---- Run the LangGraph Agent ----
    logger.info("Cache MISS → Running LangGraph Agent")
    inputs = {
        "messages": [HumanMessage(content=request.query)],
        "documents": [],
        "next_step": ""
    }
    
    result = app_graph.invoke(inputs)
    final_message = result["messages"][-1].content
    
    # Save the result to cache for next time
    save_to_cache(request.query, final_message)
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Agent response generated in {elapsed:.0f}ms")
    
    return {"response": final_message, "source": "agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
