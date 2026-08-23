import time
import os
import uuid
import shutil
from dotenv import load_dotenv

# Load .env into os.environ before anything else (crucial for LangSmith tracing when running locally without Docker)
load_dotenv()

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, status
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

from app.services.db.metadata_service import get_all_active_documents, create_document, update_document_status
from app.services.retrieval.qdrant_service import delete_document_by_id, get_qdrant_client
from app.services.cache.redis_semantic_cache import clear_cache
from app.ingestion.processor import process_file

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
    sources: list[str] = []

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
    
    result = await app_graph.ainvoke(inputs, config=config)
    final_message = result["messages"][-1].content
    plan = result.get("plan", [])
    status = result.get("status", "Done")
    documents = result.get("documents", [])
    
    # Determine source for UI
    source = "cache" if result.get("next_step") == "cache" else "agent"
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Agent response generated in {elapsed:.0f}ms")
    
    return {
        "response": final_message, 
        "source": source,
        "plan": plan,
        "status": status,
        "sources": documents
    }

# ==========================================
# Document Lifecycle Management (CRUD)
# ==========================================

@app.get("/api/documents")
def get_documents():
    """Lists all active and processing documents in the knowledge base."""
    docs = get_all_active_documents()
    return {"documents": docs}

@app.post("/api/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Async upload and ingestion of a new document."""
    doc_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    
    # Save the file temporarily
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", f"{doc_id}_{filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Write processing status to SQLite
    create_document(doc_id, filename)
    
    # Dispatch processing to a background task
    qdrant_client = get_qdrant_client()
    background_tasks.add_task(process_file, file_path, qdrant_client, extract_meta=True, force=True, doc_id=doc_id)
    
    return {"message": "Document accepted for processing", "doc_id": doc_id, "status": "processing"}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Purges a document from Qdrant, invalidates the cache, and soft-deletes in SQLite."""
    # 1. Phase 1: Qdrant Hard Delete
    success = delete_document_by_id(doc_id)
    if not success:
        # We don't fail immediately, we just log it. 
        # Maybe Qdrant already didn't have it, but we still want to soft-delete.
        logger.warning(f"Could not delete vectors for {doc_id} from Qdrant.")
        
    # 2. Phase 2: Cache Invalidation
    clear_cache()
    
    # 3. Phase 3: SQLite Soft Delete
    update_document_status(doc_id, "deleted")
    
    return {"message": f"Document {doc_id} successfully deleted and cache invalidated."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
