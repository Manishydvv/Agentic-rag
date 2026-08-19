# 🤖 Agentic RAG

A highly optimized, enterprise-ready Retrieval-Augmented Generation (RAG) system powered by an Agentic Architecture. This system leverages LangGraph for orchestration, multi-tier semantic caching for ultra-low latency, and cross-encoder reranking for maximum retrieval accuracy.

---

## 🏗️ Architecture & Core Components

### 1. Agentic Workflow (LangGraph)
Unlike traditional linear RAG chains, this system relies on a directed graph of specialized nodes:
- **NeMo Guardrails Gatekeeper**: Validates the safety, relevance, and policy alignment of user queries *before* any LLM processing occurs.
- **Planner Node**: Analyzes intent, checks the local L1 Cache, and refines the query for optimal retrieval.
- **Retriever Node**: Executes a two-stage semantic search process.
- **Responder Node**: Synthesizes the final answer using the LLM and commits the result to the cache.

### 2. Multi-Tier Semantic Caching
To dramatically reduce API costs and latency, the system implements two layers of caching:
- **L1 Cache (Redis)**: Runs locally. The `planner_node` performs a vector similarity check against past queries. If a new query is >95% similar to an existing one, the entire LLM generation pipeline is bypassed.
- **L2 Edge Cache (Portkey)**: If the local cache misses, requests sent to the LLM are cached at the edge API layer using Portkey's `x-portkey-cache: semantic` headers.

### 3. Advanced Two-Stage Retrieval
Ensures the LLM is only fed the highest quality context:
- **Base Retrieval (Qdrant)**: A local SQLite-backed Qdrant vector database quickly fetches the top 15 candidate document chunks using embeddings.
- **Cross-Encoder Reranking (Local ML)**: A dedicated HuggingFace model (`ms-marco-MiniLM-L-6-v2`) runs locally to cross-reference the user's query against the 15 candidate chunks. It filters and compresses the result set down to the **top 5** most highly relevant chunks, drastically reducing noise and LLM hallucination.

### 4. Enterprise LLM Gateway (Portkey)
Instead of hardcoding LLM providers, all API calls are routed through Portkey AI. This enables:
- **Failover Routing**: Seamlessly falls back from primary models (e.g., OpenAI `gpt-4o-mini`) to secondary models (e.g., Groq) if rate limits or downtimes occur.
- **Observability**: Centralized logging and tracing for all LLM interactions.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, Python 3.12+
- **Package Manager**: `uv` (Ultra-fast Rust-based package management)
- **Agent Orchestration**: LangGraph, LangChain
- **Vector Database**: Qdrant (Local)
- **Caching Engine**: Redis
- **Security**: NVIDIA NeMo Guardrails

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.12+
2. [uv](https://github.com/astral-sh/uv) package manager
3. Docker (for running local Redis)
4. API Keys for OpenAI/Groq and Portkey AI

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Manishydvv/Agentic-rag.git
   cd Agentic-rag
   ```
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
3. Set up environment variables:
   Copy `.env.example` to `.env` and fill in your keys:
   ```env
   OPENAI_API_KEY=your_key_here
   PORTKEY_API_KEY=your_key_here
   REDIS_URL=redis://localhost:6379
   ```

### Running the System
1. Start the local Redis container:
   ```bash
   docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
   ```
2. Start the FastAPI application:
   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```
3. Test the query endpoint:
   ```bash
   Invoke-RestMethod -Method Post -Uri "http://localhost:8000/query" -Headers @{"Content-Type"="application/json"} -Body '{"query":"How do I deploy Agentic RAG?", "session_id":"test-1"}'
   ```

---

## 🗺️ Roadmap (Phase 5)
The next major iteration (Phase 5) will transition the system from local file-based ingestion to a **Cloud-Native Automated Pipeline**.
- **S3 Triggered Ingestion**: Uploading documents to AWS S3 will trigger Lambda functions to automatically parse, chunk, and embed documents into a managed Qdrant Cloud cluster.
- **Serverless Scaling**: Decoupling the ingestion pipeline from the query API for massive scalability.
