# 🧠 Agentic RAG

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-FF9900.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)
![AWS Fargate](https://img.shields.io/badge/AWS-Fargate-232F3E.svg?logo=amazon-aws)

An enterprise-grade, stateful Agentic Retrieval-Augmented Generation (RAG) system. This project leverages intelligent AI agents to route queries, retrieve context from massive document stores, cache semantic answers, and block adversarial inputs.

## ✨ Features

- **Agentic Routing (LangGraph):** Stateful agents dynamically route queries between L1 Semantic Cache, Qdrant Vector Search, or conversational memory.
- **AI Gateway (Portkey):** Centralized observability and routing. Uses **OpenAI** as the primary LLM, with automatic fallback to **Groq** for high reliability.
- **Enterprise Guardrails:** NVIDIA NeMo Guardrails intercept and block adversarial prompts, jailbreaks, and off-topic queries before they hit the LLM.
- **Semantic Caching:** Redis Enterprise Cloud caches previous answers, returning instant results for repeated or semantically identical questions.
- **Serverless PostgreSQL:** Neon Postgres tracks document ingestion status and provides persistent memory (checkpoints) for the LangGraph agents.
- **Evaluation Suite (`/evals`):** A dedicated Streamlit UI utilizing RAGAS to continuously evaluate RAG metrics, using Groq directly as an LLM-as-a-Judge.
- **Cloud-Ready CI/CD:** Fully containerized with Docker, optimized with BuildKit caching, and ready for serverless deployment on AWS Fargate via GitHub Actions.

---

## 🏗️ Architecture

### 1. API Flow (LangGraph & Portkey)
```mermaid
graph TD
    %% Main API Flow
    User([User Request]) --> API[FastAPI /query]
    API --> Guardrails[NeMo Guardrails]
    
    %% Guardrails
    Guardrails -- Blocked --> Reject([Return Blocked Message])
    Guardrails -- Passed --> GraphEntry[Enter LangGraph]
    
    %% LangGraph Flow
    subgraph LangGraph [LangGraph Stateful Agent]
        GraphEntry --> Planner[Planner Node]
        
        Planner -- next_step = cache --> CacheResponder[Cache Responder Node]
        Planner -- next_step = retrieve --> Retriever[Retriever Node]
        Planner -- next_step = respond --> Responder[Responder Node]
        
        Retriever -- fetches from Qdrant --> Responder
    end
    
    %% Exits
    CacheResponder --> ReturnCache([Return Cached Answer])
    Responder -- saves to Redis --> ReturnAnswer([Return LLM Answer via Portkey])
    
    %% Document Ingestion (Background)
    UI([Document Upload UI]) --> DB[(Neon Postgres metadata)]
    DB --> BgTask[FastAPI Background Task]
    BgTask -- chunks & embeds --> Qdrant[(Qdrant Cloud Vector DB)]
```

### 2. Evaluation Suite (RAGAS)
```mermaid
graph TD
    %% Eval Flow
    StartEval([Run Evals Dashboard]) --> Streamlit[Streamlit UI /evals/app.py]
    Streamlit --> LoadData[Load Golden Dataset]
    
    %% Phase 1
    subgraph Phase1 [Live Pipeline Testing]
        LoadData --> HitAPI[Hit FastAPI /query]
        HitAPI --> Capture[Capture Actual Responses & Contexts]
    end
    
    %% Phase 2
    subgraph Phase2 [RAGAS LLM-as-a-Judge]
        Capture --> Ragas[Ragas Metrics Engine]
        Ragas -- uses JUDGE_GROQ --> Groq[Groq API direct]
        
        Groq --> Faithfulness[Compute Faithfulness]
        Groq --> Relevancy[Compute Answer Relevancy]
        Groq --> Precision[Compute Context Precision]
    end
    
    Phase2 --> Results([Display Metrics & Badges])
```

---

## 💻 Tech Stack

- **Framework:** FastAPI
- **AI & Agents:** LangChain, LangGraph
- **LLM Inference:** OpenAI (Primary) & Groq (Fallback), routed via Portkey AI Gateway
- **Vector Store:** Qdrant Cloud
- **Database (Metadata & State):** Neon PostgreSQL (Serverless)
- **Cache:** Redis Enterprise Cloud
- **Security:** NeMo Guardrails
- **Observability:** LangSmith
- **Evaluation (LLM-as-a-Judge):** Ragas & Streamlit (using Groq)
- **DevOps:** Docker Compose, AWS ECS Fargate, GitHub Actions

---

## 🧪 Evaluation Suite (`/evals`)

The repository includes a production-grade automated evaluation suite located in the `evals/` directory.

- **Dashboard:** A visual Streamlit UI (`evals/app.py`) for running and visualizing evaluation pipelines.
- **LLM-as-a-Judge:** Uses the Ragas framework to compute metrics like Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
- **Direct Groq Integration:** While the main API uses Portkey (OpenAI -> Groq), the Evaluation suite hits **Groq directly** (`groq/compound-mini`) as the judge model. 
- **Rate-Limit Optimized:** The pipeline uses a separate `JUDGE_GROQ` API key to prevent exhausting production limits, and automatically batches requests with cooldowns to stay under Groq's strict 6,000 TPM limit.

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- [Docker & Docker Compose](https://www.docker.com/)
- API Keys for OpenAI, Groq, Portkey, Qdrant, Neon Postgres, and Redis.

### 1. Environment Setup
Create a `.env` file in the root directory and populate it with your cloud credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
PORTKEY_API_KEY=your_portkey_api_key_here
JUDGE_GROQ=your_secondary_groq_key_for_evals # Optional: protects production limits

# Qdrant Cloud Configuration
QDRANT_URL=https://your-qdrant-cloud-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here

# LangSmith Tracing (Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Agentic-RAG

# Managed PostgreSQL Configuration
DATABASE_URL=postgresql://your_postgres_username:your_postgres_password@your-db-endpoint.co:5432/postgres

# Managed Redis Configuration
REDIS_URL=rediss://default:your_redis_password@your-redis-endpoint.upstash.io:30000
```

### 2. Run the Application
The project utilizes Docker BuildKit caching to ensure lightning-fast builds. 

```bash
docker-compose up --build
```

This will spin up two services:
1. **API Backend:** `http://localhost:8000/docs` (FastAPI Swagger UI)
2. **Evals Dashboard:** `http://localhost:8501` (Streamlit RAGAS Evaluation)

### 3. Ingesting Documents
You can ingest PDF, TXT, DOCX, or HTML files by opening the `index.html` frontend UI, or by directly calling the API endpoint:
```bash
curl -X POST "http://localhost:8000/api/documents" -F "file=@your_document.pdf"
```

---

## ☁️ Deployment

This project is configured for a highly scalable, serverless deployment on **AWS ECS Fargate**.

For step-by-step AWS CLI deployment instructions, please refer to the [Deployment Guide](deployment.md). Automated CI/CD is configured via `.github/workflows/deploy.yml`.

---

## 🛡️ Security
**Never commit your `.env` or `task-def.json` files to source control.** They are ignored via `.gitignore`.
