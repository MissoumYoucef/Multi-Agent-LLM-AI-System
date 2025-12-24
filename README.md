# 🤖 Multi-Agent RAG System with LLM Orchestration

A production-ready **Multi-Agent LLM System** built with **LangChain** and **LangGraph**, featuring **Retrieval-Augmented Generation (RAG)**, **self-reflection**, **ReAct reasoning**, and comprehensive **guardrails**. It is designed for scalable, reliable AI-powered document Q&A, capable of running both in the cloud (Google Gemini) and locally (Ollama).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

-   [Features](#-features)
-   [Architecture Overview](#-architecture-overview)
-   [System Modules](#-system-modules)
    -   [Agents](#agents-srcagents)
    -   [RAG Pipeline](#rag-pipeline-srcrag)
    -   [Memory](#memory-management-srcmemory)
    -   [Monitoring](#monitoring--observability-srcmonitoring)
-   [Installation](#-installation)
-   [Configuration](#-configuration)
-   [Running the System](#-running-the-system)
    -   [Cloud Mode (Gemini)](#cloud-mode-gemini)
    -   [Local Mode (Ollama)](#local-mode-ollama)
-   [Docker Deployment](#-docker-deployment)
-   [Testing](#-testing)
-   [Project Structure](#-project-structure)

---

## ✨ Features

### Multi-Agent Orchestration
-   **LangGraph Workflow** - State machine-based agent coordination.
-   **Dynamic Routing** - Automatic task delegation based on query type.
-   **Self-Reflection** - Agents critique and improve their own responses.
-   **ReAct Reasoning** - Explicit "Thought → Action → Observation" loops.

### RAG Pipeline
-   **Hybrid Search** - BM25 keyword + vector semantic search ensemble.
-   **PDF Document Loading** - Native PDF parsing and chunking.
-   **Chroma Vector Store** - Persistent vector database.
-   **Dual Model Support** - Google Gemini (Cloud) or Llama/Sentence-Transformers (Local).

### Production Features
-   **Input/Output Guardrails** - Prompt injection detection, content filtering.
-   **Conversation Memory** - Short-term buffer + long-term summarization.
-   **Cost Control** - Token tracking and budget management.
-   **Response Caching** - Redis-backed caching for repeated queries.
-   **Distributed Tracing** - OpenTelemetry + Jaeger integration.

---

## 🏗 Architecture Overview

The system uses a hub-and-spoke architecture where an **Orchestrator** manages the flow of information between specialized agents and the user.

```mermaid
graph TD
    User([User Input]) --> G_In[Input Guardrails]
    G_In --> Orch[Orchestrator]
    
    subgraph "Agent Pool"
        Orch -->|General Chat| Chat[Chatbot Agent]
        Orch -->|Complex Logic| React[ReAct Agent]
        Orch -->|Math/Logic| Solver[Solver Agent]
        
        React -->|Verify| Ana[Analyzer Agent]
        Solver -->|Verify| Ana
        
        Ana -->|Critique| Refl[Reflective Agent]
        Refl -->|Refine| React
        Refl -->|Refine| Solver
        
        Chat --> Tools[Tools]
        React --> Tools
    end
    
    Chat --> G_Out[Output Guardrails]
    React --> G_Out
    Solver --> G_Out
    Refl --> G_Out
    
    G_Out --> Final([Final Response])
    
    style Orch fill:#f9f,stroke:#333,stroke-width:2px
    style React fill:#bbf,stroke:#333,stroke-width:2px
    style Solver fill:#bbf,stroke:#333,stroke-width:2px
    style Refl fill:#bfb,stroke:#333,stroke-width:2px
```

---

## 📦 System Modules (`src`)

The system is organized into modular packages to ensure separation of concerns.

```mermaid
graph TD
    subgraph "Core Logic"
        Agents[src/agents]
        RAG[src/rag]
        Memory[src/memory]
    end
    
    subgraph "Support Infrastructure"
        Utils[src/utils]
        Eval[src/evaluation]
        Monitor[src/monitoring]
        Scale[src/scaling]
    end
    
    Agents -->|Retrieves Context| RAG
    Agents -->|Reads/Writes| Memory
    
    Agents -->|Uses| Utils
    RAG -->|Uses| Utils
    
    Eval -->|Tests| Agents
    Monitor -->|Observes| Agents
    Monitor -->|Observes| RAG
    
    Agents -.->|Traced By| Scale
```

### Module Breakdown

#### 🧠 [Agents](src/agents/README.md)
The brain of the application.
| Component | Description |
|-----------|-------------|
| **Orchestrator** | Central `LangGraph` state machine that manages the workflow and routing. |
| **ReAct Agent** | Implements the **Reason+Act** paradigm for complex problem solving. |
| **Solver Agent** | Specialized in breaking down logic/math problems. |
| **Chatbot Agent** | Handles general conversational queries and maintains persona. |
| **Reflective Agent** | Performs self-reflection and detailed critique to improve quality. |
| **Guardrails** | Security layers ensuring input safety and output privacy. |

#### 📚 [RAG Pipeline](src/rag/README.md)
Handles document ingestion and context retrieval.
| Component | Description |
|-----------|-------------|
| **Hybrid Retriever** | Combines **BM25** (sparse) and **Vector** (dense) search with weighted fusion. |
| **Vector Store** | Interface for **ChromaDB**. Supports Google & HuggingFace embeddings. |
| **Document Loader** | Parses PDFs and handles text chunking with recursive splitting. |
| **Freshness Tracker** | Monitors source files to only re-index modified documents. |

#### 💾 [Memory Management](src/memory/README.md)
| Component | Description |
|-----------|-------------|
| **Memory Manager** | Unified interface for short-term and persistent memory. |
| **Conversation Buffer** | Manages context windows, automatically summarizing old terms. |

#### 🔭 [Monitoring](src/monitoring/README.md)
| Component | Description |
|-----------|-------------|
| **Drift Detector** | Analyzes query distribution to detect data/concept drift. |
| **Alert Manager** | Routes critical system health alerts to configured channels. |

#### ⚖️ [Scaling](src/scaling/README.md)
| Component | Description |
|-----------|-------------|
| **Request Batcher** | Aggregates multiple API calls to optimize throughput. |
| **Distributed Tracing** | OpenTelemetry integration for full-stack request visibility. |

#### 📊 [Evaluation](src/evaluation/README.md)
| Component | Description |
|-----------|-------------|
| **Metrics** | Implements ROUGE, BLEU, and custom correctness scores. |
| **Continuous Eval** | Automated testing of agent performance against ground truth. |

#### 🛠 [Utils](src/utils/README.md)
| Component | Description |
|-----------|-------------|
| **Config** | Centralized environment and configuration management. |
| **Caching** | Redis-backed response caching for latency reduction. |


---

## 🚀 Installation

### Prerequisites
-   **Python 3.9+**
-   **Docker & Docker Compose** (optional, for containerized deployment)
-   **Google AI API Key** (if using Cloud mode)
-   **Ollama** (if using Local mode)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Math-Agent---LLM.git
cd AI-Math-Agent---LLM/Project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to match your desired setup.

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | None | Required for Cloud Mode (Gemini). |
| `USE_LOCAL` | `false` | Set to `true` to use Ollama instead of Google API. |
| `LOCAL_LLM_MODEL` | `llama3.2:1b` | Ollama model name for text generation. |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers...` | HuggingFace model for embeddings. |
| `LLM_MODEL` | `gemini-pro` | Cloud model name. |
| `RETRIEVER_K` | `3` | Number of documents to retrieve. |
| `BM25_WEIGHT` | `0.5` | Weight for keyword search (0.0 - 1.0). |

---

## ▶️ Running the System

You can run the system in two modes: **Cloud** (using Google Gemini) or **Local** (using Ollama).

### Cloud Mode (Gemini)
Best for performance and reasoning capability.

1.  Ensure `GOOGLE_API_KEY` is set in `.env`.
2.  Set `USE_LOCAL=false`.
3.  Add PDF documents to `data/pdfs/`.
4.  Run the application:
    ```bash
    python main.py
    ```

### Local Mode (Ollama)
Best for privacy and offline usage.

1.  **Install & Start Ollama:**
    Follow instructions at [ollama.com](https://ollama.com).
    ```bash
    ollama serve
    ```
2.  **Pull Required Models:**
    ```bash
    ollama pull llama3.2:1b
    ```
3.  **Configure `.env`:**
    Set `USE_LOCAL=true`.
4.  **Run the application:**
    ```bash
    python main.py
    ```

---

## 🐳 Docker Deployment

To deploy the full microservices stack (Inference, RAG, Observability):

```bash
docker-compose up -d
```

### Service Endpoints
| Service | URL | Description |
|---------|-----|-------------|
| **Inference Service** | `http://localhost:8000` | Main Agent API |
| **RAG Service** | `http://localhost:8001` | Retrieval API |
| **Jaeger** | `http://localhost:16686` | Tracing UI |
| **Grafana** | `http://localhost:3000` | Monitoring Dashboards |

---

## 🧪 Testing

Run quality assurance tests to verify system integrity.

```bash
# Run all tests
pytest tests/ -v

# Run agent-specific tests
pytest tests/test_agents.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📁 Project Structure

```
Project/
├── main.py                      # CLI Entry point
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Dependencies
├── .env.example                 # Config template
│
├── src/
│   ├── agents/                  # [Module] Agent logic & Orchestrator
│   ├── rag/                     # [Module] Retrieval (PDFs + Vector)
│   ├── memory/                  # [Module] Context management
│   ├── monitoring/              # [Module] Drift detection & Alerts
│   ├── evaluation/              # [Module] Metrics & Auto-eval
│   ├── scaling/                 # [Module] Batching & Tracing
│   └── utils/                   # [Module] Config & Caching
│
├── tests/                       # Pytest suite
└── data/                        # PDF storage & persistent DBs
```
