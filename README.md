# Jinsie RAG Support System

Production‑oriented Retrieval‑Augmented Generation (RAG) backend
designed with enterprise AI platform standards and LLMOps principles.

------------------------------------------------------------------------

# 🚀 Positioning

This project demonstrates how to evolve a basic RAG demo into a
production‑style AI backend with:

-   Structured execution pipeline
-   Explicit stage separation
-   Built‑in observability
-   Unified response contract
-   Cloud‑native deployment readiness

Target architecture style aligns with large‑scale AI application
platforms (e.g., Tongyi / enterprise LLM platforms).

------------------------------------------------------------------------

# 🧠 High‑Level Architecture

``` mermaid
flowchart TD

Client[Client / API Consumer]
Client --> API[FastAPI API Layer]

API --> Pipeline[RAG Orchestration Layer]

Pipeline --> Retriever[Retrieval Stage]
Retriever -->|keyword| Keyword[keyword_retrieve]
Retriever -->|vector| Vector[vector_retrieve]

Retriever --> Context[Context Builder]
Context --> LLM[LLM Inference Layer]

LLM --> Result[Structured Result]
Result --> Metrics[Metrics Collector]

Metrics --> API
API --> Response[Unified Envelope Response]
```

------------------------------------------------------------------------

# 🏗 Layered Design

## 1️⃣ API Layer

File: `app/main.py`

Responsibilities:

-   REST interface (`/v1/rag/run`)
-   Unified response envelope
-   Standardized error handling
-   Request ID management

Response contract:

``` json
{
  "success": true,
  "request_id": "uuid",
  "data": { ... }
}
```

------------------------------------------------------------------------

## 2️⃣ Orchestration Layer

File: `app/rag/pipeline.py`

Pipeline stages:

1.  Retrieval
2.  Context augmentation
3.  LLM inference
4.  Metrics collection

Characteristics:

-   Explicit stage boundaries
-   Pluggable execution model
-   Deterministic return shape

------------------------------------------------------------------------

## 3️⃣ Retrieval Layer

Supports:

-   Keyword retrieval (fast, deterministic)
-   Vector retrieval (semantic, embedding‑based)

Designed for extension to:

-   Hybrid retrieval
-   Reranking
-   External vector DB

------------------------------------------------------------------------

## 4️⃣ Inference Layer

-   OpenAI‑compatible Chat Completion interface
-   Model‑agnostic
-   Compatible with Qwen / Tongyi‑style APIs
-   Supports token usage reporting

------------------------------------------------------------------------

## 5️⃣ Observability Layer

Each request exposes latency metrics:

``` json
"metrics": {
  "total_ms": 1382.4,
  "retrieval_ms": 1.3,
  "llm_ms": 1381.1
}
```

Meaning:

-   `retrieval_ms`: document retrieval latency
-   `llm_ms`: model inference latency
-   `total_ms`: end‑to‑end request latency

This enables:

-   Bottleneck identification
-   Performance regression detection
-   Basic LLMOps observability

------------------------------------------------------------------------

# 📦 Endpoint Specification

### POST `/v1/rag/run`

Request:

``` json
{
  "query": "string",
  "retrieval_mode": "keyword | vector",
  "top_k": 3,
  "include_context": false
}
```

Response:

``` json
{
  "success": true,
  "request_id": "uuid",
  "data": {
    "answer": { ... },
    "retrieval": {
      "mode": "keyword | vector",
      "hit_count": 3,
      "doc_ids": ["md_1", "md_2"]
    },
    "metrics": {
      "total_ms": 1382.4,
      "retrieval_ms": 1.3,
      "llm_ms": 1381.1
    }
  }
}
```

------------------------------------------------------------------------

# 🌩 Cloud‑Native Readiness

Designed to be:

-   Containerizable (Docker‑ready)
-   Deployable on ECS / Kubernetes
-   Compatible with API gateway layers
-   Extensible with:
    -   Logging aggregation
    -   Rate limiting
    -   Distributed tracing

------------------------------------------------------------------------

# 🎯 Engineering Highlights

-   Clear separation of concerns
-   Deterministic execution flow
-   Unified response contract
-   Built‑in latency observability
-   Production‑style structure without over‑engineering

------------------------------------------------------------------------

# 🔮 Roadmap (Structured Evolution)

-   Request‑level tracing
-   Structured logging pipeline
-   Hybrid retrieval strategy
-   LangGraph‑based orchestration upgrade
-   Distributed deployment mode

------------------------------------------------------------------------

# 📌 Summary

This backend reflects a progression:

Simple RAG → Structured Backend → Observable AI Service → Platform‑ready
Architecture

It emphasizes engineering maturity, scalability boundaries, and
operational visibility.