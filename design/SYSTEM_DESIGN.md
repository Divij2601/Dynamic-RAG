# Dynamic-RAG — System Design

## 1. Purpose

This document defines the technical implementation blueprint for **Dynamic-RAG**, an evaluation-first adaptive retrieval-augmented generation system.

The purpose of this document is to translate the architecture and roadmap into a concrete software design that can be implemented reliably, tested independently, and extended without architectural drift.

Dynamic-RAG is designed to:
- route queries intelligently,
- retrieve the right evidence,
- generate grounded answers,
- verify answer faithfulness,
- preserve memory appropriately,
- expose observability data,
- support benchmarking and regression testing,
- remain production-ready.

This document is the implementation reference for backend development.

---

## 2. Design Identity

Dynamic-RAG is not a simple chat application with retrieval attached.

It is a modular knowledge system built around four technical commitments:

1. **Adaptive control flow**
   - Different queries may require different execution paths.

2. **Evaluation-first engineering**
   - Retrieval quality, generation faithfulness, and system behavior must be measurable.

3. **Evidence-grounded output**
   - Answers should be supported by retrieved or researched evidence.

4. **Operational accountability**
   - Every important request decision should be traceable.

These commitments affect every module in the system.

---

## 3. Design Objectives

### 3.1 Correctness
The system should maximize answer correctness through strong retrieval, faithful generation, and verification.

### 3.2 Measurability
Every major subsystem should emit metrics and traces.

### 3.3 Modularity
Each component should be replaceable independently.

### 3.4 Persistence
Indexed documents, sessions, memory, and traces should survive restarts.

### 3.5 Scalability
The design should support growth in documents, users, and request volume.

### 3.6 Debuggability
Failures should be localizable to a specific stage of the pipeline.

---

## 4. Technology Stack

The first implementation should use the following stack.

### Backend
- FastAPI
- Uvicorn
- Pydantic
- LangGraph
- LangChain

### Retrieval and AI
- Qdrant
- sentence-transformers or equivalent embedding provider
- BM25 or equivalent sparse retrieval component
- reranker model
- OpenAI-compatible LLM or equivalent provider
- Tavily or another web search API for external evidence

### Storage
- MongoDB for sessions, memory, logs, and traces
- Qdrant for embeddings and vector metadata
- local file storage for raw uploads and artifacts

### Observability
- structured logging
- timing hooks
- token accounting
- trace persistence
- benchmark export

---

## 5. Repository Structure

A clean modular structure is required.

```text
repo/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── dependencies.py
│   │
│   ├── agents/
│   │   ├── planner.py
│   │   ├── retriever_agent.py
│   │   ├── web_research_agent.py
│   │   ├── critic_agent.py
│   │   ├── formatter_agent.py
│   │   └── memory_agent.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── metadata.py
│   │   ├── embedder.py
│   │   └── indexer.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   └── evidence.py
│   │
│   ├── memory/
│   │   ├── session_store.py
│   │   ├── semantic_memory.py
│   │   ├── summarizer.py
│   │   └── policy.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── router.py
│   │   └── graph_builder.py
│   │
│   ├── database/
│   │   ├── qdrant_client.py
│   │   ├── mongo_client.py
│   │   ├── repositories.py
│   │   └── schemas.py
│   │
│   ├── observability/
│   │   ├── logger.py
│   │   ├── tracing.py
│   │   ├── metrics.py
│   │   └── audit.py
│   │
│   ├── evaluation/
│   │   ├── datasets.py
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   ├── system_metrics.py
│   │   ├── offline_eval.py
│   │   └── regression.py
│   │
│   ├── config.py
│   └── utils/
│       ├── text.py
│       ├── ids.py
│       ├── time.py
│       └── constants.py
│
├── docs/
├── tests/
├── configs/
├── notebooks/
├── .env
├── requirements.txt
└── README.md
```

---

## 6. Core Runtime Model

Dynamic-RAG should operate as a stateful request-processing system.

### The runtime flow
1. API receives request.
2. Context is loaded.
3. Planner decides route.
4. Selected route executes.
5. Evidence is assembled.
6. Generator creates answer.
7. Critic verifies answer.
8. Formatter prepares final payload.
9. Logger and metric collector store the trace.

The runtime should be implemented as a graph or state machine so each step is explicit and inspectable.

---

## 7. Key Data Objects

The system should rely on explicit structured objects.

### 7.1 QueryState
Represents the full state of one request.

Suggested fields:
- request_id
- session_id
- query_text
- chat_history
- memory_context
- planner_output
- selected_route
- internal_evidence
- web_evidence
- candidate_answer
- verification_result
- final_response
- metrics
- error

### 7.2 PlannerOutput
Represents route selection and budget planning.

Suggested fields:
- intent
- complexity
- route
- confidence
- needs_retrieval
- needs_web
- needs_decomposition
- subqueries
- budget

### 7.3 EvidenceItem
Represents one supporting evidence unit.

Suggested fields:
- source_type
- source_id
- chunk_id
- page
- section
- text
- score
- metadata

### 7.4 RetrievalResult
Represents retrieval output.

Suggested fields:
- mode
- dense_candidates
- sparse_candidates
- merged_candidates
- reranked_candidates
- top_k
- retrieval_confidence

### 7.5 VerificationResult
Represents critic output.

Suggested fields:
- faithful
- supported
- complete
- issues
- retry_required
- severity

### 7.6 FinalResponse
Represents the response returned to the client.

Suggested fields:
- answer
- sources
- route
- confidence
- status
- query_id
- session_id

---

## 8. API Design

The API layer should remain thin and delegate all business logic.

## 8.1 Health Endpoint

### `GET /health`
Returns service status.

Example:
```json
{
  "status": "ok"
}
```

## 8.2 Document Upload

### `POST /documents/upload`

Responsibilities:
- validate file type,
- store raw file,
- trigger ingestion,
- return document metadata.

## 8.3 Query Endpoint

### `POST /chat/query`

Responsibilities:
- load context,
- execute graph,
- return final response.

## 8.4 Session Endpoint

### `GET /chat/{session_id}`
Returns session history and metadata.

## 8.5 Source Endpoint

### `GET /query/{query_id}/sources`
Returns evidence references associated with a query.

## 8.6 Metrics Endpoint

### `GET /system/metrics`
Returns high-level operational metrics.

---

## 9. LangGraph Workflow Design

LangGraph should serve as the orchestration layer.

## 9.1 Graph Nodes

### Node 1 — Context Loader
Loads:
- chat history
- semantic memory
- document scope

### Node 2 — Planner
Classifies query and selects route.

### Node 3 — Route Dispatcher
Branches to the appropriate path.

### Node 4A — Internal Retriever
Retrieves evidence from Qdrant and supporting indexes.

### Node 4B — Web Researcher
Collects external evidence when needed.

### Node 4C — General Reasoner
Handles direct reasoning without retrieval.

### Node 5 — Evidence Assembler
Normalizes retrieved material into a consistent evidence bundle.

### Node 6 — Generator
Produces the candidate answer.

### Node 7 — Critic
Evaluates faithfulness and completeness.

### Node 8 — Retry Controller
Retries, rewrites, or aborts based on critic outcome.

### Node 9 — Formatter
Produces the final structured response.

### Node 10 — Logger / Metrics Sink
Persists traces, metrics, and audit data.

---

## 9.2 State Transition Rules

The graph should obey these rules:

1. Every node has a defined input and output schema.
2. Every failure path is explicit.
3. Retry count is bounded.
4. Verification failure cannot be ignored.
5. Final response cannot be emitted without a terminal state.

---

## 10. Agent Design

## 10.1 Planner Agent

### Purpose
Select the best route for a query.

### Responsibilities
- classify intent,
- estimate complexity,
- detect freshness requirements,
- decide whether retrieval is needed,
- decide whether web search is needed,
- set a budget,
- generate subqueries when needed.

### Output
Structured planner output only.

---

## 10.2 Retriever Agent

### Purpose
Fetch relevant internal evidence.

### Responsibilities
- hybrid retrieval,
- reranking,
- metadata filtering,
- evidence normalization,
- confidence estimation.

### Output
A ranked evidence bundle with source metadata.

---

## 10.3 Web Research Agent

### Purpose
Fetch external evidence when freshness or broader coverage is required.

### Responsibilities
- build search queries,
- retrieve web results,
- normalize snippets,
- preserve source references.

---

## 10.4 Critic Agent

### Purpose
Verify that the answer is supported by evidence.

### Responsibilities
- detect unsupported claims,
- score faithfulness,
- check citation alignment,
- measure completeness,
- request retry or abstention.

### Design Requirement
This agent is the enforcement mechanism for grounded output.

---

## 10.5 Formatter Agent

### Purpose
Prepare the final answer payload.

### Responsibilities
- format answer text,
- attach sources,
- attach route metadata,
- attach confidence,
- produce client-ready output.

---

## 10.6 Memory Agent

### Purpose
Manage session continuity and semantic memory.

### Responsibilities
- retrieve recent turns,
- summarize long sessions,
- retrieve relevant semantic memory,
- inject memory selectively.

---

## 11. Retrieval System Design

Retrieval must be hybrid and measurable.

## 11.1 Dense Retrieval
Used for semantic similarity.

## 11.2 Sparse Retrieval
Used for lexical precision and exact-match terms.

## 11.3 Hybrid Fusion
Combines dense and sparse scores.

## 11.4 Reranker
Reorders candidates using a stronger relevance model.

## 11.5 Evidence Builder
Converts candidate chunks into structured evidence items.

### Retrieval Output Requirements
The retrieval layer must provide:
- top-k candidates,
- scores,
- source metadata,
- retrieval mode,
- retrieval confidence.

---

## 12. Ingestion Pipeline Design

The ingestion pipeline must be deterministic and metadata-rich.

### Stages
1. file validation
2. extraction
3. normalization
4. chunking
5. metadata enrichment
6. embedding generation
7. indexing
8. persistence

### Chunk Requirements
Each chunk should preserve:
- document ID,
- filename,
- version,
- page or section,
- hash,
- upload timestamp,
- tags.

### Why This Matters
The retrieval layer depends on structured ingestion quality. Weak chunking creates weak retrieval metrics.

---

## 13. Storage Design

## 13.1 Qdrant
Primary storage for:
- embeddings,
- chunk metadata,
- filtered retrieval,
- persistent document search.

### Payload fields
- doc_id
- chunk_id
- file_name
- page
- section
- version
- text
- tags
- owner_id
- upload_time
- source_type
- hash

## 13.2 MongoDB
Stores:
- sessions,
- chat messages,
- semantic memory,
- traces,
- ingestion state,
- evaluation summaries.

## 13.3 File Storage
Stores:
- raw uploads,
- local artifacts,
- benchmark files,
- reports.

---

## 14. Data Model Design

### 14.1 SessionRecord
- session_id
- summary
- start_time
- last_updated
- active_doc_scope

### 14.2 MessageRecord
- message_id
- session_id
- role
- content
- timestamp

### 14.3 MemoryRecord
- memory_id
- type
- content
- relevance
- created_at

### 14.4 TraceRecord
- trace_id
- request_id
- route
- latency
- cost
- faithfulness
- groundedness
- retry_count
- status

### 14.5 DocumentRecord
- doc_id
- filename
- version
- status
- ingestion_time
- indexing_status

---

## 15. Observability Design

Every request must produce a trace.

### Required Fields
- request_id
- session_id
- route
- retrieval_mode
- retrieval_latency
- rerank_latency
- generation_latency
- faithfulness_score
- groundedness_score
- retry_count
- token_usage
- total_cost
- final_status

### Logs Must Support
- debugging,
- regression analysis,
- performance analysis,
- benchmark correlation.

---

## 16. Evaluation Design

Evaluation is built into the system design.

### Plane 1 — Retrieval
Metrics:
- Context Recall
- Context Precision
- Recall@K
- MRR
- NDCG@K
- Hit Rate
- retrieval latency

### Plane 2 — Generation
Metrics:
- Faithfulness
- Answer Relevance
- Groundedness
- Citation Accuracy
- Completeness
- Noise Robustness
- Counterfactual Robustness

### Plane 3 — System
Metrics:
- end-to-end accuracy
- rejection rate
- cost per query
- latency percentiles
- retry frequency
- failure rate

### Evaluation Modules
The evaluation package should support:
- dataset loading,
- metric computation,
- regression comparison,
- benchmark reports,
- adversarial testing.

---

## 17. Error Handling Design

### Error Classes
- validation error
- ingestion error
- retrieval error
- generation error
- verification error
- external API failure
- database failure
- timeout
- configuration failure

### Handling Rules
- classify errors explicitly,
- log errors with context,
- return safe responses,
- do not silently suppress failures.

---

## 18. Performance Design

### Priorities
1. reduce unnecessary route cost,
2. minimize retrieval noise,
3. rerank efficiently,
4. avoid prompt bloat,
5. keep retries bounded,
6. maintain low tail latency.

### Metrics to Track
- P50 latency
- P95 latency
- P99 latency
- cost per query
- retries per query

---

## 19. Security and Safety Design

### Security
- no hardcoded secrets,
- validate all uploads,
- sanitize inputs,
- control external API usage,
- preserve data isolation where relevant.

### Safety
- abstain when confidence is too low,
- do not fabricate sources,
- do not bypass verification,
- do not return unsupported claims.

---

## 20. Testing Strategy

### Unit Tests
- planner output,
- chunking,
- retrieval scoring,
- verification behavior,
- formatter output.

### Integration Tests
- document ingestion to retrieval,
- query to answer,
- query to verification retry,
- session continuity.

### Evaluation Tests
- retrieval benchmark runs,
- generation faithfulness runs,
- system-level metric tracking,
- regression comparisons.

---

## 21. Deployment Design

### Local
- FastAPI backend
- MongoDB
- Qdrant
- .env configuration

### Production-Ready Goals
- containerized backend,
- persistent databases,
- restart safety,
- trace persistence,
- reproducible environment setup.

---

## 22. Design Constraints

1. API layer must remain thin.
2. Retrieval and generation must stay separate.
3. Verification must be independent from generation.
4. Memory must be selective.
5. Observability must be mandatory.
6. Evaluation must be versioned and reproducible.
7. Every route decision must be explicit.
8. Every failure must be traceable.

---

## 23. Definition of a Well-Designed System

Dynamic-RAG is well designed only if:
- queries are routed intelligently,
- retrieval is measurable,
- answers are grounded,
- verification works,
- memory is controlled,
- observability is complete,
- evaluation is reproducible,
- failures are localizable.

---

## 24. Final Statement

Dynamic-RAG is an adaptive, evaluation-first RAG platform with explicit orchestration, persistent storage, bounded verification loops, and strong observability.

Its implementation should make the system explainable in operational terms:
- what was retrieved,
- why it was retrieved,
- how it was used,
- whether it was faithful,
- how much it cost,
- how fast it ran,
- where it failed.

That is the technical identity of the system.
