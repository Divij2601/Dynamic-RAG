# Dynamic-RAG — System Design

## 1. Purpose

This document defines the technical design of **Dynamic-RAG**, a modular adaptive retrieval-augmented generation system built with explicit orchestration, persistent storage, verification, and multi-agent execution.

The purpose of this document is to describe how the system should be implemented in code. It translates the project architecture and roadmap into concrete engineering structure:
- module boundaries,
- class responsibilities,
- state models,
- database schemas,
- API contracts,
- workflow transitions,
- failure handling,
- deployment assumptions.

This document should be treated as the implementation reference for backend development and agentic coding.

---

## 2. Design Objectives

Dynamic-RAG is designed to satisfy the following technical objectives:

1. **Adaptive execution**
   - Route each query to the correct reasoning path.

2. **Grounded answer generation**
   - Generate answers from evidence whenever possible.

3. **Persistent retrieval**
   - Store and retrieve embeddings reliably across restarts.

4. **Verification-first response flow**
   - Validate candidate answers before release.

5. **Multi-agent separation**
   - Keep planning, retrieval, generation, verification, and formatting isolated.

6. **Operational visibility**
   - Log and trace route selection, latency, and quality signals.

7. **Modular codebase**
   - Support future replacement of any subsystem without full rewrite.

---

## 3. Technology Stack

The initial implementation should use the following stack:

### Backend
- **FastAPI** for HTTP APIs
- **Uvicorn** as ASGI server
- **Pydantic** for request/response schemas
- **LangGraph** for workflow orchestration
- **LangChain** for model and retriever integrations

### AI / Retrieval
- **GROQ-compatible LLM**
- **Sentence-transformers or embedding provider**
- **Qdrant** as primary vector store
- **BM25** for sparse retrieval
- **Reranker model** for candidate reordering
- **Tavily or equivalent search API** for web research

### Storage
- **MongoDB** for sessions, memory, and logs
- **Qdrant** for document embeddings and payload metadata
- Optional local filesystem for raw documents and debugging artifacts

### Observability
- structured logging,
- request tracing,
- token and latency measurement,
- evaluation hooks.

---

## 4. System Boundary

Dynamic-RAG is a backend-first application. The frontend is a thin client.

### In scope
- document upload,
- indexing,
- query orchestration,
- retrieval,
- answer generation,
- verification,
- memory management,
- source exposure,
- logging,
- metrics.

### Out of scope for core backend design
- model training,
- distributed orchestration clusters,
- custom UI rendering logic,
- streaming media processing,
- unrelated automation workflows.

---

## 5. Repository Structure

The codebase should be organized into clean modules.

```text
repo/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── documents.py
│   │   │   ├── health.py
│   │   │   └── metrics.py
│   │   ├── schemas/
│   │   │   ├── chat.py
│   │   │   ├── documents.py
│   │   │   ├── common.py
│   │   │   └── responses.py
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
│   │   └── memory_policy.py
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
│   │   ├── metrics.py
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

## 6. Core Runtime Flow

The runtime flow is the same for every query, though the selected route changes.

### Request flow
1. API receives query.
2. Session and memory context are loaded.
3. Planner agent analyzes the request.
4. Route is selected.
5. Relevant execution path runs.
6. Evidence is assembled.
7. Generator produces candidate answer.
8. Critic verifies response.
9. Formatter prepares the final payload.
10. Result is returned and logged.

This flow should be implemented through LangGraph or an equivalent graph-based state machine.

---

## 7. Main Data Objects

The system should rely on explicit data structures rather than ad hoc dictionaries wherever possible.

### 7.1 QueryState
Represents the full state of one user request.

Suggested fields:
- `query_id`
- `session_id`
- `user_query`
- `chat_history`
- `memory_context`
- `planner_output`
- `route`
- `retrieved_evidence`
- `web_evidence`
- `candidate_answer`
- `critic_result`
- `final_answer`
- `metadata`
- `error`

### 7.2 PlannerOutput
Represents route selection output.

Suggested fields:
- `intent`
- `complexity`
- `route`
- `needs_retrieval`
- `needs_web`
- `needs_decomposition`
- `confidence`
- `subqueries`
- `budget`

### 7.3 EvidenceItem
Represents a unit of supporting evidence.

Suggested fields:
- `source_type`
- `source_id`
- `chunk_id`
- `page`
- `text`
- `score`
- `metadata`

### 7.4 RetrievalResult
Represents retrieval output.

Suggested fields:
- `retrieval_mode`
- `dense_results`
- `sparse_results`
- `merged_results`
- `reranked_results`
- `top_k`
- `confidence`

### 7.5 VerificationResult
Represents critic output.

Suggested fields:
- `faithful`
- `supported`
- `completeness`
- `issues`
- `retry_required`
- `severity`

### 7.6 FinalResponse
Represents response sent to the user.

Suggested fields:
- `answer`
- `sources`
- `route`
- `confidence`
- `status`
- `session_id`
- `query_id`

---

## 8. API Design

## 8.1 Health Endpoint

### `GET /health`
Returns service status.

Example response:
```json
{
  "status": "ok"
}
```

---

## 8.2 Document Upload

### `POST /documents/upload`

Accepts:
- file,
- optional description,
- optional metadata.

Responsibilities:
- validate file type,
- store raw file,
- parse text,
- chunk content,
- embed chunks,
- persist vectors and metadata,
- return document ID.

---

## 8.3 Query Endpoint

### `POST /chat/query`

Accepts:
- user query,
- session ID,
- optional client metadata.

Responsibilities:
- load memory,
- plan route,
- execute selected flow,
- verify output,
- return structured answer.

---

## 8.4 Session Endpoint

### `GET /chat/{session_id}`
Returns session history and metadata.

---

## 8.5 Sources Endpoint

### `GET /query/{query_id}/sources`
Returns cited sources or evidence items associated with a query.

---

## 8.6 Metrics Endpoint

### `GET /system/metrics`
Returns system health and operational statistics.

---

## 9. LangGraph State Machine Design

Dynamic-RAG should use LangGraph for explicit control flow.

## 9.1 Graph Nodes

### Node 1: Context Loader
Loads:
- chat history,
- semantic memory,
- document scope.

### Node 2: Planner
Classifies the query and sets the route.

### Node 3: Route Dispatcher
Sends the state to the appropriate branch.

### Node 4a: Internal Retriever
Retrieves from Qdrant and supporting indexes.

### Node 4b: Web Researcher
Fetches external evidence.

### Node 4c: General Reasoner
Handles direct non-retrieval responses.

### Node 5: Evidence Assembler
Consolidates selected evidence into a uniform input format.

### Node 6: Generator
Produces the candidate answer.

### Node 7: Critic
Verifies faithfulness and completeness.

### Node 8: Retry Controller
Re-routes if verification fails.

### Node 9: Formatter
Builds final response payload.

### Node 10: Logger
Stores traces and request summary.

---

## 9.2 State Transitions

Example flow:

```text
load_context
→ plan
→ route_dispatch
→ retrieve / web_research / direct_reason
→ assemble_evidence
→ generate
→ critic
→ [pass] formatter
→ [fail] retry_controller → retrieve/rewrite/generate
```

### Retry Policy
- maximum retries should be bounded,
- repeated failure should lead to abstention or safe fallback,
- retry conditions should be explicit.

---

## 10. Agent Design

## 10.1 Planner Agent

### Purpose
Select the best route for a query.

### Inputs
- query,
- memory,
- session context,
- available collections,
- optional document scope.

### Outputs
- route,
- complexity,
- need for retrieval,
- need for web search,
- subqueries,
- confidence.

### Design Constraint
The planner must return structured output. Free-form text is not acceptable for routing decisions.

---

## 10.2 Retriever Agent

### Purpose
Find relevant internal evidence.

### Subsystems Used
- dense retriever,
- sparse retriever,
- fusion module,
- reranker,
- evidence normalizer.

### Behavior
- fetch candidate chunks,
- re-score them,
- return evidence bundle,
- handle empty result cases.

---

## 10.3 Web Research Agent

### Purpose
Fetch fresh external evidence.

### Behavior
- create search queries,
- execute search,
- normalize snippets,
- optionally summarize source content,
- return structured evidence.

### Constraint
This agent should never be called unless the planner or route rules require it.

---

## 10.4 Critic Agent

### Purpose
Assess whether the generated answer is supported.

### Checks
- evidence coverage,
- unsupported claims,
- contradiction risk,
- missing important details,
- answer completeness,
- source alignment.

### Output
- pass/fail,
- issue list,
- retry recommendation.

---

## 10.5 Formatter Agent

### Purpose
Render the final response into a consistent schema and presentation format.

### Responsibilities
- clean answer,
- attach sources,
- attach metadata,
- enforce output style.

---

## 10.6 Memory Agent

### Purpose
Manage both session memory and semantic memory.

### Responsibilities
- store recent turns,
- summarize older turns,
- retrieve relevant past context,
- inject only relevant memory into the request state.

---

## 11. Retrieval System Design

The retrieval system should be hybrid and layered.

### 11.1 Dense Retrieval
Used for semantic similarity.

#### Input
- query embedding,
- Qdrant vectors.

#### Output
- top semantic candidates.

### 11.2 Sparse Retrieval
Used for exact keywords and rare entity matching.

#### Input
- query tokens,
- lexical index.

#### Output
- keyword-matched candidates.

### 11.3 Fusion Layer
Combines dense and sparse results.

### 11.4 Reranker
Applies stronger relevance scoring on the fused candidate set.

### 11.5 Evidence Builder
Normalizes and deduplicates final chunks into evidence objects.

### Retrieval Requirements
- stable output schema,
- source metadata preserved,
- ranking scores available,
- filter support by document scope and metadata.

---

## 12. Ingestion Pipeline Design

The ingestion pipeline should be a deterministic preprocessing pipeline.

### Stages
1. file validation,
2. extraction,
3. normalization,
4. chunking,
5. metadata enrichment,
6. embedding,
7. indexing,
8. verification.

### Chunk Strategy
Chunks should be:
- semantically coherent,
- not too small,
- not excessively large,
- annotated with page and source metadata when available.

### Metadata Strategy
Each chunk should store:
- document ID,
- file name,
- version,
- page,
- section,
- upload time,
- owner scope,
- hash.

### Failure Handling
If ingestion fails at any stage:
- record the reason,
- avoid partial silent success,
- allow safe retry or cleanup.

---

## 13. Database Design

## 13.1 Qdrant Collections

### Collection Name
`dynamic_rag_documents`

### Vector Schema
- cosine distance,
- embedding dimension determined by chosen embedding model.

### Payload Schema
Suggested fields:
- `doc_id`
- `chunk_id`
- `file_name`
- `page`
- `section`
- `version`
- `text`
- `tags`
- `upload_time`
- `owner_id`
- `source_type`
- `hash`

### Use Cases
- semantic search,
- metadata filtering,
- source traceability,
- version-aware retrieval.

---

## 13.2 MongoDB Collections

### `sessions`
Stores:
- session metadata,
- summary,
- active scope.

### `messages`
Stores:
- turn-by-turn conversation history.

### `memory`
Stores:
- long-term semantic memory entries,
- normalized preference and fact records.

### `traces`
Stores:
- request logs,
- route data,
- latency and token data,
- critic outcomes.

### `documents`
Stores:
- document metadata,
- ingestion status,
- indexing state.

---

## 14. Configuration Design

Configuration should be centralized.

### Sources
- `.env`
- environment variables
- optional config profiles

### Required Settings
- GROQ API key or provider key,
- Qdrant URL and API key,
- MongoDB URI,
- web search API key,
- embedding model name,
- reranker model name,
- logging level,
- retry limits,
- chunk size,
- overlap size.

### Design Requirement
No secrets should be committed to the repository.

---

## 15. Error Handling Design

Dynamic-RAG should use explicit error classification.

### Error Types
- validation error,
- ingestion error,
- retrieval error,
- generation error,
- verification error,
- external API error,
- database error,
- configuration error.

### Behavior
- log the error,
- classify the error,
- return safe API response,
- preserve diagnostic context.

### Response Strategy
- user-caused errors should be explained clearly,
- system failures should not leak sensitive internals,
- unsupported or unsafe requests should be handled gracefully.

---

## 16. Logging and Tracing Design

Every request should emit a structured trace.

### Trace Fields
- request ID,
- session ID,
- route,
- planner output,
- retrieval stats,
- generation latency,
- verification verdict,
- final status,
- token counts,
- failure state.

### Logging Levels
- DEBUG for development,
- INFO for normal operation,
- WARNING for recoverable issues,
- ERROR for failures,
- CRITICAL for unrecoverable failures.

### Why This Matters
Without trace data, evaluation and debugging become guesswork.

---

## 17. Evaluation Design

The system should be evaluated independently of the UI.

### Evaluation Dimensions
- retrieval relevance,
- answer faithfulness,
- answer completeness,
- routing accuracy,
- latency,
- cost efficiency,
- retry frequency.

### Evaluation Assets
- benchmark question set,
- expected source set,
- offline replay runner,
- regression comparison script.

### Required Output
Evaluation should produce measurable, repeatable results.

---

## 18. Security and Safety Design

### Security Principles
- do not expose secrets,
- validate uploaded files,
- sanitize inputs,
- restrict unsafe operations,
- isolate document scopes if multi-user support is added.

### Safety Principles
- abstain when confidence is too low,
- do not fabricate sources,
- do not return unsupported claims,
- use verification before response release.

---

## 19. Performance Design

### Performance Targets
The exact targets can be refined later, but the system should aim for:
- low latency on simple queries,
- bounded retry loops,
- efficient retrieval,
- minimal unnecessary LLM calls,
- reusable cached context where appropriate.

### Optimization Priorities
1. reduce unnecessary routing cost,
2. reduce retrieval candidate volume,
3. rerank efficiently,
4. minimize prompt bloat,
5. cache stable artifacts where safe.

---

## 20. Deployment Design

The system should support local development first and containerized deployment later.

### Local Development
- FastAPI on local host,
- Qdrant and MongoDB locally or via managed services,
- `.env` configuration.

### Containerized Deployment
- backend container,
- database containers or managed endpoints,
- environment-based configuration,
- health checks.

### Deployment Requirement
The system should recover cleanly after restart without losing indexed data or session state.

---

## 21. Testing Strategy

The codebase should include tests for each layer.

### Unit Tests
- planner output parsing,
- chunking behavior,
- retrieval scoring,
- verification decisions,
- formatter output.

### Integration Tests
- document ingestion to retrieval,
- query to answer,
- query to verification retry,
- session memory continuity.

### Regression Tests
- route selection stability,
- retrieval quality stability,
- source attachment correctness.

---

## 22. Design Constraints

The following constraints should be respected throughout implementation:

1. Keep API handlers thin.
2. Keep agent responsibilities narrow.
3. Keep retrieval evidence separate from final wording.
4. Keep verification independent from generation.
5. Keep memory selective.
6. Keep route decisions structured.
7. Keep observability mandatory.
8. Keep storage persistent.
9. Keep failure handling explicit.
10. Keep the codebase modular.

---

## 23. Definition of a Well-Designed System

Dynamic-RAG is well designed only if:
- every query has a visible route,
- every answer has an evidence basis,
- every retrieval path is inspectable,
- every failure is logged,
- every component is replaceable,
- every major decision is explicit.

That is the technical identity of the system.

---

## 24. Final Design Statement

Dynamic-RAG is a modular adaptive retrieval and reasoning platform built around planning, retrieval, verification, and memory. The system is intentionally structured to choose the correct path per query rather than forcing a single one-size-fits-all pipeline.

This design is meant to support implementation from scratch, future extension, and production hardening without architectural drift.
