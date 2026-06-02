# PROJECT ARCHITECTURE

## 1. Overview

This project is a production-oriented **Dynamic RAG system** built as a modular, multi-agent knowledge application. The system accepts natural-language questions, determines the most appropriate reasoning path, retrieves relevant evidence when needed, optionally researches external sources when the query depends on freshness or web context, generates grounded answers, and verifies those answers before returning them to the user.

The architecture is designed around a central principle:

> **Do not force every query through the same pipeline.**
>  
> Instead, route each query to the lightest reliable path that can answer it correctly.

That principle drives the system design. Some queries require only reasoning over prior conversation or internal memory. Some require retrieval from uploaded documents. Some require live web research. Some require multi-step evidence gathering and answer verification. The architecture must support all of these cases without becoming brittle, slow, or difficult to extend.

This document defines the project from first principles so that the codebase can be implemented, extended, maintained, and evaluated in a disciplined way.

---

## 2. Project Vision

The system aims to become a high-trust assistant for document-grounded and knowledge-grounded question answering. It should be able to:

- answer questions from uploaded documents,
- support conversations across multiple turns,
- selectively use retrieval only when it is needed,
- fetch fresh information from the web when required,
- cite or expose supporting evidence,
- detect unsupported or low-confidence answers,
- log behavior for debugging and evaluation,
- scale from local development to production deployment.

The long-term goal is not merely to build a chatbot. The goal is to build a **query-orchestrated knowledge system** where routing, retrieval, generation, verification, and memory are separate concerns with explicit interfaces.

---

## 3. Design Goals

### 3.1 Accuracy
The system should favor grounded answers over fluent but unsupported answers. Retrieval and verification must reduce hallucination risk.

### 3.2 Adaptivity
The system should choose the answer path dynamically based on query intent, complexity, and freshness requirements.

### 3.3 Modularity
Each subsystem must be replaceable independently. The project should not depend on one monolithic chain.

### 3.4 Observability
Every important step in the request path should be measurable and traceable: route selection, retrieval quality, latency, token usage, verification outcomes, and failure modes.

### 3.5 Persistence
Documents, embeddings, session memory, and operational traces must persist beyond process restarts.

### 3.6 Extensibility
The codebase should allow future addition of new retrieval strategies, new agents, new memory types, and new storage backends without major rewrites.

### 3.7 Production Readiness
The project must support API separation, configuration management, logging, error handling, rate limiting hooks, and safe deployment patterns.

---

## 4. Core Architectural Principles

### 4.1 Separate Planning from Execution
The system first decides *what to do*, then performs the chosen task. Planning and execution must not be collapsed into a single opaque LLM call.

### 4.2 Separate Retrieval from Generation
Retrieval produces evidence. Generation turns evidence into an answer. These are different problems and should remain isolated.

### 4.3 Separate Evidence from Memory
Conversation memory is not the same as document knowledge. The system must treat them as distinct data sources.

### 4.4 Verify Before Responding
Generation alone is not enough. A verification layer should assess faithfulness and completeness before the final response is released.

### 4.5 Keep Route Decisions Explicit
Every request should have a visible route outcome such as `internal_rag`, `web_research`, or `general_reasoning`.

### 4.6 Prefer Structured Outputs
Wherever the system produces internal decisions, it should use structured objects instead of free-form text.

### 4.7 Build for Reusability
All core logic should be packaged into reusable modules instead of being embedded directly in API endpoints.

---

## 5. System Scope

### 5.1 In Scope
- document upload and indexing,
- query routing,
- hybrid retrieval,
- multi-stage retrieval refinement,
- answer generation,
- verification and retry loops,
- web research path,
- short-term and long-term memory,
- logging and observability,
- API-based interaction,
- UI integration,
- configuration via environment variables and config files.

### 5.2 Out of Scope for the First Version
- voice interfaces,
- multimodal image understanding,
- fully autonomous tool execution beyond defined agents,
- heavy workflow automation unrelated to RAG,
- arbitrary plugin execution,
- user-generated code execution,
- multi-tenant enterprise permission systems beyond foundational support,
- training a foundation language model from scratch.

The architecture should allow these later, but they are not required to achieve the first stable release.

---

## 6. High-Level Architecture

The system is organized into the following layers:

1. **Interface Layer**
   - FastAPI endpoints
   - Streamlit or web frontend
   - Request validation

2. **Orchestration Layer**
   - Planner agent
   - LangGraph workflow
   - route selection
   - retry control

3. **Knowledge Layer**
   - document ingestion
   - chunking
   - embeddings
   - Qdrant vector store
   - hybrid search
   - reranking

4. **Reasoning Layer**
   - generation agent
   - web research agent
   - general reasoning path

5. **Verification Layer**
   - critic agent
   - faithfulness checking
   - evidence alignment
   - retry decisions

6. **Memory Layer**
   - session memory
   - semantic memory
   - conversation state

7. **Persistence and Observability Layer**
   - MongoDB
   - Qdrant
   - logs
   - traces
   - metrics

This separation ensures that each layer can evolve independently.

---

## 7. Request Lifecycle

A request follows a structured lifecycle.

### 7.1 Ingestion or Query Entry
The system receives either:
- a document upload,
- a user query,
- a follow-up query in an existing session.

### 7.2 Context Assembly
The system gathers available context:
- current query,
- recent conversation turns,
- relevant persistent memory,
- user/session metadata,
- document collections available to the session.

### 7.3 Planning
The planner agent estimates:
- query intent,
- query complexity,
- whether retrieval is needed,
- whether web search is needed,
- whether the query should be decomposed,
- confidence in routing,
- budget for tokens and retrieval steps.

### 7.4 Route Selection
The orchestrator selects one of the core routes:
- no retrieval / direct reasoning,
- internal retrieval,
- web research,
- hybrid or iterative retrieval,
- fallback or abstain path.

### 7.5 Evidence Gathering
The chosen route gathers supporting material:
- internal document chunks,
- web snippets,
- prior session context,
- semantic memory items.

### 7.6 Answer Generation
The generator forms a candidate response using:
- the query,
- relevant evidence,
- system constraints,
- formatting instructions.

### 7.7 Verification
A critic agent checks:
- whether the answer is supported,
- whether important claims are grounded,
- whether the answer is complete enough,
- whether a retry is needed.

### 7.8 Response Formatting
The formatter prepares the final response with:
- answer text,
- supporting sources,
- confidence,
- route used,
- optional citations or reference metadata.

### 7.9 Logging and Persistence
The request outcome is stored for:
- debugging,
- analytics,
- evaluation,
- auditing,
- future memory retrieval.

---

## 8. Main Subsystems

## 8.1 Interface Layer

The interface layer exposes the project to external clients.

### Responsibilities
- accept user messages,
- validate input payloads,
- handle file uploads,
- manage sessions,
- expose status and metrics endpoints,
- return structured responses.

### Implementation Requirements
- FastAPI as the primary backend framework,
- request and response schemas defined with Pydantic,
- route handlers thin and delegation-only,
- consistent error responses,
- versioned API paths.

### Expected Endpoints
- `POST /documents/upload`
- `POST /chat/query`
- `GET /chat/{session_id}`
- `GET /query/{query_id}/sources`
- `GET /system/metrics`
- `POST /documents/reindex`
- `DELETE /documents/{doc_id}`

The interface layer must not contain business logic beyond validation and orchestration calls.

---

## 8.2 Orchestration Layer

This layer coordinates the end-to-end reasoning path.

### Core Components
- Planner agent
- LangGraph workflow
- route dispatcher
- retry controller
- failure fallback rules

### Responsibilities
- decide route,
- manage state transitions,
- call the correct downstream modules,
- stop repeated loops,
- enforce retry limits,
- pass structured state objects between nodes.

### Required Behavior
The orchestration layer should be deterministic in structure and probabilistic only where LLM-based decisions are intentionally used. Each node must have a clear input and output schema.

---

## 8.3 Query Planner Agent

This agent is the first decision-maker for every query.

### Responsibilities
- classify the query,
- estimate complexity,
- determine freshness requirements,
- determine whether document retrieval is needed,
- decide whether external search is required,
- decide whether the query should be decomposed,
- produce route confidence.

### Output Schema
A planner output should be structured and machine-readable. Example fields:
- `intent`
- `route`
- `complexity`
- `needs_web`
- `needs_retrieval`
- `confidence`
- `subqueries`
- `budget`

### Why It Matters
The planner avoids unnecessary retrieval and prevents the system from treating all queries alike. This is the key mechanism behind adaptive behavior.

---

## 8.4 Ingestion Pipeline

The ingestion pipeline converts raw documents into indexed knowledge.

### Responsibilities
- load files,
- extract text,
- normalize content,
- chunk documents,
- generate embeddings,
- attach metadata,
- store in vector database,
- update searchable indexes.

### Supported Inputs
Initially:
- PDF
- TXT

Later:
- DOCX
- HTML
- Markdown
- scanned PDFs with OCR

### Chunking Policy
Chunking should preserve semantic coherence while maintaining retrieval efficiency. Chunks should include:
- chunk text,
- source document ID,
- page or section metadata,
- chunk position,
- content hash.

### Metadata Requirements
Each chunk should store:
- document ID,
- filename,
- document type,
- version,
- page number or section,
- upload timestamp,
- access scope,
- optional tags.

### Why This Layer Exists
Without disciplined ingestion, retrieval quality degrades quickly. Document quality and metadata quality are primary drivers of answer quality.

---

## 8.5 Retrieval Layer

The retrieval layer finds evidence relevant to the current query.

### Retrieval Strategy
The project should use **hybrid retrieval**, not only dense vectors.

### Components
- dense vector retrieval,
- lexical retrieval,
- score fusion,
- metadata filtering,
- reranking,
- deduplication.

### Dense Retrieval
Captures semantic similarity between query and document chunks.

### Lexical Retrieval
Captures exact term matches and rare entity references.

### Reranking
A cross-encoder or similar reranker should reorder candidate chunks using higher-quality relevance scoring.

### Output
The retrieval layer should return:
- top chunks,
- source metadata,
- scores,
- retrieval path used,
- retrieval confidence.

### Design Requirement
Retrieval should be repeatable, inspectable, and testable independently of generation.

---

## 8.6 Web Research Agent

The web research agent is used when:
- the query needs current information,
- the internal corpus is insufficient,
- external corroboration is needed,
- the user asks for non-document knowledge that may have changed.

### Responsibilities
- search the web using a trusted search provider,
- gather snippets or page content,
- extract candidate evidence,
- normalize results into the same evidence format used by internal retrieval.

### Important Constraint
Web research should not be used for every query. It is a selective fallback and should be invoked only when the planner indicates need.

### Output
A structured evidence bundle including:
- search query used,
- titles,
- URLs or references,
- snippets,
- recency indicators if available.

---

## 8.7 General Reasoning Path

Some queries do not require retrieval.

### Examples
- simple conceptual questions,
- small talk,
- explanatory requests,
- brainstorming prompts,
- low-risk open-domain requests.

### Responsibilities
- answer directly,
- avoid unnecessary retrieval cost,
- remain within policy and system instructions,
- preserve consistency with the project’s grounded-answer philosophy.

This path should still be traceable and logged, even if it does not use evidence retrieval.

---

## 8.8 Generation Layer

The generation layer produces the candidate answer.

### Responsibilities
- synthesize evidence into a coherent answer,
- remain faithful to source material,
- avoid unsupported claims,
- follow response format requirements,
- maintain concise and useful language.

### Input
- user query,
- selected route,
- retrieved evidence,
- memory context,
- generation instructions.

### Output
- candidate answer,
- source references,
- optional uncertainty statements,
- confidence estimate.

### Design Constraint
The generator should not be the only authority on correctness. It is a synthesis component, not a truth engine.

---

## 8.9 Critic / Verification Layer

This layer validates the candidate response before it is returned.

### Responsibilities
- check faithfulness against evidence,
- detect unsupported claims,
- detect missing key points,
- assess completeness,
- decide whether to accept or retry,
- enforce abstention when confidence is too low.

### Verification Signals
- support coverage,
- contradiction risk,
- source alignment,
- answer completeness,
- retrieval sufficiency.

### Retry Policy
If the critic rejects the answer:
1. reformulate the query or subquery,
2. rerun retrieval or research,
3. regenerate,
4. verify again,
5. stop after a bounded number of attempts.

### Why This Layer Exists
It reduces hallucination and makes the system safer and more reliable.

---

## 8.10 Formatter Layer

The formatter prepares the final response for the client.

### Responsibilities
- clean raw output,
- enforce consistent structure,
- attach sources,
- include confidence information if desired,
- format markdown or JSON depending on client needs.

### Required Output Properties
- readable answer,
- source references,
- route metadata,
- error or abstain messages when applicable.

### Why It Matters
The formatter separates presentation from reasoning.

---

## 8.11 Memory Layer

The memory layer preserves state across interactions.

### Short-Term Memory
Used for:
- recent conversation turns,
- immediate session continuity,
- pronoun resolution,
- follow-up questions.

### Long-Term Memory
Used for:
- stable user preferences,
- recurring topics,
- persistent semantic facts,
- previous resolution patterns.

### Design Requirement
Memory should be filtered and scored before injection into prompts. It should not dump raw history into every request.

### Storage
Session history may live in MongoDB. Semantic memory may be stored in a separate structure or collection with retrieval support.

---

## 8.12 Persistence Layer

The persistence layer stores all durable artifacts.

### Qdrant
Used for:
- document embeddings,
- searchable chunk vectors,
- metadata payloads,
- hybrid retrieval support.

### MongoDB
Used for:
- chat sessions,
- conversation events,
- memory records,
- request logs,
- operational state.

### File System / Object Storage
Used for:
- raw uploaded files,
- intermediate parsing artifacts,
- evaluation datasets,
- exported logs if needed.

### Design Requirement
No important state should exist only in process memory.

---

## 8.13 Observability Layer

The observability layer allows the system to be measured and debugged.

### What Must Be Logged
- request ID,
- session ID,
- route selected,
- retrieval timing,
- generation timing,
- verification result,
- token usage,
- error type,
- retries performed,
- source count,
- confidence score,
- response length.

### Why This Matters
A production system must be explainable operationally. If something fails, the team must know where and why.

### Evaluation Support
The same observability data should support:
- regression tests,
- offline evaluation,
- quality comparisons between versions,
- performance benchmarks.

---

## 9. Data Model Overview

The project should use explicit models for core entities.

### 9.1 Query State
Contains:
- user query,
- session ID,
- memory context,
- planner output,
- selected route,
- evidence bundles,
- generated answer,
- critic result.

### 9.2 Document Chunk
Contains:
- chunk text,
- source document ID,
- chunk ID,
- position,
- metadata,
- embedding vector reference.

### 9.3 Session Record
Contains:
- session ID,
- user messages,
- assistant messages,
- timestamps,
- memory summary,
- active document scope.

### 9.4 Retrieval Result
Contains:
- retrieved chunks,
- retrieval scores,
- retriever type,
- reranker scores,
- selected evidence set.

### 9.5 Verification Record
Contains:
- answer ID,
- verdict,
- issues found,
- retry count,
- supporting evidence coverage.

---

## 10. Routing Modes

The system should support several explicit routes.

### 10.1 Direct Reasoning
Used when the query is simple and does not require evidence retrieval.

### 10.2 Internal Retrieval
Used when the answer should come from uploaded or indexed documents.

### 10.3 Web Research
Used when fresh or external information is needed.

### 10.4 Hybrid Route
Used when internal documents and web evidence both matter.

### 10.5 Abstain Route
Used when the system cannot answer safely or confidently.

Each route should be visible in logs and response metadata.

---

## 11. Non-Functional Requirements

### 11.1 Reliability
The system should degrade gracefully. If web search fails, internal retrieval may still work. If one document store is temporarily unavailable, the system should report the failure clearly.

### 11.2 Scalability
The architecture should support more documents, more users, and more requests without major redesign.

### 11.3 Maintainability
Each module should have a single responsibility and a clear interface.

### 11.4 Testability
Core logic should be unit-testable without requiring the full UI or live external services.

### 11.5 Security
The system should avoid exposing secrets, should validate inputs, and should be safe against unsafe file or prompt content where feasible.

### 11.6 Cost Awareness
The planner should reduce unnecessary model calls and avoid expensive pathways for simple requests.

---

## 12. Error Handling Philosophy

The system should not fail silently.

### Required Behaviors
- return structured errors,
- log failure reasons,
- preserve request context,
- distinguish between user error and system error,
- provide fallback answers or safe abstentions where possible.

### Examples of Failure Classes
- invalid file format,
- missing environment variable,
- empty retrieval result,
- vector database failure,
- web search failure,
- generation timeout,
- verification failure.

---

## 13. Extensibility Roadmap Built Into the Architecture

The current architecture should allow these future additions without breaking core design:

- additional retrieval strategies,
- more robust reranking models,
- document-level summaries,
- source citation rendering,
- multi-tenant document isolation,
- OCR for scanned documents,
- tool-using subagents,
- evaluation dashboards,
- policy-based response filters,
- multilingual support,
- advanced memory compression.

These are not required immediately, but the architecture should not block them.

---

## 14. Implementation Boundaries

To keep the codebase clean:

### API Layer
Should not contain retrieval, embedding, or reasoning logic.

### Agent Layer
Should not directly manage file uploads or database schema definitions.

### Retrieval Layer
Should not format user-facing responses.

### Persistence Layer
Should not make routing decisions.

### Formatter Layer
Should not query databases directly.

This separation prevents architectural entanglement.

---

## 15. Success Criteria for the Architecture

The architecture can be considered successful when the following are true:

1. The system can ingest documents reliably.
2. The system can answer document-based questions grounded in retrieved evidence.
3. The system can selectively route non-document questions away from retrieval.
4. The system can use web research for fresh information.
5. The system can verify answer quality before responding.
6. The system can preserve and reuse session memory.
7. The system can be observed, tested, and debugged effectively.
8. The system can be extended without rewriting core logic.

---

## 16. Final Architectural Statement

This project is a modular adaptive knowledge system that combines query planning, retrieval, generation, verification, and memory into a single disciplined workflow. The architecture is intentionally built to avoid a common failure pattern in RAG applications: forcing every question through the same retrieval-and-generation path.

Instead, the system chooses the correct path for each query, gathers the right evidence, verifies the result, and returns a grounded response. That is the core architectural identity of the project.

This document defines the target state of the system. The implementation should follow it closely and preserve the separation of concerns described here.
