# Dynamic-RAG — Project Architecture

## 1. Overview

**Dynamic-RAG** is a production-grade adaptive retrieval-augmented generation system built to answer questions through the most appropriate reasoning path while maintaining strong retrieval quality, faithful generation, and system-level accountability.

The architecture is designed around a central principle:

> **Do not force every query through the same pipeline.**
>
> Each query should be routed to the lightest reliable path that can answer it correctly, and every major decision should be observable and measurable.

Dynamic-RAG is not only a question-answering application. It is an engineered knowledge platform with explicit support for:
- query planning,
- retrieval,
- generation,
- verification,
- memory,
- observability,
- benchmarking,
- accountability.

The system is deliberately organized to make it possible to answer three distinct questions for every request:

1. **Did we retrieve the right evidence?**
2. **Did the model use that evidence faithfully?**
3. **Did the overall system behave reliably and efficiently?**

That is the architectural identity of Dynamic-RAG.

---

## 2. Project Vision

The project exists to build a knowledge system that is:
- adaptive,
- grounded,
- observable,
- measurable,
- robust,
- production-ready.

The design should support both local experimentation and real deployment. It should not treat retrieval quality, generation faithfulness, latency, and cost as separate concerns. They are all part of the same engineering problem.

Dynamic-RAG should be able to:
- answer from uploaded documents,
- answer from recent conversation context,
- selectively use web search when freshness is needed,
- verify whether an answer is supported,
- abstain when evidence is insufficient,
- log and measure every step,
- compare performance across versions.

---

## 3. Core Architectural Identity

Dynamic-RAG is defined by four architectural commitments.

### 3.1 Adaptive Execution
The system should not use a single fixed route for every query. Instead, it should plan execution based on:
- intent,
- complexity,
- evidence availability,
- freshness requirements,
- budget.

### 3.2 Evaluation-First Design
Retrieval quality, grounding quality, and system behavior must be measurable from the beginning. Evaluation is not a final-stage feature.

### 3.3 Evidence-Grounded Answers
The system should use retrieved or researched evidence whenever possible, and it should not present unsupported claims as fact.

### 3.4 Observability and Accountability
Every query should be traceable. The system should expose:
- route taken,
- evidence used,
- confidence estimates,
- retry behavior,
- latency,
- cost,
- verification outcome.

---

## 4. Design Goals

### 4.1 Strong Retrieval
The system should retrieve the most relevant and useful evidence, not merely topically similar chunks.

### 4.2 Faithful Generation
The system should produce answers that remain aligned with retrieved context.

### 4.3 Robust Routing
The system should choose the right route for each query rather than overusing retrieval or overusing generation.

### 4.4 Operational Accountability
The system should expose metrics and traces that allow debugging, benchmarking, and regression analysis.

### 4.5 Modular Implementation
Every subsystem should be independently replaceable.

### 4.6 Production Readiness
The system should support persistent storage, configurable deployment, and predictable failure handling.

---

## 5. Scope

### 5.1 In Scope
- document ingestion,
- chunking and indexing,
- hybrid retrieval,
- reranking,
- adaptive routing,
- answer generation,
- verification and retries,
- web research fallback,
- session memory,
- semantic memory,
- trace logging,
- benchmarking,
- system observability,
- API exposure,
- frontend integration.

### 5.2 Out of Scope for the Initial Release
- model training from scratch,
- multimodal reasoning,
- autonomous code execution,
- arbitrary plugin ecosystems,
- enterprise authorization layers beyond basic support,
- heavy workflow automation unrelated to RAG.

The architecture should allow these later, but they are not required for the first stable version.

---

## 6. High-Level Layered Architecture

Dynamic-RAG is built as a layered system.

### 6.1 Interface Layer
Handles:
- HTTP APIs,
- request validation,
- upload endpoints,
- query endpoints,
- metrics endpoints,
- client-facing responses.

### 6.2 Orchestration Layer
Handles:
- query planning,
- route selection,
- workflow control,
- retry control,
- state transitions.

### 6.3 Knowledge Layer
Handles:
- ingestion,
- embedding,
- vector indexing,
- hybrid retrieval,
- reranking,
- evidence assembly.

### 6.4 Reasoning Layer
Handles:
- generation,
- direct reasoning,
- web research,
- answer synthesis.

### 6.5 Verification Layer
Handles:
- faithfulness checks,
- completeness checks,
- support validation,
- retry or abstain decisions.

### 6.6 Memory Layer
Handles:
- session memory,
- semantic memory,
- context summarization.

### 6.7 Observability and Evaluation Layer
Handles:
- traces,
- logs,
- metrics,
- retrieval evaluation,
- generation evaluation,
- system-level benchmarking.

### 6.8 Persistence Layer
Handles:
- document storage,
- vector storage,
- session storage,
- trace storage,
- memory storage.

---

## 7. Request Lifecycle

Every query should follow a structured lifecycle.

### Step 1 — Entry
The system receives a query or document upload through the API.

### Step 2 — Context Assembly
The system loads:
- session history,
- relevant memory,
- available document scope,
- optional metadata.

### Step 3 — Planning
A planner agent estimates:
- intent,
- complexity,
- freshness need,
- retrieval need,
- route confidence,
- budget.

### Step 4 — Routing
The system selects one of the supported routes:
- direct reasoning,
- internal retrieval,
- web research,
- hybrid retrieval,
- abstain / fallback.

### Step 5 — Evidence Gathering
The chosen path retrieves or assembles:
- document chunks,
- web results,
- memory context,
- supporting metadata.

### Step 6 — Generation
The generator produces a candidate answer from the assembled evidence.

### Step 7 — Verification
The critic checks:
- support,
- faithfulness,
- completeness,
- citation correctness,
- answer relevance.

### Step 8 — Retry or Return
If verification fails, the system can:
- rewrite the query,
- retrieve again,
- research again,
- regenerate,
- abstain.

### Step 9 — Formatting and Logging
The final answer is formatted, traced, and persisted.

---

## 8. Main Subsystems

## 8.1 Interface Layer

The interface layer is the external contract for the system.

### Responsibilities
- accept user questions,
- accept document uploads,
- validate payloads,
- return structured answers,
- expose health and metrics endpoints.

### Design Rules
- API handlers should be thin,
- business logic must live in service modules,
- responses must be structured,
- errors must be explicit and consistent.

---

## 8.2 Orchestration Layer

This layer coordinates the overall workflow.

### Components
- planner agent,
- route dispatcher,
- graph controller,
- retry manager,
- failure manager.

### Responsibilities
- determine what happens next,
- route requests to the correct node,
- keep state consistent,
- stop infinite retry loops,
- preserve observability data.

### Key Principle
Planning and execution must remain separate.

---

## 8.3 Query Planner Agent

The planner is the system’s first decision point.

### Responsibilities
- classify query type,
- estimate query complexity,
- decide whether retrieval is needed,
- decide whether web search is needed,
- estimate confidence,
- produce subqueries when necessary,
- set execution budget.

### Output Requirements
The planner must return structured output, not free-form text.

### Why It Matters
A good planner reduces unnecessary retrieval, improves latency, and increases accuracy by selecting the right route early.

---

## 8.4 Ingestion Pipeline

The ingestion pipeline transforms raw documents into retrievable knowledge.

### Responsibilities
- load files,
- extract text,
- clean and normalize content,
- chunk documents,
- attach metadata,
- generate embeddings,
- persist chunks in vector storage.

### Supported Inputs
Initial support:
- PDF
- TXT

Future support:
- DOCX
- HTML
- Markdown
- scanned documents with OCR

### Ingestion Requirements
Each chunk should preserve:
- source document ID,
- file name,
- section or page reference,
- chunk position,
- version,
- hash,
- upload time.

### Why It Matters
Retrieval quality depends heavily on ingestion quality.

---

## 8.5 Retrieval Layer

The retrieval layer finds evidence for the query.

### Retrieval Modes
- dense retrieval,
- sparse retrieval,
- hybrid retrieval,
- metadata-filtered retrieval,
- reranked retrieval.

### Responsibilities
- retrieve candidate chunks,
- merge candidate sets,
- rerank results,
- deduplicate noisy results,
- preserve source traceability.

### Evidence Requirement
Retrieval should return evidence objects, not just text fragments.

Each evidence item should carry:
- source type,
- source identifier,
- chunk identifier,
- content text,
- score,
- metadata.

---

## 8.6 Web Research Agent

This agent is used only when current or external information is needed.

### Responsibilities
- formulate search queries,
- gather web results,
- normalize snippets,
- preserve source metadata,
- pass evidence to the generator.

### Design Constraint
The web route should be selective. It should not be the default path for every request.

---

## 8.7 Generation Layer

The generation layer synthesizes evidence into a candidate answer.

### Responsibilities
- combine the query and evidence,
- generate a grounded answer,
- follow formatting constraints,
- avoid unsupported claims,
- expose uncertainty when needed.

### Design Constraint
The generator is not the truth source. It is an evidence synthesis component.

---

## 8.8 Verification Layer

The verification layer validates the generated response.

### Responsibilities
- score faithfulness,
- detect unsupported claims,
- check source-to-claim alignment,
- verify answer completeness,
- decide whether to retry or abstain.

### Why It Exists
Generation alone can be fluent while still being incorrect. Verification prevents unsupported confidence from reaching the user.

---

## 8.9 Memory Layer

The memory layer manages continuity across sessions.

### Short-Term Memory
Stores recent conversation turns for immediate continuity.

### Long-Term Memory
Stores stable user preferences, semantic facts, and reusable context.

### Design Rule
Memory must be selective. It should not be dumped blindly into every prompt.

---

## 8.10 Observability Layer

This layer makes the system measurable.

### Must Be Observable
- route decisions,
- retrieval latency,
- generation latency,
- verification outcomes,
- retry count,
- token usage,
- cost,
- source count,
- failure states.

### Why It Matters
Without observability, the system cannot be improved systematically.

---

## 8.11 Evaluation Layer

This layer measures system quality across three planes:

### Plane 1 — Retrieval Quality
- Context Recall
- Context Precision
- Recall@K
- MRR
- NDCG@K
- Hit Rate

### Plane 2 — Generation Quality
- Faithfulness
- Answer Relevance
- Groundedness
- Citation Accuracy
- Completeness
- Robustness

### Plane 3 — System-Level Quality
- End-to-end accuracy
- Rejection rate
- Latency
- Cost per query
- Retry frequency
- Stability across versions

Evaluation is a core architectural function, not a reporting add-on.

---

## 9. Data and Persistence Architecture

## 9.1 Qdrant
Qdrant is the primary vector store for:
- chunk embeddings,
- source metadata,
- filtered retrieval,
- persistent indexing.

### Stored Data
Each vector record should contain:
- vector representation,
- chunk text,
- chunk ID,
- document ID,
- page or section reference,
- document version,
- tags,
- upload metadata.

---

## 9.2 MongoDB
MongoDB should store:
- chat sessions,
- conversation messages,
- semantic memory,
- request traces,
- document ingestion state,
- evaluation summaries.

---

## 9.3 File Storage
The file layer stores:
- raw uploads,
- intermediate parsing artifacts,
- benchmark datasets,
- generated evaluation reports.

---

## 10. Data Model Overview

### QueryState
Contains the evolving state of one request:
- request ID,
- session ID,
- query text,
- route,
- evidence,
- answer,
- critic result,
- final response,
- metrics.

### DocumentChunk
Contains one retrievable unit:
- chunk text,
- source metadata,
- vector reference,
- version,
- hash.

### PlannerOutput
Contains routing decisions:
- intent,
- complexity,
- route,
- confidence,
- budget,
- subqueries.

### RetrievalResult
Contains retrieval outputs:
- candidate chunks,
- reranked chunks,
- scores,
- confidence.

### VerificationResult
Contains validation outputs:
- faithful or not,
- supported or not,
- issues,
- retry decision.

### FinalResponse
Contains the user-facing output:
- answer,
- source references,
- route,
- confidence,
- status.

---

## 11. LangGraph Workflow Design

LangGraph should orchestrate the system as a state machine.

### Suggested Nodes
1. Context Loader  
2. Planner  
3. Route Dispatcher  
4. Internal Retriever  
5. Web Researcher  
6. General Reasoner  
7. Evidence Assembler  
8. Generator  
9. Critic  
10. Retry Controller  
11. Formatter  
12. Logger

### Workflow Principle
Each node should have:
- a clear input,
- a clear output,
- a bounded responsibility.

### State Transition Rule
The graph should make the decision path explicit and traceable.

---

## 12. Routing Modes

### 12.1 Direct Reasoning
Used for queries that do not require retrieval.

### 12.2 Internal Retrieval
Used for document-grounded queries.

### 12.3 Web Research
Used for current or external information.

### 12.4 Hybrid Retrieval
Used when both document evidence and external evidence are useful.

### 12.5 Abstain
Used when the system does not have sufficient support to answer safely.

---

## 13. Production Constraints

### 13.1 Reliability
The system must recover gracefully from:
- missing documents,
- empty retrieval,
- search failures,
- generation failures,
- verification failures.

### 13.2 Traceability
Every answer should be traceable back to:
- route selection,
- evidence,
- verification,
- logs.

### 13.3 Persistence
Important state must survive process restarts.

### 13.4 Cost Awareness
The architecture must allow controlled use of retrieval and LLM calls.

### 13.5 Modularity
Each subsystem should be replaceable without rewriting the entire application.

---

## 14. Testing and Benchmark Readiness

The architecture should support:
- unit testing,
- integration testing,
- offline benchmark evaluation,
- regression comparisons,
- route correctness tests,
- retrieval quality tests,
- generation faithfulness tests,
- system-level stability tests.

The design must make these tests easy to run and interpret.

---

## 15. Success Criteria

Dynamic-RAG has a successful architecture when:
- queries are routed intelligently,
- retrieval quality is measurable,
- answers are grounded,
- unsupported claims are caught,
- latency and cost are visible,
- routes and failures are traceable,
- the system can be benchmarked and improved over time.

---

## 16. Final Architectural Statement

Dynamic-RAG is an evaluation-first adaptive RAG system built to combine:
- intelligent routing,
- high-quality retrieval,
- faithful generation,
- verification,
- memory,
- observability,
- accountability.

Its defining characteristic is not just that it answers questions. It is that it can explain, measure, and defend how those answers were produced.
