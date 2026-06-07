# Dynamic-RAG — Roadmap

## 1. Purpose

This roadmap defines the implementation sequence for **Dynamic-RAG**, an evaluation-first, production-grade adaptive retrieval-augmented generation system.

The project is intentionally structured to prioritize:
- measurable retrieval quality,
- faithful generation,
- observable system behavior,
- reproducible benchmarking,
- accountable response generation,
- stable production deployment.

This is not a feature list. It is a build order.

The central rule is:

> **Do not add advanced orchestration before the system can prove that retrieval, grounding, and operational behavior are measurable.**

---

## 2. Project Identity

Dynamic-RAG is a system designed to answer questions by selecting the best available reasoning path and proving that the chosen path performed well.

Its identity is defined by four principles:

1. **Adaptive**
   - Route each query through the lightest correct path.

2. **Grounded**
   - Use evidence whenever evidence is available.

3. **Observable**
   - Expose route decisions, latency, cost, and failure modes.

4. **Accountable**
   - Measure retrieval quality, generation faithfulness, and system robustness.

This means the project is not merely a chatbot.
It is an engineered knowledge system with built-in evaluation hooks.

---

## 3. Roadmap Philosophy

The roadmap follows five engineering rules:

### Rule 1 — Build the measurable core first
If a subsystem cannot be measured, it cannot be trusted.

### Rule 2 — Add complexity only after the base path works
Retrieval before orchestration. Verification before scale. Observability before release.

### Rule 3 — Every phase must end in a runnable increment
No phase should produce only design artifacts.

### Rule 4 — Evaluation is part of development
Metrics and benchmarks are not final-stage extras.

### Rule 5 — Each layer must be independently testable
Retrieval, generation, critic behavior, and routing must all be debugged separately.

---

## 4. Release Strategy

Dynamic-RAG should progress through capability-driven versions.

### v0.1 — Runtime Foundation
Backend skeleton, config, logging, health checks.

### v0.2 — Observable Ingestion
Document parsing, chunking, metadata, indexing, traceable ingestion.

### v0.3 — Measurable Retrieval
Hybrid retrieval, reranking, retrieval metrics, benchmark support.

### v0.4 — Grounded Generation
Evidence-based answering, source attachment, answer evaluation.

### v0.5 — Verification Layer
Critic agent, retry controller, faithfulness gating, abstention logic.

### v0.6 — Adaptive Routing
Planner agent, route selection, query complexity analysis, web fallback.

### v0.7 — Memory and Session Intelligence
Short-term memory, long-term memory, session continuity.

### v0.8 — Observability and Benchmarking
Full traces, dashboards, offline benchmarks, regression tracking.

### v0.9 — UI Integration
End-to-end user interaction with visible sources and statuses.

### v1.0 — Production-Ready Stable Release
Deployment readiness, error handling, persistence safety, reproducibility.

---

## 5. Phase Plan

# Phase 0 — Project Definition Lock

## Objective
Finalize the identity and implementation standards for the project before code expansion.

## Deliverables
- `PROJECT_ARCHITECTURE.md`
- `ROADMAP.md`
- `SYSTEM_DESIGN.md`
- `EVALUATION_AND_OBSERVABILITY.md`
- repo naming consistency
- final folder structure
- `.env.example`
- `.gitignore`

## Exit Criteria
- the system name is standardized as Dynamic-RAG,
- architecture and roadmap are aligned,
- evaluation-first positioning is explicit.

---

# Phase 1 — Runtime Foundation

## Objective
Create a minimal backend that can run locally and expose stable health endpoints.

## Work Items
- FastAPI application bootstrap,
- config loader,
- environment variable parsing,
- logging framework,
- health endpoint,
- base schemas,
- error-handling scaffold,
- request ID generation.

## Deliverables
- running API server,
- `/health` endpoint,
- config module,
- logging module,
- basic request/response models.

## Exit Criteria
- the backend starts successfully,
- configuration is loaded from environment,
- the app returns predictable health responses.

## Evaluation Gate
- request logging is enabled,
- basic traces can be emitted,
- startup failure modes are visible.

---

# Phase 2 — Observability Backbone

## Objective
Make the system measurable before adding any intelligence layers.

## Work Items
- structured request logging,
- request/session IDs,
- trace object design,
- timing utilities,
- token/cost accounting hooks,
- metrics schema,
- trace persistence model.

## Deliverables
- trace logging module,
- metrics data model,
- trace repository,
- basic system metrics endpoint.

## Exit Criteria
- every request can be traced,
- latency and cost fields exist in the system model,
- logs are queryable.

## Evaluation Gate
- tracing works for health and test requests,
- trace records persist correctly,
- observability data is stable enough for later metrics.

---

# Phase 3 — Document Ingestion and Indexing

## Objective
Convert source files into searchable, versioned, metadata-rich knowledge units.

## Work Items
- file loader,
- text extraction,
- cleaning and normalization,
- chunking,
- metadata generation,
- embedding generation,
- Qdrant indexing,
- raw file persistence,
- ingestion status tracking.

## Supported Inputs
- PDF
- TXT

## Deliverables
- ingestion pipeline,
- chunk schema,
- indexing pipeline,
- document records,
- source metadata model.

## Exit Criteria
- a document can be uploaded and stored,
- chunks are indexed into Qdrant,
- metadata is retained,
- document versioning is possible.

## Evaluation Gate
- ingestion produces deterministic chunk outputs,
- indexed chunks are retrievable,
- source-page mapping is preserved.

---

# Phase 4 — Retrieval MVP

## Objective
Build a retrieval system that can find the correct evidence efficiently.

## Work Items
- dense search,
- sparse search,
- hybrid fusion,
- candidate merging,
- reranking,
- metadata filtering,
- evidence normalization,
- retrieval latency tracking.

## Deliverables
- retriever service,
- hybrid retrieval pipeline,
- reranker integration,
- evidence object schema,
- retrieval metrics calculator.

## Exit Criteria
- a query returns relevant evidence,
- retrieval output is inspectable,
- top-k results are stable.

## Evaluation Gate
- Recall@K is measurable,
- MRR is measurable,
- Context Precision is measurable,
- Hit Rate is measurable,
- retrieval latency is recorded.

---

# Phase 5 — Retrieval Evaluation Layer

## Objective
Make retrieval quality a first-class engineering concern.

## Work Items
- offline retrieval benchmarks,
- gold chunk mapping,
- context recall calculation,
- context precision calculation,
- NDCG calculation,
- retrieval regression tests,
- noise robustness tests.

## Deliverables
- evaluation datasets,
- retrieval benchmark runner,
- retrieval report generator,
- regression comparison scripts.

## Exit Criteria
- retrieval quality can be scored across datasets,
- retrieval regressions are detectable,
- noisy-context failure modes are visible.

## Evaluation Gate
- retrieval scores are recorded before proceeding to large-scale generation work,
- low-quality retrieval blocks downstream progress.

---

# Phase 6 — Grounded Answer Generation

## Objective
Generate answers from retrieved evidence with controlled prompting and source awareness.

## Work Items
- generation prompt templates,
- evidence-aware answer construction,
- source formatting,
- confidence output,
- abstention-ready response style.

## Deliverables
- generation module,
- structured answer schema,
- source-aware response format,
- evidence-to-answer linking.

## Exit Criteria
- the system can answer grounded document questions,
- generated answers reference evidence instead of free-form guessing.

## Evaluation Gate
- answer relevance is measurable,
- response completeness is measurable,
- groundedness is measurable.

---

# Phase 7 — Generation Evaluation Layer

## Objective
Measure whether the generated answer is faithful, relevant, and complete.

## Work Items
- claim decomposition,
- faithfulness scoring,
- citation accuracy scoring,
- answer relevance scoring,
- completeness evaluation,
- noise robustness evaluation,
- counterfactual robustness evaluation.

## Deliverables
- generation benchmark suite,
- critic-compatible scoring functions,
- generation quality reports,
- answer-level regression tests.

## Exit Criteria
- the system can distinguish faithful from unfaithful answers,
- generation regressions are visible,
- unsupported answers trigger measurable failures.

## Evaluation Gate
- faithfulness must reach a defined minimum threshold before later phases are expanded,
- unsupported claims must be detectable.

---

# Phase 8 — Verification and Critic Layer

## Objective
Prevent weak or unsupported answers from being returned.

## Work Items
- critic agent,
- answer validation,
- issue classification,
- retry controller,
- rewrite loop,
- abstention logic.

## Deliverables
- critic module,
- verification schema,
- retry policy,
- abstain/fallback response path.

## Exit Criteria
- answers are checked before release,
- low-confidence outputs can be rejected,
- the system can retry or abstain safely.

## Evaluation Gate
- faithfulness gating is active,
- rejection rate is measurable,
- retry loops are bounded.

---

# Phase 9 — Adaptive Query Planning

## Objective
Route each query through the best available execution path.

## Work Items
- planner agent,
- complexity detection,
- intent classification,
- freshness detection,
- route selection,
- subquery generation,
- confidence estimation.

## Supported Routes
- direct reasoning
- internal retrieval
- web research
- hybrid retrieval
- abstain path

## Deliverables
- planner module,
- structured route object,
- route decision policy,
- orchestration graph integration.

## Exit Criteria
- different query types follow different routes,
- routing decisions are structured and logged,
- retrieval is avoided when unnecessary.

## Evaluation Gate
- route accuracy is measurable,
- route confusion is visible,
- planner mistakes can be diagnosed.

---

# Phase 10 — Web Research Path

## Objective
Support queries that depend on fresh or external information.

## Work Items
- external search integration,
- search query formulation,
- snippet normalization,
- result ranking,
- source trace preservation,
- web evidence passing to generator.

## Deliverables
- web research agent,
- external evidence model,
- web-grounded answering path,
- web source metadata support.

## Exit Criteria
- freshness-sensitive queries can be answered with web evidence,
- web and internal evidence remain distinguishable.

## Evaluation Gate
- external source correctness is verifiable,
- web path latency and cost are measured.

---

# Phase 11 — Memory System

## Objective
Add conversational continuity and reusable long-term context.

## Work Items
- session history storage,
- short-term memory retrieval,
- semantic memory storage,
- memory summarization,
- memory filtering,
- memory injection policy.

## Deliverables
- MongoDB session store,
- memory retrieval module,
- semantic memory schema,
- memory summarizer.

## Exit Criteria
- follow-up questions preserve context,
- memory remains separate from document retrieval.

## Evaluation Gate
- memory usefulness is measurable,
- irrelevant memory injection is minimized.

---

# Phase 12 — System-Level Evaluation

## Objective
Measure production behavior across the full stack.

## Work Items
- end-to-end accuracy evaluation,
- rejection-rate analysis,
- latency percentiles,
- cost-per-query analysis,
- retry-frequency analysis,
- system robustness tests,
- benchmark report generation.

## Deliverables
- system benchmark runner,
- observability dashboard data,
- cross-version comparison reports,
- failure analysis summaries.

## Exit Criteria
- full pipeline performance is measurable,
- bottlenecks are visible,
- production risks are surfaced.

## Evaluation Gate
- end-to-end metrics meet release thresholds,
- system-level regressions are blocked.

---

# Phase 13 — UI Integration

## Objective
Expose the system through a usable interface without weakening backend discipline.

## Work Items
- chat UI,
- document upload UI,
- source display,
- route status display,
- confidence or trace display,
- error visibility.

## Deliverables
- Streamlit or equivalent frontend,
- upload workflow,
- interactive chat,
- answer presentation with sources.

## Exit Criteria
- users can upload documents and ask questions end to end,
- backend routing and verification remain the source of truth.

---

# Phase 14 — Hardening and Production Readiness

## Objective
Prepare the system for stable repeatable usage.

## Work Items
- containerization,
- deployment config,
- secret handling,
- rate limiting hooks,
- timeout handling,
- restart safety,
- persistence verification,
- test coverage improvements,
- final documentation.

## Deliverables
- deployment-ready service,
- environment profiles,
- operational runbook,
- release checklist,
- stable version tag.

## Exit Criteria
- the system survives restart,
- retrieval persists correctly,
- evaluation artifacts remain available,
- the project can be demonstrated reliably.

---

## 6. Mandatory Evaluation Gates

Dynamic-RAG must not progress blindly between phases.

### Gate A — Retrieval Gate
Required before expanding generation complexity.

Minimum requirements:
- retrieval metrics are measurable,
- retrieval regressions are visible,
- noisy retrieval behavior is understood.

### Gate B — Faithfulness Gate
Required before adding richer orchestration.

Minimum requirements:
- answer claims are checked against evidence,
- unsupported claims are detected,
- retry or abstention works.

### Gate C — Route Gate
Required before large-scale memory expansion.

Minimum requirements:
- planner decisions are stable,
- route selection is traceable,
- query types are classified correctly.

### Gate D — Production Gate
Required before release.

Minimum requirements:
- latency is within tolerance,
- cost is within tolerance,
- observability is complete,
- benchmark results are reproducible.

---

## 7. Build Priorities

The implementation order must always respect the following priority stack:

1. Runtime foundation
2. Observability backbone
3. Ingestion and indexing
4. Retrieval MVP
5. Retrieval evaluation
6. Grounded generation
7. Generation evaluation
8. Verification
9. Adaptive routing
10. Web research
11. Memory
12. System-level evaluation
13. UI
14. Hardening

This order prevents premature complexity and protects the evaluation-first architecture.

---

## 8. Non-Negotiable Rules

1. Do not build agents before observability exists.
2. Do not build orchestration before retrieval works.
3. Do not trust generation before faithfulness is measurable.
4. Do not trust retrieval before it is benchmarked.
5. Do not move to production before latency and cost are known.
6. Do not hide failures.
7. Do not skip evaluation gates.
8. Do not treat evaluation as an afterthought.

---

## 9. Definition of Done for Dynamic-RAG v1.0

Dynamic-RAG v1.0 is complete only when all of the following are true:

- documents can be ingested and persisted,
- retrieval quality is measurable and benchmarked,
- generation faithfulness is measurable and enforced,
- route decisions are adaptive and observable,
- web research works when needed,
- memory improves continuity without contaminating retrieval,
- system-level latency and cost are tracked,
- failures are diagnosable,
- the system can be evaluated across versions,
- the project can be deployed and demonstrated reliably.

---

## 10. Final Statement

Dynamic-RAG is a build-order driven system, not a feature pile.

The project should evolve through:
- measurable foundation,
- observable ingestion,
- benchmarked retrieval,
- grounded generation,
- verified answers,
- adaptive routing,
- robust memory,
- system-level accountability,
- production hardening.

That is the implementation strategy for the project.
