# Dynamic-RAG — Evaluation and Observability

## 1. Purpose

This document defines the evaluation framework, observability strategy, benchmarking methodology, and accountability mechanisms for **Dynamic-RAG**.

Unlike conventional RAG systems that focus only on producing answers, Dynamic-RAG is designed as an **evaluation-first, measurable, and accountable retrieval system**.

The objective is not merely to answer questions.

The objective is to answer questions while being able to:
- explain why the answer was produced,
- measure retrieval quality,
- measure grounding quality,
- identify hallucinations,
- expose bottlenecks,
- benchmark improvements,
- understand failure modes.

Dynamic-RAG should never operate as a black box.

Every major decision in the pipeline should be measurable.

## 2. Evaluation Philosophy

Dynamic-RAG evaluation is divided into three independent but connected planes:

1. Retrieval Quality  
2. Generation Quality  
3. System-Level Quality

The philosophy is:

> Measure retrieval separately from generation, and measure both separately from overall system behavior.

A system may retrieve excellent evidence but generate poor answers. Another may generate faithful answers but fail under latency or cost constraints.

Without plane separation, diagnosis becomes impossible.

## 3. Plane 1 — Retrieval Quality

### Core Question
Did the system retrieve the correct evidence?

### Metrics

#### Context Recall
Formula:

GT sentences covered / GT total sentences

Purpose:
Measures whether retrieved evidence contains enough information to answer the query.

Failure mode:
The retriever returns topically similar chunks but misses decisive evidence.

#### Context Precision
Formula:

Relevant chunks / Retrieved chunks

Purpose:
Penalizes noisy retrieval.

Failure mode:
Top-k contains many distractors, diluting signal.

#### Recall@K
Tracks:

- Recall@1
- Recall@3
- Recall@5
- Recall@10

Purpose:
Checks if relevant evidence appears in top-k.

#### Mean Reciprocal Rank (MRR)
Formula:

MRR = (1 / |Q|) * Σ (1 / rank_i)

Purpose:
Measures how early the first relevant chunk appears.

Failure mode:
Relevant chunk exists at rank 8 while only top-3 are passed to the model.

#### NDCG@K
Purpose:
Measures ranking quality with graded relevance.

Recommended:
- NDCG@3
- NDCG@5
- NDCG@10

#### Hit Rate
Formula:

Queries with answer hit / Total queries

Purpose:
Production dashboard signal for quick triage.

#### Retrieval Latency
Track:
- P50
- P90
- P95
- P99

Components:
- embedding latency
- ANN search latency
- rerank latency

Tail latency matters more than mean latency.

#### Noise Robustness
Inject distractor chunks and measure degradation.

Goal:
Minimal faithfulness degradation under noisy retrieval.

## 4. Plane 2 — Generation Quality

### Core Question
Did the LLM use retrieved evidence faithfully?

#### Faithfulness
Most important metric.

Formula:

Claims entailed by retrieved evidence / Total claims

Measurement:
- claim decomposition
- NLI
- LLM-as-judge
- RAGAS

Policy:
Faithfulness acts as a release gate.

Low faithfulness triggers:
- retry
- query rewrite
- abstention

#### Answer Relevance
Measures whether the answer actually addresses the question.

Failure mode:
Correct but irrelevant answer.

#### Groundedness / Citation Accuracy
Measures:

Correct source-to-claim mapping / Total citations

Prevents:
- citation hallucination
- source mixing
- fabricated grounding

#### Answer Completeness
Measures:

Subquestions answered / Total subquestions

Purpose:
Ensure multi-part questions are fully answered.

#### Counterfactual Robustness
Tests whether Dynamic-RAG obeys retrieved evidence even when it contradicts LLM priors.

#### Noise Robustness
Inject adversarial chunks and evaluate degradation.

## 5. Plane 3 — System-Level Quality

### Core Question
Does the system behave reliably in production?

#### End-to-End Accuracy
Metrics:
- Exact Match
- F1
- LLM-as-judge

#### Rejection Rate
Measures:

Correct abstentions / Unanswerable queries

Dynamic-RAG should abstain when evidence is insufficient.

#### Cost Per Query
Measures:
- embedding cost
- retrieval cost
- reranking cost
- generation tokens

#### Retry Frequency
Measures critic effectiveness.

High retry frequency indicates:
- poor retrieval
- poor prompt engineering
- planner failures

#### Failure Rate
Track:
- retrieval failures
- generation failures
- timeout failures
- API failures

#### Robustness
Test:
- adversarial prompts
- malformed inputs
- contradictory context
- empty retrieval

## 6. Runtime Observability

Every request must emit a structured trace.

### Required Trace Fields

- request_id
- query_id
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

### Example Trace

```json
{
  "query_id": "Q001",
  "route": "internal_rag",
  "retrieval_latency": 140,
  "generation_latency": 800,
  "faithfulness": 0.92,
  "groundedness": 0.90,
  "retry_count": 1,
  "cost": 0.003
}
```

## 7. Benchmark Frameworks

Dynamic-RAG should align with:

### Retrieval
- BEIR
- MTEB

### Generation
- RAGAS

### End-to-End
- CRAG
- RGB
- FRAMES

## 8. Evaluation Dataset Structure

```text
evaluation/
├── benchmark_queries.json
├── gold_answers.json
├── expected_chunks.json
├── unanswerable_queries.json
└── adversarial_queries.json
```

Purpose:
Dynamic-RAG should be benchmarkable from Day 1.

## 9. Evaluation Flow

```text
Query
 ↓
Retrieval Evaluation
 ↓
Generation Evaluation
 ↓
System Evaluation
 ↓
Dashboard + Logs
```

## 10. Design Principles

1. Never trust retrieval without metrics.
2. Never trust generation without faithfulness.
3. Never trust benchmarks without robustness.
4. Never trust accuracy without latency and cost.
5. Always measure failures independently.

## 11. Success Criteria

Dynamic-RAG evaluation succeeds if:

- retrieval quality is measurable,
- grounding is measurable,
- hallucination is measurable,
- failures are diagnosable,
- regressions are detectable,
- benchmark comparisons are reproducible.

## 12. Final Statement

Dynamic-RAG is not only a retrieval system.

It is an **observable, accountable, and measurable RAG engineering platform** focused on:
- retrieval quality,
- grounded generation,
- robustness,
- production reliability,
- system accountability.
