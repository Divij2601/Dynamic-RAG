# Dynamic-RAG Usage Guide

This document covers everything you need to run, query, ingest documents into,
evaluate, and extend Dynamic-RAG. It assumes you have already completed the
installation steps in the root README and that your `.env` file is configured
with valid API keys.

---

## Table of Contents

1. [Running the Full Stack](#1-running-the-full-stack)
2. [Using the Streamlit UI](#2-using-the-streamlit-ui)
3. [Session Management](#3-session-management)
4. [Using the REST API](#4-using-the-rest-api)
5. [Ingesting Documents](#5-ingesting-documents)
6. [Understanding Query Routing](#6-understanding-query-routing)
7. [Running the Benchmark](#7-running-the-benchmark)
8. [Running Tests](#8-running-tests)
9. [Production Tips](#9-production-tips)

---

## 1. Running the Full Stack

Dynamic-RAG requires three services running before the UI becomes usable.
The services must be started in the order below because FastAPI imports
`mongo_client` and `qdrant_client` at startup time — if either database is
unreachable during import, the API server will log a warning but continue;
however, the first query will fail until the database responds.

### Service dependency diagram

```
MongoDB (27017) <─── FastAPI (8000) <─── Streamlit (8501)
    |                     |
    └── stores traces,    └── imports DB clients
        sessions,             at import time;
        memory                compiles LangGraph
                              on first request
```

### Terminal 1 — MongoDB

```bash
mongod --dbpath /var/lib/mongodb
```

On Windows (if MongoDB is installed as a service, skip this step; if not):

```powershell
mongod --dbpath "C:\data\db"
```

Verify it is up:

```bash
mongosh --eval "db.adminCommand({ ping: 1 })"
```

Expected output: `{ ok: 1 }`.

### Terminal 2 — Qdrant

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

On Windows (PowerShell):

```powershell
docker run -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Verify it is up:

```bash
curl http://localhost:6333/healthz
```

Expected output: `{"title":"qdrant - vector search engine","version":"..."}`.

### Terminal 3 — FastAPI

```bash
cd /path/to/Dynamic-RAG
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Wait for the log line:

```
INFO:     Application startup complete.
```

The LangGraph is compiled on first import. Swagger UI is available at
`http://127.0.0.1:8000/docs`.

### Terminal 4 — Streamlit UI

```bash
cd /path/to/Dynamic-RAG
streamlit run app.py
```

The browser should open automatically to `http://localhost:8501`. The sidebar
shows a green dot labeled "API healthy" when all three services are reachable.

---

## 2. Using the Streamlit UI

The UI is organized into four main tabs: **Chat**, **Metrics**,
**Session History**, and **Evaluation**.

---

### 2a. Chat Tab

**Typing a query**

Type your question in the text field at the bottom of the Chat area and press
Enter or click "Send". The system forwards the text to `POST /chat/query` via
the FastAPI backend.

**Route badge**

Each assistant response is prefixed by two badges:

| Badge label | What it means |
|---|---|
| Internal RAG | Answer came from the local Qdrant corpus |
| Web Research | Answer came from Tavily web search results |
| Hybrid | Answer combined corpus and web evidence |
| Memory | Answer was drawn from this session's conversation history |
| Direct | No retrieval — pure language model generation |

The badge color provides a quick visual cue: blue for internal, amber for web,
purple for hybrid, green for memory, orange for direct.

**Confidence and faithfulness scores**

Click the "Details" expander below any assistant message to see two score bars:

- **Confidence** (0 to 1): the planner's self-reported certainty that the
  chosen route will produce a good answer. Scores above 0.8 are shown in
  green, 0.5–0.8 in amber, below 0.5 in red.
- **Faithfulness** (0 to 1): the verifier model's judgment of how well the
  generated answer is grounded in the retrieved evidence. A score close to 1.0
  means every claim is directly supported by a source passage.

The "Grounded" field is a boolean summary: Yes when the verifier found adequate
evidence support, No when it did not (and the answer may have been abstained).

**Expanding sources**

Click the "Sources (N)" expander to see all evidence chunks used to construct
the answer. Each source is labeled:

```
Source 1 — geopolitics_cold_war.pdf, page 4
Source 2 — india_history.txt
Source 3 — https://example.com/article (web)
```

Document sources and web sources are separated into sub-tabs when both types
are present. Expanding a source card shows the raw chunk text (up to 600
characters) and the retrieval relevance score.

The numbering in the source list matches the `[Source N]` citation markers
that appear in the answer text, so you can trace every claim back to its origin.

**Evidence numbering convention**

Document chunks occupy Source 1 through N. Web results continue from N+1.
This numbering is consistent within a single response — the LLM is prompted
to cite by the same indices.

**Follow-up questions**

The planner receives the last 5 turns of conversation history with every
request. Queries that reference prior context are automatically detected and
rewritten into standalone queries before retrieval. For example:

- You ask: "Who founded the United Nations?"
- Assistant answers with details about 1945.
- You type: "Same as before but search the web."
- The planner detects a follow-up, rewrites the query to
  "United Nations founding history web search", and routes it to `web_research`.

You can also reference previous topics implicitly:
"What did it say about the Cold War?" or "Tell me more about that treaty."

---

### 2b. Metrics Tab

The Metrics tab displays live operational data aggregated from the MongoDB
`traces` collection. It refreshes each time you navigate to the tab.

**KPI cards (top row)**

| Card | What it measures |
|---|---|
| Total Queries | Number of requests processed since the server started |
| Mean Latency | Average end-to-end time per query in milliseconds |
| P95 Latency | 95th-percentile latency — 95% of queries are faster than this value |
| Abstention Rate | Fraction of queries where the system declined to answer due to low evidence quality |
| Total Cost | Accumulated Groq API cost in USD across all traced requests |

**Route distribution donut chart**

Shows the proportion of queries handled by each of the five routes. A healthy
distribution for a domain-specific corpus should be dominated by
`internal_rag`. High `web_research` volume indicates the corpus does not cover
the topics being asked about.

**P95 latency interpretation**

P95 latency is the most operationally useful latency metric. Mean latency can
be dragged down by fast `direct_generation` queries. P95 reveals the tail
behavior — if P95 is much higher than the mean, there are slow outliers, which
usually indicates reranker or Groq API rate-limit delays.

A P95 below 8000ms is typical for `internal_rag` queries with the current
corpus (4888 chunks). `web_research` queries are slower because of the Tavily
network round-trip.

---

### 2c. Session History Tab

Use this tab to inspect a specific session's full conversation without
switching to it.

1. The current session ID is pre-filled in the text input.
2. Enter any session ID (shown in the sidebar or returned by the API).
3. Click "Load session".

The tab renders each turn with route badge, timestamp, and confidence score in
read-only mode. This is useful for auditing past conversations or debugging
unexpected answers.

---

### 2d. Evaluation Tab

The Evaluation tab displays results from benchmark reports saved to
`evaluation/reports/`. Reports are generated by the benchmark runner (see
[Section 7](#7-running-the-benchmark)).

**Selecting a report**

Use the dropdown at the top of the tab to select a report by filename. Reports
are named `{experiment_name}_{YYYYMMDD_HHMMSS}.json` and sorted newest-first.

**Reading Plane 1 — Retrieval Quality**

Six metric cards are shown:

| Metric | What it means |
|---|---|
| Recall@K | Fraction of relevant chunks that appear anywhere in the top-K retrieved results |
| MRR | Mean Reciprocal Rank — how highly the first relevant chunk is ranked |
| NDCG@K | Normalized Discounted Cumulative Gain — rewards relevant chunks ranked higher |
| Hit Rate | Fraction of queries where at least one relevant chunk was retrieved |
| Context Precision | Fraction of retrieved chunks that are actually relevant |
| Context Recall | Fraction of all relevant chunks that were retrieved |

The radar chart visualizes all six metrics simultaneously. A well-balanced
system produces a wide, regular polygon. Narrow spikes indicate a metric that
is strong but at the expense of others.

**Reading Gate C — Routing Accuracy**

This section shows how accurately the planner assigned each query to the
correct route. The score bars break down accuracy per route class, and the
confusion matrix heatmap shows which routes are being confused with each other.
High confusion between `internal_rag` and `hybrid` is normal for ambiguous
queries.

**Reading Plane 2 — Generation Quality**

Five score bars show the LLM judge's assessment of the generated answers:

| Metric | What it means |
|---|---|
| Faithfulness | Claims in the answer are supported by retrieved evidence |
| Groundedness | Answer stays within the bounds of the evidence without hallucination |
| Answer Relevance | Answer addresses what the query was actually asking |
| Completeness | Answer covers all aspects required by the ground truth |
| Citation Accuracy | In-text citations match the retrieved source indices |

**Reading Plane 3 — System Quality**

Six operational metrics: E2E Accuracy, Rejection Rate, Mean Latency, P95
Latency, Failure Count, and estimated Cost per Query. A Failure Count of 0
means no queries raised an unhandled exception during the benchmark run.

---

## 3. Session Management

**How sessions work**

Each session is identified by a UUID-derived string (`session_XXXXXXXX`).
A new session is created automatically when you first open the UI or click
"+ New". Sessions are stored in MongoDB (`conversations` collection) and
persist across server and browser restarts.

**Auto-naming**

Sessions are auto-named from the first message typed in that session. Before
a message is sent, the session name displays as the raw session ID. After the
first query, the name still shows the ID until you rename it manually.

**Session order**

The sidebar lists all sessions ordered by most-recent activity (the session
with the most recent query appears at the top). Each session card shows:
- Session name (or ID if unnamed)
- A preview of the last message
- Message count
- Relative timestamp ("5m ago", "2h ago")

**Opening a past session**

Click the "Open" button below any session card in the sidebar. This loads the
full message history from MongoDB and reconstructs the chat display including
all source panels. The active session is highlighted with a blue dot.

**Renaming a session**

Edit the text field at the top of the sidebar (under "Active Session"). The
name is saved to MongoDB as soon as you change the field value and click
elsewhere — no separate save button is needed.

**Starting a new session**

Click "+ New" in the sidebar. The current session is already persisted
automatically (every query is saved immediately after generation). Clicking
"+ New" creates a fresh session ID, clears the visible chat, and moves the
old session to the history list where it is accessible via "Open".

**Clear vs. New**

"Clear" erases the in-memory message display but does not create a new session
ID — subsequent queries are still appended to the same session in MongoDB.
"+ New" creates an entirely new session.

---

## 4. Using the REST API

All endpoints accept and return JSON. The base URL is `http://localhost:8000`.
Interactive documentation is available at `http://localhost:8000/docs`.

---

### POST /chat/query

Submit a query and receive a structured answer.

**Basic query (curl)**

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "When did the Cold War end?"}'
```

**Session-aware query (curl)**

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What role did NATO play?",
    "session_id": "session_a1b2c3d4"
  }'
```

**Basic query (Python)**

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/query",
    json={"query": "When did the Cold War end?"},
    timeout=120,
)
response.raise_for_status()
data = response.json()

print(data["answer"])
print(data["route"])
print(data["confidence"])
print(data["faithfulness_score"])
print(data["query_id"])   # save this to fetch sources
```

**Session-aware query (Python)**

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/query",
    json={
        "query": "What role did NATO play?",
        "session_id": "session_a1b2c3d4",
    },
    timeout=120,
)
data = response.json()
```

**Response fields**

| Field | Type | Description |
|---|---|---|
| `answer` | string | The generated answer |
| `query_id` | string | UUID for this specific query (use to fetch sources) |
| `session_id` | string | Session the query belongs to |
| `route` | string | One of the five route names |
| `confidence` | float | Planner confidence (0–1) |
| `faithfulness_score` | float | Verifier score (0–1) |
| `latency_ms` | float | Server-side latency in milliseconds |
| `status` | string | `success`, `abstained`, or `error` |
| `sources` | list | Abbreviated source list (use /sources for full text) |

---

### GET /query/{query_id}/sources

Retrieve the full evidence source list for a previously answered query. The
`query_id` is returned in the `/chat/query` response.

**curl**

```bash
curl http://localhost:8000/query/req_7f3a2b1c4d5e6f8a/sources
```

**Python**

```python
import requests

query_id = "req_7f3a2b1c4d5e6f8a"
response = requests.get(
    f"http://localhost:8000/query/{query_id}/sources",
    timeout=10,
)
data = response.json()

for source in data["sources"]:
    print(source["source_type"], source["chunk_id"], source["score"])
```

**Response fields**

| Field | Type | Description |
|---|---|---|
| `query_id` | string | The query ID |
| `route` | string | Route used for this query |
| `num_sources` | int | Total source count |
| `sources` | list | Full source objects with text, metadata, page, score |

---

### GET /chat/{session_id}

Retrieve the conversation history for a session.

**curl**

```bash
curl http://localhost:8000/chat/session_a1b2c3d4
```

**Python**

```python
import requests

response = requests.get(
    "http://localhost:8000/chat/session_a1b2c3d4",
    timeout=10,
)
data = response.json()

print(f"Session has {data['message_count']} turns")
for msg in data["messages"]:
    print(msg["query"])
    print(msg["route"], msg["confidence"])
    print(msg["answer"][:200])
    print("---")
```

Returns up to 50 most recent turns, each with `query`, `answer`, `route`,
`confidence`, and `timestamp`.

---

### POST /documents/upload

Upload a PDF or TXT file and ingest it into the knowledge base. The file is
parsed, chunked, embedded, and indexed into Qdrant in one synchronous call.
Large PDFs may take 30–90 seconds.

**curl**

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/manual_relay_v1.pdf"
```

**Python**

```python
import requests

with open("/path/to/manual_relay_v1.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": ("manual_relay_v1.pdf", f, "application/pdf")},
        timeout=300,
    )

response.raise_for_status()
data = response.json()
print(data["document_id"])
print(data["status"])      # "indexed" or "failed"
print(data["message"])     # e.g. "143 chunks indexed into the knowledge base."
```

Supported file types: `.pdf`, `.txt`.

**Response fields**

| Field | Type | Description |
|---|---|---|
| `document_id` | string | Content-derived unique ID for the document |
| `filename` | string | Original filename |
| `status` | string | `indexed` or `failed` |
| `message` | string | Human-readable summary including chunk count |

---

### GET /health

Check the API server and its database connections.

**curl**

```bash
curl http://localhost:8000/health
```

**Python**

```python
import requests

data = requests.get("http://localhost:8000/health", timeout=4).json()
print(data["status"])               # "healthy" or "degraded"
print(data["services"]["mongodb"])  # "connected" or "disconnected"
print(data["services"]["qdrant"])   # "connected" or "disconnected"
```

Status is `degraded` if either MongoDB or Qdrant is unreachable.

---

### GET /system/metrics

Retrieve aggregated operational metrics.

**curl**

```bash
curl http://localhost:8000/system/metrics
```

**Python**

```python
import requests

data = requests.get("http://localhost:8000/system/metrics", timeout=10).json()
print(f"Total queries:    {data['total_requests']}")
print(f"Mean latency:     {data['mean_latency_ms']}ms")
print(f"P95 latency:      {data['p95_latency_ms']}ms")
print(f"Abstention rate:  {data['abstention_rate']:.1%}")
print(f"Route breakdown:  {data['route_distribution']}")
```

---

## 5. Ingesting Documents

### Method A — Streamlit UI upload widget

1. In the sidebar, locate the "Documents" section.
2. Click the file uploader and select a `.pdf` or `.txt` file.
3. Click "Ingest document".
4. A spinner shows while the pipeline runs; a success message confirms the
   chunk count when indexing completes.
5. The sidebar's indexed documents list updates automatically.

### Method B — CLI pipeline

Use the `src.ingestion.pipeline` module directly. This bypasses the API server
and is suitable for bulk ingestion or scripted workflows.

**Ingest a single file**

```bash
python -m src.ingestion.pipeline data/raw/primary/manual_relay_v1.pdf
```

**Ingest an entire directory (recursive)**

```bash
python -m src.ingestion.pipeline data/raw/primary
```

The pipeline walks the directory recursively and processes every `.pdf` and
`.txt` file it finds.

**Expected terminal output**

```
INFO  Ingesting: data/raw/primary/manual_relay_v1.pdf
INFO  Parsed 24 pages
INFO  Produced 143 chunks
SUCCESS  Ingested manual_relay_v1.pdf -> 143 chunks (doc_id=doc_7a3b...)
{'document_id': 'doc_7a3b...', 'filename': 'manual_relay_v1.pdf', 'chunks': 143, 'indexed': True}
```

### What happens automatically on every ingestion

1. **BM25 cache invalidation** — the sparse retriever discards its cached
   BM25 corpus index so the next query rebuilds it with the new document
   included. No restart is required.
2. **Corpus description rebuild** — `corpus_description_builder.invalidate()`
   is called so the planner's knowledge-base description is regenerated on the
   next query, keeping routing decisions accurate.
3. **MongoDB metadata save** — a document record is written to the
   `documents` collection with the document ID, filename, chunk count, and
   ingestion timestamp.

### Verifying ingestion with Qdrant

Check the total indexed point count directly via the Qdrant HTTP API:

```bash
curl http://localhost:6333/collections/dynamic_rag_documents
```

Look for `"points_count"` in the response. Each successfully ingested chunk
corresponds to one Qdrant point.

You can also query by document ID to confirm a specific file was indexed:

```python
from src.database.qdrant_client import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = qdrant_client.client
results = client.scroll(
    collection_name="dynamic_rag_documents",
    scroll_filter=Filter(
        must=[FieldCondition(
            key="document_id",
            match=MatchValue(value="doc_7a3b...")
        )]
    ),
    limit=5,
    with_payload=True,
)
print(f"Points found: {len(results[0])}")
```

---

## 6. Understanding Query Routing

The planner (`llama-3.1-8b-instant`) reads the query, the last 5 turns of
conversation history, and a dynamically generated description of the corpus
contents. It selects one of five routes. Below are representative examples
with reasoning.

---

**Example 1 — Factual historical question -> internal_rag**

```
Query: "What was the outcome of the Cuban Missile Crisis?"
Route: internal_rag
```

Reason: The corpus explicitly covers the Cold War, Cuban Missile Crisis, and
related events. The answer is stable historical fact that does not require live
data. The planner identifies the topic as covered by the knowledge base and
routes to the dense+sparse hybrid retriever backed by Qdrant.

---

**Example 2 — Real-time news -> web_research**

```
Query: "Latest news on China-Taiwan tensions this week"
Route: web_research
```

Reason: The corpus covers geopolitical background but not live or rolling news.
The planner detects the "this week" temporal qualifier and the live-data
requirement, and routes to Tavily web search. No Qdrant lookup occurs.

---

**Example 3 — Compare corpus with current research -> hybrid**

```
Query: "Compare our documents on NATO with the latest 2026 expansion news"
Route: hybrid
```

Reason: "Our documents" signals internal corpus content; "latest 2026" signals
real-time external data. The planner sets `needs_retrieval=True` and
`needs_web=True`, and the hybrid node runs both the Qdrant retriever and Tavily
in parallel before merging evidence.

---

**Example 4 — Memory reference -> memory**

```
Query: "What did we discuss earlier about India's foreign policy?"
Route: memory
```

Reason: "We discussed earlier" is a direct reference to the prior conversation,
not the corpus. The planner routes to the memory node, which fetches recent
turns and semantic memory embeddings for the current session from MongoDB and
the `dynamic_rag_memory` Qdrant collection. No document retrieval occurs.

---

**Example 5 — Language task -> direct_generation**

```
Query: "Rewrite this sentence in a more formal tone: 'The deal fell through.'"
Route: direct_generation
```

Reason: The task requires language generation only — no factual lookup is
needed. The planner routes directly to the generator, bypassing all retrieval
stages. This is also the route used for simple math, creative writing requests,
and casual greetings.

---

**Example 6 — Follow-up rewrite -> web_research**

```
Turn 1: "What is India's stance on the Belt and Road Initiative?"
         (answered by internal_rag)
Turn 2: "Same as before but search the internet."
Route: web_research
rewritten_query: "India stance on Belt and Road Initiative 2025 current"
```

Reason: The planner detects "same as before" as a follow-up trigger. It reads
Turn 1's query from history, rewrites the standalone query, and changes the
route to `web_research` because the user explicitly requested a live search.
The rewritten query is used for the Tavily call instead of the raw follow-up
text.

---

## 7. Running the Benchmark

The benchmark runner executes all three evaluation planes sequentially and
saves a JSON report.

### Step 1 — Prepare the test set

The test set lives at `evaluation/data/test_set.json`. Each entry follows this
schema:

```json
{
  "query": "When did the Soviet Union dissolve?",
  "ground_truth_answer": "The Soviet Union dissolved on December 26, 1991.",
  "relevant_chunk_ids": [
    "chunk_273b5efc9311",
    "chunk_a8864d665cf8"
  ],
  "answerable": true,
  "metadata": {
    "category": "factual",
    "topic": "cold_war"
  }
}
```

Set `"answerable": false` for queries that should be rejected by the system.

`relevant_chunk_ids` are used by Plane 1 to measure retrieval quality. To
find chunk IDs for a document you have ingested, scroll the Qdrant collection
and inspect the `chunk_id` field in each point's payload.

### Step 2 — Run the benchmark

```bash
python -m evaluation.runner evaluation/data/test_set.json
```

With a custom experiment name:

```bash
python -m evaluation.runner evaluation/data/test_set.json my_experiment_v2
```

The runner prints a JSON summary to stdout and also saves the report to
`evaluation/reports/`.

### Step 3 — Find the report

```
evaluation/reports/dynamic_rag_20260607_154331.json
evaluation/reports/my_experiment_v2_20260607_160045.json
```

Reports are named `{experiment_name}_{YYYYMMDD_HHMMSS}.json`.

### Step 4 — Read the JSON structure

```json
{
  "experiment_name": "dynamic_rag",
  "timestamp": "2026-06-07T15:43:31",
  "dataset": "evaluation/data/test_set.json",
  "plane_1_retrieval": {
    "Recall@K": 0.9338,
    "MRR": 1.0,
    "NDCG@K": 0.9538,
    "Hit Rate": 1.0,
    "Context Precision": 0.9266,
    "Context Recall": 0.9338
  },
  "gate_c_routing": {
    "Route Accuracy": 0.9888,
    "Per-Class Accuracy": { ... },
    "Confusion Matrix": { ... }
  },
  "plane_2_generation": {
    "Faithfulness": 0.9899,
    "Groundedness": 0.9873,
    "Answer Relevance": 0.8572,
    "Citation Accuracy": 0.9886,
    "Completeness": 0.8494,
    "Evaluated (answerable)": 79
  },
  "plane_3_system": {
    "End-to-End Accuracy": 0.8437,
    "Rejection Rate": 0.3333,
    "Mean Latency (ms)": 15206.52,
    "P95 Latency (ms)": 34763.05,
    "Failure Count": 0,
    "Estimated Cost / Query": 0.0
  }
}
```

### Groq rate limits and pacing

The benchmark makes LLM calls to three separate models on separate Groq rate
limit buckets:

| Role | Model | Bucket |
|---|---|---|
| Planner | `llama-3.1-8b-instant` | Separate per-model quota |
| Generator | `llama-3.3-70b-versatile` | Separate per-model quota |
| Critic (verifier) | `qwen/qwen3-32b` | Separate per-model quota |

The `execute_dataset` function in `evaluation/pipeline_exec.py` applies a
15-second pacing delay between queries. This is intentional: without pacing,
back-to-back benchmark queries exhaust the per-minute token limits on the
70B and 32B models, producing rate limit errors that inflate failure counts
and latency measurements.

Do not remove the pacing delay. For the current test set of ~35 queries,
the benchmark takes approximately 10–12 minutes to complete.

---

## 8. Running Tests

The test suite lives in the `tests/` directory. Tests are split into fast
unit tests and integration tests that require live Qdrant, MongoDB, and Groq.

### Fast tests only (no services required)

```bash
pytest -m "not integration"
```

This runs chunker, metadata, parsing, confidence calibration, and retrieval
metric unit tests. These pass without any running services and complete in
under 30 seconds.

Expected output (abbreviated):

```
..............................
30 passed in 12.44s
```

### Integration tests only (requires live services)

```bash
pytest -m integration
```

This runs end-to-end ingestion, retrieval, and Groq provider tests. All three
services (MongoDB, Qdrant, Groq API key) must be available.

### All tests

```bash
pytest
```

Runs both suites. Expect integration tests to take 60–120 seconds due to
Groq API calls and embedding model warm-up.

### Verbose output

```bash
pytest -v
```

Shows each test name and pass/fail status individually.

---

## 9. Production Tips

### Rate limit management across three model buckets

Groq enforces limits per model, per minute (TPM) and per day (TPD). Because
Dynamic-RAG uses three distinct models, each has its own independent quota.
Rate limits on `llama-3.3-70b-versatile` (the generator) are typically the
binding constraint under load.

The system automatically retries rate-limited calls with exponential backoff
(base 2 seconds, max wait 30 seconds, up to 3 retries). If the generator is
still rate-limited after all retries, the fallback model
`llama-3.1-8b-instant` is used for that single request. This is logged at
WARNING level.

To check current usage and remaining quota, visit the Groq Console at
`https://console.groq.com` and navigate to the Usage tab.

### Reading the Groq dashboard

Useful signals from the Groq dashboard:

- **Requests/minute by model**: confirm that `llama-3.3-70b-versatile` and
  `qwen/qwen3-32b` are not saturating their per-minute limits.
- **Token throughput**: if token throughput is near the TPM limit, responses
  will slow because the client is waiting for retry backoff.
- **Error rate**: a nonzero error rate during normal chat usage (not benchmark
  runs) indicates systematic rate limiting; reduce concurrent users or upgrade
  the Groq plan.

### What to do when a model is decommissioned

Groq occasionally deprecates model IDs. If you see an error like
`model 'X' not found` or `model 'X' has been deprecated`:

1. Check the current available model list:

   ```bash
   curl https://api.groq.com/openai/v1/models \
     -H "Authorization: Bearer $GROQ_API_KEY" | python -m json.tool | grep '"id"'
   ```

2. Find a replacement model in the same capability tier.

3. Update the relevant variable in your `.env` file:

   ```
   DEFAULT_LLM=llama-3.3-70b-versatile
   FAST_MODEL=llama-3.1-8b-instant
   CRITIC_MODEL=qwen/qwen3-32b
   ```

4. Restart the FastAPI server. The planner, generator, and verifier all read
   these values from `settings` at import time, so a server restart is
   required for the change to take effect.

### The 15-second benchmark pacing and why it matters

The pacing delay in the benchmark pipeline is not a workaround — it reflects
the actual throughput constraint imposed by Groq's per-model rate limits at
the free tier. If you upgrade to a paid Groq plan with higher TPM limits, you
can reduce or remove the pacing. Edit `evaluation/pipeline_exec.py` and adjust
the `time.sleep` value.

However, removing pacing entirely from a large benchmark run without verified
higher quotas will produce intermittent 429 errors that inflate the
`Failure Count` metric and may produce misleadingly low Faithfulness scores
(because the verifier call fails and defaults to 0 rather than a genuine
judgment).

### Corpus updates and the knowledge-base description

When you add new documents to the corpus, the `KNOWLEDGE_BASE_DESCRIPTION`
setting in `.env` (and in `src/config.py`) should also be updated to reflect
the new topics. The planner uses this description verbatim to decide whether a
query can be answered internally or requires web research. Keeping it accurate
improves routing accuracy (Gate C) and reduces unnecessary web search calls.

The corpus description is also rebuilt dynamically from indexed document
metadata via `corpus_description_builder`. After a CLI ingestion, the dynamic
description updates automatically on the next query. The static
`KNOWLEDGE_BASE_DESCRIPTION` in `.env` acts as a fallback and override —
keep it roughly in sync with what is actually indexed.
