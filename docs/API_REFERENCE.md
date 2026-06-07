# Dynamic-RAG API Reference

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
   - [GET /health](#get-health)
   - [POST /chat/query](#post-chatquery)
   - [GET /chat/{session_id}](#get-chatsession_id)
   - [GET /query/{query_id}/sources](#get-queryquery_idsources)
   - [POST /documents/upload](#post-documentsupload)
   - [GET /system/metrics](#get-systemmetrics)
4. [Query ID Workflow](#query-id-workflow)
5. [Session Continuity Workflow](#session-continuity-workflow)
6. [Error Reference](#error-reference)
7. [Rate Limits](#rate-limits)
8. [Extending the API](#extending-the-api)

---

## Overview

The Dynamic-RAG API is a RESTful HTTP API exposed by a FastAPI application running locally. It provides endpoints for document ingestion, conversational query answering, source attribution, session history, and system observability.

| Property | Value |
|---|---|
| Base URL | `http://localhost:8000` |
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| OpenAPI schema | `http://localhost:8000/openapi.json` |
| Default content-type | `application/json` |
| File upload content-type | `multipart/form-data` |
| Python version | 3.11.9 |
| FastAPI version | 0.136.3 |

All request and response bodies use `application/json` unless the endpoint involves file upload, which uses `multipart/form-data`. All responses include standard HTTP status codes. Timestamps are in ISO 8601 format (UTC).

The system is backed by:
- **Qdrant** (vector store, `http://localhost:6333`) — stores 384-dimensional embeddings produced by `BAAI/bge-small-en-v1.5`
- **MongoDB** (`mongodb://localhost:27017`) — stores sessions, traces, query records, and memory
- **Groq** — LLM provider for planning, generation, and critique
- **Tavily** — web search provider for the `web_research` route

The RAG pipeline is orchestrated via a LangGraph graph with the following node sequence:

```
context_loader → planner → [route node] → generate → verify → [retry/format] → format → persist → END
```

The five available routes are: `internal_rag`, `web_research`, `hybrid`, `memory`, `direct_generation`.

---

## Authentication

The API currently has no authentication. It is designed for local deployment and trusts all requests.

**For production deployment**, add API key middleware in `src/api/main.py` before exposing the service publicly. A recommended pattern is an `X-API-Key` header validated against an environment variable or secrets store. FastAPI's dependency injection system makes this straightforward to bolt on without modifying route handlers.

---

## Endpoints

---

### GET /health

Returns the operational status of the API and its backing services.

#### Request

No parameters, no request body.

#### Response

**Status 200 — application/json**

| Field | Type | Description |
|---|---|---|
| `status` | string | Overall system status. One of `"healthy"`, `"degraded"`, `"unreachable"`. `"degraded"` means at least one service is disconnected but the API is still accepting requests. |
| `app_name` | string | Application name, e.g. `"Dynamic-RAG"`. |
| `environment` | string | Deployment environment, e.g. `"development"` or `"production"`. |
| `timestamp` | string | ISO 8601 UTC timestamp of the health check. |
| `services` | object | Sub-status for each backing service. |
| `services.mongodb` | string | `"connected"` or `"disconnected"`. |
| `services.qdrant` | string | `"connected"` or `"disconnected"`. |

**Example response (healthy)**

```json
{
  "status": "healthy",
  "app_name": "Dynamic-RAG",
  "environment": "development",
  "timestamp": "2026-06-07T10:34:21.004Z",
  "services": {
    "mongodb": "connected",
    "qdrant": "connected"
  }
}
```

**Example response (degraded)**

```json
{
  "status": "degraded",
  "app_name": "Dynamic-RAG",
  "environment": "development",
  "timestamp": "2026-06-07T10:34:21.004Z",
  "services": {
    "mongodb": "connected",
    "qdrant": "disconnected"
  }
}
```

#### Status Codes

| Code | Meaning |
|---|---|
| 200 | Health check completed. Inspect `status` field — a 200 does not guarantee `"healthy"`. |
| 500 | Health check itself failed unexpectedly. |

#### curl

```bash
curl -X GET http://localhost:8000/health
```

#### Python

```python
import requests

response = requests.get("http://localhost:8000/health")
response.raise_for_status()
health = response.json()

print(health["status"])           # "healthy" | "degraded" | "unreachable"
print(health["services"])         # {"mongodb": "connected", "qdrant": "connected"}
```

#### Notes

- A `200` response with `status: "degraded"` is expected when one of the backing services is temporarily unavailable. The API remains reachable but query answering may fail.
- Use this endpoint for liveness and readiness probes in containerised deployments.

---

### POST /chat/query

Submit a natural language query to the RAG pipeline. The system plans a retrieval route, assembles evidence, generates an answer, verifies faithfulness, and returns a structured response.

#### Request

**Content-Type: application/json**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | The natural language question or instruction. Cannot be empty. |
| `session_id` | string | No | An existing session ID to continue a conversation. If omitted, a new session is created and a fresh `session_id` is returned. Reuse the returned value for all follow-up questions to maintain context. |

**Example request body**

```json
{
  "query": "What triggered the escalation of the Russia-Ukraine conflict in 2022?",
  "session_id": "sess_abc123"
}
```

#### Response

**Status 200 — application/json**

| Field | Type | Description |
|---|---|---|
| `answer` | string | The generated answer. If the system abstained, this will be an explicit abstention message rather than a hallucinated answer. |
| `query_id` | string | A unique identifier for this specific query execution. Use this with `GET /query/{query_id}/sources` to retrieve source attribution. |
| `session_id` | string | The session identifier (echoed back, or newly generated if none was provided). |
| `route` | string | The retrieval route chosen by the planner. One of `internal_rag`, `web_research`, `hybrid`, `memory`, `direct_generation`. |
| `confidence` | float | Overall confidence score in the answer, in the range [0.0, 1.0]. Derived from retrieval scores and verifier output. |
| `faithfulness_score` | float or null | Faithfulness score produced by the verifier (critic model). In range [0.0, 1.0]. `null` if verification was skipped (e.g., `direct_generation` route). |
| `status` | string | Execution outcome. One of `"success"`, `"abstained"`, `"error"`. |
| `sources` | array | Abbreviated source list. Each element contains `source_type`, `chunk_id`, and `score`. For full source metadata, use `GET /query/{query_id}/sources`. |
| `sources[].source_type` | string | `"internal"`, `"web"`, or `"memory"`. |
| `sources[].chunk_id` | string | Unique chunk identifier within the document store. |
| `sources[].score` | float | Relevance score after reranking, in range [0.0, 1.0]. |
| `latency_ms` | float | Total end-to-end latency in milliseconds for this query, measured server-side. |

**Example response**

```json
{
  "answer": "The escalation of the Russia-Ukraine conflict in 2022 was primarily triggered by Russia's full-scale military invasion on 24 February 2022. This followed years of tension stemming from the 2014 Maidan Revolution, Russia's annexation of Crimea, and the conflict in the Donbas region. NATO expansion eastward and Ukraine's aspirations toward EU and NATO membership were cited by Russia as security concerns motivating the invasion.",
  "query_id": "qry_7f3c2a1e9b04",
  "session_id": "sess_abc123",
  "route": "internal_rag",
  "confidence": 0.91,
  "faithfulness_score": 0.98,
  "status": "success",
  "sources": [
    {
      "source_type": "internal",
      "chunk_id": "doc_35427eba9d37_chunk_012",
      "score": 0.94
    },
    {
      "source_type": "internal",
      "chunk_id": "doc_649bc4198e06_chunk_007",
      "score": 0.89
    }
  ],
  "latency_ms": 2341.7
}
```

**Example abstention response**

```json
{
  "answer": "I was unable to find sufficient information in the available sources to answer this question confidently. Please try rephrasing or uploading relevant documents.",
  "query_id": "qry_9a1b3c5d7e2f",
  "session_id": "sess_abc123",
  "route": "internal_rag",
  "confidence": 0.21,
  "faithfulness_score": null,
  "status": "abstained",
  "sources": [],
  "latency_ms": 891.2
}
```

#### Status Codes

| Code | Meaning |
|---|---|
| 200 | Query executed. Inspect `status` field — `"abstained"` is a valid 200 outcome, not an error. |
| 400 | Malformed request (e.g., empty query string). |
| 500 | Query execution failed due to internal error (services unreachable, LLM error, etc.). |

#### curl

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What triggered the escalation of the Russia-Ukraine conflict in 2022?",
    "session_id": "sess_abc123"
  }'
```

**First query (no existing session):**

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are the main parties in the Israel-Palestine conflict?"}'
```

#### Python

```python
import requests

BASE_URL = "http://localhost:8000"

payload = {
    "query": "What triggered the escalation of the Russia-Ukraine conflict in 2022?",
    "session_id": "sess_abc123"   # omit to start a new session
}

response = requests.post(f"{BASE_URL}/chat/query", json=payload)
response.raise_for_status()
data = response.json()

print("Answer:", data["answer"])
print("Route:", data["route"])
print("Confidence:", data["confidence"])
print("Faithfulness:", data["faithfulness_score"])
print("Status:", data["status"])
print("Query ID:", data["query_id"])        # save this to fetch sources later
print("Session ID:", data["session_id"])    # reuse for follow-up questions
print("Latency (ms):", data["latency_ms"])
```

#### Notes

- **Session continuity**: pass the same `session_id` in all follow-up questions. The system loads recent conversation history and semantic memory from MongoDB before planning, enabling it to understand references like "same as before", "tell me more", or "what did we say about X".
- **Route selection**: the planner (llama-3.1-8b-instant) chooses the retrieval route automatically based on query characteristics. You cannot force a specific route via the API — this is intentional to keep the interface simple and route selection observable via the `route` field.
- **Abstention**: a `status` of `"abstained"` with a low `confidence` score means the verifier determined the answer was not sufficiently grounded. The system prefers explicit abstention over confident hallucination.
- **`faithfulness_score`**: this is produced by the critic model (qwen/qwen3-32b) and measures how well the answer is supported by retrieved evidence. Scores above 0.9 indicate high fidelity.
- **Sources in response vs. full sources**: the `sources` array in this response is abbreviated. For the full source metadata (document title, page number, etc.) call `GET /query/{query_id}/sources`.

---

### GET /chat/{session_id}

Retrieve the full conversation history for a session.

#### Request

**Path Parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | Yes | The session ID returned by `POST /chat/query`. |

No query parameters. No request body.

#### Response

**Status 200 — application/json**

| Field | Type | Description |
|---|---|---|
| `session_id` | string | The session identifier. |
| `message_count` | integer | Total number of query-answer turns in this session. |
| `messages` | array | Ordered list of conversation turns, oldest first. |
| `messages[].query` | string | The user's question for this turn. |
| `messages[].answer` | string | The system's answer for this turn. |
| `messages[].route` | string | The retrieval route used for this turn. |
| `messages[].confidence` | float | Confidence score for this turn's answer. |
| `messages[].timestamp` | string | ISO 8601 UTC timestamp of when this turn was executed. |

**Example response**

```json
{
  "session_id": "sess_abc123",
  "message_count": 2,
  "messages": [
    {
      "query": "Who are the main parties in the Israel-Palestine conflict?",
      "answer": "The main parties in the Israel-Palestine conflict are the State of Israel and the Palestinian people, represented politically by the Palestinian Authority in the West Bank and Hamas in the Gaza Strip...",
      "route": "internal_rag",
      "confidence": 0.88,
      "timestamp": "2026-06-07T10:12:44.312Z"
    },
    {
      "query": "What role does the United States play in this conflict?",
      "answer": "The United States has historically been a close ally of Israel, providing military aid, diplomatic support at the UN, and brokering multiple peace negotiations...",
      "route": "hybrid",
      "confidence": 0.84,
      "timestamp": "2026-06-07T10:14:07.551Z"
    }
  ]
}
```

#### Status Codes

| Code | Meaning |
|---|---|
| 200 | Session found and history returned. |
| 404 | No session found for the given `session_id`. |
| 500 | Unexpected server error while reading session history. |

#### curl

```bash
curl -X GET http://localhost:8000/chat/sess_abc123
```

#### Python

```python
import requests

BASE_URL = "http://localhost:8000"
session_id = "sess_abc123"

response = requests.get(f"{BASE_URL}/chat/{session_id}")

if response.status_code == 404:
    print("Session not found")
else:
    response.raise_for_status()
    data = response.json()
    print(f"Session has {data['message_count']} turns")
    for i, msg in enumerate(data["messages"], 1):
        print(f"\nTurn {i} [{msg['route']}] confidence={msg['confidence']}")
        print(f"  Q: {msg['query']}")
        print(f"  A: {msg['answer'][:120]}...")
```

#### Notes

- Messages are returned in chronological order (oldest turn first).
- This endpoint is useful for rendering conversation history in a UI or for auditing what the system said in a previous session.
- Session data is persisted in MongoDB. Sessions are not time-limited by default, but this may be subject to MongoDB collection TTL settings configured in the deployment.

---

### GET /query/{query_id}/sources

Retrieve full source attribution for a specific query. Use the `query_id` returned by `POST /chat/query`.

#### Request

**Path Parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query_id` | string | Yes | The unique query identifier returned in the `query_id` field of `POST /chat/query`. |

No query parameters. No request body.

#### Response

**Status 200 — application/json**

| Field | Type | Description |
|---|---|---|
| `query_id` | string | The query identifier. |
| `route` | string | The retrieval route used when the query was executed. |
| `num_sources` | integer | Total number of sources returned. |
| `sources` | array | Full source metadata for each piece of evidence used. |
| `sources[].source_type` | string | Origin of this source. One of `"internal"` (from Qdrant document store), `"web"` (from Tavily web search), or `"memory"` (from session memory). |
| `sources[].source_id` | string | Document-level identifier. For internal sources, this is the document ID (content-hash based). For web sources, this is the URL. |
| `sources[].chunk_id` | string | Chunk-level identifier within the document. For web sources, this may be a derived segment ID. |
| `sources[].page` | integer or null | Page number within the source document. `null` for web and memory sources. |
| `sources[].score` | float | Final relevance score after reranking by the cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Range [0.0, 1.0]. |
| `sources[].title` | string | Human-readable title. For internal sources, derived from the document filename. For web sources, the page title returned by Tavily. |

**Example response (internal_rag route)**

```json
{
  "query_id": "qry_7f3c2a1e9b04",
  "route": "internal_rag",
  "num_sources": 3,
  "sources": [
    {
      "source_type": "internal",
      "source_id": "doc_35427eba9d37",
      "chunk_id": "doc_35427eba9d37_chunk_012",
      "page": 4,
      "score": 0.94,
      "title": "conflict_escalation_ukraine_2022.pdf"
    },
    {
      "source_type": "internal",
      "source_id": "doc_649bc4198e06",
      "chunk_id": "doc_649bc4198e06_chunk_007",
      "page": 2,
      "score": 0.89,
      "title": "geopolitical_events_eastern_europe.pdf"
    },
    {
      "source_type": "internal",
      "source_id": "doc_836098d6074c",
      "chunk_id": "doc_836098d6074c_chunk_019",
      "page": 7,
      "score": 0.81,
      "title": "global_crisis_report_2022.pdf"
    }
  ]
}
```

**Example response (web_research route)**

```json
{
  "query_id": "qry_9b2d4f6a8c0e",
  "route": "web_research",
  "num_sources": 2,
  "sources": [
    {
      "source_type": "web",
      "source_id": "https://www.cfr.org/global-conflict-tracker/conflict/war-ukraine",
      "chunk_id": "web_qry_9b2d4f6a8c0e_seg_0",
      "page": null,
      "score": 0.91,
      "title": "War in Ukraine | Global Conflict Tracker | CFR"
    },
    {
      "source_type": "web",
      "source_id": "https://www.bbc.com/news/world-europe-56720589",
      "chunk_id": "web_qry_9b2d4f6a8c0e_seg_1",
      "page": null,
      "score": 0.85,
      "title": "Ukraine crisis: What is happening and why? - BBC News"
    }
  ]
}
```

#### Status Codes

| Code | Meaning |
|---|---|
| 200 | Sources found and returned. |
| 404 | No query record found for the given `query_id`. This happens if the `query_id` was never created or the trace was not persisted (e.g., due to a persistence failure). |
| 500 | Unexpected server error while reading source data. |

#### curl

```bash
curl -X GET http://localhost:8000/query/qry_7f3c2a1e9b04/sources
```

#### Python

```python
import requests

BASE_URL = "http://localhost:8000"
query_id = "qry_7f3c2a1e9b04"

response = requests.get(f"{BASE_URL}/query/{query_id}/sources")

if response.status_code == 404:
    print("Query ID not found — check the query_id from POST /chat/query")
else:
    response.raise_for_status()
    data = response.json()
    print(f"Route: {data['route']}, Sources: {data['num_sources']}")
    for src in data["sources"]:
        print(f"  [{src['source_type']}] score={src['score']:.2f} page={src['page']} — {src['title']}")
```

#### Notes

- Sources are listed in descending order of relevance `score` (highest first).
- The pipeline retrieves `RERANK_TOP_K=20` candidates from the vector store, then the cross-encoder reranker reduces this to `FINAL_TOP_K=8`. The sources returned here are the final post-reranking set actually used to generate the answer.
- The `score` field reflects the cross-encoder score, not the raw vector similarity score. Cross-encoder scores are more reliable for ranking.
- For the `hybrid` route, sources will contain a mix of `"internal"` and `"web"` entries.

---

### POST /documents/upload

Upload and ingest a document into the RAG corpus. The pipeline parses, chunks, embeds, and indexes the document into Qdrant, then updates the BM25 sparse index.

#### Request

**Content-Type: multipart/form-data**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | The document to ingest. Accepted formats: `.pdf`, `.txt`. |

**Supported file types**

| Extension | Parser | Notes |
|---|---|---|
| `.pdf` | PyMuPDF + EasyOCR fallback | Scanned PDFs are processed with EasyOCR (1.7.2). OCR adds significant processing time. |
| `.txt` | Plain text reader | UTF-8 encoding assumed. |

#### Response

**Status 200 — application/json**

| Field | Type | Description |
|---|---|---|
| `document_id` | string | A content-hash-based unique identifier for the document. If the same file is uploaded again, it will produce the same `document_id`. |
| `filename` | string | The original filename as provided in the upload. |
| `status` | string | `"indexed"` if the full pipeline succeeded, `"failed"` if ingestion encountered an error. |
| `message` | string | Human-readable summary, e.g. `"147 chunks indexed into Qdrant"`. |

**Example response (success)**

```json
{
  "document_id": "doc_35427eba9d37",
  "filename": "conflict_escalation_ukraine_2022.pdf",
  "status": "indexed",
  "message": "147 chunks indexed into Qdrant"
}
```

**Example response (failure)**

```json
{
  "document_id": null,
  "filename": "scan_corrupt.pdf",
  "status": "failed",
  "message": "Parser failed: unable to extract text from document"
}
```

#### Status Codes

| Code | Meaning |
|---|---|
| 200 | Request accepted and pipeline ran. Inspect `status` field — `"failed"` is a valid 200 outcome. |
| 400 | Unsupported file type (not `.pdf` or `.txt`). |
| 500 | Ingestion pipeline crashed unexpectedly. |

#### curl

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/conflict_escalation_ukraine_2022.pdf"
```

#### Python

```python
import requests

BASE_URL = "http://localhost:8000"
file_path = "/path/to/conflict_escalation_ukraine_2022.pdf"

with open(file_path, "rb") as f:
    response = requests.post(
        f"{BASE_URL}/documents/upload",
        files={"file": (file_path.split("/")[-1], f, "application/pdf")}
    )

response.raise_for_status()
data = response.json()

print("Document ID:", data["document_id"])
print("Status:", data["status"])
print("Message:", data["message"])
```

#### Notes

- **Synchronous processing**: the pipeline runs synchronously within the request. The HTTP response is not returned until chunking, embedding, and indexing are complete. For large PDFs (100+ pages), this can take 30 seconds to several minutes if OCR is required.
- **OCR**: EasyOCR is invoked automatically as a fallback when PyMuPDF cannot extract text from a page (typical for scanned documents). OCR is GPU-accelerated if a CUDA device is available; otherwise it runs on CPU and is significantly slower.
- **Idempotency**: uploading the same file twice produces the same `document_id` (content-hash based). Duplicate chunks may be inserted into Qdrant unless deduplication is enforced at the indexer level. Check your Qdrant collection configuration.
- **BM25 index update**: after successful ingestion, the in-memory BM25 corpus is rebuilt to include the new document's tokens, making the new content immediately available for sparse retrieval.
- **Corpus description update**: the system updates its internal corpus description metadata, which the planner uses when deciding whether `internal_rag` or `web_research` is more appropriate for a given query.

---

### GET /system/metrics

Returns aggregated operational metrics for the current server session.

#### Request

No parameters. No request body.

#### Response

**Status 200 — application/json**

| Field | Type | Description |
|---|---|---|
| `status` | string | Service status at time of metrics collection. `"healthy"` or `"degraded"`. |
| `total_requests` | integer | Total number of queries processed since the server started (or since last reset). |
| `route_distribution` | object | Map from route name to query count. Keys: `internal_rag`, `web_research`, `hybrid`, `memory`, `direct_generation`. |
| `mean_latency_ms` | float | Mean end-to-end query latency in milliseconds across all requests. |
| `p95_latency_ms` | float | 95th percentile latency in milliseconds. |
| `p99_latency_ms` | float | 99th percentile latency in milliseconds. |
| `mean_confidence` | float | Mean confidence score across all non-abstained responses. Range [0.0, 1.0]. |
| `abstention_rate` | float | Fraction of queries that resulted in `"abstained"` status. Range [0.0, 1.0]. |
| `total_retries` | integer | Total number of LangGraph retry cycles triggered (e.g., due to low faithfulness scores prompting re-generation). |
| `total_cost_usd` | float | Estimated total Groq API cost in USD for all queries since server start, based on token counts and Groq pricing. |
| `mean_cost_per_query_usd` | float | Mean estimated cost per query in USD. |

**Example response**

```json
{
  "status": "healthy",
  "total_requests": 214,
  "route_distribution": {
    "internal_rag": 142,
    "web_research": 31,
    "hybrid": 24,
    "memory": 11,
    "direct_generation": 6
  },
  "mean_latency_ms": 2187.4,
  "p95_latency_ms": 4831.2,
  "p99_latency_ms": 7102.5,
  "mean_confidence": 0.847,
  "abstention_rate": 0.037,
  "total_retries": 8,
  "total_cost_usd": 0.1423,
  "mean_cost_per_query_usd": 0.000665
}
```

#### Status Codes

| Code | Meaning |
|---|---|
| 200 | Metrics collected and returned successfully. |
| 500 | Metrics collection failed (e.g., metrics store unavailable). |

#### curl

```bash
curl -X GET http://localhost:8000/system/metrics
```

#### Python

```python
import requests

BASE_URL = "http://localhost:8000"

response = requests.get(f"{BASE_URL}/system/metrics")
response.raise_for_status()
metrics = response.json()

print(f"Total requests: {metrics['total_requests']}")
print(f"Route distribution: {metrics['route_distribution']}")
print(f"Mean latency: {metrics['mean_latency_ms']:.1f} ms")
print(f"P95 latency:  {metrics['p95_latency_ms']:.1f} ms")
print(f"Abstention rate: {metrics['abstention_rate']:.1%}")
print(f"Total cost: ${metrics['total_cost_usd']:.4f}")
```

#### Notes

- Metrics are accumulated in-memory on the server process. They reset when the server restarts.
- `abstention_rate` is computed as `abstained_count / total_requests`. A healthy system operating on a well-formed corpus should have abstention rates below 10%. The current benchmark abstention rate is 0.6 (60%) on the out-of-domain portion of the test set, which is the intended behavior.
- `total_retries` counts LangGraph retry edges — these occur when the verifier returns a faithfulness score below the configured threshold and the graph re-enters the generation node with tighter constraints.
- Cost estimates are approximations based on token counts reported by the Groq API. Actual billing may differ.

---

## Query ID Workflow

A common pattern when building UIs or citation-aware clients is to fetch sources separately from the answer, since sources are verbose and not always needed for every display context.

**Step 1 — Submit the query**

```
POST /chat/query
{
  "query": "What caused the 2008 financial crisis?",
  "session_id": "sess_xyz789"
}
```

The response includes `query_id: "qry_4a9c1e3f7b20"`.

**Step 2 — Display the answer immediately**

The UI can render the `answer`, `confidence`, `route`, and `status` fields without waiting for sources to be fetched separately.

**Step 3 — Fetch full source attribution**

```
GET /query/qry_4a9c1e3f7b20/sources
```

The response contains the full list of documents, chunk IDs, page numbers, and scores. Render these as footnotes, citations, or an expandable panel.

**Python pattern**

```python
import requests

BASE_URL = "http://localhost:8000"
SESSION_ID = "sess_xyz789"

# Step 1: submit the query
query_resp = requests.post(f"{BASE_URL}/chat/query", json={
    "query": "What caused the 2008 financial crisis?",
    "session_id": SESSION_ID
})
query_resp.raise_for_status()
result = query_resp.json()

print("Answer:", result["answer"])
print("Route:", result["route"])
print("Confidence:", result["confidence"])

# Step 2: fetch sources for citation rendering
query_id = result["query_id"]
sources_resp = requests.get(f"{BASE_URL}/query/{query_id}/sources")
sources_resp.raise_for_status()
sources = sources_resp.json()

print(f"\nSources ({sources['num_sources']}):")
for s in sources["sources"]:
    page = f" p.{s['page']}" if s["page"] else ""
    print(f"  - [{s['source_type']}]{page} score={s['score']:.2f} — {s['title']}")
```

This two-step pattern keeps the primary query response lightweight while making full attribution available on demand.

---

## Session Continuity Workflow

The `session_id` field is the mechanism for multi-turn conversation. Pass the same `session_id` in every query to maintain context. The system loads recent conversation history and semantic memory before each planning step, allowing it to resolve references, pronouns, and follow-up phrasing.

**Example: 3-turn conversation**

### Turn 1 — Start a new session

Do not provide `session_id`. The system creates one and returns it.

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are the major stakeholders in the South China Sea dispute?"}'
```

Response excerpt:
```json
{
  "answer": "The major stakeholders in the South China Sea dispute include China, which claims the majority of the sea under its nine-dash line; Vietnam, the Philippines, Malaysia, and Brunei, which have overlapping territorial claims; and Taiwan. The United States is also a significant actor through its freedom of navigation operations.",
  "session_id": "sess_gen_8f3d1a2c",
  "route": "internal_rag",
  "confidence": 0.92
}
```

Save `session_id = "sess_gen_8f3d1a2c"`.

### Turn 2 — Follow up using a pronoun reference

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is China'\''s primary legal justification for its claim?",
    "session_id": "sess_gen_8f3d1a2c"
  }'
```

Response excerpt:
```json
{
  "answer": "China's primary legal justification for its expansive claim in the South China Sea is the so-called nine-dash line, a demarcation it asserts reflects historic fishing and navigation rights predating modern international law. China does not formally invoke UNCLOS as the basis for this claim, and the Permanent Court of Arbitration ruled in 2016 that the nine-dash line has no legal basis under UNCLOS.",
  "session_id": "sess_gen_8f3d1a2c",
  "route": "internal_rag",
  "confidence": 0.89
}
```

### Turn 3 — Request synthesis across topics discussed

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key tensions we have discussed so far",
    "session_id": "sess_gen_8f3d1a2c"
  }'
```

Response excerpt:
```json
{
  "answer": "Based on our conversation, the key tensions in the South China Sea dispute are: (1) China's nine-dash line claim versus the overlapping territorial claims of Vietnam, the Philippines, Malaysia, Brunei, and Taiwan; (2) China's rejection of the 2016 UNCLOS arbitration ruling; and (3) the United States' freedom of navigation operations, which China views as provocative.",
  "session_id": "sess_gen_8f3d1a2c",
  "route": "memory",
  "confidence": 0.86
}
```

Note that Turn 3 was routed via `memory` — the planner recognized the query was asking for a summary of previous exchanges rather than new document retrieval.

**Python: complete 3-turn conversation**

```python
import requests

BASE_URL = "http://localhost:8000"
session_id = None  # will be set from the first response

queries = [
    "Who are the major stakeholders in the South China Sea dispute?",
    "What is China's primary legal justification for its claim?",
    "Summarize the key tensions we have discussed so far",
]

for i, query in enumerate(queries, 1):
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(f"{BASE_URL}/chat/query", json=payload)
    response.raise_for_status()
    data = response.json()

    # capture session_id from the first response
    if session_id is None:
        session_id = data["session_id"]
        print(f"Session started: {session_id}\n")

    print(f"Turn {i} [{data['route']}] confidence={data['confidence']:.2f}")
    print(f"  Q: {query}")
    print(f"  A: {data['answer'][:200]}...")
    print()
```

---

## Error Reference

| HTTP Status | Error Code | Cause | Fix |
|---|---|---|---|
| 400 | `EMPTY_QUERY` | `query` field is an empty string or contains only whitespace. | Ensure the `query` field is a non-empty string before sending the request. |
| 400 | `UNSUPPORTED_FILE_TYPE` | The uploaded file has an extension other than `.pdf` or `.txt`. | Convert the document to PDF or TXT before uploading, or contact the maintainers to add support for the new format. |
| 404 | `QUERY_NOT_FOUND` | The `query_id` provided to `GET /query/{query_id}/sources` does not exist in the trace store. | Verify the `query_id` was copied correctly from the `POST /chat/query` response. If persistence failed silently, re-run the query. |
| 404 | `SESSION_NOT_FOUND` | The `session_id` provided to `GET /chat/{session_id}` does not exist in MongoDB. | Verify the `session_id`. New sessions are only created on the first successful `POST /chat/query` call. |
| 500 | `QUERY_EXECUTION_FAILED` | The LangGraph pipeline raised an unhandled exception. Most commonly caused by Qdrant or MongoDB being unreachable, or a Groq API error. | Check that Qdrant (`http://localhost:6333`) and MongoDB (`mongodb://localhost:27017`) are running. Check the Groq API key in `.env`. Inspect `logs/dynamic_rag.log` for the stack trace. |
| 500 | `INGESTION_FAILED` | The document ingestion pipeline failed. Can be caused by a corrupt file, a PDF with no extractable text and OCR failure, an embedding model error, or a Qdrant write failure. | Inspect `logs/dynamic_rag.log`. Verify the file is a valid PDF or UTF-8 TXT. Ensure Qdrant is running and has sufficient disk space. |

---

## Rate Limits

The Dynamic-RAG API itself has no built-in rate limiting. However, the system calls the Groq API internally for every query (planner, generator, and verifier steps), and Groq imposes its own rate limits per model.

**Groq rate limits (as of June 2026)**

| Model | Role | Tokens Per Minute (TPM) |
|---|---|---|
| `llama-3.1-8b-instant` | Planner | 6,000 TPM |
| `llama-3.3-70b-versatile` | Generator | 12,000 TPM |
| `qwen/qwen3-32b` | Critic / Verifier | Separate bucket (check Groq console) |

These limits apply to your Groq API key across all callers, not just Dynamic-RAG. If the system appears to slow down significantly or returns errors referencing rate limits, it is likely backing off from Groq's rate limits.

The `GroqProvider` in `src/models/groq_provider.py` handles Groq rate limit responses (HTTP 429) automatically with exponential backoff and retry logic. Retries are counted in the `total_retries` field of `GET /system/metrics`.

To reduce exposure to rate limits:
- Keep the critic model (`qwen/qwen3-32b`) usage bounded by raising the faithfulness threshold only when needed.
- For high-throughput deployments, consider caching answers for repeated queries.
- Monitor `mean_latency_ms` via `GET /system/metrics` — sustained increases often indicate rate limit backoff.

---

## Extending the API

### Adding new routes

New API route modules belong in `src/api/routes/`. Each module should export an `APIRouter` instance. The thin-handler pattern must be preserved: route handler functions should contain no business logic. They are responsible only for:

1. parsing and validating the incoming request schema,
2. calling into the appropriate service or use-case layer,
3. serializing and returning the response schema.

All business logic belongs in `src/` service modules (e.g., `src/graph/`, `src/retrieval/`, `src/generation/`).

**Example: adding a `GET /documents` route**

Create `src/api/routes/documents_list.py`:

```python
from fastapi import APIRouter
from src.database.repositories import DocumentRepository

router = APIRouter()

@router.get("/documents")
async def list_documents():
    repo = DocumentRepository()
    docs = await repo.list_all()
    return {"documents": docs, "count": len(docs)}
```

### Registering a new router

Open `src/api/main.py` and add the router with `app.include_router()`:

```python
from src.api.routes.documents_list import router as documents_list_router

app.include_router(documents_list_router, prefix="/documents", tags=["documents"])
```

### Schema definitions

Request and response schemas are Pydantic models located in `src/api/schemas/`. Define new schemas there to keep type validation co-located and to ensure they appear correctly in the Swagger UI at `/docs`.

### Middleware

Cross-cutting concerns (authentication, logging, CORS, request ID injection) belong as ASGI middleware registered in `src/api/main.py` via `app.add_middleware()`. Do not embed these concerns in individual route handlers.
