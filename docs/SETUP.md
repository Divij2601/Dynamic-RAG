# Dynamic-RAG Setup Guide

This guide walks you from a fresh machine to a fully running Dynamic-RAG
instance. Follow every section in order the first time. Experienced users
can jump to the relevant section.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installing MongoDB](#2-installing-mongodb)
3. [Installing Qdrant](#3-installing-qdrant)
4. [Python Environment](#4-python-environment)
5. [Environment Configuration](#5-environment-configuration)
6. [Getting API Keys](#6-getting-api-keys)
7. [Groq Model Reference](#7-groq-model-reference)
8. [First Run Verification](#8-first-run-verification)
9. [Ingesting Your First Document](#9-ingesting-your-first-document)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Before starting, make sure you have every item on this list. Missing any one
of them will cause the setup to fail at a later step.

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.11 | 3.11.9 was used in development. 3.12+ is untested. |
| Git | Any recent | For cloning the repo. |
| MongoDB | 6.0 | Community Edition is free. |
| Qdrant | 1.9+ | Binary or Docker. See Section 3. |
| RAM | 4 GB | 8 GB recommended. Embedding model loads into RAM. |
| Disk | 3 GB free | PyTorch + sentence-transformers + EasyOCR ~1.5 GB, models ~500 MB on first run. |
| Groq API key | Free tier | ~14,400 requests/day free. Required for LLM calls. |
| Tavily API key | Free tier | 1,000 searches/month free. Required for web research route. |

The system does not require a GPU. All embedding and reranking runs on CPU.
First-run model downloads are automated by `sentence-transformers` and
`easyocr`; they require a working internet connection.

---

## 2. Installing MongoDB

MongoDB stores conversation sessions, query traces, and document metadata.

### Windows

1. Download the MSI installer from
   https://www.mongodb.com/try/download/community
   (select version 6.x, Windows, MSI).
2. Run the installer. Accept defaults. Install as a Windows service so it
   starts automatically.
3. Add `C:\Program Files\MongoDB\Server\6.0\bin` to your `PATH`.

Start / stop the service:

```powershell
# Start
net start MongoDB

# Stop
net stop MongoDB

# Verify
mongosh --eval "db.adminCommand('ping')"
```

### macOS

```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb-community@6.0

# Verify
mongosh --eval "db.adminCommand('ping')"
```

### Linux (Ubuntu / Debian)

```bash
# Import the public key
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc \
  | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor

# Add the repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] \
  https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" \
  | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

sudo apt-get update
sudo apt-get install -y mongodb-org

# Start and enable on boot
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
mongosh --eval "db.adminCommand('ping')"
```

Expected output from the verify command:

```
{ ok: 1 }
```

The default URI `mongodb://localhost:27017` needs no authentication for a
local installation. Do not expose this port to the internet.

---

## 3. Installing Qdrant

Qdrant is the vector database that stores document embeddings and semantic
memory. Two installation options are provided. Choose one.

### Option A: Docker (recommended)

This is the simplest approach if Docker is already installed.

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`:

```powershell
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v "${PWD}/qdrant_storage:/qdrant/storage" `
  qdrant/qdrant:latest
```

The volume mount persists your vector data across container restarts. Without
it, all indexed documents are lost when the container stops.

Restart Qdrant after a reboot:

```bash
docker start qdrant
```

### Option B: Binary (no Docker required)

Download the latest release binary from
https://github.com/qdrant/qdrant/releases

```bash
# Linux / macOS example
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz
./qdrant
```

On Windows, download the `.zip` for `x86_64-pc-windows-msvc`, extract, and
run `qdrant.exe` from a terminal.

Qdrant listens on port `6333` (REST) and `6334` (gRPC) by default. Data is
stored in `./storage` relative to where the binary runs.

### Verifying Qdrant is running

```bash
curl http://localhost:6333/collections
```

Expected response:

```json
{"result":{"collections":[]},"status":"ok","time":0.0001}
```

If you have already ingested documents, `collections` will list
`dynamic_rag_documents` and `dynamic_rag_memory`.

---

## 4. Python Environment

### Clone and enter the repository

```bash
git clone https://github.com/Divij2601/Dynamic-RAG.git
cd Dynamic-RAG
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

If you see an execution-policy error, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

Your prompt should now show `(venv)` as a prefix.

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, sentence-transformers, EasyOCR, LangGraph, FastAPI,
Streamlit, Groq, Qdrant client, and all other dependencies. Total download
size is approximately 1.5 GB including PyTorch wheels.

**Note on first-run model downloads:** When you first call an embedding or
reranking function, `sentence-transformers` will automatically download
`BAAI/bge-small-en-v1.5` (~130 MB) and
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB) from Hugging Face. EasyOCR
downloads its own detection models (~200 MB) the first time it processes a
scanned PDF. These downloads happen once and are cached in
`~/.cache/huggingface` and `~/.EasyOCR`. Subsequent runs are instant.

---

## 5. Environment Configuration

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` in any text editor. The full contents of `.env.example` are
shown below with every variable explained.

```dotenv
# ============================================================
# Dynamic-RAG Environment Configuration
# ============================================================


# ────────────────────────────────────────────────────────────
# App Config
# ────────────────────────────────────────────────────────────

# Human-readable name shown in API responses and logs.
# No consequence if changed; cosmetic only.
APP_NAME=Dynamic-RAG

# Controls log verbosity and certain FastAPI behaviors.
# Use "development" locally and "production" in deployment.
# Missing/wrong: defaults to "development".
ENVIRONMENT=development

# Set to True to enable verbose output. FastAPI will also
# show detailed error tracebacks in API responses.
# Set to False in production.
DEBUG=True

# Interface the FastAPI server binds to. Use 127.0.0.1 for
# local-only access. Use 0.0.0.0 to expose to the network.
# Missing: defaults to 127.0.0.1.
HOST=127.0.0.1

# Port FastAPI listens on. The Streamlit UI (app.py) is
# hardcoded to call http://localhost:8000, so change both
# if you change this.
# Missing: defaults to 8000.
PORT=8000

# Python logging level for the application logger.
# One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
# Missing: defaults to INFO.
LOG_LEVEL=INFO


# ────────────────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────────────────

# Full MongoDB connection string. For a local default install
# this is always mongodb://localhost:27017. Add credentials
# if your instance requires auth:
# mongodb://username:password@localhost:27017/
# REQUIRED: the app will crash at startup if missing.
MONGO_URI=mongodb://localhost:27017

# Base URL of the Qdrant REST API. For Docker or binary on
# the same machine, this is always http://localhost:6333.
# Change the host/port if Qdrant runs on a remote server.
# REQUIRED: retrieval and indexing will fail if missing.
QDRANT_URL=http://localhost:6333

# API key for Qdrant. Only needed for Qdrant Cloud instances
# or self-hosted deployments with authentication enabled.
# For a local Docker/binary install, leave this empty.
# Missing/empty: no authentication is sent (correct for local).
QDRANT_API_KEY=


# ────────────────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────────────────

# LLM provider. Only "groq" is implemented.
# Missing: defaults to groq. Do not change.
LLM_PROVIDER=groq

# Your Groq API key. Obtain from console.groq.com.
# REQUIRED: every LLM call (planning, generation, criticism)
# will fail with a 401 error if this is missing or wrong.
GROQ_API_KEY=gsk_YOUR_KEY_HERE

# Primary generation model. Used for the main answer
# generation node. Should be a large, high-quality model.
# Recommended: llama-3.3-70b-versatile
# Missing: the app will raise a ValidationError at startup.
DEFAULT_LLM=llama-3.3-70b-versatile

# Fast model used for planning (intent classification, route
# selection). Speed matters more than size here.
# Recommended: llama-3.1-8b-instant
# Missing: the app will raise a ValidationError at startup.
FAST_MODEL=llama-3.1-8b-instant

# Critic model used for faithfulness verification. This model
# reads the generated answer and evidence and scores
# groundedness. qwen3-32b is a reasoning model — it produces
# a chain-of-thought before the verdict. Allocate generous
# MAX_TOKENS (2048+) so the reasoning trace is not truncated.
# Missing: the app will raise a ValidationError at startup.
CRITIC_MODEL=qwen/qwen3-32b

# Sampling temperature for generation. Lower = more
# deterministic. 0.2 is recommended for factual RAG.
# Range: 0.0 to 2.0. Missing: defaults to 0.2.
TEMPERATURE=0.2

# Maximum output tokens for the generator and critic.
# 2048 is the minimum safe value. Increase to 4096 if the
# critic's reasoning traces are being cut off.
# Missing: defaults to 2048.
MAX_TOKENS=2048

# Retry settings for transient LLM errors (rate limits,
# timeouts). LLM_MAX_RETRIES controls how many times a
# failed LLM call is retried before giving up.
LLM_MAX_RETRIES=3

# Base for exponential backoff between retries (seconds).
LLM_BACKOFF_BASE=2.0

# Maximum time to wait between retries (seconds).
LLM_RETRY_MAX_WAIT=30.0

# Fallback model used when the primary model is rate-limited
# beyond LLM_RETRY_MAX_WAIT. Set to empty string to disable.
LLM_FALLBACK_MODEL=llama-3.1-8b-instant

# Maximum conversation turns to keep in full before the
# memory summarizer compresses older turns into a summary.
# Higher = more context, more tokens. Missing: defaults to 8.
MAX_HISTORY_TURNS=8

# Number of recent turns to always keep verbatim after
# summarization. Missing: defaults to 3.
SUMMARY_KEEP_RECENT=3


# ────────────────────────────────────────────────────────────
# Retrieval
# ────────────────────────────────────────────────────────────

# Legacy top-k setting. Kept for backward compatibility.
# The active retrieval parameters are RERANK_TOP_K and
# FINAL_TOP_K below. Missing: defaults to 5.
TOP_K=5

# Number of candidate chunks retrieved by the hybrid
# retriever (dense + sparse) before the cross-encoder
# reranker narrows them down. 20 is the measured optimum for
# the current corpus. Increase for larger corpora.
# Missing: defaults to 20.
RERANK_TOP_K=20

# Final number of chunks passed to the generator after
# reranking. 8 achieves Recall=1.0 on the current benchmark.
# Increasing beyond 8 adds no recall gain but increases
# prompt size and latency. Missing: defaults to 8.
FINAL_TOP_K=8

# Score fusion strategy for hybrid retrieval.
# "weighted" — weighted sum of normalised dense and sparse
#   scores. Best measured mode for the current corpus.
# "rrf" — Reciprocal Rank Fusion. Better for larger or more
#   lexically diverse corpora.
# Missing: defaults to weighted.
FUSION_MODE=weighted

# Weight applied to the dense (embedding) retrieval score
# when FUSION_MODE=weighted. Must sum to 1.0 with
# SPARSE_WEIGHT. Missing: defaults to 0.7.
DENSE_WEIGHT=0.7

# Weight applied to the sparse (BM25) retrieval score when
# FUSION_MODE=weighted. Missing: defaults to 0.3.
SPARSE_WEIGHT=0.3

# The k constant in Reciprocal Rank Fusion. Only used when
# FUSION_MODE=rrf. Higher k = smoother rank distribution.
# Missing: defaults to 60.
RRF_K=60

# Character length of each text chunk during ingestion.
# Larger chunks = more context per retrieved passage but
# slower reranking. Missing: defaults to 1000.
CHUNK_SIZE=1000

# Overlap between consecutive chunks (characters). Prevents
# answer spans from falling at chunk boundaries.
# Missing: defaults to 200.
CHUNK_OVERLAP=200


# ────────────────────────────────────────────────────────────
# Runtime
# ────────────────────────────────────────────────────────────

# HTTP timeout in seconds for outbound API calls (Groq,
# Tavily). Increase for slow network connections.
# Missing: defaults to 30.
REQUEST_TIMEOUT=30

# Maximum number of answer retry cycles in the LangGraph
# (verify -> retry -> generate loop). Each retry costs one
# extra generation call. Missing: defaults to 2.
MAX_RETRIES=2


# ────────────────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────────────────

# Sentence-transformers model used to embed document chunks
# and queries. BAAI/bge-small-en-v1.5 produces 384-dim
# vectors with cosine similarity. Do not change unless you
# also rebuild the Qdrant collection (dimension mismatch
# will cause indexing to fail).
# Missing: defaults to BAAI/bge-small-en-v1.5.
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Cross-encoder model used to rerank the top-k candidates.
# Missing: defaults to cross-encoder/ms-marco-MiniLM-L-6-v2.
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Device for PyTorch inference. "cpu" for all machines.
# "cuda" if you have an NVIDIA GPU — speeds up embedding
# and reranking significantly but is not required.
# Missing: defaults to cpu.
EMBEDDING_DEVICE=cpu

# Number of chunks embedded in a single forward pass.
# Larger batches use more RAM but are faster.
# Missing: defaults to 16.
EMBEDDING_BATCH_SIZE=16

# Qdrant collection name for document chunks.
# Changing this requires deleting and recreating the
# collection (all indexed documents must be re-ingested).
# Missing: defaults to dynamic_rag_documents.
QDRANT_COLLECTION_NAME=dynamic_rag_documents

# Qdrant collection name for semantic conversation memory.
# Missing: defaults to dynamic_rag_memory.
MEMORY_COLLECTION_NAME=dynamic_rag_memory

# Vector dimension. Must match the embedding model output.
# BAAI/bge-small-en-v1.5 outputs 384 dimensions.
# Wrong value: Qdrant will reject the collection creation.
# Missing: defaults to 384.
VECTOR_DIMENSION=384


# ────────────────────────────────────────────────────────────
# Memory
# ────────────────────────────────────────────────────────────

# Conversation memory is stored in MongoDB (turn-by-turn)
# and Qdrant (semantic embeddings). No extra variables are
# needed beyond the database settings above.
# MAX_HISTORY_TURNS and SUMMARY_KEEP_RECENT (LLM section)
# control compression behaviour.


# ────────────────────────────────────────────────────────────
# Web Research
# ────────────────────────────────────────────────────────────

# Your Tavily API key. Obtain from app.tavily.com.
# If empty, the web_research and hybrid routes will fail
# with an authentication error. The internal_rag, memory,
# and direct_generation routes are unaffected.
TAVILY_API_KEY=tvly-YOUR_KEY_HERE

# Number of web search results to retrieve per Tavily query.
# Higher values add latency. 5 is the recommended default.
# Missing: defaults to 5.
WEB_TOP_K=5
```

After editing, verify the file is saved and the `.env` is in the project
root (the same directory as `requirements.txt`).

---

## 6. Getting API Keys

### Groq

Groq provides a free tier with no credit card required. The free tier allows
approximately 14,400 requests per day split across models.

1. Go to https://console.groq.com
2. Sign up or log in.
3. In the left sidebar, click **API Keys**.
4. Click **Create API Key**.
5. Give it a name (e.g. `dynamic-rag-local`).
6. Copy the key immediately — it is only shown once.
7. Paste it into `.env` as the value of `GROQ_API_KEY`.

Dynamic-RAG uses three distinct Groq calls per query:

- **Planner** (`FAST_MODEL`): classifies intent and selects the retrieval
  route. Uses a small, fast model because latency matters.
- **Generator** (`DEFAULT_LLM`): produces the grounded answer from evidence.
  Uses a large model for quality.
- **Critic** (`CRITIC_MODEL`): verifies the answer against the evidence and
  scores faithfulness. Uses a reasoning model.

Each of these calls counts separately against your rate limit. Queries that
trigger the retry loop (verify -> retry -> generate) make one additional
generator call.

### Tavily

Tavily is a search API designed for RAG and AI agents. The free tier provides
1,000 searches per month.

1. Go to https://app.tavily.com
2. Click **Sign Up** (or **Sign In** if you have an account).
3. After signing in, go to **API Keys** in the dashboard.
4. Copy the key shown (format: `tvly-...`).
5. Paste it into `.env` as the value of `TAVILY_API_KEY`.

Tavily is only called when the planner routes to `web_research` or `hybrid`.
If your queries are exclusively internal (answered from indexed documents),
you can leave `TAVILY_API_KEY` empty and the web routes will simply fail
gracefully with an abstention.

---

## 7. Groq Model Reference

The table below lists all models available on Groq as of mid-2026, their
context windows, and their recommended role in Dynamic-RAG.

| Model ID | Context | Speed | Recommended Role |
|---|---|---|---|
| `llama-3.3-70b-versatile` | 128k | Medium | Generator (`DEFAULT_LLM`) — best quality |
| `llama-3.1-8b-instant` | 128k | Very fast | Planner (`FAST_MODEL`) and fallback |
| `llama-3.1-70b-versatile` | 128k | Medium | Generator alternative |
| `qwen/qwen3-32b` | 32k | Medium | Critic (`CRITIC_MODEL`) — reasoning model |
| `qwen/qwen3-14b` | 32k | Fast | Lighter critic alternative |
| `qwen/qwen3-8b` | 32k | Very fast | Critic for low-latency setups |
| `mixtral-8x7b-32768` | 32k | Fast | Generator fallback |
| `gemma2-9b-it` | 8k | Fast | Lightweight generator |
| `llama-guard-3-8b` | 8k | Fast | Content moderation (not used by default) |

**Important note on `CRITIC_MODEL=qwen/qwen3-32b`:** This is a reasoning
model. It produces a chain-of-thought trace inside `<think>...</think>` tags
before emitting the JSON verdict. This trace can be several hundred tokens
long. Set `MAX_TOKENS` to at least `2048` (preferably `4096`) to ensure the
full output is generated. If the model is cut off mid-output, the verifier's
JSON parser will fall back to a default non-faithful verdict, which will
trigger unnecessary retries.

**Model deprecations:** Groq rotates its model catalogue. If you see
`model not found` errors, check https://console.groq.com/docs/models for the
current list and update the three `*_MODEL` variables in `.env` accordingly.
The planner and generator roles are interchangeable across the Llama 3 family.

---

## 8. First Run Verification

Start all four services in separate terminals. The order matters: databases
first, API server second, UI last.

### Terminal 1 — MongoDB

MongoDB should already be running as a service (see Section 2). If not:

**Windows:**
```powershell
net start MongoDB
```

**macOS:**
```bash
brew services start mongodb-community@6.0
```

**Linux:**
```bash
sudo systemctl start mongod
```

### Terminal 2 — Qdrant

**Docker:**
```bash
docker start qdrant
```

**Binary:**
```bash
./qdrant
```

Wait until you see `Qdrant HTTP listening on 0.0.0.0:6333`.

### Terminal 3 — FastAPI server

From the project root with your virtual environment activated:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Wait until you see `Application startup complete.`

The `--reload` flag is for development only. Remove it in production.

### Terminal 4 — Streamlit UI

```bash
streamlit run app.py
```

Streamlit will open the UI automatically at http://localhost:8501

### Health check

Run this Python one-liner from the project root to verify all four services
respond correctly:

```bash
python -c "
import requests, pymongo, sys
from qdrant_client import QdrantClient

results = {}

# FastAPI
try:
    r = requests.get('http://localhost:8000/health', timeout=5)
    results['FastAPI'] = r.json().get('status', 'unknown')
except Exception as e:
    results['FastAPI'] = f'FAILED: {e}'

# MongoDB
try:
    c = pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=3000)
    c.admin.command('ping')
    results['MongoDB'] = 'connected'
except Exception as e:
    results['MongoDB'] = f'FAILED: {e}'

# Qdrant
try:
    q = QdrantClient(url='http://localhost:6333', timeout=5)
    q.get_collections()
    results['Qdrant'] = 'connected'
except Exception as e:
    results['Qdrant'] = f'FAILED: {e}'

# Streamlit (basic HTTP check)
try:
    r = requests.get('http://localhost:8501', timeout=5)
    results['Streamlit'] = 'reachable' if r.ok else f'HTTP {r.status_code}'
except Exception as e:
    results['Streamlit'] = f'FAILED: {e}'

print()
all_ok = True
for svc, status in results.items():
    icon = 'OK' if 'FAILED' not in status else 'FAIL'
    print(f'  [{icon}] {svc}: {status}')
    if 'FAILED' in status:
        all_ok = False
print()
sys.exit(0 if all_ok else 1)
"
```

Expected output when everything is healthy:

```
  [OK] FastAPI: healthy
  [OK] MongoDB: connected
  [OK] Qdrant: connected
  [OK] Streamlit: reachable
```

If FastAPI reports `degraded` instead of `healthy`, one of the database
connections is failing. Check the relevant service terminal for error messages.

The FastAPI interactive docs (Swagger UI) are available at
http://localhost:8000/docs — you can test all endpoints directly from the
browser.

---

## 9. Ingesting Your First Document

The repository includes a sample corpus under `data/raw/primary` (geopolitics,
Indian history, and world affairs documents — 30 files, ~4,888 chunks when
fully ingested).

### Ingest the full sample corpus

With the FastAPI server running and the virtual environment activated, run:

```bash
python -m src.ingestion.pipeline data/raw/primary
```

This recursively finds all `.pdf` and `.txt` files, parses them, chunks them,
embeds them, and indexes them into Qdrant. First-run model downloads happen
here if you have not run anything yet. Expect 5 to 15 minutes depending on
corpus size and hardware.

Progress is logged to the console and to `logs/dynamic_rag.log`.

### Verify indexing

```bash
curl http://localhost:6333/collections/dynamic_rag_documents
```

The response includes a `vectors_count` field showing how many chunks were
indexed.

### Test a query via the API

```bash
curl -s -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What caused the Cuban Missile Crisis?", "session_id": "test-001"}' \
  | python -m json.tool
```

A successful response has this structure:

```json
{
    "answer": "The Cuban Missile Crisis was caused by ...",
    "query_id": "req_...",
    "session_id": "test-001",
    "route": "internal_rag",
    "confidence": 0.91,
    "sources": [...],
    "faithfulness_score": 0.98,
    "latency_ms": 4200.0,
    "status": "success"
}
```

### Ingest a custom document

Upload any PDF or TXT through the API:

```bash
curl -s -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/your/document.pdf" \
  | python -m json.tool
```

Or use the document uploader in the Streamlit sidebar.

---

## 10. Troubleshooting

| Error Message | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Python cannot find the package because the script is not run from the project root. | Always run commands from the `Dynamic-RAG/` directory. Do not `cd` into subdirectories. |
| `ModuleNotFoundError: No module named 'groq'` | Virtual environment is not activated, or `pip install -r requirements.txt` was not run. | Activate the venv (`.\venv\Scripts\Activate.ps1` on Windows, `source venv/bin/activate` on Mac/Linux) then re-run `pip install -r requirements.txt`. |
| `ServerSelectionTimeoutError` / `MongoDB connection failed` | MongoDB is not running, or `MONGO_URI` in `.env` is wrong. | Start MongoDB (see Section 2). Confirm URI with `mongosh --eval "db.adminCommand('ping')"`. |
| `Connection refused` on `http://localhost:6333` | Qdrant is not running. | Start Qdrant container (`docker start qdrant`) or relaunch the binary. |
| `httpx.ConnectError` calling Groq API | No internet connection, or firewall is blocking outbound HTTPS to `api.groq.com`. | Check network connectivity. Try `curl https://api.groq.com` to confirm. |
| `AuthenticationError` / `401` from Groq | `GROQ_API_KEY` is missing, empty, or invalid. | Verify the key at console.groq.com. Copy it again and update `.env`. |
| `RateLimitError` / `429` from Groq | Free-tier rate limit exceeded. | Wait 60 seconds and retry. Reduce query frequency. The `LLM_MAX_RETRIES` and `LLM_BACKOFF_BASE` settings manage automatic retries for transient 429s. |
| `model not found` / `404` from Groq | A model ID in `.env` has been decommissioned by Groq. | Check https://console.groq.com/docs/models for the current model list. Update `DEFAULT_LLM`, `FAST_MODEL`, or `CRITIC_MODEL` in `.env`. |
| `ValidationError` at startup: `DEFAULT_LLM` / `FAST_MODEL` / `CRITIC_MODEL` field required | One of the required model env vars is missing from `.env`. | Add the missing variable. All three are required; there are no hardcoded defaults for these three. |
| `ValueError: Vector dimension mismatch` from Qdrant | You changed `EMBEDDING_MODEL` or `VECTOR_DIMENSION` after the collection was already created with different dimensions. | Delete the existing Qdrant collection (`curl -X DELETE http://localhost:6333/collections/dynamic_rag_documents`), update the env vars, and re-ingest all documents. |
| EasyOCR first run takes 5+ minutes | EasyOCR downloads ~200 MB of detection and recognition models on first use. This is a one-time cost. | Wait for the download to complete. Progress is shown in the terminal. Subsequent runs use the cached models. |
| EasyOCR returns empty text on a PDF | The PDF is image-only (scanned). EasyOCR should handle it, but extremely low DPI scans produce empty OCR output. | Check the PDF quality. If it is below 150 DPI, re-scan at 300 DPI. The parser logs a warning when OCR yields no text. |
| `tavily` / `401` on web research route | `TAVILY_API_KEY` is missing or invalid. | Obtain a key from app.tavily.com and add it to `.env`. If you do not need web research, leave the key empty — only the `web_research` and `hybrid` routes will be affected. |
| Streamlit shows `API unreachable` in the sidebar | The FastAPI server is not running, or it is on a different port. | Start the API server with `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`. Check that `PORT=8000` in `.env` matches. |
| `No chunks produced` warning during ingestion | The document is empty, has no extractable text, or all pages were blank after parsing. | Inspect the file manually. For scanned PDFs with no embedded text, EasyOCR should activate — check that `easyocr` is installed correctly. |
| Answers are consistently `abstained` | Retrieval is returning no evidence (empty corpus or collection not found). | Confirm ingestion completed successfully. Check Qdrant `vectors_count` as shown in Section 9. |
