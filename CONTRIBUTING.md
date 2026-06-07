# Contributing to Dynamic-RAG

Thank you for considering a contribution to Dynamic-RAG. This project is being built to a serious engineering standard, and contributions that uphold that standard are genuinely valuable. The goal is a measurable, reproducible, evaluation-first RAG system — not a demo, not a prototype. Every line of code that lands in this repository is expected to be tested, typed, and accountable to the evaluation planes.

Please read this document in full before opening a pull request.

---

## Table of Contents

1. [Welcome and Project Ethos](#1-welcome-and-project-ethos)
2. [Before You Start](#2-before-you-start)
3. [Development Setup](#3-development-setup)
4. [Code Style](#4-code-style)
5. [Running Tests](#5-running-tests)
6. [Adding a New Route](#6-adding-a-new-route)
7. [Adding a New Evaluation Metric](#7-adding-a-new-evaluation-metric)
8. [Adding a New File Type](#8-adding-a-new-file-type)
9. [Pull Request Process](#9-pull-request-process)
10. [What NOT to Do](#10-what-not-to-do)

---

## 1. Welcome and Project Ethos

Dynamic-RAG is an adaptive retrieval-augmented generation system built around three core values:

**Evaluation first.** No subsystem is considered working until it has been exercised by a test and measured by a metric. The three evaluation planes (retrieval quality, generation faithfulness, system-level behavior) are not an afterthought — they are the acceptance criteria.

**No untested code.** If a module has not been executed and its outputs inspected, it does not count as working. This applies equally to new contributions. You are expected to run the code you write, observe its outputs, and confirm it behaves correctly before submitting.

**Observability over magic.** Every decision the system makes should be traceable. Routing decisions, retrieval scores, faithfulness verdicts, and generation latencies are all logged and measurable. Contributions should preserve or extend this observability, never reduce it.

**Faithful over verbose.** A response that says "I do not have enough information to answer this" is better than a response that makes up a confident-sounding answer. The faithfulness verifier exists for a reason — do not work around it.

If a contribution does not fit these values, it will not be merged regardless of how clever it is.

---

## 2. Before You Start

### Understand the Three Evaluation Planes

All contributions that touch retrieval, generation, or routing are evaluated across three planes:

**Plane 1 — Retrieval Quality**

Measures whether the retrieval layer is returning relevant chunks.

Key metrics: Recall@K, MRR, Hit Rate, Context Precision, Context Recall.

Current baseline: Recall@K = 0.87, MRR = 1.0, Hit Rate = 1.0, Context Precision = 0.85.

A contribution must not regress these numbers.

**Gate C — Routing Accuracy**

Measures whether the planner is selecting the correct route for a given query.

Current baseline: Routing Accuracy = 0.93.

**Plane 2 — Generation Faithfulness**

Measures whether the generator is grounded in the retrieved evidence.

Key metrics: Faithfulness, Groundedness, Citation Accuracy, Completeness.

Current baseline: Faithfulness = 0.98, Groundedness = 1.0, Citation Accuracy = 0.99, Completeness = 0.85.

**Plane 3 — System-Level Behavior**

Measures end-to-end accuracy, rejection rate, and failure count.

Current baseline: E2E Accuracy = 0.81, Rejection Rate = 0.6, Failure Count = 0.

Before submitting a PR that touches any of these layers, run the relevant evaluation script and include the output in your PR description. If a metric regresses, explain why and what the tradeoff is.

---

## 3. Development Setup

### Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/Dynamic-RAG.git
cd Dynamic-RAG
```

Add the upstream remote so you can pull in future changes:

```bash
git remote add upstream https://github.com/Divij2601/Dynamic-RAG.git
```

### Python Version

This project requires Python 3.11.9. Use pyenv, conda, or any version manager to pin this version before creating a virtual environment.

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key package versions this project is tested against:

| Package | Version |
|---|---|
| langgraph | 1.2.4 |
| fastapi | 0.136.3 |
| streamlit | 1.58.0 |
| groq | 1.4.0 |
| qdrant-client | 1.18.0 |
| sentence-transformers | 5.5.1 |
| easyocr | 1.7.2 |
| rank-bm25 | 0.2.2 |

If you add a dependency, pin the exact version in `requirements.txt` and note it in your PR.

### Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values. At minimum you need:

- `GROQ_API_KEY` — your Groq API key
- `TAVILY_API_KEY` — your Tavily web search key
- `QDRANT_URL` — defaults to `http://localhost:6333`
- `MONGO_URI` — defaults to `mongodb://localhost:27017`

Never commit `.env`. It is listed in `.gitignore`. See section 10 for more on this.

### Start Local Infrastructure

Qdrant and MongoDB must be running for integration tests and for the full pipeline to work.

Qdrant (Docker):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

MongoDB (Docker):

```bash
docker run -p 27017:27017 mongo:7
```

Alternatively, use locally installed instances if you have them.

### Verify the Setup

```bash
python -c "from src.config import settings; print(settings.groq_model_generator)"
```

If this prints `llama-3.3-70b-versatile` without errors, the configuration layer is working.

---

## 4. Code Style

### Vertical Argument Formatting

This project uses a consistent vertical formatting style. Each argument or keyword argument goes on its own line. This applies to function definitions, function calls, dataclass constructors, Pydantic models, and any other multi-argument structure.

Do this:

```python
result = retrieve_dense(
    query=query_text,
    collection=settings.qdrant_collection,
    top_k=settings.rerank_top_k,
    score_threshold=settings.dense_score_threshold,
)
```

Not this:

```python
result = retrieve_dense(query=query_text, collection=settings.qdrant_collection, top_k=settings.rerank_top_k)
```

The trailing comma after the last argument is required. This makes diffs cleaner when arguments are added or removed.

### Type Hints Required

Every function must have type annotations on all parameters and on the return type. No exceptions.

```python
def fuse_scores(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float,
    sparse_weight: float,
) -> list[dict]:
```

Do not use `Any` unless you are explicitly wrapping an untyped third-party interface, and even then, leave a comment explaining why.

### Comments: Only When the Why Is Non-Obvious

Do not comment what the code does. Comment why it does something, and only when a reasonable reader would not immediately understand the reason.

Do this:

```python
# BM25 scores are not normalized; dividing by max prevents dominating dense scores
# when sparse results cluster at high token overlap.
bm25_norm = score / max_bm25_score if max_bm25_score > 0 else 0.0
```

Not this:

```python
# Divide score by max score
bm25_norm = score / max_bm25_score if max_bm25_score > 0 else 0.0
```

### No Docstrings on Simple Methods

If a function's name, parameter names, and type hints make its behavior obvious, do not add a docstring. Docstrings add noise when they restate what is already clear.

Use docstrings only for:

- public-facing functions in the API layer,
- functions with non-obvious contracts (pre/postconditions, side effects, ordering requirements),
- evaluation utilities where the metric definition is not universally known.

### Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers within a module: prefix with `_`

### No Magic Numbers

All thresholds, limits, and tunable values belong in `src/config.py` as settings fields. Do not hardcode `0.7` or `20` or `8` inside a function body.

---

## 5. Running Tests

### Fast Test Suite (No Live Infrastructure Required)

```bash
pytest -m "not integration"
```

This runs all unit tests that do not require a live Qdrant instance, a live MongoDB instance, or a live Groq API key. These tests cover:

- configuration loading and validation,
- chunker logic and overlap behavior,
- metadata enrichment,
- score fusion math (weighted and RRF),
- planner heuristics and route classification,
- prompt builder formatting,
- response builder schema construction,
- evaluation metric calculations (Recall@K, MRR, NDCG@K, Faithfulness, etc.),
- state schema serialization.

The fast suite should complete in under 60 seconds on a standard laptop with no external connections.

### Integration Test Suite (Requires Live Infrastructure)

```bash
pytest -m "integration"
```

Before running this, confirm that Qdrant is reachable at `http://localhost:6333` and MongoDB is reachable at `mongodb://localhost:27017`. Your `.env` must have valid API keys for Groq and Tavily.

Integration tests cover:

- end-to-end ingestion (load, parse, chunk, embed, index),
- dense and sparse retrieval against an indexed collection,
- hybrid retrieval fusion and deduplication,
- cross-encoder reranking,
- LangGraph graph execution across all five routes,
- memory store write and recall,
- web search and evidence construction,
- full generation with faithfulness verification,
- API endpoint responses.

Integration tests are slower (several minutes depending on model download and API latency) and require credits. Do not run them in a tight loop.

### Running Evaluation Scripts

Evaluation scripts are not part of the pytest suite. Run them directly:

```bash
python evaluation/retrieval_eval.py
python evaluation/generation_eval.py
python evaluation/system_eval.py
python evaluation/runner.py
```

Reports are written to `evaluation/reports/` as JSON files with timestamps. Include the relevant report file in your PR if your change touches a measured subsystem.

### Test File Locations

```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_chunker.py
│   ├── test_hybrid.py
│   ├── test_planner.py
│   ├── test_metrics.py
│   └── ...
└── integration/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_graph.py
    ├── test_memory.py
    └── ...
```

Place new unit tests under `tests/unit/` and new integration tests under `tests/integration/`. Mark integration tests with `@pytest.mark.integration`.

---

## 6. Adding a New Route

The five current routes are: `internal_rag`, `web_research`, `hybrid`, `memory`, `direct_generation`.

If you are adding a sixth route (for example, a `structured_db` route that queries a SQL database), follow these steps exactly.

### Step 1: Register the Route Key in planner.py

Open `src/planner/planner.py` and locate `VALID_ROUTES`. Add your new key:

```python
VALID_ROUTES = {
    "internal_rag",
    "web_research",
    "hybrid",
    "memory",
    "direct_generation",
    "structured_db",   # add here
}
```

Also update the heuristics in `src/planner/heuristics.py` if the new route can be selected by keyword pattern matching. Add the pattern and the route label to the appropriate classifier block.

### Step 2: Write the Node Function in graph/nodes.py

Each route corresponds to a node function in `src/graph/nodes.py`. A node function takes the full `QueryState` and returns a partial state update.

```python
async def structured_db_node(
    state: QueryState,
) -> dict:
    # retrieve from your structured source
    # build EvidenceItem objects
    # return partial state update
    return {
        "evidence": evidence_items,
        "retrieval_metrics": metrics,
    }
```

The function must:

- accept `QueryState` as its only argument,
- return a `dict` of state keys to update (not the full state),
- never mutate `state` in place,
- populate the `evidence` list with properly constructed `EvidenceItem` objects,
- populate `retrieval_metrics` if retrieval was performed.

### Step 3: Register the Node and Edge in graph_builder.py

Open `src/graph/graph_builder.py`. Add your node to the graph:

```python
graph.add_node("structured_db", structured_db_node)
```

Then add the edge from your node to the `generate` node:

```python
graph.add_edge("structured_db", "generate")
```

### Step 4: Add the Conditional Edge Mapping in graph/router.py

Open `src/graph/router.py` and locate the routing function that maps planner output to node names. Add your new route:

```python
def route_after_planner(
    state: QueryState,
) -> str:
    route = state.planner_output.route
    mapping = {
        "internal_rag": "internal_rag",
        "web_research": "web_research",
        "hybrid": "hybrid",
        "memory": "memory",
        "direct_generation": "direct_generation",
        "structured_db": "structured_db",   # add here
    }
    return mapping.get(route, "direct_generation")
```

Also register the new destination in the `add_conditional_edges` call in `graph_builder.py`:

```python
graph.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "internal_rag": "internal_rag",
        "web_research": "web_research",
        "hybrid": "hybrid",
        "memory": "memory",
        "direct_generation": "direct_generation",
        "structured_db": "structured_db",   # add here
    },
)
```

### Step 5: Update KNOWLEDGE_BASE_DESCRIPTION in config.py if Needed

If the planner LLM uses a description of available knowledge sources to decide routing, update `KNOWLEDGE_BASE_DESCRIPTION` in `src/config.py` so it knows the new source exists and when to prefer it.

### Step 6: Add Test Queries to the Evaluation Dataset

Open `evaluation/data/test_set.json` and add at least three benchmark entries that should route to your new node:

```json
{
    "query": "How many rows are in the transactions table?",
    "ground_truth_answer": "...",
    "relevant_chunk_ids": [],
    "answerable": true,
    "expected_route": "structured_db",
    "category": "structured_db"
}
```

Run `evaluation/retrieval_eval.py` after adding the route and confirm that Gate C (Routing Accuracy) does not regress below 0.93.

---

## 7. Adding a New Evaluation Metric

### Step 1: Implement the Metric in evaluation/utils/metrics.py

All metric computation functions live in `evaluation/utils/metrics.py`. Add your function there:

```python
def compute_context_f1(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> float:
    if not relevant_chunk_ids:
        return 0.0
    retrieved_set = set(retrieved_chunk_ids)
    relevant_set = set(relevant_chunk_ids)
    precision = len(retrieved_set & relevant_set) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(retrieved_set & relevant_set) / len(relevant_set)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

Requirements:

- the function must be pure (no side effects, no I/O),
- it must accept only typed primitives or standard Python collections,
- it must handle the empty-input case gracefully (return 0.0, not raise),
- it must be unit-testable in the fast suite.

### Step 2: Add a Unit Test

Add a test in `tests/unit/test_metrics.py`:

```python
def test_context_f1_perfect_recall():
    score = compute_context_f1(
        retrieved_chunk_ids=["a", "b", "c"],
        relevant_chunk_ids=["a", "b"],
    )
    assert score == pytest.approx(0.8, abs=0.01)

def test_context_f1_empty_relevant():
    score = compute_context_f1(
        retrieved_chunk_ids=["a"],
        relevant_chunk_ids=[],
    )
    assert score == 0.0
```

### Step 3: Wire into the Evaluation Script

For a retrieval metric, add the computation to `evaluation/retrieval_eval.py` inside the per-sample loop and add the aggregated result to the report dict.

For a generation metric, add it to `evaluation/generation_eval.py` in the same way.

For a system-level metric, add it to `evaluation/system_eval.py`.

Follow the existing pattern for each file — compute per-sample, then aggregate (mean or sum), then write to the report.

### Step 4: Update the Schema

If the metric is part of the structured report, add a field to the relevant dataclass or Pydantic model in `evaluation/schemas.py`.

---

## 8. Adding a New File Type

The ingestion parser lives in `src/ingestion/parser.py`. It currently handles PDF files (via pdfplumber with EasyOCR fallback) and plain text files.

To add support for a new file type (for example, `.docx` or `.html`):

### Step 1: Add the Extension to the Supported Types Registry

In `src/ingestion/parser.py`, locate the block that maps file extensions to handler functions and add your extension:

```python
EXTENSION_HANDLERS = {
    ".pdf": _parse_pdf,
    ".txt": _parse_txt,
    ".docx": _parse_docx,   # add here
}
```

### Step 2: Write the Handler Function

Each handler receives the file path and returns a list of page dicts. The structure must match what the rest of the pipeline expects:

```python
def _parse_docx(
    file_path: Path,
) -> list[dict]:
    # extract text page by page or section by section
    # return list of {"page_number": int, "text": str}
    ...
```

Requirements:

- return type is always `list[dict]` with keys `page_number` (int) and `text` (str),
- empty pages (whitespace-only text) must be filtered out,
- the function must not raise on a well-formed file of that type,
- add the required third-party package to `requirements.txt` with a pinned version.

### Step 3: Update the Loader Validation

In `src/ingestion/loader.py`, the loader validates file extensions before passing to the parser. Add your extension to the allowlist there as well.

### Step 4: Write a Unit Test

Add a fixture file (small, synthetic, no real personal data) under `tests/fixtures/` and a test in `tests/unit/test_parser.py`:

```python
def test_parse_docx_returns_pages():
    pages = _parse_docx(Path("tests/fixtures/sample.docx"))
    assert len(pages) > 0
    assert all("text" in p for p in pages)
    assert all("page_number" in p for p in pages)
```

### Step 5: Run an Ingestion Smoke Test

Ingest a real file of the new type through the full pipeline (loader → parser → chunker → embedder → indexer) and confirm that chunks appear in Qdrant and are retrievable.

---

## 9. Pull Request Process

### Branch Naming

Use one of these prefixes:

| Prefix | Use for |
|---|---|
| `feature/` | New functionality (new route, new file type, new API endpoint) |
| `fix/` | Bug fixes, schema corrections, error handling improvements |
| `eval/` | Changes to evaluation scripts, metrics, or benchmark datasets |
| `refactor/` | Internal restructuring with no behavior change |
| `docs/` | Documentation only |

Example branch names:

```
feature/structured-db-route
fix/hybrid-fusion-zero-division
eval/add-context-f1-metric
```

### Before Opening a PR

Work through this checklist yourself before requesting review:

- [ ] All imports resolve cleanly (`python -c "from src.<module> import <thing>"` for each changed module)
- [ ] Fast test suite passes: `pytest -m "not integration"` exits 0
- [ ] If the change touches retrieval, Plane 1 metrics are included in the PR description
- [ ] If the change touches generation or the verifier, Plane 2 metrics are included
- [ ] If the change touches routing, Gate C routing accuracy is included
- [ ] `requirements.txt` is updated if any new package was added, with a pinned version
- [ ] `.env` is not staged: `git status` shows no `.env` file
- [ ] No hardcoded API keys, tokens, passwords, or local file paths in any committed file
- [ ] Type hints are present on all new function definitions
- [ ] New test queries added to `evaluation/data/test_set.json` if a new route was added
- [ ] The PR description explains what changed, why, and what the evaluation impact is

### PR Description Template

```
## What changed
<one paragraph>

## Why
<one paragraph>

## Evaluation impact
Plane 1: Recall@K=X, MRR=X, HitRate=X, CtxPrecision=X
Gate C: RoutingAccuracy=X
Plane 2: Faithfulness=X, Groundedness=X, CitationAccuracy=X
Plane 3: E2EAccuracy=X, RejectionRate=X, FailureCount=X

## Tests added or modified
<list>

## Notes for reviewer
<any tradeoffs, open questions, or follow-up items>
```

### Review Criteria

A PR will be reviewed for:

1. Correctness — does it do what it claims?
2. Evaluation impact — do the numbers hold or improve?
3. Type safety — are all signatures annotated?
4. Test coverage — are new behaviors tested?
5. Style compliance — vertical formatting, no magic numbers, no unnecessary comments?
6. Security — no secrets, no hardcoded paths?

PRs that do not include evaluation output for subsystems they affect will be returned for revision before review begins.

---

## 10. What NOT to Do

These are hard rules. Violations will result in the PR being closed without merge.

### Never skip the evaluation gates

Every change to retrieval, routing, or generation must be measured against the three evaluation planes before submission. "It looks right" is not a measurement. If you do not have access to the full corpus, say so in the PR — do not silently omit evaluation output.

### Never commit .env

The `.env` file contains API keys. It is in `.gitignore` for this reason. Do not add it to a commit, do not rename it and commit it, do not paste its contents into any other committed file. If you accidentally staged it, remove it with:

```bash
git rm --cached .env
```

Then rotate any keys that were exposed.

### Never add an untested module

If you write a new file, it must be imported and executed at least once before the PR is opened. "I'll test it later" is not acceptable. The project's operating rule is: untested code does not exist until it runs. A module that imports cleanly but has never been called is not tested.

### Never bypass the faithfulness verifier

The verifier in `src/generation/verifier.py` exists to catch hallucinations before they reach the user. Do not add logic that skips verification, that returns early before verification runs, or that overrides a low-faithfulness verdict with a higher one without the critic actually re-scoring. The verifier is not optional and is not a performance optimization target.

### Never hardcode thresholds in function bodies

If you introduce a new threshold (score cutoff, top-k limit, temperature, weight), it belongs in `src/config.py` as a field with a documented default. A hardcoded `0.5` buried in a retrieval function will not survive the next configuration change.

### Never use print for observability

Use the logger from `src/observability/logger.py`. Print statements do not appear in log files, are not searchable, and are not traceable to a request ID. There are no exceptions to this rule in production code paths.

### Never merge a PR that increases Failure Count above 0

The system-level Failure Count metric must remain at 0. A failure is an unhandled exception that reaches the API surface. If your change introduces a new code path that can raise an unhandled exception, you must add error handling before the PR is mergeable.

---

If you have questions about any of these guidelines, open a GitHub Discussion before starting work. It is much easier to align on approach before code is written than after.
