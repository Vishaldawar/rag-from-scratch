# Local RAG Pipeline

A fully local, end-to-end Retrieval Augmented Generation (RAG) pipeline built from scratch — no frameworks, no API costs, runs entirely on your machine.

Built as a learning project to deeply understand how RAG systems work under the hood, rather than abstracting everything away behind LangChain or LlamaIndex.

---

## What it does

Takes a mix of PDF and CSV files, chunks and embeds them locally, stores them in a vector database, and lets you ask natural language questions about your data — all powered by a local LLM via Ollama. Includes a custom evaluation framework, hybrid search with cross-encoder reranking, query caching, a pandas agent for aggregation queries, and an LLM-based router that picks the right path automatically.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Ollama (llama3.2) | Free, fully local, no API key needed |
| Embeddings | Ollama (nomic-embed-text) | Local embedding model, no cost per call |
| Vector Store | ChromaDB | Simple, persistent, no infrastructure setup |
| Keyword Search | rank_bm25 (BM25Okapi) | Exact term matching, complements semantic search |
| PDF Extraction | pypdf | Lightweight, no external dependencies |
| CSV Handling | pandas | Natural language row conversion |
| Chunking | tiktoken | Token-based chunking for consistency |
| Faithfulness Evaluation | cross-encoder/nli-deberta-v3-small | Purpose-built NLI model for entailment scoring |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Query-aware relevance scoring on top of hybrid retrieval |
| Caching | hashlib + JSON | Deterministic answers for repeated queries |
| Aggregation | pandas + LLM-generated code via `exec()` | True full-dataset calculations RAG can't do |
| Routing | Ollama (llama3.2) as classifier | Decides lookup (RAG) vs aggregation (pandas agent) per query |

---

## Pipeline Architecture

```
Documents (PDF + CSV)
        ↓
   Text Extraction        ← ingest.py
        ↓
  Token-based Chunking    ← chunker.py
        ↓
 Embedding Generation
        ↓
  Vector Store (ChromaDB) ← vectorstore.py
        ↓
     User Query
        ↓
  Query Classification    ← router.py  ("lookup" or "aggregation")
        ↓
        ├── lookup ──────────────────────────────┐
        │                                         ↓
        │                          Semantic Search + BM25 Keyword Search
        │                                         ↓
        │                          Reciprocal Rank Fusion (RRF) ← hybrid_retrieve() → top 20
        │                                         ↓
        │                          Cross-Encoder Reranking ← rerank() → top 5
        │                                         ↓
        │                                  Context Assembly
        │                                         ↓
        │                                  Cache Lookup    ← cache.py
        │                                         ↓ (miss)
        │                                  Local LLM (Ollama) ← rag.py
        │                                         ↓
        │                                  Cache Write
        │                                         ↓
        │                          Answer + Source (retrieved chunks)
        │
        └── aggregation ─────────────────────────┐
                                                   ↓
                                    Schema + Sample Rows → LLM ← pandas_agent.py
                                                   ↓
                                    Generated pandas code
                                                   ↓
                                    exec() against full dataframe
                                                   ↓
                                    Answer + Source (generated code)
```

---

## Project Structure

```
rag_pipeline/
├── ingest.py          # PDF and CSV text extraction
├── chunker.py         # Token-based chunking with overlap
├── vectorstore.py     # Embeddings, ChromaDB, BM25 keyword search, hybrid RRF retrieval, cross-encoder reranking
├── rag.py             # Context assembly, cache lookup, and LLM response generation
├── cache.py           # Query + context hashing and JSON-based response caching
├── pandas_agent.py    # LLM-generated pandas code for aggregation queries, executed via exec()
├── router.py          # LLM-based query classifier — routes to RAG or pandas agent
├── main.py            # Entry point with indexing check, routing, and query loop
├── manual_eval.py     # Custom evaluation framework with four metrics
└── data/              # Place your PDF and CSV files here
```

---

## Setup

**Prerequisites**
- Python 3.11
- [Ollama](https://ollama.com) installed and running

**Pull required models**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Clone and install dependencies**
```bash
git clone <your-repo-url>
cd rag_pipeline
python3.11 -m venv venv
source venv/bin/activate
pip install chromadb pypdf pandas tiktoken ollama sentence-transformers rank_bm25
```

**Add your data**

Place any `.pdf` or `.csv` files into the `data/` folder.

> **Note:** `query_cache.json` is generated automatically at runtime and is excluded via `.gitignore`, along with `chroma_db/`, `venv/`, and `__pycache__/`.

**Run the pipeline**
```bash
python main.py
```

On first run, the pipeline will automatically build the vector index. Subsequent runs skip indexing and go straight to the query loop.

**Run the evaluator**
```bash
python manual_eval.py
```

The NLI model downloads automatically on first run (~400MB). Subsequent runs use the cached version.

---

## Evaluation Framework

Rather than using RAGAs out of the box, I built a custom evaluation framework from scratch. This was partly forced by dependency incompatibilities between RAGAs, LangChain, and Python 3.11 — but turned out to be a significantly better learning experience.

The evaluator measures four metrics:

**Answer Relevancy** — cosine similarity between the question and answer embeddings. Measures whether the answer is topically aligned with the question. Note: this metric has a known limitation — responses that mirror the question vocabulary score deceptively high even when they don't actually answer it.

**Context Precision** — average cosine similarity between the question and each retrieved chunk. Measures whether retrieval is fetching relevant or noisy context.

**Context Recall** — maximum cosine similarity between the ground truth answer and any single retrieved chunk. Measures whether the correct information exists anywhere in what was retrieved.

**Faithfulness** — uses a dedicated NLI (Natural Language Inference) model rather than embeddings. For each chunk, the model predicts whether the answer is entailed by, neutral to, or contradicted by the context. Raw logits are converted to probabilities via softmax, and the maximum entailment score across all chunks is returned.

### Why NLI for faithfulness and not embeddings?

Embeddings capture semantic similarity but cannot distinguish entailment from contradiction. Consider:

```
Context:  "Lucretia paid $48.27"
Answer 1: "Lucretia paid $48.27"     ← high similarity, entailment
Answer 2: "Lucretia did not pay $48.27" ← high similarity, contradiction
```

Both answers would score similarly with embeddings. The NLI model correctly identifies one as entailment and the other as contradiction.

### Sample evaluation output

```
Answer Relevancy:  0.9149  ← misleadingly high (non-answer mirrors question)
Context Precision: 0.6486  ← moderate (topically related but not entity-specific)
Context Recall:    0.5672  ← low (entity's row not in top-k chunks)
Faithfulness:      0.8196  ← high ("I don't know" is faithful to context)
```

These scores confirmed the pipeline's core weakness — retrieval precision for specific entity lookups — which led directly into the next piece of work.

---

## Hybrid Search (Semantic + Keyword)

Pure semantic search struggles with exact entity lookups. Asking "What did Appolonia Blewitt purchase?" reliably failed to retrieve her data row, because a proper noun like "Appolonia" carries little semantic weight — the embedding captures the general concept of "a purchase question" rather than the specific name.

**The fix — combine two retrieval methods**

- **Semantic search** (existing) — embeddings + cosine similarity via ChromaDB, good at conceptual matching
- **Keyword search** (new) — BM25 via `rank_bm25`, good at exact term matching

Each method returns its own ranked list of chunks. The two lists are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(chunk) = 1 / (semantic_rank + k) + 1 / (keyword_rank + k)
```

Where `k = 60` is a standard constant that dampens the impact of very low ranks. A chunk missing from one list entirely is assigned a heavy penalty rank rather than being excluded — this preserves the complementary benefit of combining both methods instead of only keeping chunks both methods agree on.

**Why BM25 needed preprocessing to work**

The first implementation of keyword search scored the target chunk **0.0** — completely failing to surface it. The cause: the BM25 corpus was left untokenized and unfiltered (raw `.split(" ")`) while the query was preprocessed, so query and corpus tokens never matched on format. Lowercasing, stripping punctuation, and removing stop words from both the query *and* the corpus fixed this — the target chunk's BM25 score went from 0.0 to the highest score in the entire corpus.

**Result**

Before hybrid search, "What did Appolonia Blewitt purchase?" returned a hallucinated, incorrect answer regardless of retrieval attempts. After hybrid search, the correct chunk is retrieved consistently and the LLM, given the right context, produces the correct answer: *"Appolonia Blewitt purchased a product in the category of Clothing - Outerwear and paid $215.38."*

---

## Cross-Encoder Reranking

Hybrid search fixed *recall* — the right chunk now reliably appears somewhere in the retrieved set. But it didn't always rank at the top. RRF combines two rank-based signals, and the correct chunk for Appolonia, for example, landed 2nd rather than 1st — good enough most of the time, but not a guarantee, especially as the candidate pool grows.

**Why embeddings and BM25 alone aren't enough for precise ranking**

Both semantic search and BM25 score a query against each chunk *independently* — the chunk's representation (its embedding, or its token statistics) is computed once, without ever being processed alongside the specific query. This means two superficially similar chunks (e.g. two different customers who both bought "Clothing - Outerwear") can score nearly identically, even though only one is the actual answer.

A cross-encoder processes the query and a candidate chunk *together*, in a single forward pass, letting the model directly attend to how specific words in the query relate to specific words in the chunk. This is far more precise — but also far more expensive, since there's no precomputation possible; every query-chunk pair requires a fresh model call.

**The two-stage retrieval pattern this implies**

1. **Stage 1 — Hybrid retrieval (cheap, approximate)** — widened from top 5 to top 20 candidates, using existing semantic + BM25 + RRF
2. **Stage 2 — Reranking (expensive, precise)** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 20 query-chunk pairs in a single batched call, chunks are sorted by this score, and only the top 5 are kept for the LLM

This mirrors the same `CrossEncoder` pattern already used for faithfulness scoring in the evaluation framework — same library, same `.predict()` call shape — just a different pretrained checkpoint (trained for query-passage relevance rather than NLI) and a direct sort instead of a softmax over three classes.

**Result**

With reranking added, Appolonia's chunk moved from rank 2 (post-RRF) to rank 1 (post-rerank) — and across testing, unique entity-lookup queries are now answered correctly far more consistently than with hybrid search alone.

**A bug worth noting**

The first implementation attempted to use `[query, chunk]` pairs as Python dictionary keys to track scores. Lists aren't hashable in Python, so this fails — the fix was to key the score dictionary by the chunk text itself (a string, which is hashable) rather than the pair.

---

## Pandas Agent (Aggregation Queries)

RAG fundamentally cannot answer aggregation questions correctly — "what is the average transaction amount for Food" requires seeing every matching row, but RAG only ever retrieves a sample of chunks. Increasing `top_k` doesn't fix this; it was tried early on (`top_k=400`) and instead caused the LLM to lose focus entirely and produce a generic, ungrounded summary instead of an answer.

**The fix — let the LLM write code instead of reason over text**

Rather than retrieving chunks, the pandas agent gives the LLM:
- The dataframe's schema (column names and types)
- A handful of sample rows for context
- The question

…and asks it to generate **pandas code** that computes the answer, which is then executed against the **full** dataframe with `exec()`. The LLM never sees the whole dataset — it only ever needs to understand the shape of the data well enough to write correct code. Pandas does the actual computation, not the LLM, so the result is mathematically exact rather than reasoned-and-possibly-wrong.

```
Question + schema + sample rows → LLM
        ↓
Generated pandas code (stored in a 'result' variable)
        ↓
exec(code, {}, {"df": full_dataframe, "pd": pd})
        ↓
result extracted from the exec() namespace
```

**Design choice — a class, not a function**

The agent is implemented as a class (`pandas_agent_class`) rather than a plain function specifically so that intermediate state — the generated code, the full dataframe, the final result — stays accessible as object attributes after the call (`self.code`, `self.result`). This made debugging significantly easier than earlier function-based attempts, where returning the code *or* the result meant choosing one and losing visibility into the other.

**A bug worth noting**

An early version computed the schema and sample rows but never actually inserted them into the prompt sent to the LLM — the variables existed but weren't referenced in the f-string. The code ran without error, but the LLM was effectively working blind, with no idea what columns existed. A reminder that "the code runs" and "the code does what you intended" are different things worth checking separately.

**Safety note**

This executes LLM-generated code directly via `exec()` with access to the real dataframe. This is acceptable for local, single-user experimentation but would need sandboxing (e.g. a restricted execution environment, an allowlist of permitted operations) before being exposed to untrusted input in any shared or production setting.

---

## Query Router

With both a RAG path and a pandas agent path now available, the pipeline needs to decide which one to use for a given question. A keyword-based approach (checking for words like "average" or "total") was considered but rejected as too brittle — many aggregation questions don't contain obvious trigger words, and many lookup questions do.

**The fix — classify with the LLM itself**

A lightweight classification call asks the LLM to label the query as either `"lookup"` or `"aggregation"` before any retrieval happens, based on intent rather than keyword matching. The query is then routed to the RAG pipeline or the pandas agent accordingly. An unrecognised classification result defaults to the RAG path, since it's the safer fallback for ambiguous queries.

**Source attribution**

Rather than relying on the LLM to introspect on its own process when asked something like "what is your source" — which it's unreliable at and could hallucinate about — the pipeline deterministically prints the actual source alongside every answer:

- **RAG path** — the retrieved (and reranked) chunks used as context
- **Pandas agent path** — the generated pandas code that computed the answer

This sidesteps needing the LLM to explain itself after the fact; the source is shown automatically because the pipeline already has it as state from generating the answer.

---

## LLM Non-Determinism

Even with the correct context retrieved consistently (verified by hashing the assembled context string across repeated calls), the LLM's answer to the identical question varied from call to call — sometimes correct, sometimes a hallucinated wrong answer, sometimes a false "no data available."

**Diagnosis process**

1. Confirmed retrieval was fully deterministic — identical context hash across repeated calls
2. Set `temperature: 0` in the Ollama call — reduced but did not eliminate variation
3. Added a fixed `seed` alongside temperature — still did not fully eliminate variation

**Root cause**

With identical input, identical temperature, and identical seed, output still varied. This points to floating point non-determinism in GPU-accelerated inference (Apple Metal backend on macOS) — parallel computation can execute operations in different orders across runs, and when two candidate tokens have very close probabilities, tiny numerical differences can flip which one is selected. Once one token differs early in generation, the rest of the response can diverge completely.

This is a known limitation of local LLM inference on consumer GPU hardware, not a configuration error. CPU-only inference is more reliably deterministic but significantly slower.

---

## Query Caching

Rather than fight GPU non-determinism directly, a query + context cache was added to guarantee consistent answers for repeated questions and avoid redundant LLM calls.

**Design**

- **Cache key** — SHA-256 hash of `query + context` combined, not query alone. This means the cache naturally invalidates itself whenever retrieval returns different context (e.g. after re-indexing new data), without needing explicit invalidation logic.
- **Cache value** — the original query, the answer, and a timestamp, stored as JSON for human-readable inspection.
- **Storage** — a flat `query_cache.json` file, checked before every LLM call and written to after every cache miss.

**A subtle bug along the way**

The cache initially appeared not to work — every call printed "No cache exists!" even on the second identical question. The cause was using `os.listdir()` to check file existence relative to whatever directory the script happened to be run from, which is unreliable. Switching to `os.path.exists(file_name)` — which checks the exact given path directly — fixed it immediately.

**What caching does and doesn't solve**

Caching guarantees the *same* answer for a *repeated* identical query + context pair. It does not fix the underlying non-determinism — a genuinely new question will still be subject to the same GPU floating point variability described above. This is an explicit tradeoff: consistency for repeated queries, not a fix for the root cause.

---

## Design Decisions and Learnings

**Why build without frameworks first**

The first instinct when building RAG is to reach for LangChain or LlamaIndex. I deliberately avoided this to understand every component independently. When something breaks in a framework, you need to know what's happening underneath to debug it effectively.

**Token-based chunking over character or line splitting**

Early iterations split text by lines or characters. This produced inconsistent chunk sizes since a line could be 3 tokens or 30 tokens. Switching to tiktoken-based chunking ensured consistent 300-token chunks with 50-token overlap, giving the embedding model uniform input sizes.

**CSV to natural language conversion**

Instead of passing raw CSV rows or markdown tables to the embedding model, each row is converted to natural language format — `name: John, age: 30, salary: 50000`. This reduces noise characters and improves semantic similarity matching during retrieval.

**Why nomic-embed-text over sentence-transformers**

Since Ollama was already part of the stack for the LLM, using it for embeddings too kept the architecture consistent and eliminated an additional Python dependency. nomic-embed-text is purpose-built for retrieval tasks and performs well for this use case.

**The retrieval quality problem**

One key learning was that RAG retrieval quality is heavily dependent on query language matching data language. A query like "what is the average salary?" performs poorly against data containing `txn_amt` and `product_category` fields. This highlighted the importance of query transformation in production systems.

**RAG is not suitable for aggregations**

Attempting to ask aggregate questions like "what is the average transaction amount" revealed a fundamental limitation — RAG retrieves a sample of chunks, not the full dataset. Any aggregation over retrieved chunks is mathematically incomplete. The right solution for structured data aggregations is a SQL or pandas agent layer, which is a planned extension of this project.

**Hallucination through system prompt tightening**

An important discovery was that small models like llama3.2 will hallucinate confidently when given irrelevant context. Tightening the system prompt to explicitly instruct the model to say "I don't have enough information" when the answer isn't in the context dramatically improved reliability over trying to increase top-k retrieval.

**Why RAGAs was replaced with a custom evaluator**

RAGAs was the first choice for evaluation but introduced unresolvable dependency conflicts between its LangChain dependencies, langchain-community, and Python 3.11. Rather than downgrading the entire environment, I built equivalent metrics from scratch — which revealed exactly how RAGAs works under the hood and exposed limitations that using it as a black box would have hidden.

**The answer relevancy limitation discovery**

Building answer relevancy with embeddings revealed a critical flaw — a response that says "I don't have enough data to answer the question about X" scores higher than a correct answer phrased differently, because it mirrors the question vocabulary. This is why RAGAs uses LLM as judge for this specific metric rather than embeddings.

---

## Known Limitations

- **Small model reasoning** — llama3.2 at 2GB occasionally hallucinates despite prompt constraints and a tightened system prompt. Swapping to mistral or gemma3 improves reasoning quality at the cost of speed.
- **Answer relevancy metric** — embedding-based relevancy rewards non-answers that mirror the question. LLM as judge would be more reliable for this specific metric.
- **LLM non-determinism on GPU** — even with temperature 0 and a fixed seed, identical query + context can occasionally produce different answers due to floating point non-determinism in GPU-accelerated inference. Mitigated for repeated queries via caching, not fully solved for novel queries.
- **BM25 index rebuilt per call** — `keyword_retrieve` currently rebuilds the BM25 index from the full corpus on every call rather than building it once at indexing time. Fine at this dataset's scale (370 chunks), would not scale to large corpora.
- **Reranking adds latency** — every RAG-path query pays the cost of a cross-encoder forward pass over ~20 candidates, on top of embedding + BM25 retrieval. Not yet measured precisely, but noticeably slower than hybrid search alone.
- **Reranker model is general-purpose** — `ms-marco-MiniLM-L-6-v2` was trained on web search query-passage pairs, not transaction-style tabular text. It works well here but hasn't been validated against alternatives for this specific data shape.
- **`exec()` on LLM-generated code is unsandboxed** — the pandas agent executes generated code with full access to the dataframe and pandas itself. Fine for local single-user use, not safe for untrusted or multi-user input without a restricted execution environment.
- **Router has no evaluation of its own** — query classification accuracy (lookup vs aggregation) hasn't been measured against a labelled test set; correctness is currently judged by spot-checking.
- **Pandas agent has no protection against malformed code** — if the LLM generates code that doesn't define a `result` variable, or that errors out, there's currently no graceful fallback or retry.

---

## Planned Extensions

- [ ] Query transformation before retrieval (HyDE)
- [ ] LLM as judge for answer relevancy metric
- [ ] Build BM25 index once at indexing time instead of per-query
- [ ] Re-run evaluation metrics with hybrid retrieval + reranking for a quantified before/after comparison
- [ ] Benchmark reranker latency and evaluate alternative cross-encoder checkpoints
- [ ] Labelled test set to measure router classification accuracy
- [ ] Graceful fallback/retry when pandas agent generates malformed code
- [ ] Sandboxed execution environment for the pandas agent
- [ ] Support for additional file types (`.txt`, `.docx`)
- [ ] Swap local LLM for Claude API with minimal code changes

---

## Branch History

Each branch represents a milestone in the learning journey:

| Branch | What it contains |
|---|---|
| `v1-rag-pipeline-core` | Core RAG pipeline — ingestion, chunking, vector store, retrieval, generation |
| `v2-rag-pipeline-eval` | Core pipeline + custom evaluation framework (4 metrics, NLI faithfulness) |
| `v3-hybrid-search-caching` | Hybrid search (BM25 + RRF), LLM non-determinism diagnosis, query caching |
| `v4-reranking` | Cross-encoder reranking on top of hybrid search (widened candidate pool, two-stage retrieval) |
| `v5-pandas-agent-router` | Pandas agent for aggregation queries, LLM-based router, source attribution |
| `main` | Latest — all of the above |

---

## Why this project

I watched a team demo a RAG-based agent built on Databricks and realised I didn't fully understand every layer of what they had built. Rather than accepting a surface level understanding, I built the entire pipeline from scratch to understand each component — ingestion, chunking, embedding, retrieval, generation, evaluation, hybrid search, reranking, caching, aggregation via a pandas agent, and query routing — independently before using any framework to abstract it away. Several of the most useful lessons came from debugging — a BM25 corpus that wasn't tokenized the same way as the query, a cache that silently failed because of `os.listdir()` vs `os.path.exists()`, a deep dive into why a local LLM doesn't always return the same output even at temperature 0, a reminder that Python lists aren't hashable when reranking scores needed a dictionary key, and a prompt that computed the right variables but forgot to actually use them.

This project also marks closing the original gap that started it — the team's RAG agent now has a fully understood, working, locally-built counterpart, with a documented account of where it's stronger, where it's weaker, and why.