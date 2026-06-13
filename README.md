# Local RAG Pipeline

A fully local, end-to-end Retrieval Augmented Generation (RAG) pipeline built from scratch — no frameworks, no API costs, runs entirely on your machine.

Built as a learning project to deeply understand how RAG systems work under the hood, rather than abstracting everything away behind LangChain or LlamaIndex.

---

## What it does

Takes a mix of PDF and CSV files, chunks and embeds them locally, stores them in a vector database, and lets you ask natural language questions about your data — all powered by a local LLM via Ollama. Includes a custom evaluation framework built from scratch to measure pipeline quality across four metrics.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Ollama (llama3.2) | Free, fully local, no API key needed |
| Embeddings | Ollama (nomic-embed-text) | Local embedding model, no cost per call |
| Vector Store | ChromaDB | Simple, persistent, no infrastructure setup |
| PDF Extraction | pypdf | Lightweight, no external dependencies |
| CSV Handling | pandas | Natural language row conversion |
| Chunking | tiktoken | Token-based chunking for consistency |
| Faithfulness Evaluation | cross-encoder/nli-deberta-v3-small | Purpose-built NLI model for entailment scoring |

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
  Query Embedding + Retrieval
        ↓
   Context Assembly
        ↓
   Local LLM (Ollama)     ← rag.py
        ↓
     Final Answer
        ↓
   Evaluation             ← manual_eval.py
```

---

## Project Structure

```
rag_pipeline/
├── ingest.py         # PDF and CSV text extraction
├── chunker.py        # Token-based chunking with overlap
├── vectorstore.py    # Embedding generation and ChromaDB storage/retrieval
├── rag.py            # Context assembly and LLM response generation
├── main.py           # Entry point with indexing check and query loop
├── manual_eval.py    # Custom evaluation framework with four metrics
└── data/             # Place your PDF and CSV files here
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
pip install chromadb pypdf pandas tiktoken ollama sentence-transformers
```

**Add your data**

Place any `.pdf` or `.csv` files into the `data/` folder.

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

These scores confirmed the pipeline's core weakness — retrieval precision for specific entity lookups — which is the next thing to fix.

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

- **Exact lookup queries** — semantic search struggles with name or ID based lookups. Hybrid search combining semantic and keyword retrieval is the next planned improvement.
- **Aggregation queries** — RAG is not designed for calculations over full datasets. A pandas agent extension is planned to handle this.
- **Small model reasoning** — llama3.2 at 2GB occasionally hallucinates despite prompt constraints. Swapping to mistral or gemma3 improves reasoning quality at the cost of speed.
- **Answer relevancy metric** — embedding-based relevancy rewards non-answers that mirror the question. LLM as judge would be more reliable for this specific metric.

---

## Planned Extensions

- [ ] Hybrid search combining semantic and keyword retrieval
- [ ] Pandas agent for structured data aggregations
- [ ] Query transformation before retrieval
- [ ] LLM as judge for answer relevancy metric
- [ ] Support for additional file types (`.txt`, `.docx`)
- [ ] Swap local LLM for Claude API with minimal code changes

---

## Branch History

Each branch represents a milestone in the learning journey:

| Branch | What it contains |
|---|---|
| `v1-rag-pipeline-core` | Core RAG pipeline — ingestion, chunking, vector store, retrieval, generation |
| `main` | Core pipeline + custom evaluation framework |

---

## Why this project

I watched a team demo a RAG-based agent built on Databricks and realised I didn't fully understand every layer of what they had built. Rather than accepting a surface level understanding, I built the entire pipeline from scratch to understand each component — ingestion, chunking, embedding, retrieval, generation, and now evaluation — independently before using any framework to abstract it away.