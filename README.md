# Local RAG Pipeline

A fully local, end-to-end Retrieval Augmented Generation (RAG) pipeline built from scratch — no frameworks, no API costs, runs entirely on your machine.

Built as a learning project to deeply understand how RAG systems work under the hood, rather than abstracting everything away behind LangChain or LlamaIndex.

---

## What it does

Takes a mix of PDF and CSV files, chunks and embeds them locally, stores them in a vector database, and lets you ask natural language questions about your data — all powered by a local LLM via Ollama.

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
└── data/             # Place your PDF and CSV files here
```

---

## Setup

**Prerequisites**
- Python 3.9+
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
python3 -m venv venv
source venv/bin/activate
pip install chromadb pypdf pandas tiktoken ollama
```

**Add your data**

Place any `.pdf` or `.csv` files into the `data/` folder.

**Run**
```bash
python main.py
```

On first run, the pipeline will automatically build the vector index. Subsequent runs skip indexing and go straight to the query loop.

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

---

## Known Limitations

- **Exact lookup queries** — semantic search struggles with name or ID based lookups. Hybrid search combining semantic and keyword retrieval would improve this significantly.
- **Aggregation queries** — RAG is not designed for calculations over full datasets. A pandas agent extension is planned to handle this.
- **Small model reasoning** — llama3.2 at 2GB occasionally hallucinates despite prompt constraints. Swapping to mistral or gemma3 improves reasoning quality at the cost of speed.

---

## Planned Extensions

- [ ] Pandas agent for structured data aggregations
- [ ] Hybrid search combining semantic and keyword retrieval
- [ ] Query transformation before retrieval
- [ ] RAGAs evaluation framework integration
- [ ] Support for additional file types (`.txt`, `.docx`)
- [ ] Swap local LLM for Claude API with minimal code changes

---

## Why this project

I watched a team demo a RAG-based agent built on Databricks and realised I didn't fully understand every layer of what they had built. Rather than accepting a surface level understanding, I built the entire pipeline from scratch to understand each component — ingestion, chunking, embedding, retrieval, and generation — independently before using any framework to abstract it away.
