import ollama
import chromadb
import os
from ingest import read_data
from chunker import get_chunks
from rank_bm25 import BM25Okapi
import re

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_collection")

stop_words = {"what", "did", "how", "much", "she", "he", "the", "a", "an", 
              "is", "are", "was", "were", "and", "or", "for", "to", "do",
              "does", "her", "his", "pay", "purchase", "buy", "bought"}

def preprocess(text: str) -> list:
    tokens = text.lower().split()
    tokens = [re.sub(r'[^a-z0-9]', '', t) for t in tokens]
    tokens = [t for t in tokens if t and t not in stop_words]
    return tokens

def index_chunks(chunks):
    for chunk in chunks:
        text = chunk['text']
        id = chunk['chunk_index']
        embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
        collection.add(ids=[id],
                        documents=[text],
                        embeddings=[embedding]
                        )
    

def retrieve(query, top_k=5):
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return result


def keyword_retrieve(corpus: list, query: str, k: int) -> list:
    # tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_corpus = [preprocess(doc) for doc in corpus] 
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = preprocess(query)
    # tokenized_query = query.split(" ")

    scores = bm25.get_scores(tokenized_query)
    # Find chunk 0's score
    # print(f"Chunk 0 BM25 score: {scores[0]}")
    # print(f"Max BM25 score: {max(scores)}")
    # print(f"Chunk with max score index: {scores.argmax()}")
    # print(f"Top 5 scoring chunk indices: {scores.argsort()[-5:][::-1]}")
    top_n = bm25.get_top_n(tokenized_query, corpus, n=k)
    return top_n

def hybrid_retrieve(semantic_chunks:list, keyword_chunks:list, top_k: int):
    both_chunks = semantic_chunks
    both_chunks = both_chunks + [x for x in keyword_chunks if x not in both_chunks]
    rrf_scores = {}
    for chunk in both_chunks:
        if chunk in semantic_chunks:
            semantic_rank = semantic_chunks.index(chunk) + 1
        else:
            semantic_rank = 1000
        if chunk in keyword_chunks:
            keyword_rank = keyword_chunks.index(chunk) + 1
        else:
            keyword_rank = 1000
        rrf = 1 / (keyword_rank + 60) + 1 / (semantic_rank + 60)
        rrf_scores[chunk] = rrf
    sorted_rrf = dict(sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_rrf = [x for x in sorted_rrf.keys()][:top_k]
    return sorted_rrf

def get_context(query, k):
    files = os.listdir("./data")
    files = [x for x in files if '.ipynb' not in x]
    all_text = read_data(files)
    chunks = get_chunks(all_text)
    if collection.count() == 0:
        index_chunks(chunks)
    result = retrieve(query, top_k=k)
    chunks_text = [x['text'] for x in chunks]
    keyword_result = keyword_retrieve(chunks_text, query = query, k=k)
    hybrid_chunks = hybrid_retrieve(result['documents'][0], keyword_result, top_k = 5)
    return hybrid_chunks


if __name__ == "__main__":

    k = 5
    query = "What product category did Appolonia Blewitt purchase and how much did she pay?"
    # result = retrieve(query, top_k=k)
    print("#"*100)
    context = get_context(query, k)
    print(context)
    # print(result)
    # print("#"*100)
    # print(result['documents'][0])
    # print("#"*100)
    # chunks_text = [x['text'] for x in chunks]
    # keyword_result = keyword_retrieve(chunks_text, query = query, k=k)
    # hybrid_chunks = hybrid_retrieve(result['documents'][0], keyword_result, top_k = 5)
    # # print(hybrid_chunks)
    # print("SEMANTIC CHUNKS:")
    # for c in result['documents'][0]:
    #     print(c[:100])
    # print("#"*100)
    # print("KEYWORD CHUNKS:")
    # for c in keyword_result:
    #     print(c[:100])
    # print("#"*100)
    # print("HYBRID CHUNKS:")
    # for c in hybrid_chunks:
    #     if 'appolonia' in c.lower():
    #         print("APPOLONIA CONFIRMED IN HYBRID RESULTS")
    #         print(c)
    # result1 = retrieve(query, top_k=k)
    # result2 = retrieve(query, top_k=k)

    # ids1 = result1['ids'][0]
    # ids2 = result2['ids'][0]

    # print("First call IDs:", ids1)
    # print("Second call IDs:", ids2)
    # print("Are they identical?", ids1 == ids2)