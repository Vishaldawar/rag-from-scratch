import ollama
import chromadb
import os
from ingest import read_data
from chunker import get_chunks

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_collection")

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


if __name__ == "__main__":
    files = os.listdir("./data")
    files = [x for x in files if '.ipynb' not in x]
    all_text = read_data(files)
    chunks = get_chunks(all_text)
    index_chunks(chunks)
    result = retrieve("what is the transaction amount for product category 'Food'?", top_k=5)
    print("#"*100)
    print(result)
    print("#"*100)
    print(result['documents'][0])