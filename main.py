import os
from rag import get_llm_output
import chromadb
from ingest import read_data
from chunker import get_chunks
from vectorstore import index_chunks, retrieve, get_context

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("rag_collection")
    if collection.count() == 0:
        print("Vector Store empty, building embeddings!")
        files = os.listdir("./data")
        files = [x for x in files if '.ipynb' not in x]
        all_text = read_data(files)
        chunks = get_chunks(all_text)
        index_chunks(chunks)
    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        # results = retrieve(query, top_k=10)
        # # print("RETRIEVED CONTEXT:")
        # context = get_context(query, k= 5)
        # context = "\n\n".join(context)
        
        # print(context)
        # answer = get_llm_output(query, context)
        print("#"*100)
        # print(answer)
        context = get_context(query, k=5)
        context_str = "\n\n".join(context)
        # print("CONTEXT HASH:", hash(context_str))
        answer = get_llm_output(query, context_str)
        print(answer)
