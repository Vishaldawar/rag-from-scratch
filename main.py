import os
from rag import get_llm_output
import chromadb
import ollama
from ingest import read_data
from chunker import get_chunks
from vectorstore import index_chunks, retrieve, get_context
from pandas_agent import pandas_agent_class

def classify_query(query: str) -> str:
    system_prompt = """
    Classify the following question into exactly one category:
    - "lookup" — asking about a specific entity, record, or fact (e.g. "what did X purchase", "which bank does Y use")
    - "aggregation" — asking for a calculation, total, average, count, max, min, or grouping across the dataset (e.g. "average transaction amount", "how many customers paid in Euro")
    
    Respond with ONLY one word: "lookup" or "aggregation". No explanation.
    
    Question: {query}
    """
    response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
    answer = response["message"]["content"]
    return answer

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
        # print("#"*100)
        # print(answer)
        category = classify_query(query)
        if category == "aggregation":
            pandas_agent_obj = pandas_agent_class(csv_path="./data/mock_data.csv")
            answer = pandas_agent_obj.pandas_agent(query)
            try:
                print(f"\nAnswer: {answer}")
            except:
                print("Could not get an answer from the LLM!")
            print(f"\n[Source — generated code]:\n{pandas_agent_obj.code}")
        else:
            context = get_context(query, k=5)
            context_str = "\n\n".join(context)
            # print("CONTEXT HASH:", hash(context_str))
            answer = get_llm_output(query, context_str)
            print(f"\nAnswer: {answer}")
            print(f"\n[Source — retrieved context]:\n{context}")
