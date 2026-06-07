import ollama
import os
# from ingest import read_data
# from chunker import get_chunks
from vectorstore import retrieve

def get_llm_output(query, context):
    system_prompt = """
    You are a data analyst. Answer ONLY using the exact data provided in the context below.
    Do not infer, calculate averages, or reference any data not explicitly present in the context.
    If the context doesn't contain enough data to answer fully, say exactly that.
    """
    response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query + '\n' + context}
        ]
    )
    answer = response["message"]["content"]
    return answer


if __name__ == "__main__":
    # files = os.listdir("./data")
    # files = [x for x in files if '.ipynb' not in x]
    # all_text = read_data(files)
    # chunks = get_chunks(all_text)
    # index_chunks(chunks)
    query = "what is the average transaction amount for product category 'Food'?"
    results = retrieve(query, top_k=10)
    context = "\n\n".join(results["documents"][0])
    answer = get_llm_output(query, context)
    print("#"*100)
    print(answer)