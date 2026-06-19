import ollama
import os
from ingest import read_data
from chunker import get_chunks
from vectorstore import get_context
from cache import get_cached_response, cache_response

def get_llm_output(query, context):
    cached = get_cached_response(query, context)
    if cached is not None:
        return cached

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
        ],
        options={
        "temperature": 0,
        "seed": 42
    }
    )
    answer = response["message"]["content"]
        # existing LLM call logic here
    
    cache_response(query, context, answer)
    return answer


if __name__ == "__main__":
    files = os.listdir("./data")
    files = [x for x in files if '.ipynb' not in x]
    all_text = read_data(files)
    chunks = get_chunks(all_text)
    # index_chunks(chunks)
    # query = "what is the average transaction amount for product category 'Food'?"
    query = "What product category did Appolonia Blewitt purchase and how much did she pay?"
    # result = retrieve(query, top_k=10)
    # chunks_text = [x['text'] for x in chunks]
    k = 5
    # keyword_result = keyword_retrieve(chunks_text, query = query, k=k)
    # hybrid_chunks = hybrid_retrieve(result['documents'][0], keyword_result, top_k = k)
    context = get_context(query, k)

    # context = "\n\n".join(result["documents"][0])
    context = "\n\n".join(context)
    answer = get_llm_output(query, context)
    print("#"*100)
    print(answer)