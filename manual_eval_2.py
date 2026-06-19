import pandas as pd
import numpy as np
import ollama
from vectorstore import retrieve
from sentence_transformers import CrossEncoder
from vectorstore import get_context
from rag import get_llm_output
from chunker import get_chunks
from ingest import read_data
import os

def cosine_similarity(u1, v1) -> float:
    similarity = np.dot(u1, v1) / (np.linalg.norm(u1) * np.linalg.norm(v1))
    return similarity

def answer_relevancy(question: str, answer: str) -> float:
    vector_q = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
    vector_a = ollama.embeddings(model="nomic-embed-text", prompt=answer)["embedding"]
    similarity = cosine_similarity(vector_q, vector_a)
    return np.round(similarity, 4)

def context_precision(question: str, chunks: list[str]) -> float:
    vector_q = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
    precisions = []
    for chunk in chunks:
        vc = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
        precisions.append(cosine_similarity(vector_q, vc))
    return np.round(np.mean(precisions), 4)

def context_recall(ground_truth: str, chunks: list[str]) -> float:
    vector_a = ollama.embeddings(model="nomic-embed-text", prompt=ground_truth)["embedding"]
    recalls = []
    for chunk in chunks:
        vc = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
        recalls.append(cosine_similarity(vector_a, vc))
    return np.round(np.max(recalls), 4)


def faithfullness(nli_model_obj, answer: str, context_chunks: list[str]) -> float:
    scores = []
    for context in context_chunks:
        score = nli_model_obj.predict([[context, answer]])
        # Returns scores for [contradiction, neutral, entailment]
        exp_scores = np.exp(score[0])
        probabilities = exp_scores / exp_scores.sum()
        entailment_score = probabilities[2]  # index 2 is entailment
        scores.append(entailment_score)
    return max(scores)




if __name__ == "__main__":
    files = os.listdir("./data")
    files = [x for x in files if '.ipynb' not in x]
    all_text = read_data(files)
    chunks = get_chunks(all_text)
    question = "What product category did Appolonia Blewitt purchase and how much did she pay?"
    ground_truth = "Appolonia Blewitt bought a garment from outerwear clothing category for $215.38."
    # answer = "I do not have enough data to answer the question fully about what product category Appolonia Blewitt purchased and how much she paid."
    context = get_context(question, k=5)
    context = "\n\n".join(context)
    answer = get_llm_output(question, context)
    relevancy = answer_relevancy(question, answer)
    results = retrieve(question, top_k=5)
    chunks = results["documents"][0]
    print(f"For question : \n{question} \nand answer : \n{answer} : \nthe relevancy score is : {relevancy}")
    precision = context_precision(question, chunks)
    print(f"For question : \n{question} \nand answer : \n{answer} : \nthe context precision score is : {precision}")
    recall = context_recall(ground_truth, chunks)
    print(f"For question : \n{question} \nand answer : \n{answer} : \nthe context recall score is : {recall}")

    ## Natural Language Inference for faithfulness
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    faithfulness_score = faithfullness(nli_model, answer, chunks)
    print(f"For question : \n{question} \nand answer : \n{answer} : \nthe faithfulness score is : {faithfulness_score}")