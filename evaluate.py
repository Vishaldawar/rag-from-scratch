import os
import ollama
import chromadb
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# Replace with
from openai import OpenAI
from ragas.llms import llm_factory
from datasets import Dataset
from rag import get_llm_output
from vectorstore import retrieve

from openai import OpenAI
from ragas.llms import llm_factory

# Point RAGAs to Ollama's OpenAI-compatible endpoint
client = OpenAI(
    api_key="ollama",  # Ollama doesn't need a real key
    base_url="http://localhost:11434/v1"
)
# llm = llm_factory("llama3.2", provider="openai", client=client)

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Configure RAGAs to use Ollama
llm = LangchainLLMWrapper(ChatOllama(model="llama3.2"))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

# Configure RAGAs to use Ollama instead of OpenAI
# llm = LangchainLLMWrapper(ChatOllama(model="llama3.2"))
# embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
# embeddings = llm_factory("nomic-embed-text", provider="openai", client=client)

# Test dataset
test_cases = [
    {
        "question": "What product category did Appolonia Blewitt purchase and how much did she pay?",
        "ground_truth": "Appolonia Blewitt bought a garment from outerwear clothing category for $215.38."
    },
    {
        "question": "Which customers paid with Ngultrum currency?",
        "ground_truth": "Only one customer paid with Ngultrum currency, and the customer was Hedvig Trumble."
    },
    {
        "question": "Which transaction had the highest amount?",
        "ground_truth": "The transaction with transaction id 650 had the highest amount."
    }
]

def run_evaluation():
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Running pipeline for each test case...\n")
    for i, test in enumerate(test_cases):
        print(f"Test {i+1}: {test['question']}")
        
        # Run your existing pipeline
        results = retrieve(test["question"], top_k=10)
        context_docs = results["documents"][0]
        context = "\n\n".join(context_docs)
        answer = get_llm_output(test["question"], context)

        questions.append(test["question"])
        answers.append(answer)
        contexts.append(context_docs)  # RAGAs expects a list of strings per question
        ground_truths.append(test["ground_truth"])

        print(f"Answer: {answer[:200]}...")  # preview first 200 chars
        print("-" * 50)

    # Build RAGAs dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\nRunning RAGAs evaluation...\n")
    # results = evaluate(
    #     dataset=dataset,
    #     metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    #     llm=llm,
    #     embeddings=embeddings
    # )
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    print("\n===== EVALUATION RESULTS =====")
    print(results)
    
    # Save to CSV for analysis
    results_df = results.to_pandas()
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()