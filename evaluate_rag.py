import os
import pandas as pd
from rag_engine import get_rag_chain_manual
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

def run_evaluation():
    print("Starting Ragas Evaluation...")
    
    # Define test questions and expected answers (ground truth)
    test_data = [
        {
            "question": "What is the policy on remote work in the employee handbook?",
            "ground_truth": "The employee handbook states that remote work is allowed up to 2 days a week with manager approval.",
            "role": "general"
        },
        {
            "question": "What were the marketing expenses for Q1 2024?",
            "ground_truth": "Marketing expenses for Q1 2024 were $150,000 for digital advertising.",
            "role": "finance"
        }
    ]
    
    # Initialize Ragas evaluator LLM (using Groq)
    evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    results = []
    
    for item in test_data:
        role = item["role"]
        rag_chain, retriever = get_rag_chain_manual(role)
        
        # Get answer
        answer = rag_chain.invoke(item["question"])
        
        # Get context
        docs = retriever.invoke(item["question"])
        contexts = [doc.page_content for doc in docs]
        
        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })
    
    # Convert to dataset
    dataset = Dataset.from_list(results)
    
    # Run evaluation
    # Note: Ragas metrics might need OpenAI by default, we'll try to override or use a simpler check
    print("Evaluating metrics...")
    try:
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=embeddings
        )
        print("\nEvaluation Results:")
        print(score)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Note: Ragas often requires OpenAI API keys for some metrics. Manual verification in LangSmith is recommended.")

if __name__ == "__main__":
    run_evaluation()
