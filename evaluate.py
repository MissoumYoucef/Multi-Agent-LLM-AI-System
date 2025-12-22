import os
import asyncio
from src.rag.loader import PDFLoader
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.agents.orchestrator import Orchestrator
# from src.evaluation.generator import DatasetGenerator
from src.evaluation.metrics import EvaluationMetrics
from src.utils.config import PDF_DATA_PATH

async def run_evaluation():
    print("--- Starting Evaluation ---")
    
    # 1. Load Data
    loader = PDFLoader(PDF_DATA_PATH)
    documents = loader.load_documents()
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Setup RAG
    vs_manager = VectorStoreManager()
    vectorstore, splits = vs_manager.create_vector_store(documents)
    retriever = HybridRetriever(vectorstore, splits)

    # 3. Setup Orchestrator
    orchestrator = Orchestrator(retriever)

    # 4. Define Evaluation Dataset (Static)
    print("\n--- Using Static Evaluation Dataset ---")
    # Since we removed the generator, we use a static list or load from file.
    # Ideally, this should be loaded from a json file.
    dataset = [
        {"question": "What is the main topic of the document?", "answer": "Refer to text.pdf content"},
        {"question": "Summarize the key points.", "answer": "Refer to text.pdf content"}
    ]
    print(f"Using {len(dataset)} static Q&A pairs for demo.")

    # 5. Run Evaluation
    print("\n--- Running Evaluation Metrics ---")
    metrics = EvaluationMetrics()
    results = []

    for item in dataset:
        question = item.get("question")
        reference = item.get("answer")
        
        print(f"\nEvaluating Question: {question}")
        
        # Run system
        response = orchestrator.run(question)
        # Extract final answer from the state
        # The graph returns the final state. The 'analysis' node is the last one.
        # But the 'solution' might be what we want to compare against the reference answer,
        # or the 'analysis' if that's the final output.
        # Let's use the 'solution' from the solver as the answer to check correctness,
        # and maybe 'analysis' as a secondary check.
        # For this demo, let's assume the 'solution' is the direct answer.
        prediction = response.get("solution", "")
        
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")

        # Compute Metrics
        # Functional: Check if reference keywords are in prediction (simple proxy)
        # We'll just split reference into words as keywords for this simple demo
        keywords = reference.split()
        func_score = metrics.functional_correctness(prediction, keywords)
        
        # Lexical
        lex_score = metrics.lexical_exactness(prediction, reference)
        
        # AI Judge
        judge_score = metrics.ai_judge(question, prediction, reference)
        
        results.append({
            "question": question,
            "functional_score": func_score,
            "lexical_score": lex_score,
            "judge_score": judge_score
        })

    # 6. Print Summary
    print("\n--- Evaluation Results ---")
    avg_func = sum(r["functional_score"] for r in results) / len(results) if results else 0
    avg_lex = sum(r["lexical_score"] for r in results) / len(results) if results else 0
    avg_judge = sum(r["judge_score"] for r in results) / len(results) if results else 0

    print(f"Average Functional Correctness: {avg_func:.2f}")
    print(f"Average Lexical Exactness: {avg_lex:.2f}")
    print(f"Average AI Judge Score: {avg_judge:.2f}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
