import os
import sys
from src.rag.loader import PDFLoader
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.agents.orchestrator import Orchestrator
from src.utils.config import PDF_DATA_PATH

def main():
    print("--- Multi-Agent RAG System ---")
    
    # 1. Load Data
    print("Loading data...")
    loader = PDFLoader(PDF_DATA_PATH, load_all=True)
    documents = loader.load_documents()
    if not documents:
        print("No documents found. Please ensure PDFs are in data/pdfs/")
        # For first run, we might need to wait for generate_pdf to finish or run it here if missing
        return

    # 2. Setup RAG
    print("Setting up RAG pipeline...")
    vs_manager = VectorStoreManager()
    # Check if vectorstore exists to avoid re-creating every time?
    # For this demo, we'll create it fresh or load if exists logic is in vectorstore.py
    # But vectorstore.py currently creates fresh splits every time in create_vector_store.
    # Let's just run it.
    vectorstore, splits = vs_manager.create_vector_store(documents)
    retriever = HybridRetriever(vectorstore, splits)

    # 3. Setup Orchestrator
    print("Initializing Agents...")
    orchestrator = Orchestrator(retriever)

    # 4. Interactive Loop
    print("\nSystem Ready! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        try:
            print("Processing...")
            result = orchestrator.run(query)
            
            print("\n--- Result ---")
            print(f"Problem: {result.get('problem')}")
            print(f"Context Retrieved: {len(result.get('context', ''))} chars")
            print(f"Solution: {result.get('solution')}")
            print(f"Analysis: {result.get('analysis')}")
            print("----------------")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
