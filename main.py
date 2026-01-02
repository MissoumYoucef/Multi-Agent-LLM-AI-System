import os
import sys
import json
from src.rag.loader import PDFLoader
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.agents.orchestrator import Orchestrator
from src.utils.config import PDF_DATA_PATH, DAILY_BUDGET_USD, CACHE_ENABLED
from src.rag.freshness_tracker import create_freshness_tracker

def main():
    """
    Main entry point for the Multi-Agent RAG System.
    
    Initializes the RAG pipeline with document loading, vector store creation,
    and orchestrator setup. Provides an interactive loop for user queries.
    """
    print("--- Multi-Agent RAG System ---")

    # Initialize Freshness Tracker
    freshness_tracker = create_freshness_tracker(staleness_days=30)
    
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
    vs_manager = VectorStoreManager(freshness_tracker=freshness_tracker)
    vectorstore, splits = vs_manager.create_vector_store(documents)
    
    # Check for stale documents
    stale_docs = freshness_tracker.get_stale_documents()
    if stale_docs:
        print(f"Warning: {len(stale_docs)} documents are stale.")

    retriever = HybridRetriever(vectorstore, splits)

    # 3. Setup Orchestrator
    print("Initializing Agents...")
    # Initialize with all utilities enabled
    orchestrator = Orchestrator(
        retriever=retriever,
        enable_guardrails=True,
        enable_reflection=True,
        enable_react=False, # Use solver by default
        enable_caching=CACHE_ENABLED,
        enable_cost_control=True,
        enable_memory=True,
        enable_evaluation=True,
        daily_budget=DAILY_BUDGET_USD
    )

    # 4. Interactive Loop
    print("\nSystem Ready! Type 'exit' to quit.")
    
    try:
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            try:
                print("Processing...")
                result = orchestrator.run(query)
                
                print("\n--- Result ---")
                print(f"Problem: {result.get('problem')}")
                if result.get('cache_hit'):
                    print("[Cache Hit]")
                    
                print(f"Solution: {result.get('solution')}")
                print(f"Analysis: {result.get('analysis')}")
                
                if result.get('quality_score') is not None:
                     print(f"Quality Score: {result['quality_score']:.2f}")
                     
                print("----------------")
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # 5. Display Statistics
        print("\n--- Session Statistics ---")
        try:
            stats = orchestrator.get_stats()
            
            # Freshness Stats
            freshness_stats = freshness_tracker.get_stats()
            print("\nDocument Freshness:")
            print(json.dumps(freshness_stats, default=str, indent=2))
            
            # System Stats
            print("\nSystem Usage:")
            print(json.dumps(stats, default=str, indent=2))
        except Exception as e:
            print(f"Error displaying stats: {e}")

if __name__ == "__main__":
    main()
