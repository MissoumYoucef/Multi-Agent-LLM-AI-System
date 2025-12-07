from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStore

class HybridRetriever:
    def __init__(self, vectorstore: VectorStore, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self):
        # 1. Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 2. Keyword Retriever (BM25)
        # BM25 requires the raw documents (chunks) to build the index
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = 3

        # 3. Ensemble Retriever (Hybrid)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5] # Equal weighting
        )
        return ensemble_retriever

    def retrieve(self, query: str) -> List[Document]:
        print(f"Retrieving for query: {query}")
        docs = self.retriever.invoke(query)
        return docs
