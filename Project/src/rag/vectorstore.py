import os
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from ..utils.config import GOOGLE_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, VECTOR_STORE_PATH

class VectorStoreManager:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        self.vector_store_path = VECTOR_STORE_PATH

    def create_vector_store(self, documents: List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks from documents.")

        if os.path.exists(self.vector_store_path):
             # Simple check to avoid re-creating if not needed for this demo, 
             # but for a clean run we might want to overwrite or persist.
             # For this demo, let's just create a new one in memory or persist to disk.
             # We will persist to disk to allow retrieval later.
             pass

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        print(f"Vector store created at {self.vector_store_path}")
        return vectorstore, splits

    def load_vector_store(self):
        if not os.path.exists(self.vector_store_path):
            return None
            
        vectorstore = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
        )
        return vectorstore
