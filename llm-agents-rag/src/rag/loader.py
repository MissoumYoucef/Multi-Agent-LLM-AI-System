import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

class PDFLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load_documents(self) -> List[Document]:
        documents = []
        if not os.path.exists(self.directory_path):
            print(f"Directory {self.directory_path} does not exist.")
            return []

        file_path = os.path.join(self.directory_path, "text.pdf")
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
             print(f"Warning: {file_path} not found. Please ensure 'text.pdf' is in {self.directory_path}")
        
        return self.clean_documents(documents)

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        cleaned_docs = []
        seen_content = set()
        
        for doc in documents:
            content = doc.page_content.strip()
            # Remove empty pages and duplicates
            if content and content not in seen_content:
                doc.page_content = content # Update with stripped content
                cleaned_docs.append(doc)
                seen_content.add(content)
        
        print(f"Loaded {len(cleaned_docs)} unique pages from PDFs.")
        return cleaned_docs
