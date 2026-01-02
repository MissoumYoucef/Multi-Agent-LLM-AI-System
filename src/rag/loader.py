"""
PDF Loader module.

Provides functionality to load and clean PDF documents.
"""
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Optional

logger = logging.getLogger(__name__)

# Minimum content length to consider a page valid
MIN_CONTENT_LENGTH = 50


class PDFLoader:
    """
    Loads PDF documents from a directory.

    Supports loading a specific file (text.pdf) or all PDFs in the directory.
    """

    def __init__(self, directory_path: str, load_all: bool = False):
        """
        Initialize the PDF loader.

        Args:
            directory_path: Path to the directory containing PDF files.
            load_all: If True, load all PDFs in directory. If False, only load text.pdf.
        """
        self.directory_path = directory_path
        self.load_all = load_all

    def load_documents(self) -> List[Document]:
        """
        Load documents from PDF files.

        Returns:
            A list of Document objects with page content and metadata.
        """
        documents = []

        if not os.path.exists(self.directory_path):
            logger.error(f"Directory {self.directory_path} does not exist.")
            return []

        if self.load_all:
            documents = self._load_all_pdfs()
        else:
            documents = self._load_text_pdf()

        return self.clean_documents(documents)

    def _load_text_pdf(self) -> List[Document]:
        """Load only text.pdf from the directory."""
        documents = []
        file_path = os.path.join(self.directory_path, "text.pdf")

        if os.path.exists(file_path):
            logger.info(f"Loading {file_path}...")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from text.pdf")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"text.pdf not found in {self.directory_path}")

        return documents

    def _load_all_pdfs(self) -> List[Document]:
        """Load all PDF files from the directory."""
        documents = []
        pdf_files = [f for f in os.listdir(self.directory_path) if f.endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.directory_path}")
            return []

        for filename in pdf_files:
            file_path = os.path.join(self.directory_path, filename)
            logger.info(f"Loading {file_path}...")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Add source filename to metadata
                for doc in docs:
                    doc.metadata['source_file'] = filename
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {filename}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return documents

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean and deduplicate documents.

        Removes empty pages, very short pages, and duplicates.

        Args:
            documents: List of documents to clean.

        Returns:
            Cleaned list of unique documents.
        """
        cleaned_docs = []
        seen_content = set()

        for doc in documents:
            content = doc.page_content.strip()
            # Remove empty pages, short pages, and duplicates
            if (content and
                len(content) >= MIN_CONTENT_LENGTH and
                content not in seen_content):
                doc.page_content = content
                cleaned_docs.append(doc)
                seen_content.add(content)

        logger.info(f"Loaded {len(cleaned_docs)} unique pages from PDFs "
                   f"(filtered {len(documents) - len(cleaned_docs)} pages).")
        return cleaned_docs

