import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Fallback or warning if not set, though user should set it.
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# Configuration constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_STORE_PATH = "./chroma_db"
PDF_DATA_PATH = "./data/pdfs"  # Directory containing text.pdf
