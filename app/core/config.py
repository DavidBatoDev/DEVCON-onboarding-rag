# backend/app/core/config.py
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env file correctly from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # Google Drive Configuration
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "keys/devcon-460307-b6c246eeb117.json")
    GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    # RAG Configuration
    RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "app/storage/rag_index")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    
    # LLM Configuration  
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "app/storage/chroma_db")
    
    # Embedding Configuration - BGE via HuggingFace Inference API
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto, cpu, cuda
    EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./hf_cache")
    
    # BGE Specific Settings
    BGE_NORMALIZE_EMBEDDINGS = os.getenv("BGE_NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    BGE_TRUST_REMOTE_CODE = os.getenv("BGE_TRUST_REMOTE_CODE", "true").lower() == "true"
    
    # Performance Settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    ENABLE_GPU = os.getenv("ENABLE_GPU", "true").lower() == "true"
    
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    VITE_API_URL = os.getenv("VITE_API_URL", "http://localhost:8000/api/v1/ask")


settings = Settings()
