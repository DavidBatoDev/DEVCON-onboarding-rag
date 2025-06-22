"""
Enhanced compilation script for LlamaIndex RAG system with HF Inference API
Processes Google Drive documents and builds the RAG index using BAAI/bge-large-en-v1.5 via HF API
Updated to use huggingface_hub.InferenceClient with proper response handling
"""

import sys
import logging
from pathlib import Path
from typing import List
import time
import numpy as np
from huggingface_hub import InferenceClient

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.services.llamaindex_service import rag_service
from app.services.document_processor import document_processor
from app.services.drive_loader import get_drive_files
from app.core.config import settings

# Setup logging with better encoding handling
def setup_logging():
    """Setup logging with proper encoding for Windows console"""
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (can handle unicode)
    file_handler = logging.FileHandler('compilation.log', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # Console handler (avoid emojis for Windows compatibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


def print_banner():
    """Print a nice banner for the compilation process"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DEVCON RAG COMPILER                       ║
    ║            Enhanced with HF Inference API                   ║
    ║              BAAI/bge-large-en-v1.5                        ║
    ║                  Cloud-Powered Embeddings                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_environment():
    """Validate that all required environment variables are set"""
    logger.info("Validating environment...")
    
    required_settings = [
        ("GEMINI_API_KEY", settings.GEMINI_API_KEY),
        ("GOOGLE_SERVICE_ACCOUNT_JSON_DATA", settings.GOOGLE_SERVICE_ACCOUNT_JSON_DATA),
        ("HUGGINGFACE_TOKEN", settings.HUGGINGFACE_TOKEN)
    ]
    
    missing_vars = []
    for var_name, var_value in required_settings:
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        logger.error(f"Missing required configuration values: {missing_vars}")
        logger.error("HUGGINGFACE_TOKEN is required for Inference API access")
        sys.exit(1)
    
    logger.info("Environment validation passed")
    logger.info("Using Hugging Face Inference API (cloud-based)")
    logger.info("No local GPU required - embeddings computed in the cloud")


def test_hf_inference_api():
    """Test Hugging Face Inference API connectivity using InferenceClient"""
    logger.info("Testing Hugging Face Inference API...")
    
    try:
        # Initialize the modern InferenceClient with token from settings
        client = InferenceClient(
            api_key=settings.HUGGINGFACE_TOKEN
        )
        
        # Test with a simple feature extraction call
        test_text = "Test API connectivity"
        
        logger.info("Testing API connectivity...")
        
        # Use the feature_extraction method
        embeddings = client.feature_extraction(
            text=test_text,
            model=settings.EMBEDDING_MODEL
        )
        
        # Fix: Proper handling of numpy array response
        if embeddings is not None and len(embeddings) > 0:
            # Convert to list if it's a numpy array
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.info("HF Inference API is responding correctly")
            logger.info(f"Test embedding dimensions: {len(embeddings)}")
            logger.info(f"Model: {settings.EMBEDDING_MODEL}")
            logger.info("Embedding computation: Cloud-based via HF Inference API")
            return True
        else:
            logger.error("API returned empty response")
            return False
        
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg:
            logger.error("Invalid Hugging Face token")
            logger.error("Please check your HUGGINGFACE_TOKEN in settings")
        elif "404" in error_msg or "not found" in error_msg:
            logger.error(f"Model {settings.EMBEDDING_MODEL} not found or not available")
            logger.error("The model might be loading or temporarily unavailable")
        elif "503" in error_msg or "loading" in error_msg:
            logger.info("Model is loading on HF servers, this is normal for first use")
            logger.info("The model will be available shortly...")
            return True  # Consider this a success as the API is responding
        else:
            logger.error(f"Error testing HF Inference API: {e}")
            logger.error("Please check your internet connection and HF token")
        return False


class HFInferenceEmbedding:
    """Embedding class that uses HuggingFace InferenceClient"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.client = InferenceClient(api_key=settings.HUGGINGFACE_TOKEN)
        self._dimension = None
    
    def _process_embedding_response(self, embedding):
        """Process embedding response to handle different formats"""
        try:
            # Convert numpy array to list if needed
            if isinstance(embedding, np.ndarray):
                if embedding.ndim == 2:
                    # If 2D array, take first row
                    return embedding[0].tolist()
                else:
                    return embedding.tolist()
            
            # Handle list format
            if isinstance(embedding, list):
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    # Multiple embeddings returned, take the first
                    return embedding[0]
                return embedding
            
            logger.error(f"Unexpected embedding format: {type(embedding)}")
            return []
            
        except Exception as e:
            logger.error(f"Error processing embedding response: {e}")
            return []
    
    def get_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        all_embeddings = []
        
        # Process in batches to respect rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    # Get embedding for single text
                    embedding = self.client.feature_extraction(
                        text=text,
                        model=self.model_name
                    )
                    
                    # Process the response properly
                    processed_embedding = self._process_embedding_response(embedding)
                    
                    if processed_embedding:
                        batch_embeddings.append(processed_embedding)
                    else:
                        logger.warning(f"Empty embedding returned for text: {text[:50]}...")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error getting embedding for text: {e}")
                    continue
            
            all_embeddings.extend(batch_embeddings)
            
            # Brief pause between batches
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return all_embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    @property
    def dimension(self) -> int:
        """Get embedding dimensions"""
        if self._dimension is None:
            test_embedding = self.get_embedding("test")
            self._dimension = len(test_embedding) if test_embedding else 0
        return self._dimension


def test_embedding_model():
    """Test the embedding model via HF Inference API"""
    logger.info("Testing BGE embedding model via HF Inference API...")
    
    try:
        # Create embedding instance
        embedding_model = HFInferenceEmbedding(settings.EMBEDDING_MODEL)
        
        # Test with sample text
        test_text = "Testing BGE embeddings for DEVCON documents via HF API"
        embedding = embedding_model.get_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            logger.info("BGE model working successfully via HF Inference API")
            logger.info(f"Model: {settings.EMBEDDING_MODEL}")
            logger.info(f"API Type: HuggingFace InferenceClient")
            logger.info(f"Embedding dimensions: {len(embedding)}")
            logger.info(f"Sample values: {embedding[:5]}...")
            return True
        else:
            logger.error("BGE model test failed: No embedding returned")
            return False
            
    except Exception as e:
        logger.error(f"Error testing BGE model: {e}")
        return False


def get_compilation_stats():
    """Get current compilation statistics"""
    try:
        stats = rag_service.get_index_stats()
        logger.info("Current Index Stats:")
        logger.info(f"  Status: {stats.get('status', 'unknown')}")
        logger.info(f"  Documents: {stats.get('document_count', 0)}")
        logger.info(f"  Embedding Model: {stats.get('embedding_model', 'unknown')}")
        logger.info(f"  Embedding API: {stats.get('embedding_api', 'unknown')}")
        logger.info(f"  Embedding Dims: {stats.get('embedding_dimensions', 'unknown')}")
        logger.info(f"  Chunk Size: {stats.get('chunk_size', 'unknown')}")
        return stats
    except Exception as e:
        logger.warning(f"Could not get compilation stats: {e}")
        return {}


def fetch_and_process_documents(folder_id: str = None) -> List:
    """Fetch documents from Google Drive and process them"""
    logger.info("Fetching documents from Google Drive...")
    
    try:
        # Get files from Google Drive
        files = get_drive_files(folder_id)
        logger.info(f"Found {len(files)} files in Google Drive")
        
        if not files:
            logger.warning("No files found in Google Drive")
            return []
        
        # Log file details
        logger.info("Files to process:")
        for i, file in enumerate(files[:10], 1):  # Show first 10 files
            name = file.get('name', 'Unknown')
            mime_type = file.get('mimeType', 'Unknown')
            size = file.get('size', 'Unknown')
            logger.info(f"  {i}. {name} ({mime_type}) - {size} bytes")
        
        if len(files) > 10:
            logger.info(f"  ... and {len(files) - 10} more files")
        
        # Process files into LlamaIndex Documents
        logger.info("Processing documents...")
        start_time = time.time()
        
        documents = document_processor.process_drive_files(files)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} documents in {processing_time:.2f} seconds")
        
        # Log processing results
        success_rate = (len(documents) / len(files)) * 100 if files else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching/processing documents: {e}")
        raise


def estimate_api_usage(documents: List):
    """Estimate HF Inference API usage"""
    logger.info("Estimating HF Inference API usage...")
    
    try:
        total_chars = sum(len(doc.text) for doc in documents)
        estimated_chunks = total_chars // settings.CHUNK_SIZE + len(documents)
        
        logger.info("Usage estimates:")
        logger.info(f"  Total characters: {total_chars:,}")
        logger.info(f"  Estimated chunks: {estimated_chunks:,}")
        logger.info(f"  API calls needed: ~{estimated_chunks}")
        
        if estimated_chunks > 1000:
            logger.warning("Large number of API calls required")
            logger.warning("Consider processing in smaller batches to respect rate limits")
            return estimated_chunks // 50  # Suggest batch size
        elif estimated_chunks > 500:
            logger.info("Moderate API usage expected")
            return estimated_chunks // 25
        else:
            logger.info("Light API usage expected")
            return None
            
    except Exception as e:
        logger.warning(f"Could not estimate API usage: {e}")
        return None


def build_rag_index(documents: List, batch_size: int = None):
    """Build the RAG index with processed documents using HF Inference API"""
    if not documents:
        logger.warning("No documents to add to index")
        return False
    
    logger.info(f"Building RAG index with {len(documents)} documents...")
    logger.info("Using HF Inference API - embeddings computed in the cloud")
    logger.info("Processing time depends on API response times and rate limits")
    
    start_time = time.time()
    
    try:
        # Estimate and suggest batch processing if needed
        suggested_batch_size = estimate_api_usage(documents)
        if suggested_batch_size and not batch_size:
            batch_size = suggested_batch_size
            logger.info(f"Using suggested batch size: {batch_size}")
        
        # Process documents in batches if specified
        if batch_size and len(documents) > batch_size:
            logger.info(f"Processing documents in batches of {batch_size}")
            logger.info("This helps respect API rate limits and provides better progress tracking")
            
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                batch_start = time.time()
                success = rag_service.add_documents(batch)
                batch_time = time.time() - batch_start
                
                if not success:
                    logger.error(f"Failed to process batch {batch_num}")
                    return False
                
                # Show progress and timing
                processed = min(i + batch_size, len(documents))
                progress = (processed / len(documents)) * 100
                logger.info(f"Progress: {processed}/{len(documents)} ({progress:.1f}%)")
                logger.info(f"Batch processed in {batch_time:.1f}s")
                
                # Brief pause between batches to be respectful to API
                if batch_num < total_batches:
                    logger.info("Brief pause between batches...")
                    time.sleep(2)
        else:
            # Process all documents at once
            logger.info("Processing all documents in single batch")
            success = rag_service.add_documents(documents)
            if not success:
                logger.error("Failed to build RAG index")
                return False
        
        build_time = time.time() - start_time
        logger.info(f"RAG index built successfully in {build_time:.2f} seconds")
        
        # Calculate processing rate
        avg_time_per_doc = build_time / len(documents) if len(documents) > 0 else 0
        logger.info(f"Average processing time: {avg_time_per_doc:.2f} seconds/document")
        
        return True
            
    except Exception as e:
        logger.error(f"Error building RAG index: {e}")
        if "rate limit" in str(e).lower():
            logger.error("This might be due to API rate limits")
            logger.error("Try using smaller batch sizes or waiting before retrying")
        raise


def test_rag_system():
    """Test the RAG system with sample queries"""
    logger.info("Testing RAG system with HF Inference API embeddings...")
    
    test_queries = [
        "What is DEVCON?",
        "Tell me about the event schedule",
        "Who are the speakers?",
        "What are the main topics discussed?",
        "How to apply in DEVCON Internship program?",
        "Who is Dom and what is his role and achievements?"
    ]
    
    total_query_time = 0
    
    for i, query in enumerate(test_queries, 1):
        try:
            logger.info(f"Test {i}: {query}")
            start_time = time.time()
            
            response = rag_service.query(query)
            
            query_time = time.time() - start_time
            total_query_time += query_time
            
            # Log first 100 characters of response
            preview = response[:100] + "..." if len(response) > 100 else response
            logger.info(f"Response ({query_time:.2f}s): {preview}")
            
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")
    
    if total_query_time > 0:
        avg_query_time = total_query_time / len(test_queries)
        logger.info(f"Average query time: {avg_query_time:.2f}s")


def cleanup_temp_files():
    """Clean up any temporary files created during processing"""
    logger.info("Cleaning up temporary files...")
    
    try:
        # Remove any temp files that might have been created
        temp_patterns = ["temp_*.docx", "temp_*.pdf", "*.tmp"]
        
        for pattern in temp_patterns:
            for temp_file in Path(".").glob(pattern):
                temp_file.unlink()
                logger.info(f"Removed: {temp_file}")
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def main():
    """Main compilation process"""
    print_banner()
    
    try:
        # Validate environment
        validate_environment()
        
        # Test HF Inference API
        if not test_hf_inference_api():
            logger.error("HF Inference API test failed. Exiting.")
            sys.exit(1)
        
        # Test embedding model
        if not test_embedding_model():
            logger.error("BGE model test failed. Exiting.")
            sys.exit(1)
        
        # Get initial stats
        logger.info("Getting initial statistics...")
        initial_stats = get_compilation_stats()        # Fetch and process documents
        # You can pass a specific folder_id here if needed
        folder_id = settings.GOOGLE_DRIVE_FOLDER_ID  # Get from settings
        documents = fetch_and_process_documents(folder_id)
        
        if not documents:
            logger.warning("No documents to process. Exiting.")
            return
        
        # Build RAG index with HF Inference API
        # Use smaller batch size for API rate limiting
        batch_size = 25 if len(documents) > 50 else None
        success = build_rag_index(documents, batch_size)
        
        if not success:
            logger.error("Failed to build RAG index")
            sys.exit(1)
        
        # Get final stats
        logger.info("Getting final statistics...")
        final_stats = get_compilation_stats()
        
        # Test the system
        test_rag_system()
        
        # Cleanup
        cleanup_temp_files()
        
        # Final summary
        logger.info("Compilation completed successfully!")
        logger.info(f"Final document count: {final_stats.get('document_count', 0)}")
        logger.info(f"Embedding model: {final_stats.get('embedding_model', 'unknown')}")
        logger.info(f"Embedding API: {final_stats.get('embedding_api', 'unknown')}")
        
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║                   COMPILATION COMPLETE                       ║
        ║                                                              ║
        ║  Your DEVCON RAG system with HF Inference API is ready!     ║
        ║  Cloud-powered BGE embeddings for excellent search quality  ║
        ║  Start the server with: python main.py                      ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
    except KeyboardInterrupt:
        logger.info("Compilation interrupted by user")
        cleanup_temp_files()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        if "api" in str(e).lower() or "inference" in str(e).lower():
            logger.error("This appears to be an API-related error")
            logger.error("Please check your HuggingFace token and internet connection")
        cleanup_temp_files()
        sys.exit(1)


if __name__ == "__main__":
    main()