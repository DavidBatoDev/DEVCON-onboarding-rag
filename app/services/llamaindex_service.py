# backend/app/services/llamaindex_service.py
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings as LlamaSettings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response import Response
from llama_index.core.embeddings import BaseEmbedding
from typing import List, Optional, Any
from pydantic import Field

import chromadb
from chromadb.config import Settings as ChromaSettings
import os
import numpy as np
import time
from typing import List, Optional, Any
from pathlib import Path
from app.core.config import settings
from huggingface_hub import InferenceClient
import logging
import json
from app.services.prompt_engine import DEVCONPromptEngine 



logger = logging.getLogger(__name__)


class HuggingFaceInferenceEmbedding(BaseEmbedding):
    """Custom embedding class using Hugging Face InferenceClient"""
    
    # Define Pydantic fields explicitly
    model_name: str = Field(default="BAAI/bge-large-en-v1.5", description="HuggingFace model name")
    api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    timeout: int = Field(default=60, description="API timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")
    
    # These will be set during initialization
    client: Optional[InferenceClient] = Field(default=None, init=False)
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
        
        # Initialize the InferenceClient with the modern approach
        self.client = InferenceClient(
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        print(f"✅ Initialized HF InferenceClient embedding: {self.model_name}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        return self._get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        return self._call_api([text])[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        return self._call_api(texts)
    
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Make API call to Hugging Face Inference API using InferenceClient"""
        all_embeddings = []
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Getting embeddings for {len(texts)} texts (attempt {attempt + 1})")
                
                # Process texts one by one to handle API properly
                batch_embeddings = []
                for text in texts:
                    try:
                        # Use the feature_extraction method from InferenceClient
                        embedding = self.client.feature_extraction(
                            text=text,
                            model=self.model_name
                        )
                        
                        # Process the embedding response
                        processed_embedding = self._process_embedding_response(embedding)
                        if processed_embedding:
                            batch_embeddings.append(processed_embedding)
                        else:
                            print(f"⚠️ Empty embedding for text: {text[:50]}...")
                            # Return a zero vector as fallback
                            batch_embeddings.append([0.0] * 1024)  # BGE-large dimension
                            
                    except Exception as text_error:
                        print(f"⚠️ Error processing single text: {text_error}")
                        # Return a zero vector as fallback
                        batch_embeddings.append([0.0] * 1024)
                        continue
                
                all_embeddings = batch_embeddings
                
                # Normalize embeddings if requested
                if self.normalize_embeddings:
                    all_embeddings = [self._normalize_embedding(emb) for emb in all_embeddings]
                
                print(f"✅ Got {len(all_embeddings)} embeddings, dim: {len(all_embeddings[0]) if all_embeddings else 0}")
                return all_embeddings
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"⚠️ API request error (attempt {attempt + 1}): {e}")
                
                if "503" in error_msg or "loading" in error_msg:
                    # Model is loading, wait and retry
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"⏳ Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif "404" in error_msg:
                    print(f"❌ Model not found: {self.model_name}")
                    print("🔍 Check if model name is correct and accessible")
                    raise Exception(f"Model not found: {self.model_name}")
                elif "401" in error_msg or "unauthorized" in error_msg:
                    print("❌ Unauthorized - check your HuggingFace token")
                    raise Exception("Invalid HuggingFace API token")
                elif "429" in error_msg or "rate limit" in error_msg:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"⏳ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to get embeddings after {self.max_retries} attempts: {e}")
                time.sleep(self.retry_delay * (attempt + 1))
            
        # If we get here, all attempts failed
        raise Exception("Failed to get embeddings after all retry attempts")
    
    def _process_embedding_response(self, embedding) -> List[float]:
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
            
            print(f"⚠️ Unexpected embedding format: {type(embedding)}")
            return []
            
        except Exception as e:
            print(f"❌ Error processing embedding response: {e}")
            return []
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length"""
        try:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm == 0:
                return embedding
            return (embedding_array / norm).tolist()
        except Exception as e:
            print(f"⚠️ Error normalizing embedding: {e}")
            return embedding
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version - fallback to sync for now"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version - fallback to sync for now"""
        return self._get_text_embedding(text)


class LlamaIndexRAGService:
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        self._setup_llama_index()
        self._initialize_components()
        self.index_state_file = Path(settings.RAG_INDEX_DIR) / "index_state.json"
        self._load_index_state()
        self.prompt_engine = DEVCONPromptEngine(self)
    
    def _setup_llama_index(self):
        """Configure LlamaIndex global settings"""
        # Initialize Gemini LLM
        llm = Gemini(
            api_key=settings.GEMINI_API_KEY,
            model="models/gemini-2.0-flash",
            temperature=0.1,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
        )
        
        # Initialize custom Hugging Face InferenceClient Embeddings
        embed_model = HuggingFaceInferenceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.HUGGINGFACE_TOKEN,
            timeout=60,
            max_retries=3,
            retry_delay=1.0,
            normalize_embeddings=settings.BGE_NORMALIZE_EMBEDDINGS,
        )
        
        # Configure global settings
        LlamaSettings.llm = llm
        LlamaSettings.embed_model = embed_model
        LlamaSettings.chunk_size = settings.CHUNK_SIZE
        LlamaSettings.chunk_overlap = settings.CHUNK_OVERLAP
        
        print(f"✅ Initialized {settings.EMBEDDING_MODEL} embeddings via HF InferenceClient")
    
    def _initialize_components(self):
        """Initialize vector store and other components"""
        # Setup ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        collection_name = "devcon_documents_hf_client"  # Updated name for HF InferenceClient
        try:
            chroma_collection = chroma_client.get_collection(collection_name)
            print(f"✅ Loaded existing ChromaDB collection: {collection_name}")
        except:
            chroma_collection = chroma_client.create_collection(collection_name)
            print(f"✅ Created new ChromaDB collection: {collection_name}")
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Try to load existing index
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one"""
        storage_dir = Path(settings.RAG_INDEX_DIR)
        
        try:
            if storage_dir.exists() and any(storage_dir.iterdir()):
                # Load existing index
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(storage_dir),
                    vector_store=self.vector_store
                )
                self.index = load_index_from_storage(storage_context)
                print("✅ Loaded existing LlamaIndex")
            else:
                # Create new index
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex([], storage_context=storage_context)
                print("✅ Created new LlamaIndex")
                
        except Exception as e:
            print(f"⚠️ Error loading index, creating new one: {e}")
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex([], storage_context=storage_context)
        
        self._setup_query_engine()
    
    def _load_index_state(self):
        """Load index state tracking file modifications"""
        self.index_state = {}
        if self.index_state_file.exists():
            try:
                with open(self.index_state_file, "r") as f:
                    self.index_state = json.load(f)
                logger.info(f"Loaded index state with {len(self.index_state)} files")
            except Exception as e:
                logger.error(f"Error loading index state: {e}")
    
    def save_index_state(self):
        """Save current index state to disk"""
        try:
            with open(self.index_state_file, "w") as f:
                json.dump(self.index_state, f, indent=2)
            logger.info("Index state saved")
        except Exception as e:
            logger.error(f"Error saving index state: {e}")
    
    def get_index_state(self):
        """Return current index state"""
        return self.index_state

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata to ensure ChromaDB compatibility"""
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float)):
                sanitized[key] = value
            elif isinstance(value, bool):
                sanitized[key] = str(value)
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(item) for item in value) if value else ""
            elif isinstance(value, dict):
                # Convert dicts to JSON-like string representation
                sanitized[key] = str(value)
            else:
                # Convert any other type to string
                sanitized[key] = str(value)
        
        return sanitized

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the index with metadata sanitization"""
        try:
            if not self.index:
                self._load_or_create_index()
            
            # Sanitize metadata for all documents before processing
            print("🧹 Sanitizing document metadata for ChromaDB compatibility...")
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc.metadata = self._sanitize_metadata(doc.metadata)
                    print(f"📄 Sanitized metadata for document: {doc.metadata.get('title', 'Unknown')[:50]}...")
            
            # Use sentence splitter for better chunking
            parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            # Parse documents into nodes
            nodes = parser.get_nodes_from_documents(documents)
            print(f"📄 Created {len(nodes)} nodes from {len(documents)} documents")
            
            # Sanitize node metadata as well
            print("🧹 Sanitizing node metadata...")
            for node in nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    node.metadata = self._sanitize_metadata(node.metadata)
            
            # Add to index with progress tracking
            print("🔄 Adding nodes to index...")
            self.index.insert_nodes(nodes)
            
            # Persist the index
            self.persist_index()
            
            # Refresh query engine
            self._setup_query_engine()

            for doc in documents:
                file_id = doc.metadata["file_id"]
                self.index_state[file_id] = {
                    "file_id": file_id,
                    "file_name": doc.metadata["source"],
                    "title": doc.metadata.get("title", doc.metadata["source"]),
                    "modified_time": doc.metadata["modified_time"],
                    "chunk_count": len(nodes)  # Assuming nodes are created
                }
            
            self.save_index_state()
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            # Print more detailed error info for debugging
            import traceback
            print(f"🔍 Full error traceback:")
            traceback.print_exc()
            return False
        
    def remove_document(self, file_id: str) -> bool:
        """Remove a document from the index by file ID"""
        try:
            if self.index:
                # Delete from index
                self.index.delete_ref_doc(file_id, delete_from_docstore=True)
                
                # Delete from state
                if file_id in self.index_state:
                    del self.index_state[file_id]
                
                logger.info(f"✅ Removed document {file_id} from index")
                return True
        except Exception as e:
            logger.error(f"❌ Error removing document {file_id}: {e}")
        return False

    def _setup_query_engine(self):
        """Setup the query engine with optimized settings for BGE embeddings"""
        if not self.index:
            return
            
        # Create retriever with higher top-k for better recall
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.TOP_K_RETRIEVAL * 2,  # Retrieve more candidates
        )
        
        # Use a lower similarity cutoff for BGE embeddings (they have different score ranges)
        # BGE embeddings often have lower similarity scores, so we need a more permissive threshold
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)  # Lowered from 0.5
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
        )
        
        print(f"✅ Query engine configured with similarity_cutoff=0.3, top_k={settings.TOP_K_RETRIEVAL * 2}")

    def _create_custom_engine(self, system_prompt: str):
        """Create query engine with custom DEVCON prompt"""
        from llama_index.core import PromptTemplate
        from llama_index.core.response_synthesizers import get_response_synthesizer
        
        # Create the QA template with proper placeholder syntax
        qa_template_str = (
            f"{system_prompt}\n\n"
            "Context information is below:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query as a DEVCON chapter officer advisor.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        
        # Create the PromptTemplate object
        qa_template = PromptTemplate(qa_template_str)
        
        # Create response synthesizer with custom template
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
            response_mode="compact"  # Add response mode for better control
        )
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.TOP_K_RETRIEVAL * 2,
        )
        
        # Create postprocessor with relaxed threshold
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.25)  # Even more relaxed
        
        # Return custom query engine
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor],
        )

    def _create_anti_hallucination_engine(self, question: str):
        """Create query engine specifically designed to prevent hallucinations"""
        from llama_index.core import PromptTemplate
        from llama_index.core.response_synthesizers import get_response_synthesizer
        
        # First retrieve relevant nodes
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.TOP_K_RETRIEVAL * 2,
        )
        
        nodes = retriever.retrieve(question)
        
        if not nodes:
            return None, "No relevant documents found for your question."
        
        # Create context-aware prompt using the prompt engine
        context_prompt = self.prompt_engine.create_context_aware_prompt(nodes, question)
        
        # Create a more flexible QA template with emojis
        qa_template_str = """🤖 You are the DEVCON Officers' Onboarding Assistant! 

Below is some context retrieved from documents. If it's helpful to answer the question or query, feel free to use it. Otherwise, ignore it and provide your best guidance as a DEVCON advisor! 📖✨

Context:
{context_str}

Question: {query_str}

💡 Remember to:
- Use emojis to make responses engaging 😊
- Reference context naturally when it's helpful
- Be practical and actionable for chapter officers 🎯
- If context isn't relevant, provide general DEVCON guidance instead!

Answer:"""
        
        qa_template = PromptTemplate(qa_template_str)
        
        # Create response synthesizer with strict template
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
            response_mode="compact"
        )
        
        # Create postprocessor with higher threshold to ensure relevance
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.4)
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor],
        ), None

    def query_with_verification(self, question: str) -> str:
        """Query with built-in hallucination verification"""
        try:
            if not self.query_engine:
                return "❌ RAG system not initialized. Please add documents first."
            
            # Enhanced query processing
            enhanced_question = self.prompt_engine.enhance_query(question)
            print(f"🔍 Enhanced query: {enhanced_question}")

            # Create anti-hallucination engine
            custom_engine, error = self._create_anti_hallucination_engine(enhanced_question)
            
            if error:
                return f"❌ {error}"
            
            # Query with verification-focused engine
            print("🚀 Querying with anti-hallucination safeguards...")
            response = custom_engine.query(enhanced_question)
            
            # Verify response quality
            if not response.response or response.response.strip() in ["Empty Response", "", "I don't know"]:
                return "❌ I don't have enough relevant information in the available documents to answer your question accurately."
            
            # Check for low-confidence responses
            if any(phrase in response.response.lower() for phrase in [
                "i don't have that information",
                "not mentioned in the context",
                "the provided documents don't contain"
            ]):
                # This is actually good - the model is being honest about limitations
                pass
            
            # Format response with source attribution
            answer = f"{response.response}\n\n"
            
            # Add detailed source information
            if hasattr(response, 'source_nodes') and response.source_nodes:
                answer += "📚 **Sources Referenced:**\n"
                for i, node in enumerate(response.source_nodes[:3], 1):
                    source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
                    file_id = node.metadata.get('file_id')
                    score = getattr(node, 'score', 0)
                    
                    answer += f"{i}. {source_info})\n"
                    
                    if file_id:
                        answer += f"   [[View Document]](https://drive.google.com/file/d/{file_id}/view)\n"
            
            return answer
            
        except Exception as e:
            print(f"❌ Error in verified query: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ I encountered an error while processing your question. Please try rephrasing or contact support."

    def _calculate_response_consistency(self, responses: list) -> float:
        """Calculate consistency score between responses"""
        if len(responses) < 2:
            return 1.0
        
        # Simple similarity based on shared key phrases
        # You could use more sophisticated NLP similarity measures
        consistency_scores = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # Count shared words (simplified)
                words1 = set(responses[i].lower().split())
                words2 = set(responses[j].lower().split())
                
                if len(words1) + len(words2) == 0:
                    continue
                
                jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
                consistency_scores.append(jaccard_sim)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

    # Replace the existing query method with this safer version
    def query(self, question: str) -> str:
        """Main query method with anti-hallucination measures"""
        return self.query_with_verification(question)


    def test_retrieval_scores(self, test_query: str = "DEVCON") -> dict:
        """Test retrieval scores to help tune similarity thresholds"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10,
            )
            
            nodes = retriever.retrieve(test_query)
            
            if not nodes:
                return {"error": "No nodes retrieved"}
            
            scores = [node.score for node in nodes]
            
            return {
                "query": test_query,
                "num_results": len(nodes),
                "score_stats": {
                    "min": min(scores),
                    "max": max(scores),
                    "mean": sum(scores) / len(scores),
                    "all_scores": scores
                },
                "sample_results": [
                    {
                        "score": node.score,
                        "text_preview": node.text[:100] + "...",
                        "source": node.metadata.get('source', 'Unknown')
                    }
                    for node in nodes[:3]
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def persist_index(self):
        """Persist the index to disk"""
        try:
            if self.index:
                storage_dir = Path(settings.RAG_INDEX_DIR)
                storage_dir.mkdir(parents=True, exist_ok=True)
                self.index.storage_context.persist(persist_dir=str(storage_dir))
                print(f"✅ Index persisted to {storage_dir}")
        except Exception as e:
            print(f"❌ Error persisting index: {e}")
    
    def get_index_stats(self) -> dict:
        """Get statistics about the current index"""
        try:
            if not self.index:
                return {"status": "not_initialized", "document_count": 0}
            
            # Get document count from vector store
            collection_count = len(self.vector_store._collection.get()['ids'])
            
            return {
                "status": "ready",
                "document_count": collection_count,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "top_k_retrieval": settings.TOP_K_RETRIEVAL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_api": "Hugging Face InferenceClient",
                "embedding_dimensions": 1024 if "bge-large" in settings.EMBEDDING_MODEL else 768,
                "normalize_embeddings": settings.BGE_NORMALIZE_EMBEDDINGS,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def test_embedding(self, text: str = "This is a test sentence.") -> dict:
        """Test the embedding model"""
        try:
            embed_model = LlamaSettings.embed_model
            embedding = embed_model._get_text_embedding(text)
            
            return {
                "status": "success",
                "text": text,
                "embedding_dim": len(embedding),
                "embedding_sample": embedding[:5],  # First 5 dimensions
                "model_name": embed_model.model_name,
                "api_type": "Hugging Face InferenceClient",
                "normalized": embed_model.normalize_embeddings
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance
rag_service = LlamaIndexRAGService()