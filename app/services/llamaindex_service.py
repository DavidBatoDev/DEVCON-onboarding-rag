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
from typing import List, Optional, Dict
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

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history into a readable context string"""
        if not history:
            return ""
        
        formatted_history = []
        for msg in history[-5:]:  # Only use last 5 exchanges to avoid token limits
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)

    def _create_history_aware_prompt(self, question: str, history: List[Dict[str, str]]) -> str:
        """Create a context-aware prompt that considers conversation history"""
        if not history:
            return question
        
        # Format recent conversation
        conversation_context = self._format_conversation_history(history)
        
        # Create enhanced prompt that considers context
        enhanced_prompt = f"""
    Previous conversation context:
    {conversation_context}

    Current question: {question}

    Please provide a helpful response considering the conversation history above. 
    If the current question refers to something mentioned earlier, use that context.
    If it's a new topic, focus on the current question.
    """
        
        return enhanced_prompt

    def _create_history_aware_anti_hallucination_engine(self, question: str, history: List[Dict[str, str]] = None):
        """Create query engine that considers conversation history AND prevents hallucinations"""
        from llama_index.core import PromptTemplate
        from llama_index.core.response_synthesizers import get_response_synthesizer
        
        # Format conversation history
        conversation_context = self._format_conversation_history(history) if history else ""
        
        # Enhanced query with history context
        enhanced_query = self._create_history_aware_prompt(question, history) if history else question
        
        # First retrieve relevant nodes using enhanced query
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.TOP_K_RETRIEVAL * 2,
        )
        
        nodes = retriever.retrieve(enhanced_query)
        
        if not nodes:
            return None, "No relevant documents found for your question.", enhanced_query
        
        # Create context-aware prompt using the prompt engine if available
        if hasattr(self, 'prompt_engine') and self.prompt_engine:
            context_prompt = self.prompt_engine.create_context_aware_prompt(nodes, question)
        
        # Updated QA template with reference decision logic
        qa_template_str = """🤖 You are the DEVCON Officers' Onboarding Assistant! 

    {conversation_history}

    Below is some context retrieved from documents. If it's helpful to answer the current question, use it. If the context isn't directly relevant, provide your best guidance as a DEVCON advisor while being honest about what information is available.

    Context from documents:
    {context_str}

    Current question: {query_str}

    💡 Instructions:
    - Consider the conversation history when answering
    - If the question refers to something mentioned earlier, acknowledge that context
    - Use emojis to make responses engaging 😊
    - Be practical and actionable for chapter officers 🎯
    - If context from documents isn't relevant, provide general DEVCON guidance
    - Be honest if you don't have specific information - don't make things up!
    - If the current question is related to previous ones, provide a cohesive response

    🔍 Reference Decision:
    At the end of your response, add EXACTLY ONE of these markers:
    - Add "SHOW_REFERENCES: true" if your answer is based on or references specific information from the provided documents
    - Add "SHOW_REFERENCES: false" if your answer is general guidance not specifically from the documents

    Answer:"""
        
        # Insert conversation history into template
        if conversation_context:
            formatted_template = qa_template_str.replace(
                "{conversation_history}", 
                f"Recent conversation:\n{conversation_context}\n"
            )
        else:
            formatted_template = qa_template_str.replace("{conversation_history}\n", "")
        
        qa_template = PromptTemplate(formatted_template)
        
        # Create response synthesizer with history-aware template
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_template,
            response_mode="compact"
        )
        
        # Create postprocessor with balanced threshold for relevance
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor],
        ), None, enhanced_query


    def query_with_history_and_verification(self, question: str, history: List[Dict[str, str]] = None) -> str:
        """Main query method that handles conversation history AND prevents hallucinations"""
        try:
            if not self.query_engine:
                return "❌ RAG system not initialized. Please add documents first."
            
            if history is None:
                history = []
            
            print(f"🔍 Processing query with {len(history)} history messages")
            
            # Enhanced query processing with history context
            if history:
                enhanced_question = self._create_history_aware_prompt(question, history)
            else:
                # Use prompt engine if available for single queries
                if hasattr(self, 'prompt_engine') and self.prompt_engine:
                    enhanced_question = self.prompt_engine.enhance_query(question)
                else:
                    enhanced_question = question
            
            print(f"🔍 Enhanced query: {enhanced_question}")

            # Create history-aware anti-hallucination engine
            custom_engine, error, final_query = self._create_history_aware_anti_hallucination_engine(question, history)
            
            if error:
                return f"❌ {error}"
            
            # Query with both history context and verification safeguards
            print("🚀 Querying with conversation history context and anti-hallucination safeguards...")
            response = custom_engine.query(final_query)
            
            # Verify response quality
            if not response.response or response.response.strip() in ["Empty Response", "", "I don't know"]:
                return "❌ I don't have enough relevant information in the available documents to answer your question accurately."
            
            # Check for low-confidence responses (this is actually good - shows honesty)
            if any(phrase in response.response.lower() for phrase in [
                "i don't have that information",
                "not mentioned in the context",
                "the provided documents don't contain"
            ]):
                # Model is being appropriately cautious - this is good behavior
                pass
            
            # Parse the show_reference decision from the response
            response_text = response.response
            show_references = False
            
            # Look for the reference decision marker
            if "SHOW_REFERENCES: true" in response_text:
                show_references = True
                # Remove the marker from the response
                response_text = response_text.replace("SHOW_REFERENCES: true", "").strip()
            elif "SHOW_REFERENCES: false" in response_text:
                show_references = False
                # Remove the marker from the response
                response_text = response_text.replace("SHOW_REFERENCES: false", "").strip()
            else:
                # Fallback: if no marker found, default to showing references if we have source nodes
                show_references = hasattr(response, 'source_nodes') and response.source_nodes
            
            # Start building the final answer
            answer = f"{response_text}\n\n"
            
            # Only add source information if the LLM decided to show references
            if show_references and hasattr(response, 'source_nodes') and response.source_nodes:
                answer += "📚 **Sources Referenced:**\n"
                for i, node in enumerate(response.source_nodes[:3], 1):
                    source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
                    file_id = node.metadata.get('file_id')
                    
                    answer += f"{i}. {source_info}\n"
                    
                    if file_id:
                        answer += f"   [[View Document]](https://drive.google.com/file/d/{file_id}/view)\n"
            
            print(f"🔍 Reference display decision: {show_references}")
            return answer
            
        except Exception as e:
            print(f"❌ Error in history-aware verified query: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ I encountered an error while processing your question. Please try rephrasing or contact support."

    # 3. Optional: Add a method to explicitly control reference display
    def query_with_reference_control(self, question: str, history: List[Dict[str, str]] = None, force_show_references: bool = None) -> str:
        """Query method with explicit reference display control"""
        try:
            # Get the normal response
            response = self.query_with_history_and_verification(question, history)
            
            # If force_show_references is specified, override the LLM's decision
            if force_show_references is not None:
                # Remove any existing source section
                if "📚 **Sources Referenced:**" in response:
                    response = response.split("📚 **Sources Referenced:**")[0].strip() + "\n\n"
                
                # Add sources if forced to show
                if force_show_references:
                    # Re-run query to get source nodes
                    custom_engine, error, final_query = self._create_history_aware_anti_hallucination_engine(question, history)
                    if not error:
                        query_response = custom_engine.query(final_query)
                        if hasattr(query_response, 'source_nodes') and query_response.source_nodes:
                            response += "📚 **Sources Referenced:**\n"
                            for i, node in enumerate(query_response.source_nodes[:3], 1):
                                source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
                                file_id = node.metadata.get('file_id')
                                
                                response += f"{i}. {source_info}\n"
                                
                                if file_id:
                                    response += f"   [[View Document]](https://drive.google.com/file/d/{file_id}/view)\n"
            
            return response
            
        except Exception as e:
            print(f"❌ Error in reference-controlled query: {e}")
            return f"❌ I encountered an error while processing your question. Please try rephrasing or contact support."

    # Update the main query method to use the merged functionality
    def query(self, question: str, history: List[Dict[str, str]] = None) -> str:
        """Main query method that handles both history and verification"""
        return self.query_with_history_and_verification(question, history)

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