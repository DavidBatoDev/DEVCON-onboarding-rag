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
        
        # Get or create collection with NO embedding function
        collection_name = "devcon_documents_hf_client"  # Updated name for HF InferenceClient
        try:
            chroma_collection = chroma_client.get_collection(collection_name)
            print(f"✅ Loaded existing ChromaDB collection: {collection_name}")
        except:
            # Create collection with NO embedding function - we'll provide pre-computed embeddings
            chroma_collection = chroma_client.create_collection(
                collection_name, 
                embedding_function=None
            )
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
        

    def start_new_index_build(self) -> bool:
        """Start building a new index without affecting the current one"""
        try:
            # Create a new ChromaDB collection for the new index
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Use a temporary collection name
            temp_collection_name = "devcon_documents_hf_client_temp"
            
            # Delete existing temp collection if it exists
            try:
                chroma_client.delete_collection(temp_collection_name)
                logger.info(f"Deleted existing temp collection: {temp_collection_name}")
            except Exception:
                pass  # Collection doesn't exist, which is fine
            
            # Create new temporary collection with NO embedding function
            # This is important - we'll provide pre-computed embeddings
            self.temp_chroma_collection = chroma_client.create_collection(
                temp_collection_name,
                embedding_function=None
            )
            self.temp_vector_store = ChromaVectorStore(chroma_collection=self.temp_chroma_collection)
            
            # Create new temporary index
            temp_storage_context = StorageContext.from_defaults(vector_store=self.temp_vector_store)
            self.temp_index = VectorStoreIndex([], storage_context=temp_storage_context)
            
            # Initialize temporary index state
            self.temp_index_state = {}
            
            logger.info("✅ Started new index build with temporary collection")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting new index build: {e}")
            import traceback
            traceback.print_exc()
            return False


    def add_documents_to_new_index(self, documents: List[Document]) -> bool:
        """Add documents to the new index being built"""
        try:
            if not hasattr(self, 'temp_index') or not self.temp_index:
                logger.error("❌ Temporary index not initialized. Call start_new_index_build() first.")
                return False
            
            # Sanitize metadata for all documents
            logger.info("🧹 Sanitizing document metadata for ChromaDB compatibility...")
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc.metadata = self._sanitize_metadata(doc.metadata)
            
            # Use sentence splitter for better chunking
            parser = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            # Parse documents into nodes
            nodes = parser.get_nodes_from_documents(documents)
            logger.info(f"📄 Created {len(nodes)} nodes from {len(documents)} documents")
            
            # Sanitize node metadata as well
            for node in nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    node.metadata = self._sanitize_metadata(node.metadata)
            
            # CRITICAL FIX: Explicitly generate embeddings for nodes before adding to temp index
            logger.info("🔄 Generating embeddings for nodes...")
            embed_model = LlamaSettings.embed_model
            
            # Process nodes in batches to avoid memory issues
            batch_size = 10
            for i in range(0, len(nodes), batch_size):
                batch_nodes = nodes[i:batch_size + i]
                logger.info(f"🔄 Processing embedding batch {i//batch_size + 1}/{(len(nodes) + batch_size - 1)//batch_size}")
                
                # Generate embeddings for this batch
                for node in batch_nodes:
                    if not hasattr(node, 'embedding') or node.embedding is None:
                        # Generate embedding for the node text
                        node.embedding = embed_model._get_text_embedding(node.text)
                        logger.debug(f"Generated embedding for node: {len(node.embedding)} dims")
            
            # Add to temporary index - now with embeddings
            logger.info("🔄 Adding nodes with embeddings to temporary index...")
            self.temp_index.insert_nodes(nodes)
            
            # Update temporary index state
            for doc in documents:
                file_id = doc.metadata.get("file_id")
                if file_id:
                    self.temp_index_state[file_id] = {
                        "file_id": file_id,
                        "file_name": doc.metadata.get("source", "Unknown"),
                        "title": doc.metadata.get("title", doc.metadata.get("source", "Unknown")),
                        "modified_time": doc.metadata.get("modified_time"),
                        "chunk_count": len([n for n in nodes if n.metadata.get("file_id") == file_id])
                    }
            
            logger.info(f"✅ Added {len(documents)} documents with embeddings to temporary index")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents to new index: {e}")
            import traceback
            traceback.print_exc()
            return False


    def replace_index_with_new(self) -> bool:
        """Replace the current index with the newly built one"""
        try:
            if not hasattr(self, 'temp_index') or not self.temp_index:
                logger.error("❌ No temporary index found to replace with")
                return False
            
            # Get ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Collection names
            current_collection_name = "devcon_documents_hf_client"
            temp_collection_name = "devcon_documents_hf_client_temp"
            backup_collection_name = "devcon_documents_hf_client_backup"
            
            # Step 1: Backup current collection (if it exists)
            backup_created = False
            try:
                current_collection = chroma_client.get_collection(current_collection_name)
                # Delete existing backup if it exists
                try:
                    chroma_client.delete_collection(backup_collection_name)
                except Exception:
                    pass
                
                # Get all data from current collection
                current_data = current_collection.get()
                if current_data['ids']:
                    # Create backup collection with NO embedding function - we'll provide embeddings
                    backup_collection = chroma_client.create_collection(
                        backup_collection_name, 
                        embedding_function=None
                    )
                    
                    # Check if data exists (avoid ambiguous truth value error)
                    has_current_embeddings = current_data.get('embeddings') is not None and len(current_data.get('embeddings', [])) > 0
                    has_current_metadatas = current_data.get('metadatas') is not None and len(current_data.get('metadatas', [])) > 0
                    has_current_documents = current_data.get('documents') is not None and len(current_data.get('documents', [])) > 0
                    
                    # Add data to backup collection in batches to avoid memory issues
                    batch_size = 100
                    for i in range(0, len(current_data['ids']), batch_size):
                        end_idx = min(i + batch_size, len(current_data['ids']))
                        
                        backup_collection.add(
                            ids=current_data['ids'][i:end_idx],
                            embeddings=current_data['embeddings'][i:end_idx] if has_current_embeddings else None,
                            metadatas=current_data['metadatas'][i:end_idx] if has_current_metadatas else None,
                            documents=current_data['documents'][i:end_idx] if has_current_documents else None
                        )
                    
                    backup_created = True
                    logger.info(f"✅ Backed up current collection with {len(current_data['ids'])} documents")
            
            except Exception as e:
                logger.warning(f"⚠️ Could not backup current collection: {e}")
                # Continue anyway - the temp index should still be valid
            
            # Step 2: FIXED - Get temp collection data and verify embeddings exist
            try:
                temp_collection = chroma_client.get_collection(temp_collection_name)
                temp_data = temp_collection.get(include=['embeddings', 'metadatas', 'documents'])
                
                if not temp_data['ids']:
                    logger.error("❌ Temporary collection is empty")
                    return False
                
                # CRITICAL FIX: Check embeddings properly
                embeddings_exist = False
                if temp_data.get('embeddings') is not None:
                    # Check if we have embeddings and they're not empty
                    if len(temp_data['embeddings']) > 0:
                        # Check if the first embedding is not None/empty
                        first_embedding = temp_data['embeddings'][0]
                        if first_embedding is not None and len(first_embedding) > 0:
                            embeddings_exist = True
                
                if not embeddings_exist:
                    logger.error("❌ Temporary collection has no embeddings!")
                    logger.info("🔄 Attempting to regenerate embeddings from temp collection...")
                    
                    # Try to regenerate embeddings from the documents in temp collection
                    if temp_data.get('documents') and len(temp_data['documents']) > 0:
                        embed_model = LlamaSettings.embed_model
                        regenerated_embeddings = []
                        
                        logger.info(f"🔄 Regenerating embeddings for {len(temp_data['documents'])} documents...")
                        for idx, doc_text in enumerate(temp_data['documents']):
                            if doc_text and doc_text.strip():
                                try:
                                    embedding = embed_model._get_text_embedding(doc_text)
                                    regenerated_embeddings.append(embedding)
                                    if (idx + 1) % 10 == 0:
                                        logger.info(f"🔄 Regenerated {idx + 1}/{len(temp_data['documents'])} embeddings")
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to generate embedding for document {idx}: {e}")
                                    # Use zero embedding as fallback
                                    regenerated_embeddings.append([0.0] * 1024)
                            else:
                                logger.warning(f"⚠️ Found empty document text at index {idx}, using zero embedding")
                                regenerated_embeddings.append([0.0] * 1024)
                        
                        temp_data['embeddings'] = regenerated_embeddings
                        logger.info(f"✅ Regenerated {len(regenerated_embeddings)} embeddings")
                    else:
                        logger.error("❌ No documents found in temp collection to regenerate embeddings")
                        return False
                
                logger.info(f"✅ Temp collection has {len(temp_data['ids'])} documents with embeddings")
                    
            except Exception as e:
                logger.error(f"❌ Could not retrieve temp collection data: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Step 3: Delete current collection
            try:
                chroma_client.delete_collection(current_collection_name)
                logger.info(f"✅ Deleted current collection: {current_collection_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete current collection: {e}")
            
            # Step 4: Create new current collection and copy data with embeddings
            try:
                # Create new current collection with NO embedding function
                new_current_collection = chroma_client.create_collection(
                    current_collection_name, 
                    embedding_function=None
                )
                
                # Copy data from temp to new current collection in batches
                batch_size = 100
                total_batches = (len(temp_data['ids']) + batch_size - 1) // batch_size
                
                # Check if embeddings exist (avoid ambiguous truth value error)
                has_embeddings = temp_data.get('embeddings') is not None and len(temp_data.get('embeddings', [])) > 0
                has_metadatas = temp_data.get('metadatas') is not None and len(temp_data.get('metadatas', [])) > 0
                has_documents = temp_data.get('documents') is not None and len(temp_data.get('documents', [])) > 0
                
                for i in range(0, len(temp_data['ids']), batch_size):
                    end_idx = min(i + batch_size, len(temp_data['ids']))
                    batch_num = i // batch_size + 1
                    
                    # Prepare batch data - avoid ambiguous truth value errors
                    batch_ids = temp_data['ids'][i:end_idx]
                    batch_embeddings = temp_data['embeddings'][i:end_idx] if has_embeddings else None
                    batch_metadatas = temp_data['metadatas'][i:end_idx] if has_metadatas else None
                    batch_documents = temp_data['documents'][i:end_idx] if has_documents else None
                    
                    # FIXED: Properly check embeddings without ambiguous truth value error
                    if batch_embeddings is None or len(batch_embeddings) == 0:
                        logger.error(f"❌ No embeddings found in temp collection batch {i}-{end_idx}")
                        raise Exception("Temp collection missing embeddings")
                    
                    # Verify embedding dimensions - check each embedding individually
                    for idx, embedding in enumerate(batch_embeddings):
                        if embedding is None:
                            logger.error(f"❌ None embedding at index {i + idx}")
                            raise Exception(f"None embedding at index {i + idx}")
                        
                        # Convert to list if it's a numpy array to avoid ambiguous truth value
                        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                        if not embedding_list or len(embedding_list) == 0:
                            logger.error(f"❌ Empty embedding at index {i + idx}")
                            raise Exception(f"Empty embedding at index {i + idx}")
                    
                    new_current_collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_documents
                    )
                    
                    logger.info(f"✅ Added batch {batch_num}/{total_batches} with {len(batch_ids)} documents")
                
                logger.info(f"✅ Created new current collection with {len(temp_data['ids'])} documents")
                
            except Exception as e:
                logger.error(f"❌ Error creating new current collection: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to restore from backup if we created one
                if backup_created:
                    logger.info("🔄 Attempting to restore from backup...")
                    try:
                        self._restore_from_backup()
                        logger.info("✅ Restored from backup after failed replacement")
                    except Exception as restore_error:
                        logger.error(f"❌ Failed to restore from backup: {restore_error}")
                
                return False
            
            # Step 5: Update service to use new collection
            try:
                self.vector_store = ChromaVectorStore(chroma_collection=new_current_collection)
                
                # Create new index with the new vector store
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex([], storage_context=storage_context)
                
                # Update index state
                self.index_state = self.temp_index_state.copy()
                self.save_index_state()
                
                # Setup query engine with new index
                self._setup_query_engine()
                
                # Persist the new index
                self.persist_index()
                
                logger.info("✅ Updated service to use new index")
                
            except Exception as e:
                logger.error(f"❌ Error updating service with new index: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Step 6: Cleanup temp collection
            try:
                chroma_client.delete_collection(temp_collection_name)
                logger.info(f"✅ Cleaned up temporary collection: {temp_collection_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete temp collection: {e}")
            
            # Step 7: Cleanup temp variables
            self.cleanup_temp_variables()
            
            logger.info("✅ Successfully replaced current index with new index")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error replacing index with new: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to restore from backup if possible
            try:
                self._restore_from_backup()
            except Exception as restore_error:
                logger.error(f"❌ Failed to restore from backup: {restore_error}")
            
            return False


    def cleanup_failed_index_build(self):
        """Clean up resources from a failed index build attempt"""
        try:
            logger.info("🧹 Cleaning up failed index build...")
            
            # Delete temporary collection if it exists
            if hasattr(self, 'temp_chroma_collection'):
                try:
                    chroma_client = chromadb.PersistentClient(
                        path=settings.CHROMA_PERSIST_DIR,
                        settings=ChromaSettings(anonymized_telemetry=False)
                    )
                    chroma_client.delete_collection("devcon_documents_hf_client_temp")
                    logger.info("✅ Deleted temporary collection")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete temp collection: {e}")
            
            # Cleanup temp variables
            self.cleanup_temp_variables()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def cleanup_temp_variables(self):
        """Clean up temporary variables used during index rebuild"""
        temp_vars = ['temp_index', 'temp_vector_store', 'temp_chroma_collection', 'temp_index_state']
        for var in temp_vars:
            if hasattr(self, var):
                delattr(self, var)
                logger.debug(f"Cleaned up temp variable: {var}")

    def _restore_from_backup(self):
        """Restore the index from backup collection (emergency recovery)"""
        try:
            logger.info("🔄 Attempting to restore from backup...")
            
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            backup_collection_name = "devcon_documents_hf_client_backup"
            current_collection_name = "devcon_documents_hf_client"
            
            # Check if backup exists
            try:
                backup_collection = chroma_client.get_collection(backup_collection_name)
                backup_data = backup_collection.get()
                
                if not backup_data['ids']:
                    logger.warning("⚠️ Backup collection is empty")
                    return False
                
                # Delete current collection if it exists
                try:
                    chroma_client.delete_collection(current_collection_name)
                except Exception:
                    pass
                
                # Create new current collection from backup with NO embedding function
                current_collection = chroma_client.create_collection(
                    current_collection_name, 
                    embedding_function=None
                )
                
                # Check if data exists (avoid ambiguous truth value error)
                has_backup_embeddings = backup_data.get('embeddings') is not None and len(backup_data.get('embeddings', [])) > 0
                has_backup_metadatas = backup_data.get('metadatas') is not None and len(backup_data.get('metadatas', [])) > 0
                has_backup_documents = backup_data.get('documents') is not None and len(backup_data.get('documents', [])) > 0
                
                # Add backup data in batches with explicit embeddings
                batch_size = 100
                for i in range(0, len(backup_data['ids']), batch_size):
                    end_idx = min(i + batch_size, len(backup_data['ids']))
                    
                    current_collection.add(
                        ids=backup_data['ids'][i:end_idx],
                        embeddings=backup_data['embeddings'][i:end_idx] if has_backup_embeddings else None,
                        metadatas=backup_data['metadatas'][i:end_idx] if has_backup_metadatas else None,
                        documents=backup_data['documents'][i:end_idx] if has_backup_documents else None
                    )
                
                # Update service to use restored collection
                self.vector_store = ChromaVectorStore(chroma_collection=current_collection)
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self.index = VectorStoreIndex([], storage_context=storage_context)
                self._setup_query_engine()
                
                logger.info(f"✅ Restored from backup with {len(backup_data['ids'])} documents")
                return True
                
            except Exception as e:
                logger.error(f"❌ Could not restore from backup: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in restore process: {e}")
            return False

    def get_rebuild_capabilities(self) -> dict:
        """Get information about rebuild capabilities and current state"""
        try:
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Check what collections exist
            collections = chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            # Check current index state
            current_stats = self.get_index_stats()
            
            return {
                "zero_downtime_rebuild": True,
                "current_index_status": current_stats.get("status", "unknown"),
                "current_document_count": current_stats.get("document_count", 0),
                "available_collections": collection_names,
                "has_backup": "devcon_documents_hf_client_backup" in collection_names,
                "temp_build_in_progress": hasattr(self, 'temp_index') and self.temp_index is not None,
                "supported_operations": [
                    "start_new_index_build",
                    "add_documents_to_new_index", 
                    "replace_index_with_new",
                    "cleanup_failed_index_build",
                    "restore_from_backup"
                ]
            }
            
        except Exception as e:
            return {
                "zero_downtime_rebuild": False,
                "error": str(e)
            }

    
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
        

    def clear_index(self) -> bool:
        """Completely clear the RAG index and reset its state"""
        try:
            # Clear the ChromaDB collection
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            collection_name = "devcon_documents_hf_client"
            try:
                chroma_client.delete_collection(collection_name)
                print(f"✅ Deleted ChromaDB collection: {collection_name}")
            except Exception as e:
                print(f"⚠️ Could not delete collection: {e}. Creating new one anyway.")
            
            # Clear the ChromaDB persistent storage directory completely
            chroma_persist_dir = Path(settings.CHROMA_PERSIST_DIR)
            if chroma_persist_dir.exists():
                import shutil
                try:
                    shutil.rmtree(chroma_persist_dir)
                    print(f"✅ Deleted ChromaDB storage directory: {chroma_persist_dir}")
                    # Recreate the directory
                    chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"⚠️ Could not delete ChromaDB storage directory: {e}")
                    # Try to delete individual collection folders
                    for item in chroma_persist_dir.iterdir():
                        if item.is_dir():
                            try:
                                shutil.rmtree(item)
                                print(f"✅ Deleted ChromaDB collection folder: {item.name}")
                            except Exception as folder_error:
                                print(f"⚠️ Could not delete folder {item.name}: {folder_error}")
            
            # Clear the index storage directory (LlamaIndex storage)
            storage_dir = Path(settings.RAG_INDEX_DIR)
            if storage_dir.exists():
                import shutil
                try:
                    shutil.rmtree(storage_dir)
                    print(f"✅ Deleted LlamaIndex storage at {storage_dir}")
                except Exception as e:
                    print(f"⚠️ Could not delete LlamaIndex storage: {e}")
            
            # Reset index state file
            self.index_state = {}
            if self.index_state_file.exists():
                try:
                    self.index_state_file.unlink()
                    print(f"✅ Deleted index state file: {self.index_state_file}")
                except Exception as e:
                    print(f"⚠️ Could not delete index state file: {e}")
            
            # Reinitialize ChromaDB client and collection
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            chroma_collection = chroma_client.create_collection(collection_name, data_loader=None, embedding_function=None)
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Reinitialize the index components
            self.index = None
            self._load_or_create_index()
            
            # Save fresh index state
            self.save_index_state()
            
            print("✅ Index completely cleared and reset - all storage files deleted")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global instance
rag_service = LlamaIndexRAGService()