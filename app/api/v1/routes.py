# backend/app/api/v1/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.api.v1.models import AskRequest, AskResponse, RefreshStatus, FileProcessingDetail
from app.services.llamaindex_service import rag_service
from app.services.document_processor import document_processor
from app.services.drive_loader import get_drive_files
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import threading
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)
router = APIRouter()


class IndexStatsResponse(BaseModel):
    status: str
    document_count: int
    chunk_size: int = None
    chunk_overlap: int = None
    top_k_retrieval: int = None
    error: str = None


class RebuildRequest(BaseModel):
    folder_id: str = None  # Optional: specific folder to rebuild from
    batch_size: int = 25  # Batch size for processing


class RebuildResponse(BaseModel):
    status: str
    message: str
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    processing_time: float = 0
    files_details: List[Dict[str, Any]] = []


rebuild_state = {
    "in_progress": False,
    "last_started": None,
    "last_completed": None,
    "last_duration": None,
    "last_status": None
}

rebuild_lock = threading.Lock()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question using the RAG system with conversation history"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received query: {request.query}")
        logger.info(f"History length: {len(request.history)}")
        
        # Convert history to format expected by RAG service
        history_context = []
        for msg in request.history:
            history_context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Query with history context
        response = rag_service.query(request.query, history_context)
        
        return AskResponse(answer=response)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get statistics about the current RAG index"""
    try:
        stats = rag_service.get_index_stats()
        return IndexStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return IndexStatsResponse(
            status="error",
            document_count=0,
            error=str(e)
        )


@router.get("/rebuild/status")
async def get_rebuild_status():
    """Get current rebuild status"""
    return rebuild_state


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest):
    """Completely rebuild the RAG index from scratch (synchronous)
    
    This endpoint will:
    1. Fetch all files from Google Drive
    2. Process them into documents
    3. Build a new index with all documents
    4. Only clear the old index after successful rebuild
    
    Designed to run every 2 hours for complete index refresh.
    """
    with rebuild_lock:
        if rebuild_state["in_progress"]:
            return RebuildResponse(
                status="already_running",
                message="Rebuild is already in progress",
            )
        
        rebuild_state["in_progress"] = True
        rebuild_state["last_started"] = str(datetime.now())
        rebuild_state["last_completed"] = None
        rebuild_state["last_duration"] = None
        rebuild_state["last_status"] = "in_progress"

    try:
        logger.info("🔄 Starting index rebuild (keeping current index until completion)...")
        start_time = time.time()
        
        if request.folder_id is None:
            request.folder_id = "1eocL8T8BH6EwnP5siOtDz3FG2CqGHveS"
        
        # Step 1: Get files from Google Drive
        logger.info("📁 Fetching files from Google Drive...")
        files = get_drive_files(request.folder_id)
        logger.info(f"Found {len(files)} files to process")
        
        if not files:
            rebuild_state["last_status"] = "completed_no_files"
            return RebuildResponse(
                status="completed",
                message="No files found to process",
                processed_files=0,
                failed_files=0,
                processing_time=time.time() - start_time
            )
        
        # Step 2: Process files into LlamaIndex Documents
        logger.info("🔧 Processing documents...")
        documents = document_processor.process_drive_files(files)
        processed_count = len(documents)
        failed_count = len(files) - processed_count
        
        # Calculate total chunks (rough estimate)
        total_chunks = sum(len(doc.text) // 512 + 1 for doc in documents)
        
        # Prepare file details for response
        files_details = []
        for i, file in enumerate(files):
            status = "processed" if i < processed_count else "failed"
            chunk_count = len(documents[i].text) // 512 + 1 if i < processed_count else 0
            
            files_details.append({
                "name": file.get("name", "Unknown"),
                "id": file.get("id", "Unknown"),
                "status": status,
                "mime_type": file.get("mimeType", "Unknown"),
                "chunk_count": chunk_count
            })
        
        if not documents:
            rebuild_state["last_status"] = "completed_no_documents"
            return RebuildResponse(
                status="completed",
                message="No documents were successfully processed",
                processed_files=0,
                failed_files=len(files),
                processing_time=time.time() - start_time,
                files_details=files_details
            )
        
        # Step 3: Build new index with batch processing
        logger.info(f"🚀 Building new RAG index with {len(documents)} documents...")
        logger.info(f"Using batch size: {request.batch_size}")
        
        # Create a new temporary index or use the service's rebuild functionality
        # This depends on how your rag_service is implemented
        success = True
        
        if len(documents) > request.batch_size:
            # Process in batches
            total_batches = (len(documents) + request.batch_size - 1) // request.batch_size
            logger.info(f"Processing in {total_batches} batches")
            
            # Start building new index (this should create a new index without clearing the old one)
            new_index_success = rag_service.start_new_index_build()
            if not new_index_success:
                logger.error("Failed to start new index build")
                success = False
            else:
                for i in range(0, len(documents), request.batch_size):
                    batch = documents[i:i + request.batch_size]
                    batch_num = (i // request.batch_size) + 1
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    batch_success = rag_service.add_documents_to_new_index(batch)
                    if not batch_success:
                        logger.error(f"Failed to process batch {batch_num}")
                        success = False
                        break
                    
                    # Brief pause between batches for API rate limiting
                    if batch_num < total_batches:
                        time.sleep(1)
        else:
            # Process all at once
            new_index_success = rag_service.start_new_index_build()
            if new_index_success:
                success = rag_service.add_documents_to_new_index(documents)
            else:
                success = False
        
        processing_time = time.time() - start_time
        
        if success:
            # Step 4: Only now replace the old index with the new one
            logger.info("🔄 Replacing old index with newly built index...")
            replace_success = rag_service.replace_index_with_new()
            
            if replace_success:
                logger.info(f"✅ Index rebuild completed successfully in {processing_time:.2f} seconds")
                rebuild_state["last_status"] = "completed_success"
                return RebuildResponse(
                    status="completed",
                    message=f"Successfully rebuilt index with {processed_count} documents",
                    processed_files=processed_count,
                    failed_files=failed_count,
                    total_chunks=total_chunks,
                    processing_time=processing_time,
                    files_details=files_details
                )
            else:
                logger.error("Failed to replace old index with new index")
                rebuild_state["last_status"] = "failed_index_replacement"
                return RebuildResponse(
                    status="error",
                    message="Failed to replace old index with newly built index",
                    processed_files=processed_count,
                    failed_files=failed_count,
                    total_chunks=total_chunks,
                    processing_time=processing_time,
                    files_details=files_details
                )
        else:
            logger.error("Failed to build new index with documents")
            rebuild_state["last_status"] = "failed_build"
            # Clean up the failed new index attempt
            if hasattr(rag_service, 'cleanup_failed_index_build'):
                rag_service.cleanup_failed_index_build()
            
            return RebuildResponse(
                status="error",
                message="Failed to build new index with documents",
                processed_files=0,
                failed_files=len(files),
                processing_time=processing_time,
                files_details=files_details
            )
    
    except Exception as e:
        logger.error(f"Error during index rebuild: {e}")
        rebuild_state["last_status"] = f"error: {str(e)}"
        
        # Clean up any partial new index build
        try:
            if hasattr(rag_service, 'cleanup_failed_index_build'):
                rag_service.cleanup_failed_index_build()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Error during rebuild: {str(e)}")
    
    finally:
        duration = time.time() - start_time
        with rebuild_lock:
            rebuild_state["in_progress"] = False
            rebuild_state["last_completed"] = str(datetime.now())
            rebuild_state["last_duration"] = duration


@router.delete("/index")
async def clear_index():
    """Clear the RAG index (for testing purposes only)"""
    try:
        # Handle missing clear_index method gracefully
        if hasattr(rag_service, 'clear_index'):
            success = rag_service.clear_index()
            if success:
                return {"status": "success", "message": "Index cleared successfully"}
            else:
                return {"status": "error", "message": "Failed to clear index"}
        else:
            return {"status": "error", "message": "Clear index method not available in RAG service"}
    
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing index: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = rag_service.get_index_stats()
        return {
            "status": "healthy",
            "rag_status": stats.get("status", "unknown"),
            "document_count": stats.get("document_count", 0),
            "rebuild_status": rebuild_state["last_status"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "rebuild_status": rebuild_state.get("last_status", "unknown")
        }