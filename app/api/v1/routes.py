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
from app.core.config import settings
import os

logger = logging.getLogger(__name__)
router = APIRouter()


class IndexStatsResponse(BaseModel):
    status: str
    document_count: int
    chunk_size: int = None
    chunk_overlap: int = None
    top_k_retrieval: int = None
    error: str = None


class FetchFilesRequest(BaseModel):
    folder_id: str = None  # Optional: specific folder to fetch from


class DriveFileInfo(BaseModel):
    id: str
    name: str
    mime_type: str
    size: int = None
    modified_time: str = None
    status: str = "available"  # available, processed, failed


class FetchFilesResponse(BaseModel):
    status: str
    message: str
    total_files: int = 0
    supported_files: int = 0
    files: List[DriveFileInfo] = []
    processing_time: float = 0


class RebuildRequest(BaseModel):
    folder_id: str = None  # Optional: specific folder to rebuild from
    batch_size: int = 25  # Batch size for processing
    file_ids: List[str] = None  # Optional: specific files to process


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


@router.post("/files/fetch", response_model=FetchFilesResponse)
async def fetch_drive_files(request: FetchFilesRequest):
    """Fetch and list files from Google Drive without processing them
    
    This endpoint will:
    1. Connect to Google Drive
    2. List all files in the specified folder
    3. Return file information without processing or indexing
    
    Use this to see what files are available before deciding to rebuild the index.
    """
    try:
        logger.info("📁 Fetching files from Google Drive...")
        start_time = time.time()
        
        # Use provided folder_id or default
        folder_id = request.folder_id or "1eocL8T8BH6EwnP5siOtDz3FG2CqGHveS"
        
        # Get files from Google Drive
        files = get_drive_files(folder_id)
        total_files = len(files)
        
        if not files:
            return FetchFilesResponse(
                status="success",
                message="No files found in the specified folder",
                total_files=0,
                supported_files=0,
                files=[],
                processing_time=time.time() - start_time
            )
        
        # Convert to response format
        drive_files = []
        for file in files:
            drive_file = DriveFileInfo(
                id=file.get("id", ""),
                name=file.get("name", "Unknown"),
                mime_type=file.get("mimeType", "Unknown"),
                size=int(file.get("size", 0)) if file.get("size") else None,
                modified_time=file.get("modifiedTime", ""),
                status="available"
            )
            drive_files.append(drive_file)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Successfully fetched {total_files} files in {processing_time:.2f} seconds")
        
        return FetchFilesResponse(
            status="success",
            message=f"Successfully fetched {total_files} files from Google Drive",
            total_files=total_files,
            supported_files=total_files,  # All returned files are supported types
            files=drive_files,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error fetching files from Google Drive: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")


@router.get("/files/list")
async def list_drive_files(folder_id: str = None):
    """Simple endpoint to list files (alias for fetch with GET method)"""
    request = FetchFilesRequest(folder_id=folder_id)
    return await fetch_drive_files(request)


@router.get("/rebuild/status")
async def get_rebuild_status():
    """Get current rebuild status"""
    return rebuild_state


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest):
    """Rebuild the RAG index from Google Drive files
    
    This endpoint will:
    1. Fetch files from Google Drive (or use specific file_ids if provided)
    2. Process them into documents
    3. Build a new index with fresh embeddings
    
Note: Use /files/fetch first to see available files before rebuilding.
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
        logger.info("🔄 Starting index rebuild...")
        start_time = time.time()
        
        # Use provided folder_id or default
        folder_id = request.folder_id or "1eocL8T8BH6EwnP5siOtDz3FG2CqGHveS"
        
        # Step 1: Get files from Google Drive
        logger.info("📁 Fetching files from Google Drive...")
        all_files = get_drive_files(folder_id)
        
        # Filter files if specific file_ids are provided
        if request.file_ids:
            files = [f for f in all_files if f.get("id") in request.file_ids]
            logger.info(f"Filtered to {len(files)} specific files out of {len(all_files)} total files")
        else:
            files = all_files
            logger.info(f"Processing all {len(files)} files")
        
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
        
        # Step 3: Rebuild index with processed documents
        logger.info(f"🚀 Rebuilding RAG index with {len(documents)} documents...")
        
        success = rag_service.rebuild_index_from_scratch(documents)
        
        processing_time = time.time() - start_time
        
        if success:
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
            logger.error("Failed to rebuild index")
            rebuild_state["last_status"] = "failed_rebuild"
            return RebuildResponse(
                status="error",
                message="Failed to rebuild index",
                processed_files=0,
                failed_files=len(files),
                processing_time=processing_time,
                files_details=files_details
            )
    
    except Exception as e:
        logger.error(f"Error during index rebuild: {e}")
        rebuild_state["last_status"] = f"error: {str(e)}"
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