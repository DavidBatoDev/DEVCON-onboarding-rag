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


class RefreshRequest(BaseModel):
    folder_id: str = None  # Optional: specific folder to refresh
    force_reprocess: bool = False  # Added this field


class RefreshResponse(BaseModel):
    status: str
    message: str
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    files_details: List[Dict[str, Any]] = []


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


refresh_state = {
    "in_progress": False,
    "last_started": None,
    "last_completed": None,
    "last_duration": None
}

rebuild_state = {
    "in_progress": False,
    "last_started": None,
    "last_completed": None,
    "last_duration": None
}

refresh_lock = threading.Lock()
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


@router.get("/refresh/status", response_model=RefreshStatus)
async def get_refresh_status():
    """Get current refresh status"""
    return refresh_state


@router.get("/rebuild/status")
async def get_rebuild_status():
    """Get current rebuild status"""
    return rebuild_state


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_index(request: RefreshRequest, background_tasks: BackgroundTasks):
    """Refresh the RAG index with latest documents from Google Drive (incremental)"""
    with refresh_lock:
        if refresh_state["in_progress"]:
            return RefreshResponse(
                status="already_running",
                message="Refresh is already in progress",
            )
        
        refresh_state["in_progress"] = True
        refresh_state["last_started"] = str(datetime.now())
        refresh_state["last_completed"] = None
        refresh_state["last_duration"] = None
    
    # Start background task
    background_tasks.add_task(
        _refresh_index_background,
        request.folder_id,
        request.force_reprocess
    )
    
    return RefreshResponse(
        status="started",
        message="Index refresh started in background",
    )


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest, background_tasks: BackgroundTasks):
    """Completely rebuild the RAG index from scratch (like compile script)"""
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
    
    # Start background task
    background_tasks.add_task(
        _rebuild_index_background,
        request.folder_id,
        request.batch_size
    )
    
    return RebuildResponse(
        status="started",
        message="Index rebuild started in background (complete re-embedding)",
    )


@router.post("/rebuild/sync", response_model=RebuildResponse)
async def rebuild_index_sync(request: RebuildRequest):
    """Synchronously rebuild the RAG index from scratch"""
    try:
        logger.info("🔄 Starting synchronous index rebuild...")
        start_time = time.time()
        
        # Clear existing index first
        logger.info("🗑️ Clearing existing index...")
        rag_service.clear_index()
        
        # Get files from Google Drive
        logger.info("📁 Fetching files from Google Drive...")
        files = get_drive_files(request.folder_id)
        logger.info(f"Found {len(files)} files to process")
        
        if not files:
            return RebuildResponse(
                status="completed",
                message="No files found to process",
                processed_files=0,
                failed_files=0,
                processing_time=time.time() - start_time
            )
        
        # Process files into LlamaIndex Documents
        logger.info("🔧 Processing documents...")
        documents = document_processor.process_drive_files(files)
        processed_count = len(documents)
        failed_count = len(files) - processed_count
        
        # Calculate total chunks
        total_chunks = sum(len(doc.text) // 512 + 1 for doc in documents)  # Rough estimate
        
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
        
        if documents:
            # Build RAG index with batch processing
            logger.info(f"🚀 Building RAG index with {len(documents)} documents...")
            logger.info(f"Using batch size: {request.batch_size}")
            
            if len(documents) > request.batch_size:
                # Process in batches
                total_batches = (len(documents) + request.batch_size - 1) // request.batch_size
                logger.info(f"Processing in {total_batches} batches")
                
                for i in range(0, len(documents), request.batch_size):
                    batch = documents[i:i + request.batch_size]
                    batch_num = (i // request.batch_size) + 1
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    success = rag_service.add_documents(batch)
                    if not success:
                        logger.error(f"Failed to process batch {batch_num}")
                        break
                    
                    # Brief pause between batches for API rate limiting
                    if batch_num < total_batches:
                        time.sleep(1)
            else:
                # Process all at once
                success = rag_service.add_documents(documents)
            
            processing_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ Index rebuild completed in {processing_time:.2f} seconds")
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
                return RebuildResponse(
                    status="error",
                    message="Failed to build index with documents",
                    processed_files=0,
                    failed_files=len(files),
                    processing_time=processing_time,
                    files_details=files_details
                )
        else:
            return RebuildResponse(
                status="completed",
                message="No documents were successfully processed",
                processed_files=0,
                failed_files=len(files),
                processing_time=time.time() - start_time,
                files_details=files_details
            )
    
    except Exception as e:
        logger.error(f"Error during synchronous index rebuild: {e}")
        raise HTTPException(status_code=500, detail=f"Error during rebuild: {str(e)}")


async def _refresh_index_background(folder_id: str = None, force_reprocess: bool = False):
    """Background task to refresh the index with incremental updates"""
    start_time = time.time()
    processed_files = 0
    failed_files = 0
    skipped_files = 0
    total_chunks = 0
    files_details = []
    
    try:
        # 1. Get current state from index
        current_index = rag_service.get_index_state()  # New method to implement
        
        # 2. Get files from Google Drive
        logger.info("🔍 Fetching files from Google Drive...")
        drive_files = get_drive_files(folder_id)
        logger.info(f"📁 Found {len(drive_files)} files in Google Drive")
        
        if not drive_files:
            logger.warning("No files found to process")
            return
        
        # 3. Identify changes
        files_to_process = []
        for file in drive_files:
            file_id = file["id"]
            modified_time = file["modifiedTime"]
            file_detail = FileProcessingDetail(
                name=file.get("name", "Unknown"),
                id=file_id,
                mime_type=file.get("mimeType", "Unknown"),
                status="pending",
            )
            
            # Check if we need to process this file
            if force_reprocess:
                file_detail.status = "queued"
                files_to_process.append(file)
            elif file_id in current_index:
                # File exists - check if modified
                if modified_time > current_index[file_id]["modified_time"]:
                    file_detail.status = "queued"
                    files_to_process.append(file)
                else:
                    file_detail.status = "skipped"
                    file_detail.error_message = "Unchanged since last processing"
                    skipped_files += 1
            else:
                # New file
                file_detail.status = "queued"
                print(f"New file detected: {file.get('name', 'Unknown')}")
                files_to_process.append(file)
            
            files_details.append(file_detail)
        
        logger.info(f"🔄 {len(files_to_process)} files to process "
                    f"({skipped_files} skipped, {len(drive_files) - len(files_to_process) - skipped_files} unchanged)")
        
        # 4. Process files
        if files_to_process:
            documents = document_processor.process_drive_files(files_to_process)
            processed_count = len(documents)
            failed_count = len(files_to_process) - processed_count
            
            # Update file details with processing results
            for detail in files_details:
                if detail.status == "queued":
                    # Find matching document
                    doc = next((d for d in documents if d.metadata["file_id"] == detail.id), None)
                    if doc:
                        detail.status = "processed"
                        detail.chunk_count = len(doc.metadata.get("chunks", []))
                        processed_files += 1
                        total_chunks += detail.chunk_count
                    else:
                        detail.status = "failed"
                        detail.error_message = "Processing failed"
                        failed_files += 1
        
        # 5. Identify deleted files
        current_file_ids = {f["id"] for f in drive_files}
        deleted_files = [fid for fid in current_index if fid not in current_file_ids]
        
        if deleted_files:
            logger.info(f"🗑️ Found {len(deleted_files)} deleted files to remove")
            for file_id in deleted_files:
                rag_service.remove_document(file_id)
        
        # 6. Add/update documents
        if documents:
            success = rag_service.add_documents(documents)
            if not success:
                logger.error("Failed to add documents to index")
        
        # 7. Persist the new state
        rag_service.save_index_state()
        
        logger.info(f"✅ Index refresh completed: "
                    f"{processed_files} processed, {failed_files} failed, "
                    f"{skipped_files} skipped, {len(deleted_files)} deleted")
    
    except Exception as e:
        logger.error(f"❌ Error during background index refresh: {e}")
        # Update error status for all queued files
        for detail in files_details:
            if detail.status == "queued":
                detail.status = "failed"
                detail.error_message = str(e)
                failed_files += 1
    
    finally:
        duration = time.time() - start_time
        with refresh_lock:
            refresh_state["in_progress"] = False
            refresh_state["last_completed"] = str(datetime.now())
            refresh_state["last_duration"] = duration


async def _rebuild_index_background(folder_id: str = None, batch_size: int = 25):
    """Background task to completely rebuild the index from scratch"""
    start_time = time.time()
    processed_files = 0
    failed_files = 0
    total_chunks = 0
    files_details = []
    
    try:
        logger.info("🔄 Starting background index rebuild...")
        
        # 1. Clear existing index
        logger.info("🗑️ Clearing existing index...")
        rag_service.clear_index()
        
        # 2. Get files from Google Drive
        logger.info("📁 Fetching files from Google Drive...")
        files = get_drive_files(folder_id)
        logger.info(f"Found {len(files)} files to process")
        
        if not files:
            logger.warning("No files found to process")
            return
        
        # 3. Process files into LlamaIndex Documents
        logger.info("🔧 Processing documents...")
        documents = document_processor.process_drive_files(files)
        processed_files = len(documents)
        failed_files = len(files) - processed_files
        
        # Create file details
        for i, file in enumerate(files):
            status = "processed" if i < processed_files else "failed"
            chunk_count = 0
            
            if i < processed_files:
                # Rough estimate of chunks
                chunk_count = len(documents[i].text) // 512 + 1
                total_chunks += chunk_count
            
            files_details.append({
                "name": file.get("name", "Unknown"),
                "id": file.get("id", "Unknown"),
                "status": status,
                "mime_type": file.get("mimeType", "Unknown"),
                "chunk_count": chunk_count
            })
        
        # 4. Build RAG index with batch processing
        if documents:
            logger.info(f"🚀 Building RAG index with {len(documents)} documents...")
            logger.info(f"Using batch size: {batch_size}")
            
            if len(documents) > batch_size:
                # Process in batches
                total_batches = (len(documents) + batch_size - 1) // batch_size
                logger.info(f"Processing in {total_batches} batches")
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    success = rag_service.add_documents(batch)
                    if not success:
                        logger.error(f"Failed to process batch {batch_num}")
                        break
                    
                    # Brief pause between batches for API rate limiting
                    if batch_num < total_batches:
                        time.sleep(1)
            else:
                # Process all at once
                success = rag_service.add_documents(documents)
            
            if success:
                logger.info("✅ Background index rebuild completed successfully")
            else:
                logger.error("❌ Failed to build index with documents")
        else:
            logger.warning("No documents were successfully processed")
    
    except Exception as e:
        logger.error(f"❌ Error during background index rebuild: {e}")
        # Update error status for all files
        for detail in files_details:
            if detail["status"] == "processed":
                detail["status"] = "failed"
                detail["error_message"] = str(e)
                failed_files += 1
                processed_files -= 1
    
    finally:
        duration = time.time() - start_time
        with rebuild_lock:
            rebuild_state["in_progress"] = False
            rebuild_state["last_completed"] = str(datetime.now())
            rebuild_state["last_duration"] = duration
        
        logger.info(f"Background rebuild completed in {duration:.2f} seconds")


@router.post("/refresh/sync", response_model=RefreshResponse)
async def refresh_index_sync(request: RefreshRequest):
    """Synchronously refresh the RAG index (for testing/small datasets)"""
    try:
        logger.info("Starting synchronous index refresh...")
        
        # Get files from Google Drive
        files = get_drive_files(request.folder_id)
        logger.info(f"Found {len(files)} files to process")
        
        if not files:
            return RefreshResponse(
                status="completed",
                message="No files found to process",
                processed_files=0,
                failed_files=0
            )
        
        # Process files into LlamaIndex Documents
        documents = document_processor.process_drive_files(files)
        processed_count = len(documents)
        failed_count = len(files) - processed_count
        
        files_details = []
        for i, file in enumerate(files):
            files_details.append({
                "name": file.get("name", "Unknown"),
                "id": file.get("id", "Unknown"),
                "status": "processed" if i < processed_count else "failed",
                "mime_type": file.get("mimeType", "Unknown")
            })
        
        if documents:
            # Add documents to the RAG index
            success = rag_service.add_documents(documents)
            if success:
                return RefreshResponse(
                    status="completed",
                    message=f"Successfully processed {processed_count} documents",
                    processed_files=processed_count,
                    failed_files=failed_count,
                    files_details=files_details
                )
            else:
                return RefreshResponse(
                    status="error",
                    message="Failed to add documents to index",
                    processed_files=0,
                    failed_files=len(files),
                    files_details=files_details
                )
        else:
            return RefreshResponse(
                status="completed",
                message="No documents were successfully processed",
                processed_files=0,
                failed_files=len(files),
                files_details=files_details
            )
    
    except Exception as e:
        logger.error(f"Error during synchronous index refresh: {e}")
        raise HTTPException(status_code=500, detail=f"Error during refresh: {str(e)}")


@router.delete("/index")
async def clear_index():
    """Clear the RAG index (for testing purposes)"""
    try:
        success = rag_service.clear_index()
        if success:
            return {"status": "success", "message": "Index cleared successfully"}
        else:
            return {"status": "error", "message": "Failed to clear index"}
    
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
            "document_count": stats.get("document_count", 0)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }