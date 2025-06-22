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


class RefreshResponse(BaseModel):
    status: str
    message: str
    processed_files: int = 0
    failed_files: int = 0
    files_details: List[Dict[str, Any]] = []

refresh_state = {
    "in_progress": False,
    "last_started": None,
    "last_completed": None,
    "last_duration": None,
    "next_scheduled": None
}

refresh_lock = threading.Lock()


def schedule_periodic_refresh(interval_hours=2):
    """Schedule periodic index refreshes"""
    def refresh_worker():
        while True:
            try:
                with refresh_lock:
                    refresh_state["next_scheduled"] = str(datetime.now() + timedelta(hours=interval_hours))
                
                # Wait until next scheduled time
                time.sleep(interval_hours * 3600)
                
                # Start refresh if not already running
                with refresh_lock:
                    if not refresh_state["in_progress"]:
                        logger.info("🚀 Starting scheduled index refresh")
                        _refresh_index_background()
            except Exception as e:
                logger.error(f"Periodic refresh scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    # Start the scheduler thread
    scheduler_thread = threading.Thread(target=refresh_worker, daemon=True)
    scheduler_thread.start()
    logger.info("⏰ Periodic refresh scheduler started")


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question using the RAG system"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received query: {request.query}")
        response = rag_service.query(request.query)
        
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

@router.post("/refresh", response_model=RefreshResponse)
async def refresh_index(request: RefreshRequest, background_tasks: BackgroundTasks):
    """Refresh the RAG index with latest documents from Google Drive"""
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
        
        return RefreshResponse(
            status="completed",
            message=f"Processed {processed_files} files in {duration:.1f} seconds",
            processed_files=processed_files,
            failed_files=failed_files,
            skipped_files=skipped_files,
            total_chunks=total_chunks,
            files_details=files_details
        )

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
        # This would need to be implemented in the RAG service
        # For now, just return success
        return {"status": "success", "message": "Index cleared (placeholder)"}
    
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