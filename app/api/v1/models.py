from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask the RAG system")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of relevant chunks to retrieve")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature for response generation")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the RAG system")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the response")


class DocumentMetadata(BaseModel):
    source: str
    file_id: str
    mime_type: str
    file_size: str
    created_time: str
    modified_time: str
    owners: List[str]
    drive_link: str


class IndexStatsResponse(BaseModel):
    status: str = Field(..., description="Status of the RAG index")
    document_count: int = Field(..., description="Number of documents in the index")
    chunk_size: Optional[int] = Field(None, description="Size of text chunks")
    chunk_overlap: Optional[int] = Field(None, description="Overlap between chunks")
    top_k_retrieval: Optional[int] = Field(None, description="Default number of chunks retrieved")
    error: Optional[str] = Field(None, description="Error message if any")


class RefreshRequest(BaseModel):
    folder_id: Optional[str] = Field(None, description="Specific Google Drive folder ID to refresh")
    force_reprocess: Optional[bool] = Field(False, description="Force reprocessing of all documents")


class FileProcessingDetail(BaseModel):
    name: str
    id: str
    status: str  # "processed", "failed", "skipped"
    mime_type: str
    error_message: Optional[str] = None
    chunk_count: Optional[int] = None


class RefreshResponse(BaseModel):
    status: str = Field(..., description="Status of the refresh operation")
    message: str = Field(..., description="Human-readable message about the operation")
    processed_files: int = Field(0, description="Number of files successfully processed")
    failed_files: int = Field(0, description="Number of files that failed processing")
    skipped_files: int = Field(0, description="Number of files skipped")
    total_chunks: int = Field(0, description="Total number of text chunks created")
    files_details: List[FileProcessingDetail] = Field([], description="Detailed information about each file")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system health status")
    rag_status: str = Field(..., description="RAG system status")
    document_count: int = Field(..., description="Number of documents in index")
    uptime: Optional[str] = Field(None, description="System uptime")
    last_refresh: Optional[str] = Field(None, description="Last time the index was refreshed")


class SearchResult(BaseModel):
    content: str = Field(..., description="Text content of the chunk")
    score: float = Field(..., description="Relevance score")
    metadata: DocumentMetadata = Field(..., description="Source document metadata")


class QueryAnalysis(BaseModel):
    intent: str = Field(..., description="Detected intent of the query")
    entities: List[str] = Field([], description="Extracted entities from the query")
    keywords: List[str] = Field([], description="Key terms from the query")
    complexity: str = Field(..., description="Query complexity level")


class EnhancedAskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(..., description="Confidence score of the answer")
    sources: List[SearchResult] = Field([], description="Source chunks used for the answer")
    query_analysis: Optional[QueryAnalysis] = Field(None, description="Analysis of the input query")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used for generation")

class RefreshStatus(BaseModel):
    in_progress: bool = Field(..., description="Whether a refresh is currently running")
    last_started: Optional[str] = Field(None, description="Timestamp of last refresh start")
    last_completed: Optional[str] = Field(None, description="Timestamp of last refresh completion")
    last_duration: Optional[float] = Field(None, description="Duration of last refresh in seconds")
    next_scheduled: Optional[str] = Field(None, description="Next scheduled refresh time")