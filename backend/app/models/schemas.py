"""
Pydantic models for API request/response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query to search in documentation"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking conversations"
    )
    top_k: Optional[int] = Field(
        5,
        ge=1,
        le=10,
        description="Number of documents to retrieve"
    )
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class SourceCitation(BaseModel):
    """Model for source citation"""
    source_file: str = Field(..., description="Source file path")
    filename: Optional[str] = Field(None, description="Filename if available")
    section: Optional[str] = Field(None, description="Section name if available")
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")
    similarity_score: Optional[float] = Field(None, description="Similarity score")
    excerpt: Optional[str] = Field(None, description="Text excerpt used")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Generated answer from the bot")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="List of source citations"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the answer"
    )
    session_id: Optional[str] = Field(None, description="Session ID")
    query: str = Field(..., description="Original query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To use Accelerate, you need to install it using pip...",
                "sources": [
                    {
                        "source_file": "huggingface/accelerate/docs/quickstart.md",
                        "filename": "quickstart.md",
                        "section": "Installation",
                        "similarity_score": 0.89
                    }
                ],
                "confidence": 0.89,
                "session_id": "session_123",
                "query": "How do I install Accelerate?"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    vector_db_status: str = Field(..., description="Vector database status")
    total_documents: Optional[int] = Field(None, description="Total documents in vector DB")
    timestamp: str = Field(..., description="Timestamp of health check")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Timestamp of error")


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint (optional)"""
    subset_filter: str = Field(
        "accelerate",
        description="Documentation subset to ingest"
    )
    force_refresh: bool = Field(
        False,
        description="Force refresh of existing data"
    )


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")