"""
Custom exception classes for the application
"""

from typing import Optional, Dict, Any


class VeltrisException(Exception):
    """Base exception for Veltris application"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class VectorStoreError(VeltrisException):
    """Exception for vector store operations"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            status_code=503,
            details=details
        )


class LLMError(VeltrisException):
    """Exception for LLM operations"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=503,
            details=details
        )


class EmbeddingError(VeltrisException):
    """Exception for embedding operations"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            status_code=503,
            details=details
        )


class QueryValidationError(VeltrisException):
    """Exception for query validation"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="QUERY_VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class RAGServiceError(VeltrisException):
    """Exception for RAG service errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RAG_SERVICE_ERROR",
            status_code=500,
            details=details
        )


class ConfigurationError(VeltrisException):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details
        )


class DocumentIngestionError(VeltrisException):
    """Exception for document ingestion"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INGESTION_ERROR",
            status_code=400,
            details=details
        )
