"""
FastAPI Application - Veltris Intelligent Doc-Bot
"""
# File: backend/app/main.py
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Union

from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse
)
from app.services.rag_service import RAGService
from app.utils.logger import get_logger
from app.exceptions import (
    VeltrisException,
    ConfigurationError,
    RAGServiceError,
    QueryValidationError,
    VectorStoreError,
    LLMError,
    EmbeddingError
)

logger = get_logger(__name__, settings.log_level)

# Global RAG service instance
rag_service: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global rag_service
    logger.info("Starting up Veltris Doc-Bot API...")
    
    try:
        rag_service = RAGService()
        logger.info("RAG Service initialized successfully")
    except ConfigurationError as e:
        logger.error(f"Configuration error during startup: {e.message}")
        raise RuntimeError(f"Startup failed: {e.message}")
    except VectorStoreError as e:
        logger.error(f"Vector store initialization failed: {e.message}")
        raise RuntimeError(f"Vector store unavailable: {e.message}")
    except EmbeddingError as e:
        logger.error(f"Embedding model initialization failed: {e.message}")
        raise RuntimeError(f"Embedding service unavailable: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during startup: {type(e).__name__}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize RAG Service: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Veltris Doc-Bot API...")
    try:
        if rag_service:
            rag_service.cleanup()
    except Exception as e:
        logger.warning(f"Error during cleanup: {type(e).__name__}: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Veltris Intelligent Doc-Bot API",
        "version": settings.api_version,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns service status and vector database information
    """
    print(f"\n[HEALTH_CHECK] Performing health check...")
    
    try:
        if rag_service is None:
            print(f"[HEALTH_CHECK] ERROR: RAG Service not initialized")
            logger.error("RAG Service not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        print(f"[HEALTH_CHECK] RAG Service is initialized, checking health...")
        health_data = rag_service.health_check()
        
        print(f"[HEALTH_CHECK] Health status: {health_data['status']}")
        print(f"[HEALTH_CHECK] Vector DB status: {health_data['vector_db_status']}")
        print(f"[HEALTH_CHECK] Total documents: {health_data.get('total_documents', 0)}")
        
        return HealthResponse(
            status=health_data["status"],
            vector_db_status=health_data["vector_db_status"],
            total_documents=health_data.get("total_documents"),
            timestamp=datetime.utcnow().isoformat()
        )
    except VectorStoreError as e:
        print(f"[HEALTH_CHECK] VectorStoreError: {e.message}")
        logger.error(f"Vector store health check failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[HEALTH_CHECK] Exception: {type(e).__name__}: {e}")
        logger.error(f"Unexpected error in health check: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint - Query the documentation bot
    
    Args:
        request: ChatRequest containing query and optional parameters
    
    Returns:
        ChatResponse with answer, sources, and confidence
    
    Raises:
        HTTPException: If query processing fails
    """
    print(f"\n\n{'*'*80}")
    print(f"* CHAT ENDPOINT CALLED")
    print(f"* Query: {request.query}")
    print(f"* Session ID: {request.session_id}")
    print(f"* Top-K: {request.top_k}")
    print(f"{'*'*80}\n")
    
    try:
        if rag_service is None:
            print(f"[CHAT_ENDPOINT] ERROR: RAG Service not initialized")
            logger.error("RAG Service not initialized when processing chat request")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
        
        print(f"[CHAT_ENDPOINT] RAG Service is initialized")
        logger.info(f"Received chat request: {request.query[:100]}...")
        
        # Process query using RAG service
        try:
            print(f"[CHAT_ENDPOINT] Calling rag_service.query()...")
            response = rag_service.query(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k
            )
            
            print(f"[CHAT_ENDPOINT] Response received from RAG service")
            print(f"[CHAT_ENDPOINT] Answer: {response.answer}")
            print(f"[CHAT_ENDPOINT] Confidence: {response.confidence}")
            print(f"[CHAT_ENDPOINT] Sources count: {len(response.sources)}")
            
            logger.info(f"Successfully processed query, confidence: {response.confidence}")
            return response
            
        except QueryValidationError as e:
            print(f"[CHAT_ENDPOINT] QueryValidationError: {e.message}")
            logger.warning(f"Query validation failed: {e.message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=e.message
            )
        except VectorStoreError as e:
            print(f"[CHAT_ENDPOINT] VectorStoreError: {e.message}")
            logger.error(f"Vector store error during query: {e.message}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database temporarily unavailable"
            )
        except LLMError as e:
            print(f"[CHAT_ENDPOINT] LLMError: {e.message}")
            logger.error(f"LLM error during query: {e.message}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service temporarily unavailable"
            )
        except RAGServiceError as e:
            print(f"[CHAT_ENDPOINT] RAGServiceError: {e.message}")
            logger.error(f"RAG service error: {e.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process query"
            )
            
    except HTTPException:
        print(f"[CHAT_ENDPOINT] HTTPException raised - re-raising")
        raise
    except ValueError as e:
        print(f"[CHAT_ENDPOINT] ValueError: {str(e)}")
        logger.error(f"Validation error in chat request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except TimeoutError as e:
        print(f"[CHAT_ENDPOINT] TimeoutError: {str(e)}")
        logger.error(f"Query processing timeout: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Query processing took too long"
        )
    except Exception as e:
        print(f"[CHAT_ENDPOINT] Unexpected exception: {type(e).__name__}: {e}")
        logger.error(f"Unexpected error processing chat request: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@app.exception_handler(VeltrisException)
async def veltris_exception_handler(request: Request, exc: VeltrisException):
    """Handle custom Veltris exceptions"""
    logger.error(
        f"Veltris exception occurred: {exc.error_code}",
        error_code=exc.error_code,
        status_code=exc.status_code,
        details=exc.details
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error_code,
            detail=exc.message,
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(
        f"HTTP exception: {exc.detail}",
        status_code=exc.status_code
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            detail=str(exc.detail),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors"""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exception_type=type(exc).__name__,
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            detail="An unexpected error occurred. Please try again later.",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )