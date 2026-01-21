"""
Structured logging utility for the application
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict


class StructuredLogger:
    """Structured JSON logger for production"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def _format_message(
        self,
        level: str,
        message: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Format log message as structured JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        }
        
        # Add additional context
        if kwargs:
            log_data["context"] = kwargs
        
        return log_data
    
    def info(self, message: str, **kwargs: Any):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs: Any):
        """Log error message"""
        self.logger.error(message, extra=kwargs, exc_info=True)
    
    def warning(self, message: str, **kwargs: Any):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs: Any):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def log_query(
        self,
        query: str,
        retrieval_time: float,
        llm_time: float,
        total_time: float,
        sources_count: int
    ):
        """Log query metrics"""
        self.info(
            "Query processed",
            query_length=len(query),
            retrieval_time_ms=round(retrieval_time * 1000, 2),
            llm_time_ms=round(llm_time * 1000, 2),
            total_time_ms=round(total_time * 1000, 2),
            sources_count=sources_count
        )


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Get or create a structured logger"""
    return StructuredLogger(name, level)