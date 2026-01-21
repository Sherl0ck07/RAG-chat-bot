"""
API Client for Backend Communication
"""

import requests
from typing import Dict, Optional, List
import time


class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class APIClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 60  # 60 seconds for LLM responses
    
    def health_check(self) -> Optional[Dict]:
        """
        Check backend health status
        
        Returns:
            Dict with health status or None if failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"status": "offline", "message": "Cannot connect to backend"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "message": "Backend not responding"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> Optional[Dict]:
        """
        Send a query to the backend
        
        Args:
            query: User's question
            session_id: Optional session identifier
            top_k: Number of documents to retrieve
        
        Returns:
            Dict with answer, sources, and confidence
        
        Raises:
            APIError: If the request fails
        """
        try:
            payload = {
                "query": query,
                "top_k": top_k
            }
            
            if session_id:
                payload["session_id"] = session_id
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            # Handle different status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                error_data = response.json()
                raise APIError(
                    f"Bad Request: {error_data.get('detail', 'Invalid query')}",
                    status_code=400
                )
            elif response.status_code == 503:
                raise APIError(
                    "Service temporarily unavailable. Please try again.",
                    status_code=503
                )
            else:
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            raise APIError(
                "Request timed out. The query might be too complex.",
                status_code=504
            )
        except requests.exceptions.ConnectionError:
            raise APIError(
                "Cannot connect to backend. Please ensure the API server is running.",
                status_code=503
            )
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")