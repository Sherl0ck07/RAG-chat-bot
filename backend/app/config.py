"""
Configuration management for the application
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Nebius API Configuration
    nebius_api_key: str
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nebius_model_name: str = "deepseek-ai/DeepSeek-R1-0528"
    
    # Vector DB Configuration
    chroma_persist_directory: str = "../vector_db"
    collection_name: str = "accelerate_docs"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.5
    
    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API Configuration
    api_title: str = "Veltris Intelligent Doc-Bot API"
    api_version: str = "1.0.0"
    api_description: str = "Production-Ready RAG System for Technical Documentation"
    
    # Logging
    log_level: str = "INFO"
    
    # CORS
    cors_origins: list = ["http://localhost:8501", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
print("[CONFIG] Loading configuration from environment and defaults...")
settings = Settings()
print("[CONFIG] Configuration loaded successfully")
print(f"[CONFIG] API Title: {settings.api_title}")
print(f"[CONFIG] API Version: {settings.api_version}")
print(f"[CONFIG] Nebius Base URL: {settings.nebius_base_url}")
print(f"[CONFIG] Nebius Model: {settings.nebius_model_name}")
print(f"[CONFIG] Chroma Directory: {settings.chroma_persist_directory}")
print(f"[CONFIG] Collection Name: {settings.collection_name}")
print(f"[CONFIG] Similarity Threshold: {settings.similarity_threshold}")
print(f"[CONFIG] Top-K Retrieval: {settings.top_k_retrieval}")
print(f"[CONFIG] Embedding Model: {settings.embedding_model}")
print(f"[CONFIG] Log Level: {settings.log_level}")
print(f"[CONFIG] CORS Origins: {settings.cors_origins}")