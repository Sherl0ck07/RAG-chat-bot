"""
Data Ingestion Pipeline for Veltris Doc-Bot
Filters accelerate documentation, chunks it, and stores in ChromaDB
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DocumentIngestionError(Exception):
    """Custom exception for ingestion errors"""
    pass


class DocumentIngestionPipeline:
    """Pipeline for ingesting and processing documentation"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "../vector_db",
        collection_name: str = "accelerate_docs"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        try:
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize embeddings
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {type(e).__name__}: {e}")
            raise DocumentIngestionError(f"Pipeline initialization failed: {str(e)}")
    
    def load_huggingface_docs(self, subset_filter: str = "accelerate") -> List[Dict]:
        """Load HuggingFace documentation dataset and filter by subset"""
        logger.info(f"Loading HuggingFace documentation dataset (filter: {subset_filter})...")
        
        try:
            ds = load_dataset("m-ric/huggingface_doc", split="train")
            logger.info(f"Total documents loaded: {len(ds)}")
            
            if not ds:
                raise DocumentIngestionError("Dataset is empty")
            
            # Filter for accelerate docs
            filtered_docs = []
            for idx, doc in enumerate(ds):
                try:
                    source = doc.get("source", "")
                    text = doc.get("text", "")
                    
                    if not source or not text:
                        logger.warning(f"Skipping document {idx} with missing source or text")
                        continue
                    
                    if subset_filter.lower() in source.lower():
                        filtered_docs.append({
                            "text": text,
                            "source": source
                        })
                except Exception as e:
                    logger.warning(f"Error processing document {idx}: {type(e).__name__}: {e}")
                    continue
            
            logger.info(f"Filtered documents ({subset_filter}): {len(filtered_docs)}")
            
            if not filtered_docs:
                raise DocumentIngestionError(f"No documents found matching filter: {subset_filter}")
            
            return filtered_docs
            
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {type(e).__name__}: {e}")
            raise DocumentIngestionError(f"Failed to load dataset: {str(e)}")
    
    def extract_metadata(self, source: str) -> Dict[str, str]:
        """Extract metadata from source path"""
        try:
            if not source or not isinstance(source, str):
                logger.warning(f"Invalid source: {source}")
                return {
                    "source_file": str(source),
                    "repository": "unknown",
                    "project": "unknown"
                }
            
            # Example source: 'huggingface/accelerate/blob/main/docs/source/usage_guides/gradient_accumulation.mdx'
            parts = source.split('/')
            
            metadata = {
                "source_file": source,
                "repository": parts[0] if len(parts) > 0 else "unknown",
                "project": parts[1] if len(parts) > 1 else "unknown",
            }
            
            # Extract filename if available
            if '.md' in source or '.mdx' in source:
                filename = source.split('/')[-1]
                metadata["filename"] = filename
                metadata["section"] = filename.replace('.md', '').replace('.mdx', '').replace('_', ' ').title()
            
            return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata: {type(e).__name__}: {e}")
            return {
                "source_file": str(source),
                "repository": "unknown",
                "project": "unknown"
            }
    
    def create_chunks(self, documents: List[Dict]) -> List[Document]:
        """Chunk documents with metadata preservation"""
        logger.info("Creating document chunks...")
        
        try:
            all_chunks = []
            
            for idx, doc in enumerate(documents):
                try:
                    text = doc.get("text", "")
                    source = doc.get("source", "")
                    
                    if not text or not source:
                        logger.warning(f"Skipping document {idx} with missing text or source")
                        continue
                    
                    # Extract metadata
                    metadata = self.extract_metadata(source)
                    metadata["doc_index"] = idx
                    metadata["ingestion_timestamp"] = datetime.now().isoformat()
                    
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(text)
                    
                    if not chunks:
                        logger.warning(f"Document {idx} produced no chunks")
                        continue
                    
                    # Create Document objects with metadata
                    for chunk_idx, chunk in enumerate(chunks):
                        if not chunk or not chunk.strip():
                            continue
                        
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = chunk_idx
                        chunk_metadata["total_chunks"] = len(chunks)
                        
                        all_chunks.append(
                            Document(
                                page_content=chunk.strip(),
                                metadata=chunk_metadata
                            )
                        )
                except Exception as e:
                    logger.warning(f"Error processing document {idx}: {type(e).__name__}: {e}")
                    continue
            
            if not all_chunks:
                raise DocumentIngestionError("No chunks were created from documents")
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            return all_chunks
            
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Error creating chunks: {type(e).__name__}: {e}")
            raise DocumentIngestionError(f"Failed to create chunks: {str(e)}")
    
    def store_in_vectordb(self, chunks: List[Document]) -> Chroma:
        """Store chunks in ChromaDB with embeddings"""
        logger.info("Storing chunks in vector database...")
        
        try:
            if not chunks:
                raise DocumentIngestionError("No chunks provided for storage")
            
            # Create persist directory if it doesn't exist
            try:
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create persist directory: {type(e).__name__}: {e}")
                raise DocumentIngestionError(f"Cannot create directory: {self.persist_directory}")
            
            # Create Chroma vector store
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            # Persist the database
            vectordb.persist()
            
            logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
            logger.info(f"Persist directory: {self.persist_directory}")
            
            return vectordb
            
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Error storing in vector database: {type(e).__name__}: {e}")
            raise DocumentIngestionError(f"Failed to store in vector database: {str(e)}")
    
    def run_pipeline(self, subset_filter: str = "accelerate"):
        """Run the complete ingestion pipeline"""
        logger.info("="*60)
        logger.info("Starting Document Ingestion Pipeline")
        logger.info("="*60)
        
        try:
            # Step 1: Load documents
            try:
                documents = self.load_huggingface_docs(subset_filter)
            except DocumentIngestionError:
                raise
            except Exception as e:
                raise DocumentIngestionError(f"Failed to load documents: {str(e)}")
            
            # Step 2: Create chunks
            try:
                chunks = self.create_chunks(documents)
            except DocumentIngestionError:
                raise
            except Exception as e:
                raise DocumentIngestionError(f"Failed to create chunks: {str(e)}")
            
            # Step 3: Store in vector database
            try:
                vectordb = self.store_in_vectordb(chunks)
            except DocumentIngestionError:
                raise
            except Exception as e:
                raise DocumentIngestionError(f"Failed to store in vector database: {str(e)}")
            
            # Step 4: Verify storage
            try:
                collection_count = vectordb._collection.count()
                logger.info(f"Vector DB collection count: {collection_count}")
                
                if collection_count == 0:
                    logger.warning("Vector DB collection is empty after ingestion")
            except Exception as e:
                logger.warning(f"Failed to verify storage: {type(e).__name__}: {e}")
            
            logger.info("="*60)
            logger.info("Ingestion Pipeline Completed Successfully!")
            logger.info("="*60)
            
            return vectordb
            
        except DocumentIngestionError as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {type(e).__name__}: {e}", exc_info=True)
            raise DocumentIngestionError(f"Pipeline failed unexpectedly: {str(e)}")


def main():
    """Main execution function"""
    try:
        # Load configuration from environment or use defaults
        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "../vector_db")
        collection_name = os.getenv("COLLECTION_NAME", "accelerate_docs")
        
        # Initialize pipeline
        try:
            pipeline = DocumentIngestionPipeline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        except DocumentIngestionError as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            return 1
        
        # Run pipeline
        try:
            pipeline.run_pipeline(subset_filter="accelerate")
            return 0
        except DocumentIngestionError as e:
            logger.error(f"Ingestion failed: {str(e)}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error in main: {type(e).__name__}: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())