"""
Data Ingestion Pipeline for Veltris Doc-Bot
Filters accelerate documentation, chunks it, and stores in ChromaDB
"""
# File: backend/ingestion/ingest.py
import os
import sys
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime
from tqdm import tqdm

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

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using embedding device: {device}")

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
                model_kwargs={"device": device},
    encode_kwargs={"batch_size": 64}
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {type(e).__name__}: {e}")
            raise DocumentIngestionError(f"Pipeline initialization failed: {str(e)}")
    
    def load_huggingface_docs(self, subset_filter: str = "accelerate") -> List[Dict]:
        """Load HuggingFace documentation dataset and filter by subset"""
        print(f"[INGEST_LOAD] Loading HuggingFace documentation dataset...")
        logger.info(f"Loading HuggingFace documentation dataset (filter: {subset_filter})...")
        
        try:
            print(f"[INGEST_LOAD] Fetching dataset from Hugging Face...")
            ds = load_dataset("m-ric/huggingface_doc", split="train")
            print(f"[INGEST_LOAD] Total documents in dataset: {len(ds)}")
            logger.info(f"Total documents loaded: {len(ds)}")
            
            if not ds:
                raise DocumentIngestionError("Dataset is empty")
            
            # Filter for accelerate docs
            print(f"[INGEST_LOAD] Filtering documents for '{subset_filter}'...")
            filtered_docs = []
            for idx, doc in enumerate(tqdm(ds, desc="Filtering documents")):
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
            
            print(f"[INGEST_LOAD] Filtered documents: {len(filtered_docs)} matching '{subset_filter}'")
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
        print(f"[INGEST_CHUNK] Starting chunking process for {len(documents)} documents")
        print(f"[INGEST_CHUNK] Chunk size={self.chunk_size}, overlap={self.chunk_overlap}")
        logger.info("Creating document chunks...")
        
        try:
            all_chunks = []
            
            for idx, doc in enumerate(tqdm(documents, desc="Chunking documents")):

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
                    print(f"[INGEST_CHUNK] Document {idx}: {len(chunks)} chunks created (text_len={len(text)})")
                    
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
            
            print(f"[INGEST_CHUNK] Chunking complete: {len(all_chunks)} total chunks from {len(documents)} documents")
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            return all_chunks
            
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Error creating chunks: {type(e).__name__}: {e}")
            raise DocumentIngestionError(f"Failed to create chunks: {str(e)}")
    
    def store_in_vectordb(self, chunks: List[Document]) -> Chroma:
        print(f"[INGEST_STORE] Starting vector database storage for {len(chunks)} chunks")
        logger.info("Storing chunks in vector database...")

        if not chunks:
            raise DocumentIngestionError("No chunks provided for storage")

        print(f"[INGEST_STORE] Creating persist directory: {self.persist_directory}")
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        print(f"[INGEST_STORE] Initializing ChromaDB with collection: {self.collection_name}")
        vectordb = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        BATCH_SIZE = 256
        print(f"[INGEST_STORE] Adding documents in batches of {BATCH_SIZE}")

        for i in tqdm(
            range(0, len(chunks), BATCH_SIZE),
            desc="Embedding + storing chunks"
        ):
            batch = chunks[i:i + BATCH_SIZE]
            print(f"[INGEST_STORE] Adding batch {i//BATCH_SIZE + 1}: {len(batch)} chunks")
            vectordb.add_documents(batch)

        print(f"[INGEST_STORE] Persisting ChromaDB to disk...")
        vectordb.persist()

        print(f"[INGEST_STORE] Successfully stored {len(chunks)} chunks in ChromaDB")
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
        return vectordb

    
    def run_pipeline(self, subset_filter: str = "accelerate"):
        """Run the complete ingestion pipeline"""
        print("\n[INGEST_PIPELINE] " + "="*60)
        print("[INGEST_PIPELINE] Starting Document Ingestion Pipeline")
        print("[INGEST_PIPELINE] " + "="*60)
        logger.info("="*60)
        logger.info("Starting Document Ingestion Pipeline")
        logger.info("="*60)
        
        try:
            # Step 1: Load documents
            print(f"[INGEST_PIPELINE] Step 1: Loading documents")
            try:
                documents = self.load_huggingface_docs(subset_filter)
                print(f"[INGEST_PIPELINE] Step 1 complete: {len(documents)} documents loaded")
            except DocumentIngestionError:
                raise
            except Exception as e:
                raise DocumentIngestionError(f"Failed to load documents: {str(e)}")
            
            # Step 2: Create chunks
            print(f"[INGEST_PIPELINE] Step 2: Creating chunks")
            try:
                chunks = self.create_chunks(documents)
                print(f"[INGEST_PIPELINE] Step 2 complete: {len(chunks)} chunks created")
            except DocumentIngestionError:
                raise
            except Exception as e:
                raise DocumentIngestionError(f"Failed to create chunks: {str(e)}")
            
            # Step 3: Store in vector database
            print(f"[INGEST_PIPELINE] Step 3: Storing in vector database")
            try:
                vectordb = self.store_in_vectordb(chunks)
                print(f"[INGEST_PIPELINE] Step 3 complete: Chunks stored in ChromaDB")
            except DocumentIngestionError:
                raise
            except Exception as e:
                raise DocumentIngestionError(f"Failed to store in vector database: {str(e)}")
            
            # Step 4: Verify storage
            print(f"[INGEST_PIPELINE] Step 4: Verifying storage")
            try:
                collection_count = vectordb._collection.count()
                print(f"[INGEST_PIPELINE] Vector DB collection count: {collection_count}")
                logger.info(f"Vector DB collection count: {collection_count}")
                
                if collection_count == 0:
                    print(f"[INGEST_PIPELINE] WARNING: Vector DB collection is empty after ingestion")
                    logger.warning("Vector DB collection is empty after ingestion")
            except Exception as e:
                print(f"[INGEST_PIPELINE] Warning verifying storage: {type(e).__name__}: {e}")
                logger.warning(f"Failed to verify storage: {type(e).__name__}: {e}")
            
            print("[INGEST_PIPELINE] " + "="*60)
            print("[INGEST_PIPELINE] Ingestion Pipeline Completed Successfully!")
            print("[INGEST_PIPELINE] " + "="*60)
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
    print("\n[MAIN] Starting document ingestion...")
    try:
        # Load configuration from environment or use defaults
        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "../vector_db")
        collection_name = os.getenv("COLLECTION_NAME", "accelerate_docs")
        
        print(f"[MAIN] Configuration loaded:")
        print(f"[MAIN]   chunk_size={chunk_size}")
        print(f"[MAIN]   chunk_overlap={chunk_overlap}")
        print(f"[MAIN]   embedding_model={embedding_model}")
        print(f"[MAIN]   persist_directory={persist_directory}")
        print(f"[MAIN]   collection_name={collection_name}")
        
        # Initialize pipeline
        try:
            print(f"[MAIN] Initializing pipeline...")
            pipeline = DocumentIngestionPipeline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            print(f"[MAIN] Pipeline initialized successfully")
        except DocumentIngestionError as e:
            print(f"[MAIN] ERROR: Failed to initialize pipeline: {str(e)}")
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            return 1
        
        # Run pipeline
        try:
            print(f"[MAIN] Running ingestion pipeline...")
            pipeline.run_pipeline(subset_filter="accelerate")
            print(f"[MAIN] Ingestion completed successfully")
            return 0
        except DocumentIngestionError as e:
            print(f"[MAIN] ERROR: Ingestion failed: {str(e)}")
            logger.error(f"Ingestion failed: {str(e)}")
            return 1
            
    except Exception as e:
        print(f"[MAIN] ERROR: Unexpected error: {type(e).__name__}: {e}")
        logger.error(f"Unexpected error in main: {type(e).__name__}: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())