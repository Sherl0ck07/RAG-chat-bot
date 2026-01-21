"""
RAG Service - Core retrieval and generation logic
"""

import time
from typing import List, Dict, Tuple, Optional
from functools import wraps
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from openai import APIError, APIConnectionError, APITimeoutError

from app.config import settings
from app.models.schemas import SourceCitation, ChatResponse
from app.utils.logger import get_logger
from app.exceptions import (
    VectorStoreError,
    EmbeddingError,
    LLMError,
    RAGServiceError,
    ConfigurationError
)

logger = get_logger(__name__, settings.log_level)


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (APITimeoutError, APIConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}, "
                            f"retrying in {wait_time}s: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}")
                except Exception as e:
                    # Don't retry on non-transient errors
                    raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RAGService:
    """Retrieval-Augmented Generation Service"""
    
    # System prompt for strict context-only answering
    SYSTEM_PROMPT = """You are a technical documentation assistant for HuggingFace Accelerate library.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. Answer ONLY using the provided context below
2. If the answer is not in the context or you are unsure, you MUST respond EXACTLY with: "I cannot find the answer in the provided documentation."
3. Be precise, technical, and concise
4. Do not make assumptions or use external knowledge
5. When answering, reference specific parts of the context
6. If the context is partially relevant but doesn't fully answer the question, state what you can answer and what's missing

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(self):
        """Initialize RAG service"""
        logger.info("Initializing RAG Service...")
        
        try:
            # Validate configuration
            self._validate_config()
            
            # Initialize embeddings
            self.embeddings = self._initialize_embeddings()
            
            # Initialize vector store
            self.vectorstore = self._load_vectorstore()
            
            # Initialize LLM (Nebius OpenAI-compatible)
            self.llm = self._initialize_llm()
            
            logger.info("RAG Service initialized successfully")
        except ConfigurationError:
            raise
        except EmbeddingError:
            raise
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during RAG Service initialization: {type(e).__name__}: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize RAG Service: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate required configuration"""
        try:
            if not settings.nebius_api_key:
                raise ConfigurationError("Nebius API key is not set")
            if not settings.nebius_base_url:
                raise ConfigurationError("Nebius base URL is not set")
            if not settings.chroma_persist_directory:
                raise ConfigurationError("Chroma persist directory is not set")
            
            logger.info("Configuration validation passed")
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embedding model with error handling"""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embedding model loaded successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embedding model: {type(e).__name__}: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")
    
    def _load_vectorstore(self) -> Chroma:
        """Load existing vector store with error handling"""
        try:
            logger.info(f"Loading vector store from {settings.chroma_persist_directory}")
            vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=self.embeddings,
                collection_name=settings.collection_name
            )
            
            count = vectorstore._collection.count()
            logger.info(f"Loaded vector store with {count} documents")
            
            if count == 0:
                logger.warning("Vector store is empty - no documents found")
            
            return vectorstore
            
        except FileNotFoundError as e:
            logger.error(f"Vector store directory not found: {settings.chroma_persist_directory}")
            raise VectorStoreError(
                f"Vector store not found at {settings.chroma_persist_directory}",
                details={"path": settings.chroma_persist_directory}
            )
        except Exception as e:
            logger.error(f"Error loading vector store: {type(e).__name__}: {e}")
            raise VectorStoreError(f"Failed to load vector store: {str(e)}")
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize LLM with error handling"""
        try:
            logger.info(f"Initializing LLM: {settings.nebius_model_name}")
            llm = ChatOpenAI(
                model=settings.nebius_model_name,
                openai_api_key=settings.nebius_api_key,
                openai_api_base=settings.nebius_base_url,
                temperature=0.1,
                max_tokens=1000,
                request_timeout=30
            )
            logger.info("LLM initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {type(e).__name__}: {e}")
            raise LLMError(f"Failed to initialize LLM: {str(e)}")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            logger.info("Cleaning up RAG Service resources...")
            if hasattr(self, 'vectorstore') and self.vectorstore:
                # Vectorstore cleanup if needed
                pass
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {type(e).__name__}: {e}")
            raise
    
    def retrieve_documents(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores"""
        start_time = time.time()
        
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        print(f"\n{'='*80}")
        print(f"[RETRIEVE_DOCUMENTS] Starting retrieval")
        print(f"[RETRIEVE_DOCUMENTS] Query: {query}")
        print(f"[RETRIEVE_DOCUMENTS] Top-K: {top_k}")
        print(f"[RETRIEVE_DOCUMENTS] Similarity threshold: {settings.similarity_threshold}")
        
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Retrieve documents with scores
            print(f"[RETRIEVE_DOCUMENTS] Calling vectorstore.similarity_search_with_score...")
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )
            
            print(f"[RETRIEVE_DOCUMENTS] Retrieved {len(docs_with_scores)} raw documents")
            
            # Print details of all retrieved documents
            for idx, (doc, score) in enumerate(docs_with_scores):
                print(f"\n  [DOC {idx}] Score (distance): {score:.6f}")
                print(f"  [DOC {idx}] Similarity: {1-score:.6f}")
                print(f"  [DOC {idx}] Source: {doc.metadata.get('source_file', 'Unknown')}")
                print(f"  [DOC {idx}] Section: {doc.metadata.get('section', 'Unknown')}")
                print(f"  [DOC {idx}] Content preview: {doc.page_content[:100]}...")
            
            retrieval_time = time.time() - start_time
            
            # Filter by similarity threshold
            print(f"\n[RETRIEVE_DOCUMENTS] Filtering by threshold: {settings.similarity_threshold}")
            print(f"[RETRIEVE_DOCUMENTS] Max distance allowed: {1 - settings.similarity_threshold:.6f}")
            
            filtered_docs = [
                (doc, score) 
                for doc, score in docs_with_scores 
                if score <= settings.similarity_threshold  # ChromaDB uses distance
            ]
            
            print(f"[RETRIEVE_DOCUMENTS] After filtering: {len(filtered_docs)}/{len(docs_with_scores)} documents passed threshold")
            
            for idx, (doc, score) in enumerate(filtered_docs):
                print(f"  [FILTERED_DOC {idx}] Similarity: {1-score:.6f}")
            
            logger.info(
                f"Retrieved {len(filtered_docs)}/{len(docs_with_scores)} documents",
                retrieval_time_ms=round(retrieval_time * 1000, 2),
                query_length=len(query)
            )
            
            print(f"[RETRIEVE_DOCUMENTS] Retrieval completed in {retrieval_time:.3f}s")
            print(f"{'='*80}\n")
            
            return filtered_docs
            
        except ValueError as e:
            print(f"[RETRIEVE_DOCUMENTS] ValueError: {str(e)}")
            logger.error(f"Invalid query for retrieval: {str(e)}")
            raise
        except Exception as e:
            print(f"[RETRIEVE_DOCUMENTS] Exception: {type(e).__name__}: {e}")
            logger.error(f"Error retrieving documents: {type(e).__name__}: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to retrieve documents: {str(e)}")
    
    def format_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Format retrieved documents into context string"""
        print(f"\n{'='*80}")
        print(f"[FORMAT_CONTEXT] Starting context formatting")
        print(f"[FORMAT_CONTEXT] Number of documents: {len(documents)}")
        
        try:
            if not documents:
                print(f"[FORMAT_CONTEXT] No documents provided")
                logger.warning("No documents provided for context formatting")
                return ""
            
            context_parts = []
            for idx, (doc, score) in enumerate(documents, 1):
                if not doc or not hasattr(doc, 'page_content'):
                    print(f"[FORMAT_CONTEXT] Skipping invalid document {idx}")
                    logger.warning(f"Invalid document at index {idx}")
                    continue
                
                source = doc.metadata.get('source_file', 'Unknown') if doc.metadata else 'Unknown'
                section = doc.metadata.get('section', 'Unknown Section') if doc.metadata else 'Unknown Section'
                
                print(f"[FORMAT_CONTEXT] Document {idx}: {section} from {source}")
                print(f"[FORMAT_CONTEXT]   Content length: {len(doc.page_content)} chars")
                
                context_parts.append(
                    f"[Source {idx}: {section} from {source}]\n{doc.page_content}\n"
                )
            
            if not context_parts:
                print(f"[FORMAT_CONTEXT] No valid documents found")
                logger.warning("No valid documents found to format context")
                return ""
            
            formatted_context = "\n".join(context_parts)
            print(f"[FORMAT_CONTEXT] Final context length: {len(formatted_context)} chars")
            print(f"[FORMAT_CONTEXT] Context preview: {formatted_context[:200]}...")
            print(f"{'='*80}\n")
            
            return formatted_context
        except Exception as e:
            print(f"[FORMAT_CONTEXT] Exception: {type(e).__name__}: {e}")
            logger.error(f"Error formatting context: {type(e).__name__}: {e}")
            raise RAGServiceError(f"Failed to format context: {str(e)}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retry logic"""
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"[GENERATE_ANSWER] Starting answer generation")
        print(f"[GENERATE_ANSWER] Query: {query}")
        print(f"[GENERATE_ANSWER] Context length: {len(context)} chars")
        print(f"[GENERATE_ANSWER] Context preview: {context[:300]}...")
        
        try:
            if not context or not context.strip():
                print(f"[GENERATE_ANSWER] Empty context provided")
                logger.warning("Empty context provided to generate_answer")
                return "I cannot find the answer in the provided documentation."
            
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template(self.SYSTEM_PROMPT)
            print(f"[GENERATE_ANSWER] ChatPromptTemplate created")
            
            # Format prompt
            formatted_prompt = prompt.format(context=context, question=query)
            print(f"[GENERATE_ANSWER] Formatted prompt length: {len(formatted_prompt)} chars")
            print(f"[GENERATE_ANSWER] Formatted prompt preview:")
            print(f"{formatted_prompt[:500]}")
            print(f"[GENERATE_ANSWER] Full formatted prompt:\n{formatted_prompt}\n")
            
            # Generate response
            print(f"[GENERATE_ANSWER] Calling LLM with model: {settings.nebius_model_name}")
            response = self.llm.predict(formatted_prompt)
            print(f"[GENERATE_ANSWER] LLM response received")
            
            if not response or not isinstance(response, str):
                print(f"[GENERATE_ANSWER] Invalid response type: {type(response)}")
                logger.error("LLM returned invalid response")
                raise LLMError("LLM returned invalid response type")
            
            print(f"[GENERATE_ANSWER] Response length: {len(response)} chars")
            print(f"[GENERATE_ANSWER] Response: {response}")
            
            generation_time = time.time() - start_time
            
            logger.info(
                f"Generated answer",
                generation_time_ms=round(generation_time * 1000, 2),
                response_length=len(response)
            )
            
            print(f"[GENERATE_ANSWER] Generation completed in {generation_time:.3f}s")
            print(f"{'='*80}\n")
            
            # Remove thinking tags if present
            response_clean = response.strip()
            if response_clean.startswith('<think>'):
                response_clean = response_clean.split('</think>')[-1].strip()
            return response_clean
            
        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"[GENERATE_ANSWER] LLM API error: {type(e).__name__}: {e}")
            logger.error(f"LLM API error: {type(e).__name__}: {e}")
            raise LLMError(f"LLM service error: {str(e)}", details={"error_type": type(e).__name__})
        except ValueError as e:
            print(f"[GENERATE_ANSWER] ValueError: {str(e)}")
            logger.error(f"Invalid input to generate_answer: {str(e)}")
            raise
        except Exception as e:
            print(f"[GENERATE_ANSWER] Exception: {type(e).__name__}: {e}")
            logger.error(f"Error generating answer: {type(e).__name__}: {e}", exc_info=True)
            raise LLMError(f"Failed to generate answer: {str(e)}")
    
    def extract_citations(
        self,
        documents: List[Tuple[Document, float]]
    ) -> List[SourceCitation]:
        """Extract source citations from retrieved documents"""
        try:
            if not documents:
                logger.warning("No documents provided for citation extraction")
                return []
            
            citations = []
            
            for doc, score in documents:
                try:
                    if not doc or not doc.metadata:
                        logger.warning("Skipping document with missing metadata")
                        continue
                    
                    metadata = doc.metadata
                    
                    # Convert distance to similarity (ChromaDB uses distance)
                    similarity = 1 - score
                    
                    citation = SourceCitation(
                        source_file=metadata.get('source_file', 'Unknown'),
                        filename=metadata.get('filename'),
                        section=metadata.get('section'),
                        chunk_index=metadata.get('chunk_index'),
                        similarity_score=round(similarity, 4),
                        excerpt=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    )
                    
                    citations.append(citation)
                except Exception as e:
                    logger.warning(f"Error extracting citation from document: {type(e).__name__}: {e}")
                    continue
            
            return citations
        except Exception as e:
            logger.error(f"Error extracting citations: {type(e).__name__}: {e}")
            return []
    
    def query(self, query: str, session_id: Optional[str] = None, top_k: int = None) -> ChatResponse:
        """
        Main query method - orchestrates retrieval and generation
        """
        total_start_time = time.time()
        
        print(f"\n\n{'#'*80}")
        print(f"# QUERY PROCESSING STARTED")
        print(f"# Query: {query}")
        print(f"# Session ID: {session_id}")
        print(f"# Top-K: {top_k}")
        print(f"{'#'*80}\n")
        
        try:
            # Validate input
            if not query or not query.strip():
                print(f"[QUERY] Invalid query - empty or whitespace")
                raise ValueError("Query cannot be empty or whitespace only")
            
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Retrieve relevant documents
            try:
                retrieval_start = time.time()
                print(f"[QUERY] Step 1: Retrieving documents...")
                documents = self.retrieve_documents(query, top_k)
                retrieval_time = time.time() - retrieval_start
                print(f"[QUERY] Documents retrieved: {len(documents)} in {retrieval_time:.3f}s")
            except VectorStoreError:
                raise
            except Exception as e:
                print(f"[QUERY] Retrieval failed: {type(e).__name__}: {e}")
                logger.error(f"Retrieval failed: {type(e).__name__}: {e}")
                raise VectorStoreError(f"Document retrieval failed: {str(e)}")
            
            # Check if we have relevant documents
            if not documents:
                print(f"[QUERY] No relevant documents found - returning default response")
                return ChatResponse(
                    answer="I cannot find the answer in the provided documentation.",
                    sources=[],
                    confidence=0.0,
                    session_id=session_id,
                    query=query
                )
            
            # Step 2: Format context
            try:
                print(f"[QUERY] Step 2: Formatting context...")
                context = self.format_context(documents)
                if not context:
                    print(f"[QUERY] Context formatting returned empty - returning default response")
                    return ChatResponse(
                        answer="I cannot find the answer in the provided documentation.",
                        sources=[],
                        confidence=0.0,
                        session_id=session_id,
                        query=query
                    )
                print(f"[QUERY] Context formatted: {len(context)} chars")
            except RAGServiceError:
                raise
            except Exception as e:
                print(f"[QUERY] Context formatting failed: {type(e).__name__}: {e}")
                logger.error(f"Context formatting failed: {type(e).__name__}: {e}")
                raise RAGServiceError(f"Failed to format context: {str(e)}")
            
            # Step 3: Generate answer
            try:
                print(f"[QUERY] Step 3: Generating answer...")
                llm_start = time.time()
                answer = self.generate_answer(query, context)
                llm_time = time.time() - llm_start
                print(f"[QUERY] Answer generated: {len(answer)} chars in {llm_time:.3f}s")
                print(f"[QUERY] Answer content: {answer}")
            except LLMError:
                raise
            except Exception as e:
                print(f"[QUERY] Answer generation failed: {type(e).__name__}: {e}")
                logger.error(f"Answer generation failed: {type(e).__name__}: {e}")
                raise LLMError(f"Failed to generate answer: {str(e)}")
            
            # Step 4: Extract citations
            try:
                print(f"[QUERY] Step 4: Extracting citations...")
                citations = self.extract_citations(documents)
                print(f"[QUERY] Citations extracted: {len(citations)}")
            except Exception as e:
                print(f"[QUERY] Citation extraction failed, continuing without citations: {e}")
                logger.warning(f"Citation extraction failed, returning response without citations: {e}")
                citations = []
            
            # Calculate confidence (average similarity of top documents)
# Calculate confidence - clamp to 0-1 range
            avg_similarity = max(0.0, sum(1 - score for _, score in documents) / len(documents)) if documents else 0.0            
            total_time = time.time() - total_start_time
            
            # Log metrics
            try:
                logger.log_query(
                    query=query,
                    retrieval_time=retrieval_time,
                    llm_time=llm_time,
                    total_time=total_time,
                    sources_count=len(citations)
                )
            except Exception as e:
                print(f"[QUERY] Failed to log metrics: {e}")
                logger.warning(f"Failed to log query metrics: {e}")
            
            print(f"\n[QUERY] FINAL RESPONSE:")
            print(f"  Answer: {answer}")
            print(f"  Confidence: {avg_similarity:.4f}")
            print(f"  Sources: {len(citations)}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"{'#'*80}\n")
            
            return ChatResponse(
                answer=answer,
                sources=citations,
                confidence=round(avg_similarity, 4),
                session_id=session_id,
                query=query
            )
            
        except (VectorStoreError, LLMError, RAGServiceError):
            print(f"[QUERY] Expected error - re-raising")
            raise
        except ValueError as e:
            print(f"[QUERY] Validation error: {str(e)}")
            logger.error(f"Validation error in query: {str(e)}")
            raise
        except Exception as e:
            print(f"[QUERY] Unexpected error: {type(e).__name__}: {e}")
            logger.error(f"Unexpected error processing query: {type(e).__name__}: {e}", exc_info=True)
            raise RAGServiceError(f"Failed to process query: {str(e)}")
    
    def health_check(self) -> Dict:
        """Check service health"""
        try:
            if not self.vectorstore or not hasattr(self.vectorstore, '_collection'):
                logger.error("Vector store is not properly initialized")
                return {
                    "status": "unhealthy",
                    "vector_db_status": "not_initialized",
                    "total_documents": 0
                }
            
            try:
                count = self.vectorstore._collection.count()
            except Exception as e:
                logger.warning(f"Failed to get document count: {type(e).__name__}: {e}")
                count = 0
            
            return {
                "status": "healthy",
                "vector_db_status": "connected",
                "total_documents": count
            }
        except Exception as e:
            logger.error(f"Health check failed: {type(e).__name__}: {e}")
            return {
                "status": "unhealthy",
                "vector_db_status": "error",
                "total_documents": 0
            }