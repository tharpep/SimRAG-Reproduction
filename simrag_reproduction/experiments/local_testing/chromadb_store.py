"""
ChromaDB Vector Store
Handles ChromaDB setup and document management
Matches Colab notebook exactly
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List

from ...logging_config import get_logger

logger = get_logger(__name__)


class ChromaDBStore:
    """
    ChromaDB vector store wrapper
    Matches Colab notebook setup exactly
    """
    
    def __init__(self, collection_name: str = "model_test", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB store
        
        Args:
            collection_name: Name for the collection
            embedding_model: Embedding model to use (matches Colab default)
        """
        logger.info("Initializing ChromaDB...")
        
        self.client = chromadb.Client()
        self.embedding_model = embedding_model
        
        # CRITICAL: Use exact same embedding model as Colab
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        
        logger.info(f"✓ Using {embedding_model} (384 dimensions)")
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the collection
        
        Args:
            documents: List of document texts
        """
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=doc_ids
        )
        logger.info(f"✓ Documents indexed in ChromaDB")
    
    def query(self, question: str, top_k: int = 5) -> tuple[List[str], List[float]]:
        """
        Query the collection for similar documents
        
        Args:
            question: Query text
            top_k: Number of results to return
            
        Returns:
            Tuple of (context_docs, context_scores)
            context_scores are converted from distances to similarity (1 - distance)
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        
        context_docs = results['documents'][0] if results['documents'] else []
        context_scores = results['distances'][0] if results['distances'] else []
        
        # Convert distances to similarity scores (1 - distance)
        context_scores = [max(0.0, 1.0 - d) for d in context_scores]
        
        return context_docs, context_scores

