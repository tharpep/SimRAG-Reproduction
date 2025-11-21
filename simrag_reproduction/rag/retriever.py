"""
Document Retriever
Handles embedding generation and document retrieval logic
"""

from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from typing import List, Tuple
import uuid


class DocumentRetriever:
    """Handles document embedding and retrieval operations"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize document retriever
        
        Args:
            model_name: Name of the sentence transformer model
            
        Raises:
            ValueError: If model_name is invalid
            Exception: If model loading fails
        """
        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got: {model_name}")
        
        try:
            self.model_name = model_name
            self.retriever = SentenceTransformer(model_name)
            self.embedding_dim = self.retriever.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")
    
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Encode documents into embeddings
        
        Args:
            documents: List of text documents
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If documents list is empty or invalid
            Exception: If encoding fails
        """
        if not documents:
            raise ValueError("documents list cannot be empty")
        if not isinstance(documents, list):
            raise ValueError(f"documents must be a list, got: {type(documents)}")
        
        try:
            embeddings = self.retriever.encode(documents)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            raise RuntimeError(f"Failed to encode documents: {e}")
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a query into an embedding
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If encoding fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")
        
        try:
            embedding = self.retriever.encode(query)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {e}")
    
    def create_points(self, documents: List[str], embeddings: List[List[float]], 
                     start_doc_id: int = 0) -> List[PointStruct]:
        """
        Create Qdrant points from documents and embeddings
        
        Args:
            documents: List of text documents
            embeddings: List of embedding vectors
            start_doc_id: Starting document ID for indexing
            
        Returns:
            List of Qdrant PointStruct objects
            
        Raises:
            ValueError: If documents and embeddings lists don't match or are empty
        """
        if not documents:
            raise ValueError("documents list cannot be empty")
        if not embeddings:
            raise ValueError("embeddings list cannot be empty")
        if len(documents) != len(embeddings):
            raise ValueError(f"documents ({len(documents)}) and embeddings ({len(embeddings)}) lists must have the same length")
        
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": doc, 
                    "doc_id": start_doc_id + idx,
                    "chunk_id": idx
                }
            )
            points.append(point)
        return points
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension
        
        Returns:
            Embedding dimension size
        """
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": self.retriever.max_seq_length
        }
