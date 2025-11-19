"""
RAG System Orchestrator
Coordinates vector storage, retrieval, and generation components
"""

from ai_providers.gateway import AIGateway
from .vector_store import VectorStore
from .retriever import DocumentRetriever
from ..config import get_rag_config
from typing import Optional


class BasicRAG:
    """RAG system that orchestrates vector storage, retrieval, and generation"""
    
    def __init__(self, collection_name=None, use_persistent=None, force_provider=None, model_path=None):
        """
        Initialize RAG system
        
        Args:
            collection_name: Name for Qdrant collection (uses config default if None)
            use_persistent: If True, use persistent Qdrant storage (uses config default if None)
            force_provider: Force provider to use ("purdue", "ollama", or "huggingface"). If None, uses config default.
            model_path: Path to fine-tuned HuggingFace model (only used if force_provider="huggingface")
        """
        self.config = get_rag_config()
        self.collection_name = collection_name or self.config.collection_name
        
        # Initialize components
        gateway_config = {}
        
        # If model_path is provided, use HuggingFace client
        if model_path:
            gateway_config["huggingface"] = {"model_path": model_path}
            force_provider = "huggingface"
        
        self.gateway = AIGateway(gateway_config)
        self.force_provider = force_provider
        
        # Override gateway chat if provider is forced
        if force_provider:
            original_chat = self.gateway.chat
            def forced_chat(message: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
                return original_chat(message, provider=force_provider, model=model, force_provider=True)
            self.gateway.chat = forced_chat
        
        self.vector_store = VectorStore(use_persistent=use_persistent if use_persistent is not None else self.config.use_persistent)
        self.retriever = DocumentRetriever()
        
        # Setup collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup the vector collection"""
        embedding_dim = self.retriever.get_embedding_dimension()
        success = self.vector_store.setup_collection(self.collection_name, embedding_dim)
        if not success:
            raise Exception(f"Failed to setup collection: {self.collection_name}")
    
    def add_documents(self, documents):
        """
        Add documents to the vector database
        
        Args:
            documents: List of text documents to index
            
        Returns:
            Number of documents added
        """
        # Create embeddings
        embeddings = self.retriever.encode_documents(documents)
        
        # Create points for vector store
        points = self.retriever.create_points(documents, embeddings)
        
        # Add to vector store
        return self.vector_store.add_points(self.collection_name, points)
    
    def search(self, query, limit=None):
        """
        Search for relevant documents
        
        Args:
            query: Search query
            limit: Number of results to return (uses config default if None)
            
        Returns:
            List of (text, score) tuples
        """
        # Use config default if limit not specified
        if limit is None:
            limit = self.config.top_k
            
        # Create query embedding
        query_embedding = self.retriever.encode_query(query)
        
        # Search vector store
        return self.vector_store.search(self.collection_name, query_embedding, limit)
    
    def query(self, question, context_limit=None):
        """
        Answer a question using RAG
        
        Args:
            question: Question to answer
            context_limit: Number of documents to retrieve for context (uses config default if None)
            
        Returns:
            Answer string
        """
        # Use config default if context_limit not specified
        if context_limit is None:
            context_limit = self.config.top_k
            
        # Retrieve relevant documents
        retrieved_docs = self.search(question, limit=context_limit)
        
        if not retrieved_docs:
            return "No relevant documents found.", [], []
        
        # Build RAG context from retrieved documents
        rag_context = "\n\n".join([doc for doc, _ in retrieved_docs])
        
        # Create prompt
        prompt = f"""Context:
{rag_context}

Question: {question}

Answer:"""
        
        # Generate answer
        answer = self.gateway.chat(prompt)
        
        # Return answer along with context details for logging
        context_docs = [doc for doc, _ in retrieved_docs]
        context_scores = [score for _, score in retrieved_docs]
        
        return answer, context_docs, context_scores
    
    def get_stats(self):
        """Get collection statistics"""
        stats = self.vector_store.get_collection_stats(self.collection_name)
        if "error" not in stats:
            stats.update({
                "collection_name": stats.get("name", self.collection_name),
                "document_count": stats.get("points_count", 0),
                "vector_size": self.retriever.get_embedding_dimension(),
                "distance": "cosine",
                "model_info": self.retriever.get_model_info()
            })
        else:
            # If there's an error, still provide basic info
            stats.update({
                "collection_name": self.collection_name,
                "document_count": 0,
                "vector_size": self.retriever.get_embedding_dimension(),
                "distance": "cosine",
                "model_info": self.retriever.get_model_info()
            })
        return stats
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            embedding_dim = self.retriever.get_embedding_dimension()
            
            # Clean up old collections first
            self.vector_store.cleanup_old_collections([self.collection_name])
            
            # Clear the main collection
            self.vector_store.clear_collection(self.collection_name, embedding_dim)
            return {"success": True, "message": f"Cleared collection {self.collection_name}"}
        except Exception as e:
            return {"error": f"Failed to clear collection: {str(e)}"}
