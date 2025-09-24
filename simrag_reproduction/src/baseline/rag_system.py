"""
Baseline RAG System Implementation

Implements a vanilla RAG system with dense retriever and language model generator.
"""

import torch
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
from loguru import logger

# from ..utils.logging import setup_logging


class BaselineRAGSystem:
    """
    Baseline RAG system with dense retriever and causal language model.
    
    This implements the vanilla RAG approach as described in the original RAG paper,
    serving as our baseline for comparison with SimRAG.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the baseline RAG system.
        
        Args:
            config: Configuration dictionary containing model and system parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._setup_retriever()
        self._setup_generator()
        self._setup_index()
        
        logger.info(f"Baseline RAG system initialized on {self.device}")
    
    def _setup_retriever(self):
        """Initialize the dense retriever model."""
        retriever_config = self.config["model"]["retriever"]
        self.retriever = SentenceTransformer(retriever_config["name"])
        self.retriever.to(self.device)
        logger.info(f"Retriever loaded: {retriever_config['name']}")
    
    def _setup_generator(self):
        """Initialize the language model generator."""
        generator_config = self.config["model"]["generator"]
        self.tokenizer = AutoTokenizer.from_pretrained(generator_config["name"])
        self.generator = AutoModelForCausalLM.from_pretrained(
            generator_config["name"],
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Generator loaded: {generator_config['name']}")
    
    def _setup_index(self):
        """Initialize the FAISS index for document retrieval."""
        self.index = None
        self.documents = []
        self.document_embeddings = None
        logger.info("FAISS index initialized")
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the retrieval index.
        
        Args:
            documents: List of document texts to index
        """
        logger.info(f"Adding {len(documents)} documents to index")
        
        # Generate embeddings
        embeddings = self.retriever.encode(
            documents, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_np = embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Store documents and embeddings
        self.documents = documents
        self.document_embeddings = embeddings_np
        
        logger.info(f"Index built with {self.index.ntotal} documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: Input query string
            k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document text and relevance scores
        """
        if self.index is None:
            raise ValueError("No documents have been added to the index")
        
        # Generate query embedding
        query_embedding = self.retriever.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)
        
        # Search index
        scores, indices = self.index.search(query_embedding_np, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    "text": self.documents[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        
        return results
    
    def generate(self, query: str, context: str, max_length: int = 512) -> str:
        """
        Generate an answer given a query and retrieved context.
        
        Args:
            query: Input query
            context: Retrieved context documents
            max_length: Maximum generation length
            
        Returns:
            Generated answer text
        """
        # Create input prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 100,
                temperature=self.config["model"]["generator"]["temperature"],
                do_sample=self.config["model"]["generator"]["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Answer a query using retrieval-augmented generation.
        
        Args:
            query: Input query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer, retrieved documents, and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k)
        
        # Combine context
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        
        # Generate answer
        answer = self.generate(query, context)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context": context
        }
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluate the RAG system on test data.
        
        Args:
            test_data: List of dictionaries with 'question' and 'answer' keys
            
        Returns:
            Dictionary of evaluation metrics
        """
        from ..evaluation.metrics import compute_metrics
        
        predictions = []
        references = []
        
        for item in test_data:
            result = self.answer(item["question"])
            predictions.append(result["answer"])
            references.append(item["answer"])
        
        return compute_metrics(predictions, references)
