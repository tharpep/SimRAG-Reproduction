"""
RAG System Tests
Tests the RAG system with domain documents using pytest
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rag.rag_setup import BasicRAG
from rag.document_ingester import DocumentIngester
from config import RAGConfig


class TestRAGSystem:
    """Test class for RAG system functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method"""
        self.config = RAGConfig(
            use_ollama=True,
            use_persistent=False,  # Use in-memory storage for tests to avoid lock conflicts
            use_laptop=True  # Use laptop model (qwen3:1.7b)
        )
        
        self.documents_folder = str(project_root / "data" / "documents")
    
    def test_rag_initialization(self):
        """Test RAG system initialization"""
        print("\n=== Testing RAG Initialization ===")
        
        rag = BasicRAG(
            collection_name="test_docs_init",
            use_persistent=self.config.use_persistent
        )
        
        assert rag is not None
        assert rag.collection_name == "test_docs_init"
        print("[OK] RAG system initialized successfully")
    
    def test_document_ingestion(self):
        """Test document ingestion process"""
        print("\n=== Testing Document Ingestion ===")
        
        rag = BasicRAG(
            collection_name="test_docs_ingest",
            use_persistent=self.config.use_persistent
        )
        
        ingester = DocumentIngester(rag)
        
        # Check if documents folder exists and has files
        if not os.path.exists(self.documents_folder):
            pytest.skip(f"Documents folder not found: {self.documents_folder}. Create it with test documents to run this test.")
        
        supported_files = ingester.get_supported_files(self.documents_folder)
        
        if len(supported_files) == 0:
            pytest.skip(f"No markdown files found in documents folder: {self.documents_folder}")
        
        print(f"[OK] Found {len(supported_files)} markdown files")
        
        # Ingest documents
        result = ingester.ingest_folder(self.documents_folder)
        
        assert result["success"], f"Document ingestion failed: {result.get('error', 'Unknown error')}"
        assert result["processed"] > 0, "No files were processed"
        print(f"[OK] Processed {result['processed']} files, {result['failed']} failed")
        print(f"[OK] Total chunks indexed: {sum(f['chunks'] for f in result['files'])}")
    
    def test_vector_search(self):
        """Test vector search functionality"""
        print("\n=== Testing Vector Search ===")
        
        rag = BasicRAG(
            collection_name="test_docs_search",
            use_persistent=self.config.use_persistent
        )
        
        # Use sample documents if real documents folder doesn't exist
        if not os.path.exists(self.documents_folder):
            sample_docs = [
                "Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers.",
                "Python is a versatile programming language commonly used for web development, data science, and automation.",
                "DevOps is a set of practices that combines software development and IT operations to shorten the development lifecycle."
            ]
            rag.add_documents(sample_docs)
        else:
            ingester = DocumentIngester(rag)
            ingester.ingest_folder(self.documents_folder)
        
        # Test search
        query = "What is Docker?"
        results = rag.search(query, limit=3)
        
        assert len(results) > 0, "No search results returned"
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results), "Invalid result format"
        
        print(f"[OK] Search returned {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  [{i}] (score: {score:.3f}) {doc[:100]}...")
    
    def test_rag_query_without_llm(self):
        """Test RAG query functionality (without LLM - just retrieval)"""
        print("\n=== Testing RAG Query (Retrieval Only) ===")
        
        rag = BasicRAG(
            collection_name="test_docs_query",
            use_persistent=self.config.use_persistent
        )
        
        # Use sample documents if real documents folder doesn't exist
        if not os.path.exists(self.documents_folder):
            sample_docs = [
                "Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers.",
                "Python is a versatile programming language commonly used for web development, data science, and automation.",
                "DevOps is a set of practices that combines software development and IT operations to shorten the development lifecycle."
            ]
            rag.add_documents(sample_docs)
        else:
            ingester = DocumentIngester(rag)
            ingester.ingest_folder(self.documents_folder)
        
        # Test retrieval (without LLM generation)
        query = "What is Docker and how does it work?"
        retrieved_docs = rag.search(query, limit=3)
        
        assert len(retrieved_docs) > 0, "No documents retrieved"
        
        # Build context manually (simulating what RAG would do)
        context = "\n\n".join([doc for doc, score in retrieved_docs])
        assert len(context) > 0, "No context built from retrieved documents"
        
        print(f"[OK] Retrieved {len(retrieved_docs)} relevant documents")
        print(f"[OK] Built context of {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
    
    def test_collection_stats(self):
        """Test collection statistics"""
        print("\n=== Testing Collection Stats ===")
        
        rag = BasicRAG(
            collection_name="test_docs_stats",
            use_persistent=self.config.use_persistent
        )
        
        # Use sample documents if real documents folder doesn't exist
        if not os.path.exists(self.documents_folder):
            sample_docs = [
                "Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers.",
                "Python is a versatile programming language commonly used for web development, data science, and automation."
            ]
            rag.add_documents(sample_docs)
        else:
            ingester = DocumentIngester(rag)
            ingester.ingest_folder(self.documents_folder)
        
        stats = rag.get_stats()
        
        assert "points_count" in stats or "document_count" in stats, "Missing points_count/document_count in stats"
        doc_count = stats.get("points_count") or stats.get("document_count", 0)
        assert doc_count > 0, "No points in collection"
        assert "vector_size" in stats, "Missing vector_size in stats"
        
        print(f"[OK] Collection stats: {stats}")


def test_quick_demo():
    """Quick demo with sample documents"""
    print("\n=== Quick Demo Test ===")
    
    sample_docs = [
        "Docker is a containerization platform that allows you to package applications and their dependencies into lightweight, portable containers.",
        "Python is a versatile programming language commonly used for web development, data science, and automation.",
        "DevOps is a set of practices that combines software development and IT operations to shorten the development lifecycle.",
        "Binary search is an efficient algorithm for finding an item in a sorted array by repeatedly dividing the search space in half.",
        "Docker containers provide isolated environments for running applications consistently across different systems."
    ]
    
    rag = BasicRAG(use_persistent=False)  # In-memory for demo
    count = rag.add_documents(sample_docs)
    
    assert count == len(sample_docs), f"Expected {len(sample_docs)} documents, got {count}"
        
    # Test search
    query = "What is Docker?"
    results = rag.search(query, limit=2)
    
    assert len(results) > 0, "No search results in demo"
    
    print(f"[OK] Demo: Added {count} documents, retrieved {len(results)} results")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  [{i}] (score: {score:.3f}) {doc[:100]}...")


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v", "-s"])
