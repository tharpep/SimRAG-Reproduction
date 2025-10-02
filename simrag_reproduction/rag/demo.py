"""
RAG Demo System
Interactive and automated testing for RAG functionality
"""

import os
from typing import List, Optional
from .rag_setup import BasicRAG
from .document_ingester import DocumentIngester
from config import RAGConfig


class RAGDemo:
    """Demo system for testing RAG functionality"""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize demo system
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.rag = None
        self.ingester = None
        
        # Set environment variables for AI providers
        env_vars = config.to_env_dict()
        for key, value in env_vars.items():
            if value:  # Only set non-empty values
                os.environ[key] = value
    
    def initialize(self):
        """Initialize RAG system and components"""
        print("Initializing RAG system...")
        
        # Initialize RAG
        self.rag = BasicRAG(
            collection_name=self.config.collection_name,
            use_persistent=self.config.use_persistent
        )
        
        # Initialize document ingester
        self.ingester = DocumentIngester(self.rag)
        
        print("RAG system initialized successfully!")
    
    def load_demo_documents(self):
        """Load demo documents if enabled"""
        if not self.config.demo_documents or not self.rag:
            return
        
        print("\nLoading demo documents...")
        
        demo_docs = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
            "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            "Natural language processing (NLP) combines computational linguistics with machine learning to help computers understand human language.",
            "Computer vision enables machines to interpret and understand visual information from images and videos.",
            "Reinforcement learning is where agents learn optimal behavior by interacting with an environment and receiving rewards.",
            "Qdrant is a vector database that provides fast similarity search and supports metadata filtering.",
            "RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation for more accurate answers.",
            "Sentence transformers convert text into dense vector representations for semantic similarity search."
        ]
        
        count = self.rag.add_documents(demo_docs)
        print(f"Loaded {count} demo documents")
    
    def load_user_documents(self):
        """Load documents from data/documents folder"""
        if not self.ingester:
            return
            
        documents_folder = "./data/documents"
        
        if not os.path.exists(documents_folder):
            print(f"Documents folder not found: {documents_folder}")
            return
        
        print(f"\nLoading documents from {documents_folder}...")
        result = self.ingester.ingest_folder(documents_folder)
        
        if result["success"]:
            print(f"Processed: {result['processed']} files")
            print(f"Failed: {result['failed']} files")
            if result["errors"]:
                print(f"Errors: {result['errors']}")
        else:
            print(f"Error: {result['error']}")
    
    def show_stats(self):
        """Display system statistics"""
        if not self.rag:
            return
            
        print("\n=== System Statistics ===")
        stats = self.rag.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    def run_demo_queries(self):
        """Run predefined demo queries"""
        print("\n=== Running Demo Queries ===")
        
        demo_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is RAG?",
            "What is Qdrant used for?"
        ]
        
        for query in demo_queries:
            print(f"\nQuery: {query}")
            if self.rag:
                answer = self.rag.query(query, context_limit=self.config.context_limit)
                print(f"Answer: {answer}")
                
                # Show retrieved context
                retrieved = self.rag.search(query, limit=2)
                print(f"Retrieved {len(retrieved)} documents")
                for i, (doc, score) in enumerate(retrieved, 1):
                    print(f"  [{i}] (score: {score:.3f}) {doc[:80]}...")
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n=== Interactive Mode ===")
        print("Type your questions (or 'quit' to exit):")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print(f"\nQuery: {query}")
                if self.rag:
                    answer = self.rag.query(query, context_limit=self.config.context_limit)
                    print(f"Answer: {answer}")
                    
                    # Show retrieved context
                    retrieved = self.rag.search(query, limit=2)
                    print(f"\nRetrieved {len(retrieved)} documents:")
                    for i, (doc, score) in enumerate(retrieved, 1):
                        print(f"  [{i}] (score: {score:.3f}) {doc[:100]}...")
                    
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run_full_demo(self):
        """Run complete demo with all features"""
        self.config.print_config()
        
        # Initialize system
        self.initialize()
        
        # Load documents
        self.load_demo_documents()
        self.load_user_documents()
        
        # Show stats
        self.show_stats()
        
        # Run demo or interactive mode
        if self.config.interactive_mode:
            self.interactive_mode()
        else:
            self.run_demo_queries()
    
    def run_quick_test(self):
        """Run a quick test of the system"""
        print("Running quick test...")
        
        self.initialize()
        self.load_demo_documents()
        
        # Test a simple query
        query = "What is machine learning?"
        if self.rag:
            answer = self.rag.query(query)
            print(f"\nTest Query: {query}")
            print(f"Answer: {answer}")
        
        # Show stats
        self.show_stats()


def main():
    """Main demo function"""
    from config import RAGConfig
    
    # Load configuration
    config = RAGConfig.from_env()
    
    # Create and run demo
    demo = RAGDemo(config)
    demo.run_full_demo()


if __name__ == "__main__":
    main()
