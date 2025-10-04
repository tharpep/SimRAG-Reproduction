"""
Simple RAG Demo
Basic testing for RAG functionality
"""

import os
from pathlib import Path
from .rag_setup import BasicRAG
from .document_ingester import DocumentIngester
from config import get_rag_config


def run_rag_demo(mode="automated"):
    """Run RAG demo in automated or interactive mode"""
    print("=== RAG Demo ===")
    
    # Get configuration
    config = get_rag_config()
    print(f"Using model: {config.model_name}")
    print(f"Provider: {'Ollama' if config.use_ollama else 'Purdue API'}")
    
    # Set environment variables BEFORE initializing RAG system
    os.environ["USE_LAPTOP"] = "true" if config.use_laptop else "false"
    os.environ["USE_OLLAMA"] = "true" if config.use_ollama else "false"
    os.environ["MODEL_NAME"] = config.model_name
    os.environ["USE_PERSISTENT"] = "true" if config.use_persistent else "false"
    os.environ["COLLECTION_NAME"] = config.collection_name
    
    try:
        # Initialize RAG system (after env vars are set)
        print("\nInitializing RAG system...")
        rag = BasicRAG(
            collection_name=config.collection_name,
            use_persistent=config.use_persistent
        )
        
        # Load documents
        print("Loading documents...")
        documents_folder = Path(__file__).parent.parent / "data" / "documents"
        
        if documents_folder.exists():
            ingester = DocumentIngester(rag)
            result = ingester.ingest_folder(str(documents_folder))
            
            if result["success"]:
                print(f"✅ Loaded {result['processed']} documents")
            else:
                print(f"❌ Error loading documents: {result['error']}")
                return
        else:
            print(f"❌ Documents folder not found: {documents_folder}")
            return
        
        # Show stats
        stats = rag.get_stats()
        print(f"\nCollection: {stats['collection_name']}")
        print(f"Documents: {stats['document_count']}")
        
        if mode == "interactive":
            # Interactive mode
            print("\n=== Interactive Mode ===")
            print("Ask questions (type 'quit' to exit):")
            
            # Test connection first
            try:
                test_answer = rag.query("Hello, are you working?")
                print(f"✅ Connection test successful: {test_answer[:50]}...")
            except Exception as e:
                print(f"⚠️  Connection test failed: {e}")
                if config.use_ollama:
                    print("💡 Make sure Ollama is running: 'ollama serve'")
                else:
                    print("💡 Check your Purdue API key configuration")
                print("Continuing anyway...")
            
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question:
                    print("RAG: Thinking...")
                    try:
                        answer = rag.query(question)
                        print(f"RAG: {answer}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        if config.use_ollama:
                            print("💡 Tip: Make sure Ollama is running: 'ollama serve'")
                        else:
                            print("💡 Tip: Check your Purdue API key configuration")
        else:
            # Automated mode - run test queries
            print("\n=== Automated Mode ===")
            test_queries = [
                "What is Docker?",
                "How does binary search work?",
                "What is DevOps?",
                "Explain Python programming"
            ]
            
            for query in test_queries:
                print(f"\nQ: {query}")
                try:
                    answer = rag.query(query)
                    print(f"A: {answer[:200]}...")  # Truncate long answers
                except Exception as e:
                    print(f"Error: {e}")
        
        print("\n✅ RAG demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main function for direct execution"""
    run_rag_demo("automated")


if __name__ == "__main__":
    main()