"""
Simple RAG Demo
Basic testing for RAG functionality
"""

import os
import time
from pathlib import Path
from .rag_setup import BasicRAG
from .document_ingester import DocumentIngester
from config import get_rag_config
from logging_config import log_rag_result


def run_rag_demo(mode="automated"):
    """Run RAG demo in automated or interactive mode"""
    print("=== RAG Demo ===")
    
    # Get configuration
    config = get_rag_config()
    print(f"Using model: {config.model_name}")
    print(f"Provider: {'Ollama' if config.use_ollama else 'Purdue API'}")
    
    try:
        # Initialize RAG system (uses config defaults automatically)
        print("\nInitializing RAG system...")
        rag = BasicRAG()
        
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
                print(f"⚠️  Connection test failed:")
                print(f"Raw error: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                print("Continuing anyway...")
            
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question:
                    print("RAG: Thinking...")
                    try:
                        start_time = time.time()
                        answer = rag.query(question)
                        response_time = time.time() - start_time
                        
                        print(f"RAG: {answer}")
                        
                        # Log the result
                        log_rag_result(
                            question=question,
                            answer=answer,
                            response_time=response_time,
                            model_name=config.model_name,
                            provider="Ollama" if config.use_ollama else "Purdue API",
                            context_length=len(answer)
                        )
                        
                    except Exception as e:
                        print(f"❌ Error:")
                        print(f"Raw error: {e}")
                        import traceback
                        print(f"Full traceback:\n{traceback.format_exc()}")
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
                    start_time = time.time()
                    answer = rag.query(query)
                    response_time = time.time() - start_time
                    
                    print(f"A: {answer[:200]}...")  # Truncate long answers
                    
                    # Log the result
                    log_rag_result(
                        question=query,
                        answer=answer,
                        response_time=response_time,
                        model_name=config.model_name,
                        provider="Ollama" if config.use_ollama else "Purdue API",
                        context_length=len(answer)  # Simplified context length
                    )
                    
                except Exception as e:
                    print(f"Error:")
                    print(f"Raw error: {e}")
                    import traceback
                    print(f"Full traceback:\n{traceback.format_exc()}")
        
        print("\n✅ RAG demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed:")
        print(f"Raw error: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")


def main():
    """Main function for direct execution"""
    run_rag_demo("automated")


if __name__ == "__main__":
    main()