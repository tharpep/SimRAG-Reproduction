"""
SimRAG Synthetic QA Generation
Stage 2: Generate synthetic question-answer pairs from domain documents
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base import SimRAGBase
from rag.rag_setup import BasicRAG
from ai_providers.gateway import AIGateway


class SyntheticQAGeneration(SimRAGBase):
    """SimRAG Stage 2: Synthetic QA generation from domain documents"""
    
    def __init__(self, model_name: str = "llama3.2:1b", config=None, stage_1_model_path: Optional[str] = None):
        """
        Initialize synthetic QA generator
        
        Args:
            model_name: Model to use for QA generation
            config: TuningConfig instance
            stage_1_model_path: Path to Stage 1 model (if available)
        """
        super().__init__(model_name, config)
        self.stage_1_model_path = stage_1_model_path
        self.gateway = AIGateway()
        
        # Initialize RAG system (will use Stage 1 model if available)
        self.rag_system = self._initialize_rag_system()
        
        print(f"Synthetic QA Generator initialized")
        if stage_1_model_path:
            print(f"Using Stage 1 model: {stage_1_model_path}")
        else:
            print("Using vanilla RAG system")
    
    def _initialize_rag_system(self) -> BasicRAG:
        """Initialize RAG system, using Stage 1 model if available"""
        # For now, use vanilla RAG
        # TODO: Integrate Stage 1 model for improved QA generation
        return BasicRAG()
    
    def generate_questions_from_document(self, document: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions from a single document using LLM
        
        Args:
            document: Source document text
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        prompt = f"""Given this document, generate {num_questions} diverse questions that test understanding:

Document: {document[:500]}...

Generate questions that cover:
1. Key concepts and definitions
2. Process explanations  
3. Practical applications
4. Technical details

Questions:"""
        
        try:
            response = self.gateway.chat(prompt)
            questions = self._parse_questions(response)
            return questions[:num_questions]  # Limit to requested number
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from LLM response"""
        questions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.startswith(('What', 'How', 'Why', 'When', 'Where', 'Explain', 'Describe'))):
                # Clean up the question
                question = line.replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').strip()
                if question and len(question) > 10:  # Filter out very short responses
                    questions.append(question)
        
        return questions
    
    def create_qa_pairs_from_documents(self, documents: List[str], 
                                     questions_per_doc: int = 2) -> List[Dict[str, Any]]:
        """
        Create synthetic QA pairs from a list of documents
        
        Args:
            documents: List of document texts
            questions_per_doc: Number of questions to generate per document
            
        Returns:
            List of QA pairs with metadata
        """
        print(f"=== Generating Synthetic QA Pairs ===")
        print(f"Processing {len(documents)} documents...")
        
        qa_pairs = []
        
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}...")
            
            # Generate questions from document
            questions = self.generate_questions_from_document(doc, questions_per_doc)
            
            for question in questions:
                try:
                    # Generate answer using RAG system
                    answer, context_docs, context_scores = self.rag_system.query(question)
                    
                    # Create QA pair
                    qa_pair = {
                        "question": question,
                        "answer": answer,
                        "context": doc[:200] + "...",  # Truncate for storage
                        "context_docs": context_docs,
                        "context_scores": context_scores,
                        "source_doc_index": i
                    }
                    
                    qa_pairs.append(qa_pair)
                    print(f"  Generated Q: {question[:50]}...")
                    
                except Exception as e:
                    print(f"  Error processing question: {e}")
                    continue
        
        print(f"✅ Generated {len(qa_pairs)} synthetic QA pairs")
        return qa_pairs
    
    def filter_high_quality_qa_pairs(self, qa_pairs: List[Dict[str, Any]], 
                                   min_context_score: float = 0.7,
                                   min_answer_length: int = 20) -> List[Dict[str, Any]]:
        """
        Filter QA pairs based on quality criteria
        
        Args:
            qa_pairs: List of QA pairs to filter
            min_context_score: Minimum context similarity score
            min_answer_length: Minimum answer length in characters
            
        Returns:
            Filtered list of high-quality QA pairs
        """
        print(f"=== Filtering High-Quality QA Pairs ===")
        print(f"Original pairs: {len(qa_pairs)}")
        
        filtered_pairs = []
        
        for qa_pair in qa_pairs:
            # Check context scores
            if qa_pair.get("context_scores"):
                avg_score = sum(qa_pair["context_scores"]) / len(qa_pair["context_scores"])
                if avg_score < min_context_score:
                    continue
            
            # Check answer quality
            if len(qa_pair["answer"]) < min_answer_length:
                continue
            
            # Check for meaningful content
            if qa_pair["answer"].lower() in ["no relevant documents found", "i don't know", "cannot answer"]:
                continue
            
            filtered_pairs.append(qa_pair)
        
        print(f"Filtered pairs: {len(filtered_pairs)}")
        print(f"Quality retention: {len(filtered_pairs)/len(qa_pairs)*100:.1f}%")
        
        return filtered_pairs
    
    def prepare_training_data(self, qa_pairs: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare QA pairs for training in instruction format
        
        Args:
            qa_pairs: List of QA pairs
            
        Returns:
            List of training examples in instruction format
        """
        training_examples = []
        
        for qa_pair in qa_pairs:
            # Format as instruction-following example
            example = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
            training_examples.append(example)
        
        print(f"Prepared {len(training_examples)} training examples")
        return training_examples
    
    def generate_synthetic_dataset(self, documents: List[str], 
                                 questions_per_doc: int = 2,
                                 min_context_score: float = 0.7) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset for Stage 2 training
        
        Args:
            documents: List of domain documents
            questions_per_doc: Questions to generate per document
            min_context_score: Minimum context similarity threshold
            
        Returns:
            Complete synthetic dataset with metadata
        """
        print("=== SimRAG Stage 2: Synthetic Dataset Generation ===")
        
        # Generate QA pairs
        qa_pairs = self.create_qa_pairs_from_documents(documents, questions_per_doc)
        
        # Filter for quality
        high_quality_pairs = self.filter_high_quality_qa_pairs(qa_pairs, min_context_score)
        
        # Prepare training data
        training_data = self.prepare_training_data(high_quality_pairs)
        
        # Create dataset summary
        dataset_info = {
            "total_documents": len(documents),
            "total_qa_pairs": len(qa_pairs),
            "high_quality_pairs": len(high_quality_pairs),
            "training_examples": len(training_data),
            "quality_retention": len(high_quality_pairs) / len(qa_pairs) * 100 if qa_pairs else 0,
            "avg_context_score": sum(sum(pair.get("context_scores", [0])) for pair in high_quality_pairs) / len(high_quality_pairs) if high_quality_pairs else 0
        }
        
        print(f"✅ Synthetic dataset generated!")
        print(f"   Documents: {dataset_info['total_documents']}")
        print(f"   QA pairs: {dataset_info['total_qa_pairs']}")
        print(f"   High quality: {dataset_info['high_quality_pairs']}")
        print(f"   Training examples: {dataset_info['training_examples']}")
        print(f"   Quality retention: {dataset_info['quality_retention']:.1f}%")
        
        return {
            "qa_pairs": high_quality_pairs,
            "training_data": training_data,
            "dataset_info": dataset_info
        }


def main():
    """Demo of synthetic QA generation"""
    print("=== SimRAG Stage 2: Synthetic QA Generation Demo ===\n")
    
    # Initialize synthetic QA generator
    qa_generator = SyntheticQAGeneration()
    
    # Sample domain documents
    sample_documents = [
        "Docker is a containerization platform that allows developers to package applications and their dependencies into lightweight, portable containers. Containers provide consistent environments across development, testing, and production.",
        
        "Binary search is an efficient algorithm for finding elements in sorted arrays. It works by repeatedly dividing the search interval in half, comparing the target value with the middle element, and eliminating half of the remaining elements each time.",
        
        "DevOps is a set of practices that combines software development and IT operations. It aims to shorten the systems development life cycle and provide continuous delivery with high software quality through automation, collaboration, and shared responsibility."
    ]
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset = qa_generator.generate_synthetic_dataset(
        documents=sample_documents,
        questions_per_doc=2,
        min_context_score=0.6
    )
    
    # Show sample results
    print("\nSample generated QA pairs:")
    for i, qa_pair in enumerate(dataset["qa_pairs"][:3], 1):
        print(f"\nPair {i}:")
        print(f"  Q: {qa_pair['question']}")
        print(f"  A: {qa_pair['answer'][:100]}...")
        print(f"  Context scores: {qa_pair['context_scores']}")
    
    print(f"\n✅ Demo completed!")
    print(f"Generated {dataset['dataset_info']['training_examples']} training examples")


if __name__ == "__main__":
    main()
