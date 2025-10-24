"""
SimRAG Stage 1: Instruction-Oriented Fine-tuning
Fine-tunes LLM on instruction-following, QA, and search data
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tuning.basic_tuning import BasicTuner
from tuning.model_registry import get_model_registry
from config import get_tuning_config
from rag.rag_setup import BasicRAG


class SimRAGStageI:
    """SimRAG Stage 1: Retrieval-oriented fine-tuning"""
    
    def __init__(self, model_name: str = "llama3.2:1b", config=None):
        """
        Initialize SimRAG Stage 1
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance
        """
        self.model_name = model_name
        self.config = config or get_tuning_config()
        self.tuner = BasicTuner(model_name, config=self.config)
        self.registry = get_model_registry(self.config)
        
        print(f"SimRAG Stage I initialized with model: {model_name}")
    
    def prepare_instruction_data(self) -> List[str]:
        """
        Prepare instruction-following and QA datasets for Stage 1
        
        Returns:
            List of instruction-following text examples
        """
        # Create instruction-following examples from domain knowledge
        instruction_examples = [
            "Question: What is Docker?\nAnswer: Docker is a platform for developing, shipping, and running applications in containers. It allows you to package your application and its dependencies into a lightweight, portable container.",
            
            "Question: How does binary search work?\nAnswer: Binary search works by repeatedly dividing the search interval in half. It compares the target value with the middle element and eliminates half of the remaining elements each time.",
            
            "Question: What is DevOps?\nAnswer: DevOps is a set of practices that combines software development and IT operations. It aims to shorten the systems development life cycle and provide continuous delivery with high software quality.",
            
            "Question: Explain Python programming concepts.\nAnswer: Python is a high-level programming language known for its simple syntax and readability. Key concepts include variables, data types, control structures, functions, and object-oriented programming.",
            
            "Question: What are the key principles of machine learning?\nAnswer: Machine learning principles include supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).",
            
            "Question: How do neural networks work?\nAnswer: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and learn patterns through training on data.",
            
            "Question: What is the difference between supervised and unsupervised learning?\nAnswer: Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds hidden patterns in data without labeled examples.",
            
            "Question: Explain the concept of overfitting in machine learning.\nAnswer: Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, leading to poor performance on new, unseen data.",
            
            "Question: What is version control and why is it important?\nAnswer: Version control is a system that tracks changes to files over time. It's important for collaboration, maintaining code history, and enabling rollbacks to previous versions.",
            
            "Question: How does a hash table work?\nAnswer: A hash table uses a hash function to map keys to array indices, allowing for average O(1) time complexity for insertions, deletions, and lookups."
        ]
        
        print(f"Prepared {len(instruction_examples)} instruction-following examples")
        return instruction_examples
    
    def train_stage_i(self, notes: str = "SimRAG Stage I - Instruction Following") -> Optional[Any]:
        """
        Train model on instruction-following data (Stage I)
        
        Args:
            notes: Training notes for versioning
            
        Returns:
            Model version object if successful
        """
        print("=== SimRAG Stage I Training ===")
        
        # Load model
        print("Loading model...")
        self.tuner.load_model()
        
        # Prepare instruction data
        print("Preparing instruction-following data...")
        instruction_data = self.prepare_instruction_data()
        
        # Prepare training dataset
        print("Preparing training dataset...")
        train_dataset = self.tuner.prepare_data(instruction_data, max_length=self.config.max_length)
        
        # Setup trainer for Stage I
        print("Setting up trainer for Stage I...")
        output_dir = f"./tuned_models/llama_1b/stage_i"
        self.tuner.setup_trainer(
            train_dataset=train_dataset,
            output_dir=output_dir,
            num_epochs=self.config.optimized_num_epochs,
            batch_size=self.config.optimized_batch_size,
            learning_rate=self.config.learning_rate
        )
        
        # Train the model
        print("Starting Stage I training...")
        new_version = self.tuner.train(notes=notes)
        
        if new_version:
            print(f"✅ Stage I training completed!")
            print(f"Version: {new_version.version}")
            print(f"Training time: {new_version.training_time_seconds:.1f}s")
            if new_version.final_loss:
                print(f"Final loss: {new_version.final_loss:.4f}")
        
        return new_version
    
    def test_stage_i_performance(self, rag_system: BasicRAG, test_questions: List[str]) -> Dict[str, Any]:
        """
        Test Stage I model performance on RAG tasks
        
        Args:
            rag_system: RAG system to test
            test_questions: List of test questions
            
        Returns:
            Performance metrics
        """
        print("=== Testing Stage I Performance ===")
        
        results = {
            "questions": test_questions,
            "answers": [],
            "response_times": [],
            "context_scores": []
        }
        
        for question in test_questions:
            print(f"Testing: {question}")
            
            # Test with RAG system
            answer, context_docs, context_scores = rag_system.query(question)
            
            results["answers"].append(answer)
            results["context_scores"].append(context_scores)
            
            print(f"Answer: {answer[:100]}...")
            print(f"Context scores: {context_scores}")
            print()
        
        return results


def main():
    """Demo of SimRAG Stage I"""
    print("=== SimRAG Stage I Demo ===\n")
    
    # Initialize Stage I
    stage_i = SimRAGStageI()
    
    # Train Stage I
    print("1. Training Stage I...")
    version = stage_i.train_stage_i()
    
    if version:
        print(f"\n✅ Stage I training completed!")
        print(f"Model version: {version.version}")
        
        # Test performance
        print("\n2. Testing Stage I performance...")
        rag = BasicRAG()
        
        test_questions = [
            "What is Docker and how does it work?",
            "Explain binary search algorithm",
            "What are the benefits of DevOps practices?"
        ]
        
        results = stage_i.test_stage_i_performance(rag, test_questions)
        print(f"Tested {len(results['questions'])} questions")
    else:
        print("❌ Stage I training failed")


if __name__ == "__main__":
    main()
