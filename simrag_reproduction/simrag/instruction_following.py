"""
SimRAG Instruction Following
Stage 1: Fine-tune on instruction-following datasets
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base import SimRAGBase
# from datasets import load_dataset  # Will be imported when needed


class InstructionFollowing(SimRAGBase):
    """SimRAG Stage 1: Instruction-following fine-tuning"""
    
    def __init__(self, model_name: str = "llama3.2:1b", config=None):
        """
        Initialize instruction following trainer
        
        Args:
            model_name: Model to fine-tune
            config: TuningConfig instance
        """
        super().__init__(model_name, config)
        print(f"Instruction Following trainer initialized")
    
    def load_instruction_datasets(self, dataset_names: Optional[List[str]] = None) -> List[str]:
        """
        Load real instruction-following datasets from Hugging Face
        
        Args:
            dataset_names: List of dataset names to load
            
        Returns:
            List of instruction-following examples
        """
        if dataset_names is None:
            dataset_names = ["alpaca"]
        
        print(f"Loading instruction datasets: {dataset_names}")
        
        # For now, return test data to avoid dependency issues
        # TODO: Implement real dataset loading when needed
        return self._generate_test_instruction_data()
    
    def _process_alpaca_dataset(self, dataset) -> List[str]:
        """Process Alpaca dataset into instruction format"""
        examples = []
        
        for item in dataset:
            if item.get("instruction") and item.get("output"):
                example = f"Question: {item['instruction']}\nAnswer: {item['output']}"
                examples.append(example)
        
        return examples
    
    def _process_sharegpt_dataset(self, dataset) -> List[str]:
        """Process ShareGPT dataset into instruction format"""
        examples = []
        
        for item in dataset:
            if item.get("conversations"):
                # Extract Q&A pairs from conversations
                conversations = item["conversations"]
                for i in range(0, len(conversations) - 1, 2):
                    if (conversations[i].get("from") == "human" and 
                        conversations[i + 1].get("from") == "gpt"):
                        question = conversations[i]["value"]
                        answer = conversations[i + 1]["value"]
                        example = f"Question: {question}\nAnswer: {answer}"
                        examples.append(example)
        
        return examples
    
    def _generate_test_instruction_data(self) -> List[str]:
        """Generate test instruction data for testing purposes"""
        return [
            "Question: What is Docker?\nAnswer: Docker is a platform for developing, shipping, and running applications in containers.",
            "Question: How does binary search work?\nAnswer: Binary search works by repeatedly dividing the search interval in half.",
            "Question: What is DevOps?\nAnswer: DevOps is a set of practices that combines software development and IT operations.",
            "Question: Explain Python programming.\nAnswer: Python is a high-level programming language known for its simple syntax and readability.",
            "Question: What is machine learning?\nAnswer: Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        ]
    
    def prepare_instruction_data(self, use_real_datasets: bool = True, 
                               dataset_names: Optional[List[str]] = None) -> List[str]:
        """
        Prepare instruction-following data for Stage 1
        
        Args:
            use_real_datasets: Whether to use real datasets or test data
            dataset_names: List of dataset names to load
            
        Returns:
            List of instruction-following examples
        """
        if use_real_datasets:
            return self.load_instruction_datasets(dataset_names or ["alpaca"])
        else:
            return self._generate_test_instruction_data()
    
    def train_stage_1(self, use_real_datasets: bool = True, 
                     dataset_names: Optional[List[str]] = None,
                     notes: str = "SimRAG Stage 1 - Instruction Following") -> Optional[Any]:
        """
        Train Stage 1: Instruction-following fine-tuning
        
        Args:
            use_real_datasets: Whether to use real datasets or test data
            dataset_names: List of dataset names to load
            notes: Training notes for versioning
            
        Returns:
            Model version object if successful
        """
        print("=== SimRAG Stage 1: Instruction Following Training ===")
        
        # Prepare instruction data
        print("Preparing instruction-following data...")
        instruction_data = self.prepare_instruction_data(use_real_datasets, dataset_names or ["alpaca"])
        
        # Load model and prepare training data
        self.load_model()
        train_dataset = self.prepare_training_data(instruction_data)
        
        # Setup trainer for Stage 1
        print("Setting up trainer for Stage 1...")
        output_dir = f"./tuned_models/llama_1b/stage_1"
        self.setup_trainer(
            train_dataset=train_dataset,
            output_dir=output_dir,
            notes=notes
        )
        
        # Train Stage 1
        print("Starting Stage 1 training...")
        version = self.train_model(notes)
        
        if version:
            print(f"✅ Stage 1 training completed!")
            print(f"Version: {version.version}")
            print(f"Training time: {version.training_time_seconds:.1f}s")
            if version.final_loss:
                print(f"Final loss: {version.final_loss:.4f}")
        
        return version
    
    def test_stage_1_performance(self, rag_system, test_questions: List[str]) -> Dict[str, Any]:
        """
        Test Stage 1 model performance on RAG tasks
        
        Args:
            rag_system: RAG system to test
            test_questions: List of test questions
            
        Returns:
            Performance metrics
        """
        print("=== Testing Stage 1 Performance ===")
        return self.test_performance(rag_system, test_questions)


def main():
    """Demo of instruction following training"""
    print("=== SimRAG Stage 1: Instruction Following Demo ===\n")
    
    # Initialize instruction following trainer
    trainer = InstructionFollowing()
    
    # Train Stage 1 (using test data for demo)
    print("1. Training Stage 1 (using test data)...")
    version = trainer.train_stage_1(use_real_datasets=False, notes="Demo Stage 1 training")
    
    if version:
        print(f"✅ Stage 1 training completed!")
        print(f"Model version: {version.version}")
        
        # Test performance
        print("\n2. Testing Stage 1 performance...")
        from rag.rag_setup import BasicRAG
        
        rag = BasicRAG()
        test_questions = [
            "What is Docker and how does it work?",
            "Explain binary search algorithm",
            "What are the benefits of DevOps practices?"
        ]
        
        results = trainer.test_stage_1_performance(rag, test_questions)
        print(f"Tested {len(results['questions'])} questions")
    else:
        print("❌ Stage 1 training failed")


if __name__ == "__main__":
    main()
