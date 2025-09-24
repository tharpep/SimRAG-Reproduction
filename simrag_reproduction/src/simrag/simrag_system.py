"""
SimRAG System Implementation

Implements the self-improving RAG system as described in the SimRAG paper.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from baseline.rag_system import BaselineRAGSystem
from .self_improvement import SelfImprovementModule
from .synthetic_data_generator import SyntheticDataGenerator
from evaluation.metrics import compute_metrics


class SimRAGSystem(BaselineRAGSystem):
    """
    SimRAG: Self-improving Retrieval-Augmented Generation system.
    
    Extends the baseline RAG system with self-improvement capabilities through
    synthetic data generation and iterative fine-tuning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SimRAG system.
        
        Args:
            config: Configuration dictionary containing model and system parameters
        """
        super().__init__(config)
        
        # Initialize SimRAG-specific components
        self._setup_quantization()
        self._setup_self_improvement()
        self._setup_synthetic_generator()
        
        # Track improvement metrics
        self.improvement_history = []
        self.current_iteration = 0
        
        logger.info("SimRAG system initialized with self-improvement capabilities")
    
    def _setup_quantization(self):
        """Setup QLoRA quantization for efficient fine-tuning."""
        if self.config.get("hardware", {}).get("use_qlora", False):
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Reinitialize generator with quantization
            generator_config = self.config["model"]["generator"]
            self.generator = AutoModelForCausalLM.from_pretrained(
                generator_config["name"],
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            logger.info("Generator configured with QLoRA quantization")
    
    def _setup_self_improvement(self):
        """Initialize the self-improvement module."""
        self.self_improvement = SelfImprovementModule(self.config)
        logger.info("Self-improvement module initialized")
    
    def _setup_synthetic_generator(self):
        """Initialize the synthetic data generator."""
        self.synthetic_generator = SyntheticDataGenerator(
            self.generator, 
            self.tokenizer, 
            self.config
        )
        logger.info("Synthetic data generator initialized")
    
    def generate_synthetic_qa_pairs(self, num_pairs: int) -> List[Dict[str, str]]:
        """
        Generate synthetic question-answer pairs for self-improvement.
        
        Args:
            num_pairs: Number of synthetic pairs to generate
            
        Returns:
            List of synthetic QA pairs
        """
        logger.info(f"Generating {num_pairs} synthetic QA pairs")
        
        # Sample documents from the corpus
        if len(self.documents) == 0:
            raise ValueError("No documents available for synthetic data generation")
        
        # Generate synthetic pairs
        synthetic_pairs = self.synthetic_generator.generate_pairs(
            self.documents, 
            num_pairs
        )
        
        logger.info(f"Generated {len(synthetic_pairs)} synthetic QA pairs")
        return synthetic_pairs
    
    def self_improve(self, real_data: List[Dict[str, str]], 
                    synthetic_data: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Perform one iteration of self-improvement.
        
        Args:
            real_data: Real QA pairs for training
            synthetic_data: Synthetic QA pairs for training
            
        Returns:
            Dictionary of improvement metrics
        """
        logger.info(f"Starting self-improvement iteration {self.current_iteration + 1}")
        
        # Combine real and synthetic data
        combined_data = real_data + synthetic_data
        
        # Fine-tune retriever if enabled
        if self.config["self_improvement"]["fine_tune_retriever"]:
            self._fine_tune_retriever(combined_data)
        
        # Fine-tune generator if enabled
        if self.config["self_improvement"]["fine_tune_generator"]:
            self._fine_tune_generator(combined_data)
        
        # Evaluate improvement
        improvement_metrics = self._evaluate_improvement(real_data)
        
        # Update history
        self.improvement_history.append(improvement_metrics)
        self.current_iteration += 1
        
        logger.info(f"Self-improvement iteration {self.current_iteration} completed")
        return improvement_metrics
    
    def _fine_tune_retriever(self, training_data: List[Dict[str, str]]):
        """Fine-tune the retriever on combined real and synthetic data."""
        logger.info("Fine-tuning retriever")
        
        # Prepare training data for retriever
        queries = [item["question"] for item in training_data]
        contexts = [item["answer"] for item in training_data]  # Use answers as positive contexts
        
        # Create training pairs
        training_pairs = list(zip(queries, contexts))
        
        # Fine-tune using contrastive learning
        self.self_improvement.fine_tune_retriever(
            self.retriever, 
            training_pairs,
            learning_rate=self.config["self_improvement"]["retriever_lr"]
        )
        
        # Rebuild index with updated retriever
        self._rebuild_index()
        
        logger.info("Retriever fine-tuning completed")
    
    def _fine_tune_generator(self, training_data: List[Dict[str, str]]):
        """Fine-tune the generator using QLoRA."""
        logger.info("Fine-tuning generator with QLoRA")
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Apply LoRA to model
        model = get_peft_model(self.generator, lora_config)
        
        # Prepare training data
        train_dataset = self._prepare_generator_dataset(training_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results/simrag/generator_checkpoints",
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["self_improvement"]["generator_lr"],
            warmup_steps=self.config["training"]["warmup_steps"],
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            save_total_limit=2,
            fp16=True,
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Update generator
        self.generator = model
        
        logger.info("Generator fine-tuning completed")
    
    def _prepare_generator_dataset(self, training_data: List[Dict[str, str]]):
        """Prepare dataset for generator fine-tuning."""
        from torch.utils.data import Dataset
        
        class RAGDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=512):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                
                # Create input prompt
                prompt = f"Context: {item.get('context', '')}\n\nQuestion: {item['question']}\n\nAnswer: {item['answer']}"
                
                # Tokenize
                encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": encoding["input_ids"].flatten()
                }
        
        return RAGDataset(training_data, self.tokenizer)
    
    def _rebuild_index(self):
        """Rebuild the document index with the updated retriever."""
        if self.documents:
            logger.info("Rebuilding document index with updated retriever")
            documents = self.documents.copy()
            self.documents = []
            self.add_documents(documents)
    
    def _evaluate_improvement(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate the current system performance."""
        logger.info("Evaluating system improvement")
        
        # Get current performance
        current_metrics = self.evaluate(test_data)
        
        # Compare with previous iteration if available
        if self.improvement_history:
            previous_metrics = self.improvement_history[-1]
            improvement = {
                metric: current_metrics[metric] - previous_metrics[metric]
                for metric in current_metrics.keys()
            }
        else:
            improvement = current_metrics
        
        return improvement
    
    def run_self_improvement_loop(self, real_data: List[Dict[str, str]], 
                                 max_iterations: int = 3) -> List[Dict[str, float]]:
        """
        Run the complete self-improvement loop.
        
        Args:
            real_data: Real QA pairs for training and evaluation
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            List of improvement metrics for each iteration
        """
        logger.info(f"Starting self-improvement loop with max {max_iterations} iterations")
        
        # Split data for training and evaluation
        split_idx = int(len(real_data) * 0.8)
        train_data = real_data[:split_idx]
        eval_data = real_data[split_idx:]
        
        all_improvements = []
        
        for iteration in range(max_iterations):
            logger.info(f"Self-improvement iteration {iteration + 1}/{max_iterations}")
            
            # Generate synthetic data
            num_synthetic = int(len(train_data) * self.config["data"]["synthetic_data"]["num_synthetic_pairs"] / 1000)
            synthetic_data = self.generate_synthetic_qa_pairs(num_synthetic)
            
            # Perform self-improvement
            improvement = self.self_improve(train_data, synthetic_data)
            all_improvements.append(improvement)
            
            # Check if improvement threshold is met
            improvement_threshold = self.config["self_improvement"]["improvement_threshold"]
            if improvement.get("f1", 0) < improvement_threshold:
                logger.info(f"Improvement below threshold ({improvement_threshold}), stopping early")
                break
        
        logger.info("Self-improvement loop completed")
        return all_improvements
