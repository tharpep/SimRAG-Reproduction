"""
Synthetic Data Generator for SimRAG

Generates synthetic question-answer pairs for self-improvement.
"""

import random
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger


class SyntheticDataGenerator:
    """
    Generates synthetic question-answer pairs for self-improvement.
    
    This component is crucial for SimRAG's self-improvement loop, creating
    additional training data to enhance the system's performance.
    """
    
    def __init__(self, generator_model, tokenizer, config: Dict[str, Any]):
        """
        Initialize the synthetic data generator.
        
        Args:
            generator_model: The language model for generation
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        self.generator = generator_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Synthetic data generator initialized")
    
    def generate_pairs(self, documents: List[str], num_pairs: int) -> List[Dict[str, str]]:
        """
        Generate synthetic question-answer pairs from documents.
        
        Args:
            documents: List of source documents
            num_pairs: Number of pairs to generate
            
        Returns:
            List of synthetic QA pairs
        """
        logger.info(f"Generating {num_pairs} synthetic QA pairs from {len(documents)} documents")
        
        synthetic_pairs = []
        
        # Sample documents for generation
        sampled_docs = random.sample(documents, min(len(documents), num_pairs))
        
        for i, doc in enumerate(sampled_docs):
            if len(synthetic_pairs) >= num_pairs:
                break
            
            # Generate multiple pairs per document
            pairs_per_doc = max(1, num_pairs // len(sampled_docs))
            
            for _ in range(pairs_per_doc):
                if len(synthetic_pairs) >= num_pairs:
                    break
                
                try:
                    pair = self._generate_single_pair(doc)
                    if pair and self._validate_pair(pair):
                        synthetic_pairs.append(pair)
                except Exception as e:
                    logger.warning(f"Failed to generate pair from document {i}: {e}")
                    continue
        
        logger.info(f"Generated {len(synthetic_pairs)} valid synthetic QA pairs")
        return synthetic_pairs
    
    def _generate_single_pair(self, document: str) -> Dict[str, str]:
        """
        Generate a single question-answer pair from a document.
        
        Args:
            document: Source document text
            
        Returns:
            Dictionary with 'question', 'answer', and 'context' keys
        """
        # Create a prompt for question generation
        prompt = self._create_question_generation_prompt(document)
        
        # Generate question
        question = self._generate_text(prompt, max_length=50)
        
        # Create a prompt for answer generation
        answer_prompt = self._create_answer_generation_prompt(document, question)
        
        # Generate answer
        answer = self._generate_text(answer_prompt, max_length=100)
        
        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "context": document
        }
    
    def _create_question_generation_prompt(self, document: str) -> str:
        """
        Create a prompt for question generation.
        
        Args:
            document: Source document
            
        Returns:
            Formatted prompt for question generation
        """
        # Truncate document if too long
        max_doc_length = 500
        if len(document) > max_doc_length:
            document = document[:max_doc_length] + "..."
        
        prompt = f"""Based on the following document, generate a relevant question:

Document: {document}

Question:"""
        
        return prompt
    
    def _create_answer_generation_prompt(self, document: str, question: str) -> str:
        """
        Create a prompt for answer generation.
        
        Args:
            document: Source document
            question: Generated question
            
        Returns:
            Formatted prompt for answer generation
        """
        # Truncate document if too long
        max_doc_length = 500
        if len(document) > max_doc_length:
            document = document[:max_doc_length] + "..."
        
        prompt = f"""Based on the following document, answer the question:

Document: {document}

Question: {question}

Answer:"""
        
        return prompt
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the language model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=self.config["data"]["synthetic_data"]["generation_temperature"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _validate_pair(self, pair: Dict[str, str]) -> bool:
        """
        Validate a generated QA pair.
        
        Args:
            pair: Generated QA pair
            
        Returns:
            True if pair is valid, False otherwise
        """
        question = pair["question"]
        answer = pair["answer"]
        
        # Check minimum length
        if len(question) < 10 or len(answer) < 10:
            return False
        
        # Check for reasonable content
        if question.lower().startswith(("document:", "based on", "the following")):
            return False
        
        if answer.lower().startswith(("document:", "based on", "the following")):
            return False
        
        # Check for diversity (avoid repetitive questions)
        if self._is_repetitive_question(question):
            return False
        
        return True
    
    def _is_repetitive_question(self, question: str) -> bool:
        """
        Check if a question is too repetitive or generic.
        
        Args:
            question: Question text
            
        Returns:
            True if question is repetitive, False otherwise
        """
        # Common repetitive patterns
        repetitive_patterns = [
            "what is the main",
            "what are the key",
            "what does this",
            "how does this",
            "what can we learn",
            "what is the purpose",
            "what is the goal"
        ]
        
        question_lower = question.lower()
        
        # Check if question starts with repetitive patterns
        for pattern in repetitive_patterns:
            if question_lower.startswith(pattern):
                return True
        
        return False
    
    def generate_diverse_pairs(self, documents: List[str], num_pairs: int) -> List[Dict[str, str]]:
        """
        Generate diverse synthetic QA pairs using different strategies.
        
        Args:
            documents: List of source documents
            num_pairs: Number of pairs to generate
            
        Returns:
            List of diverse synthetic QA pairs
        """
        logger.info(f"Generating {num_pairs} diverse synthetic QA pairs")
        
        # Use different generation strategies
        strategies = [
            self._generate_factual_questions,
            self._generate_analytical_questions,
            self._generate_comparative_questions
        ]
        
        pairs_per_strategy = num_pairs // len(strategies)
        all_pairs = []
        
        for strategy in strategies:
            try:
                strategy_pairs = strategy(documents, pairs_per_strategy)
                all_pairs.extend(strategy_pairs)
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        # Shuffle and return requested number
        random.shuffle(all_pairs)
        return all_pairs[:num_pairs]
    
    def _generate_factual_questions(self, documents: List[str], num_pairs: int) -> List[Dict[str, str]]:
        """Generate factual questions from documents."""
        # Implementation for factual question generation
        return self.generate_pairs(documents, num_pairs)
    
    def _generate_analytical_questions(self, documents: List[str], num_pairs: int) -> List[Dict[str, str]]:
        """Generate analytical questions from documents."""
        # Implementation for analytical question generation
        return self.generate_pairs(documents, num_pairs)
    
    def _generate_comparative_questions(self, documents: List[str], num_pairs: int) -> List[Dict[str, str]]:
        """Generate comparative questions from documents."""
        # Implementation for comparative question generation
        return self.generate_pairs(documents, num_pairs)
