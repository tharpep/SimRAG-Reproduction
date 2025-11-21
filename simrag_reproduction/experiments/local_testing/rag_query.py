"""
RAG Query Logic
Handles querying RAG system with models
"""

from typing import Tuple, List, Any

from ...logging_config import get_logger

logger = get_logger(__name__)

# Constants for magic numbers
MAX_TOKENIZER_LENGTH = 2048


class RAGQuery:
    """
    RAG query handler for executing queries with baseline and fine-tuned models
    """
    
    @staticmethod
    def build_prompt(context_text: str, question: str) -> str:
        """
        Build RAG prompt using standard format
        
        Args:
            context_text: Retrieved context documents (concatenated)
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"""Context:
{context_text}

Question: {question}

Answer:"""
    
    @staticmethod
    def extract_answer(full_output: str, prompt: str) -> str:
        """
        Extract answer from model output
        
        Args:
            full_output: Full model output including prompt
            prompt: Original prompt used
            
        Returns:
            Extracted answer text
        """
        if prompt in full_output:
            # Replace only the first occurrence, not all occurrences
            answer = full_output.replace(prompt, "", 1).strip()
        else:
            answer_marker = "Answer:"
            if answer_marker in full_output:
                answer = full_output.split(answer_marker)[-1].strip()
            else:
                answer = full_output.strip()
        
        return answer
    
    @classmethod
    def query_baseline(
        cls,
        question: str,
        model: Any,
        tokenizer: Any,
        vector_store: Any,
        top_k: int,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, List[str], List[float]]:
        """
        Query RAG system with baseline model
        
        Args:
            question: Question to answer
            model: Baseline model
            tokenizer: Tokenizer
            vector_store: ChromaDBStore instance
            top_k: Number of documents to retrieve
            temperature: Generation temperature
            max_tokens: Maximum new tokens to generate
            
        Returns:
            Tuple of (answer, context_docs, context_scores)
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if model is None:
            raise ValueError("model cannot be None")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got: {max_tokens}")
        
        context_docs, context_scores = vector_store.query(question, top_k=top_k)
        context_text = "\n\n".join(context_docs)
        prompt = cls.build_prompt(context_text, question)
        
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENIZER_LENGTH).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        if not outputs or len(outputs) == 0:
            raise ValueError("Model generation returned empty output")
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = cls.extract_answer(answer, prompt)
        
        return answer, context_docs, context_scores
    
    @classmethod
    def query_finetuned(
        cls,
        question: str,
        model: Any,
        tokenizer: Any,
        vector_store: Any,
        top_k: int,
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, List[str], List[float]]:
        """
        Query RAG system with fine-tuned model
        
        Args:
            question: Question to answer
            model: Fine-tuned model
            tokenizer: Tokenizer
            vector_store: ChromaDBStore instance
            top_k: Number of documents to retrieve
            temperature: Generation temperature
            max_tokens: Maximum new tokens to generate
            
        Returns:
            Tuple of (answer, context_docs, context_scores)
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if model is None:
            raise ValueError("model cannot be None")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got: {max_tokens}")
        
        context_docs, context_scores = vector_store.query(question, top_k=top_k)
        context_text = "\n\n".join(context_docs)
        prompt = cls.build_prompt(context_text, question)
        
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENIZER_LENGTH).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        if not outputs or len(outputs) == 0:
            raise ValueError("Model generation returned empty output")
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = cls.extract_answer(answer, prompt)
        
        return answer, context_docs, context_scores

