"""
RAG Query Logic
Handles querying RAG system with models
Matches Colab notebook exactly
"""

from typing import Tuple, List, Any

from ...logging_config import get_logger

logger = get_logger(__name__)


class RAGQuery:
    """
    RAG query handler
    Matches Colab notebook query functions exactly
    """
    
    @staticmethod
    def build_prompt(context_text: str, question: str) -> str:
        """
        Build RAG prompt (EXACT format from rag_setup.py, matches Colab)
        
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
        Extract answer from model output (matches Colab)
        
        Args:
            full_output: Full model output including prompt
            prompt: Original prompt used
            
        Returns:
            Extracted answer text
        """
        # Extract just the answer (remove prompt)
        if prompt in full_output:
            answer = full_output.replace(prompt, "").strip()
        else:
            # Fallback: try to find where answer starts
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
        Query RAG system with baseline model (matches Colab query_rag_baseline)
        
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
        """
        # Retrieve context
        context_docs, context_scores = vector_store.query(question, top_k=top_k)
        context_text = "\n\n".join(context_docs)
        
        # Build prompt - EXACT format from rag_setup.py (matches Colab)
        prompt = cls.build_prompt(context_text, question)
        
        # Generate answer using pre-loaded baseline model
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
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
        Query RAG system with fine-tuned model (matches Colab query_rag)
        
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
        """
        # Retrieve context
        context_docs, context_scores = vector_store.query(question, top_k=top_k)
        context_text = "\n\n".join(context_docs)
        
        # Build prompt - EXACT format from rag_setup.py (matches Colab)
        prompt = cls.build_prompt(context_text, question)
        
        # Generate answer with same parameters as baseline
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = cls.extract_answer(answer, prompt)
        
        return answer, context_docs, context_scores

