"""
SimRAG Synthetic QA Generation
Stage 2: Generate synthetic question-answer pairs from domain documents
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..logging_config import get_logger
from .base import SimRAGBase
from ..rag.rag_setup import BasicRAG
from ..ai_providers.gateway import AIGateway
from ..config import get_rag_config

logger = get_logger(__name__)

# Constants for magic numbers
DOCUMENT_PREVIEW_LENGTH = 500
CONTEXT_TRUNCATE_LENGTH = 200
MIN_QUESTION_LENGTH = 10
QUESTION_LOG_PREVIEW_LENGTH = 50


class SyntheticQAGeneration(SimRAGBase):
    """SimRAG Stage 2: Synthetic QA generation from domain documents"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", config: Optional[Any] = None, stage_1_model_path: Optional[str] = None):
        """
        Initialize synthetic QA generator
        
        Args:
            model_name: Model to use for QA generation
            config: TuningConfig instance (optional)
            stage_1_model_path: Path to Stage 1 model (optional)
        """
        super().__init__(model_name, config)
        self.stage_1_model_path = stage_1_model_path
        try:
            # Initialize gateway (will auto-select provider based on config and availability)
            self.gateway = AIGateway()
            # Initialize RAG system with preferred provider for answer generation
            self.rag_system = self._initialize_rag_system()
            rag_config = get_rag_config()
            preferred_provider = rag_config.qa_provider if hasattr(rag_config, 'qa_provider') else "purdue"
            available_providers = self.gateway.get_available_providers()
            logger.info("Synthetic QA Generator initialized")
            logger.info(f"Preferred QA provider: {preferred_provider}")
            logger.info(f"Available providers: {', '.join(available_providers) if available_providers else 'none'}")
            if stage_1_model_path:
                logger.info(f"Stage 1 model available: {stage_1_model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Synthetic QA Generator: {e}")
            raise
    
    def _initialize_rag_system(self) -> BasicRAG:
        """
        Initialize RAG system for synthetic QA generation
        
        Uses preferred provider from config (claude, purdue, or huggingface)
        """
        rag_config = get_rag_config()
        preferred_provider = rag_config.qa_provider if hasattr(rag_config, 'qa_provider') else "purdue"
        # Use preferred provider, but allow fallback if not available
        return BasicRAG(force_provider=preferred_provider)
    
    def generate_questions_from_document(self, document: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions from a single document using LLM
        
        Args:
            document: Source document text
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
            
        Raises:
            ValueError: If document is empty or num_questions is invalid
        """
        if not document or not isinstance(document, str) or not document.strip():
            raise ValueError("document must be a non-empty string")
        if not isinstance(num_questions, int) or num_questions <= 0:
            raise ValueError(f"num_questions must be a positive integer, got: {num_questions}")
        prompt = f"""Given this document, generate {num_questions} diverse questions that test understanding:

Document: {document[:DOCUMENT_PREVIEW_LENGTH]}...

Generate questions that cover:
1. Key concepts and definitions
2. Process explanations  
3. Practical applications
4. Technical details

Questions:"""
        
        try:
            # Use preferred provider from config (auto-selects with fallback)
            rag_config = get_rag_config()
            preferred_provider = rag_config.qa_provider if hasattr(rag_config, 'qa_provider') else None
            response = self.gateway.chat(prompt, provider=preferred_provider, force_provider=False)
            questions = self._parse_questions(response)
            result = questions[:num_questions]  # Limit to requested number
            logger.debug(f"Generated {len(result)} questions from document")
            return result
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
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
                if question and len(question) > MIN_QUESTION_LENGTH:  # Filter out very short responses
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
            
        Raises:
            ValueError: If documents list is empty or invalid
        """
        logger.info(f"=== Generating Synthetic QA Pairs ===")
        logger.info(f"Processing {len(documents)} documents...")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        if not isinstance(documents, list):
            raise ValueError(f"documents must be a list, got: {type(documents)}")
        if not isinstance(questions_per_doc, int) or questions_per_doc <= 0:
            raise ValueError(f"questions_per_doc must be a positive integer, got: {questions_per_doc}")
        
        qa_pairs = []
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}...")
            
            # Generate questions from document
            questions = self.generate_questions_from_document(doc, questions_per_doc)
            
            if not questions:
                logger.warning(f"  No questions generated for document {i}")
                continue
            
            for question in questions:
                try:
                    # Generate answer using RAG system
                    answer, context_docs, context_scores = self.rag_system.query(question)
                    
                    # Create QA pair
                    qa_pair = {
                        "question": question,
                        "answer": answer,
                        "context": doc[:CONTEXT_TRUNCATE_LENGTH] + "..." if len(doc) > CONTEXT_TRUNCATE_LENGTH else doc,
                        "context_docs": context_docs,
                        "context_scores": context_scores,
                        "source_doc_index": i - 1
                    }
                    
                    qa_pairs.append(qa_pair)
                    question_preview = question[:QUESTION_LOG_PREVIEW_LENGTH] + "..." if len(question) > QUESTION_LOG_PREVIEW_LENGTH else question
                    logger.debug(f"  Generated Q: {question_preview}")
                    
                except Exception as e:
                    question_preview = question[:QUESTION_LOG_PREVIEW_LENGTH] + "..." if len(question) > QUESTION_LOG_PREVIEW_LENGTH else question
                    logger.error(f"  Error processing question '{question_preview}': {e}")
                    continue
        
        logger.info(f"Generated {len(qa_pairs)} synthetic QA pairs")
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
        """
        Filter QA pairs based on quality criteria
        
        Args:
            qa_pairs: List of QA pairs to filter
            min_context_score: Minimum context similarity score
            min_answer_length: Minimum answer length in characters
            
        Returns:
            Filtered list of high-quality QA pairs
        """
        logger.info(f"=== Filtering High-Quality QA Pairs ===")
        logger.info(f"Original pairs: {len(qa_pairs)}")
        
        if not qa_pairs:
            logger.warning("No QA pairs to filter")
            return []
        
        filtered_pairs = []
        
        for qa_pair in qa_pairs:
            try:
                # Check context scores
                context_scores = qa_pair.get("context_scores", [])
                if context_scores:
                    avg_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
                    if avg_score < min_context_score:
                        continue
                
                # Check answer quality
                if len(qa_pair.get("answer", "")) < min_answer_length:
                    continue
                
                # Check for meaningful content
                answer_lower = qa_pair.get("answer", "").lower()
                if answer_lower in ["no relevant documents found", "i don't know", "cannot answer"]:
                    continue
                
                filtered_pairs.append(qa_pair)
            except Exception as e:
                logger.warning(f"Error filtering QA pair: {e}")
                continue
        
        logger.info(f"Filtered pairs: {len(filtered_pairs)}")
        if qa_pairs:
            retention = (len(filtered_pairs) / len(qa_pairs)) * 100 if qa_pairs else 0.0
            logger.info(f"Quality retention: {retention:.1f}%")
        else:
            logger.warning("Quality retention: 0.0% (no pairs to filter)")
        
        return filtered_pairs
    
    def prepare_qa_training_data(self, qa_pairs: List[Dict[str, Any]]) -> List[str]:
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
        
        logger.info(f"Prepared {len(training_examples)} training examples")
        return training_examples
    
    def generate_synthetic_dataset(self, documents: List[str], 
                                 questions_per_doc: Optional[int] = None,
                                 min_context_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset for Stage 2 training
        
        Args:
            documents: List of domain documents
            questions_per_doc: Questions to generate per document (uses config default if None)
            min_context_score: Minimum context similarity threshold (uses config default if None)
            
        Returns:
            Complete synthetic dataset with metadata
        """
        logger.info("=== SimRAG Stage 2: Synthetic Dataset Generation ===")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Use config defaults if not provided
        questions_per_doc = questions_per_doc or self.config.simrag_questions_per_doc
        min_context_score = min_context_score or self.config.simrag_min_context_score
        
        # Generate QA pairs
        try:
            qa_pairs = self.create_qa_pairs_from_documents(documents, questions_per_doc)
        except Exception as e:
            logger.error(f"Failed to create QA pairs: {e}")
            raise
        
        # Filter for quality
        try:
            high_quality_pairs = self.filter_high_quality_qa_pairs(qa_pairs, min_context_score)
        except Exception as e:
            logger.error(f"Failed to filter QA pairs: {e}")
            raise
        
        # Prepare training data
        training_data = self.prepare_qa_training_data(high_quality_pairs)
        
        # Create dataset summary
        try:
            all_scores = []
            for pair in high_quality_pairs:
                scores = pair.get("context_scores", [])
                if scores:
                    all_scores.extend(scores)
            
            dataset_info = {
                "total_documents": len(documents),
                "total_qa_pairs": len(qa_pairs),
                "high_quality_pairs": len(high_quality_pairs),
                "training_examples": len(training_data),
                "quality_retention": (len(high_quality_pairs) / len(qa_pairs) * 100) if qa_pairs else 0.0,
                "avg_context_score": sum(all_scores) / len(all_scores) if all_scores else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to calculate dataset info: {e}")
            raise
        
        logger.info(f"Synthetic dataset generated!")
        logger.info(f"   Documents: {dataset_info['total_documents']}")
        logger.info(f"   QA pairs: {dataset_info['total_qa_pairs']}")
        logger.info(f"   High quality: {dataset_info['high_quality_pairs']}")
        logger.info(f"   Training examples: {dataset_info['training_examples']}")
        logger.info(f"   Quality retention: {dataset_info['quality_retention']:.1f}%")
        
        return {
            "qa_pairs": high_quality_pairs,
            "training_data": training_data,
            "dataset_info": dataset_info
        }


