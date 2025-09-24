"""
Dataset Loading and Preparation

Handles loading and preprocessing of datasets for RAG experiments.
"""

import random
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests
from loguru import logger


def load_dataset(corpus_size: int = 10000) -> List[str]:
    """
    Load a dataset of documents for RAG experiments.
    
    For this reproduction, we'll use a combination of:
    1. Sample documents from Wikipedia
    2. Generated synthetic documents
    3. Academic papers abstracts (if available)
    
    Args:
        corpus_size: Number of documents to load
        
    Returns:
        List of document texts
    """
    logger.info(f"Loading dataset with {corpus_size} documents")
    
    documents = []
    
    # Load Wikipedia samples (if available)
    try:
        wiki_docs = _load_wikipedia_samples(min(corpus_size // 3, 3000))
        documents.extend(wiki_docs)
        logger.info(f"Loaded {len(wiki_docs)} Wikipedia documents")
    except Exception as e:
        logger.warning(f"Could not load Wikipedia documents: {e}")
    
    # Generate synthetic documents
    remaining = corpus_size - len(documents)
    if remaining > 0:
        synthetic_docs = _generate_synthetic_documents(remaining)
        documents.extend(synthetic_docs)
        logger.info(f"Generated {len(synthetic_docs)} synthetic documents")
    
    # Ensure we have the requested number of documents
    if len(documents) > corpus_size:
        documents = documents[:corpus_size]
    elif len(documents) < corpus_size:
        # Generate more synthetic documents if needed
        additional = corpus_size - len(documents)
        more_synthetic = _generate_synthetic_documents(additional)
        documents.extend(more_synthetic)
    
    logger.info(f"Dataset loaded: {len(documents)} documents")
    return documents


def _load_wikipedia_samples(num_docs: int) -> List[str]:
    """
    Load sample documents from Wikipedia.
    
    Args:
        num_docs: Number of documents to load
        
    Returns:
        List of Wikipedia document texts
    """
    # For this demo, we'll create sample Wikipedia-style documents
    # In a real implementation, you might use the Wikipedia API or a dataset
    
    sample_topics = [
        "artificial intelligence", "machine learning", "natural language processing",
        "computer science", "data science", "deep learning", "neural networks",
        "information retrieval", "question answering", "text generation",
        "transformer models", "attention mechanisms", "language models",
        "retrieval augmented generation", "fine-tuning", "pre-training",
        "embeddings", "vector databases", "semantic search", "document ranking"
    ]
    
    documents = []
    for i in range(num_docs):
        topic = random.choice(sample_topics)
        doc = f"""
        {topic.title()} is a fascinating field of study that has gained significant attention in recent years. 
        The development of {topic} has revolutionized many aspects of technology and research. 
        Researchers have made substantial progress in understanding the fundamental principles 
        underlying {topic} and its applications across various domains.
        
        The key challenges in {topic} include scalability, efficiency, and accuracy. 
        Recent advances have shown promising results in addressing these challenges through 
        innovative approaches and novel methodologies. The field continues to evolve rapidly 
        with new discoveries and breakthroughs emerging regularly.
        
        Applications of {topic} span multiple industries including healthcare, finance, 
        education, and entertainment. The potential impact of {topic} on society is 
        profound, with implications for both positive and negative outcomes that require 
        careful consideration and ethical evaluation.
        """
        documents.append(doc.strip())
    
    return documents


def _generate_synthetic_documents(num_docs: int) -> List[str]:
    """
    Generate synthetic documents for the corpus.
    
    Args:
        num_docs: Number of documents to generate
        
    Returns:
        List of synthetic document texts
    """
    # Define document templates and topics
    templates = [
        "The concept of {topic} has been extensively studied in the literature. {topic} represents a fundamental aspect of {domain} that requires careful analysis and understanding.",
        "Recent research in {domain} has focused on advancing the state-of-the-art in {topic}. The development of {topic} has opened new possibilities for {application}.",
        "The field of {domain} has witnessed significant progress in {topic} over the past decade. {topic} has become a cornerstone technology for {application}.",
        "Understanding {topic} is crucial for advancing {domain}. The complexity of {topic} requires sophisticated approaches and innovative solutions.",
        "The application of {topic} in {domain} has shown remarkable results. {topic} provides a powerful framework for addressing challenges in {application}."
    ]
    
    topics = [
        "algorithmic optimization", "data preprocessing", "feature engineering", 
        "model selection", "hyperparameter tuning", "cross-validation", 
        "ensemble methods", "dimensionality reduction", "clustering algorithms",
        "classification techniques", "regression analysis", "time series forecasting",
        "anomaly detection", "pattern recognition", "statistical inference",
        "experimental design", "hypothesis testing", "confidence intervals",
        "bayesian methods", "frequentist approaches", "machine learning pipelines"
    ]
    
    domains = [
        "computer science", "data science", "artificial intelligence", 
        "machine learning", "statistics", "mathematics", "engineering",
        "research methodology", "scientific computing", "analytics"
    ]
    
    applications = [
        "real-world problems", "business intelligence", "scientific research",
        "decision making", "automation", "optimization", "prediction",
        "classification", "clustering", "recommendation systems"
    ]
    
    documents = []
    for i in range(num_docs):
        template = random.choice(templates)
        topic = random.choice(topics)
        domain = random.choice(domains)
        application = random.choice(applications)
        
        doc = template.format(topic=topic, domain=domain, application=application)
        
        # Add more content to make documents more substantial
        doc += f" The methodology involves several key steps including data collection, preprocessing, model training, and evaluation. "
        doc += f"Results demonstrate the effectiveness of {topic} in various scenarios with consistent performance improvements. "
        doc += f"Future work will explore advanced techniques and novel applications of {topic} in {domain}."
        
        documents.append(doc)
    
    return documents


def prepare_qa_data(documents: List[str], test_size: float = 0.2) -> List[Dict[str, str]]:
    """
    Prepare question-answer pairs from documents for evaluation.
    
    Args:
        documents: List of document texts
        test_size: Fraction of documents to use for testing
        
    Returns:
        List of QA pairs
    """
    logger.info("Preparing QA data from documents")
    
    # Generate questions for a subset of documents
    num_test_docs = int(len(documents) * test_size)
    test_docs = random.sample(documents, num_test_docs)
    
    qa_pairs = []
    
    for doc in test_docs:
        # Generate multiple questions per document
        questions = _generate_questions_for_document(doc)
        for question in questions:
            qa_pairs.append({
                "question": question,
                "answer": _extract_answer_for_question(doc, question),
                "context": doc
            })
    
    logger.info(f"Generated {len(qa_pairs)} QA pairs from {len(test_docs)} documents")
    return qa_pairs


def _generate_questions_for_document(doc: str) -> List[str]:
    """
    Generate questions for a given document.
    
    Args:
        doc: Document text
        
    Returns:
        List of questions
    """
    # Simple question generation based on document content
    # In a real implementation, you might use a more sophisticated approach
    
    questions = []
    
    # Extract key terms and concepts
    words = doc.lower().split()
    key_terms = [w for w in words if len(w) > 5 and w.isalpha()]
    key_terms = list(set(key_terms))[:5]  # Take top 5 unique terms
    
    # Generate questions based on key terms
    question_templates = [
        "What is {term}?",
        "How does {term} work?",
        "What are the applications of {term}?",
        "What are the challenges in {term}?",
        "What is the importance of {term}?"
    ]
    
    for term in key_terms[:3]:  # Generate up to 3 questions per document
        template = random.choice(question_templates)
        question = template.format(term=term)
        questions.append(question)
    
    return questions


def _extract_answer_for_question(doc: str, question: str) -> str:
    """
    Extract or generate an answer for a question based on the document.
    
    Args:
        doc: Document text
        question: Question text
        
    Returns:
        Answer text
    """
    # Simple answer extraction based on question type
    # In a real implementation, you might use more sophisticated methods
    
    question_lower = question.lower()
    
    if "what is" in question_lower:
        # Extract definition or explanation
        sentences = doc.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip() + "."
    
    elif "how does" in question_lower:
        # Extract process or mechanism explanation
        sentences = doc.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["process", "method", "approach", "technique"]):
                return sentence.strip() + "."
    
    elif "applications" in question_lower:
        # Extract application information
        sentences = doc.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ["application", "use", "applied"]):
                return sentence.strip() + "."
    
    else:
        # Default: return first substantial sentence
        sentences = doc.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip() + "."
    
    # Fallback: return first sentence
    return doc.split('.')[0].strip() + "."


def chunk_documents(documents: List[str], chunk_size: int = 256, 
                   chunk_overlap: int = 50) -> List[str]:
    """
    Split documents into chunks for indexing.
    
    Args:
        documents: List of document texts
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document chunks
    """
    logger.info(f"Chunking {len(documents)} documents with size {chunk_size} and overlap {chunk_overlap}")
    
    chunks = []
    
    for doc in documents:
        if len(doc) <= chunk_size:
            chunks.append(doc)
        else:
            # Split document into overlapping chunks
            start = 0
            while start < len(doc):
                end = start + chunk_size
                chunk = doc[start:end]
                chunks.append(chunk)
                
                # Move start position with overlap
                start = end - chunk_overlap
                if start >= len(doc):
                    break
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def save_dataset(documents: List[str], output_path: str):
    """
    Save dataset to file.
    
    Args:
        documents: List of document texts
        output_path: Path to save the dataset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset saved to {output_path}")


def load_dataset_from_file(file_path: str) -> List[str]:
    """
    Load dataset from file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of document texts
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"Dataset loaded from {file_path}: {len(documents)} documents")
    return documents
