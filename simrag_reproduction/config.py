"""
Simple Configuration Management
Basic settings for RAG system testing and operation
"""

import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Simple configuration for RAG system"""
    
    # Model size settings
    model_size: str = "small"  # "small" for llama3.2:1b, "medium" for llama3:8b
    
    # AI Provider settings
    use_ollama: bool = True  # True for Ollama (local), False for Purdue API
    
    # Vector store settings
    use_persistent: bool = True  # True for persistent storage, False for in-memory only
    collection_name: str = "simrag_docs"  # Name for Qdrant collection
    clear_on_ingest: bool = True  # Clear collection before ingesting new documents
    
    # Retrieval settings
    top_k: int = 5  # Number of documents to retrieve (1-20 recommended)
    similarity_threshold: float = 0.7  # Minimum similarity score (0.0-1.0)
    
    # Generation settings
    max_tokens: int = 200  # Maximum tokens in response (50-500 recommended)
    temperature: float = 0.7  # Creativity level (0.0-1.0, lower = more focused)
    
    @property
    def model_name(self) -> str:
        """Get model name based on model size configuration"""
        return "llama3.2:1b" if self.model_size == "small" else "llama3:8b"


@dataclass
class TuningConfig:
    """Simple configuration for model fine-tuning"""
    
    # Model size settings
    model_size: str = "small"  # "small" for llama3.2:1b, "medium" for llama3:8b
    
    # Model settings
    device: str = "auto"  # Options: "auto", "cpu", "cuda", "mps" (for Apple Silicon)
    max_length: int = 512  # Maximum sequence length (256-1024 recommended)
    
    # Training settings (will be optimized based on model_size)
    num_epochs: int = 3  # Number of training epochs (1-10 recommended)
    batch_size: int = 4  # Batch size (1-16, adjust based on GPU memory)
    learning_rate: float = 5e-5  # Learning rate (1e-5 to 1e-3 recommended)
    
    # Model versioning
    version: str = "v1.0"  # Model version (e.g., v1.0, v1.1, v2.0)
    create_version_dir: bool = True  # Whether to create versioned subdirectories
    
    # SimRAG specific settings
    simrag_stage_1_epochs: int = 1  # Epochs for Stage 1 (instruction following)
    simrag_stage_2_epochs: int = 1  # Epochs for Stage 2 (domain adaptation)
    simrag_improvement_rounds: int = 2  # Number of self-improvement rounds
    simrag_questions_per_doc: int = 2  # Questions to generate per document
    simrag_min_context_score: float = 0.7  # Minimum context similarity threshold
    
    @property
    def optimized_batch_size(self) -> int:
        """Get batch size optimized for model size"""
        return 1 if self.model_size == "small" else 8  # Small: 1, Medium: 8
    
    @property
    def optimized_num_epochs(self) -> int:
        """Get number of epochs optimized for model size"""
        return 1 if self.model_size == "small" else 3  # Small: 1 for speed, Medium: 3 for quality
    
    @property
    def model_name(self) -> str:
        """Get model name based on model size configuration"""
        return "llama3.2:1b" if self.model_size == "small" else "llama3:8b"
    
    @property
    def output_dir(self) -> str:
        """Get output directory based on model size configuration"""
        model_suffix = "1b" if self.model_size == "small" else "8b"
        base_dir = f"./tuned_models/llama_{model_suffix}"
        
        if self.create_version_dir:
            return f"{base_dir}/{self.version}"
        return base_dir
    
    @property
    def model_registry_path(self) -> str:
        """Get path to model registry metadata file"""
        model_suffix = "1b" if self.model_size == "small" else "8b"
        return f"./tuned_models/llama_{model_suffix}/model_registry.json"
    
    def get_stage_output_dir(self, stage: str) -> str:
        """
        Get output directory for a specific SimRAG stage
        
        Args:
            stage: Stage name (e.g., "stage_1", "stage_2")
            
        Returns:
            Path to stage-specific output directory
        """
        model_suffix = "1b" if self.model_size == "small" else "8b"
        base_dir = f"./tuned_models/llama_{model_suffix}"
        return f"{base_dir}/{stage}"


# Default configurations
DEFAULT_RAG_CONFIG = RAGConfig()
DEFAULT_TUNING_CONFIG = TuningConfig()


def get_rag_config() -> RAGConfig:
    """Get RAG configuration with environment variable overrides
    
    Environment variables that can be set:
        - MODEL_SIZE: "small" or "medium" (small=llama3.2:1b, medium=llama3:8b)
    - USE_OLLAMA: "true" or "false" (use Ollama vs Purdue API)
    - USE_PERSISTENT: "true" or "false" (persistent vs in-memory storage)
    - COLLECTION_NAME: name for Qdrant collection
    """
    config = RAGConfig()
    
    # Override with environment variables if set
    model_size_env = os.getenv("MODEL_SIZE")
    if model_size_env:
        model_size_lower = model_size_env.lower()
        if model_size_lower in ["small", "medium"]:
            config.model_size = model_size_lower
        else:
            print(f"Warning: Invalid MODEL_SIZE '{model_size_env}', must be 'small' or 'medium'. Using default.")
    
    use_ollama_env = os.getenv("USE_OLLAMA")
    if use_ollama_env:
        config.use_ollama = use_ollama_env.lower() == "true"
    
    use_persistent_env = os.getenv("USE_PERSISTENT")
    if use_persistent_env:
        config.use_persistent = use_persistent_env.lower() == "true"
    
    collection_name_env = os.getenv("COLLECTION_NAME")
    if collection_name_env:
        config.collection_name = collection_name_env
    
    return config


def get_tuning_config() -> TuningConfig:
    """Get tuning configuration with environment variable overrides
    
    Environment variables that can be set:
        - MODEL_SIZE: "small" or "medium" (small=llama3.2:1b, medium=llama3:8b)
    - TUNING_BATCH_SIZE: batch size as integer (1-16)
    - TUNING_EPOCHS: number of epochs as integer (1-10)
    - TUNING_DEVICE: device like "auto", "cpu", "cuda", "mps"
    """
    config = TuningConfig()
    
    # Override with environment variables if set
    model_size_env = os.getenv("MODEL_SIZE")
    if model_size_env:
        model_size_lower = model_size_env.lower()
        if model_size_lower in ["small", "medium"]:
            config.model_size = model_size_lower
        else:
            print(f"Warning: Invalid MODEL_SIZE '{model_size_env}', must be 'small' or 'medium'. Using default.")
    
    tuning_batch_size_env = os.getenv("TUNING_BATCH_SIZE")
    if tuning_batch_size_env:
        try:
            batch_size = int(tuning_batch_size_env)
            if 1 <= batch_size <= 16:
                config.batch_size = batch_size
            else:
                print(f"Warning: TUNING_BATCH_SIZE {batch_size} out of range (1-16), using default")
        except ValueError:
            print(f"Warning: Invalid TUNING_BATCH_SIZE '{tuning_batch_size_env}', using default")
    
    tuning_epochs_env = os.getenv("TUNING_EPOCHS")
    if tuning_epochs_env:
        try:
            epochs = int(tuning_epochs_env)
            if 1 <= epochs <= 10:
                config.num_epochs = epochs
            else:
                print(f"Warning: TUNING_EPOCHS {epochs} out of range (1-10), using default")
        except ValueError:
            print(f"Warning: Invalid TUNING_EPOCHS '{tuning_epochs_env}', using default")
    
    tuning_device_env = os.getenv("TUNING_DEVICE")
    if tuning_device_env:
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if tuning_device_env.lower() in valid_devices:
            config.device = tuning_device_env.lower()
        else:
            print(f"Warning: Invalid TUNING_DEVICE '{tuning_device_env}', using default")
    
    return config