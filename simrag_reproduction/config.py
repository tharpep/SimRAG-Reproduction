"""
Simple Configuration Management
Basic settings for RAG system testing and operation
"""

import os
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """Simple configuration for RAG system"""
    
    # Model size settings
    model_size: str = "small"  # "small" for Qwen/Qwen2.5-1.5B-Instruct, "medium" for Qwen/Qwen2.5-7B-Instruct (both non-gated)
    
    # AI Provider settings
    use_ollama: bool = False  # True for Ollama (intermediate steps), False for Purdue API (default: False for synthetic QA)
    baseline_provider: str = "ollama"  # Default to Ollama for baseline (fast, reliable)
    
    # Vector store settings
    use_persistent: bool = True  # True for persistent storage, False for in-memory only
    collection_name: str = "simrag_docs"  # Name for Qdrant collection
    clear_on_ingest: bool = True  # Clear collection before ingesting new documents
    
    # Retrieval settings
    top_k: int = 5  # Number of documents to retrieve (1-20 recommended)
    similarity_threshold: float = 0.7  # Minimum similarity score (0.0-1.0)
    
    # Generation settings
    max_tokens: int = 100  # Maximum tokens in response (50-500 recommended, 100 is sufficient for most answers)
    temperature: float = 0.7  # Creativity level (0.0-1.0, lower = more focused)
    
    @property
    def model_name(self) -> str:
        """Get HuggingFace model ID based on model size configuration"""
        # Default to Qwen 2.5 Instruct (non-gated, non-thinking) - no HuggingFace authentication required
        return "Qwen/Qwen2.5-1.5B-Instruct" if self.model_size == "small" else "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class TuningConfig:
    """Simple configuration for model fine-tuning"""
    
    # Model size settings
    model_size: str = "small"  # "small" for Qwen/Qwen2.5-1.5B-Instruct, "medium" for Qwen/Qwen2.5-7B-Instruct (both non-gated)
    
    # Model settings
    device: str = "auto"  # Options: "auto", "cpu", "cuda", "mps" (for Apple Silicon)
    max_length: int = 512  # Maximum sequence length (256-1024 recommended)
    
    # Training settings (will be optimized based on model_size)
    num_epochs: int = 3  # Number of training epochs (1-10 recommended)
    batch_size: int = 8  # Batch size (8 for better GPU utilization, ~7-8GB VRAM)
    learning_rate: float = 5e-5  # Learning rate (1e-5 to 1e-3 recommended)
    # Note: 5e-5 is standard for QLoRA. With 3 epochs and effective batch=32, this is optimal.
    # Can lower to 4e-5 for more stability or raise to 6e-5 for faster convergence if needed.
    
    # QLoRA (Quantized Low-Rank Adaptation) settings
    # Note: QLoRA is always enabled - these settings control QLoRA hyperparameters
    lora_r: int = 16  # LoRA rank (8-64, higher = more expressive but larger adapters)
    lora_alpha: int = 32  # LoRA scaling factor (typically 2x lora_r)
    lora_dropout: float = 0.05  # Dropout for LoRA layers
    lora_target_modules: list = field(default_factory=lambda: None)  # Auto-detect from model if None
    bnb_4bit_quant_type: str = "nf4"  # Quantization type: "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True  # Use nested quantization
    
    # Model versioning
    version: str = "v1.0"  # Model version (e.g., v1.0, v1.1, v2.0)
    create_version_dir: bool = True  # Whether to create versioned subdirectories
    
    # SimRAG specific settings
    simrag_stage_1_epochs: int = 3  # Epochs for Stage 1 (instruction following) - increased from 1 for better convergence
    simrag_stage_2_epochs: int = 1  # Epochs for Stage 2 (domain adaptation) - kept at 1 as 2 epochs increased loss
    simrag_improvement_rounds: int = 5  # Number of Stage 2 rounds (1=no self-improvement, 2+=iterative refinement)
    simrag_questions_per_doc: int = 2  # Questions to generate per document - kept at 2 as 4 questions increased loss (~18 QA pairs)
    simrag_min_context_score: float = 0.7  # Minimum context similarity threshold
    
    # Reproducibility settings
    random_seed: int = 42  # Random seed for reproducibility (set via RANDOM_SEED env var)
    # Note: We do NOT use cuDNN deterministic mode as it causes 10x slowdown
    # Random seeds provide sufficient reproducibility without performance impact
    
    @property
    def optimized_batch_size(self) -> int:
        """Get batch size optimized for model size and GPU memory"""
        return 8 if self.model_size == "small" else 16  # Small: 8 (~7-8GB VRAM), Medium: 16
    
    @property
    def optimized_num_epochs(self) -> int:
        """Get number of epochs optimized for model size"""
        return 1 if self.model_size == "small" else 3  # Small: 1 for speed, Medium: 3 for quality
    
    @property
    def model_name(self) -> str:
        """Get HuggingFace model ID based on model size configuration"""
        # Default to Qwen 2.5 Instruct (non-gated, non-thinking) - no HuggingFace authentication required
        return "Qwen/Qwen2.5-1.5B-Instruct" if self.model_size == "small" else "Qwen/Qwen2.5-7B-Instruct"
    
    @property
    def output_dir(self) -> str:
        """Get output directory based on model size configuration"""
        # Use generic naming that works for any model
        model_suffix = "1b" if self.model_size == "small" else "8b"
        base_dir = f"./tuned_models/model_{model_suffix}"
        
        if self.create_version_dir:
            return f"{base_dir}/{self.version}"
        return base_dir
    
    @property
    def model_registry_path(self) -> str:
        """Get path to model registry metadata file"""
        model_suffix = "1b" if self.model_size == "small" else "8b"
        return f"./tuned_models/model_{model_suffix}/model_registry.json"
    
    def get_stage_output_dir(self, stage: str) -> str:
        """
        Get output directory for a specific SimRAG stage
        
        Args:
            stage: Stage name (e.g., "stage_1", "stage_2")
            
        Returns:
            Path to stage-specific output directory
        """
        model_suffix = "1b" if self.model_size == "small" else "8b"
        base_dir = f"./tuned_models/model_{model_suffix}"
        return f"{base_dir}/{stage}"


# Default configurations
DEFAULT_RAG_CONFIG = RAGConfig()
DEFAULT_TUNING_CONFIG = TuningConfig()


def get_rag_config() -> RAGConfig:
    """Get RAG configuration with environment variable overrides
    
    Environment variables that can be set:
        - MODEL_SIZE: "small" or "medium" (small=Qwen/Qwen2.5-1.5B-Instruct, medium=Qwen/Qwen2.5-7B-Instruct)
    - USE_OLLAMA: "true" or "false" (use Ollama vs Purdue API)
    - BASELINE_PROVIDER: "ollama" or "huggingface" (provider for baseline testing, default: "huggingface")
    - USE_PERSISTENT: "true" or "false" (persistent vs in-memory storage)
    - COLLECTION_NAME: name for Qdrant collection
    - MAX_TOKENS: maximum tokens in response (default: 100)
    - TEMPERATURE: creativity level 0.0-1.0 (default: 0.7)
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
    
    baseline_provider_env = os.getenv("BASELINE_PROVIDER")
    if baseline_provider_env:
        baseline_provider_lower = baseline_provider_env.lower()
        if baseline_provider_lower in ["ollama", "huggingface"]:
            config.baseline_provider = baseline_provider_lower
        else:
            print(f"Warning: Invalid BASELINE_PROVIDER '{baseline_provider_env}', must be 'ollama' or 'huggingface'. Using default.")
    # BASELINE_PROVIDER is independent of USE_OLLAMA
    # USE_OLLAMA controls intermediate steps (synthetic QA), not baseline
    # Baseline defaults to "huggingface" unless explicitly overridden
    
    use_persistent_env = os.getenv("USE_PERSISTENT")
    if use_persistent_env:
        config.use_persistent = use_persistent_env.lower() == "true"
    
    collection_name_env = os.getenv("COLLECTION_NAME")
    if collection_name_env:
        config.collection_name = collection_name_env
    
    max_tokens_env = os.getenv("MAX_TOKENS")
    if max_tokens_env:
        try:
            config.max_tokens = int(max_tokens_env)
        except ValueError:
            print(f"Warning: Invalid MAX_TOKENS '{max_tokens_env}', must be an integer. Using default.")
    
    temperature_env = os.getenv("TEMPERATURE")
    if temperature_env:
        try:
            config.temperature = float(temperature_env)
        except ValueError:
            print(f"Warning: Invalid TEMPERATURE '{temperature_env}', must be a float. Using default.")
    
    return config


def get_tuning_config() -> TuningConfig:
    """Get tuning configuration with environment variable overrides
    
    Environment variables that can be set:
        - MODEL_SIZE: "small" or "medium" (small=Qwen/Qwen2.5-1.5B-Instruct, medium=Qwen/Qwen2.5-7B-Instruct - both non-gated)
    - TUNING_BATCH_SIZE: batch size as integer (1-16)
    - TUNING_EPOCHS: number of epochs as integer (1-10)
    - TUNING_DEVICE: device like "auto", "cpu", "cuda", "mps"
    - LORA_R: LoRA rank as integer (8-64, default: 16)
    - LORA_ALPHA: LoRA alpha as integer (1-128, default: 32)
    - LORA_DROPOUT: LoRA dropout as float (0.0-1.0, default: 0.05)
    - SIMRAG_IMPROVEMENT_ROUNDS: number of Stage 2 self-improvement rounds (1-10, default: 2)
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
    
    # QLoRA hyperparameters
    lora_r_env = os.getenv("LORA_R")
    if lora_r_env:
        try:
            lora_r = int(lora_r_env)
            if 8 <= lora_r <= 64:
                config.lora_r = lora_r
            else:
                print(f"Warning: LORA_R {lora_r} out of range (8-64), using default")
        except ValueError:
            print(f"Warning: Invalid LORA_R '{lora_r_env}', using default")
    
    lora_alpha_env = os.getenv("LORA_ALPHA")
    if lora_alpha_env:
        try:
            lora_alpha = int(lora_alpha_env)
            if 1 <= lora_alpha <= 128:
                config.lora_alpha = lora_alpha
            else:
                print(f"Warning: LORA_ALPHA {lora_alpha} out of range (1-128), using default")
        except ValueError:
            print(f"Warning: Invalid LORA_ALPHA '{lora_alpha_env}', using default")
    
    lora_dropout_env = os.getenv("LORA_DROPOUT")
    if lora_dropout_env:
        try:
            lora_dropout = float(lora_dropout_env)
            if 0.0 <= lora_dropout <= 1.0:
                config.lora_dropout = lora_dropout
            else:
                print(f"Warning: LORA_DROPOUT {lora_dropout} out of range (0.0-1.0), using default")
        except ValueError:
            print(f"Warning: Invalid LORA_DROPOUT '{lora_dropout_env}', using default")
    
    # SimRAG self-improvement rounds
    simrag_improvement_rounds_env = os.getenv("SIMRAG_IMPROVEMENT_ROUNDS")
    if simrag_improvement_rounds_env:
        try:
            improvement_rounds = int(simrag_improvement_rounds_env)
            if 1 <= improvement_rounds <= 10:
                config.simrag_improvement_rounds = improvement_rounds
            else:
                print(f"Warning: SIMRAG_IMPROVEMENT_ROUNDS {improvement_rounds} out of range (1-10), using default")
        except ValueError:
            print(f"Warning: Invalid SIMRAG_IMPROVEMENT_ROUNDS '{simrag_improvement_rounds_env}', using default")
    
    # Random seed for reproducibility
    random_seed_env = os.getenv("RANDOM_SEED")
    if random_seed_env:
        try:
            random_seed = int(random_seed_env)
            config.random_seed = random_seed
        except ValueError:
            print(f"Warning: Invalid RANDOM_SEED '{random_seed_env}', using default")
    
    return config