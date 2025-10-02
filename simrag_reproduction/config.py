"""
Configuration Management
Handles settings for RAG system testing and operation
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TuningConfig:
    """Configuration for model fine-tuning"""
    
    # Model settings
    model_name: str = "qwen3:1.7b"  # Default to smaller model
    device: str = "auto"
    max_length: int = 512
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    
    # Output settings
    output_dir: str = "./tuned_models/qwen_1.7b"
    save_steps: int = 500
    logging_steps: int = 10
    
    # Generation settings
    max_generation_length: int = 100
    temperature: float = 0.7
    
    # Data settings
    train_split: float = 0.8
    validation_split: float = 0.2
    
    @classmethod
    def for_qwen_1_7b(cls) -> 'TuningConfig':
        """Configuration for Qwen 1.7B model (laptop-friendly)"""
        return cls(
            model_name="qwen3:1.7b",
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=5,
            output_dir="./tuned_models/qwen_1.7b",
            max_length=512
        )
    
    @classmethod
    def for_qwen_8b(cls) -> 'TuningConfig':
        """Configuration for Qwen 8B model (PC with more VRAM)"""
        return cls(
            model_name="qwen3:8b",
            batch_size=2,  # Smaller batch for 8B model
            learning_rate=5e-5,
            num_epochs=3,
            output_dir="./tuned_models/qwen_8b",
            max_length=512
        )
    
    @classmethod
    def for_quick_test(cls) -> 'TuningConfig':
        """Configuration for quick testing with 1.7B model"""
        return cls(
            model_name="qwen3:1.7b",
            batch_size=4,
            learning_rate=5e-5,
            num_epochs=1,
            warmup_steps=10,
            save_steps=100,
            logging_steps=5,
            output_dir="./tuned_models/quick_test"
        )
    
    def print_config(self):
        """Print current configuration"""
        print("=== Tuning Configuration ===")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Max Length: {self.max_length}")
        print(f"Output Dir: {self.output_dir}")
        print("=" * 30)
    
    def create_output_dirs(self):
        """Create output directories if they don't exist"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    
    # AI Provider settings
    use_ollama: bool = True
    use_purdue: bool = False
    model_name: str = "qwen3:1.7b"
    purdue_api_key: Optional[str] = None
    
    # Storage settings
    use_persistent: bool = False
    collection_name: str = "documents"
    
    # RAG settings
    context_limit: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    # Demo settings
    demo_documents: bool = True
    interactive_mode: bool = False
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        return cls(
            use_ollama=os.getenv('USE_OLLAMA', 'true').lower() == 'true',
            use_purdue=bool(os.getenv('PURDUE_API_KEY')),
            model_name=os.getenv('MODEL_NAME', 'qwen3:1.7b'),
            purdue_api_key=os.getenv('PURDUE_API_KEY'),
            use_persistent=os.getenv('USE_PERSISTENT', 'false').lower() == 'true',
            collection_name=os.getenv('COLLECTION_NAME', 'documents'),
            context_limit=int(os.getenv('CONTEXT_LIMIT', '3')),
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '100')),
            demo_documents=os.getenv('DEMO_DOCUMENTS', 'true').lower() == 'true',
            interactive_mode=os.getenv('INTERACTIVE_MODE', 'false').lower() == 'true'
        )
    
    def to_env_dict(self) -> dict:
        """Convert config to environment variable dictionary"""
        return {
            'USE_OLLAMA': str(self.use_ollama).lower(),
            'USE_PURDUE': str(self.use_purdue).lower(),
            'MODEL_NAME': self.model_name,
            'PURDUE_API_KEY': self.purdue_api_key or '',
            'USE_PERSISTENT': str(self.use_persistent).lower(),
            'COLLECTION_NAME': self.collection_name,
            'CONTEXT_LIMIT': str(self.context_limit),
            'CHUNK_SIZE': str(self.chunk_size),
            'CHUNK_OVERLAP': str(self.chunk_overlap),
            'DEMO_DOCUMENTS': str(self.demo_documents).lower(),
            'INTERACTIVE_MODE': str(self.interactive_mode).lower()
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=== RAG Configuration ===")
        print(f"AI Provider: {'Ollama' if self.use_ollama else 'Purdue'}")
        if self.use_ollama:
            print(f"  Model: {self.model_name}")
        if self.use_purdue:
            print(f"  API Key: {'Set' if self.purdue_api_key else 'Not set'}")
        print(f"Storage: {'Persistent' if self.use_persistent else 'In-memory'}")
        print(f"Collection: {self.collection_name}")
        print(f"Context Limit: {self.context_limit}")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Demo Documents: {self.demo_documents}")
        print(f"Interactive Mode: {self.interactive_mode}")
        print("=" * 25)


# Default configurations for common use cases
DEFAULT_CONFIGS = {
    "laptop": RAGConfig(
        use_ollama=True,
        use_purdue=False,
        use_persistent=False,
        model_name="qwen3:1.7b",
        demo_documents=True
    ),
    "pc": RAGConfig(
        use_ollama=True,
        use_purdue=False,
        use_persistent=True,
        model_name="qwen3:8b",
        demo_documents=True
    ),
    "local": RAGConfig(
        use_ollama=True,
        use_purdue=False,
        use_persistent=False,
        demo_documents=True
    ),
    "persistent": RAGConfig(
        use_ollama=True,
        use_purdue=False,
        use_persistent=True,
        demo_documents=True
    ),
    "purdue": RAGConfig(
        use_ollama=False,
        use_purdue=True,
        use_persistent=False,
        demo_documents=True
    ),
    "production": RAGConfig(
        use_ollama=True,
        use_purdue=False,
        use_persistent=True,
        demo_documents=False,
        context_limit=5
    )
}

# Tuning configurations
TUNING_CONFIGS = {
    "qwen_1.7b": TuningConfig.for_qwen_1_7b(),
    "qwen_8b": TuningConfig.for_qwen_8b(),
    "quick": TuningConfig.for_quick_test(),
    "default": TuningConfig(),
    # Aliases for easy switching
    "laptop": TuningConfig.for_qwen_1_7b(),
    "pc": TuningConfig.for_qwen_8b()
}
