"""
Basic Model Tuning
Simple fine-tuning implementation for language models
"""

import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from typing import List, Dict, Any, Optional
import os
import time
from dotenv import load_dotenv
from .model_registry import ModelRegistry, ModelVersion, get_model_registry

# Load environment variables (for HF_TOKEN)
load_dotenv()

# Set HuggingFace token from environment if available
# Transformers library automatically uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token


class BasicTuner:
    """Basic model fine-tuning system"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto", config=None):
        """
        Initialize the tuner
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ("auto", "cuda", "cpu")
            config: TuningConfig instance for versioning
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.config = config
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Model registry for versioning
        self.registry = None
        if self.config:
            self.registry = get_model_registry(self.config)
        
        print(f"Initializing tuner with model: {model_name}")
        print(f"Device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load the tokenizer and model
        
        Args:
            model_path: Optional path to fine-tuned model to load from.
                       If None, loads from base model (HuggingFace).
        """
        if model_path:
            print(f"Loading fine-tuned model from: {model_path}")
            
            # Check if this is a LoRA adapter (has adapter_config.json)
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            is_lora_adapter = os.path.exists(adapter_config_path)
            
            if is_lora_adapter:
                print("✓ Detected LoRA adapter model")
                
                # Load adapter config to get base model info
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", self.model_name)
                
                # Load tokenizer from adapter path (it's saved there)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load base model in 4-bit (QLoRA - always enabled)
                print(f"Loading base model '{base_model_name}' in 4-bit...")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type if self.config else "nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant if self.config else True,
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                
                # Load LoRA adapters
                print(f"Loading LoRA adapters from {model_path}...")
                self.model = PeftModel.from_pretrained(base_model, model_path)
                
                self.model.train()
                print(f"✓ LoRA adapter model loaded on {self.device}")
                
        else:
            print("Loading tokenizer...")
            
            # Use model name directly (should be HuggingFace Hub model ID)
            base_model = self.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Loading model...")
            
            # QLoRA: Always load in 4-bit for efficient training
            print("🔧 Using QLoRA: Loading base model in 4-bit...")
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type if self.config else "nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant if self.config else True,
            )
            
            # Load model in 4-bit
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            # Prepare model for k-bit training (gradient checkpointing disabled to avoid hangs)
            # Note: use_reentrant=False is recommended for PyTorch 2.5+ but we disable checkpointing entirely
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
            
            # Configure LoRA
            target_modules = self.config.lora_target_modules if self.config else None
            if target_modules is None:
                # Auto-detect target modules based on model architecture
                # Qwen2.5 uses standard transformer architecture (q_proj, k_proj, v_proj, o_proj)
                # Older Qwen1 models used c_attn, but Qwen2.5 is standard
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            lora_config = LoraConfig(
                r=self.config.lora_r if self.config else 16,
                lora_alpha=self.config.lora_alpha if self.config else 32,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout if self.config else 0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA adapters
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✓ LoRA adapters applied: {trainable_params:,} / {total_params:,} parameters trainable ({100 * trainable_params / total_params:.2f}%)")
            
            print(f"Model loaded on {self.device}")
    
    def prepare_data(self, texts: List[str], max_length: int = 512) -> Dataset:
        """
        Prepare training data
        
        Args:
            texts: List of training texts
            max_length: Maximum sequence length
            
        Returns:
            Hugging Face Dataset
        """
        print(f"Preparing {len(texts)} training examples...")
        
        # Tokenize texts
        # Note: Don't use return_tensors="pt" here - DataCollator will handle tensor conversion
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # DataCollator will handle padding
                max_length=max_length,
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        # Tokenize and remove the original "text" field - DataCollator only needs tokenized fields
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]  # Remove original text field after tokenization
        )
        
        return tokenized_dataset
    
    def setup_trainer(self, 
                     train_dataset: Dataset,
                     output_dir: str = "./tuned_model",
                     num_epochs: int = 3,
                     batch_size: int = 4,
                     learning_rate: float = 5e-5,
                     warmup_steps: int = 100):
        """
        Setup the trainer
        
        Args:
            train_dataset: Training dataset
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        print("Setting up trainer...")
        
        # Training arguments
        # Note: output_dir may be updated by train_model() to include version number
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=True,  # Remove "text" field after tokenization - DataCollator only needs tokenized fields
            # Speed optimizations (Conservative settings - 2x speedup, ~7-8GB VRAM)
            fp16=True,  # Mixed precision training (2x faster on GPU, uses less memory)
            dataloader_num_workers=2,  # Parallel data loading (0→2 for faster preprocessing)
            gradient_accumulation_steps=2,  # Simulate larger batch size (effective batch = batch_size * 2)
            dataloader_pin_memory=True,  # Faster GPU transfer (only useful with GPU)
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print("Trainer setup complete")
    
    def train(self, notes: Optional[str] = None, version_str: Optional[str] = None, experiment_run_id: Optional[str] = None):
        """
        Start training with version tracking
        
        Args:
            notes: Training notes for versioning
            version_str: Optional pre-created version string. If provided, uses this version
                        instead of creating a new one. This allows setting output_dir before training.
            experiment_run_id: Optional experiment run ID to link versions from same experiment
        
        Returns:
            ModelVersion object if successful, None otherwise
        """
        if not self.trainer:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        # Set random seeds for reproducibility (if config has random_seed)
        if self.config and hasattr(self.config, 'random_seed'):
            seed = self.config.random_seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Create or get version BEFORE training so we can set output_dir correctly
        new_version = None
        if self.registry and self.config:
            if version_str:
                # Use provided version (already created)
                new_version = self.registry.get_version(version_str)
                if not new_version:
                    raise ValueError(f"Version {version_str} not found in registry")
                # Update experiment_run_id if provided
                if experiment_run_id and new_version.experiment_run_id != experiment_run_id:
                    new_version.experiment_run_id = experiment_run_id
                    self.registry.register_version(new_version)
            else:
                # Create new version with experiment_run_id if provided
                # QLoRA is always enabled, so always include LoRA metadata
                lora_r = self.config.lora_r if self.config else 16
                lora_alpha = self.config.lora_alpha if self.config else 32
                
                new_version = self.registry.create_new_version(
                    model_name=self.model_name,
                    base_model=self.model_name,  # For now, same as model_name
                    training_epochs=self.config.optimized_num_epochs,
                    batch_size=self.config.optimized_batch_size,
                    learning_rate=self.config.learning_rate,
                    device=self.device,
                    notes=notes,
                    experiment_run_id=experiment_run_id,
                    is_lora=True,  # QLoRA is always enabled
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
                # Register it now so it's available
                self.registry.register_version(new_version)
            
            # Update trainer output_dir to include version BEFORE training
            # This ensures trainer saves directly to the correct location
            original_output_dir = self.trainer.args.output_dir
            version_output_dir = os.path.join(original_output_dir, new_version.version)
            self.trainer.args.output_dir = version_output_dir
            os.makedirs(version_output_dir, exist_ok=True)
            print(f"Training will save to: {version_output_dir}")
        
        print("Starting training...")
        start_time = time.time()
        
        # Train the model
        self.trainer.train()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds!")
        
        # Update version with training results
        if new_version:
            # Get training metrics
            final_loss = None
            if hasattr(self.trainer.state, 'log_history') and self.trainer.state.log_history:
                final_loss = self.trainer.state.log_history[-1].get('train_loss')
            
            # Get model size
            model_size_mb = None
            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            # Update version with training results
            new_version.training_time_seconds = training_time
            new_version.final_loss = final_loss
            new_version.model_size_mb = model_size_mb
            
            # Re-register with updated metrics
            self.registry.register_version(new_version)
            
            return new_version
        
        return None
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the fine-tuned model with versioning"""
        if not self.trainer:
            raise ValueError("No trainer available. Train a model first.")
        
        # Use config output directory if available, otherwise use provided or default
        if output_dir is None:
            if self.config:
                output_dir = self.config.output_dir
            else:
                output_dir = "./tuned_model"
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if model is a PEFT model (LoRA)
        is_peft_model = hasattr(self.model, 'peft_config')
        
        # Always save LoRA adapters (QLoRA is the only method)
        print(f"Saving LoRA adapters to {output_dir}...")
        # Save only the adapters (lightweight)
        self.model.save_pretrained(output_dir)
        # Save tokenizer too
        self.tokenizer.save_pretrained(output_dir)
        adapter_size_mb = sum(
            os.path.getsize(os.path.join(output_dir, f)) 
            for f in os.listdir(output_dir) 
            if os.path.isfile(os.path.join(output_dir, f))
        ) / (1024 * 1024)
        print(f"✓ LoRA adapters saved! ({adapter_size_mb:.1f} MB)")
        
        # Save model info if registry is available
        if self.registry:
            model_info = self.get_model_info()
            info_path = os.path.join(output_dir, "model_info.json")
            with open(info_path, 'w') as f:
                import json
                json.dump(model_info, f, indent=2)
            print(f"Model info saved to {info_path}")
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text using the fine-tuned model
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get actual device from model (handles device_map="auto" case)
        # Try to get device from model parameters
        try:
            model_device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            # Fallback to self.device if model has no parameters or device_map wasn't used
            model_device = self.device
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(model_device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }
        
        # Add LoRA-specific info if applicable
        if hasattr(self.model, 'peft_config') and self.model.peft_config:
            info["is_lora"] = True
            info["trainable_percentage"] = 100 * info["trainable_parameters"] / info["num_parameters"]
            # Get LoRA config if available
            peft_config_values = list(self.model.peft_config.values())
            if peft_config_values:
                peft_config = peft_config_values[0]  # Get first config
                info["lora_r"] = peft_config.r
                info["lora_alpha"] = peft_config.lora_alpha
                info["lora_target_modules"] = peft_config.target_modules
        else:
            info["is_lora"] = False
        
        return info

