"""
Ollama Integration for Fine-tuned Models
Automatically convert and register fine-tuned models with Ollama
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
import time

logger = logging.getLogger(__name__)


def check_ollama_available() -> bool:
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_ollama_model_from_adapter(
    adapter_path: str,
    model_name: str,
    base_model: str = "qwen2.5:1.5b",
    force: bool = False
) -> bool:
    """
    Create an Ollama model from LoRA adapters (SIMPLE METHOD - no GGUF conversion needed!)
    
    This uses Ollama's built-in adapter support which is much simpler than GGUF conversion.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        model_name: Name for the model in Ollama (e.g., "simrag-stage1-v1.0")
        base_model: Base model in Ollama (must already be pulled)
        force: If True, overwrite existing model
        
    Returns:
        True if successful, False otherwise
    """
    adapter_path = Path(adapter_path)
    
    if not check_ollama_available():
        logger.warning("Ollama not available - skipping model creation")
        return False
    
    logger.info(f"Creating Ollama model '{model_name}' from adapters at {adapter_path}")
    
    # Check if adapter files exist
    adapter_file = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    
    if not adapter_file.exists():
        logger.error(f"Adapter file not found: {adapter_file}")
        return False
    
    if not adapter_config.exists():
        logger.error(f"Adapter config not found: {adapter_config}")
        logger.error("Ollama requires adapter_config.json in the same directory as adapter_model.safetensors")
        return False
    
    # Use absolute path for adapter file in Modelfile
    # Ollama's ADAPTER directive expects the path to the adapter_model.safetensors file
    # Note: adapter_config.json must be in the same directory as the adapter file
    # Convert to forward slashes for cross-platform compatibility
    adapter_file_abs = adapter_file.resolve()
    adapter_file_str = str(adapter_file_abs).replace('\\', '/')
    
    # Create Modelfile
    modelfile_path = adapter_path / "Modelfile"
    modelfile_content = f"""# Fine-tuned model: {model_name}
FROM {base_model}
ADAPTER {adapter_file_str}

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# System message
SYSTEM \"\"\"You are a helpful AI assistant trained to answer questions accurately and concisely based on provided context.\"\"\"
"""
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    logger.info(f"Modelfile created at {modelfile_path}")
    logger.info(f"Adapter file path: {adapter_file_abs}")
    logger.info(f"Adapter config path: {adapter_config.resolve()}")
    logger.info(f"Modelfile content:")
    logger.info(f"---")
    for line in modelfile_content.split('\n')[:5]:  # Show first 5 lines
        logger.info(f"  {line}")
    logger.info(f"---")
    
    # Create model in Ollama
    # Run from adapter directory so Ollama can find adapter_config.json
    # Ollama needs adapter_config.json in its working directory when processing ADAPTER directive
    modelfile_path_abs = modelfile_path.resolve()
    adapter_dir_abs = adapter_path.resolve()
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path_abs)]
    
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"Working directory: {adapter_dir_abs}")
    
    try:
        # Use UTF-8 encoding with error handling for cross-platform compatibility
        # Ollama output may contain binary data or special characters
        # Run from adapter directory so adapter_config.json is in the working directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid characters instead of failing
            timeout=300,  # 5 minutes timeout
            cwd=str(adapter_dir_abs)  # Run from adapter directory
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully created Ollama model: {model_name}")
            logger.info(f"  You can now use it with: ollama run {model_name}")
            return True
        else:
            # Log both stdout and stderr for debugging
            error_msg = result.stderr if result.stderr else result.stdout
            logger.error(f"Failed to create Ollama model (exit code {result.returncode})")
            if error_msg:
                logger.error(f"Error output: {error_msg[:500]}")  # Limit error message length
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Ollama model creation timed out")
        return False
    except Exception as e:
        logger.error(f"Error creating Ollama model: {e}")
        return False


def ensure_base_model_pulled(base_model: str) -> bool:
    """
    Ensure base model is pulled in Ollama
    
    Args:
        base_model: Base model name (e.g., "qwen2.5:1.5b")
        
    Returns:
        True if model is available, False otherwise
    """
    if not check_ollama_available():
        return False
    
    # Check if model exists
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if base_model in result.stdout:
        logger.info(f"✓ Base model {base_model} already available")
        return True
    
    # Pull the model
    logger.info(f"Pulling base model {base_model}...")
    result = subprocess.run(
        ["ollama", "pull", base_model],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode == 0:
        logger.info(f"✓ Successfully pulled {base_model}")
        return True
    else:
        logger.error(f"Failed to pull {base_model}: {result.stderr}")
        return False


def get_ollama_model_name(stage: str, version: str, model_size: str = "small") -> str:
    """
    Generate consistent Ollama model name
    
    Args:
        stage: Stage name (e.g., "stage_1", "stage_2")
        version: Version string (e.g., "v1.0")
        model_size: Model size ("small" or "medium")
        
    Returns:
        Ollama model name (e.g., "simrag-1b-stage1-v1.0")
    """
    size_suffix = "1b" if model_size == "small" else "7b"
    stage_num = stage.replace("stage_", "")
    version_clean = version.replace(".", "-")
    return f"simrag-{size_suffix}-stage{stage_num}-{version_clean}"


def register_model_with_ollama(
    adapter_path: str,
    stage: str,
    version: str,
    model_size: str = "small"
) -> Optional[str]:
    """
    High-level function to register a fine-tuned model with Ollama
    
    Args:
        adapter_path: Path to LoRA adapter directory
        stage: Stage name (e.g., "stage_1", "stage_2")
        version: Version string (e.g., "v1.0")
        model_size: Model size ("small" or "medium")
        
    Returns:
        Ollama model name if successful, None otherwise
    """
    # Determine base model
    if model_size == "small":
        base_model = "qwen2.5:1.5b"
    else:
        base_model = "qwen2.5:7b"
    
    # Ensure base model is available
    if not ensure_base_model_pulled(base_model):
        logger.warning("Could not ensure base model is available")
        return None
    
    # Generate model name
    model_name = get_ollama_model_name(stage, version, model_size)
    
    # Create model
    success = create_ollama_model_from_adapter(
        adapter_path=adapter_path,
        model_name=model_name,
        base_model=base_model
    )
    
    if success:
        return model_name
    else:
        return None


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Register fine-tuned model with Ollama")
    parser.add_argument("adapter_path", help="Path to LoRA adapter directory")
    parser.add_argument("--stage", required=True, help="Stage name (e.g., stage_1)")
    parser.add_argument("--version", required=True, help="Version (e.g., v1.0)")
    parser.add_argument("--model-size", default="small", choices=["small", "medium"], help="Model size")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    model_name = register_model_with_ollama(
        adapter_path=args.adapter_path,
        stage=args.stage,
        version=args.version,
        model_size=args.model_size
    )
    
    if model_name:
        print(f"\n✓ Model registered with Ollama: {model_name}")
        print(f"  Test it with: ollama run {model_name}")
    else:
        print("\n✗ Failed to register model with Ollama")
        exit(1)


if __name__ == "__main__":
    main()

