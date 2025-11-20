"""
Convert HuggingFace Model to GGUF Format
Utility to convert merged models to GGUF format for Ollama
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "Q4_K_M"
) -> str:
    """
    Convert HuggingFace model to GGUF format using llama.cpp
    
    Args:
        model_path: Path to HuggingFace model directory
        output_path: Path to save GGUF file
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
        
    Returns:
        Path to GGUF file
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    logger.info(f"Converting {model_path} to GGUF format")
    logger.info(f"Quantization: {quantization}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if llama.cpp convert script is available
    # Note: This requires llama.cpp to be installed
    # Users can install it via: git clone https://github.com/ggerganov/llama.cpp
    
    try:
        # Try using llama-cpp-python's conversion
        import llama_cpp
        logger.info("Using llama-cpp-python for conversion")
        
        # llama-cpp-python doesn't have built-in conversion
        # We need to use the llama.cpp convert.py script
        raise ImportError("llama-cpp-python doesn't support conversion directly")
        
    except ImportError:
        logger.info("llama-cpp-python not available, checking for llama.cpp scripts")
    
    # Look for llama.cpp convert script
    # Common locations
    possible_paths = [
        Path.home() / "llama.cpp" / "convert.py",
        Path("llama.cpp") / "convert.py",
        Path("../llama.cpp") / "convert.py",
    ]
    
    convert_script = None
    for path in possible_paths:
        if path.exists():
            convert_script = path
            break
    
    if convert_script is None:
        raise FileNotFoundError(
            "llama.cpp convert.py not found. Please install llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make\n"
            "Or use the simpler create_ollama_modelfile() function which doesn't require GGUF conversion."
        )
    
    # Convert to GGUF
    logger.info(f"Running {convert_script}")
    
    # First convert to F16 GGUF
    temp_gguf = output_path.parent / f"{output_path.stem}_f16.gguf"
    
    cmd_convert = [
        "python",
        str(convert_script),
        str(model_path),
        "--outfile", str(temp_gguf),
        "--outtype", "f16"
    ]
    
    logger.info(f"Command: {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise RuntimeError(f"Failed to convert model: {result.stderr}")
    
    # Then quantize to target format
    quantize_script = convert_script.parent / "quantize"
    if not quantize_script.exists():
        quantize_script = convert_script.parent / "quantize.exe"  # Windows
    
    if quantize_script.exists():
        logger.info(f"Quantizing to {quantization}")
        
        cmd_quantize = [
            str(quantize_script),
            str(temp_gguf),
            str(output_path),
            quantization
        ]
        
        logger.info(f"Command: {' '.join(cmd_quantize)}")
        result = subprocess.run(cmd_quantize, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            # Use F16 version if quantization fails
            logger.warning(f"Using F16 version instead")
            temp_gguf.rename(output_path)
        else:
            # Remove temp file
            temp_gguf.unlink()
    else:
        logger.warning("Quantize binary not found, using F16 version")
        temp_gguf.rename(output_path)
    
    logger.info(f"✓ GGUF file saved to {output_path}")
    
    return str(output_path)


def create_ollama_modelfile(
    model_name: str,
    base_model: str = "qwen2.5:1.5b",
    adapter_path: Optional[str] = None,
    gguf_path: Optional[str] = None,
    output_path: str = "Modelfile"
) -> str:
    """
    Create Ollama Modelfile (simpler alternative to GGUF conversion)
    
    Ollama can use adapters directly or reference merged models!
    
    Args:
        model_name: Name for the model in Ollama
        base_model: Base model in Ollama (e.g., "qwen2.5:1.5b")
        adapter_path: Path to LoRA adapters (if using adapters)
        gguf_path: Path to GGUF file (if using converted model)
        output_path: Path to save Modelfile
        
    Returns:
        Path to Modelfile
    """
    output_path = Path(output_path)
    
    logger.info(f"Creating Ollama Modelfile for {model_name}")
    
    modelfile_content = f"# Modelfile for {model_name}\n\n"
    
    if gguf_path:
        # Use converted GGUF model
        modelfile_content += f"FROM {gguf_path}\n"
    elif adapter_path:
        # Use base model + adapters
        modelfile_content += f"FROM {base_model}\n"
        modelfile_content += f"ADAPTER {adapter_path}/adapter_model.safetensors\n"
    else:
        raise ValueError("Either gguf_path or adapter_path must be provided")
    
    # Add default parameters
    modelfile_content += "\n# Parameters\n"
    modelfile_content += "PARAMETER temperature 0.7\n"
    modelfile_content += "PARAMETER top_p 0.9\n"
    modelfile_content += "PARAMETER stop \"<|endoftext|>\"\n"
    modelfile_content += "PARAMETER stop \"<|im_end|>\"\n"
    
    # Add system message
    modelfile_content += "\n# System message\n"
    modelfile_content += 'SYSTEM """You are a helpful AI assistant trained to answer questions accurately and concisely based on provided context."""\n'
    
    # Write Modelfile
    with open(output_path, 'w') as f:
        f.write(modelfile_content)
    
    logger.info(f"✓ Modelfile saved to {output_path}")
    logger.info(f"\nTo create the model in Ollama, run:")
    logger.info(f"  ollama create {model_name} -f {output_path}")
    
    return str(output_path)


def main():
    """CLI entry point for GGUF conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert model to GGUF or create Ollama Modelfile")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # GGUF conversion subcommand
    gguf_parser = subparsers.add_parser("gguf", help="Convert to GGUF format")
    gguf_parser.add_argument("model_path", help="Path to HuggingFace model")
    gguf_parser.add_argument("output_path", help="Path to save GGUF file")
    gguf_parser.add_argument("--quantization", default="Q4_K_M", help="Quantization type")
    
    # Modelfile creation subcommand
    modelfile_parser = subparsers.add_parser("modelfile", help="Create Ollama Modelfile")
    modelfile_parser.add_argument("model_name", help="Name for model in Ollama")
    modelfile_parser.add_argument("--base-model", default="qwen2.5:1.5b", help="Base model in Ollama")
    modelfile_parser.add_argument("--adapter-path", help="Path to LoRA adapters")
    modelfile_parser.add_argument("--gguf-path", help="Path to GGUF file")
    modelfile_parser.add_argument("--output", default="Modelfile", help="Output Modelfile path")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.command == "gguf":
        convert_to_gguf(
            model_path=args.model_path,
            output_path=args.output_path,
            quantization=args.quantization
        )
    elif args.command == "modelfile":
        create_ollama_modelfile(
            model_name=args.model_name,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            gguf_path=args.gguf_path,
            output_path=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

