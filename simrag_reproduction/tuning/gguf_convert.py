"""
Convert merged HuggingFace models to GGUF format for Ollama
This provides much better GPU utilization than loading FP16 models directly
"""

import subprocess
import logging
from pathlib import Path
import os
import sys

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model_path: str,
    output_path: str = None,
    quantization: str = "Q4_K_M"
) -> str:
    """
    Convert a HuggingFace model to GGUF format using llama.cpp

    Args:
        model_path: Path to HuggingFace model directory (merged model)
        output_path: Output path for GGUF file (optional, auto-generated if None)
        quantization: Quantization level (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)

    Returns:
        Path to the converted GGUF file
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Auto-generate output path if not provided
    if output_path is None:
        output_path = model_path.parent / f"{model_path.name}_{quantization.lower()}.gguf"
    else:
        output_path = Path(output_path)

    # Check if already converted
    if output_path.exists():
        logger.info(f"GGUF model already exists: {output_path}")
        return str(output_path)

    logger.info(f"Converting {model_path} to GGUF format ({quantization})...")
    logger.info("This requires llama.cpp. Checking for llama-quantize...")

    # Try to find llama.cpp tools
    quantize_cmd = _find_llama_cpp_tool()

    if not quantize_cmd:
        logger.error("llama.cpp not found!")
        logger.error("Please install llama.cpp:")
        logger.error("  git clone https://github.com/ggerganov/llama.cpp.git")
        logger.error("  cd llama.cpp && make -j8")
        logger.error("Then add it to PATH or set LLAMA_CPP_PATH environment variable")
        raise RuntimeError("llama.cpp tools not found")

    # First convert to F16 GGUF
    f16_path = model_path.parent / f"{model_path.name}_f16.gguf"

    if not f16_path.exists():
        logger.info("Step 1: Converting to F16 GGUF...")

        # For CMake builds, the script is in the root llama.cpp directory, not build/bin
        quantize_dir = Path(quantize_cmd).parent
        if "build" in str(quantize_dir):
            # Go up to llama.cpp root directory
            llama_cpp_root = quantize_dir.parent.parent
        else:
            llama_cpp_root = quantize_dir

        convert_script = llama_cpp_root / "convert_hf_to_gguf.py"

        if not convert_script.exists():
            # Try alternative name
            convert_script = llama_cpp_root / "convert-hf-to-gguf.py"

        if not convert_script.exists():
            raise RuntimeError(f"convert_hf_to_gguf.py not found in {llama_cpp_root}")

        result = subprocess.run(
            [sys.executable, str(convert_script), str(model_path), "--outfile", str(f16_path), "--outtype", "f16"],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            raise RuntimeError(f"Failed to convert to F16 GGUF: {result.stderr}")

        logger.info(f"✓ F16 GGUF created: {f16_path}")
    else:
        logger.info(f"Using existing F16 GGUF: {f16_path}")

    # Then quantize if needed
    if quantization.upper() != "F16":
        logger.info(f"Step 2: Quantizing to {quantization}...")
        result = subprocess.run(
            [quantize_cmd, str(f16_path), str(output_path), quantization],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            raise RuntimeError(f"Failed to quantize: {result.stderr}")

        logger.info(f"✓ Quantized GGUF created: {output_path}")

        # Clean up F16 if we quantized to something else
        if f16_path.exists() and f16_path != output_path:
            f16_path.unlink()
            logger.info(f"Cleaned up intermediate F16 file")
    else:
        # Just use the F16 version
        output_path = f16_path

    logger.info(f"✓ GGUF conversion complete: {output_path}")
    return str(output_path)


def _find_llama_cpp_tool() -> str:
    """Find llama.cpp quantize tool"""
    # Check environment variable
    llama_cpp_path = os.getenv("LLAMA_CPP_PATH")
    if llama_cpp_path:
        # Check both old and new build locations
        for tool_name in ["llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"]:
            quantize_tool = Path(llama_cpp_path) / tool_name
            if quantize_tool.exists():
                return str(quantize_tool)

    # Check common locations (both old Makefile and new CMake builds)
    common_locations = [
        "llama.cpp/build/bin/llama-quantize",
        "llama.cpp/build/bin/quantize",
        "llama.cpp/llama-quantize",
        "llama.cpp/quantize",
        "../llama.cpp/build/bin/llama-quantize",
        "../llama.cpp/build/bin/quantize",
        "../llama.cpp/llama-quantize",
        "../llama.cpp/quantize",
        str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"),
        str(Path.home() / "llama.cpp" / "build" / "bin" / "quantize"),
        str(Path.home() / "llama.cpp" / "llama-quantize"),
        str(Path.home() / "llama.cpp" / "quantize"),
    ]

    for location in common_locations:
        path = Path(location)
        if path.exists():
            return str(path)
        # Try with .exe on Windows
        if os.name == 'nt':
            path_exe = Path(str(location) + ".exe")
            if path_exe.exists():
                return str(path_exe)

    # Try PATH
    try:
        result = subprocess.run(
            ["which", "llama-quantize"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    try:
        result = subprocess.run(
            ["which", "quantize"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace model to GGUF")
    parser.add_argument("model_path", help="Path to HuggingFace model directory")
    parser.add_argument("--output", "-o", help="Output GGUF file path")
    parser.add_argument("--quantization", "-q", default="Q4_K_M",
                       help="Quantization level (Q4_K_M, Q5_K_M, Q8_0, F16)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        gguf_path = convert_to_gguf(args.model_path, args.output, args.quantization)
        print(f"\n✓ Success! GGUF model: {gguf_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)
