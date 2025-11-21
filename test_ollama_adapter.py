"""
Test script to diagnose and fix Ollama adapter loading issue
"""

import subprocess
import sys
from pathlib import Path
import shutil
import tempfile
import os

# Fix Unicode output on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def test_ollama_adapter(checkpoint_path: str):
    """Test Ollama adapter loading with diagnostic output"""

    checkpoint_dir = Path(checkpoint_path).resolve()

    print(f"=== Testing Ollama Adapter Loading ===")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Directory exists: {checkpoint_dir.exists()}")
    print(f"Is directory: {checkpoint_dir.is_dir()}")

    # Check required files
    adapter_file = checkpoint_dir / "adapter_model.safetensors"
    adapter_config = checkpoint_dir / "adapter_config.json"

    print(f"\nRequired files:")
    print(f"  adapter_model.safetensors: {adapter_file.exists()} ({adapter_file})")
    print(f"  adapter_config.json: {adapter_config.exists()} ({adapter_config})")

    if not adapter_file.exists() or not adapter_config.exists():
        print("\n❌ Required files missing!")
        return False

    # Check Ollama availability
    print(f"\nChecking Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print(f"❌ Ollama not running!")
            return False
        print(f"✓ Ollama is running")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return False

    # Check base model
    base_model = "qwen2.5:1.5b"
    print(f"\nChecking base model: {base_model}")
    if base_model not in result.stdout:
        print(f"Base model not found. Pulling...")
        result = subprocess.run(
            ["ollama", "pull", base_model],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to pull base model")
            return False
    print(f"✓ Base model available")

    # Method 1: Try absolute path with forward slashes
    print(f"\n=== Method 1: Absolute path with forward slashes ===")
    adapter_abs = str(adapter_file.resolve()).replace('\\', '/')
    modelfile_content_1 = f"""FROM {base_model}
ADAPTER {adapter_abs}
"""

    modelfile_path_1 = checkpoint_dir / "Modelfile_test1"
    with open(modelfile_path_1, 'w') as f:
        f.write(modelfile_content_1)

    print(f"Modelfile path: {modelfile_path_1}")
    print(f"Adapter path in Modelfile: {adapter_abs}")
    print(f"Working directory: {checkpoint_dir}")

    result = subprocess.run(
        ["ollama", "create", "test-adapter-1", "-f", str(modelfile_path_1.resolve())],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=str(checkpoint_dir),
        timeout=60
    )

    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout[:500]}")
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")

    if result.returncode == 0:
        print(f"✓ Method 1 SUCCESS!")
        subprocess.run(["ollama", "rm", "test-adapter-1"], capture_output=True)
        return True

    # Method 2: Try relative path from checkpoint directory
    print(f"\n=== Method 2: Relative path from checkpoint directory ===")
    modelfile_content_2 = f"""FROM {base_model}
ADAPTER ./adapter_model.safetensors
"""

    modelfile_path_2 = checkpoint_dir / "Modelfile_test2"
    with open(modelfile_path_2, 'w') as f:
        f.write(modelfile_content_2)

    print(f"Modelfile path: {modelfile_path_2}")
    print(f"Adapter path in Modelfile: ./adapter_model.safetensors")
    print(f"Working directory: {checkpoint_dir}")

    result = subprocess.run(
        ["ollama", "create", "test-adapter-2", "-f", str(modelfile_path_2.resolve())],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=str(checkpoint_dir),
        timeout=60
    )

    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout[:500]}")
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")

    if result.returncode == 0:
        print(f"✓ Method 2 SUCCESS!")
        subprocess.run(["ollama", "rm", "test-adapter-2"], capture_output=True)
        return True

    # Method 3: Try copying files to temp directory and using relative path
    print(f"\n=== Method 3: Copy to temp directory with clean paths ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy adapter files
        shutil.copy2(adapter_file, temp_path / "adapter_model.safetensors")
        shutil.copy2(adapter_config, temp_path / "adapter_config.json")

        modelfile_content_3 = f"""FROM {base_model}
ADAPTER ./adapter_model.safetensors
"""

        modelfile_path_3 = temp_path / "Modelfile"
        with open(modelfile_path_3, 'w') as f:
            f.write(modelfile_content_3)

        print(f"Temp directory: {temp_path}")
        print(f"Modelfile path: {modelfile_path_3}")
        print(f"Adapter path in Modelfile: ./adapter_model.safetensors")

        # List files in temp directory
        print(f"Files in temp directory:")
        for f in temp_path.iterdir():
            print(f"  - {f.name} ({f.stat().st_size} bytes)")

        result = subprocess.run(
            ["ollama", "create", "test-adapter-3", "-f", str(modelfile_path_3.resolve())],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(temp_path),
            timeout=60
        )

        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout[:500]}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")

        if result.returncode == 0:
            print(f"✓ Method 3 SUCCESS!")
            subprocess.run(["ollama", "rm", "test-adapter-3"], capture_output=True)
            return True

    # Method 4: Try specifying adapter config alongside adapter path
    print(f"\n=== Method 4: Use adapter directory path instead of file ===")
    modelfile_content_4 = f"""FROM {base_model}
ADAPTER {str(checkpoint_dir).replace(chr(92), '/')}
"""

    modelfile_path_4 = checkpoint_dir / "Modelfile_test4"
    with open(modelfile_path_4, 'w') as f:
        f.write(modelfile_content_4)

    print(f"Modelfile path: {modelfile_path_4}")
    print(f"Adapter path in Modelfile: {str(checkpoint_dir).replace(chr(92), '/')}")
    print(f"Working directory: {checkpoint_dir}")

    result = subprocess.run(
        ["ollama", "create", "test-adapter-4", "-f", str(modelfile_path_4.resolve())],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=str(checkpoint_dir),
        timeout=60
    )

    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout[:500]}")
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")

    if result.returncode == 0:
        print(f"✓ Method 4 SUCCESS!")
        subprocess.run(["ollama", "rm", "test-adapter-4"], capture_output=True)
        return True

    print(f"\n❌ All methods failed!")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ollama_adapter.py <checkpoint_path>")
        print("Example: python test_ollama_adapter.py tuned_models/model_1b/stage_1/v1.8/checkpoint-1000")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    success = test_ollama_adapter(checkpoint_path)

    if success:
        print("\n✅ Found working method!")
        sys.exit(0)
    else:
        print("\n❌ No working method found")
        sys.exit(1)
