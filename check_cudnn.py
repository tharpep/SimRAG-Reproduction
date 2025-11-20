#!/usr/bin/env python3
"""Quick diagnostic to check cuDNN settings."""
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("\ncuDNN settings:")
    print(f"  deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  benchmark: {torch.backends.cudnn.benchmark}")
    print("\n✓ Training should be FAST with deterministic=False and benchmark=True")
else:
    print("\n⚠ CUDA not available - training will be slow on CPU")

