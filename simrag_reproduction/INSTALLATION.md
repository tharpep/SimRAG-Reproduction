# Installation Guide for SimRAG Reproduction Project

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA RTX 3080 or similar (12GB VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models and data
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.9 or 3.10
- **CUDA**: 11.8 or compatible version
- **Git**: For cloning the repository

## Installation Steps

### 1. Install Prerequisites

#### Windows
```bash
# Install Anaconda/Miniconda
# Download from: https://www.anaconda.com/products/distribution

# Install Git
# Download from: https://git-scm.com/download/win

# Install CUDA Toolkit 11.8
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
```

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git
brew install git

# Install Python 3.9
brew install python@3.9
```

#### Ubuntu/Linux
```bash
# Update package list
sudo apt update

# Install Python 3.9 and pip
sudo apt install python3.9 python3.9-pip python3.9-venv

# Install Git
sudo apt install git

# Install CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 2. Clone the Repository
```bash
git clone <repository-url>
cd simrag_reproduction
```

### 3. Create Conda Environment
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate simrag-reproduction
```

### 4. Install Additional Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test key imports
python -c "import transformers, sentence_transformers, faiss, peft; print('All imports successful')"
```

## Alternative Installation Methods

### Using pip only (without conda)
```bash
# Create virtual environment
python -m venv simrag-env

# Activate environment
# Windows:
simrag-env\Scripts\activate
# macOS/Linux:
source simrag-env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### Using Docker (Advanced)
```dockerfile
# Create Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.9 python3-pip git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

# Set working directory
WORKDIR /app
COPY . .

# Run experiments
CMD ["python", "experiments/run_baseline.py", "--config", "config/baseline_config.yaml"]
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in config files
# Enable gradient checkpointing
# Use mixed precision training
```

#### Import Errors
```bash
# Reinstall problematic packages
pip uninstall <package-name>
pip install <package-name>

# Check Python version compatibility
python --version
```

#### Model Download Issues
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Performance Optimization

#### For RTX 3080 (12GB VRAM)
```yaml
# In config files, use these settings:
model:
  generator:
    name: "microsoft/DialoGPT-medium"  # Smaller model
training:
  batch_size: 4  # Smaller batch size
  gradient_accumulation_steps: 2
hardware:
  mixed_precision: true
  gradient_checkpointing: true
```

#### Memory Optimization
```python
# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache between experiments
torch.cuda.empty_cache()
```

## Quick Start

After installation, run a quick test:

```bash
# Test baseline system
python experiments/run_baseline.py --config config/baseline_config.yaml --output ./test_results/baseline

# Test SimRAG system
python experiments/run_simrag.py --config config/simrag_config.yaml --output ./test_results/simrag

# Compare results
python experiments/compare_results.py --baseline ./test_results/baseline --simrag ./test_results/simrag
```

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your CUDA installation: `nvidia-smi`
3. Check available GPU memory: `nvidia-smi`
4. Review the logs in the output directories
5. Ensure all dependencies are correctly installed

## Next Steps

After successful installation:

1. Review the configuration files in `config/`
2. Adjust parameters for your hardware
3. Run the baseline experiment first
4. Then run the SimRAG experiment
5. Compare results using the comparison script
