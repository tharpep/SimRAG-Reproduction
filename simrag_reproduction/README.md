# SimRAG Reproduction Project

## Overview
This project reproduces the SimRAG (Self-improving Retrieval-Augmented Generation) paper on constrained hardware (single RTX 3080) to demonstrate the democratization of domain-specific RAG research.

## Project Structure
```
simrag_reproduction/
├── README.md
├── requirements.txt
├── environment.yml
├── config/
│   ├── baseline_config.yaml
│   ├── simrag_config.yaml
│   └── evaluation_config.yaml
├── src/
│   ├── __init__.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── rag_system.py
│   ├── simrag/
│   │   ├── __init__.py
│   │   ├── self_improvement.py
│   │   ├── synthetic_data_generator.py
│   │   └── simrag_system.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── benchmark.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── helpers.py
├── experiments/
│   ├── run_baseline.py
│   ├── run_simrag.py
│   └── compare_results.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── results/
│   ├── baseline/
│   ├── simrag/
│   └── comparisons/
└── tests/
    ├── test_baseline.py
    ├── test_simrag.py
    └── test_evaluation.py
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (RTX 3080 or similar)
- 16GB+ RAM recommended

### Setup
1. Clone the repository
2. Create conda environment: `conda env create -f environment.yml`
3. Activate environment: `conda activate simrag-reproduction`
4. Install additional requirements: `pip install -r requirements.txt`

## Usage

### Running Baseline RAG
```bash
python experiments/run_baseline.py --config config/baseline_config.yaml
```

### Running SimRAG
```bash
python experiments/run_simrag.py --config config/simrag_config.yaml
```

### Comparing Results
```bash
python experiments/compare_results.py --baseline results/baseline/ --simrag results/simrag/
```

## Evaluation Metrics
- Exact Match (EM)
- F1 Score
- Recall@k (k=1,5,10)
- nDCG
- Latency (p95)
- GPU Memory Usage

## Success Criteria
- **Mid-term:** +2-3% Recall@k or EM/F1 gain
- **Final:** ≥ +5% EM/F1 and Recall@10, with ≤ +5% latency/memory overhead

## References
- SimRAG: Self-improving retrieval-augmented generation (Cheng et al., 2025)
- RAFT: Adapting language models to domain-specific retrieval-augmented generation (Zhang et al., 2024)
- QLoRA: Efficient finetuning of quantized LLMs (Dettmers et al., 2023)
