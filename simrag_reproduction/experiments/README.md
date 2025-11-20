# Experiments

This folder contains scripts for running baseline, SimRAG, and comparison experiments.

## Structure

```
experiments/
├── baseline/          # Baseline RAG experiments
│   ├── run_baseline.py
│   └── results/       # Baseline results (JSON files)
│
├── simrag/            # SimRAG training experiments
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── run_full_pipeline.py
│   └── results/       # SimRAG results (JSON files)
│
├── comparison/        # Comparison experiments
│   ├── compare_results.py
│   └── results/       # Comparison results (JSON files)
│
├── utils.py           # Shared utilities (HTML extraction, etc.)
└── run_experiment.py  # Main orchestrator script
```

## Quick Start

### Run Complete Pipeline

```bash
# From simrag_reproduction directory
cd experiments
python run_experiment.py
```

This will:
1. Run baseline RAG test
2. Run SimRAG Stage 1 training (Alpaca dataset)
3. Run SimRAG Stage 2 training (domain documents)
4. Compare results

### Run Individual Experiments

```bash
# Baseline only
python baseline/run_baseline.py --documents ../../data/documents

# Stage 1 only
python simrag/run_stage1.py

# Stage 2 only
python simrag/run_stage2.py --documents ../../data/documents

# Full SimRAG pipeline
python simrag/run_full_pipeline.py --documents ../../data/documents

# Compare existing results
python comparison/compare_results.py \
    --baseline baseline/results/baseline_results.json \
    --simrag simrag/results/full_pipeline_results.json
```

## Options

### Use Test Data (Faster)

For quick testing, use test data instead of Alpaca:

```bash
python simrag/run_stage1.py --test-data
python run_experiment.py --test-data
```

### Skip Steps

If you already have results:

```bash
# Skip baseline, use existing
python run_experiment.py --skip-baseline

# Skip SimRAG, use existing
python run_experiment.py --skip-simrag
```

## Output Files

All results are saved as JSON files in respective `results/` folders:

- `baseline/results/baseline_results.json`
- `simrag/results/stage1_results.json`
- `simrag/results/stage2_results.json`
- `simrag/results/full_pipeline_results.json`
- `comparison/results/comparison_results.json`

## Notes

- All scripts import from `simrag_reproduction` (no code duplication)
- HTML documents are automatically extracted using `utils.py`
- Results include timestamps, configs, and performance metrics
- Use `--help` on any script for detailed options

