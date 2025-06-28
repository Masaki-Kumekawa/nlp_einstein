# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing a geometric language model that uses concepts from general relativity (spacetime curvature) to model contextual meaning changes in natural language processing. The project aims to create a fully automated research pipeline from concept to final paper.

## Key Commands

### Main Automation Command
```bash
make do
```
This single command executes the entire research pipeline: environment setup, data preparation, model training, evaluation, and paper generation.

### Individual Steps
- `make setup` - Install dependencies and create directories
- `make data` - Prepare datasets
- `make train` - Train the geometric language model
- `make evaluate` - Run evaluation on similarity and GLUE tasks
- `make paper` - Generate final LaTeX paper with results
- `make test` - Run unit tests
- `make clean` - Clean generated files

### Python Execution
The main experiment code should be implemented in `実験コード.py` (or `experiment_code.py`) with these key functions:
- `run_training()` - Execute model training
- `run_evaluation()` - Perform evaluations
- `generate_paper()` - Create final paper with results

## Architecture

### Core Concept
The project implements a modified BERT model with geometric attention mechanisms:
- **Metric Tensor**: Models context-induced curvature in semantic space
- **Geodesic Distance**: Computes semantic similarity using curved geometry
- **Modified Attention**: `Attention(Q,K,V) = softmax(Q G^(-1) K^T) V`

### File Structure
```
project/
├── src/                        # Source code modules
│   ├── geometric_transformer.py   # Main model implementation
│   ├── evaluation.py             # Evaluation metrics
│   └── utils.py                  # Utility functions
├── data/                       # Datasets (auto-downloaded)
├── results/                    # Experiment results
│   └── experiment_results.json   # All metrics and comparisons
└── output/                     # Final outputs
    └── 論文.tex/pdf             # Generated paper
```

### Key Classes to Implement

1. **GeometricBERT** (in `src/geometric_transformer.py`):
   - Extends BERT with geometric attention
   - Implements metric tensor computation
   - Handles geodesic distance calculations

2. **MetricTensor** (module within GeometricBERT):
   - Learnable parameters for context-dependent metric
   - Supports diagonal and low-rank approximations

3. **SimilarityEvaluator** (in `src/evaluation.py`):
   - Evaluates on CoSimLex, SCWS, WordSim-353
   - Computes Spearman/Pearson correlations

4. **GLUEEvaluator** (in `src/evaluation.py`):
   - Handles GLUE benchmark tasks
   - Returns task-specific metrics

## Implementation Notes

### Model Configuration
```python
{
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "metric_rank": 64,  # For computational efficiency
    "learning_rate": 2e-5,
    "batch_size": 32,
    "num_epochs": 3
}
```

### Expected Results Format
The pipeline generates `results/experiment_results.json` with:
- Similarity task correlations (CoSimLex, SCWS, etc.)
- GLUE benchmark scores
- Baseline comparisons with BERT
- Computational metrics (training time, inference speed)
- Visualization paths

### Paper Generation
The system automatically fills a LaTeX template with experimental results using placeholders like:
- `{{IMPROVEMENT_COSIMLX}}` - Percentage improvement on CoSimLex
- `{{OURS_SCWS}}` - Our model's SCWS score
- `{{BERT_BASELINE}}` - BERT baseline scores

## Development Workflow

1. Implement core geometric modules in `src/`
2. Create main experiment script (`実験コード.py`)
3. Run `make do` to execute full pipeline
4. Check `results/experiment_results.json` for metrics
5. Review generated paper in `output/論文.pdf`

## Testing
Run unit tests with `make test` or `pytest` to verify:
- Metric tensor properties (positive definiteness)
- Geodesic distance calculations
- Attention mechanism modifications
- Evaluation metric correctness