# Geometric Language Models: Contextual Meaning Change as Spacetime Curvature

This repository contains the complete implementation of a novel geometric approach to modeling contextual meaning change in natural language processing, inspired by general relativity.

## 🚀 Quick Start

Run the complete research pipeline with one command:

```bash
make do
```

This will:
1. Install all dependencies
2. Download real datasets (WordSim-353, SimLex-999, etc.)
3. Train the geometric BERT model
4. Evaluate on similarity and GLUE tasks
5. Generate visualizations
6. Create the final research paper

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM
- Internet connection for dataset downloads

## 🏗️ Architecture

### Core Components

1. **GeometricBERT** (`src/geometric_transformer.py`)
   - Modified transformer with learnable metric tensors
   - Geometric attention mechanism using geodesic distances
   - Riemannian manifold modeling of semantic space

2. **Metric Tensor Module**
   - Low-rank decomposition for efficiency
   - Context-dependent curvature computation
   - Positive definiteness guarantees

3. **Evaluation Framework** (`src/evaluation.py`)
   - Similarity task evaluation (WordSim-353, SimLex-999, CoSimLex, SCWS)
   - GLUE benchmark support
   - Baseline comparison with BERT

## 🔬 Research Methodology

### Theoretical Foundation

The model treats semantic space as a Riemannian manifold where:
- **Flat regions** represent context-independent meanings
- **Curved regions** capture context-induced meaning changes
- **Geodesic distances** provide true semantic similarity

Mathematical formulation:
```
g_μν(x) = η_μν + h_μν(context)
Attention(Q,K,V) = softmax(Q G^(-1) K^T) V
```

### Experimental Setup

- **Model**: 768-dim, 12-layer, 12-head transformer
- **Metric rank**: 64 (low-rank approximation)
- **Training**: MLM on WikiText with geometric attention
- **Evaluation**: Similarity tasks + GLUE benchmarks

## 📊 Expected Results

Based on our implementation, we expect:
- **10-20% improvement** on contextual similarity tasks
- **Competitive performance** on GLUE benchmarks
- **Interpretable attention patterns** via geodesic paths

## 🗂️ Project Structure

```
nlp_einstein/
├── src/                          # Core implementation
│   ├── geometric_transformer.py     # Main model
│   ├── evaluation.py               # Evaluation metrics
│   └── utils.py                    # Utilities
├── data/                         # Datasets (auto-downloaded)
├── results/                      # Experimental results
├── output/                       # Generated papers
├── real_experiment.py           # Main experiment script
├── download_real_datasets.py    # Dataset acquisition
├── Makefile                     # Automation system
└── requirements.txt             # Dependencies
```

## 🎯 Manual Execution

If you prefer step-by-step execution:

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download datasets
python download_real_datasets.py

# 3. Run experiment
python real_experiment.py

# 4. Generate paper
python generate_paper_from_real_results.py
```

## 📈 Monitoring Progress

Check experiment progress:
```bash
tail -f experiment.log
```

View results:
```bash
cat results/experiment_report.txt
```

## 🔧 Configuration

Edit model configuration in `real_experiment.py`:

```python
config = {
    'model': {
        'hidden_size': 768,
        'metric_rank': 64,     # Adjust for efficiency
        'num_hidden_layers': 12
    },
    'training': {
        'batch_size': 16,      # Adjust for GPU memory
        'num_epochs': 3,
        'learning_rate': 2e-5
    }
}
```

## 🎨 Visualizations

The system generates:
- Training loss curves
- Similarity performance comparisons
- t-SNE semantic space visualizations
- Metric tensor heatmaps
- Attention pattern analysis

## 📄 Output

Final deliverables:
- **LaTeX paper** (`output/paper.tex`)
- **Experimental results** (`results/real_experiment_results.json`)
- **Detailed report** (`results/experiment_report.txt`)
- **Visualizations** (`results/*.png`)

## 🚨 Important Notes

- **Actual Research**: This implements a real geometric language model, not a demo
- **Computational Requirements**: Training may take several hours on GPU
- **Memory Usage**: Requires significant RAM for large datasets
- **Scientific Rigor**: Includes proper baselines and statistical evaluation

## 🤝 Contributing

This is a research implementation. For issues or improvements:
1. Check experiment logs for debugging
2. Verify dataset downloads completed successfully
3. Ensure sufficient computational resources

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{geometric_language_models,
  title={Geometric Language Models: Contextual Meaning Change as Spacetime Curvature},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

## ⚖️ License

This research code is provided for academic use. Please respect dataset licenses and usage terms.

---

**Note**: This is a complete research implementation that performs actual machine learning experiments. Execution time and results will vary based on hardware and dataset availability.