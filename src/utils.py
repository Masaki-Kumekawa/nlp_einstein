"""
Utility functions for the geometric language model project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import json
import os
from datetime import datetime


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path}")
    return epoch, loss


def visualize_attention_patterns(attention_weights, tokens, save_path):
    """Visualize attention patterns as heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Average attention across heads
    avg_attention = attention_weights.mean(axis=0)
    
    # Create heatmap
    sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens, 
                cmap='Blues', cbar=True, square=True)
    plt.title('Attention Pattern Visualization')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_tsne_visualization(model, sample_words, contexts, save_path):
    """Generate t-SNE visualization of word embeddings in semantic space."""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for word, context in zip(sample_words, contexts):
            # Tokenize
            tokens = model.tokenizer(context, return_tensors='pt', padding=True, truncation=True)
            input_ids = tokens['input_ids'].to(model.device)
            attention_mask = tokens['attention_mask'].to(model.device)
            
            # Get embedding
            outputs = model(input_ids, attention_mask)
            embedding = outputs['last_hidden_state'][:, 0, :].cpu().numpy()  # CLS token
            
            embeddings.append(embedding[0])
            labels.append(word)
    
    # Apply t-SNE
    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=100)
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Semantic Space Visualization (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_metric_tensor(metric_tensor, save_path):
    """Visualize the metric tensor as a heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy if tensor
    if torch.is_tensor(metric_tensor):
        metric_tensor = metric_tensor.detach().cpu().numpy()
    
    # If batch, take first example
    if len(metric_tensor.shape) == 3:
        metric_tensor = metric_tensor[0]
    
    # Create heatmap
    sns.heatmap(metric_tensor, cmap='RdBu_r', center=0, square=True, 
                cbar_kws={'label': 'Metric Tensor Values'})
    plt.title('Metric Tensor Visualization')
    plt.xlabel('Dimension')
    plt.ylabel('Dimension')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_curvature(metric_tensor):
    """
    Compute scalar curvature from metric tensor (simplified).
    This is a placeholder - full Riemann curvature computation is complex.
    """
    if torch.is_tensor(metric_tensor):
        metric_tensor = metric_tensor.detach().cpu().numpy()
    
    # Simplified curvature measure: deviation from identity
    n = metric_tensor.shape[-1]
    identity = np.eye(n)
    
    if len(metric_tensor.shape) == 3:
        # Batch of metric tensors
        curvatures = []
        for i in range(metric_tensor.shape[0]):
            deviation = np.linalg.norm(metric_tensor[i] - identity, 'fro')
            curvatures.append(deviation)
        return np.array(curvatures)
    else:
        # Single metric tensor
        return np.linalg.norm(metric_tensor - identity, 'fro')


def save_results_json(results, filepath):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    results = convert_types(results)
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'framework': 'PyTorch'
    }
    
    # Save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results_json(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def format_latex_table(data, caption, label):
    """Format data as LaTeX table."""
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\begin{{tabular}}{{"""
    
    # Determine columns
    if isinstance(data, dict):
        cols = list(data.keys())
        latex += 'l' + 'c' * len(cols) + '}\n\\toprule\n'
        latex += ' & '.join(['Method'] + cols) + ' \\\\\n\\midrule\n'
        
        # Add data rows
        for method, values in data.items():
            if isinstance(values, dict):
                row_values = [f"{values.get(col, 'N/A'):.3f}" if isinstance(values.get(col, 0), (int, float)) else str(values.get(col, 'N/A')) for col in cols]
            else:
                row_values = [f"{values:.3f}" if isinstance(values, (int, float)) else str(values)]
            latex += method + ' & ' + ' & '.join(row_values) + ' \\\\\n'
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    return latex


def calculate_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
    else:
        return 0


class MetricLogger:
    """Logger for tracking metrics during training."""
    
    def __init__(self):
        self.metrics = {}
    
    def log(self, metric_name, value, step=None):
        """Log a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        entry = {'value': value}
        if step is not None:
            entry['step'] = step
        
        self.metrics[metric_name].append(entry)
    
    def get_metric(self, metric_name):
        """Get logged values for a metric."""
        return self.metrics.get(metric_name, [])
    
    def save(self, filepath):
        """Save metrics to file."""
        save_results_json(self.metrics, filepath)
    
    def load(self, filepath):
        """Load metrics from file."""
        self.metrics = load_results_json(filepath)