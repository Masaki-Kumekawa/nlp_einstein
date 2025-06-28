"""
Evaluation modules for similarity tasks and GLUE benchmark.
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import BertTokenizer
import pandas as pd


class SimilarityEvaluator:
    """Evaluator for word similarity tasks using geodesic distances."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def get_word_position(self, tokens, word):
        """Find position of word in tokenized sequence."""
        # Convert word to string if it's not already
        if not isinstance(word, str):
            word = str(word)
        
        # Skip if word is too short or numeric
        if len(word) < 2 or word.isdigit():
            return 1  # Default to position 1 (after CLS token)
            
        try:
            word_tokens = self.tokenizer.tokenize(word)
            for i in range(len(tokens) - len(word_tokens) + 1):
                if tokens[i:i+len(word_tokens)] == word_tokens:
                    return i
        except Exception:
            pass
        return 1  # Default to position 1 if not found
    
    def evaluate(self, dataset):
        """
        Evaluate on similarity dataset.
        Args:
            dataset: DataFrame with columns ['word1', 'word2', 'context1', 'context2', 'score']
        Returns:
            (spearman_corr, pearson_corr)
        """
        self.model.eval()
        predicted_similarities = []
        human_scores = []
        
        with torch.no_grad():
            for _, row in dataset.iterrows():
                try:
                    word1, word2 = str(row['word1']), str(row['word2'])
                    
                    # Skip invalid entries
                    if (len(word1) < 2 or len(word2) < 2 or 
                        word1.isdigit() or word2.isdigit()):
                        continue
                        
                    context1 = row.get('context1', f"The {word1} is here.")
                    context2 = row.get('context2', f"The {word2} is here.")
                    human_score = float(row['score'])
                    
                    # Convert contexts to strings
                    context1 = str(context1)
                    context2 = str(context2)
                    
                except (ValueError, KeyError, TypeError):
                    continue
                
                # Tokenize contexts
                tokens1 = self.tokenizer(context1, return_tensors='pt', padding=True, truncation=True)
                tokens2 = self.tokenizer(context2, return_tensors='pt', padding=True, truncation=True)
                
                # Move to device
                input_ids1 = tokens1['input_ids'].to(self.device)
                attention_mask1 = tokens1['attention_mask'].to(self.device)
                input_ids2 = tokens2['input_ids'].to(self.device)
                attention_mask2 = tokens2['attention_mask'].to(self.device)
                
                # Find word positions
                tokens_list1 = self.tokenizer.convert_ids_to_tokens(input_ids1[0].cpu().numpy())
                tokens_list2 = self.tokenizer.convert_ids_to_tokens(input_ids2[0].cpu().numpy())
                
                pos1 = self.get_word_position(tokens_list1, word1)
                pos2 = self.get_word_position(tokens_list2, word2)
                
                # Get contextual embeddings
                emb1 = self.model.get_contextual_embedding(
                    input_ids1, attention_mask1, word_positions=[pos1]
                )
                emb2 = self.model.get_contextual_embedding(
                    input_ids2, attention_mask2, word_positions=[pos2]
                )
                
                # Compute geodesic distance and convert to similarity
                distance = self.model.geodesic_distance(emb1, emb2)
                similarity = self.model.similarity_from_distance(distance).item()
                
                predicted_similarities.append(similarity)
                human_scores.append(human_score)
        
        # Compute correlations
        spearman_corr = spearmanr(predicted_similarities, human_scores)[0]
        pearson_corr = pearsonr(predicted_similarities, human_scores)[0]
        
        return spearman_corr, pearson_corr


class GLUEEvaluator:
    """Evaluator for GLUE benchmark tasks."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Task-specific configurations
        self.task_configs = {
            'cola': {'num_labels': 2, 'metric': 'matthews_corrcoef'},
            'sst2': {'num_labels': 2, 'metric': 'accuracy'},
            'mrpc': {'num_labels': 2, 'metric': 'f1'},
            'qqp': {'num_labels': 2, 'metric': 'f1'},
            'sts-b': {'num_labels': 1, 'metric': 'pearson'},
            'mnli': {'num_labels': 3, 'metric': 'accuracy'},
            'qnli': {'num_labels': 2, 'metric': 'accuracy'},
            'rte': {'num_labels': 2, 'metric': 'accuracy'},
            'wnli': {'num_labels': 2, 'metric': 'accuracy'}
        }
        
        # Classification heads for each task
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(model.config.hidden_size, config['num_labels'])
            for task, config in self.task_configs.items()
        }).to(self.device)
    
    def evaluate(self, dataset, task_name):
        """
        Evaluate on GLUE task.
        Args:
            dataset: Hugging Face dataset
            task_name: Name of GLUE task
        Returns:
            Task-specific metric score
        """
        if task_name not in self.task_configs:
            raise ValueError(f"Unknown GLUE task: {task_name}")
        
        self.model.eval()
        predictions = []
        labels = []
        
        # Get task-specific head
        task_head = self.task_heads[task_name]
        task_config = self.task_configs[task_name]
        
        with torch.no_grad():
            for batch in self._create_dataloader(dataset, task_name):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['labels'].to(self.device)
                
                # Get model outputs
                outputs = self.model(input_ids, attention_mask)
                pooled_output = outputs['last_hidden_state'][:, 0, :]  # CLS token
                
                # Task-specific prediction
                logits = task_head(pooled_output)
                
                if task_config['num_labels'] == 1:
                    # Regression task
                    preds = logits.squeeze(-1)
                else:
                    # Classification task
                    preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(label.cpu().numpy())
        
        # Compute metric
        metric_name = task_config['metric']
        if metric_name == 'accuracy':
            return accuracy_score(labels, predictions)
        elif metric_name == 'f1':
            return f1_score(labels, predictions, average='macro')
        elif metric_name == 'matthews_corrcoef':
            return matthews_corrcoef(labels, predictions)
        elif metric_name == 'pearson':
            return pearsonr(predictions, labels)[0]
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _create_dataloader(self, dataset, task_name, batch_size=32):
        """Create dataloader for GLUE task."""
        # This is a simplified version - in practice, you'd use proper data processing
        dataloader = []
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Prepare texts based on task
            if task_name in ['mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'sts-b']:
                # Sentence pair tasks
                texts = [f"{ex['sentence1']} [SEP] {ex['sentence2']}" for ex in batch]
            else:
                # Single sentence tasks
                texts = [ex['sentence'] for ex in batch]
            
            # Tokenize
            encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Add labels
            encoding['labels'] = torch.tensor([ex['label'] for ex in batch])
            
            dataloader.append(encoding)
        
        return dataloader


# Utility functions for loading datasets
def load_similarity_dataset(dataset_name):
    """Load similarity dataset (mock implementation)."""
    # In practice, you would load actual datasets
    # For now, return synthetic data
    np.random.seed(42)
    
    if dataset_name == 'wordsim353':
        n_samples = 353
    elif dataset_name == 'simlex999':
        n_samples = 999
    elif dataset_name == 'cosimlx':
        n_samples = 1024
    elif dataset_name == 'scws':
        n_samples = 2003
    else:
        n_samples = 100
    
    # Generate synthetic data
    words = ['apple', 'banana', 'car', 'computer', 'book', 'phone', 'tree', 'house', 'water', 'fire']
    data = []
    
    for _ in range(n_samples):
        word1 = np.random.choice(words)
        word2 = np.random.choice(words)
        context1 = f"The {word1} is very important in this context."
        context2 = f"I need a {word2} for my work."
        score = np.random.uniform(0, 10)
        
        data.append({
            'word1': word1,
            'word2': word2,
            'context1': context1,
            'context2': context2,
            'score': score
        })
    
    return pd.DataFrame(data)


def load_glue_dataset(task_name):
    """Load GLUE dataset (mock implementation)."""
    # In practice, you would use datasets library
    # For now, return synthetic data
    np.random.seed(42)
    
    if task_name in ['cola', 'sst2']:
        # Single sentence tasks
        sentences = [
            "This is a great movie.",
            "I don't like this at all.",
            "The weather is nice today.",
            "This doesn't make sense.",
            "Everything works perfectly."
        ]
        
        data = []
        for _ in range(100):
            data.append({
                'sentence': np.random.choice(sentences),
                'label': np.random.randint(0, 2)
            })
    
    elif task_name in ['mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']:
        # Sentence pair tasks
        sentences1 = [
            "The cat is on the mat.",
            "I went to the store.",
            "The movie was interesting.",
            "She likes to read books.",
            "The weather is sunny."
        ]
        sentences2 = [
            "A feline rests on the rug.",
            "I visited the shop.",
            "The film was boring.",
            "Reading is her hobby.",
            "It's a beautiful day."
        ]
        
        data = []
        num_labels = 3 if task_name == 'mnli' else 2
        for _ in range(100):
            data.append({
                'sentence1': np.random.choice(sentences1),
                'sentence2': np.random.choice(sentences2),
                'label': np.random.randint(0, num_labels)
            })
    
    else:
        raise ValueError(f"Unknown GLUE task: {task_name}")
    
    return data


# Add missing import
import torch.nn as nn