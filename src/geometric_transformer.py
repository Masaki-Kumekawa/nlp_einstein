"""
Geometric Transformer implementation with Riemannian metric tensors.
Models contextual meaning change as spacetime curvature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import numpy as np


class MetricTensor(nn.Module):
    """Learnable metric tensor for Riemannian geometry in semantic space."""
    
    def __init__(self, hidden_size, rank=None, epsilon=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank or hidden_size
        self.epsilon = epsilon
        
        # Low-rank decomposition for efficiency: G = L L^T + diag
        self.L = nn.Parameter(torch.randn(hidden_size, self.rank) * 0.01)
        self.diag = nn.Parameter(torch.ones(hidden_size))
        
        # Context projection layer
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, context_embeddings):
        """
        Compute metric tensor from context.
        Args:
            context_embeddings: [batch_size, seq_len, hidden_size]
        Returns:
            metric: [batch_size, head_size, head_size] where head_size = hidden_size // num_heads
        """
        batch_size = context_embeddings.size(0)
        hidden_size = context_embeddings.size(-1)
        
        # For now, assume head_size is hidden_size // num_heads
        # This is a simplification - in practice you'd pass head_size explicitly
        head_size = 64  # Fixed head size for simplicity
        
        # Aggregate context information
        context_vector = torch.mean(context_embeddings, dim=1)  # [batch_size, hidden_size]
        context_features = self.context_proj(context_vector)   # [batch_size, hidden_size]
        
        # Take only head_size dimensions
        context_features = context_features[:, :head_size]  # [batch_size, head_size]
        
        # Modulate metric based on context
        context_weight = torch.sigmoid(context_features).unsqueeze(-1)  # [batch_size, head_size, 1]
        
        # Construct positive definite metric tensor for head dimension
        L_head = self.L[:head_size, :min(self.rank, head_size)]  # [head_size, min(rank, head_size)]
        L_modulated = L_head.unsqueeze(0) * context_weight  # [batch_size, head_size, min(rank, head_size)]
        metric = torch.bmm(L_modulated, L_modulated.transpose(1, 2))  # [batch_size, head_size, head_size]
        
        # Add diagonal component for stability
        diag_head = self.diag[:head_size]  # [head_size]
        diag_matrix = torch.diag_embed(diag_head.unsqueeze(0).expand(batch_size, -1) + self.epsilon)
        metric = metric + diag_matrix
        
        return metric


class GeometricAttention(nn.Module):
    """Attention mechanism with Riemannian geometry."""
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.metric_layer = MetricTensor(config.hidden_size, rank=config.metric_rank)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute metric tensor for each attention head
        # Use aggregated context for all heads
        metric = self.metric_layer(hidden_states)  # [batch_size, head_size, head_size]
        
        # Geometric attention scores: Q G^{-1} K^T
        attention_scores = []
        for i in range(self.num_attention_heads):
            q = query_layer[:, i, :, :]  # [batch_size, seq_len, head_size]
            k = key_layer[:, i, :, :]    # [batch_size, seq_len, head_size]
            
            # Use the same metric for all heads
            metric_i = metric  # [batch_size, head_size, head_size]
            
            # Stabilize matrix inversion with stronger regularization
            identity = torch.eye(self.attention_head_size, device=metric_i.device).unsqueeze(0).expand(batch_size, -1, -1)
            regularized_metric = metric_i + 0.1 * identity
            
            try:
                # Use pseudo-inverse for better stability
                g_inv = torch.linalg.pinv(regularized_metric)
            except:
                # Fallback to standard attention if inversion fails
                g_inv = identity
            
            # Compute Q G^{-1}
            qg_inv = torch.bmm(q, g_inv)  # [batch_size, seq_len, head_size]
            
            # Compute (Q G^{-1}) K^T
            scores = torch.bmm(qg_inv, k.transpose(1, 2))  # [batch_size, seq_len, seq_len]
            attention_scores.append(scores.unsqueeze(1))
        
        attention_scores = torch.cat(attention_scores, dim=1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs


class GeometricBertLayer(nn.Module):
    """BERT layer with geometric attention."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = GeometricAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm1(attention_output + hidden_states)
        
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm2(layer_output + attention_output)
        
        return layer_output, attention_probs


class GeometricBERT(nn.Module):
    """BERT model with geometric attention mechanisms."""
    
    def __init__(self, config_dict, from_pretrained=True):
        super().__init__()
        
        # Create BERT config
        self.config = BertConfig(
            hidden_size=config_dict.get('hidden_size', 768),
            num_hidden_layers=config_dict.get('num_hidden_layers', 12),
            num_attention_heads=config_dict.get('num_attention_heads', 12),
            intermediate_size=config_dict.get('intermediate_size', 3072),
            max_position_embeddings=512,
            type_vocab_size=2,
            vocab_size=30522
        )
        
        # Add custom config for geometric components
        self.config.metric_rank = config_dict.get('metric_rank', 64)
        self.config.attention_probs_dropout_prob = 0.1
        self.config.hidden_dropout_prob = 0.1
        self.config.layer_norm_eps = 1e-12
        
        if from_pretrained:
            # Load pre-trained BERT model
            pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
            
            # Copy embeddings from pre-trained BERT
            self.embeddings = pretrained_bert.embeddings.word_embeddings
            self.position_embeddings = pretrained_bert.embeddings.position_embeddings
            self.token_type_embeddings = pretrained_bert.embeddings.token_type_embeddings
            self.LayerNorm = pretrained_bert.embeddings.LayerNorm
            self.dropout = pretrained_bert.embeddings.dropout
            
            # Initialize geometric encoder layers with pre-trained weights
            self.encoder = nn.ModuleList()
            for i in range(self.config.num_hidden_layers):
                if i < len(pretrained_bert.encoder.layer):
                    # Create geometric layer and copy non-attention weights
                    geom_layer = GeometricBertLayer(self.config)
                    orig_layer = pretrained_bert.encoder.layer[i]
                    
                    # Copy feed-forward components
                    geom_layer.intermediate.weight.data = orig_layer.intermediate.dense.weight.data.clone()
                    geom_layer.intermediate.bias.data = orig_layer.intermediate.dense.bias.data.clone()
                    geom_layer.output.weight.data = orig_layer.output.dense.weight.data.clone()
                    geom_layer.output.bias.data = orig_layer.output.dense.bias.data.clone()
                    geom_layer.LayerNorm1.weight.data = orig_layer.attention.output.LayerNorm.weight.data.clone()
                    geom_layer.LayerNorm1.bias.data = orig_layer.attention.output.LayerNorm.bias.data.clone()
                    geom_layer.LayerNorm2.weight.data = orig_layer.output.LayerNorm.weight.data.clone()
                    geom_layer.LayerNorm2.bias.data = orig_layer.output.LayerNorm.bias.data.clone()
                    
                    # Copy attention projection weights (Q, K, V)
                    geom_layer.attention.query.weight.data = orig_layer.attention.self.query.weight.data.clone()
                    geom_layer.attention.query.bias.data = orig_layer.attention.self.query.bias.data.clone()
                    geom_layer.attention.key.weight.data = orig_layer.attention.self.key.weight.data.clone()
                    geom_layer.attention.key.bias.data = orig_layer.attention.self.key.bias.data.clone()
                    geom_layer.attention.value.weight.data = orig_layer.attention.self.value.weight.data.clone()
                    geom_layer.attention.value.bias.data = orig_layer.attention.self.value.bias.data.clone()
                    
                    self.encoder.append(geom_layer)
                else:
                    # Create new geometric layer for additional layers
                    self.encoder.append(GeometricBertLayer(self.config))
        else:
            # Initialize from scratch (original behavior)
            self.embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
            self.token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
            self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
            
            # Geometric encoder layers
            self.encoder = nn.ModuleList([
                GeometricBertLayer(self.config) for _ in range(self.config.num_hidden_layers)
            ])
            
            # Initialize weights
            self.apply(self._init_weights)
        
        # Global metric tensor for similarity computation (always new)
        self.global_metric = MetricTensor(self.config.hidden_size, rank=self.config.metric_rank)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self, input_ids, token_type_ids=None, position_ids=None):
        batch_size, seq_length = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        inputs_embeds = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=True):
        batch_size, seq_length = input_ids.size()
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        hidden_states = self.get_input_embeddings(input_ids, token_type_ids)
        
        # Pass through encoder layers
        all_hidden_states = []
        all_attentions = []
        
        for layer in self.encoder:
            hidden_states, attention_probs = layer(hidden_states, extended_attention_mask)
            all_hidden_states.append(hidden_states)
            all_attentions.append(attention_probs)
        
        # Output
        outputs = {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }
        
        return outputs
    
    def get_contextual_embedding(self, input_ids, attention_mask=None, word_positions=None):
        """Get contextual embeddings for specific word positions."""
        outputs = self.forward(input_ids, attention_mask)
        hidden_states = outputs['last_hidden_state']
        
        if word_positions is not None:
            # Extract embeddings at specific positions
            batch_size = hidden_states.size(0)
            word_embeddings = []
            for i in range(batch_size):
                pos = word_positions[i]
                word_embeddings.append(hidden_states[i, pos, :])
            return torch.stack(word_embeddings)
        else:
            # Return CLS token embedding
            return hidden_states[:, 0, :]
    
    def geodesic_distance(self, emb1, emb2, context_embeddings=None):
        """Compute geodesic distance between embeddings."""
        if context_embeddings is None:
            context_embeddings = torch.cat([emb1.unsqueeze(1), emb2.unsqueeze(1)], dim=1)
        
        metric = self.global_metric(context_embeddings)  # [batch_size, head_size, head_size]
        
        # First-order approximation of geodesic distance
        diff = emb1 - emb2  # [batch_size, hidden_size]
        
        # Project to head dimension for distance computation
        head_size = metric.size(-1)  # Get actual head size from metric
        diff_head = diff[:, :head_size]  # [batch_size, head_size]
        
        # Compute sqrt(diff^T G diff)
        batch_size = diff_head.size(0)
        distances = []
        for i in range(batch_size):
            d = diff_head[i].unsqueeze(0)  # [1, head_size]
            g = metric[i]  # [head_size, head_size]
            
            # Stabilize computation
            try:
                dist_sq = torch.mm(torch.mm(d, g), d.t())  # [1, 1]
                distances.append(torch.sqrt(torch.clamp(dist_sq, min=1e-8)))
            except:
                # Fallback to Euclidean distance
                distances.append(torch.norm(d, dim=1, keepdim=True))
        
        return torch.cat(distances)
    
    def similarity_from_distance(self, distances):
        """Convert geodesic distances to similarity scores."""
        # Use exponential decay for better similarity scaling
        return torch.exp(-distances.squeeze())