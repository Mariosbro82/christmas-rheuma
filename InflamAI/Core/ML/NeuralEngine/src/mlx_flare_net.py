#!/usr/bin/env python3
"""
mlx_flare_net.py - MLX-optimized Neural Network for AS Flare Prediction
Optimized for Apple Silicon using MLX framework
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, Tuple, List


class MLXFlareNetLSTM(nn.Module):
    """LSTM-based architecture for AS flare prediction using MLX"""
    
    def __init__(
        self,
        input_dim: int = 92,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM layers
        self.lstms = []
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim * self.num_directions
            self.lstms.append(nn.LSTM(layer_input_dim, hidden_dim, bias=True))
            if bidirectional:
                self.lstms.append(nn.LSTM(layer_input_dim, hidden_dim, bias=True))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention pooling
        attention_input = hidden_dim * self.num_directions
        self.attention_weights = nn.Linear(attention_input, 1)
        
        # Classification head
        self.fc1 = nn.Linear(attention_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def __call__(self, x):
        """
        Forward pass
        Args:
            x: Input array of shape (batch, seq_len, input_dim)
        Returns:
            Dictionary with 'logits' and 'risk_score'
        """
        batch_size, seq_len, _ = x.shape
        
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM layers
        for i in range(self.num_layers):
            if self.bidirectional:
                # Forward LSTM
                lstm_fwd = self.lstms[i * 2]
                h_fwd = mx.zeros((batch_size, self.hidden_dim))
                c_fwd = mx.zeros((batch_size, self.hidden_dim))
                
                # Backward LSTM
                lstm_bwd = self.lstms[i * 2 + 1]
                h_bwd = mx.zeros((batch_size, self.hidden_dim))
                c_bwd = mx.zeros((batch_size, self.hidden_dim))
                
                fwd_outputs = []
                bwd_outputs = []
                
                # Forward pass
                for t in range(seq_len):
                    h_fwd, c_fwd = lstm_fwd(x[:, t, :], (h_fwd, c_fwd))
                    fwd_outputs.append(h_fwd)
                
                # Backward pass
                for t in range(seq_len - 1, -1, -1):
                    h_bwd, c_bwd = lstm_bwd(x[:, t, :], (h_bwd, c_bwd))
                    bwd_outputs.insert(0, h_bwd)
                
                # Concatenate forward and backward
                fwd_stack = mx.stack(fwd_outputs, axis=1)
                bwd_stack = mx.stack(bwd_outputs, axis=1)
                x = mx.concatenate([fwd_stack, bwd_stack], axis=-1)
            else:
                # Unidirectional LSTM
                lstm = self.lstms[i]
                h = mx.zeros((batch_size, self.hidden_dim))
                c = mx.zeros((batch_size, self.hidden_dim))
                
                outputs = []
                for t in range(seq_len):
                    h, c = lstm(x[:, t, :], (h, c))
                    outputs.append(h)
                
                x = mx.stack(outputs, axis=1)
            
            # Dropout between layers
            if i < self.num_layers - 1:
                x = self.dropout(x)
        
        # Attention pooling
        attention_scores = self.attention_weights(x)  # (batch, seq_len, 1)
        attention_weights = mx.softmax(attention_scores, axis=1)
        context_vector = mx.sum(x * attention_weights, axis=1)  # (batch, hidden_dim * dirs)
        
        # Classification head
        x = self.fc1(context_vector)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        logits = self.fc3(x)
        
        # Calculate risk score (probability of flare)
        probs = mx.softmax(logits, axis=-1)
        risk_score = probs[:, 1]
        
        return {
            'logits': logits,
            'risk_score': risk_score
        }


def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """
    Focal Loss for addressing class imbalance
    Args:
        logits: Model predictions (batch, 2)
        targets: Ground truth labels (batch,)
        gamma: Focusing parameter
        alpha: Class weights (2,)
    """
    # Convert targets to one-hot
    targets_one_hot = mx.zeros((targets.shape[0], 2))
    targets_one_hot = mx.scatter(
        targets_one_hot,
        mx.expand_dims(targets, axis=1),
        mx.ones_like(mx.expand_dims(targets, axis=1), dtype=mx.float32),
        axis=1
    )
    
    # Compute softmax probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Compute focal loss
    ce_loss = -mx.log(probs + 1e-8) * targets_one_hot
    p_t = mx.sum(probs * targets_one_hot, axis=-1)
    focal_weight = mx.power(1 - p_t, gamma)
    
    loss = focal_weight * mx.sum(ce_loss, axis=-1)
    
    # Apply class weights
    if alpha is not None:
        alpha_t = mx.sum(alpha * targets_one_hot, axis=-1)
        loss = alpha_t * loss
    
    return mx.mean(loss)


def create_mlx_model(config: Dict):
    """Create MLX model and loss function"""
    model = MLXFlareNetLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=True
    )
    
    # Count parameters
    def count_params(tree):
        total = 0
        if isinstance(tree, dict):
            for v in tree.values():
                total += count_params(v)
        elif hasattr(tree, 'size'):
            total += tree.size
        return total
    
    num_params = count_params(model.parameters())
    print(f"Created MLX LSTM model with {num_params:,} parameters")
    
    return model


if __name__ == "__main__":
    # Test model
    config = {
        'input_dim': 92,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.3
    }
    
    model = create_mlx_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 30
    test_input = mx.random.normal((batch_size, seq_len, config['input_dim']))
    
    output = model(test_input)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output risk score shape: {output['risk_score'].shape}")
    print("MLX model test successful!")
