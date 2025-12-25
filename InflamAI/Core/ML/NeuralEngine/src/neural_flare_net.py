#!/usr/bin/env python3
"""
neural_flare_net.py - Advanced Neural Architecture for AS Flare Prediction
Implements both LSTM and Transformer architectures with automatic switching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer architecture"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence outputs"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)
        weighted = x * attn_weights
        return weighted.sum(dim=1)

class NeuralFlareNetLSTM(nn.Module):
    """LSTM-based architecture for AS flare prediction"""

    def __init__(
        self,
        input_dim: int = 35,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.architecture = "LSTM"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_output_dim)

        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

        # Auxiliary outputs for interpretability
        self.risk_score = nn.Linear(lstm_output_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Dictionary with 'logits' and 'risk_score'
        """
        # Input projection and normalization
        x = self.input_proj(x)
        x = self.input_norm(x)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention pooling over sequence
        pooled = self.attention_pool(lstm_out)

        # Classification
        logits = self.classifier(pooled)

        # Risk score (0-1 continuous value)
        risk = torch.sigmoid(self.risk_score(pooled))

        return {
            'logits': logits,
            'risk_score': risk,
            'attention_weights': None  # Can add attention weights if needed
        }

class NeuralFlareNetTransformer(nn.Module):
    """Transformer-based architecture for AS flare prediction"""

    def __init__(
        self,
        input_dim: int = 35,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_len: int = 30
    ):
        super().__init__()
        self.architecture = "Transformer"
        self.d_model = d_model

        # Input embedding and positional encoding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

        # Risk score head
        self.risk_score = nn.Linear(d_model, 1)

        # Learned CLS token for sequence classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Dictionary with predictions
        """
        batch_size = x.size(0)

        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.input_norm(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Use CLS token for classification
        cls_output = x[:, 0]

        # Also use attention pooling on sequence (skip CLS)
        seq_output = self.attention_pool(x[:, 1:])

        # Combine both representations
        combined = cls_output + seq_output

        # Classification
        logits = self.classifier(combined)

        # Risk score
        risk = torch.sigmoid(self.risk_score(combined))

        return {
            'logits': logits,
            'risk_score': risk,
            'attention_weights': None
        }

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in medical data"""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        return focal_loss.mean()

class NeuralFlareNet(nn.Module):
    """
    Main model class that can switch between LSTM and Transformer
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.architecture_type = config.get('architecture', 'LSTM')

        if self.architecture_type == 'LSTM':
            self.model = NeuralFlareNetLSTM(
                input_dim=config.get('input_dim', 35),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('num_layers', 3),
                dropout=config.get('dropout', 0.3),
                bidirectional=config.get('bidirectional', True)
            )
        elif self.architecture_type == 'Transformer':
            self.model = NeuralFlareNetTransformer(
                input_dim=config.get('input_dim', 35),
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 4),
                dim_feedforward=config.get('dim_feedforward', 512),
                dropout=config.get('dropout', 0.3),
                max_seq_len=config.get('max_seq_len', 30)
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture_type}")

        self.config = config

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def switch_architecture(self, new_architecture: str, config: Dict):
        """Switch to a different architecture (LSTM <-> Transformer)"""
        print(f"Switching architecture from {self.architecture_type} to {new_architecture}")
        self.architecture_type = new_architecture

        if new_architecture == 'LSTM':
            self.model = NeuralFlareNetLSTM(**config)
        elif new_architecture == 'Transformer':
            self.model = NeuralFlareNetTransformer(**config)
        else:
            raise ValueError(f"Unknown architecture: {new_architecture}")

        # Move to same device as before
        device = next(self.parameters()).device
        self.to(device)

def create_model(config: Dict) -> Tuple[NeuralFlareNet, FocalLoss]:
    """
    Factory function to create model and loss
    Args:
        config: Model configuration dictionary
    Returns:
        Tuple of (model, loss_function)
    """
    model = NeuralFlareNet(config)

    # Calculate class weights if provided
    alpha = None
    if 'class_weights' in config:
        alpha = torch.tensor(config['class_weights'])

    loss_fn = FocalLoss(
        gamma=config.get('focal_gamma', 2.0),
        alpha=alpha
    )

    print(f"Created {model.architecture_type} model with {model.get_num_parameters():,} parameters")

    return model, loss_fn

if __name__ == "__main__":
    # Test model creation
    config = {
        'architecture': 'LSTM',
        'input_dim': 35,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.3
    }

    model, loss_fn = create_model(config)
    print(f"Model architecture: {model.architecture_type}")
    print(f"Parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 32
    seq_len = 30
    input_dim = 35

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = model(dummy_input)

    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Risk score shape: {output['risk_score'].shape}")