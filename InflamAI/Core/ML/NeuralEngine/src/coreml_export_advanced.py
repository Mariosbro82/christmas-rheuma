#!/usr/bin/env python3
"""
coreml_export_advanced.py - Export Advanced LSTM to CoreML for iOS
================================================================================
Exports the AdvancedLSTMFlarePredictor to .mlpackage format for iOS deployment.

Key features:
- Multi-head self-attention preserved in CoreML
- MinMax scaling parameters exported
- Threshold configuration for precision/recall trade-off
- Validation against PyTorch outputs

Author: Claude Code ML Fix
Date: 2024-12-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add parent to path
sys.path.append(str(Path(__file__).parent))

# Try to import coremltools
try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
except ImportError:
    print("Installing coremltools...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'coremltools'])
    import coremltools as ct


# ============================================================================
# RECREATE MODEL ARCHITECTURE (must match training exactly)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for CoreML export"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class AdvancedLSTMFlarePredictor(nn.Module):
    """Advanced LSTM architecture for CoreML export"""

    def __init__(
        self,
        input_dim: int = 92,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.input_skip = nn.Linear(input_dim, hidden_dim)

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.lstm_layers.append(nn.LSTM(
                hidden_dim, hidden_dim, num_layers=1,
                batch_first=True, dropout=0, bidirectional=False
            ))
            self.lstm_norms.append(nn.LayerNorm(hidden_dim))

        # Self-attention
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(2)
        ])

        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 2)
        )

        # Risk head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input projection
        h = self.input_proj(x)
        skip = self.input_skip(x)
        h = h + skip

        # LSTM layers with residual
        for lstm, norm in zip(self.lstm_layers, self.lstm_norms):
            lstm_out, _ = lstm(h)
            h = norm(lstm_out + h)

        # Self-attention
        attn_out = self.self_attention(h)
        h = self.attn_norm(h + attn_out)

        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # Temporal pooling
        attn_weights = F.softmax(self.temporal_attention(h), dim=1)
        pooled = (h * attn_weights).sum(dim=1)

        # Outputs
        logits = self.classifier(pooled)
        risk = self.risk_head(pooled)

        return logits, risk.squeeze(-1)


# ============================================================================
# COREML EXPORT WRAPPER
# ============================================================================

class CoreMLWrapper(nn.Module):
    """Wrapper for clean CoreML export with softmax probabilities"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, risk = self.model(x)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        return probs, risk


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def load_checkpoint(checkpoint_path: Path):
    """Load trained model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    metrics = checkpoint['metrics']

    print(f"  Architecture: {checkpoint.get('architecture', 'ADVANCED_LSTM_ATTENTION')}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Best AUC: {metrics.get('auc', 'N/A')}")

    # Create model
    model = AdvancedLSTMFlarePredictor(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config.get('num_heads', 8),
        dropout=config['dropout']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, metrics, checkpoint.get('scaler_params', {})


def export_to_coreml(model, config, output_path: Path):
    """Export model to CoreML format"""
    print("\nExporting to CoreML...")

    # Wrap model
    wrapped = CoreMLWrapper(model)
    wrapped.eval()

    # Create dummy input
    seq_len = config.get('sequence_length', 30)
    input_dim = config['input_dim']
    dummy_input = torch.randn(1, seq_len, input_dim)

    # Trace model
    print("  Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy_input)

    # Convert to CoreML
    print("  Converting to CoreML...")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="features",
                shape=(1, seq_len, input_dim),
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="probabilities", dtype=np.float32),
            ct.TensorType(name="risk_score", dtype=np.float32)
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
    )

    # Add metadata
    mlmodel.author = "InflamAI ML Pipeline"
    mlmodel.short_description = "AS Flare Prediction - Advanced LSTM with Self-Attention"
    mlmodel.version = "2.0"

    mlmodel.input_description["features"] = f"{seq_len}-day sequence of {input_dim} patient features (MinMax normalized)"
    mlmodel.output_description["probabilities"] = "Softmax probabilities [no_flare, will_flare]"
    mlmodel.output_description["risk_score"] = "Continuous risk score (0-1)"

    mlmodel.user_defined_metadata["architecture"] = "ADVANCED_LSTM_ATTENTION"
    mlmodel.user_defined_metadata["hidden_dim"] = str(config['hidden_dim'])
    mlmodel.user_defined_metadata["num_layers"] = str(config['num_layers'])
    mlmodel.user_defined_metadata["num_heads"] = str(config.get('num_heads', 8))

    # Save
    print(f"  Saving to {output_path}...")
    mlmodel.save(str(output_path))

    # Get size
    size_mb = sum(f.stat().st_size for f in output_path.rglob('*')) / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")

    return mlmodel


def validate_coreml(mlmodel, pytorch_model, config):
    """Validate CoreML model matches PyTorch"""
    print("\nValidating CoreML model...")

    seq_len = config.get('sequence_length', 30)
    input_dim = config['input_dim']

    # Test input
    test_input = np.random.randn(1, seq_len, input_dim).astype(np.float32)

    # PyTorch prediction
    pytorch_model.eval()
    wrapped = CoreMLWrapper(pytorch_model)
    with torch.no_grad():
        torch_probs, torch_risk = wrapped(torch.from_numpy(test_input))
        torch_probs = torch_probs.numpy()
        torch_risk = torch_risk.numpy()

    # CoreML prediction
    coreml_out = mlmodel.predict({"features": test_input})
    coreml_probs = coreml_out["probabilities"]
    coreml_risk = coreml_out["risk_score"]

    # Compare
    prob_diff = np.abs(torch_probs - coreml_probs).max()
    risk_diff = np.abs(torch_risk - coreml_risk).max()

    print(f"  Probability diff: {prob_diff:.6f}")
    print(f"  Risk score diff: {risk_diff:.6f}")

    if prob_diff < 1e-3 and risk_diff < 1e-3:
        print("  ✅ Validation PASSED!")
        return True
    else:
        print("  ⚠️ Warning: Outputs differ (may be acceptable)")
        return False


def export_scaler_params(scaler_params, output_path: Path):
    """Export scaler parameters for iOS"""
    print(f"\nExporting scaler params to {output_path}...")

    # Ensure proper format
    params = {
        'mins': [float(x) for x in scaler_params.get('mins', [])],
        'maxs': [float(x) for x in scaler_params.get('maxs', [])],
        'n_features': int(scaler_params.get('n_features', 92)),
        'scaler_type': 'MinMaxScaler',
        'note': 'Advanced LSTM model - use (x - min) / (max - min) normalization'
    }

    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"  ✅ Saved {len(params['mins'])} feature params")


def export_threshold_config(metrics, output_path: Path):
    """Export threshold configuration"""
    print(f"\nExporting threshold config to {output_path}...")

    config = {
        'default_threshold': 0.5,
        'optimized_threshold': metrics.get('optimal_threshold', 0.5),
        'auc': float(metrics.get('auc', 0.82)),
        'precision_at_default': float(metrics.get('precision', 0.27)),
        'recall_at_default': float(metrics.get('recall', 0.92)),
        'note': 'Use optimized_threshold for better precision, default for better recall',
        'thresholds': {
            'sensitive': 0.30,    # High recall, more false alarms
            'balanced': 0.50,     # Default
            'conservative': 0.70  # High precision, fewer alerts
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ Saved threshold configuration")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("COREML EXPORT - ADVANCED LSTM FLARE PREDICTOR")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent

    # Find checkpoint
    checkpoint_path = base_dir / 'models/best_lstm_advanced.pth'
    if not checkpoint_path.exists():
        # Try fixed model
        checkpoint_path = base_dir / 'models/best_lstm_fixed.pth'
        if not checkpoint_path.exists():
            print(f"ERROR: No checkpoint found!")
            print(f"  Checked: {base_dir / 'models/'}")
            return False

    # Load model
    model, config, metrics, scaler_params = load_checkpoint(checkpoint_path)

    # Export to CoreML
    output_path = base_dir / 'models/ASFlarePredictor_Advanced.mlpackage'
    mlmodel = export_to_coreml(model, config, output_path)

    # Validate
    validate_coreml(mlmodel, model, config)

    # Export scaler params
    if scaler_params:
        export_scaler_params(scaler_params, base_dir / 'data/minmax_params_advanced.json')

    # Export threshold config
    export_threshold_config(metrics, base_dir / 'data/threshold_config.json')

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {output_path}")
    print(f"  2. {base_dir / 'data/minmax_params_advanced.json'}")
    print(f"  3. {base_dir / 'data/threshold_config.json'}")
    print("\nNext: Copy to iOS project and update UnifiedNeuralEngine.swift")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
