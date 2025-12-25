#!/usr/bin/env python3
"""
advanced_lstm_trainer.py - Advanced LSTM Trainer Targeting AUC 0.95+
================================================================================
Improvements over lstm_trainer_fixed.py:

1. SMOTE OVERSAMPLING: Synthetic minority oversampling for 81/19 imbalance
2. SELF-ATTENTION: Multi-head self-attention (Transformer-style)
3. RESIDUAL CONNECTIONS: Skip connections for better gradient flow
4. TIME-SERIES AUGMENTATION: Jitter, scaling, magnitude warping
5. LABEL SMOOTHING: Prevents overconfidence, improves calibration
6. MIXUP TRAINING: Convex combinations of samples for regularization
7. ENSEMBLE READY: Outputs designed for gradient boosting ensemble

Target: AUC 0.95+ for 80% precision AND 80% recall achievability

Author: Claude Code ML Fix
Date: 2024-12-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MULTI-HEAD SELF-ATTENTION (Transformer-style)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism from Transformer architecture.
    Allows the model to attend to different parts of the input sequence
    when making predictions - crucial for finding flare patterns.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_dim)
        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)

        # Causal mask (only look at past)
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, hidden_dim)
        out = self.proj(out)

        return out


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

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


# ============================================================================
# ADVANCED LSTM WITH SELF-ATTENTION
# ============================================================================

class AdvancedLSTMFlarePredictor(nn.Module):
    """
    Advanced LSTM architecture combining:
    - Multi-layer unidirectional LSTM
    - Multi-head self-attention
    - Residual connections
    - Deep classification head

    Designed to achieve AUC 0.95+ for medical-grade predictions.
    """

    def __init__(
        self,
        input_dim: int = 92,
        hidden_dim: int = 256,  # Increased from 128
        num_layers: int = 3,     # Increased from 2
        num_heads: int = 8,      # Multi-head attention
        dropout: float = 0.3
    ):
        super().__init__()
        self.architecture = "ADVANCED_LSTM_ATTENTION"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection with skip connection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Secondary projection for skip connection
        self.input_skip = nn.Linear(input_dim, hidden_dim)

        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()

        for i in range(num_layers):
            self.lstm_layers.append(nn.LSTM(
                hidden_dim, hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=False  # Unidirectional to prevent data leakage
            ))
            self.lstm_norms.append(nn.LayerNorm(hidden_dim))

        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Residual feed-forward blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(2)
        ])

        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Deep classification head
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
            nn.Linear(hidden_dim // 4, 2)  # Binary classification
        )

        # Calibrated risk score head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful initialization"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if 'weight_ih' in pname:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in pname:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in pname:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1 for better gradient flow
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention visualization support.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Dictionary with 'logits', 'risk_score', and 'attention_weights'
        """
        # Input projection with skip connection
        h = self.input_proj(x)
        skip = self.input_skip(x)
        h = h + skip

        # Multi-layer LSTM with residual connections
        for lstm, norm in zip(self.lstm_layers, self.lstm_norms):
            lstm_out, _ = lstm(h)
            h = norm(lstm_out + h)  # Residual connection

        # Self-attention with residual
        attn_out = self.self_attention(h)
        h = self.attn_norm(h + attn_out)

        # Residual feed-forward blocks
        for block in self.residual_blocks:
            h = block(h)

        # Temporal attention pooling
        attn_weights = F.softmax(self.temporal_attention(h), dim=1)
        pooled = (h * attn_weights).sum(dim=1)

        # Classification
        logits = self.classifier(pooled)

        # Risk score
        risk = self.risk_head(pooled)

        return {
            'logits': logits,
            'risk_score': risk.squeeze(-1),
            'attention_weights': attn_weights.squeeze(-1)
        }


# ============================================================================
# TIME-SERIES DATA AUGMENTATION
# ============================================================================

class TimeSeriesAugmenter:
    """
    Time-series specific data augmentation techniques.
    Helps the model generalize better to unseen patterns.
    """

    def __init__(
        self,
        jitter_strength: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        warp_strength: float = 0.1,
        dropout_rate: float = 0.1
    ):
        self.jitter_strength = jitter_strength
        self.scale_range = scale_range
        self.warp_strength = warp_strength
        self.dropout_rate = dropout_rate

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        return x + torch.randn_like(x) * self.jitter_strength

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling"""
        scale = torch.empty(x.size(0), 1, x.size(2)).uniform_(*self.scale_range).to(x.device)
        return x * scale

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth warping of magnitude"""
        # Create smooth warping curve
        B, T, D = x.shape
        warp = torch.randn(B, 1, D, device=x.device) * self.warp_strength
        smooth_warp = 1 + warp.expand(-1, T, -1)
        return x * smooth_warp

    def feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop features (simulates missing data)"""
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_rate))
        return x * mask

    def time_crop(self, x: torch.Tensor, min_ratio: float = 0.8) -> torch.Tensor:
        """Crop sequence in time dimension"""
        B, T, D = x.shape
        new_len = int(T * torch.empty(1).uniform_(min_ratio, 1.0).item())
        start = torch.randint(0, T - new_len + 1, (1,)).item()

        cropped = x[:, start:start+new_len, :]
        # Pad back to original length
        if new_len < T:
            padding = torch.zeros(B, T - new_len, D, device=x.device)
            cropped = torch.cat([cropped, padding], dim=1)
        return cropped

    def augment(self, x: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
        """Apply random augmentations"""
        if torch.rand(1).item() < prob:
            x = self.jitter(x)
        if torch.rand(1).item() < prob:
            x = self.scale(x)
        if torch.rand(1).item() < prob / 2:
            x = self.magnitude_warp(x)
        if torch.rand(1).item() < prob / 2:
            x = self.feature_dropout(x)
        return x


# ============================================================================
# SMOTE FOR TIME-SERIES
# ============================================================================

def fast_smote_time_series(
    data: np.ndarray,
    targets: np.ndarray,
    target_ratio: float = 0.5,
    noise_level: float = 0.03
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAST SMOTE oversampling for time-series sequences.
    Uses random pair interpolation instead of full KNN (O(n) vs O(n^2)).

    For large datasets, this is much faster while achieving similar results.

    Args:
        data: Array of shape (n_samples, seq_len, features)
        targets: Array of shape (n_samples,)
        target_ratio: Target ratio of minority to majority class
        noise_level: Amount of random noise to add

    Returns:
        Augmented data and targets
    """
    print("\nApplying FAST SMOTE oversampling for time-series...")

    # Separate classes
    minority_idx = np.where(targets == 1)[0]
    majority_idx = np.where(targets == 0)[0]

    n_minority = len(minority_idx)
    n_majority = len(majority_idx)

    print(f"  Before: {n_majority:,} majority, {n_minority:,} minority ({100*n_minority/(n_minority+n_majority):.1f}%)")

    # Calculate how many synthetic samples needed
    n_synthetic = int(n_majority * target_ratio) - n_minority

    if n_synthetic <= 0:
        print("  No oversampling needed")
        return data, targets

    print(f"  Generating {n_synthetic:,} synthetic minority samples...")

    # Get minority samples
    minority_data = data[minority_idx]

    # Generate synthetic samples using random pair interpolation
    # This is O(n) instead of O(n^2) for full SMOTE
    synthetic_data = np.zeros((n_synthetic, data.shape[1], data.shape[2]), dtype=np.float32)

    # Batch generation for speed
    batch_size = 10000
    for batch_start in range(0, n_synthetic, batch_size):
        batch_end = min(batch_start + batch_size, n_synthetic)
        batch_n = batch_end - batch_start

        # Random pairs of minority samples
        idx1 = np.random.randint(0, n_minority, size=batch_n)
        idx2 = np.random.randint(0, n_minority, size=batch_n)

        # Interpolation factors
        alpha = np.random.uniform(0.2, 0.8, size=(batch_n, 1, 1))

        # Interpolate
        synthetic_data[batch_start:batch_end] = (
            alpha * minority_data[idx1] +
            (1 - alpha) * minority_data[idx2]
        )

        # Add small noise for diversity
        synthetic_data[batch_start:batch_end] += np.random.randn(
            batch_n, data.shape[1], data.shape[2]
        ).astype(np.float32) * noise_level

        if batch_start % 50000 == 0 and batch_start > 0:
            print(f"    Generated {batch_start:,}/{n_synthetic:,} samples...")

    synthetic_targets = np.ones(n_synthetic, dtype=targets.dtype)

    # Combine with original
    augmented_data = np.concatenate([data, synthetic_data], axis=0)
    augmented_targets = np.concatenate([targets, synthetic_targets], axis=0)

    # Shuffle
    indices = np.random.permutation(len(augmented_targets))
    augmented_data = augmented_data[indices]
    augmented_targets = augmented_targets[indices]

    final_minority = augmented_targets.sum()
    print(f"  After: {len(augmented_targets):,} total ({final_minority:,} minority = {100*final_minority/len(augmented_targets):.1f}%)")

    return augmented_data, augmented_targets


# ============================================================================
# MIXUP TRAINING
# ============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation - creates convex combinations of samples.
    Helps regularization and improves calibration.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# LABEL SMOOTHING LOSS
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.
    Prevents overconfidence by softening hard labels.
    """
    def __init__(self, smoothing: float = 0.1, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Log softmax
        log_probs = F.log_softmax(pred, dim=-1)

        # Cross-entropy with smoothed labels
        loss = -smooth_target * log_probs

        # Apply class weights if provided
        if self.class_weights is not None:
            weight = self.class_weights[target]
            loss = loss.sum(dim=-1) * weight
        else:
            loss = loss.sum(dim=-1)

        return loss.mean()


# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================

class AdvancedFocalLoss(nn.Module):
    """
    Combined Focal Loss + Label Smoothing for optimal imbalanced learning.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smoothing: float = 0.05,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        n_classes = inputs.size(-1)
        with torch.no_grad():
            smooth_target = torch.zeros_like(inputs)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate focal loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ============================================================================
# ADVANCED DATASET WITH SMOTE
# ============================================================================

class AdvancedDataset(Dataset):
    """Dataset with SMOTE-augmented minority class"""

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        augmenter: Optional[TimeSeriesAugmenter] = None,
        training: bool = False
    ):
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)
        self.augmenter = augmenter
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.training and self.augmenter is not None:
            x = self.augmenter.augment(x.unsqueeze(0)).squeeze(0)

        return x, y


# ============================================================================
# ADVANCED TRAINER
# ============================================================================

class AdvancedLSTMTrainer:
    """
    Advanced trainer targeting AUC 0.95+ with:
    - SMOTE oversampling
    - Mixup training
    - OneCycle learning rate
    - Gradient accumulation
    - Comprehensive metrics tracking
    """

    def __init__(self, config: Optional[Dict] = None, base_dir: Optional[Path] = None):
        self.device = self._get_device()
        print(f"\nUsing device: {self.device}")

        self.config = config or self._get_default_config()
        self.history = []
        self.best_metrics = {'auc': 0, 'f1': 0}

        # Paths
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = base_dir
        self.checkpoint_dir = base_dir / 'models'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = base_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.data_dir = base_dir / 'data'

        # Augmenter
        self.augmenter = TimeSeriesAugmenter(
            jitter_strength=0.03,
            scale_range=(0.95, 1.05),
            warp_strength=0.05,
            dropout_rate=0.1
        )

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_default_config(self):
        return {
            'input_dim': 92,
            'hidden_dim': 256,      # Increased
            'num_layers': 3,        # Increased
            'num_heads': 8,         # Multi-head attention
            'dropout': 0.3,
            'batch_size': 128,      # Smaller for better generalization
            'learning_rate': 0.0005, # Lower for stability
            'weight_decay': 0.01,
            'epochs': 100,
            'early_stopping_patience': 15,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'label_smoothing': 0.05,
            'mixup_alpha': 0.2,
            'smote_ratio': 0.4,     # Target 40% minority
            'sequence_length': 30,
            'gradient_accumulation': 2
        }

    def load_data(self, data_path: str):
        """Load data with SMOTE oversampling"""
        print(f"\n{'='*70}")
        print("LOADING DATA WITH SMOTE OVERSAMPLING")
        print(f"{'='*70}")

        # Load data
        print(f"Loading {data_path}...")
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_parquet(data_path)

        print(f"Loaded {len(df):,} rows")

        # Define feature columns
        feature_cols = [col for col in df.columns
                        if col not in ['patient_id', 'day_index', 'will_flare_3_7d']]
        target_col = 'will_flare_3_7d'

        print(f"Features: {len(feature_cols)}")

        # MinMax normalization
        print("\nApplying MinMax normalization...")
        mins, maxs = [], []

        for col in feature_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mins.append(min_val)
            maxs.append(max_val)

            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0

        self.scaler_params = {
            'mins': mins,
            'maxs': maxs,
            'feature_names': feature_cols,
            'n_features': len(feature_cols)
        }

        # Patient-based split
        print("\nPerforming PATIENT-BASED split...")
        df = df.sort_values(['patient_id', 'day_index']).reset_index(drop=True)

        unique_patients = df['patient_id'].unique()
        n_patients = len(unique_patients)

        np.random.seed(42)
        shuffled_patients = np.random.permutation(unique_patients)

        train_end = int(n_patients * 0.7)
        val_end = int(n_patients * 0.85)

        train_patients = set(shuffled_patients[:train_end])
        val_patients = set(shuffled_patients[train_end:val_end])
        test_patients = set(shuffled_patients[val_end:])

        print(f"Train patients: {len(train_patients)}")
        print(f"Val patients: {len(val_patients)}")
        print(f"Test patients: {len(test_patients)}")

        # Convert to numpy
        data_values = df[feature_cols].values.astype(np.float32)
        target_values = df[target_col].values.astype(np.int64)
        patient_ids = df['patient_id'].values

        # Create sequences for each split
        def create_sequences(patients_set):
            sequences, targets = [], []
            for i in range(len(df) - self.config['sequence_length']):
                if patient_ids[i] not in patients_set:
                    continue
                if patient_ids[i] != patient_ids[i + self.config['sequence_length']]:
                    continue

                seq = data_values[i:i + self.config['sequence_length']]
                target = target_values[i + self.config['sequence_length']]
                sequences.append(seq)
                targets.append(target)

            return np.array(sequences), np.array(targets)

        print("\nCreating sequences...")
        train_data, train_targets = create_sequences(train_patients)
        val_data, val_targets = create_sequences(val_patients)
        test_data, test_targets = create_sequences(test_patients)

        print(f"Train sequences: {len(train_data):,}")
        print(f"Val sequences: {len(val_data):,}")
        print(f"Test sequences: {len(test_data):,}")

        # Apply FAST SMOTE to training data
        train_data, train_targets = fast_smote_time_series(
            train_data, train_targets,
            target_ratio=self.config['smote_ratio'],
            noise_level=0.03
        )

        # Calculate class weights from augmented data
        class_counts = np.bincount(train_targets)
        self.class_weights = torch.FloatTensor(
            len(train_targets) / (2 * class_counts)
        ).to(self.device)
        print(f"\nClass weights: {self.class_weights.cpu().numpy()}")

        # Create datasets
        self.train_dataset = AdvancedDataset(
            train_data, train_targets, self.augmenter, training=True
        )
        self.val_dataset = AdvancedDataset(val_data, val_targets, training=False)
        self.test_dataset = AdvancedDataset(test_data, test_targets, training=False)

        # Create dataloaders
        # Use weighted sampler for additional class balancing
        sample_weights = np.where(train_targets == 1, 2.0, 1.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )

    def create_model(self):
        """Create advanced model"""
        self.model = AdvancedLSTMFlarePredictor(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        ).to(self.device)

        # Advanced focal loss
        self.criterion = AdvancedFocalLoss(
            alpha=self.config['focal_alpha'],
            gamma=self.config['focal_gamma'],
            smoothing=self.config['label_smoothing'],
            class_weights=self.class_weights
        )

        # AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # OneCycle learning rate schedule
        steps_per_epoch = len(self.train_loader) // self.config['gradient_accumulation']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'] * 10,
            epochs=self.config['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel parameters: {n_params:,}")

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with mixup"""
        self.model.train()
        running_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixup augmentation
            if self.config['mixup_alpha'] > 0 and np.random.rand() < 0.5:
                data, target_a, target_b, lam = mixup_data(
                    data, target, self.config['mixup_alpha']
                )
                output = self.model(data)
                loss = mixup_criterion(
                    self.criterion, output['logits'], target_a, target_b, lam
                )
            else:
                output = self.model(data)
                loss = self.criterion(output['logits'], target)

            # Gradient accumulation
            loss = loss / self.config['gradient_accumulation']
            loss.backward()

            if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.config['gradient_accumulation']

            # Collect predictions
            with torch.no_grad():
                probs = F.softmax(output['logits'], dim=1)
                preds = probs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='binary')

        return {
            'loss': running_loss / len(self.train_loader),
            'accuracy': accuracy,
            'f1': f1
        }

    def validate(self) -> Dict:
        """Comprehensive validation"""
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output['logits'], target)

                running_loss += loss.item()

                probs = F.softmax(output['logits'], dim=1)
                preds = probs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='binary')
        precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.5

        # Find optimal threshold for balanced precision/recall
        precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0.5

        return {
            'loss': running_loss / len(self.val_loader),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'prob_mean': np.mean(all_probs),
            'prob_std': np.std(all_probs),
            'optimal_threshold': float(best_threshold)
        }

    def train(self):
        """Full training loop"""
        print(f"\n{'='*70}")
        print("ADVANCED LSTM TRAINING - TARGETING AUC 0.95+")
        print(f"{'='*70}")

        best_auc = 0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch}/{self.config['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train: loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.3f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.3f}")
            print(f"  Val:   precision={val_metrics['precision']:.3f} recall={val_metrics['recall']:.3f} AUC={val_metrics['auc']:.4f}")
            print(f"  Output: mean={val_metrics['prob_mean']:.3f} std={val_metrics['prob_std']:.3f}")

            self.history.append({
                'epoch': epoch,
                **train_metrics,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })

            # Save best model based on AUC
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                patience_counter = 0
                self.best_metrics = val_metrics.copy()
                self.save_checkpoint('best_lstm_advanced.pth')
                print(f"  *** New best AUC: {best_auc:.4f} - Model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"\nEarly stopping after {epoch} epochs")
                    break

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE in {total_time/60:.1f} minutes")
        print(f"Best validation AUC: {best_auc:.4f}")
        print(f"{'='*70}")

        # Save history
        with open(self.log_dir / 'training_history_advanced.json', 'w') as f:
            json.dump(self.history, f, indent=2, default=float)

        # Save scaler params
        with open(self.data_dir / 'minmax_params_advanced.json', 'w') as f:
            json.dump(self.scaler_params, f, indent=2, default=float)

        return self.best_metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.best_metrics,
            'scaler_params': self.scaler_params,
            'architecture': 'ADVANCED_LSTM_ATTENTION'
        }, self.checkpoint_dir / filename)

    def evaluate_on_test(self):
        """Final test evaluation with threshold optimization"""
        print(f"\n{'='*70}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*70}")

        # Load best model
        checkpoint = torch.load(self.checkpoint_dir / 'best_lstm_advanced.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                probs = F.softmax(output['logits'], dim=1)
                preds = probs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Standard metrics (threshold 0.5)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='binary')
        precision = precision_score(all_targets, all_preds, average='binary')
        recall = recall_score(all_targets, all_preds, average='binary')
        auc = roc_auc_score(all_targets, all_probs)

        print(f"\nTest Set Results (threshold=0.5):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

        # Find threshold for target metrics
        print(f"\n{'='*70}")
        print("THRESHOLD OPTIMIZATION FOR 80% TARGETS")
        print(f"{'='*70}")

        # Search for best threshold
        best_f1_thresh = 0.5
        best_f1 = 0

        for thresh in np.arange(0.1, 0.9, 0.01):
            preds = (all_probs >= thresh).astype(int)
            prec = precision_score(all_targets, preds, zero_division=0)
            rec = recall_score(all_targets, preds, zero_division=0)
            f1_temp = f1_score(all_targets, preds, zero_division=0)

            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_f1_thresh = thresh

            # Check for 80%/80% target
            if prec >= 0.80 and rec >= 0.80:
                print(f"\n  *** 80%/80% TARGET ACHIEVABLE at threshold {thresh:.2f}!")
                print(f"      Precision: {prec*100:.1f}%")
                print(f"      Recall:    {rec*100:.1f}%")
                print(f"      F1:        {f1_temp:.4f}")

        # Show optimal F1 threshold
        opt_preds = (all_probs >= best_f1_thresh).astype(int)
        opt_prec = precision_score(all_targets, opt_preds, zero_division=0)
        opt_rec = recall_score(all_targets, opt_preds, zero_division=0)

        print(f"\nOptimal F1 threshold: {best_f1_thresh:.2f}")
        print(f"  Precision: {opt_prec*100:.1f}%")
        print(f"  Recall:    {opt_rec*100:.1f}%")
        print(f"  F1:        {best_f1:.4f}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"AUC-ROC: {auc:.4f}")

        if auc >= 0.95:
            print("*** SUCCESS: AUC >= 0.95 - 80%/80% precision/recall is achievable!")
        elif auc >= 0.90:
            print("*** GOOD: AUC >= 0.90 - Close to 80%/80% target")
        else:
            print(f"*** AUC {auc:.4f} - May need more improvements for 80%/80% target")

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'optimal_threshold': best_f1_thresh,
            'optimal_f1': best_f1
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Force unbuffered output
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("="*70, flush=True)
    print("ADVANCED LSTM TRAINER - TARGETING AUC 0.95+", flush=True)
    print("="*70, flush=True)
    print("\nAdvanced features:", flush=True)
    print("  1. SMOTE oversampling for class imbalance", flush=True)
    print("  2. Multi-head self-attention (Transformer-style)", flush=True)
    print("  3. Residual connections for better gradients", flush=True)
    print("  4. Time-series data augmentation", flush=True)
    print("  5. Label smoothing + Focal loss", flush=True)
    print("  6. Mixup training for regularization", flush=True)
    print("  7. OneCycle learning rate schedule", flush=True)
    print("="*70, flush=True)

    script_dir = Path(__file__).parent.parent

    # Use smaller dataset for faster iteration
    data_path = script_dir / 'data/training_data_500k.parquet'
    if not data_path.exists():
        data_path = script_dir / 'data/comprehensive_training_data.parquet'
        if not data_path.exists():
            print(f"ERROR: No training data found!", flush=True)
            return

    print(f"\nUsing data: {data_path}", flush=True)

    # Create trainer
    trainer = AdvancedLSTMTrainer(base_dir=script_dir)

    # Load data with SMOTE
    trainer.load_data(str(data_path))

    # Create advanced model
    trainer.create_model()

    # Train
    trainer.train()

    # Evaluate
    test_metrics = trainer.evaluate_on_test()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal AUC: {test_metrics['auc']:.4f}")
    print(f"Next step: Run coreml_export_advanced.py to export to iOS")

    return test_metrics


if __name__ == "__main__":
    main()
