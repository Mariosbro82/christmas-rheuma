#!/usr/bin/env python3
"""
mlx_simple_trainer.py - Simplified MLX Trainer to Get Started Quickly
Using feedforward architecture first, can add LSTM later
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import sys


class SimpleFlareNet(nn.Module):
    """Simple feedforward model for AS flare prediction"""
    
    def __init__(self, input_dim=92):
        super().__init__()
        
        # Use mean pooling over sequence dimension
        # Then feedforward layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def __call__(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Pool over sequence
        x = mx.mean(x, axis=1)  # (batch, input_dim)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        logits = self.fc4(x)
        
        return {'logits': logits}


def load_data(data_path, sequence_length=30):
    """Load and prepare data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows")
    
    # Define features
    feature_cols = [col for col in df.columns
                   if col not in ['patient_id', 'day_index', 'will_flare_3_7d']]
    target_col = 'will_flare_3_7d'
    
    # Normalize
    print("Normalizing features...")
    for col in feature_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    
    # Create sequences
    print(f"Creating sequences with window size {sequence_length}...")
    df = df.sort_values(['patient_id', 'day_index'])
    
    all_seqs = []
    all_targets = []
    
    for patient_id, group in df.groupby('patient_id'):
        if len(group) < sequence_length + 7:
            continue
        
        p_data = group[feature_cols].values.astype(np.float32)
        p_targets = group[target_col].values.astype(np.int32)
        
        num_windows = len(p_data) - sequence_length - 7
        if num_windows <= 0:
            continue
        
        for i in range(num_windows):
            seq = p_data[i:i+sequence_length]
            target = p_targets[i+sequence_length]
            all_seqs.append(seq)
            all_targets.append(target)
    
    sequences = np.array(all_seqs, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.int32)
    
    print(f"Created {len(sequences):,} sequences")
    print(f"Class balance: {np.mean(targets):.2%} positive")
    
    # Split data
    total_size = len(sequences)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    
    indices = np.random.permutation(total_size)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    return {
        'train': (mx.array(sequences[train_idx]), mx.array(targets[train_idx])),
        'val': (mx.array(sequences[val_idx]), mx.array(targets[val_idx])),
        'test': (mx.array(sequences[test_idx]), mx.array(targets[test_idx]))
    }


def focal_loss(logits, targets, gamma=2.0):
    """Focal loss for class imbalance"""
    # Softmax probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Log probabilities (manual log_softmax)
    log_probs = mx.log(probs + 1e-8)
    
    # Get target probabilities
    target_log_probs = mx.take_along_axis(
        log_probs, 
        mx.expand_dims(targets, axis=1),
        axis=1
    ).squeeze(1)
    
    target_probs = mx.take_along_axis(
        probs,
        mx.expand_dims(targets, axis=1),
        axis=1
    ).squeeze(1)
    
    # Focal weight
    focal_weight = mx.power(1.0 - target_probs, gamma)
    loss = -focal_weight * target_log_probs
    
    return mx.mean(loss)


def train_epoch(model, optimizer, train_data, train_targets, batch_size=256):
    """Train for one epoch"""
    num_samples = len(train_data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Shuffle
    indices = mx.random.permutation(num_samples)
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        
        batch_indices = indices[start:end]
        batch_data = train_data[batch_indices]
        batch_targets = train_targets[batch_indices]
        
        # Forward and backward
        def loss_fn(model):
            output = model(batch_data)
            return focal_loss(output['logits'], batch_targets)
        
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        total_loss += loss.item()
        
        # Get predictions
        output = model(batch_data)
        preds = mx.argmax(output['logits'], axis=-1)
        all_preds.extend(preds.tolist())
        all_targets.extend(batch_targets.tolist())
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}")
            sys.stdout.flush()
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = np.mean(all_preds == all_targets)
    
    # F1 score
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / num_batches, accuracy, f1


def validate(model, val_data, val_targets, batch_size=256):
    """Validate model"""
    num_samples = len(val_data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        
        batch_data = val_data[start:end]
        batch_targets = val_targets[start:end]
        
        output = model(batch_data)
        loss = focal_loss(output['logits'], batch_targets)
        total_loss += loss.item()
        
        preds = mx.argmax(output['logits'], axis=-1)
        all_preds.extend(preds.tolist())
        all_targets.extend(batch_targets.tolist())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = np.mean(all_preds == all_targets)
    
    # F1 score
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / num_batches, accuracy, f1


def main():
    print("\n" + "="*60)
    print("MLX TRAINER - APPLE SILICON OPTIMIZED")
    print("="*60)
    
    # Load data
    data = load_data('data/training_data_500k.parquet')
    train_data, train_targets = data['train']
    val_data, val_targets = data['val']
    test_data, test_targets = data['test']
    
    print(f"\nTrain: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
    
    # Create model
    model = SimpleFlareNet(input_dim=92)
    print(f"Created SimpleFlareNet model")
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=0.001)
    
    # Training loop
    best_val_acc = 0
    no_improve = 0
    
    for epoch in range(100):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/100")
        print('='*60)
        
        # Train
        start_time = time.time()
        train_loss, train_acc, train_f1 = train_epoch(
            model, optimizer, train_data, train_targets
        )
        train_time = time.time() - start_time
        
        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_data, val_targets)
        
        print(f"\n📊 Metrics:")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"  Time: {train_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            
            # Save model
            Path('models').mkdir(exist_ok=True)
            # Flatten parameters and save
            flat_params = tree_flatten(model.parameters())
            params_dict = {f'param_{i}': v for i, v in enumerate(flat_params[0]) if hasattr(v, 'size')}
            mx.savez('models/best_model_mlx.npz', **params_dict)
            print("  ✅ New best model saved!")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= 10:
            print(f"\n⚠️ Early stopping after {epoch+1} epochs")
            break
    
    # Test
    print("\n🧪 Final test set evaluation...")
    test_loss, test_acc, test_f1 = validate(model, test_data, test_targets)
    print(f"  Test - Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    print("\n✨ Training complete!")


if __name__ == "__main__":
    main()
