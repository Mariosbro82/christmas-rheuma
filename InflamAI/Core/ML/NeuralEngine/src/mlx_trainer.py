#!/usr/bin/env python3
"""
mlx_trainer.py - MLX-Optimized Trainer for AS Flare Prediction
Fully optimized for Apple Silicon using MLX framework
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List
import sys

from mlx_flare_net import MLXFlareNetLSTM, focal_loss, create_mlx_model


class MLXASFlareDataset:
    """Efficient dataset for MLX training"""
    
    def __init__(self, data_path: str, sequence_length: int = 30):
        self.sequence_length = sequence_length
        
        print(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print(f"Loaded {len(self.df):,} rows")
        
        # Define feature columns
        self.feature_cols = [col for col in self.df.columns
                            if col not in ['patient_id', 'day_index', 'will_flare_3_7d']]
        self.target_col = 'will_flare_3_7d'
        
        # Normalize features
        self._normalize_features()
        
        # Create sequences
        self._create_sequences()
    
    def _normalize_features(self):
        """Normalize features to 0-1 range"""
        for col in self.feature_cols:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
    
    def _create_sequences(self):
        """Create sliding window sequences"""
        print(f"Creating sequences with window size {self.sequence_length}...")
        
        # Sort by patient and day
        self.df = self.df.sort_values(['patient_id', 'day_index'])
        
        # Convert to numpy for speed
        data_values = self.df[self.feature_cols].values.astype(np.float32)
        target_values = self.df[self.target_col].values.astype(np.int32)
        patient_ids = self.df['patient_id'].values
        
        all_seqs = []
        all_targets = []
        
        # Process each patient
        for patient_id, group in self.df.groupby('patient_id'):
            if len(group) < self.sequence_length + 7:
                continue
            
            p_data = group[self.feature_cols].values.astype(np.float32)
            p_targets = group[self.target_col].values.astype(np.int32)
            
            num_windows = len(p_data) - self.sequence_length - 7
            if num_windows <= 0:
                continue
            
            # Create windows for this patient
            for i in range(num_windows):
                seq = p_data[i:i+self.sequence_length]
                target = p_targets[i+self.sequence_length]
                all_seqs.append(seq)
                all_targets.append(target)
        
        # Convert to arrays
        self.sequences = np.array(all_seqs, dtype=np.float32)
        self.targets = np.array(all_targets, dtype=np.int32)
        
        print(f"Created {len(self.sequences):,} sequences")
        print(f"Class balance: {np.mean(self.targets):.2%} positive")
    
    def get_train_val_test_split(self, train_ratio=0.6, val_ratio=0.2):
        """Split data into train/val/test sets"""
        total_size = len(self.sequences)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        
        # Shuffle indices
        indices = np.random.permutation(total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        return {
            'train': (self.sequences[train_indices], self.targets[train_indices]),
            'val': (self.sequences[val_indices], self.targets[val_indices]),
            'test': (self.sequences[test_indices], self.targets[test_indices])
        }


class MLXTrainer:
    """MLX-optimized trainer"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Paths
        self.checkpoint_dir = Path('models')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Training state
        self.best_metrics = {'accuracy': 0, 'f1': 0, 'val_loss': float('inf')}
        self.history = []
    
    def _get_default_config(self):
        return {
            'architecture': 'LSTM',
            'input_dim': 92,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.3,
            'batch_size': 256,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 100,
            'early_stopping_patience': 10,
            'focal_gamma': 2.0,
            'data_path': 'data/training_data_500k.parquet'  # Use smaller dataset to avoid timeout
        }
    
    def load_data(self, data_path: str = None):
        """Load and prepare datasets"""
        print("\nLoading dataset...")
        data_path = data_path or self.config['data_path']
        
        dataset = MLXASFlareDataset(data_path, sequence_length=30)
        splits = dataset.get_train_val_test_split()
        
        # Convert to MLX arrays
        self.train_data = mx.array(splits['train'][0])
        self.train_targets = mx.array(splits['train'][1])
        self.val_data = mx.array(splits['val'][0])
        self.val_targets = mx.array(splits['val'][1])
        self.test_data = mx.array(splits['test'][0])
        self.test_targets = mx.array(splits['test'][1])
        
        print(f"Train: {len(self.train_data):,} | Val: {len(self.val_data):,} | Test: {len(self.test_data):,}")
        
        # Calculate class weights
        class_counts = np.bincount(splits['train'][1])
        class_weights = (1.0 / class_counts) / (1.0 / class_counts).sum()
        self.class_weights = mx.array(class_weights)
        print(f"Class weights: {class_weights}")
    
    def create_model(self):
        """Create model and optimizer"""
        self.model = create_mlx_model(self.config)
        self.optimizer = optim.AdamW(
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def loss_fn(self, model, X, y):
        """Compute loss"""
        output = model(X)
        return focal_loss(
            output['logits'],
            y,
            gamma=self.config['focal_gamma'],
            alpha=self.class_weights
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        batch_size = self.config['batch_size']
        num_batches = len(self.train_data) // batch_size
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        # Shuffle data
        indices = mx.random.permutation(len(self.train_data))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.train_data))
            
            batch_indices = indices[start_idx:end_idx]
            batch_data = self.train_data[batch_indices]
            batch_targets = self.train_targets[batch_indices]
            
            # Forward and backward pass
            loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
            loss, grads = loss_and_grad_fn(self.model, batch_data, batch_targets)
            
            # Update parameters
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
            
            # Track metrics
            total_loss += loss.item()
            
            # Get predictions
            output = self.model(batch_data)
            preds = mx.argmax(output['logits'], axis=-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_targets.tolist())
            
            # Progress update
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}")
                sys.stdout.flush()
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = np.mean(all_preds == all_targets)
        
        # Calculate F1 score
        tp = np.sum((all_preds == 1) & (all_targets == 1))
        fp = np.sum((all_preds == 1) & (all_targets == 0))
        fn = np.sum((all_preds == 0) & (all_targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}
    
    def validate(self, data, targets):
        """Validate model"""
        batch_size = self.config['batch_size']
        num_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            
            batch_data = data[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Forward pass only
            loss = self.loss_fn(self.model, batch_data, batch_targets)
            output = self.model(batch_data)
            
            total_loss += loss.item()
            
            preds = mx.argmax(output['logits'], axis=-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_targets.tolist())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = np.mean(all_preds == all_targets)
        
        # Calculate F1 score
        tp = np.sum((all_preds == 1) & (all_targets == 1))
        fp = np.sum((all_preds == 1) & (all_targets == 0))
        fn = np.sum((all_preds == 0) & (all_targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}
    
    def train(self, epochs: int = None):
        """Main training loop"""
        epochs = epochs or self.config['epochs']
        print(f"\n🚀 Starting MLX training for {epochs} epochs...")
        
        no_improvement = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{epochs}")
            print('='*60)
            
            # Train
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time
            
            # Validate
            val_metrics = self.validate(self.val_data, self.val_targets)
            
            # Log metrics
            print(f"\n📊 Metrics:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            print(f"  Time: {train_time:.2f}s | LR: {self.config['learning_rate']:.6f}")
            
            # Track history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            })
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_metrics['accuracy']:
                self.best_metrics['accuracy'] = val_metrics['accuracy']
                self.best_metrics['f1'] = val_metrics['f1']
                self.best_metrics['val_loss'] = val_metrics['loss']
                no_improvement = 0
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics)
                print("  ✅ New best model saved!")
            else:
                no_improvement += 1
            
            # Early stopping
            if no_improvement >= self.config['early_stopping_patience']:
                print(f"\n⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save history
        self.save_history()
        
        # Final test evaluation
        print("\n🧪 Final test set evaluation...")
        test_metrics = self.validate(self.test_data, self.test_targets)
        print(f"  Test - Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / 'best_model_mlx.npz'
        # Save model parameters
        model_params = dict(self.model.parameters())
        mx.savez(str(checkpoint_path), **model_params)
        print(f"  Saved checkpoint to {checkpoint_path}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.log_dir / 'training_history_mlx.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("MLX TRAINER - APPLE SILICON OPTIMIZED")
    print("="*60)
    
    # Initialize trainer
    trainer = MLXTrainer()
    
    # Load data
    trainer.load_data()
    
    # Create model
    trainer.create_model()
    
    # Train
    test_metrics = trainer.train(epochs=100)
    
    print("\n✨ Training complete!")


if __name__ == "__main__":
    main()
