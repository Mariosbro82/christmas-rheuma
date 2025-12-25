#!/usr/bin/env python3
"""
trainer.py - Self-Correcting Training Loop for OVERLORD System
Autonomous training with automatic hyperparameter adjustment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List, Optional
import sys
import os
import argparse
import boto3
import botocore

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from neural_flare_net import create_model, NeuralFlareNet

def download_from_s3(bucket: str, key: str, local_path: str):
    """Download file from S3 if it doesn't exist locally"""
    if os.path.exists(local_path):
        print(f"File {local_path} already exists. Skipping download.")
        return

    print(f"Downloading s3://{bucket}/{key} to {local_path}...")
    s3 = boto3.client('s3')
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        print("Download complete!")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"The object s3://{bucket}/{key} does not exist.")
        else:
            print(f"Error downloading from S3: {e}")
        raise

class ASFlareDataset(Dataset):
    """PyTorch Dataset for AS flare prediction"""

    def __init__(self, data_path: str, sequence_length: int = 30, cache_data: bool = True):
        self.sequence_length = sequence_length

        print(f"Loading data from {data_path}...")
        print(f"Loading data from {data_path}...")
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_parquet(data_path)
        print(f"Loaded {len(self.df):,} rows")

        # Define feature columns (exclude target and metadata)
        self.feature_cols = [col for col in self.df.columns
                            if col not in ['patient_id', 'day_index', 'will_flare_3_7d']]

        self.target_col = 'will_flare_3_7d'

        # Normalize features
        self.normalize_features()

        # Create sequences
        self.sequences = []
        self.targets = []
        self._create_sequences()

    def normalize_features(self):
        """Normalize features to 0-1 range"""
        for col in self.feature_cols:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)

    def _create_sequences(self):
        """Create sliding window sequences using vectorized numpy operations"""
        print(f"Creating sequences with window size {self.sequence_length} (Vectorized)...")

        # Sort by patient and day
        self.df = self.df.sort_values(['patient_id', 'day_index'])
        
        # Convert to numpy for speed
        data_values = self.df[self.feature_cols].values.astype(np.float32)
        target_values = self.df[self.target_col].values.astype(np.int64)
        patient_ids = self.df['patient_id'].values
        
        # We need to ensure we don't cross patient boundaries
        # Identify valid start indices:
        # 1. The patient_id at start + sequence_length + 7 must be the same as at start
        # 2. We need enough data points
        
        # Create windows using stride tricks (much faster)
        from numpy.lib.stride_tricks import sliding_window_view
        
        # We'll process each patient group efficiently
        all_seqs = []
        all_targets = []
        
        # Group by patient to handle boundaries correctly
        # This is still a loop over patients, but we avoid the inner loop over days
        for patient_id, group in self.df.groupby('patient_id'):
            if len(group) < self.sequence_length + 7:
                continue
                
            # Extract data for this patient
            p_data = group[self.feature_cols].values.astype(np.float32)
            p_targets = group[self.target_col].values.astype(np.int64)
            
            # Create sliding windows
            # Shape: (num_windows, window_size, num_features)
            windows = sliding_window_view(p_data, window_shape=(self.sequence_length, len(self.feature_cols)))
            windows = windows.squeeze(axis=1) # Remove the singleton dimension if present, or adjust shape
            # sliding_window_view returns (N - W + 1, W, F) if input is (N, F) and window is (W, F)
            # Actually, sliding_window_view on 2D array with window_shape=(W, F) 
            # returns (N-W+1, 1, W, F). We need to squeeze.
            
            # We need targets at index i + sequence_length (which is 7 days before the prediction target? 
            # Wait, target column 'will_flare_7d' is already the target.
            # The original code:
            # seq = patient_data.iloc[i:i+self.sequence_length]
            # target = patient_data.iloc[i+self.sequence_length][self.target_col]
            # So target is the value at the step immediately FOLLOWING the sequence.
            
            # Valid indices for windows: 0 to len - seq_len
            # Valid indices for targets: seq_len to len
            
            # However, the original code had a +7 offset check?
            # if len(patient_data) < self.sequence_length + 7:
            # range(len(patient_data) - self.sequence_length - 7)
            # This implies we are predicting 7 days into the future relative to the end of the sequence?
            # But 'will_flare_7d' usually means "will flare in the next 7 days".
            # If the target column is already pre-calculated, we just need the value at the end of the sequence.
            # Let's stick to the original logic:
            # target index = i + self.sequence_length
            # But the loop range was `len - seq_len - 7`. Why -7?
            # Maybe to ensure we have 7 days of future data to verify? 
            # Or maybe the target column isn't pre-shifted?
            # "predict 7 days"
            # If `will_flare_7d` is already computed, then we just need that value.
            # If the loop range has -7, it means we are skipping the last 7 days of data?
            # Let's assume the original logic was correct about indices.
            
            num_windows = len(p_data) - self.sequence_length - 7
            if num_windows <= 0:
                continue
                
            # Windows
            # p_data shape (N, F)
            # We want windows starting at 0, 1, ... num_windows-1
            # sliding_window_view is easiest
            
            # Let's just use simple slicing with list comprehension which is faster than pandas iloc
            # But stride_tricks is best.
            
            # Using stride_tricks on the full patient array
            # We need to slice the result to num_windows
            
            try:
                win_view = sliding_window_view(p_data, window_shape=(self.sequence_length, len(self.feature_cols)))
                # win_view shape: (N - W + 1, 1, W, F) -> squeeze -> (N - W + 1, W, F)
                win_view = win_view.squeeze(axis=1)
                
                # We only want the first num_windows
                batch_seqs = win_view[:num_windows]
                
                # Targets
                # We want targets at indices: sequence_length, sequence_length+1, ...
                # corresponding to the windows.
                # The original code: target = patient_data.iloc[i+self.sequence_length]
                # So for i=0, target is at index sequence_length.
                batch_targets = p_targets[self.sequence_length : self.sequence_length + num_windows]
                
                all_seqs.append(batch_seqs)
                all_targets.append(batch_targets)
            except ValueError:
                continue

        if all_seqs:
            self.sequences = np.concatenate(all_seqs)
            self.targets = np.concatenate(all_targets)
        else:
            self.sequences = np.array([])
            self.targets = np.array([])

        print(f"Created {len(self.sequences):,} sequences")
        print(f"Class balance: {np.mean(self.targets):.2%} positive")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.targets[idx]]).squeeze()

class SelfCorrectingTrainer:
    """Autonomous trainer with self-correction capabilities"""

    def __init__(self, config: Optional[Dict] = None):
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Default configuration
        self.config = config or self._get_default_config()

        # Training history
        self.history = []
        self.best_metrics = {'accuracy': 0, 'f1': 0, 'val_loss': float('inf')}
        self.plateau_counter = 0
        self.architecture_switches = 0

        # Paths
        self.checkpoint_dir = Path('models')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)

    def _get_device(self):
        """Get best available device (MPS for Apple Silicon, CUDA, or CPU)"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _get_default_config(self):
        """Get default training configuration"""
        return {
            # Model config
            'architecture': 'LSTM',
            'input_dim': 92,  # Updated for comprehensive feature set
            'hidden_dim': 256,  # Increased for more features
            'num_layers': 3,
            'dropout': 0.3,

            # Training config
            'batch_size': 256,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 100,
            'early_stopping_patience': 10,

            # Loss config
            'focal_gamma': 2.0,

            # Self-correction thresholds
            'plateau_patience': 5,
            'switch_architecture_threshold': 10,
            'min_improvement': 0.001
        }

    def load_data(self, data_path: str = 'data/comprehensive_training_data.parquet', bucket_name: Optional[str] = None):
        """Load and prepare datasets, optionally downloading from S3"""
        
        # Handle S3 download if bucket is provided
        if bucket_name:
            # Assume the key in S3 matches the relative path structure or just the filename
            # For simplicity, let's assume the user uploads to 'data/filename'
            key = f"data/{os.path.basename(data_path)}"
            try:
                download_from_s3(bucket_name, key, data_path)
            except Exception as e:
                print(f"Failed to download data: {e}")
                # If download fails, we might still try to load if it exists, or fail hard.
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data file {data_path} not found and download failed.")

        print("\nLoading dataset...")

        # Create dataset
        dataset = ASFlareDataset(data_path, sequence_length=30)

        # Split into train/val/test (60/20/20)
        total_size = len(dataset)
        train_size = int(0.6 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 for MPS compatibility
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

        print(f"Train: {len(self.train_dataset):,} | Val: {len(self.val_dataset):,} | Test: {len(self.test_dataset):,}")

        # Calculate class weights for focal loss (Optimized)
        print("Calculating class weights...")
        if isinstance(self.train_dataset, torch.utils.data.Subset):
            # Fast path: access targets directly using indices
            train_indices = self.train_dataset.indices
            train_targets = dataset.targets[train_indices]
            class_counts = np.bincount(train_targets)
        else:
            # Fallback (slow)
            all_targets = []
            for _, target in self.train_loader:
                all_targets.extend(target.numpy())
            class_counts = np.bincount(all_targets)
            
        self.config['class_weights'] = (1.0 / class_counts) / (1.0 / class_counts).sum()
        print(f"Class weights: {self.config['class_weights']}")

    def create_model(self):
        """Create or recreate model"""
        self.model, self.criterion = create_model(self.config)
        self.model = self.model.to(self.device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=1e-6
        )

    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output['logits'], target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            preds = output['logits'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # Progress update
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
                sys.stdout.flush()

        # Calculate epoch metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        avg_loss = running_loss / len(self.train_loader)

        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1}

    def validate(self, loader: DataLoader) -> Dict:
        """Validate model on given loader"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output['logits'], target)

                running_loss += loss.item()

                probs = torch.softmax(output['logits'], dim=1)
                preds = output['logits'].argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        avg_loss = running_loss / len(loader)

        # ROC-AUC if we have both classes
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'predictions': all_preds,
            'targets': all_targets
        }

    def self_correct(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Apply self-correction based on metrics"""
        print("\n🔧 Self-Correction Analysis...")

        # Detect overfitting
        if train_metrics['loss'] < val_metrics['loss'] * 0.7:
            print("  ⚠️ Overfitting detected!")
            # Increase dropout
            self.config['dropout'] = min(0.5, self.config['dropout'] + 0.05)
            # Increase weight decay
            self.config['weight_decay'] *= 1.5
            print(f"  → Increased dropout to {self.config['dropout']:.2f}")
            print(f"  → Increased weight decay to {self.config['weight_decay']:.4f}")

        # Detect underfitting
        elif train_metrics['accuracy'] < 0.75:
            print("  ⚠️ Underfitting detected!")
            # Increase model capacity
            self.config['hidden_dim'] = min(256, int(self.config['hidden_dim'] * 1.2))
            print(f"  → Increased hidden_dim to {self.config['hidden_dim']}")

        # Detect plateau
        if self.plateau_counter >= self.config['plateau_patience']:
            print("  ⚠️ Performance plateau detected!")

            # Try learning rate reduction first
            if self.plateau_counter == self.config['plateau_patience']:
                self.config['learning_rate'] *= 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['learning_rate']
                print(f"  → Reduced learning rate to {self.config['learning_rate']:.5f}")

            # Switch architecture if still stuck
            elif self.plateau_counter >= self.config['switch_architecture_threshold']:
                if self.architecture_switches < 2:  # Limit architecture switches
                    new_arch = 'Transformer' if self.config['architecture'] == 'LSTM' else 'LSTM'
                    print(f"  → SWITCHING ARCHITECTURE: {self.config['architecture']} → {new_arch}")

                    self.config['architecture'] = new_arch
                    self.create_model()  # Recreate with new architecture
                    self.architecture_switches += 1
                    self.plateau_counter = 0

        # Check for NaN/Inf
        if np.isnan(train_metrics['loss']) or np.isinf(train_metrics['loss']):
            print("  💀 Training diverged! Restarting with lower learning rate...")
            self.config['learning_rate'] *= 0.1
            self.create_model()

    def train(self, epochs: Optional[int] = None):
        """Main training loop with self-correction"""
        epochs = epochs or self.config['epochs']
        print(f"\n🚀 Starting training for {epochs} epochs...")

        no_improvement_counter = 0

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{epochs}")
            print('='*60)

            # Train
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time

            # Validate
            val_metrics = self.validate(self.val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            print(f"\n📊 Metrics:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            print(f"  Time: {train_time:.2f}s | LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Track history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            })

            # Check for improvement
            improved = False
            if val_metrics['accuracy'] > self.best_metrics['accuracy'] + self.config['min_improvement']:
                self.best_metrics['accuracy'] = val_metrics['accuracy']
                self.best_metrics['f1'] = val_metrics['f1']
                self.best_metrics['val_loss'] = val_metrics['loss']
                improved = True
                no_improvement_counter = 0
                self.plateau_counter = 0

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics)
                print("  ✅ New best model saved!")
            else:
                no_improvement_counter += 1
                self.plateau_counter += 1

            # Self-correction
            self.self_correct(epoch, train_metrics, val_metrics)

            # Check victory condition
            if val_metrics['accuracy'] > 0.92 and val_metrics['f1'] > 0.85:
                print("\n🏆 VICTORY! Target metrics achieved!")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f} > 0.92")
                print(f"  F1 Score: {val_metrics['f1']:.4f} > 0.85")
                self.save_final_results(val_metrics)
                break

            # Early stopping
            if no_improvement_counter >= self.config['early_stopping_patience']:
                print(f"\n⚠️ Early stopping after {epoch+1} epochs")
                break

        # Save training history
        self.save_history()

        # Final test evaluation
        print("\n🧪 Final test set evaluation...")
        test_metrics = self.validate(self.test_loader)
        print(f"  Test - Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}")

        return test_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')

    def save_history(self):
        """Save training history"""
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_final_results(self, metrics: Dict):
        """Save final results for evaluator"""
        results = {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1'],
            'val_loss': metrics['loss'],
            'architecture': self.config['architecture'],
            'total_epochs': len(self.history),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.log_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("OVERLORD TRAINER - AUTONOMOUS TRAINING SYSTEM")
    print("="*60)

    # Parse arguments
    parser = argparse.ArgumentParser(description='OVERLORD Trainer')
    parser.add_argument('--bucket', type=str, help='S3 bucket name for data download')
    parser.add_argument('--data-path', type=str, default='data/comprehensive_training_data.parquet', 
                        help='Path to training data (local path, will be downloaded to here if --bucket is set)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    args = parser.parse_args()

    # Initialize trainer
    trainer = SelfCorrectingTrainer()

    # Load data
    trainer.load_data(data_path=args.data_path, bucket_name=args.bucket)

    # Create initial model
    trainer.create_model()

    # Train with self-correction
    test_metrics = trainer.train(epochs=args.epochs)

    # Evaluate with evaluator
    if Path('final_results.json').exists():
        with open('final_results.json', 'r') as f:
            results = json.load(f)

        # Measure inference time
        import timeit
        dummy_input = torch.randn(1, 30, 92).to(trainer.device)

        def inference():
            with torch.no_grad():
                trainer.model(dummy_input)

        latency = timeit.timeit(inference, number=100) / 100 * 1000  # ms

        # Run evaluator
        os.system(f'{sys.executable} evaluator.py --accuracy {results["accuracy"]} --f1 {results["f1_score"]} --latency {latency}')

if __name__ == "__main__":
    main()