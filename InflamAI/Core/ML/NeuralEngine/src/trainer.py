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
# AWS imports moved to function scope to allow local-only execution

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from neural_flare_net import create_model, NeuralFlareNet

def download_from_s3(bucket: str, key: str, local_path: str):
    """Download file from S3 if it doesn't exist locally"""
    if os.path.exists(local_path):
        print(f"File {local_path} already exists. Skipping download.")
        return

    print(f"Downloading s3://{bucket}/{key} to {local_path}...")
    try:
        import boto3
        import botocore
    except ImportError:
        print("Error: 'boto3' is not installed. Please install it to download from S3:")
        print("pip install boto3")
        raise

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
    """
    PyTorch Dataset for AS flare prediction with Lazy Loading.
    
    Memory Optimization:
    Instead of pre-calculating all sequences (which duplicates data 30x),
    we store the raw data once and slice it on-the-fly in __getitem__.
    """

    def __init__(self, data_path: str, sequence_length: int = 30):
        self.sequence_length = sequence_length

        print(f"Loading data from {data_path}...")
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_parquet(data_path)
        
        # Optimize memory types immediately
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        for col in self.df.select_dtypes(include=['int64']).columns:
            self.df[col] = self.df[col].astype('int32')
            
        print(f"Loaded {len(self.df):,} rows")

        # Define feature columns (exclude target and metadata)
        self.feature_cols = [col for col in self.df.columns
                            if col not in ['patient_id', 'day_index', 'will_flare_3_7d']]
        
        self.target_col = 'will_flare_3_7d'

        # Normalize features (in-place to save memory)
        self.normalize_features()

        # Prepare for lazy loading
        self._prepare_indices()

    def normalize_features(self):
        """Normalize features to 0-1 range"""
        print("Normalizing features...")
        for col in self.feature_cols:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)

    def _prepare_indices(self):
        """
        Identify valid start indices for sequences.
        A valid sequence must:
        1. Have length `sequence_length`
        2. Belong to the same patient
        """
        print("Indexing sequences (Lazy Loading setup)...")
        
        # Sort by patient and day to ensure contiguous blocks
        self.df = self.df.sort_values(['patient_id', 'day_index']).reset_index(drop=True)
        
        # Convert to numpy for fast indexing in __getitem__
        # We keep the whole dataset in memory, but it's only 1x size, not 30x
        self.data_values = self.df[self.feature_cols].values.astype(np.float32)
        self.target_values = self.df[self.target_col].values.astype(np.int64)
        self.patient_ids = self.df['patient_id'].values
        
        # Find valid start indices
        # We can start a sequence at index `i` if:
        # patient_id[i] == patient_id[i + sequence_length]
        # (This implies all intermediate points are also the same patient due to sorting)
        
        n_samples = len(self.df)
        indices = np.arange(n_samples - self.sequence_length)
        
        # Check patient boundaries
        # We need patient_id[i] == patient_id[i + sequence_length - 1] 
        # actually we just need the sequence to be valid.
        # And we need the target to be valid.
        # The target is at `i + sequence_length` (next day prediction? or pre-calculated?)
        # Based on previous code: target was at `i + sequence_length`.
        # So we need index `i + sequence_length` to exist and be same patient.
        
        # Vectorized check for valid sequences
        # patient at start == patient at end of sequence (plus target offset if needed)
        # Let's assume we need `sequence_length` points for input, and the target is at the end.
        
        valid_mask = (self.patient_ids[indices] == self.patient_ids[indices + self.sequence_length])
        
        self.valid_indices = indices[valid_mask]
        
        print(f"Found {len(self.valid_indices):,} valid sequences")
        
        # Free up pandas dataframe to save memory
        del self.df
        import gc
        gc.collect()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual start index in the data array
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Slice the sequence (Lazy loading!)
        # Shape: (sequence_length, num_features)
        sequence = self.data_values[start_idx : end_idx]
        
        # Get target (at the end of the sequence)
        # Note: In the original code, target was taken at `i + sequence_length`
        target = self.target_values[end_idx]
        
        return torch.FloatTensor(sequence), torch.LongTensor([target]).squeeze()

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
        # Calculate class weights for focal loss (Optimized)
        print("Calculating class weights...")
        # Since we don't have direct access to all targets in a simple array anymore (it's in self.target_values but indexed by valid_indices)
        # We need to be careful.
        
        # Access the underlying dataset from the Subset
        if isinstance(self.train_dataset, torch.utils.data.Subset):
            # Get the indices for the training set
            train_indices = self.train_dataset.indices
            # Map these to the actual data indices in the dataset
            actual_indices = dataset.valid_indices[train_indices]
            # Get targets
            # Target is at index + sequence_length
            target_indices = actual_indices + dataset.sequence_length
            train_targets = dataset.target_values[target_indices]
            
            class_counts = np.bincount(train_targets)
        else:
             # Fallback if not a subset (unlikely with random_split)
            class_counts = np.bincount(dataset.target_values[dataset.valid_indices + dataset.sequence_length])

        # Avoid division by zero
        class_counts = class_counts + 1 
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

    def self_correct(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> bool:
        """
        Apply self-correction based on metrics.
        Returns: True if a major change occurred (reset patience), False otherwise.
        """
        print("\n🔧 Self-Correction Analysis...")
        reset_patience = False

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
                    reset_patience = True

        # Check for NaN/Inf
        if np.isnan(train_metrics['loss']) or np.isinf(train_metrics['loss']):
            print("  💀 Training diverged! Restarting with lower learning rate...")
            self.config['learning_rate'] *= 0.1
            self.create_model()
            reset_patience = True

        return reset_patience

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
            if self.self_correct(epoch, train_metrics, val_metrics):
                print("  🔄 Major change detected - Resetting early stopping counter!")
                no_improvement_counter = 0

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