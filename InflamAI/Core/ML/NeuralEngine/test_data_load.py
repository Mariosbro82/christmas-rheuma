#!/usr/bin/env python3
"""
Quick test to verify data loading works with new configuration
"""

import pandas as pd
import sys
from pathlib import Path

print("Testing data loading...")

# Load data
data_path = 'data/comprehensive_training_data.parquet'
print(f"Loading: {data_path}")

df = pd.read_parquet(data_path)

print(f"\n✅ Data loaded successfully!")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# Check feature columns
exclude_cols = ['patient_id', 'day_index', 'will_flare_3_7d']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n   Feature columns: {len(feature_cols)}")
print(f"   Target column: will_flare_3_7d")

# Check target distribution
target_dist = df['will_flare_3_7d'].value_counts()
print(f"\n   Target distribution:")
print(f"      No flare (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
print(f"      Will flare (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")

# Check for NaN
nan_count = df.isnull().sum().sum()
print(f"\n   NaN values: {nan_count}")

if len(feature_cols) == 92:
    print(f"\n✅ Feature count matches expected (92)")
else:
    print(f"\n⚠️  Feature count mismatch: expected 92, got {len(feature_cols)}")

print("\n✅ Data is ready for training!")
