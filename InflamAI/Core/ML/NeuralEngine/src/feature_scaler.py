#!/usr/bin/env python3
"""
feature_scaler.py - Feature Normalization for Updatable CoreML Models
Saves and exports scaler parameters for on-device data normalization
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import pickle

class FeatureScalerManager:
    """Manage feature scaling for consistent normalization across training and inference"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """Fit scaler on training data"""
        print(f"📐 Fitting scaler on {len(feature_columns)} features...")
        
        self.feature_names = feature_columns
        X = df[feature_columns].values
        
        self.scaler.fit(X)
        self.is_fitted = True
        
        print(f"✅ Scaler fitted")
        print(f"   Mean range: [{self.scaler.mean_.min():.2f}, {self.scaler.mean_.max():.2f}]")
        print(f"   Std range: [{self.scaler.scale_.min():.2f}, {self.scaler.scale_.max():.2f}]")
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
            
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        df_scaled = df.copy()
        df_scaled[self.feature_names] = X_scaled
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def save_to_json(self, output_path: str = "data/scaler_params.json"):
        """Save scaler parameters to JSON for CoreML embedding"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        scaler_params = {
            'feature_names': self.feature_names,
            'means': self.scaler.mean_.tolist(),
            'stds': self.scaler.scale_.tolist(),
            'n_features': len(self.feature_names),
            'scaler_type': 'StandardScaler'
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"✅ Scaler parameters saved to: {output_path}")
        return scaler_params
    
    def save_to_pickle(self, output_path: str = "data/scaler.pkl"):
        """Save scaler object to pickle for Python use"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Scaler object saved to: {output_path}")
    
    @classmethod
    def load_from_json(cls, json_path: str) -> 'FeatureScalerManager':
        """Load scaler from JSON parameters"""
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        manager = cls()
        manager.feature_names = params['feature_names']
        manager.scaler.mean_ = np.array(params['means'])
        manager.scaler.scale_ = np.array(params['stds'])
        manager.is_fitted = True
        
        return manager
    
    @classmethod
    def load_from_pickle(cls, pickle_path: str) -> 'FeatureScalerManager':
        """Load scaler from pickle file"""
        with open(pickle_path, 'rb') as f:
            scaler = pickle.load(f)
        
        manager = cls()
        manager.scaler = scaler
        manager.is_fitted = True
        
        return manager
    
    def get_swift_code(self) -> str:
        """Generate Swift code for on-device normalization"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        swift_code = '''
// FeatureScaler.swift
// Auto-generated feature normalization code

import Foundation

struct FeatureScaler {
    let means: [Float]
    let stds: [Float]
    let featureNames: [String]
    
    init(from metadata: [String: String]) {
        guard let scalerJSON = metadata["scaler_params"],
              let data = scalerJSON.data(using: .utf8),
              let params = try? JSONDecoder().decode(ScalerParams.self, from: data) else {
            fatalError("Failed to load scaler parameters from model metadata")
        }
        
        self.means = params.means
        self.stds = params.stds
        self.featureNames = params.featureNames
    }
    
    func transform(_ features: [Float]) -> [Float] {
        guard features.count == means.count else {
            fatalError("Feature count mismatch: expected \\(means.count), got \\(features.count)")
        }
        
        return zip(zip(features, means), stds).map { ((value, mean), std) in
            (value - mean) / std
        }
    }
    
    func transform(_ features: [String: Float]) -> [Float] {
        // Convert dictionary to ordered array based on feature names
        let orderedFeatures = featureNames.map { features[$0] ?? 0.0 }
        return transform(orderedFeatures)
    }
}

private struct ScalerParams: Codable {
    let featureNames: [String]
    let means: [Float]
    let stds: [Float]
    let nFeatures: Int
    let scalerType: String
    
    enum CodingKeys: String, CodingKey {
        case featureNames = "feature_names"
        case means
        case stds
        case nFeatures = "n_features"
        case scalerType = "scaler_type"
    }
}
'''
        return swift_code
    
    def validate_normalization(self, df: pd.DataFrame, sample_size: int = 1000):
        """Validate that normalization produces expected distribution"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        print(f"\n🔍 Validating normalization...")
        
        # Transform sample
        sample = df.sample(min(sample_size, len(df)))
        transformed = self.transform(sample)
        
        # Check transformed features
        for col in self.feature_names:
            mean = transformed[col].mean()
            std = transformed[col].std()
            
            # Should be close to 0 mean, 1 std
            if abs(mean) > 0.1 or abs(std - 1.0) > 0.2:
                print(f"   ⚠️  {col}: mean={mean:.3f}, std={std:.3f}")
            else:
                print(f"   ✅ {col}: mean={mean:.3f}, std={std:.3f}")

def main():
    """Example usage"""
    print("=" * 70)
    print("FEATURE SCALER MANAGER")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading training data...")
    df = pd.read_parquet('data/comprehensive_training_data.parquet')
    
    # Define feature columns (exclude identifiers and target)
    exclude_cols = ['patient_id', 'day_index', 'will_flare_3_7d']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    print(f"   Features to normalize: {len(feature_columns)}")
    
    # Create and fit scaler
    scaler_manager = FeatureScalerManager()
    scaler_manager.fit(df, feature_columns)
    
    # Save scaler parameters
    scaler_params = scaler_manager.save_to_json('data/scaler_params.json')
    scaler_manager.save_to_pickle('data/scaler.pkl')
    
    # Validate normalization
    scaler_manager.validate_normalization(df, sample_size=10000)
    
    # Generate Swift code
    swift_code = scaler_manager.get_swift_code()
    swift_path = Path('data/FeatureScaler.swift')
    with open(swift_path, 'w') as f:
        f.write(swift_code)
    print(f"\n✅ Swift code generated: {swift_path}")
    
    print("\n" + "=" * 70)
    print("✅ FEATURE SCALER SETUP COMPLETE")
    print("=" * 70)
    print(f"\nScaler parameters ready for CoreML embedding:")
    print(f"  - JSON: data/scaler_params.json")
    print(f"  - Pickle: data/scaler.pkl")
    print(f"  - Swift: data/FeatureScaler.swift")

if __name__ == "__main__":
    main()
