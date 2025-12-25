#!/usr/bin/env python3
"""
data_generator.py - Massive AS Patient Data Generator
Generates 2.1M rows of realistic synthetic patient data for ML training
Based on medical literature and ASPatientSyntheticDataGenerator.swift patterns
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import time
from typing import Tuple, Dict, List
import json
from pathlib import Path

class ASDataGenerator:
    """Generate massive synthetic AS patient data with realistic correlations"""

    def __init__(self, n_patients: int = 1000, days_per_patient: int = 2100):
        self.n_patients = n_patients
        self.days_per_patient = days_per_patient
        self.total_rows = n_patients * days_per_patient

        # Set random seed for reproducibility
        np.random.seed(42)

        print(f"🔥 OVERLORD DATA GENERATOR INITIALIZED")
        print(f"Target: {self.total_rows:,} rows ({n_patients} patients × {days_per_patient} days)")

    def generate_patient_demographics(self) -> pd.DataFrame:
        """Generate realistic patient demographics based on AS epidemiology"""
        print("Generating patient demographics...")

        # Gender distribution (2:1 male:female ratio in AS)
        genders = np.random.choice(['M', 'F'], self.n_patients, p=[0.67, 0.33])

        # Age distribution (peak 25-35)
        ages = np.random.normal(30, 7, self.n_patients).clip(18, 65).astype(int)

        # HLA-B27 status (90% positive in AS patients)
        hla_b27 = np.random.choice([1, 0], self.n_patients, p=[0.90, 0.10])

        # Disease duration
        onset_ages = np.random.normal(25, 5, self.n_patients).clip(15, 40).astype(int)
        disease_duration = (ages - onset_ages).clip(0, None)

        # BMI (slightly lower in AS patients)
        bmis = np.random.normal(24.5, 4, self.n_patients).clip(18.5, 35)

        # Smoking status affects progression
        smoking = np.random.choice([0, 1, 2], self.n_patients, p=[0.60, 0.25, 0.15])
        # 0: never, 1: former, 2: current

        demographics = pd.DataFrame({
            'patient_id': range(self.n_patients),
            'gender': genders,
            'age': ages,
            'hla_b27': hla_b27,
            'disease_duration': disease_duration,
            'bmi': bmis,
            'smoking': smoking
        })

        return demographics

    def generate_time_series_features(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series data for each patient"""
        print("Generating time-series features (this may take a moment)...")
        start_time = time.time()

        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_per_patient)

        # Vectorized approach for speed
        all_data = []

        for patient_idx in range(self.n_patients):
            if patient_idx % 100 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (patient_idx + 1)) * (self.n_patients - patient_idx)
                print(f"  Patient {patient_idx}/{self.n_patients} - ETA: {eta:.1f}s", end='\r')

            patient = demographics.iloc[patient_idx]

            # Base disease activity influenced by demographics
            base_basdai = 3.0 + np.random.random() * 3.0  # 3-6 range

            # Smoking increases activity
            if patient['smoking'] == 2:  # Current smoker
                base_basdai *= 1.15

            # Disease duration effect (stabilizes over time)
            if patient['disease_duration'] > 10:
                base_basdai *= 0.9

            # Generate daily variations
            daily_basdai = np.random.normal(base_basdai, 0.8, self.days_per_patient)
            daily_basdai = np.clip(daily_basdai, 0, 10)

            # Add flare patterns (2-6 flares per year)
            flare_frequency = 2 + np.random.random() * 4
            n_flares = int(flare_frequency * (self.days_per_patient / 365))

            for _ in range(n_flares):
                flare_start = np.random.randint(0, self.days_per_patient - 14)
                flare_duration = np.random.randint(7, 21)
                flare_intensity = 1.5 + np.random.random() * 2.0

                # Apply flare
                flare_end = min(flare_start + flare_duration, self.days_per_patient)
                daily_basdai[flare_start:flare_end] *= flare_intensity

            daily_basdai = np.clip(daily_basdai, 0, 10)

            # Weather patterns (critical for AS)
            # Barometric pressure with realistic variability
            pressure = 1013 + np.random.normal(0, 10, self.days_per_patient)
            pressure = pd.Series(pressure).rolling(3, min_periods=1).mean().values  # Smooth changes

            # Pressure changes (12h differences)
            pressure_change = np.diff(pressure, prepend=pressure[0])
            pressure_change = pd.Series(pressure_change).rolling(2, min_periods=1).mean().values

            # Rapid pressure drops trigger flares
            pressure_trigger_mask = pressure_change < -5.0
            daily_basdai[pressure_trigger_mask] *= 1.2

            # Temperature with seasonal variation
            day_of_year = np.arange(self.days_per_patient) % 365
            seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            temperature = seasonal_temp + np.random.normal(0, 3, self.days_per_patient)

            # Humidity
            humidity = 60 + np.random.normal(0, 15, self.days_per_patient)
            humidity = np.clip(humidity, 20, 100)

            # Biometric data correlated with disease activity
            # HRV decreases with inflammation
            hrv = 50 - daily_basdai * 3 + np.random.normal(0, 5, self.days_per_patient)
            hrv = np.clip(hrv, 20, 80)

            # Resting HR increases with inflammation
            resting_hr = 60 + daily_basdai * 2 + np.random.normal(0, 3, self.days_per_patient)
            resting_hr = np.clip(resting_hr, 50, 90)

            # Step count decreases with pain
            steps = 10000 - daily_basdai * 1000 + np.random.normal(0, 1000, self.days_per_patient)
            steps = np.clip(steps, 1000, 15000).astype(int)

            # Sleep quality affected by pain
            sleep_hours = 8 - daily_basdai * 0.3 + np.random.normal(0, 0.5, self.days_per_patient)
            sleep_hours = np.clip(sleep_hours, 4, 9)

            # Morning stiffness (minutes)
            morning_stiffness = 30 + daily_basdai * 10 + np.random.normal(0, 10, self.days_per_patient)
            morning_stiffness = np.clip(morning_stiffness, 0, 180).astype(int)

            # Fatigue level
            fatigue = daily_basdai * 0.8 + np.random.normal(0, 1, self.days_per_patient)
            fatigue = np.clip(fatigue, 0, 10)

            # Pain levels by body region
            si_joint_pain = daily_basdai * 0.9 + np.random.normal(0, 0.5, self.days_per_patient)
            lumbar_pain = daily_basdai * 0.8 + np.random.normal(0, 0.5, self.days_per_patient)
            thoracic_pain = daily_basdai * 0.6 + np.random.normal(0, 0.5, self.days_per_patient)

            # Clip all pain scores
            si_joint_pain = np.clip(si_joint_pain, 0, 10)
            lumbar_pain = np.clip(lumbar_pain, 0, 10)
            thoracic_pain = np.clip(thoracic_pain, 0, 10)

            # Create flare labels (for next 7 days prediction)
            will_flare_7d = np.zeros(self.days_per_patient)
            for i in range(self.days_per_patient - 7):
                # Check if any of next 7 days has BASDAI > 6
                if np.any(daily_basdai[i:i+7] > 6):
                    will_flare_7d[i] = 1

            # Build patient data
            patient_data = pd.DataFrame({
                'patient_id': patient_idx,
                'day_index': range(self.days_per_patient),
                'basdai_score': daily_basdai,
                'si_joint_pain': si_joint_pain,
                'lumbar_pain': lumbar_pain,
                'thoracic_pain': thoracic_pain,
                'morning_stiffness': morning_stiffness,
                'fatigue': fatigue,
                'pressure': pressure,
                'pressure_change': pressure_change,
                'temperature': temperature,
                'humidity': humidity,
                'hrv': hrv,
                'resting_hr': resting_hr,
                'steps': steps,
                'sleep_hours': sleep_hours,
                'will_flare_7d': will_flare_7d,
                # Add patient demographics (repeated for each day)
                'age': patient['age'],
                'gender': 1 if patient['gender'] == 'M' else 0,
                'hla_b27': patient['hla_b27'],
                'disease_duration': patient['disease_duration'],
                'bmi': patient['bmi'],
                'smoking': patient['smoking']
            })

            all_data.append(patient_data)

        print(f"\n  Combining all patient data...")
        full_dataset = pd.concat(all_data, ignore_index=True)

        elapsed = time.time() - start_time
        print(f"  ✅ Generated {len(full_dataset):,} rows in {elapsed:.2f} seconds")

        return full_dataset

    def create_sliding_windows(self, df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
        """Create sliding window features for time-series prediction"""
        print(f"Creating sliding windows (lookback={lookback} days)...")

        # For efficiency, we'll create a subset with sliding windows
        # This is where the ML model will actually train on sequences

        window_features = []
        feature_cols = ['basdai_score', 'si_joint_pain', 'lumbar_pain', 'thoracic_pain',
                       'morning_stiffness', 'fatigue', 'pressure', 'pressure_change',
                       'temperature', 'humidity', 'hrv', 'resting_hr', 'steps', 'sleep_hours']

        # Add lag features for key variables
        for col in ['basdai_score', 'pressure_change', 'hrv']:
            for lag in [1, 3, 7, 14]:
                df[f'{col}_lag_{lag}'] = df.groupby('patient_id')[col].shift(lag)

        # Add rolling statistics
        for col in ['basdai_score', 'pressure']:
            df[f'{col}_rolling_mean_7'] = df.groupby('patient_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f'{col}_rolling_std_7'] = df.groupby('patient_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).std()
            )

        # Fill NaN values from lag/rolling features
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        return df

    def validate_correlations(self, df: pd.DataFrame):
        """Validate that generated data has realistic correlations"""
        print("\nValidating correlations...")

        # Check key correlations that should exist in AS data
        correlations = {
            'BASDAI vs Pressure Drop': df['basdai_score'].corr(df['pressure_change']),
            'BASDAI vs HRV': df['basdai_score'].corr(df['hrv']),
            'BASDAI vs Steps': df['basdai_score'].corr(df['steps']),
            'BASDAI vs Sleep': df['basdai_score'].corr(df['sleep_hours']),
            'Morning Stiffness vs BASDAI': df['morning_stiffness'].corr(df['basdai_score'])
        }

        print("\n  Key Correlations:")
        for name, corr in correlations.items():
            status = "✅" if abs(corr) > 0.1 else "⚠️"
            print(f"    {status} {name}: {corr:.3f}")

        # Save correlation matrix
        corr_matrix = df[['basdai_score', 'pressure_change', 'hrv', 'steps',
                         'sleep_hours', 'morning_stiffness']].corr()

        return correlations

    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """Save dataset to Parquet format for fast loading"""
        print(f"\nSaving dataset to {output_path}...")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(exist_ok=True)

        # Save to Parquet (much faster than CSV for large datasets)
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')

        # Also save a small sample as CSV for inspection
        sample_path = output_path.replace('.parquet', '_sample.csv')
        df.head(1000).to_csv(sample_path, index=False)

        # Save metadata
        metadata = {
            'n_rows': len(df),
            'n_patients': df['patient_id'].nunique(),
            'n_features': len(df.columns),
            'features': list(df.columns),
            'target': 'will_flare_7d',
            'class_balance': {
                'no_flare': int((df['will_flare_7d'] == 0).sum()),
                'will_flare': int((df['will_flare_7d'] == 1).sum())
            },
            'generated_at': datetime.now().isoformat()
        }

        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ✅ Saved {len(df):,} rows ({file_size_mb:.1f} MB)")
        print(f"  📊 Class balance: {metadata['class_balance']}")

    def run(self):
        """Execute full data generation pipeline"""
        print("\n" + "="*60)
        print("OVERLORD DATA GENERATION PIPELINE")
        print("="*60)

        overall_start = time.time()

        # Step 1: Generate demographics
        demographics = self.generate_patient_demographics()

        # Step 2: Generate time-series features
        full_dataset = self.generate_time_series_features(demographics)

        # Step 3: Create sliding windows and engineered features
        full_dataset = self.create_sliding_windows(full_dataset)

        # Step 4: Validate correlations
        correlations = self.validate_correlations(full_dataset)

        # Step 5: Save dataset
        output_path = 'data/huge_training_set.parquet'
        self.save_dataset(full_dataset, output_path)

        # Update protocol file
        self.update_protocol(len(full_dataset), time.time() - overall_start, correlations)

        print("\n" + "="*60)
        print(f"✅ DATA GENERATION COMPLETE")
        print(f"Total time: {time.time() - overall_start:.2f} seconds")
        print(f"Rows per second: {len(full_dataset) / (time.time() - overall_start):,.0f}")
        print("="*60)

        return full_dataset

    def update_protocol(self, n_rows: int, duration: float, correlations: Dict):
        """Update OVERLORD_PROTOCOL.md with generation metrics"""
        protocol_path = Path('../OVERLORD_PROTOCOL.md')
        if not protocol_path.exists():
            return

        content = protocol_path.read_text()

        # Update metrics in Phase 1 section
        new_metrics = f"""- **Metrics:**
  - Rows Generated: {n_rows:,}
  - Generation Time: {duration:.2f}s
  - Correlation Strength: {abs(correlations.get('BASDAI vs Pressure Drop', 0)):.3f}"""

        # Simple replacement (could be more sophisticated)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Rows Generated:' in line:
                # Found the metrics section, replace next 3 lines
                for j in range(3):
                    if i + j < len(lines):
                        lines[i + j] = new_metrics.split('\n')[j] if j < len(new_metrics.split('\n')) else ''
                break

        protocol_path.write_text('\n'.join(lines))

def main():
    """Main entry point"""
    # Configuration
    N_PATIENTS = 1000
    DAYS_PER_PATIENT = 2100

    # Initialize generator
    generator = ASDataGenerator(n_patients=N_PATIENTS, days_per_patient=DAYS_PER_PATIENT)

    # Run generation pipeline
    dataset = generator.run()

    print(f"\n🔥 Ready for Phase 2: Model Architecture")

if __name__ == "__main__":
    main()