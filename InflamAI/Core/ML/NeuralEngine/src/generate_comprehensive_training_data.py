#!/usr/bin/env python3
"""
generate_comprehensive_training_data.py - Comprehensive AS Patient Data Generator
Generates 2.1M+ rows of realistic synthetic patient data with 100+ features
Optimized for neural network training and updatable CoreML models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Tuple, Dict, List
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveASDataGenerator:
    """Generate massive synthetic AS patient data with all required features"""

    def __init__(self, n_patients: int = 1500, days_per_patient: int = 1400):
        self.n_patients = n_patients
        self.days_per_patient = days_per_patient
        self.total_rows = n_patients * days_per_patient
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Feature names (will be populated)
        self.feature_names = []
        
        print(f"🔥 COMPREHENSIVE AS DATA GENERATOR")
        print(f"Target: {self.total_rows:,} rows ({n_patients} patients × {days_per_patient} days)")
        print(f"Expected features: 100+")

    def generate_patient_profiles(self) -> pd.DataFrame:
        """Generate patient demographics and baseline characteristics"""
        print("\n📊 Generating patient profiles...")
        
        # Gender distribution (2:1 male:female in AS)
        genders = np.random.choice([1, 0], self.n_patients, p=[0.67, 0.33])  # 1=M, 0=F
        
        # Age distribution (peak 25-35)
        ages = np.random.normal(32, 8, self.n_patients).clip(18, 70).astype(int)
        
        # HLA-B27 status (90% positive in AS patients)
        hla_b27 = np.random.choice([1, 0], self.n_patients, p=[0.90, 0.10])
        
        # Disease duration
        onset_ages = np.random.normal(26, 6, self.n_patients).clip(16, 45).astype(int)
        disease_duration = (ages - onset_ages).clip(0, None)
        
        # BMI (slightly lower in AS patients)
        bmis = np.random.normal(24.5, 4, self.n_patients).clip(17, 38)
        
        # Smoking status (0: never, 1: former, 2: current)
        smoking = np.random.choice([0, 1, 2], self.n_patients, p=[0.60, 0.25, 0.15])
        
        # Baseline disease severity (affects all downstream features)
        base_severity = np.random.beta(2, 5, self.n_patients) * 10  # 0-10 scale, skewed low
        
        # Weather sensitivity (individual variation)
        weather_sensitivity = np.random.choice([0.5, 1.0, 1.5], self.n_patients, p=[0.2, 0.5, 0.3])
        
        # Stress susceptibility
        stress_susceptibility = np.random.beta(2, 3, self.n_patients)
        
        # Medication response (how well they respond to treatment)
        med_response = np.random.beta(5, 2, self.n_patients)  # Most respond well
        
        profiles = pd.DataFrame({
            'patient_id': range(self.n_patients),
            'gender': genders,
            'age': ages,
            'hla_b27': hla_b27,
            'disease_duration': disease_duration,
            'bmi': bmis,
            'smoking': smoking,
            'base_severity': base_severity,
            'weather_sensitivity': weather_sensitivity,
            'stress_susceptibility': stress_susceptibility,
            'med_response': med_response
        })
        
        print(f"✅ Generated {len(profiles)} patient profiles")
        return profiles

    def generate_time_series_data(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive time-series features for all patients"""
        print("\n⏱️  Generating time-series data (this will take a few minutes)...")
        start_time = time.time()
        
        all_patient_data = []
        
        for patient_idx in range(self.n_patients):
            if patient_idx % 50 == 0:
                elapsed = time.time() - start_time
                if patient_idx > 0:
                    eta = (elapsed / patient_idx) * (self.n_patients - patient_idx)
                    print(f"  Progress: {patient_idx}/{self.n_patients} patients ({patient_idx/self.n_patients*100:.1f}%) - ETA: {eta/60:.1f}min", end='\r')
            
            profile = profiles.iloc[patient_idx]
            patient_data = self._generate_patient_timeseries(profile)
            all_patient_data.append(patient_data)
        
        print(f"\n  Combining all patient data...")
        full_dataset = pd.concat(all_patient_data, ignore_index=True)
        
        elapsed = time.time() - start_time
        print(f"✅ Generated {len(full_dataset):,} rows in {elapsed/60:.2f} minutes")
        print(f"   Features: {len(full_dataset.columns)}")
        
        return full_dataset

    def _generate_patient_timeseries(self, profile: pd.Series) -> pd.DataFrame:
        """Generate time-series data for a single patient"""
        n_days = self.days_per_patient
        
        # Extract profile characteristics
        base_severity = profile['base_severity']
        weather_sens = profile['weather_sensitivity']
        stress_susc = profile['stress_susceptibility']
        med_response = profile['med_response']
        
        # ===== DISEASE ACTIVITY CORE =====
        # Base BASDAI with individual variation
        base_basdai = base_severity * 0.6 + np.random.random() * 2
        
        # Daily BASDAI with natural variation
        daily_basdai = np.random.normal(base_basdai, 0.8, n_days).clip(0, 10)
        
        # Add flare patterns (4-8 flares per year for better ML balance)
        flare_frequency = 4 + np.random.random() * 4
        n_flares = int(flare_frequency * (n_days / 365))
        
        for _ in range(n_flares):
            flare_start = np.random.randint(0, n_days - 21)
            
            # Flare build-up (3-5 days)
            buildup_days = np.random.randint(3, 6)
            # Flare peak (2-4 days)
            peak_days = np.random.randint(2, 5)
            # Flare recovery (5-10 days)
            recovery_days = np.random.randint(5, 11)
            
            flare_duration = buildup_days + peak_days + recovery_days
            flare_end = min(flare_start + flare_duration, n_days)
            
            # Build-up phase
            for i in range(buildup_days):
                if flare_start + i < n_days:
                    daily_basdai[flare_start + i] *= (1.2 + i * 0.1)
            
            # Peak phase
            peak_intensity = 1.8 + np.random.random() * 0.7
            for i in range(peak_days):
                if flare_start + buildup_days + i < n_days:
                    daily_basdai[flare_start + buildup_days + i] *= peak_intensity
            
            # Recovery phase (gradual decrease)
            for i in range(recovery_days):
                if flare_start + buildup_days + peak_days + i < n_days:
                    recovery_factor = 1.5 - (i / recovery_days) * 0.5
                    daily_basdai[flare_start + buildup_days + peak_days + i] *= recovery_factor
        
        daily_basdai = np.clip(daily_basdai, 0, 10)
        
        # ===== WEATHER FEATURES =====
        # Barometric pressure (realistic patterns)
        base_pressure = 1013.25
        pressure = base_pressure + np.random.normal(0, 8, n_days)
        pressure = pd.Series(pressure).rolling(5, min_periods=1, center=True).mean().values
        
        # Pressure change (6-hour rate)
        pressure_change = np.diff(pressure, prepend=pressure[0])
        pressure_change = pd.Series(pressure_change).rolling(2, min_periods=1).mean().values
        
        # Rapid pressure drops trigger flares (with individual sensitivity)
        pressure_trigger = (pressure_change < -4.0).astype(float) * weather_sens
        daily_basdai += pressure_trigger * np.random.uniform(0.5, 1.5, n_days)
        daily_basdai = np.clip(daily_basdai, 0, 10)
        
        # Temperature with seasonal variation
        day_of_year = np.arange(n_days) % 365
        seasonal_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temperature = seasonal_temp + np.random.normal(0, 4, n_days)
        
        # Humidity
        humidity = 60 + np.random.normal(0, 18, n_days)
        humidity = np.clip(humidity, 15, 100)
        
        # Air quality (1-10 scale, 1=excellent, 10=hazardous)
        air_quality = np.random.gamma(2, 1.5, n_days).clip(1, 10)
        
        # Weather change score (7-day rolling analysis)
        weather_change_score = np.zeros(n_days)
        for i in range(7, n_days):
            temp_change = abs(temperature[i] - temperature[i-7])
            pressure_delta = abs(pressure[i] - pressure[i-7])
            humidity_delta = abs(humidity[i] - humidity[i-7])
            weather_change_score[i] = (temp_change / 10 + pressure_delta / 20 + humidity_delta / 30).clip(0, 10)
        
        # ===== VITAL SIGNS & CARDIO =====
        # Blood oxygen (SpO2) - decreases slightly with inflammation
        blood_oxygen = 98 - daily_basdai * 0.3 + np.random.normal(0, 0.8, n_days)
        blood_oxygen = np.clip(blood_oxygen, 92, 100)
        
        # Cardio fitness (VO2 max estimate) - inversely related to disease activity
        base_vo2max = 40 - profile['age'] * 0.2 + profile['gender'] * 8
        cardio_fitness = base_vo2max - daily_basdai * 1.5 + np.random.normal(0, 2, n_days)
        cardio_fitness = np.clip(cardio_fitness, 20, 60)
        
        # Respiratory rate (breaths/min)
        respiratory_rate = 14 + daily_basdai * 0.4 + np.random.normal(0, 1.5, n_days)
        respiratory_rate = np.clip(respiratory_rate, 10, 25)
        
        # 6-minute walk test distance (meters) - done weekly
        walk_test_base = 500 - daily_basdai * 30
        walk_test_distance = walk_test_base + np.random.normal(0, 40, n_days)
        walk_test_distance = np.clip(walk_test_distance, 200, 650)
        
        # Resting energy expenditure (kcal/day)
        resting_energy = 1400 + profile['bmi'] * 20 + profile['gender'] * 200
        resting_energy = resting_energy + np.random.normal(0, 100, n_days)
        
        # Heart rate variability (HRV) - decreases with inflammation
        hrv = 55 - daily_basdai * 3.5 + np.random.normal(0, 6, n_days)
        hrv = np.clip(hrv, 15, 90)
        
        # Resting heart rate - increases with inflammation
        resting_hr = 62 + daily_basdai * 2.2 + np.random.normal(0, 4, n_days)
        resting_hr = np.clip(resting_hr, 45, 95)
        
        # Walking average heart rate
        walking_hr = resting_hr + 20 + np.random.normal(0, 5, n_days)
        walking_hr = np.clip(walking_hr, 70, 130)
        
        # Cardio recovery (HR drop after 1 min) - worse with disease activity
        cardio_recovery = 25 - daily_basdai * 1.5 + np.random.normal(0, 3, n_days)
        cardio_recovery = np.clip(cardio_recovery, 10, 40)
        
        # ===== PHYSICAL ACTIVITY =====
        # Steps - dramatically reduced during flares
        base_steps = 8000 - profile['age'] * 50
        steps = base_steps - daily_basdai * 800 + np.random.normal(0, 1200, n_days)
        steps = np.clip(steps, 500, 18000).astype(int)
        
        # Distance (km)
        distance = steps * 0.00075  # ~0.75m per step
        
        # Stairs ascended
        stairs_up = (steps / 200) - daily_basdai * 2 + np.random.normal(0, 5, n_days)
        stairs_up = np.clip(stairs_up, 0, 60).astype(int)
        
        # Stairs descended
        stairs_down = stairs_up + np.random.randint(-5, 6, n_days)
        stairs_down = np.clip(stairs_down, 0, 65).astype(int)
        
        # Stand hours (out of 16 waking hours)
        stand_hours = 12 - daily_basdai * 0.8 + np.random.normal(0, 1.5, n_days)
        stand_hours = np.clip(stand_hours, 2, 16).astype(int)
        
        # Stand minutes
        stand_minutes = stand_hours * 60 + np.random.randint(-30, 31, n_days)
        stand_minutes = np.clip(stand_minutes, 60, 960)
        
        # Training/exercise minutes
        training_minutes = 30 - daily_basdai * 3 + np.random.normal(0, 15, n_days)
        training_minutes = np.clip(training_minutes, 0, 120).astype(int)
        
        # Active minutes (broader than training)
        active_minutes = training_minutes + 20 + np.random.randint(-10, 21, n_days)
        active_minutes = np.clip(active_minutes, 0, 180)
        
        # Active energy (kcal)
        active_energy = steps * 0.04 + training_minutes * 5
        active_energy = active_energy + np.random.normal(0, 50, n_days)
        active_energy = np.clip(active_energy, 50, 800)
        
        # Training sessions count (weekly average)
        training_sessions = (training_minutes > 20).astype(int)
        
        # Gait metrics
        walking_tempo = 110 - daily_basdai * 3 + np.random.normal(0, 8, n_days)  # steps/min
        walking_tempo = np.clip(walking_tempo, 70, 130)
        
        step_length = 0.7 - daily_basdai * 0.02 + np.random.normal(0, 0.05, n_days)  # meters
        step_length = np.clip(step_length, 0.4, 0.9)
        
        gait_asymmetry = daily_basdai * 2 + np.random.normal(0, 3, n_days)  # percentage
        gait_asymmetry = np.clip(gait_asymmetry, 0, 30)
        
        bipedal_support = 40 - daily_basdai * 0.5 + np.random.normal(0, 2, n_days)  # percentage
        bipedal_support = np.clip(bipedal_support, 30, 45)
        
        # ===== SLEEP METRICS =====
        # Total sleep hours
        sleep_hours = 7.5 - daily_basdai * 0.35 + np.random.normal(0, 0.8, n_days)
        sleep_hours = np.clip(sleep_hours, 3.5, 10)
        
        # Sleep phases (as percentage of total sleep)
        rem_percentage = 20 - daily_basdai * 0.8 + np.random.normal(0, 3, n_days)
        rem_percentage = np.clip(rem_percentage, 10, 30)
        rem_duration = sleep_hours * (rem_percentage / 100)
        
        deep_percentage = 15 - daily_basdai * 0.6 + np.random.normal(0, 2.5, n_days)
        deep_percentage = np.clip(deep_percentage, 5, 25)
        deep_duration = sleep_hours * (deep_percentage / 100)
        
        core_percentage = 50 + np.random.normal(0, 5, n_days)
        core_percentage = np.clip(core_percentage, 40, 60)
        core_duration = sleep_hours * (core_percentage / 100)
        
        awake_duration = sleep_hours - rem_duration - deep_duration - core_duration
        awake_duration = np.clip(awake_duration, 0, 2)
        
        # Sleep score (1-100)
        sleep_score = 85 - daily_basdai * 5 + np.random.normal(0, 8, n_days)
        sleep_score = np.clip(sleep_score, 30, 100)
        
        # Sleep schedule consistency (lower is better, 0-100)
        sleep_consistency = 20 + daily_basdai * 3 + np.random.normal(0, 10, n_days)
        sleep_consistency = np.clip(sleep_consistency, 0, 100)
        
        # ===== TRAINING APP DATA =====
        burned_calories = active_energy + resting_energy / 24  # Daily portion
        
        exertion_level = training_minutes / 10 + daily_basdai * 0.5  # 1-10 scale
        exertion_level = np.clip(exertion_level, 0, 10)
        
        # ===== MENTAL WELLBEING =====
        # Mood score (1-10, inversely related to disease activity)
        mood_current = 7 - daily_basdai * 0.5 + np.random.normal(0, 1, n_days)
        mood_current = np.clip(mood_current, 1, 10)
        
        # Mood valence (-5 to +5)
        mood_valence = 2 - daily_basdai * 0.4 + np.random.normal(0, 1.5, n_days)
        mood_valence = np.clip(mood_valence, -5, 5)
        
        # Mood stability (0-10, higher is more stable)
        mood_stability = 7 - daily_basdai * 0.3 + np.random.normal(0, 1.2, n_days)
        mood_stability = np.clip(mood_stability, 2, 10)
        
        # Anxiety level (0-10)
        anxiety = 3 + daily_basdai * 0.4 + np.random.normal(0, 1.5, n_days)
        anxiety = np.clip(anxiety, 0, 10)
        
        # Stress level (0-10)
        base_stress = 4 + stress_susc * 3
        stress_level = base_stress + daily_basdai * 0.3 + np.random.normal(0, 1.8, n_days)
        stress_level = np.clip(stress_level, 0, 10)
        
        # Stress also feeds back into disease activity
        daily_basdai += stress_level * 0.1 * stress_susc
        daily_basdai = np.clip(daily_basdai, 0, 10)
        
        # Stress resilience (0-10, higher is better)
        stress_resilience = 6 - stress_susc * 3 + np.random.normal(0, 1, n_days)
        stress_resilience = np.clip(stress_resilience, 1, 10)
        
        # Mental fatigue (0-10)
        mental_fatigue = daily_basdai * 0.6 + stress_level * 0.3 + np.random.normal(0, 1, n_days)
        mental_fatigue = np.clip(mental_fatigue, 0, 10)
        
        # Cognitive function (0-10, higher is better)
        cognitive_function = 8 - mental_fatigue * 0.5 + np.random.normal(0, 0.8, n_days)
        cognitive_function = np.clip(cognitive_function, 2, 10)
        
        # Emotional regulation (0-10, higher is better)
        emotional_regulation = 7 - anxiety * 0.3 - daily_basdai * 0.2 + np.random.normal(0, 1, n_days)
        emotional_regulation = np.clip(emotional_regulation, 2, 10)
        
        # Social engagement (0-10)
        social_engagement = 6 - daily_basdai * 0.4 - anxiety * 0.2 + np.random.normal(0, 1.2, n_days)
        social_engagement = np.clip(social_engagement, 0, 10)
        
        # Mental wellbeing composite (0-100)
        mental_wellbeing = (mood_current + mood_stability + (10 - anxiety) + stress_resilience + 
                           cognitive_function + emotional_regulation + social_engagement) / 7 * 10
        mental_wellbeing = np.clip(mental_wellbeing, 20, 100)
        
        # Depression risk (0-10, higher is worse)
        depression_risk = anxiety * 0.5 + (10 - mood_current) * 0.4 + daily_basdai * 0.2
        depression_risk = np.clip(depression_risk, 0, 10)
        
        # Time in daylight (minutes)
        daylight_time = 120 - daily_basdai * 8 + np.random.normal(0, 30, n_days)
        daylight_time = np.clip(daylight_time, 10, 300)
        
        # ===== DISEASE-SPECIFIC ASSESSMENTS =====
        # ASDAS-CRP (usually 1.3-5.0)
        asdas_crp = daily_basdai * 0.5 + 1.0 + np.random.normal(0, 0.3, n_days)
        asdas_crp = np.clip(asdas_crp, 0.5, 5.5)
        
        # BASFI functional index (0-10)
        basfi = daily_basdai * 0.7 + np.random.normal(0, 0.8, n_days)
        basfi = np.clip(basfi, 0, 10)
        
        # BASMI mobility index (0-10, higher is worse)
        basmi = base_severity * 0.5 + daily_basdai * 0.2 + np.random.normal(0, 0.5, n_days)
        basmi = np.clip(basmi, 0, 10)
        
        # Patient global assessment (0-10)
        patient_global = daily_basdai * 0.8 + np.random.normal(0, 0.7, n_days)
        patient_global = np.clip(patient_global, 0, 10)
        
        # Physician global assessment (0-10, slightly more objective)
        physician_global = daily_basdai * 0.75 + np.random.normal(0, 0.5, n_days)
        physician_global = np.clip(physician_global, 0, 10)
        
        # Joint counts
        tender_joint_count = (daily_basdai * 2).astype(int) + np.random.randint(-2, 3, n_days)
        tender_joint_count = np.clip(tender_joint_count, 0, 20)
        
        swollen_joint_count = (daily_basdai * 1.5).astype(int) + np.random.randint(-1, 2, n_days)
        swollen_joint_count = np.clip(swollen_joint_count, 0, 15)
        
        # Enthesitis score (0-10)
        enthesitis = daily_basdai * 0.6 + np.random.normal(0, 0.8, n_days)
        enthesitis = np.clip(enthesitis, 0, 10)
        
        # Dactylitis score (0-10, less common)
        dactylitis = daily_basdai * 0.3 + np.random.normal(0, 0.5, n_days)
        dactylitis = np.clip(dactylitis, 0, 10)
        
        # Spinal mobility score (0-10, higher is better)
        spinal_mobility = 8 - basmi * 0.7 + np.random.normal(0, 0.6, n_days)
        spinal_mobility = np.clip(spinal_mobility, 1, 10)
        
        # Disease activity composite (0-100)
        disease_activity = (daily_basdai + asdas_crp + basfi + patient_global) / 4 * 10
        disease_activity = np.clip(disease_activity, 0, 100)
        
        # ===== PAIN CHARACTERIZATION =====
        # Pain intensity current (0-10)
        pain_current = daily_basdai * 0.9 + np.random.normal(0, 0.8, n_days)
        pain_current = np.clip(pain_current, 0, 10)
        
        # Pain intensity 24h average
        pain_avg_24h = pd.Series(pain_current).rolling(24, min_periods=1).mean().values
        
        # Pain intensity 24h max
        pain_max_24h = pd.Series(pain_current).rolling(24, min_periods=1).max().values
        
        # Nocturnal pain (worse in AS)
        nocturnal_pain = pain_current * 1.2 + np.random.normal(0, 0.5, n_days)
        nocturnal_pain = np.clip(nocturnal_pain, 0, 10)
        
        # Morning stiffness duration (minutes)
        morning_stiffness_duration = 30 + daily_basdai * 12 + np.random.normal(0, 15, n_days)
        morning_stiffness_duration = np.clip(morning_stiffness_duration, 0, 240).astype(int)
        
        # Morning stiffness severity (0-10)
        morning_stiffness_severity = daily_basdai * 0.85 + np.random.normal(0, 0.7, n_days)
        morning_stiffness_severity = np.clip(morning_stiffness_severity, 0, 10)
        
        # Pain location count (number of painful areas)
        pain_location_count = (daily_basdai * 1.5).astype(int) + np.random.randint(-1, 3, n_days)
        pain_location_count = np.clip(pain_location_count, 0, 15)
        
        # Pain quality descriptors (0-10 each)
        pain_burning = pain_current * 0.4 + np.random.normal(0, 1, n_days)
        pain_burning = np.clip(pain_burning, 0, 10)
        
        pain_aching = pain_current * 0.7 + np.random.normal(0, 0.8, n_days)
        pain_aching = np.clip(pain_aching, 0, 10)
        
        pain_sharp = pain_current * 0.5 + np.random.normal(0, 1, n_days)
        pain_sharp = np.clip(pain_sharp, 0, 10)
        
        # Pain interference
        pain_interference_sleep = nocturnal_pain * 0.8 + np.random.normal(0, 0.6, n_days)
        pain_interference_sleep = np.clip(pain_interference_sleep, 0, 10)
        
        pain_interference_activity = pain_current * 0.75 + np.random.normal(0, 0.7, n_days)
        pain_interference_activity = np.clip(pain_interference_activity, 0, 10)
        
        # Pain variability coefficient (0-1, higher = more variable)
        pain_variability = np.zeros(n_days)
        for i in range(7, n_days):
            pain_variability[i] = np.std(pain_current[i-7:i]) / (np.mean(pain_current[i-7:i]) + 0.1)
        pain_variability = np.clip(pain_variability, 0, 1)
        
        # Breakthrough pain episodes (count per day)
        breakthrough_pain = ((pain_current > 7) & (pain_current > pain_avg_24h + 2)).astype(int)
        
        # ===== APP-SPECIFIC FEATURES =====
        # Medication adherence (0-1, affected by side effects and efficacy)
        med_adherence = med_response * 0.8 + 0.2 + np.random.normal(0, 0.1, n_days)
        med_adherence = np.clip(med_adherence, 0, 1)
        
        # Medication effect (reduces disease activity over time)
        med_effect = med_adherence * med_response * 3
        daily_basdai = daily_basdai - med_effect
        daily_basdai = np.clip(daily_basdai, 0, 10)
        
        # Physiotherapy adherence (0-1)
        physio_adherence = 0.6 + np.random.normal(0, 0.2, n_days)
        physio_adherence = np.clip(physio_adherence, 0, 1)
        
        # Physiotherapy effectiveness score (1-10)
        physio_effectiveness = 6 + physio_adherence * 3 + np.random.normal(0, 1, n_days)
        physio_effectiveness = np.clip(physio_effectiveness, 1, 10)
        
        # Journal mood (1-10, similar to mood_current but user-reported)
        journal_mood = mood_current + np.random.normal(0, 0.5, n_days)
        journal_mood = np.clip(journal_mood, 1, 10)
        
        # Quick log entries (binary, did user log today)
        quick_log = (np.random.random(n_days) > 0.6).astype(int)
        
        # Universal assessment score (normalized 1-10 from various assessments)
        universal_assessment = (daily_basdai + patient_global + basfi) / 3
        universal_assessment = np.clip(universal_assessment, 0, 10)
        
        # Time-weighted assessment (considers trend)
        time_weighted_assessment = np.zeros(n_days)
        for i in range(7, n_days):
            recent_trend = np.mean(universal_assessment[i-7:i])
            time_weighted_assessment[i] = universal_assessment[i] * 0.7 + recent_trend * 0.3
        time_weighted_assessment = np.clip(time_weighted_assessment, 0, 10)
        
        # ===== ENVIRONMENTAL =====
        # Ambient noise level (dB)
        ambient_noise = 55 + np.random.normal(0, 10, n_days)
        ambient_noise = np.clip(ambient_noise, 30, 90)
        
        # Season factor (0-3: winter, spring, summer, fall)
        season = ((day_of_year // 91) % 4).astype(int)
        
        # ===== CREATE TARGET VARIABLE =====
        # Will flare in next 3-7 days (binary)
        # Adjusted threshold for better class balance (20-30% positive class)
        will_flare_3_7d = np.zeros(n_days)
        for i in range(n_days - 7):
            # Check if any day in the next 3-7 days has elevated disease activity
            # Threshold 4.0 targets 20-30% positive class for optimal ML training
            if np.any(daily_basdai[i+3:i+8] > 4.0):
                will_flare_3_7d[i] = 1
        
        # ===== ASSEMBLE DATAFRAME =====
        patient_df = pd.DataFrame({
            # Identifiers
            'patient_id': profile['patient_id'],
            'day_index': range(n_days),
            
            # Demographics (static per patient)
            'age': profile['age'],
            'gender': profile['gender'],
            'hla_b27': profile['hla_b27'],
            'disease_duration': profile['disease_duration'],
            'bmi': profile['bmi'],
            'smoking': profile['smoking'],
            
            # Disease Activity
            'basdai_score': daily_basdai,
            'asdas_crp': asdas_crp,
            'basfi': basfi,
            'basmi': basmi,
            'patient_global': patient_global,
            'physician_global': physician_global,
            'tender_joint_count': tender_joint_count,
            'swollen_joint_count': swollen_joint_count,
            'enthesitis': enthesitis,
            'dactylitis': dactylitis,
            'spinal_mobility': spinal_mobility,
            'disease_activity_composite': disease_activity,
            
            # Pain
            'pain_current': pain_current,
            'pain_avg_24h': pain_avg_24h,
            'pain_max_24h': pain_max_24h,
            'nocturnal_pain': nocturnal_pain,
            'morning_stiffness_duration': morning_stiffness_duration,
            'morning_stiffness_severity': morning_stiffness_severity,
            'pain_location_count': pain_location_count,
            'pain_burning': pain_burning,
            'pain_aching': pain_aching,
            'pain_sharp': pain_sharp,
            'pain_interference_sleep': pain_interference_sleep,
            'pain_interference_activity': pain_interference_activity,
            'pain_variability': pain_variability,
            'breakthrough_pain': breakthrough_pain,
            
            # Vital Signs & Cardio
            'blood_oxygen': blood_oxygen,
            'cardio_fitness': cardio_fitness,
            'respiratory_rate': respiratory_rate,
            'walk_test_distance': walk_test_distance,
            'resting_energy': resting_energy,
            'hrv': hrv,
            'resting_hr': resting_hr,
            'walking_hr': walking_hr,
            'cardio_recovery': cardio_recovery,
            
            # Physical Activity
            'steps': steps,
            'distance_km': distance,
            'stairs_up': stairs_up,
            'stairs_down': stairs_down,
            'stand_hours': stand_hours,
            'stand_minutes': stand_minutes,
            'training_minutes': training_minutes,
            'active_minutes': active_minutes,
            'active_energy': active_energy,
            'training_sessions': training_sessions,
            'walking_tempo': walking_tempo,
            'step_length': step_length,
            'gait_asymmetry': gait_asymmetry,
            'bipedal_support': bipedal_support,
            
            # Sleep
            'sleep_hours': sleep_hours,
            'rem_duration': rem_duration,
            'deep_duration': deep_duration,
            'core_duration': core_duration,
            'awake_duration': awake_duration,
            'sleep_score': sleep_score,
            'sleep_consistency': sleep_consistency,
            
            # Training App
            'burned_calories': burned_calories,
            'exertion_level': exertion_level,
            
            # Mental Wellbeing
            'mood_current': mood_current,
            'mood_valence': mood_valence,
            'mood_stability': mood_stability,
            'anxiety': anxiety,
            'stress_level': stress_level,
            'stress_resilience': stress_resilience,
            'mental_fatigue': mental_fatigue,
            'cognitive_function': cognitive_function,
            'emotional_regulation': emotional_regulation,
            'social_engagement': social_engagement,
            'mental_wellbeing': mental_wellbeing,
            'depression_risk': depression_risk,
            'daylight_time': daylight_time,
            
            # Weather
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'pressure_change': pressure_change,
            'air_quality': air_quality,
            'weather_change_score': weather_change_score,
            
            # App-Specific
            'med_adherence': med_adherence,
            'physio_adherence': physio_adherence,
            'physio_effectiveness': physio_effectiveness,
            'journal_mood': journal_mood,
            'quick_log': quick_log,
            'universal_assessment': universal_assessment,
            'time_weighted_assessment': time_weighted_assessment,
            
            # Environmental
            'ambient_noise': ambient_noise,
            'season': season,
            
            # Target
            'will_flare_3_7d': will_flare_3_7d
        })
        
        return patient_df

    def save_dataset(self, df: pd.DataFrame, output_dir: str = 'data'):
        """Save dataset in multiple formats with metadata"""
        print(f"\n💾 Saving dataset...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as Parquet (efficient for ML)
        parquet_file = output_path / 'comprehensive_training_data.parquet'
        df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False)
        
        # Save as CSV (human-readable, but large)
        csv_file = output_path / 'comprehensive_training_data.csv'
        df.to_csv(csv_file, index=False)
        
        # Save sample
        sample_file = output_path / 'comprehensive_training_data_sample.csv'
        df.head(1000).to_csv(sample_file, index=False)
        
        # Calculate file sizes
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)
        csv_size = csv_file.stat().st_size / (1024 * 1024)
        
        print(f"✅ Saved Parquet: {parquet_file} ({parquet_size:.1f} MB)")
        print(f"✅ Saved CSV: {csv_file} ({csv_size:.1f} MB)")
        print(f"✅ Saved Sample: {sample_file}")
        
        # Save metadata
        metadata = {
            'n_rows': len(df),
            'n_patients': df['patient_id'].nunique(),
            'n_features': len(df.columns) - 3,  # Exclude patient_id, day_index, target
            'feature_names': [col for col in df.columns if col not in ['patient_id', 'day_index', 'will_flare_3_7d']],
            'target_column': 'will_flare_3_7d',
            'class_balance': {
                'no_flare': int((df['will_flare_3_7d'] == 0).sum()),
                'will_flare': int((df['will_flare_3_7d'] == 1).sum()),
                'flare_percentage': float(df['will_flare_3_7d'].mean() * 100)
            },
            'generated_at': datetime.now().isoformat(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        metadata_file = output_path / 'comprehensive_training_data_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved Metadata: {metadata_file}")
        print(f"\n📊 Dataset Statistics:")
        print(f"   Rows: {metadata['n_rows']:,}")
        print(f"   Patients: {metadata['n_patients']}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Flare rate: {metadata['class_balance']['flare_percentage']:.1f}%")
        
        return metadata

    def validate_data(self, df: pd.DataFrame):
        """Validate data quality and correlations"""
        print(f"\n🔍 Validating data quality...")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  Warning: Found {nan_counts.sum()} NaN values")
            print(nan_counts[nan_counts > 0])
        else:
            print("✅ No NaN values found")
        
        # Check key correlations
        print(f"\n📈 Key Medical Correlations:")
        correlations = {
            'BASDAI vs Pain': df['basdai_score'].corr(df['pain_current']),
            'BASDAI vs HRV': df['basdai_score'].corr(df['hrv']),
            'BASDAI vs Steps': df['basdai_score'].corr(df['steps']),
            'BASDAI vs Sleep': df['basdai_score'].corr(df['sleep_score']),
            'Pressure Change vs BASDAI': df['pressure_change'].corr(df['basdai_score']),
            'Stress vs Disease Activity': df['stress_level'].corr(df['disease_activity_composite']),
            'Morning Stiffness vs BASDAI': df['morning_stiffness_severity'].corr(df['basdai_score']),
        }
        
        for name, corr in correlations.items():
            status = "✅" if abs(corr) > 0.15 else "⚠️"
            print(f"   {status} {name}: {corr:.3f}")
        
        # Check target distribution
        flare_rate = df['will_flare_3_7d'].mean()
        print(f"\n🎯 Target Distribution:")
        print(f"   Flare rate: {flare_rate*100:.1f}%")
        if 0.20 <= flare_rate <= 0.40:
            print(f"   ✅ Good balance for ML training")
        else:
            print(f"   ⚠️  May need rebalancing")
        
        return correlations

    def run(self):
        """Execute full data generation pipeline"""
        print("\n" + "="*70)
        print("COMPREHENSIVE AS DATA GENERATION PIPELINE")
        print("="*70)
        
        overall_start = time.time()
        
        # Step 1: Generate patient profiles
        profiles = self.generate_patient_profiles()
        
        # Step 2: Generate time-series data
        full_dataset = self.generate_time_series_data(profiles)
        
        # Step 3: Validate data
        correlations = self.validate_data(full_dataset)
        
        # Step 4: Save dataset
        metadata = self.save_dataset(full_dataset)
        
        elapsed = time.time() - overall_start
        print("\n" + "="*70)
        print(f"✅ DATA GENERATION COMPLETE")
        print(f"   Total time: {elapsed/60:.2f} minutes")
        print(f"   Rows/second: {len(full_dataset) / elapsed:,.0f}")
        print("="*70)
        
        return full_dataset, metadata

def main():
    """Main entry point"""
    # Configuration
    N_PATIENTS = 1500
    DAYS_PER_PATIENT = 1400  # ~3.8 years per patient
    
    # Initialize generator
    generator = ComprehensiveASDataGenerator(
        n_patients=N_PATIENTS,
        days_per_patient=DAYS_PER_PATIENT
    )
    
    # Run generation
    dataset, metadata = generator.run()
    
    print(f"\n🔥 Ready for model training!")
    print(f"   Load data with: pd.read_parquet('data/comprehensive_training_data.parquet')")

if __name__ == "__main__":
    main()
