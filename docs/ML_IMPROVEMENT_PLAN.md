# ML Model Improvement Plan - From 57% to 85%+ Accuracy

## Executive Summary

**Current State**: Model outputs constant ~57% probability regardless of input (collapsed weights)
**Root Cause**: Training failed on synthetic data, scaler mismatch, bidirectional LSTM data leakage
**Target**: 80-85% accuracy with personalization, 90%+ after 90 days user data

---

## Phase 1: IMMEDIATE FIXES (Stop the Bleeding) - 1-2 Days

### Fix 1.1: Disable Broken CoreML Model, Use Statistical Fallback

The CoreML model is fundamentally broken. Use the statistical approach that already exists.

**File**: `InflamAI/Core/ML/UnifiedNeuralEngine.swift`

```swift
// Line ~290 - Add validation before using CoreML output
private func validateModelOutput(_ probability: Float) -> Bool {
    // If model outputs near-constant value, it's broken
    // 0.57 ± 0.05 is the collapse signature
    return abs(probability - 0.57) > 0.10
}

// In predict() method, add fallback:
let prediction = try await runPrediction(inputArray: inputArray)

// VALIDATE: Detect collapsed model
if !validateModelOutput(prediction.flareProb) {
    print("⚠️ [ML] Model output suspicious (collapsed) - using statistical fallback")
    return await calculateStatisticalFlareRisk()
}
```

### Fix 1.2: Fix Scaler Loading - FAIL FAST, Don't Fallback

**File**: `InflamAI/Core/ML/UnifiedNeuralEngine.swift` (lines 411-441)

```swift
private func loadFeatureScaler() throws {
    // ONLY use MinMax params - the model was trained with MinMax
    guard let url = Bundle.main.url(forResource: "minmax_params", withExtension: "json"),
          let data = try? Data(contentsOf: url),
          let params = try? JSONDecoder().decode(MinMaxScalerParams.self, from: data) else {
        // FAIL FAST - don't silently use wrong scaler
        throw MLError.scalerMissing("minmax_params.json not found - cannot run predictions")
    }

    featureScaler = UnifiedFeatureScaler(mins: params.mins, maxs: params.maxs)
    print("✅ [ML] Loaded MinMax scaler with \(params.mins.count) features")
}
```

### Fix 1.3: Increase Data Quality Threshold

**File**: `InflamAI/Core/ML/UnifiedNeuralEngine.swift` (line 176)

```swift
// OLD: 15% (way too low - accepts mostly-zero vectors)
private let minimumDataQualityScore: Float = 0.15

// NEW: 40% minimum (require meaningful data)
private let minimumDataQualityScore: Float = 0.40
private let minimumNonZeroFeatures: Int = 37  // 40% of 92
```

### Fix 1.4: Add Output Distribution Monitoring

**File**: `InflamAI/Core/ML/UnifiedNeuralEngine.swift`

```swift
// Track last 10 predictions to detect collapse
private var recentPredictions: [Float] = []
private let maxRecentPredictions = 10

private func trackPrediction(_ probability: Float) {
    recentPredictions.append(probability)
    if recentPredictions.count > maxRecentPredictions {
        recentPredictions.removeFirst()
    }

    // Check for collapse (all predictions within 0.1 of each other)
    if recentPredictions.count >= 5 {
        let variance = calculateVariance(recentPredictions)
        if variance < 0.01 {
            print("🚨 [ML] WARNING: Model predictions collapsed (variance=\(variance))")
        }
    }
}
```

---

## Phase 2: MODEL ARCHITECTURE FIXES - 3-5 Days

### Fix 2.1: Change to Unidirectional LSTM (Remove Data Leakage)

**Problem**: Bidirectional LSTM sees "future" data when predicting
**Solution**: Use forward-only LSTM

**File**: Python training script (neural_flare_net.py)

```python
# OLD (WRONG - sees future):
self.lstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=3,
    bidirectional=True,  # ❌ REMOVE THIS
    batch_first=True
)

# NEW (CORRECT - causal):
self.lstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=3,
    bidirectional=False,  # ✅ Forward only
    batch_first=True,
    dropout=0.3
)
```

### Fix 2.2: Reduce Feature Count (92 → 35-40)

**Problem**: 92 features is too many for limited real data
**Solution**: Keep only features you can actually collect

**Essential Features (35 total)**:
```
Demographics (5): age, gender, bmi, disease_duration, hla_b27
Clinical (6): basdai_score, basfi, current_pain, morning_stiffness, fatigue, asdas_crp
Pain (6): pain_avg, pain_max, affected_joints_count, si_joint_pain, spine_pain, peripheral_pain
Activity (8): steps, active_minutes, exercise_minutes, flights_climbed, stand_hours, hrv, resting_hr, sleep_hours
Mental (4): stress_level, mood, anxiety, depression_score
Weather (4): temperature, humidity, pressure, pressure_change_12h
Time (2): day_of_week, season
```

### Fix 2.3: Fix Train/Test Split (Patient-Level, Not Random)

**Problem**: Same patient in train AND test sets = data leakage
**Solution**: Split by patient ID first

```python
# OLD (WRONG):
indices = np.random.permutation(total_size)
train_idx = indices[:train_size]

# NEW (CORRECT):
unique_patients = df['patient_id'].unique()
np.random.shuffle(unique_patients)

train_patients = unique_patients[:int(0.7 * len(unique_patients))]
val_patients = unique_patients[int(0.7 * len(unique_patients)):int(0.85 * len(unique_patients))]
test_patients = unique_patients[int(0.85 * len(unique_patients)):]

train_df = df[df['patient_id'].isin(train_patients)]
val_df = df[df['patient_id'].isin(val_patients)]
test_df = df[df['patient_id'].isin(test_patients)]
```

---

## Phase 3: TRAINING DATA OVERHAUL - 1-2 Weeks

### Fix 3.1: Class Imbalance Handling

**Problem**: 81% no-flare, 19% flare → model learns to always predict "no flare"

```python
# Option A: Class weights in loss function
class_weights = torch.tensor([0.19, 0.81])  # Inverse of class frequency
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option B: Focal Loss (better for rare events)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Option C: SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5)  # 50% minority class
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Fix 3.2: Generate Better Synthetic Data (If No Real Data Available)

**Current Problem**: Synthetic flares are too regular and predictable

```python
# BETTER synthetic data generation:

def generate_realistic_flare(baseline_basdai, duration_days):
    """Generate more realistic flare pattern"""

    # Realistic flare characteristics (from medical literature):
    # - Onset: 1-3 days to peak
    # - Duration: 3-21 days (highly variable)
    # - Resolution: gradual, 2-7 days
    # - Intensity: varies by trigger

    onset_days = np.random.randint(1, 4)
    peak_duration = np.random.randint(1, 7)
    resolution_days = np.random.randint(2, 8)

    total_days = onset_days + peak_duration + resolution_days

    # Peak intensity varies
    peak_increase = np.random.uniform(1.5, 4.0)

    flare_curve = np.zeros(total_days)

    # Onset phase (exponential rise)
    for i in range(onset_days):
        flare_curve[i] = peak_increase * (1 - np.exp(-3 * i / onset_days))

    # Peak phase (with daily variation)
    for i in range(onset_days, onset_days + peak_duration):
        flare_curve[i] = peak_increase * np.random.uniform(0.8, 1.0)

    # Resolution phase (slower decay)
    for i in range(onset_days + peak_duration, total_days):
        days_into_resolution = i - onset_days - peak_duration
        flare_curve[i] = peak_increase * np.exp(-1.5 * days_into_resolution / resolution_days)

    return flare_curve
```

### Fix 3.3: Add Realistic Trigger Correlations

```python
# Weather triggers (based on AS literature):
def apply_weather_trigger(patient_data, weather_sensitivity):
    """
    Real AS patients show:
    - 48-72h lag between pressure drop and symptoms
    - Humidity affects stiffness more than pain
    - Cold affects peripheral joints more than axial
    """

    pressure_change = patient_data['pressure_change_24h']

    # Only triggers if pressure drops significantly (> 5 mmHg)
    if pressure_change < -5:
        trigger_probability = min(0.4, abs(pressure_change) * 0.05 * weather_sensitivity)
        lag_hours = np.random.randint(24, 72)

        if np.random.random() < trigger_probability:
            return {'trigger': True, 'lag_hours': lag_hours, 'intensity': 0.5 + abs(pressure_change) * 0.1}

    return {'trigger': False}
```

---

## Phase 4: PERSONALIZATION PIPELINE - 2-3 Weeks

### Enable On-Device Learning

Your `ContinuousLearningPipeline.swift` and `BootstrapStrategy.swift` already exist but are disabled.

**Step 1**: Make CoreML model updatable

```python
# In model export script:
import coremltools as ct

# Convert model
mlmodel = ct.convert(pytorch_model, ...)

# Make last 3 layers updatable
spec = mlmodel.get_spec()
builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)

# Mark classification layers as updatable
for layer in spec.neuralNetwork.layers[-3:]:
    layer.isUpdatable = True

# Add training inputs
builder.set_categorical_cross_entropy_loss(name='loss', input='output')
builder.set_adam_optimizer(AdamOptimizerParams(lr=0.001, batch=32, epochs=5))
builder.set_epochs(5)

mlmodel.save('ASFlarePredictor_Updatable.mlmodel')
```

**Step 2**: Enable learning pipeline in Swift

```swift
// In UnifiedNeuralEngine.swift, uncomment/add:
private var learningPipeline: ContinuousLearningPipeline?

func enablePersonalization() {
    learningPipeline = ContinuousLearningPipeline(
        baseModel: coreMLModel,
        updateInterval: .daily,
        minSamplesForUpdate: 7
    )
}

// After each daily check-in:
func recordOutcome(flareOccurred: Bool, forPredictionDate: Date) async {
    guard let pipeline = learningPipeline else { return }

    let sample = MLTrainingSample(
        features: lastExtractedFeatures,
        label: flareOccurred ? 1.0 : 0.0,
        date: forPredictionDate
    )

    await pipeline.addSample(sample)

    // Auto-update if enough samples
    if pipeline.pendingSamples >= 7 {
        try await pipeline.updateModel()
    }
}
```

### Personalization Timeline

| Days | Model State | Expected Accuracy | What Model Learns |
|------|-------------|-------------------|-------------------|
| 0-7 | 100% baseline | 55-65% | User's normal vital ranges |
| 8-14 | 30% personalized | 60-70% | Sleep/weather sensitivity |
| 15-28 | 60% personalized | 70-78% | Weekly patterns, stress triggers |
| 29-60 | 80% personalized | 75-82% | Personal flare signature |
| 60-90 | 95% personalized | 80-88% | Multi-day pattern recognition |
| 90+ | Fully personalized | 85-92% | Anomaly detection, early warnings |

---

## Phase 5: ALTERNATIVE APPROACHES (If CoreML Fails)

### Option A: Gradient Boosting (XGBoost) - Simpler, More Robust

```python
import xgboost as xgb

# XGBoost handles missing values natively
# Better for tabular data with mixed feature types
# Faster to train, smaller model size

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=4.0,  # Handle class imbalance
    use_label_encoder=False,
    eval_metric='auc'
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=10)

# Export to CoreML
import coremltools as ct
coreml_model = ct.converters.xgboost.convert(model)
```

### Option B: Statistical + Rules Hybrid

Instead of pure ML, combine:
1. **Statistical correlations** (what FlarePredictor already does)
2. **Expert rules** (from rheumatology literature)
3. **Personal thresholds** (learned from user data)

```swift
struct HybridPredictor {
    // Rule-based triggers (from medical literature)
    let pressureDropThreshold: Float = -5.0  // mmHg in 24h
    let sleepDeprivationThreshold: Float = 5.0  // hours below normal
    let stressThreshold: Float = 7.0  // 0-10 scale

    // Personalized multipliers (learned)
    var weatherSensitivity: Float = 1.0  // 0.5-2.0
    var stressSensitivity: Float = 1.0
    var sleepSensitivity: Float = 1.0

    func predictFlareRisk(features: FeatureVector) -> Float {
        var risk: Float = 0.0

        // Rule-based risk
        if features.pressureChange24h < pressureDropThreshold {
            risk += 0.2 * weatherSensitivity
        }

        if features.sleepHours < (features.avgSleepHours - sleepDeprivationThreshold) {
            risk += 0.15 * sleepSensitivity
        }

        if features.stressLevel > stressThreshold {
            risk += 0.15 * stressSensitivity
        }

        // Statistical correlation risk (existing)
        let correlationRisk = statisticalEngine.calculateRisk(features)

        // Combine (weighted average)
        return 0.4 * risk + 0.6 * correlationRisk
    }
}
```

---

## Implementation Priority

### Week 1: Emergency Fixes
- [ ] Fix 1.1: Add model collapse detection
- [ ] Fix 1.2: Fix scaler loading (fail fast)
- [ ] Fix 1.3: Increase data quality threshold
- [ ] Fix 1.4: Add output monitoring

### Week 2: Architecture Fixes
- [ ] Fix 2.1: Change to unidirectional LSTM
- [ ] Fix 2.2: Reduce to 35 features
- [ ] Fix 2.3: Fix train/test split

### Week 3-4: Training Data
- [ ] Fix 3.1: Implement class weighting
- [ ] Fix 3.2: Improve synthetic data
- [ ] Retrain model with all fixes

### Week 5-6: Personalization
- [ ] Enable updatable CoreML model
- [ ] Activate learning pipeline
- [ ] Test personalization flow

---

## Success Metrics

| Metric | Current | Target (Phase 1) | Target (Phase 4) |
|--------|---------|------------------|------------------|
| Accuracy | 57% | 70% | 85% |
| Precision | ~19% | 50% | 75% |
| Recall | ~100% | 60% | 80% |
| F1 Score | ~32% | 55% | 77% |
| AUC-ROC | ~0.50 | 0.70 | 0.88 |

---

## Files to Modify

| File | Changes |
|------|---------|
| `UnifiedNeuralEngine.swift` | Fixes 1.1-1.4, 2.2 |
| `FeatureExtractor.swift` | Fix 2.2 (reduce features) |
| `neural_flare_net.py` | Fix 2.1 (unidirectional) |
| `trainer.py` | Fixes 2.3, 3.1 |
| `generate_training_data.py` | Fixes 3.2, 3.3 |
| `ContinuousLearningPipeline.swift` | Phase 4 |
| `minmax_params.json` | Update for 35 features |

---

**Created**: 2024-12-04
**Status**: Ready for Implementation
**Priority**: CRITICAL - Current model provides no value
