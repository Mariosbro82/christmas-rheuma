# ML-Based Trigger Detection System for InflamAI

## Comprehensive Implementation Plan

**Document Version**: 1.0
**Date**: December 7, 2025
**Target**: Personal trigger detection that answers "What makes my pain worse?"

---

## Executive Summary

This plan outlines a **3-phase approach** to building an on-device, privacy-first ML system for detecting personal AS triggers. The system will help users understand what factors (coffee, sleep, weather, stress, etc.) affect their symptoms.

### Key Design Principles

1. **Privacy-First**: 100% on-device processing, zero cloud inference
2. **Explainable**: Users must understand WHY something is a trigger
3. **Progressive**: Start simple (statistics), add ML only when justified
4. **Scientifically Sound**: Clinically validated methodologies
5. **Updatable**: Continuously learns from new user data

### The Data Reality

| Days of Data | Samples | ML Approach | Explainability |
|-------------|---------|-------------|----------------|
| 1-30 | 30 | Statistics only | Excellent |
| 31-90 | 90 | Statistics + k-NN | Good |
| 90+ | 90+ | Statistics + k-NN + Neural | Good (with SHAP) |

---

## Phase 1: Enhanced Statistical Analysis (Weeks 1-3)

### Goal
Transform existing CorrelationEngine into a comprehensive trigger detection system with lagged analysis, effect sizes, and user-friendly explanations.

### Why Start Here?
- **No overfitting risk** with any dataset size
- **Fully explainable** - users see exactly why coffee is flagged
- **Clinically validated** - Pearson correlation is medical gold standard
- **Existing foundation** - FlarePredictor already has correlation logic

### 1.1 Core Components to Build

#### A. Lagged Correlation Engine
**Purpose**: Detect delayed trigger effects (coffee today → pain tomorrow)

```swift
// File: Core/ML/TriggerAnalysis/LaggedCorrelationEngine.swift

struct LagResult {
    let lag: Int                    // 0 = same day, 1 = next day, etc.
    let lagDescription: String      // "Same day", "Next day", "2 days later"
    let correlation: Double         // Pearson r (-1 to +1)
    let pValue: Double              // Statistical significance
    let isSignificant: Bool         // p < 0.05
    let effectStrength: EffectStrength

    enum EffectStrength: String {
        case strong = "Strong"      // |r| > 0.5
        case moderate = "Moderate"  // |r| > 0.3
        case weak = "Weak"          // |r| > 0.2
        case negligible = "Negligible"
    }
}

class LaggedCorrelationEngine {
    /// Analyze trigger with multiple lag offsets
    func analyzeWithLags(
        trigger: [Double],
        symptom: [Double],
        maxLag: Int = 3
    ) -> [LagResult]

    /// Find optimal lag for a trigger
    func findBestLag(
        trigger: [Double],
        symptom: [Double]
    ) -> LagResult?
}
```

**Default Lags to Analyze**:
- **0h (Same day)**: Immediate effects (medication)
- **12h (Overnight)**: Sleep quality
- **24h (Next day)**: Most common trigger delay for inflammatory responses
- **48h (2 days)**: Delayed immune responses

#### B. Effect Size Calculator
**Purpose**: Quantify clinical significance (not just statistical significance)

```swift
// File: Core/ML/TriggerAnalysis/EffectSizeCalculator.swift

struct TriggerEffect {
    let trigger: String
    let sampleSize: Int
    let meanWithTrigger: Double     // Average pain on trigger days
    let meanWithoutTrigger: Double  // Average pain on non-trigger days
    let meanDifference: Double      // How much more pain with trigger
    let cohenD: Double              // Standardized effect size
    let percentIncrease: Double     // "40% more pain with coffee"
    let clinicallySignificant: Bool // |d| > 0.5 AND |diff| > 1.0

    var userFriendlyDescription: String {
        // "Coffee increases your pain by 2.1 points on average"
    }
}

class EffectSizeCalculator {
    /// Calculate Cohen's d effect size
    func calculateCohenD(
        painWithTrigger: [Double],
        painWithoutTrigger: [Double]
    ) -> Double

    /// Generate complete trigger effect analysis
    func analyzeTriggerEffect(
        trigger: String,
        triggerDays: [Date],
        allLogs: [SymptomLog]
    ) -> TriggerEffect
}
```

**Cohen's d Interpretation**:
- **0.2**: Small effect (noticeable but minimal)
- **0.5**: Medium effect (clinically meaningful)
- **0.8**: Large effect (strong impact)

#### C. Confidence Classification
**Purpose**: Tell users how reliable the trigger detection is

```swift
// File: Core/ML/TriggerAnalysis/TriggerConfidence.swift

enum TriggerConfidence: String {
    case high = "High Confidence"
    // Criteria: n ≥ 60 days, p < 0.01, |r| > 0.5

    case medium = "Moderate Confidence"
    // Criteria: n ≥ 30 days, p < 0.05, |r| > 0.3

    case low = "Low Confidence"
    // Criteria: Anything else

    case insufficient = "Insufficient Data"
    // Criteria: n < 14 days for this trigger

    var icon: String
    var color: Color
    var recommendation: String
}

class ConfidenceClassifier {
    func classify(
        sampleSize: Int,
        pValue: Double,
        correlation: Double,
        effectSize: Double
    ) -> TriggerConfidence
}
```

#### D. Comprehensive Trigger Report
**Purpose**: User-facing summary of all findings

```swift
// File: Core/ML/TriggerAnalysis/TriggerReport.swift

struct TriggerAnalysisReport {
    let trigger: String
    let analysisDate: Date
    let sampleSize: Int
    let trackedDays: Int            // Total days with any data
    let triggerDays: Int            // Days this trigger was present

    // Statistical results
    let correlation: Double
    let pValue: Double
    let effectSize: TriggerEffect
    let lagResults: [LagResult]
    let bestLag: LagResult?

    // Classification
    let confidence: TriggerConfidence
    let category: TriggerCategory   // food, sleep, activity, weather, stress

    // User-facing output
    func generateReport() -> String
    func generateRecommendation() -> String
}

enum TriggerCategory: String, CaseIterable {
    case food = "Food & Drink"
    case sleep = "Sleep"
    case activity = "Physical Activity"
    case weather = "Weather"
    case stress = "Stress & Mental"
    case medication = "Medication"
    case other = "Other"
}
```

### 1.2 Triggers to Track

#### Food & Drink (User-Logged)
| Trigger | Why Track | Expected Lag |
|---------|-----------|--------------|
| Coffee/Caffeine | Inflammation, sleep disruption | 12-24h |
| Alcohol | Immune modulation | 24-48h |
| Sugar (high intake) | Inflammatory response | 12-24h |
| Dairy | Potential sensitivity | 24-48h |
| Gluten | Gut-immune axis | 24-72h |
| Nightshades | Inflammatory for some | 24-48h |
| Red meat | Inflammatory markers | 24-48h |
| Processed food | Inflammatory | 24h |

#### Sleep (HealthKit + User)
| Trigger | Data Source | Expected Lag |
|---------|-------------|--------------|
| Sleep duration < 6h | HealthKit | Same day - 24h |
| Sleep duration < 7h | HealthKit | Same day - 24h |
| Poor sleep quality | User rating | Same day - 24h |
| Late bedtime (>12am) | HealthKit | Same day - 24h |
| Sleep interruptions | HealthKit | Same day |

#### Physical Activity (HealthKit + User)
| Trigger | Data Source | Expected Lag |
|---------|-------------|--------------|
| Low activity (< 3000 steps) | HealthKit | Same day - 24h |
| High activity (> 15000 steps) | HealthKit | Same day - 24h |
| Prolonged sitting | User log | Same day |
| Exercise (any) | HealthKit/User | Same day - 24h |
| Exercise intensity | User rating | Same day - 48h |
| No stretching | User log | 24-48h |

#### Weather (OpenMeteo)
| Trigger | Data Source | Expected Lag |
|---------|-------------|--------------|
| Pressure drop > 5 hPa | OpenMeteo | 6-24h |
| Pressure drop > 10 hPa | OpenMeteo | 6-24h |
| High humidity > 80% | OpenMeteo | Same day |
| Cold temperature < 10°C | OpenMeteo | Same day |
| Temperature swing > 10°C | OpenMeteo | Same day |
| Rain/Storm | OpenMeteo | Same day |

#### Stress & Mental (User-Logged)
| Trigger | Data Source | Expected Lag |
|---------|-------------|--------------|
| High stress (> 7/10) | User rating | Same day - 48h |
| Anxiety (> 7/10) | User rating | Same day - 48h |
| Work hours > 10h | User log | 24-48h |
| Poor mood | User rating | Same day |

#### Medication (User-Logged)
| Trigger | Data Source | Expected Lag |
|---------|-------------|--------------|
| Missed NSAID | Medication log | 12-24h |
| Missed biologic | Medication log | 3-7 days |
| New medication | Medication log | Variable |

### 1.3 Data Schema Additions

```swift
// Core Data Entity: TriggerLog
// Purpose: Track specific trigger events for analysis

@objc(TriggerLog)
class TriggerLog: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var timestamp: Date
    @NSManaged var triggerCategory: String    // food, sleep, activity, etc.
    @NSManaged var triggerName: String        // "coffee", "alcohol", etc.
    @NSManaged var triggerValue: Double       // Quantity or intensity (0-10)
    @NSManaged var triggerUnit: String?       // "cups", "hours", "steps"
    @NSManaged var notes: String?
    @NSManaged var symptomLog: SymptomLog?    // Link to same-day symptoms
}
```

### 1.4 UI Components

#### A. Trigger Logging Flow
```
Quick Log Screen:
┌─────────────────────────────────────────┐
│  What did you have today?               │
├─────────────────────────────────────────┤
│  ☕ Coffee     [ 0 ] [ 1 ] [ 2 ] [3+]   │
│  🍺 Alcohol    [ 0 ] [ 1 ] [ 2 ] [3+]   │
│  🍬 Sugar      [Low] [Med] [High]       │
│  😰 Stress     [1] [2] [3] [4] [5]...[10]│
│  🏃 Exercise   [None] [Light] [Moderate]│
│                        [Intense]        │
├─────────────────────────────────────────┤
│  + Add custom trigger                   │
└─────────────────────────────────────────┘
```

#### B. Trigger Insights Dashboard
```
My Triggers Screen:
┌─────────────────────────────────────────┐
│  🎯 YOUR CONFIRMED TRIGGERS             │
├─────────────────────────────────────────┤
│  1. Short Sleep (< 6h)                  │
│     ✓ High Confidence                   │
│     +2.8 pain next day (p=0.001)        │
│     [View Details →]                    │
├─────────────────────────────────────────┤
│  2. Coffee (2+ cups)                    │
│     ⚠ Moderate Confidence               │
│     +1.5 pain next day (p=0.03)         │
│     Track 12 more coffee days           │
│     [View Details →]                    │
├─────────────────────────────────────────┤
│  ⚪ POTENTIAL TRIGGERS (Need more data) │
│  • High stress: +0.9 pain (8 days)      │
│  • Alcohol: +0.6 pain (5 days)          │
│  • Dairy: Insufficient data             │
└─────────────────────────────────────────┘
```

#### C. Trigger Detail View
```
Coffee Analysis:
┌─────────────────────────────────────────┐
│  ☕ COFFEE IMPACT                        │
│                                         │
│  📊 Effect Timeline                     │
│  ┌───────────────────────────────────┐  │
│  │       |                           │  │
│  │  +2.1 |    ████                   │  │
│  │       |    ████                   │  │
│  │  +1.0 |████████                   │  │
│  │       |████████████               │  │
│  │   0   |________________________________│
│  │       Same  Next   2 days  3 days │  │
│  │       day   day                   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ⏱️ Timing: Effect peaks NEXT DAY       │
│                                         │
│  📈 Statistics:                         │
│  • Days with coffee: 23                 │
│  • Days without: 45                     │
│  • Avg pain WITH coffee: 6.2           │
│  • Avg pain WITHOUT coffee: 4.7        │
│  • Difference: +1.5 points             │
│  • Correlation: r=0.42 (moderate)      │
│  • Significance: p=0.028               │
│                                         │
│  💡 Recommendation:                     │
│  "Consider reducing coffee intake,     │
│  especially when stress is high.       │
│  Your data shows the combination       │
│  increases pain significantly."        │
└─────────────────────────────────────────┘
```

### 1.5 Implementation Tasks

```
Week 1:
├── Create TriggerLog Core Data entity
├── Build LaggedCorrelationEngine
├── Build EffectSizeCalculator
├── Build ConfidenceClassifier
├── Unit tests for all calculators
└── Integration with existing CorrelationEngine

Week 2:
├── Build TriggerAnalysisReport generator
├── Create Quick Trigger Log UI
├── Create Trigger Insights Dashboard
├── Create Trigger Detail View
└── Swift Charts integration for lag visualizations

Week 3:
├── Add all trigger types to logging flow
├── Build trigger recommendation engine
├── Add push notification reminders for logging
├── Performance optimization
├── Accessibility testing (VoiceOver, Dynamic Type)
└── User testing & feedback incorporation
```

---

## Phase 2: k-Nearest Neighbors Personalization (Weeks 4-6)

### Goal
Add Core ML k-NN model for "similar day" matching and improved predictions.

### Why k-NN?
- **No overfitting**: Non-parametric, just memorizes examples
- **Updatable on-device**: Core ML KNearestNeighborsClassifier supports MLUpdateTask
- **Intuitive explanations**: "Your 5 most similar days had average pain 7.2"
- **Works with small datasets**: Even 30 samples are useful

### 2.1 Architecture

```
Input Features (15-20):
├── Lagged triggers (coffee_lag1, sleep_lag1, etc.)
├── Rolling averages (pain_7d_avg, hrv_7d_avg)
├── Weather features (pressure_change, humidity)
├── Temporal features (day_of_week, is_weekend)
└── Interaction terms (coffee_x_stress)

k-NN Model:
├── k = 5 (5 nearest neighbors)
├── Distance: Euclidean (normalized features)
├── Output: Average pain of k neighbors
└── Explanation: Show the 5 similar days

Update Mechanism:
├── On each new SymptomLog, add to k-NN training set
├── MLUpdateTask runs in background
└── Model persisted to Application Support
```

### 2.2 Core ML Model Creation

```python
# create_knn_model.py
import coremltools as ct
from coremltools.models.nearest_neighbors import NearestNeighborsSpec

# Create empty k-NN model (will be populated on-device)
spec = NearestNeighborsSpec(
    input_features=[
        ('features', ct.models.datatypes.Array(20)),  # 20 feature vector
    ],
    output_label='predicted_pain',
    k=5,
    distance_metric='euclidean'
)

model = ct.models.MLModel(spec)
model.short_description = "Personalized trigger detection via similar day matching"
model.author = "InflamAI"

# Mark as updatable
model.is_updatable = True

model.save("PersonalizedTriggerKNN.mlmodel")
```

### 2.3 Swift Integration

```swift
// File: Core/ML/TriggerAnalysis/KNNTriggerModel.swift

class KNNTriggerModel {
    private var model: MLModel?
    private let modelURL: URL

    // MARK: - Initialization
    init() {
        // Copy from bundle to writable location
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        modelURL = appSupport.appendingPathComponent("PersonalizedTriggerKNN.mlmodelc")

        copyBundleModelIfNeeded()
        loadModel()
    }

    // MARK: - Prediction
    func predictPain(features: [Double]) -> (prediction: Double, similarDays: [SimilarDay])? {
        // Returns prediction AND the 5 similar days for explanation
    }

    // MARK: - On-Device Update
    func addNewDay(features: [Double], painLevel: Double) {
        // Add to training set and trigger MLUpdateTask
    }

    // MARK: - Explanation
    struct SimilarDay {
        let date: Date
        let painLevel: Double
        let distance: Double
        let keyFeatures: [String: Double]  // What made this day similar
    }
}
```

### 2.4 Feature Engineering Pipeline

```swift
// File: Core/ML/TriggerAnalysis/TriggerFeatureExtractor.swift

class TriggerFeatureExtractor {

    /// Extract features for trigger analysis
    func extractFeatures(from logs: [SymptomLog]) -> [[String: Double]] {
        logs.enumerated().map { index, log in
            var features: [String: Double] = [:]

            // 1. Current day biometrics
            features["hrv"] = log.contextSnapshot?.hrvValue ?? 0
            features["resting_hr"] = log.contextSnapshot?.restingHeartRate ?? 0
            features["steps"] = log.contextSnapshot?.stepCount ?? 0
            features["sleep_hours"] = (log.contextSnapshot?.sleepDuration ?? 0) / 3600

            // 2. Weather features
            features["pressure"] = log.contextSnapshot?.barometricPressure ?? 1013.25
            features["humidity"] = log.contextSnapshot?.humidity ?? 50

            // 3. Trigger values (from TriggerLogs)
            features["coffee"] = getTriggerValue(log, "coffee")
            features["alcohol"] = getTriggerValue(log, "alcohol")
            features["stress"] = log.stressLevel ?? 0

            // 4. Lag features (t-1, t-2)
            if index > 0 {
                let yesterday = logs[index - 1]
                features["sleep_lag1"] = (yesterday.contextSnapshot?.sleepDuration ?? 0) / 3600
                features["coffee_lag1"] = getTriggerValue(yesterday, "coffee")
                features["stress_lag1"] = yesterday.stressLevel ?? 0
            }

            // 5. Rolling averages (7-day)
            if index >= 6 {
                let recentLogs = Array(logs[(index-6)...index])
                features["pain_7d_avg"] = recentLogs.map { $0.basdaiScore }.mean()
                features["hrv_7d_avg"] = recentLogs.compactMap { $0.contextSnapshot?.hrvValue }.mean()
            }

            // 6. Pressure change
            if index > 0 {
                let yesterdayPressure = logs[index-1].contextSnapshot?.barometricPressure ?? 1013.25
                let todayPressure = log.contextSnapshot?.barometricPressure ?? 1013.25
                features["pressure_change_24h"] = todayPressure - yesterdayPressure
            }

            // 7. Temporal features
            if let date = log.timestamp {
                let calendar = Calendar.current
                features["day_of_week"] = Double(calendar.component(.weekday, from: date))
                features["is_weekend"] = calendar.isDateInWeekend(date) ? 1.0 : 0.0
            }

            // 8. Interaction terms
            features["coffee_x_stress"] = (features["coffee"] ?? 0) * (features["stress"] ?? 0) / 10
            features["sleep_deficit_x_activity"] = max(0, 7 - (features["sleep_hours"] ?? 7)) * (features["steps"] ?? 0) / 10000

            return features
        }
    }
}
```

### 2.5 User Explanation UI

```
Similar Days Analysis:
┌─────────────────────────────────────────┐
│  📊 PREDICTION: Pain Level 6.8          │
│                                         │
│  Based on your 5 most similar days:     │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ Nov 15 - Pain: 7.2              │    │
│  │ ☕ Coffee: 2 | 😴 Sleep: 5.5h   │    │
│  │ 📊 Similar because: low sleep,  │    │
│  │    high stress, coffee intake   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ Nov 8 - Pain: 6.5               │    │
│  │ ☕ Coffee: 2 | 😴 Sleep: 6h     │    │
│  │ 📊 Similar because: coffee,     │    │
│  │    pressure drop                │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [Show 3 more similar days]             │
│                                         │
│  💡 Key patterns from similar days:     │
│  • All had coffee ≥ 2 cups              │
│  • 4/5 had sleep < 6.5 hours            │
│  • Average stress: 7.2/10               │
└─────────────────────────────────────────┘
```

### 2.6 Implementation Tasks

```
Week 4:
├── Create k-NN Core ML model (Python + coremltools)
├── Build TriggerFeatureExtractor
├── Build KNNTriggerModel wrapper
├── Implement feature normalization (min-max scaling)
└── Unit tests for feature extraction

Week 5:
├── Implement MLUpdateTask for on-device updates
├── Build "similar days" explanation generator
├── Create Similar Days UI component
├── Integrate with Trigger Insights Dashboard
└── Background update scheduling (BGProcessingTask)

Week 6:
├── Ensemble integration (Statistical + k-NN)
├── A/B comparison UI (Statistical vs k-NN predictions)
├── Performance benchmarking
├── Memory profiling (k-NN can grow with data)
└── User testing & iteration
```

---

## Phase 3: Lightweight Neural Network (Weeks 7-10)

### Goal
Add on-device updatable neural network for users with 90+ days of data who opt-in.

### Why Neural Network?
- **Non-linear patterns**: Captures complex interactions (coffee + stress + low sleep)
- **Predictive**: Can forecast tomorrow's pain, not just analyze past
- **Personalized**: Transfer learning from population model + fine-tuning

### Prerequisites
- User has 90+ days of consistent data
- At least 15 features with real values (not defaults)
- Statistical analysis shows significant correlations (something to learn)
- User opts into "Advanced AI Mode"

### 3.1 Architecture Design

```
Model: GRU-Attention (Optimized for Small Data)

Input Layer:
└── [batch_size, 7, 20] - 7 days history × 20 features

GRU Layer (32 units):
└── Captures temporal patterns across 7 days
└── Dropout 30% (aggressive regularization)

Self-Attention Layer:
└── Learns which features matter most
└── Provides explainability weights

Dense Layer (16 units):
└── ReLU activation
└── L2 regularization (λ=0.001)

Output Layer:
└── Single neuron (pain prediction 0-10)
└── Linear activation

Total Parameters: ~15,000 (acceptable for 90+ samples)
```

### 3.2 Model Training Strategy

#### A. Pre-trained Population Model
```python
# train_population_model.py
# Train on synthetic/aggregated AS patient data

import tensorflow as tf

def create_population_model():
    inputs = tf.keras.Input(shape=(7, 20))

    # GRU layer
    x = tf.keras.layers.GRU(32, return_sequences=True, dropout=0.3)(inputs)

    # Self-attention
    attention = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)

    # Dense layers
    x = tf.keras.layers.Dense(16, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Train on population data (synthetic or aggregated)
model = create_population_model()
model.fit(population_X, population_y, epochs=100, validation_split=0.2)
model.save("PopulationTriggerModel.h5")
```

#### B. Convert to Updatable Core ML

```python
# convert_to_coreml.py
import coremltools as ct

# Load trained Keras model
keras_model = tf.keras.models.load_model("PopulationTriggerModel.h5")

# Convert to Core ML
coreml_model = ct.convert(
    keras_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS17
)

# Make updatable
spec = coreml_model.get_spec()
spec.isUpdatable = True

# Mark only last 2 layers as trainable (transfer learning)
builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
builder.make_updatable(['dense_1', 'dense_2'])

# Set loss function
builder.set_mean_squared_error_loss(name='mse_loss', input='pain_prediction')

# Set optimizer
builder.set_sgd_optimizer(SgdParams(lr=0.001, batch=32, epochs=5))

# Save
ct.utils.save_spec(builder.spec, "UpdatableTriggerModel.mlmodel")
```

### 3.3 On-Device Training

```swift
// File: Core/ML/TriggerAnalysis/NeuralTriggerModel.swift

class NeuralTriggerModel {
    private var model: MLModel?
    private let modelURL: URL
    private var isTraining = false

    // MARK: - Training
    func trainOnUserData(logs: [SymptomLog]) async throws {
        guard logs.count >= 90 else {
            throw TriggerModelError.insufficientData(required: 90, available: logs.count)
        }

        isTraining = true
        defer { isTraining = false }

        // Prepare training data
        let (features, labels) = prepareTrainingData(from: logs)
        let batchProvider = try createBatchProvider(features: features, labels: labels)

        // Create update task
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let updateTask = try MLUpdateTask(
            forModelAt: modelURL,
            trainingData: batchProvider,
            configuration: config,
            completionHandler: { [weak self] context in
                if let updatedModel = context.model {
                    try? updatedModel.write(to: self?.modelURL ?? URL(fileURLWithPath: ""))
                    self?.model = updatedModel

                    // Log training metrics
                    let loss = context.metrics[.lossValue] as? Double ?? -1
                    print("Training complete. Final loss: \(loss)")
                }
            }
        )

        // Progress monitoring
        updateTask.progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            progressHandler: { context in
                let epoch = context.event.rawValue
                let loss = context.metrics[.lossValue] as? Double ?? -1
                print("Epoch \(epoch): Loss = \(loss)")
            }
        )

        updateTask.resume()
    }

    // MARK: - Prediction with Uncertainty
    func predictWithUncertainty(features: [Double]) -> (mean: Double, std: Double)? {
        // Monte Carlo Dropout: Run 50 predictions with dropout active
        var predictions: [Double] = []

        for _ in 0..<50 {
            if let pred = predict(features: features, withDropout: true) {
                predictions.append(pred)
            }
        }

        guard !predictions.isEmpty else { return nil }

        let mean = predictions.reduce(0, +) / Double(predictions.count)
        let variance = predictions.map { pow($0 - mean, 2) }.reduce(0, +) / Double(predictions.count)

        return (mean, sqrt(variance))
    }
}
```

### 3.4 SHAP-Based Explanations

Since Core ML doesn't support SHAP directly, we compute approximations:

```swift
// File: Core/ML/TriggerAnalysis/FeatureImportance.swift

class FeatureImportanceCalculator {
    private let model: NeuralTriggerModel

    /// Approximate SHAP values using permutation importance
    func calculateFeatureImportance(
        baseFeatures: [Double],
        featureNames: [String]
    ) -> [FeatureContribution] {
        let basePrediction = model.predict(features: baseFeatures) ?? 0

        var contributions: [FeatureContribution] = []

        for (index, name) in featureNames.enumerated() {
            // Permute this feature (set to mean or zero)
            var permutedFeatures = baseFeatures
            permutedFeatures[index] = 0  // or mean value

            let permutedPrediction = model.predict(features: permutedFeatures) ?? 0
            let contribution = basePrediction - permutedPrediction

            contributions.append(FeatureContribution(
                feature: name,
                contribution: contribution,
                direction: contribution > 0 ? .increases : .decreases
            ))
        }

        return contributions.sorted { abs($0.contribution) > abs($1.contribution) }
    }
}

struct FeatureContribution {
    let feature: String
    let contribution: Double
    let direction: Direction

    enum Direction {
        case increases, decreases
    }

    var userDescription: String {
        let verb = direction == .increases ? "increased" : "decreased"
        return "\(feature) \(verb) pain by \(abs(contribution), specifier: "%.1f") points"
    }
}
```

### 3.5 Attention Weight Extraction

```swift
// If using attention mechanism, extract weights directly

func extractAttentionWeights() -> [String: Double] {
    // Get attention weights from model output
    // Maps directly to feature importance

    return [
        "coffee": 0.25,      // 25% attention weight
        "sleep": 0.22,
        "stress": 0.18,
        "weather": 0.15,
        "activity": 0.10,
        "medication": 0.08,
        "other": 0.02
    ]
}
```

### 3.6 User Interface

```
Advanced AI Predictions:
┌─────────────────────────────────────────┐
│  🧠 NEURAL NETWORK ANALYSIS             │
│                                         │
│  Tomorrow's Prediction:                 │
│  ┌───────────────────────────────────┐  │
│  │       PAIN: 6.8 ± 1.2             │  │
│  │       (68% confidence)            │  │
│  │                                   │  │
│  │  ╔═══════════════════════════╗    │  │
│  │  ║████████████████░░░░░░░░░░░║    │  │
│  │  ╚═══════════════════════════╝    │  │
│  │    Low (2)      →      High (10)  │  │
│  │         Expected range: 5.6-8.0   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  📊 What's driving this prediction:     │
│                                         │
│  ☕ Coffee today        +1.8 pain       │
│  ████████████████░░░░░░░░░░░           │
│                                         │
│  😴 Sleep (5.5h)        +1.2 pain       │
│  ████████████░░░░░░░░░░░░░░░           │
│                                         │
│  📉 Pressure drop       +0.8 pain       │
│  ████████░░░░░░░░░░░░░░░░░░░           │
│                                         │
│  💊 Medication taken    -0.5 pain       │
│  ░░░░░░░░░░░░░░░████████░░░░           │
│                                         │
│  ═══════════════════════════════        │
│  Baseline: 3.5 (your average)           │
│                                         │
│  💡 Recommendation:                     │
│  "Consider skipping coffee and          │
│  prioritizing 7+ hours sleep tonight    │
│  to reduce tomorrow's expected pain."   │
└─────────────────────────────────────────┘
```

### 3.7 Implementation Tasks

```
Week 7:
├── Design neural architecture (GRU-Attention)
├── Create population training data (synthetic)
├── Train population base model (Python/TensorFlow)
├── Convert to updatable Core ML model
└── Unit tests for model conversion

Week 8:
├── Build NeuralTriggerModel Swift wrapper
├── Implement MLUpdateTask training
├── Implement Monte Carlo Dropout uncertainty
├── Build FeatureImportanceCalculator
└── Performance profiling (memory, speed)

Week 9:
├── Create opt-in flow for Advanced AI
├── Build prediction UI with uncertainty
├── Build feature contribution visualization
├── Integrate with existing trigger dashboard
└── A/B test: Statistical vs k-NN vs Neural

Week 10:
├── Background training scheduler (BGProcessingTask)
├── Model versioning and rollback
├── User feedback collection
├── Performance optimization
└── Documentation & code review
```

---

## Phase 4: Continuous Improvement (Ongoing)

### 4.1 Ensemble Integration

```swift
// Combine all three approaches for robust predictions

class TriggerEnsemble {
    let statisticalEngine: StatisticalTriggerEngine
    let knnModel: KNNTriggerModel
    let neuralModel: NeuralTriggerModel?

    func getEnsemblePrediction(features: [Double]) -> EnsemblePrediction {
        let statistical = statisticalEngine.predict(features)
        let knn = knnModel.predict(features)
        let neural = neuralModel?.predictWithUncertainty(features: features)

        // Weighted ensemble based on confidence
        let weights = calculateWeights(
            statisticalConfidence: statistical.confidence,
            knnSampleSize: knn.neighborCount,
            neuralUncertainty: neural?.std ?? Double.infinity
        )

        let prediction =
            weights.statistical * statistical.value +
            weights.knn * knn.value +
            weights.neural * (neural?.mean ?? 0)

        return EnsemblePrediction(
            value: prediction,
            statisticalComponent: statistical,
            knnComponent: knn,
            neuralComponent: neural,
            weights: weights
        )
    }
}
```

### 4.2 Federated Learning (Future)

For users who opt-in to help improve the population model:

```
Privacy-Preserving Aggregation:
├── Only model gradients leave device (not data)
├── Differential privacy noise added
├── Aggregated across 10,000+ users
├── Improves base model for new users
└── Apple Private Federated Learning framework
```

### 4.3 Trigger Discovery

```swift
// Automatically detect novel triggers user hasn't logged

class TriggerDiscovery {
    /// Analyze residuals to find unexplained variance
    func findUnexplainedPatterns(
        predictions: [Double],
        actuals: [Double],
        contextData: [ContextSnapshot]
    ) -> [PotentialTrigger] {
        // Look for days where prediction was way off
        // Check if any context feature correlates with residuals
        // Suggest new triggers to track
    }
}

// Example output:
// "We noticed your pain is often higher than predicted on Mondays.
//  Consider tracking work-related stress or weekend activities."
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│  Daily Check-in → SymptomLog (pain, fatigue, stiffness)         │
│  Quick Trigger Log → TriggerLog (coffee, alcohol, stress)       │
│  Medication Log → DoseLog (taken/missed)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AUTOMATIC COLLECTION                        │
│  HealthKit → Sleep, HRV, Steps, HR (via HealthKitService)       │
│  OpenMeteo → Pressure, Humidity, Temperature (via WeatherService)│
│  Core Data → Historical symptoms, medication adherence          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
│  TriggerFeatureExtractor                                        │
│  ├── Lag features (t-1, t-2)                                    │
│  ├── Rolling averages (7-day)                                   │
│  ├── Interaction terms (coffee × stress)                        │
│  └── Temporal encoding (day of week, weekend)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS ENGINES                            │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Statistical     │  │ k-NN Model      │  │ Neural Network  │  │
│  │ Engine          │  │                 │  │ (90+ days)      │  │
│  │ (Always active) │  │ (30+ days)      │  │                 │  │
│  │                 │  │                 │  │                 │  │
│  │ • Correlation   │  │ • Similar days  │  │ • GRU-Attention │  │
│  │ • Lag analysis  │  │ • Instance-     │  │ • Transfer      │  │
│  │ • Effect size   │  │   based         │  │   learning      │  │
│  │ • P-values      │  │ • Updateable    │  │ • MC Dropout    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│            │                   │                   │             │
│            └───────────────────┴───────────────────┘             │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                           │
│                    │ Ensemble        │                           │
│                    │ Combiner        │                           │
│                    └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EXPLAINABILITY                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Correlation     │  │ Similar Days    │  │ Feature         │  │
│  │ Reports         │  │ Explanation     │  │ Importance      │  │
│  │                 │  │                 │  │ (SHAP-like)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         USER OUTPUT                              │
│  ├── Trigger Insights Dashboard                                 │
│  ├── Individual Trigger Detail Views                            │
│  ├── Tomorrow's Prediction (with uncertainty)                   │
│  ├── Personalized Recommendations                               │
│  └── PDF Export for Rheumatologist                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Core/ML/TriggerAnalysis/
├── README.md                           # This document
├── Models/
│   ├── PersonalizedTriggerKNN.mlmodel  # k-NN model
│   └── UpdatableTriggerNN.mlmodel      # Neural network
├── Engines/
│   ├── LaggedCorrelationEngine.swift   # Lagged analysis
│   ├── EffectSizeCalculator.swift      # Cohen's d, etc.
│   ├── ConfidenceClassifier.swift      # High/Med/Low
│   └── StatisticalTriggerEngine.swift  # Main statistical engine
├── MLModels/
│   ├── KNNTriggerModel.swift           # k-NN wrapper
│   ├── NeuralTriggerModel.swift        # NN wrapper
│   └── TriggerEnsemble.swift           # Combined predictions
├── Features/
│   ├── TriggerFeatureExtractor.swift   # Feature engineering
│   └── FeatureScaler.swift             # Normalization
├── Explainability/
│   ├── TriggerAnalysisReport.swift     # User reports
│   ├── FeatureImportanceCalculator.swift # SHAP-like
│   └── SimilarDaysExplainer.swift      # k-NN explanations
├── Types/
│   ├── TriggerTypes.swift              # Enums, structs
│   ├── LagResult.swift
│   └── TriggerEffect.swift
└── Tests/
    ├── LaggedCorrelationTests.swift
    ├── EffectSizeTests.swift
    └── KNNModelTests.swift
```

---

## Success Metrics

### Phase 1 (Statistical)
- [ ] Lag analysis detects known delayed triggers
- [ ] Effect sizes match clinical literature
- [ ] User satisfaction: "I finally understand my triggers"
- [ ] 70%+ daily logging engagement

### Phase 2 (k-NN)
- [ ] Similar days match intuitive expectations
- [ ] Predictions within ±1.5 pain points
- [ ] Update time < 5 seconds
- [ ] Model size < 2 MB

### Phase 3 (Neural)
- [ ] Prediction MAE < 1.0 pain points
- [ ] Uncertainty calibrated (68% in ±1σ range)
- [ ] Feature importance aligns with statistical findings
- [ ] Training completes in < 60 seconds

### Overall
- [ ] No false trigger claims (p > 0.05)
- [ ] Medical disclaimers prominent
- [ ] 100% on-device (no cloud calls)
- [ ] WCAG AA accessibility compliance

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overfitting neural model | High | Medium | Aggressive regularization, ensemble |
| User doesn't log triggers | High | High | Gamification, reminders, auto-detection |
| False trigger claims | Medium | High | Confidence thresholds, disclaimers |
| Memory issues with k-NN | Low | Medium | Prune old data, cap at 365 days |
| Core ML training bugs | Medium | Medium | Fallback to statistical only |
| User distrust of AI | Medium | Medium | Transparent explanations, opt-in |

---

## Medical Disclaimers

All trigger analysis must display:

```
⚠️ NOT MEDICAL ADVICE

This analysis shows statistical patterns in YOUR data.
Correlation does not prove causation.

• Results may vary as you track more data
• Discuss findings with your rheumatologist
• Don't change medications without consulting a doctor
• Individual responses to triggers differ significantly

Learn more about how we analyze your data →
```

---

## References

1. Apple Core ML On-Device Training: https://developer.apple.com/documentation/coreml/mlmodel
2. SHAP Values: Lundberg & Lee, 2017 - https://arxiv.org/abs/1705.07874
3. Lagged Correlation in Health: https://pmc.ncbi.nlm.nih.gov/articles/PMC7153151/
4. Transfer Learning for Small Datasets: https://arxiv.org/html/2406.10050v1
5. Monte Carlo Dropout: Gal & Ghahramani, 2016 - https://arxiv.org/abs/1506.02142
6. Cohen's d Effect Size: Cohen, 1988 - Statistical Power Analysis

---

**Document Author**: Claude Code
**Last Updated**: December 7, 2025
**Status**: Ready for Implementation
