# Production-Ready ML System Plan

## Current State Analysis

### Critical Issues Discovered

| Issue | Severity | Impact |
|-------|----------|--------|
| `healthKitService = nil` in FeatureExtractor | **CRITICAL** | 23 HealthKit features always 0 |
| CoreML model trained on synthetic data | **HIGH** | Predictions match synthetic patterns, not user |
| No `training_data_source` metadata | **HIGH** | `isUsingSyntheticModel` always true |
| Two competing ML systems (FlarePredictor + UnifiedNeuralEngine) | **MEDIUM** | Confusing UX, unclear authoritative source |
| CalibrationEngine not integrated | **MEDIUM** | Uncalibrated probability outputs |
| No prediction tracking/validation | **MEDIUM** | Can't measure real-world accuracy |

### Data Flow Problem

```
Current (Broken):
User grants HealthKit ✓ → HealthKitService ready ✓ → UnifiedNeuralEngine.init()
    → FeatureExtractor(persistenceController) ← NO healthKitService!
    → extractActivityMetrics() → guard fails → return []
    → 23 features = 0 → Synthetic model makes predictions on incomplete data
    → UI shows synthetic predictions
```

---

## Phase 1: Fix HealthKit Data Integration (Critical)

### 1.1 Inject HealthKitService into FeatureExtractor

**File**: `Core/ML/UnifiedNeuralEngine.swift:81`

Change:
```swift
// BEFORE (broken)
self.featureExtractor = FeatureExtractor(persistenceController: persistenceController)

// AFTER (fixed)
self.featureExtractor = FeatureExtractor(
    persistenceController: persistenceController,
    healthKitService: HealthKitService.shared
)
```

### 1.2 Add HealthKit Authorization Check

**File**: `Core/ML/FeatureExtractor.swift`

Add authorization verification before extraction:
```swift
func extractWithHealthKitVerification(for date: Date) async -> [[Float]] {
    // 1. Check HealthKit authorization status
    let isAuthorized = await healthKitService?.isAuthorized() ?? false

    // 2. Log data source availability
    print("📊 Feature Extraction - HealthKit: \(isAuthorized ? "✓" : "✗")")

    // 3. Extract features with proper error handling
    return await extract30DayFeatures(endingOn: date)
}
```

### 1.3 Add Data Quality Tracking

**File**: `Core/ML/FeatureExtractor.swift` (new struct)

```swift
struct FeatureExtractionResult {
    let features: [[Float]]
    let healthKitFeaturesAvailable: Int
    let coreDateFeaturesAvailable: Int
    let weatherFeaturesAvailable: Int
    let dataQualityScore: Float  // 0-1
    let missingFeatures: [String]
}
```

### 1.4 Tasks
- [ ] Modify UnifiedNeuralEngine to pass HealthKitService.shared
- [ ] Add authorization check before feature extraction
- [ ] Create FeatureExtractionResult struct for quality tracking
- [ ] Add logging for each feature category
- [ ] Update UI to show data quality indicators

---

## Phase 2: Consolidate ML Systems

### 2.1 Make UnifiedNeuralEngine the Single Source of Truth

**Current State**: Two systems compete:
- `FlarePredictor.swift` (statistical, Pearson correlation)
- `UnifiedNeuralEngine.swift` (CoreML neural network)

**Target State**: UnifiedNeuralEngine as primary, FlarePredictor as fallback

### 2.2 Create Unified Prediction Interface

**File**: `Core/ML/MLPredictionService.swift` (new)

```swift
@MainActor
final class MLPredictionService: ObservableObject {
    static let shared = MLPredictionService()

    // Primary engine
    private let neuralEngine: UnifiedNeuralEngine
    // Fallback (statistical)
    private let statisticalPredictor: FlarePredictor

    @Published var currentPrediction: UnifiedPrediction?
    @Published var predictionSource: PredictionSource = .neuralEngine
    @Published var dataQuality: DataQualityReport?

    enum PredictionSource {
        case neuralEngine      // CoreML model
        case statistical       // FlarePredictor (fallback)
        case hybrid            // Ensemble of both
    }

    func getPrediction() async -> UnifiedPrediction {
        // 1. Try neural engine first
        // 2. Fall back to statistical if insufficient data
        // 3. Return uncertainty-aware prediction
    }
}
```

### 2.3 Update AIInsightsView

**File**: `Features/AI/AIInsightsView.swift`

- Remove dual-system display
- Use single `MLPredictionService`
- Show prediction source badge
- Display data quality indicators

### 2.4 Tasks
- [ ] Create MLPredictionService as unified interface
- [ ] Define UnifiedPrediction protocol/struct
- [ ] Implement fallback logic (neural → statistical)
- [ ] Update AIInsightsView to use single service
- [ ] Add prediction source indicator in UI
- [ ] Deprecate direct FlarePredictor usage in views

---

## Phase 3: Real Data Training Pipeline

### 3.1 Wire ContinuousLearningPipeline

**File**: `Core/ML/ContinuousLearningPipeline.swift`

Current: Pipeline exists but isn't connected to data flow

**Integration Points**:

1. **After Daily Check-In** (`DailyCheckInViewModel.completeCheckIn()`):
```swift
// After saving symptom log, add to training pipeline
await ContinuousLearningPipeline.shared.recordSymptomLog(
    features: extractedFeatures,
    log: symptomLog
)
```

2. **When Flare is Marked** (`FlareEvent` creation):
```swift
// Record outcome for retrospective labeling
await ContinuousLearningPipeline.shared.recordFlareOutcome(
    startDate: flareEvent.startDate,
    severity: flareEvent.severity
)
```

3. **Weekly Model Update** (background):
```swift
// Automatic weekly personalization check
func scheduleWeeklyPersonalization() {
    let trigger = UNTimeIntervalNotificationTrigger(
        timeInterval: 7 * 24 * 3600,
        repeats: true
    )
    // Trigger on-device training if criteria met
}
```

### 3.2 Implement Outcome Tracking

**File**: `Core/ML/OutcomeTracker.swift` (new)

```swift
class OutcomeTracker {
    // Track predictions vs actual outcomes
    struct PredictionOutcome: Codable {
        let predictionDate: Date
        let predictedProbability: Float
        let predictedFlare: Bool
        let actualFlare: Bool?  // nil until outcome known
        let outcomeDate: Date?
        let daysToFlare: Int?
    }

    // Store predictions for later validation
    func recordPrediction(_ prediction: FlareRiskPrediction)

    // Update when outcome is known (3-7 days later)
    func recordOutcome(flareOccurred: Bool, date: Date)

    // Calculate real-world accuracy
    func calculateAccuracyMetrics() -> AccuracyMetrics
}
```

### 3.3 Tasks
- [ ] Create OutcomeTracker class
- [ ] Wire DailyCheckInViewModel to ContinuousLearningPipeline
- [ ] Add flare outcome recording
- [ ] Implement weekly personalization scheduler
- [ ] Store prediction-outcome pairs for validation
- [ ] Add accuracy calculation methods

---

## Phase 4: Calibration Integration

### 4.1 Integrate CalibrationEngine

**File**: `Core/ML/CalibrationEngine.swift` (exists, needs integration)

**Integration in UnifiedNeuralEngine**:
```swift
class UnifiedNeuralEngine {
    private let calibrationEngine: CalibrationEngine

    func predict(features: [[Float]]) async throws -> FlareRiskPrediction {
        // 1. Get raw model output
        let rawProbability = try await model.predict(features)

        // 2. Apply temperature scaling
        let calibratedProbability = calibrationEngine.applyTemperatureScaling(
            rawProbability: rawProbability
        )

        // 3. Compute uncertainty (MC dropout approximation)
        let calibratedResult = try await calibrationEngine.predictWithUncertainty(
            features: features
        )

        return FlareRiskPrediction(
            probability: calibratedResult.probability,
            confidence: calibratedResult.confidence,
            uncertaintyScore: calibratedResult.uncertaintyScore,
            predictionInterval: calibratedResult.predictionInterval
        )
    }
}
```

### 4.2 Add Calibration UI Elements

**Show uncertainty in predictions**:
- Confidence interval (e.g., "45-65% risk")
- Confidence level badge (High/Medium/Low)
- Uncertainty decomposition (aleatoric vs epistemic)

### 4.3 Tasks
- [ ] Initialize CalibrationEngine in UnifiedNeuralEngine
- [ ] Apply temperature scaling to all predictions
- [ ] Add Monte Carlo uncertainty estimation
- [ ] Display confidence intervals in UI
- [ ] Show confidence level badges
- [ ] Add calibration metrics (ECE) to model info

---

## Phase 5: Observability & Transparency

### 5.1 Create ML Dashboard View

**File**: `Features/AI/MLDashboardView.swift` (new)

```swift
struct MLDashboardView: View {
    var body: some View {
        List {
            // Data Quality Section
            Section("Data Sources") {
                DataSourceRow(name: "HealthKit", status: healthKitStatus)
                DataSourceRow(name: "Weather", status: weatherStatus)
                DataSourceRow(name: "Symptom Logs", count: symptomLogCount)
            }

            // Model Performance Section
            Section("Model Performance") {
                MetricRow(name: "Real-World Accuracy", value: accuracy)
                MetricRow(name: "Calibration (ECE)", value: ece)
                MetricRow(name: "Predictions Made", value: predictionCount)
            }

            // Feature Availability Section
            Section("Feature Status") {
                FeatureGroupRow(name: "Clinical (15)", available: 15, total: 15)
                FeatureGroupRow(name: "HealthKit (23)", available: healthKitCount, total: 23)
                FeatureGroupRow(name: "Weather (8)", available: weatherCount, total: 8)
                FeatureGroupRow(name: "Mental (12)", available: mentalCount, total: 12)
            }

            // Personalization Section
            Section("Personalization") {
                ProgressRow(name: "Data Collection", progress: dataProgress)
                InfoRow(name: "Model Version", value: "v\(modelVersion)")
                InfoRow(name: "Last Update", value: lastUpdateDate)
            }
        }
    }
}
```

### 5.2 Add Real-Time Feature Display

**File**: `Features/AI/MLFeaturesDisplayView.swift` (enhance existing)

Show live values for all 92 features:
- Group by category
- Show source (HealthKit, CoreData, Weather, etc.)
- Highlight missing/zero values
- Show normalized vs raw values

### 5.3 Tasks
- [ ] Create MLDashboardView
- [ ] Enhance MLFeaturesDisplayView with live values
- [ ] Add data source status indicators
- [ ] Show real-time accuracy metrics
- [ ] Add feature availability breakdown
- [ ] Create export functionality for debugging

---

## Phase 6: Model Metadata & Versioning

### 6.1 Fix Synthetic Model Flag

**File**: `Core/ML/NeuralEnginePredictionService.swift`

Update model metadata handling:
```swift
private func checkModelMetadata() {
    guard let userDefined = model.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String] else {
        // No metadata = synthetic model
        self.isUsingSyntheticModel = true
        self.modelTrainingSource = .synthetic
        return
    }

    if let source = userDefined["training_data_source"] {
        switch source {
        case "real_patient_data":
            self.isUsingSyntheticModel = false
            self.modelTrainingSource = .realPatient
        case "user_personalized":
            self.isUsingSyntheticModel = false
            self.modelTrainingSource = .personalized
        default:
            self.isUsingSyntheticModel = true
            self.modelTrainingSource = .synthetic
        }
    }
}
```

### 6.2 Model Version Tracking

**File**: `Core/ML/ModelVersionManager.swift` (new)

```swift
class ModelVersionManager {
    struct ModelVersion: Codable {
        let version: Int
        let createdAt: Date
        let trainingDataPoints: Int
        let validationAccuracy: Float
        let calibrationECE: Float
        let trainingSource: TrainingSource
    }

    enum TrainingSource: String, Codable {
        case synthetic = "Synthetic Research Data"
        case personalized = "Your Personal Data"
        case hybrid = "Hybrid (Baseline + Personal)"
    }

    func recordModelUpdate(version: ModelVersion)
    func getVersionHistory() -> [ModelVersion]
    func getCurrentVersion() -> ModelVersion
}
```

### 6.3 Tasks
- [ ] Create ModelVersionManager
- [ ] Track model version history
- [ ] Store training metadata with each version
- [ ] Add rollback capability
- [ ] Show version info in UI
- [ ] Export model performance reports

---

## Phase 7: Testing & Validation

### 7.1 Unit Tests for Feature Extraction

**File**: `Tests/ML/FeatureExtractorTests.swift` (new)

```swift
class FeatureExtractorTests: XCTestCase {
    func testHealthKitFeaturesExtraction() async {
        // Given: Mock HealthKit data
        // When: Extract features
        // Then: HealthKit features populated correctly
    }

    func testFeatureNormalization() {
        // Given: Raw feature values
        // When: Apply scaler
        // Then: Values normalized correctly
    }

    func testMissingDataHandling() async {
        // Given: HealthKit unavailable
        // When: Extract features
        // Then: Graceful fallback, quality score reflects missing data
    }
}
```

### 7.2 Integration Tests

**File**: `Tests/ML/MLPipelineIntegrationTests.swift` (new)

```swift
class MLPipelineIntegrationTests: XCTestCase {
    func testEndToEndPrediction() async {
        // Given: User with 30+ days of data
        // When: Request prediction
        // Then: Valid prediction with calibrated uncertainty
    }

    func testPersonalizationPipeline() async {
        // Given: Sufficient training data
        // When: Trigger personalization
        // Then: Model updated, version incremented
    }
}
```

### 7.3 Validation Framework Usage

**File**: `Core/ML/ValidationFramework.swift` (exists)

Integrate validation checks:
- Run validation after each model update
- Track metrics over time
- Alert on performance degradation

### 7.4 Tasks
- [ ] Create FeatureExtractorTests
- [ ] Create MLPipelineIntegrationTests
- [ ] Add mock HealthKit data generator for tests
- [ ] Integrate ValidationFramework into ContinuousLearningPipeline
- [ ] Add CI/CD test pipeline
- [ ] Create performance regression tests

---

## Phase 8: UI Polish & User Experience

### 8.1 Update UnifiedNeuralEngineView

**Enhancements**:
- Show data quality score prominently
- Display confidence intervals on predictions
- Add "What's driving this?" expandable section
- Show feature importance (top 5 factors)
- Add "Improve Prediction" actionable tips

### 8.2 Update AIInsightsView

**Enhancements**:
- Single prediction card (not two systems)
- Clear "Research Model" vs "Personalized Model" indicator
- Progress toward personalization
- Historical accuracy display

### 8.3 Add Onboarding for ML Features

**File**: `Features/Onboarding/MLOnboardingView.swift` (new)

Explain to users:
- How predictions work
- Data requirements
- Privacy guarantees
- Personalization timeline
- Limitations/disclaimers

### 8.4 Tasks
- [ ] Redesign prediction cards with confidence intervals
- [ ] Add feature importance display
- [ ] Create actionable "Improve Prediction" section
- [ ] Add ML onboarding flow
- [ ] Update medical disclaimers
- [ ] Add accessibility support for all ML views

---

## Implementation Order

### Sprint 1: Critical Fixes (Phase 1)
**Priority: CRITICAL**
1. Fix HealthKitService injection in UnifiedNeuralEngine
2. Add authorization verification
3. Add data quality tracking
4. Verify HealthKit features flow correctly

### Sprint 2: System Consolidation (Phase 2)
**Priority: HIGH**
1. Create MLPredictionService
2. Update AIInsightsView
3. Implement fallback logic
4. Deprecate direct FlarePredictor usage

### Sprint 3: Training Pipeline (Phase 3)
**Priority: HIGH**
1. Wire ContinuousLearningPipeline
2. Create OutcomeTracker
3. Implement prediction-outcome validation
4. Add weekly personalization scheduler

### Sprint 4: Calibration (Phase 4)
**Priority: MEDIUM**
1. Integrate CalibrationEngine
2. Add temperature scaling
3. Implement uncertainty estimation
4. Update UI with confidence intervals

### Sprint 5: Observability (Phase 5)
**Priority: MEDIUM**
1. Create MLDashboardView
2. Enhance feature display
3. Add performance metrics
4. Create export functionality

### Sprint 6: Versioning & Testing (Phases 6-7)
**Priority: MEDIUM**
1. Create ModelVersionManager
2. Write unit tests
3. Write integration tests
4. Integrate ValidationFramework

### Sprint 7: UI Polish (Phase 8)
**Priority: LOW**
1. Update views with new designs
2. Add onboarding
3. Accessibility pass
4. Final polish

---

## Success Criteria

### Data Quality
- [ ] 100% of HealthKit features populated when authorized
- [ ] Data quality score visible in UI
- [ ] Clear indication of missing data sources

### Prediction Quality
- [ ] Calibrated probabilities (ECE < 0.1)
- [ ] Confidence intervals displayed
- [ ] Real-world accuracy tracked (target: > 60%)

### User Experience
- [ ] Single, clear prediction source
- [ ] Transparent data usage
- [ ] Actionable insights
- [ ] Clear path to personalization

### Technical Quality
- [ ] Unit test coverage > 60% for ML code
- [ ] Integration tests passing
- [ ] Model versioning working
- [ ] Rollback capability tested

---

## Files to Modify

| File | Phase | Change Type |
|------|-------|-------------|
| `Core/ML/UnifiedNeuralEngine.swift` | 1, 4 | Fix + Enhance |
| `Core/ML/FeatureExtractor.swift` | 1 | Enhance |
| `Core/ML/MLPredictionService.swift` | 2 | New |
| `Core/ML/OutcomeTracker.swift` | 3 | New |
| `Core/ML/ContinuousLearningPipeline.swift` | 3 | Wire |
| `Core/ML/CalibrationEngine.swift` | 4 | Integrate |
| `Core/ML/ModelVersionManager.swift` | 6 | New |
| `Features/AI/AIInsightsView.swift` | 2, 8 | Refactor |
| `Features/AI/UnifiedNeuralEngineView.swift` | 8 | Enhance |
| `Features/AI/MLDashboardView.swift` | 5 | New |
| `Features/CheckIn/DailyCheckInViewModel.swift` | 3 | Wire |
| `Tests/ML/*.swift` | 7 | New |

---

## Risk Mitigation

### Risk: Model Performance Degrades After Personalization
**Mitigation**:
- Keep baseline model as fallback
- Validate before deploying updated model
- Implement rollback capability

### Risk: HealthKit Authorization Revoked
**Mitigation**:
- Graceful degradation with clear UI indication
- Use cached biometric data when fresh data unavailable
- Statistical model as fallback

### Risk: Insufficient Data for Personalization
**Mitigation**:
- Clear progress indicators
- Reminders to log daily
- Hybrid model (baseline + partial personalization)

---

## Design Decisions (User Confirmed)

### 1. Partial HealthKit Data: **Graceful Degradation**
- Make predictions with available data
- Show clear warnings about missing data sources
- Display data quality score
- UI indicates which features are unavailable

### 2. Fallback Behavior: **Hybrid Approach**
- Show BOTH Neural Engine and Statistical predictions
- Clear source labeling for each prediction
- User can see which system is more confident
- Helps validate predictions against each other

### 3. Personalization Trigger: **Automatic with Opt-Out**
- Automatically personalize when 37+ days available
- User can disable auto-personalization in Settings
- Show notification before first personalization
- "Self-Learning" toggle in Privacy settings

### 4. Accuracy Metrics: **Developer Mode**
- Hidden by default in production UI
- Toggle in Settings → Developer Options
- Shows: Accuracy, ECE, prediction history, feature availability
- Useful for debugging without confusing regular users

---

**Plan Version**: 1.0
**Created**: 2024-11-30
**Estimated Total Effort**: 7 sprints (flexible scope)
