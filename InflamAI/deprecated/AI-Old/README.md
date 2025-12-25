# Deprecated AI Files

**Moved**: 2025-01-22
**Reason**: iOS incompatibility, replaced by FlarePredictor

---

## Why These Files Were Deprecated

These 26 files were removed from the active Xcode build because:

1. **iOS Incompatibility**: Import `CreateML` and `TabularData` frameworks (macOS-only)
2. **Build Failures**: Cannot compile for iPhone/iPad
3. **Functional Replacement**: Superseded by `Core/ML/FlarePredictor.swift`
4. **Not in Use**: No active UI components reference these files

---

## Files in This Directory

### Core AI Engines (with CreateML)
- `AIMLEngine.swift` (1096 lines) - Main singleton, used heuristics disguised as ML
- `PainPredictionEngine.swift` - Runtime ML training (impossible on iOS)
- `PainPredictionEngine-Core.swift` - Duplicate from Core/AI/
- `AIHealthAnalyticsModule.swift` - TabularData-based analytics
- `DataAnalyticsEngine.swift` - Synthetic data generators
- `PredictiveModelEngine.swift`
- `HealthDataAnalyticsEngine.swift`
- `MLModelManager.swift`

### Specialized Engines (with CreateML)
- `VoiceCommandSystem.swift`
- `PersonalizedTreatmentRecommendationEngine.swift`
- `NaturalLanguageProcessingEngine.swift`
- `GestureRecognitionEngine.swift`
- `RealTimeVitalSignsMonitor.swift`
- `VitalSignsMonitor.swift`
- `SentimentAnalysisEngine.swift`

### ML Models (with CreateML)
- `ReminderPersonalizationModel.swift`
- `OptimalTimingModel.swift`
- `AdherencePredictionModel.swift`

### View Components (used AIMLEngine)
- `DetailedPainAnalysisView.swift`
- `AdvancedPainTrackingView.swift`
- `VoiceCommandPainTrackingView.swift`
- `AppleWatchPainMonitoringView.swift`
- `ARBodyScanningView.swift`
- `PainIntensityHistoryView.swift`
- `PainIntensityControlView.swift`

### Data Layer (incompatible schema)
- `PainDataStore.swift` - Used different PainEntry model
- `VoiceCommandEngine.swift` - Coordinated voice+AI

---

## Active Replacement

**Current System**: `Core/ML/FlarePredictor.swift`

**Key Differences**:
- ✅ Uses statistical pattern analysis (no ML)
- ✅ iOS-compatible (no CreateML)
- ✅ Real Core Data queries (no synthetic data)
- ✅ Proper error handling (throws if insufficient data)
- ✅ Medical disclaimers in UI
- ✅ Privacy-first (100% on-device)

---

## Technical Issues with Deprecated Files

### 1. CreateML Import Problem
```swift
// ❌ This fails on iOS
import CreateML
import TabularData

let regressor = try MLRegressor(...)  // iOS runtime error
```

### 2. Runtime Training Attempts
```swift
// ❌ Cannot train on iOS
func trainInitialModel() {
    let regressor = try MLRegressor(
        trainingData: table,
        targetColumn: "painLevel"
    )
    // This crashes on iPhone
}
```

### 3. Missing Model Assets
```swift
// ❌ No .mlmodel files exist in repo
let modelURL = getModelURL() // Returns Documents/Model.mlmodel
let model = try MLModel(contentsOf: modelURL) // File not found
```

### 4. Synthetic Data Generation
```swift
// ❌ Fabricated training data
func generateSyntheticTrainingData() -> [PainDataEntry] {
    var entries: [PainDataEntry] = []
    for _ in 0..<1000 {
        let temperature = Double.random(in: -10...35)
        let pain = calculatePain(temperature: temperature) + randomNoise()
        entries.append(PainDataEntry(pain: pain, ...))
    }
    return entries // Not real patient data
}
```

### 5. Misleading Confidence Scores
```swift
// ❌ Arbitrary confidence values
func analyzePainPattern() -> PainInsight {
    let avgPain = painEntries.map { $0.level }.average()
    return PainInsight(
        text: avgPain > 7 ? "High risk" : "Low risk",
        confidence: 0.95 // No validation basis
    )
}
```

---

## Why Not Delete Permanently?

These files are **moved, not deleted** because:

1. **Reference Value**: Implementation patterns may be useful for future features
2. **Code History**: Preserves development context
3. **Refactoring Opportunity**: Some non-ML logic could be extracted
4. **Educational**: Shows what NOT to do in medical apps

---

## Migration Path (If Needed)

If you want to reintroduce these features:

### Option 1: Port to Statistical Methods
1. Replace CreateML logic with comparative statistics
2. Remove synthetic data generators
3. Add proper error handling
4. Update UI to clarify methodology

### Option 2: Offline ML Training
1. Train models in Python (scikit-learn, TensorFlow)
2. Export to Core ML (`.mlmodel`)
3. Compile to `.mlmodelc`
4. Bundle in app for inference-only
5. Never train at runtime

### Option 3: Remove Entirely
1. Delete this directory
2. These features were never activated
3. Current system is sufficient

---

## Verification

To verify these files are NOT in the build:

```bash
cd /path/to/InflamAI

# List all Swift files in Xcode build
ruby -e "
  require 'xcodeproj'
  project = Xcodeproj::Project.open('InflamAI.xcodeproj')
  files = project.targets.first.source_build_phase.files
  swift_files = files.map { |f| f.file_ref&.path }.compact.select { |f| f.end_with?('.swift') }
  puts swift_files.sort
"

# Should return 30 files, none from deprecated/AI-Old/
```

---

## Questions?

See [ARCHITECTURE.md](../../../ARCHITECTURE.md) for full system documentation.
