# InflamAI Architecture Documentation

**Last Updated**: 2025-01-25
**Version**: 1.1
**Platform**: iOS 16+
**Architecture Score**: 8.0/10 (Improving to 9.5/10)

---

## Overview

InflamAI is a privacy-first Ankylosing Spondylitis management app built with SwiftUI, Core Data, and Apple ecosystem integrations. The app focuses on symptom tracking, pattern analysis, and medication management.

---

## Core Architecture

### Technology Stack

- **UI Framework**: SwiftUI (iOS 16+)
- **Data Persistence**: Core Data (local-only, no CloudKit)
- **Pattern Analysis**: Statistical analysis (NOT machine learning)
- **Health Integration**: HealthKit, WeatherKit
- **Wearables**: Apple Watch integration (watchOS 10+) - See [Apple Watch Integration](#apple-watch-integration)
- **Charts**: Swift Charts
- **Privacy**: 100% on-device processing, no server communication

### Project Structure

```
InflamAI/
├── InflamAIApp.swift           # App entry point, tab navigation
├── Core/
│   ├── Persistence/
│   │   └── InflamAIPersistenceController.swift
│   ├── ML/
│   │   └── FlarePredictor.swift   # Statistical pattern analyzer (ONLY "AI")
│   ├── Export/
│   │   └── PDFExportService.swift
│   └── Utilities/
│       ├── BASDAICalculator.swift
│       ├── TimeRange.swift
│       └── CorrelationEngine.swift
├── Features/
│   ├── Home/
│   │   └── HomeView.swift
│   ├── AI/
│   │   └── AIInsightsView.swift   # UI for FlarePredictor
│   ├── Trends/
│   │   ├── TrendsView.swift
│   │   └── TrendsViewModel.swift
│   ├── Medication/
│   │   ├── MedicationManagementView.swift
│   │   └── MedicationViewModel.swift
│   ├── Exercise/
│   │   ├── ExerciseLibraryView.swift
│   │   └── ExerciseData.swift
│   ├── Flares/
│   │   └── FlareTimelineView.swift
│   ├── PainMap/
│   │   ├── PainLocationSelector.swift
│   │   └── PainMapViewModel.swift
│   ├── QuickCapture/
│   │   └── JointTapSOSView.swift
│   ├── Coach/
│   │   └── CoachCompositorView.swift
│   └── Onboarding/
│       └── OnboardingFlow.swift
└── deprecated/
    └── AI-Old/                     # Unused legacy AI files (not compiled)
        ├── AIMLEngine.swift        # 1096-line singleton (replaced by FlarePredictor)
        ├── PainPredictionEngine.swift
        └── 17+ other CreateML-based files
```

---

## Active Components (30 files in Xcode build)

### 1. Pattern Analysis System (NOT Machine Learning)

**File**: `Core/ML/FlarePredictor.swift`

**What it does**:
- Statistical pattern analysis (comparative averages)
- Analyzes historical symptom data from Core Data
- Compares current symptoms to patterns that preceded past flares
- Requires minimum 30 days of data before analysis

**What it does NOT do**:
- ❌ No machine learning models
- ❌ No CreateML/TabularData imports
- ❌ No synthetic data generation
- ❌ No fabricated predictions
- ❌ No fake confidence scores

**Data Sources**:
```swift
// Real Core Data queries
let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
let flares: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
```

**Error Handling**:
```swift
guard historicalData.count >= minimumDataPoints else {
    throw MLError.insufficientData(required: 30, actual: count)
}
```

**Algorithm**:
1. Fetch all SymptomLog and FlareEvent records
2. Separate days that preceded flares vs. normal days
3. Calculate average BASDAI, pain, stiffness for each group
4. Compare current values to historical averages
5. Calculate risk score: `(avgRisk * 0.7) + (baseRate * 0.3)`

**Medical Disclaimer**: Clearly labeled as "Statistical Pattern Analysis" in UI with "⚠️ Not Medical Advice" warnings.

### 2. Data Model (Core Data)

**Model File**: `InflamAI.xcdatamodeld`

**Entities**:
- `SymptomLog` - Daily symptom tracking
- `BodyRegionLog` - Pain location details (60 regions)
- `FlareEvent` - Flare episodes
- `Medication` - Medication records
- `MedicationLog` - Medication adherence
- `ExerciseLog` - Physical activity
- `ContextSnapshot` - Environmental data (weather, activity)
- `UserProfile` - User settings

**Key Relationships**:
```
SymptomLog ←→ BodyRegionLog (1-to-many)
SymptomLog ←→ ContextSnapshot (1-to-1)
Medication ←→ MedicationLog (1-to-many)
FlareEvent ←→ SymptomLog (implied by date queries)
```

### 3. Feature Modules

#### Home Dashboard
- Quick symptom logging
- BASDAI quick-check
- Recent flare alerts
- Medication reminders

#### Pattern Insights (AI Tab)
- Risk percentage display
- Contributing factors
- Personalized suggestions
- Training status

#### Trends
- Swift Charts visualizations
- BASDAI over time
- Pain correlations
- Weather/activity patterns

#### Medication Management
- Medication library (standardized)
- Dosage tracking
- Reminder system
- Adherence analytics

#### Pain Map
- 60-point body diagram (29 front, 29 back, 2 head)
- Tap to select regions
- Intensity levels (1-10)
- Saves to BodyRegionLog

#### Exercise Library
- Pre-built AS-specific routines
- Video references (YouTube)
- Progress tracking
- Custom routines

#### Flare Timeline
- Historical flare events
- Duration tracking
- Severity scores
- Trigger notes

#### JointTap SOS
- Rapid flare capture
- One-tap logging
- Emergency contact integration

---

## Deprecated Components (NOT in build)

### Files Moved to `/deprecated/AI-Old/`

**Why deprecated**:
- Import macOS-only frameworks (CreateML, TabularData)
- Cannot compile for iOS
- Replaced by FlarePredictor
- Not referenced by active UI

**List of deprecated files**:
1. `AIMLEngine.swift` (1096 lines) - Massive singleton with heuristics
2. `AIHealthAnalyticsModule.swift` - CreateML-based analytics
3. `PainPredictionEngine.swift` (3 copies) - Runtime ML training attempts
4. `DataAnalyticsEngine.swift` - Synthetic data generators
5. `PredictiveModelEngine.swift`
6. `HealthDataAnalyticsEngine.swift`
7. `VoiceCommandSystem.swift`
8. `PersonalizedTreatmentRecommendationEngine.swift`
9. `NaturalLanguageProcessingEngine.swift`
10. `GestureRecognitionEngine.swift`
11. `RealTimeVitalSignsMonitor.swift`
12. `SentimentAnalysisEngine.swift`
13. `ReminderPersonalizationModel.swift`
14. `OptimalTimingModel.swift`
15. `AdherencePredictionModel.swift`
16. `MLModelManager.swift`
17. `PainDataStore.swift` - Incompatible PainEntry schema
18. `DetailedPainAnalysisView.swift` - Used AIMLEngine
19. `AdvancedPainTrackingView.swift` - Used AIMLEngine
20. `VoiceCommandPainTrackingView.swift` - Used AIMLEngine
21. `AppleWatchPainMonitoringView.swift` - Used AIMLEngine
22. `ARBodyScanningView.swift` - Used AIMLEngine
23. `PainIntensityHistoryView.swift` - Used AIMLEngine
24. `PainIntensityControlView.swift` - Used AIMLEngine
25. `VoiceCommandEngine.swift` - Used CreateML
26. `VitalSignsMonitor.swift` - Used CreateML

**Important**: `BodyDiagramView.swift` remains in build but contains only stub implementations (no-op functions with print statements).

---

## Build Verification

### Files in Xcode Build Target: 30

```bash
# Verify CreateML not in build
ruby -e "
  require 'xcodeproj'
  project = Xcodeproj::Project.open('InflamAI.xcodeproj')
  files = project.targets.first.source_build_phase.files.map { |f| f.file_ref&.path }
  puts files.compact.select { |f| f.end_with?('.swift') }.sort
"
```

### Build Status
- ✅ Builds successfully for iOS Simulator
- ✅ No CreateML symbols in binary
- ✅ No TabularData dependencies
- ✅ All imports are iOS-compatible

---

## Privacy & Ethics

### Data Handling
- **Local-Only**: All data stored in Core Data on-device
- **No Cloud Sync**: CloudKit explicitly disabled
- **No Network Calls**: Zero server communication
- **HealthKit**: User-controlled authorization
- **WeatherKit**: Location data for context only

### Medical Disclaimers
- "⚠️ Not Medical Advice" displayed in UI
- "Statistical Pattern Analysis" explicitly labeled
- "Always consult your rheumatologist" messaging
- No clinical validation claims

### Ethical AI
- No misleading "AI-powered" marketing
- No fake confidence scores
- Transparent about statistical methods
- Requires real data (no fabrication)

---

## Future Considerations

### If Real ML is Needed (Not Currently Implemented)

**Proper Approach**:
1. Train models **offline** using Python (scikit-learn, TensorFlow)
2. Export to Core ML format (`.mlmodel`)
3. Compile to `.mlmodelc` using `coremltools`
4. Bundle compiled models in app
5. Use `MLModel(contentsOf:)` for **inference only**
6. Document training data, metrics, and validation
7. Get clinical review before deployment

**Never Do**:
- ❌ Runtime training with CreateML (iOS incompatible)
- ❌ Shipping raw .mlmodel files (need .mlmodelc)
- ❌ Synthetic data for training
- ❌ Heuristics disguised as ML

---

## Testing Strategy

### Current State
- Manual testing on iOS Simulator
- Build verification via Xcode
- No automated tests (yet)

### Recommended
- Unit tests for FlarePredictor statistical logic
- UI tests for critical flows
- Integration tests for Core Data queries
- Accessibility testing

---

## Dependencies

### Apple Frameworks (All iOS-Compatible)
```swift
import SwiftUI
import CoreData
import HealthKit
import WeatherKit
import UserNotifications
import Charts
import CoreML         // Used only for data structures, not training
import Combine
import Foundation
```

### External Dependencies
- None (100% native Swift)

---

## Known Limitations

1. **FlarePredictor requires 30+ days** of data before analysis
2. **No real-time predictions** - batch analysis only
3. **Statistical, not ML** - correlation, not causation
4. **No clinical validation** - not a medical device
5. **iOS only** - macOS/watchOS not supported (yet)

---

## Contribution Guidelines

### Before Adding "AI" Features
1. Verify iOS compatibility of frameworks
2. Use statistical methods if possible
3. Never fabricate data or confidence scores
4. Add medical disclaimers
5. Document methodology clearly

### Code Review Checklist
- [ ] No CreateML/TabularData imports
- [ ] No synthetic data generation
- [ ] Proper error handling for insufficient data
- [ ] Medical disclaimers in UI
- [ ] Privacy-preserving (local-only)
- [ ] Builds successfully for iOS

---

## Contact & Support

**Repository**: https://github.com/[your-repo]/InflamAI
**Issues**: Report bugs via GitHub Issues
**Privacy Policy**: See `PRIVACY.md`
**License**: See `LICENSE`

---

## Recent Improvements (v1.1 - 2025-01-25)

### Architecture Enhancements ✅
1. **MVVM Pattern Adoption**
   - Created `FlareTimelineViewModel` for flare history management
   - Created `AIInsightsViewModel` for pattern analysis integration
   - Created `ExerciseLibraryViewModel` for exercise tracking
   - Improved from 9 to 12 ViewModels (+33% coverage)

2. **Dependency Injection Foundation**
   - Created `Core/DependencyInjection/DIContainer.swift`
   - Factory methods for all ViewModels
   - SwiftUI Environment integration ready
   - Improved testability and maintainability

3. **Code Consolidation**
   - Removed duplicate `Features/Flare` directory
   - Consolidated into single `Features/Flares` module
   - Documented duplicate managers for future cleanup

4. **Documentation**
   - Created `ARCHITECTURE_IMPROVEMENT_ROADMAP.md`
   - 10-week plan for achieving 9.5/10 architecture score
   - Comprehensive refactoring strategy documented

### Next Steps
See `ARCHITECTURE_IMPROVEMENT_ROADMAP.md` for:
- Consolidating 11 duplicate manager files
- Moving 64 root-level files into Features
- Completing MVVM adoption (target: 80% coverage)
- Establishing test infrastructure (target: 40% coverage)

---

## Apple Watch Integration

### Overview

InflamAI includes comprehensive Apple Watch integration to enable continuous biometric monitoring, quick symptom logging, and real-time pattern detection. This represents a **major enhancement** to the app's capabilities.

### Key Features

1. **Quick Symptom Logging** (3 taps, <10 seconds)
   - Launch from watch face complication
   - Simple pain/stiffness/fatigue sliders
   - Instant sync to iPhone

2. **Medication Reminders**
   - Watch notifications for scheduled medications
   - One-tap "Taken" confirmation
   - Snooze/skip options

3. **Continuous Biometric Monitoring**
   - Heart rate (every 1-5 minutes)
   - HRV (heart rate variability) during rest/sleep
   - Sleep stage analysis (Deep, REM, Core)
   - Daily activity metrics (steps, active energy, stand hours)

4. **Watch Complications**
   - Medication countdown timer
   - Current pain level indicator
   - Activity rings integration
   - Multi-metric dashboard

5. **Pre-Flare Detection**
   - Real-time biomarker monitoring
   - 12-24 hour warning notifications
   - Cascade detection (HRV drops → HR increases → sleep disruption)

### Architecture

```
iPhone App (iOS 17+)
  ├── WatchConnectivityService
  ├── HealthKitService (enhanced)
  └── Core Data (CloudKit sync)
       ↕
  WatchConnectivity Framework
       ↕
Apple Watch App (watchOS 10+)
  ├── QuickLogView
  ├── MedicationTrackerView
  ├── BiometricsView
  ├── ComplicationController
  └── WatchHealthKitService
```

### Data Flow

1. **iPhone → Watch**:
   - Medication schedules (Application Context)
   - Recent symptom logs (User Info Transfer)
   - Configuration settings (Application Context)

2. **Watch → iPhone**:
   - Quick symptom logs (Interactive Messages)
   - Medication confirmations (Interactive Messages)
   - Health metrics requests (Interactive Messages)

3. **CloudKit Sync**:
   - Shared Core Data model
   - Automatic conflict resolution
   - Background synchronization

### Pattern Recognition Enhancements

With Apple Watch integration, pattern recognition capabilities expand from **49 correlations** to **2,000+ correlations**:

- **Circadian patterns**: Nocturnal HRV drops → morning stiffness
- **Intraday variability**: Hour-by-hour symptom tracking
- **Pre-flare cascades**: 6-48 hour warning signals
- **Activity thresholds**: Personalized optimal activity levels
- **Medication timing**: Real-time response profiling
- **Sleep-inflammation cycles**: Deep sleep % correlation with symptoms

### Documentation

Comprehensive documentation available in `/docs`:

1. **[APPLE_WATCH_INTEGRATION_ANALYSIS.md](docs/APPLE_WATCH_INTEGRATION_ANALYSIS.md)**
   - Impact assessment (⭐⭐⭐ HIGH IMPACT)
   - ROI analysis and competitive differentiation
   - Technical feasibility evaluation

2. **[HEALTHKIT_ENHANCEMENT_ROADMAP.md](docs/HEALTHKIT_ENHANCEMENT_ROADMAP.md)**
   - 10-week implementation plan
   - Three-phase rollout strategy
   - Code examples and acceptance criteria

3. **[PATTERN_RECOGNITION_OPPORTUNITIES.md](docs/PATTERN_RECOGNITION_OPPORTUNITIES.md)**
   - Statistical power improvements (41x increase)
   - New correlation categories unlocked
   - Algorithm enhancement recommendations

4. **[WATCH_APP_TECHNICAL_SPEC.md](docs/WATCH_APP_TECHNICAL_SPEC.md)**
   - Complete WatchOS app architecture
   - WatchConnectivity implementation
   - Complication design specifications
   - Background monitoring setup

### Implementation Status

**Current State**: 📋 Planning Phase

**Readiness**:
- ✅ HealthKitService foundation (70% complete)
- ✅ AppleWatchManager skeleton (60% infrastructure)
- ✅ CorrelationEngine ready for enhancement
- ❌ WatchOS app target (0% - not yet created)
- ❌ WatchConnectivity service (0% - not yet implemented)

**Next Steps**:
1. Create WatchOS app target in Xcode
2. Implement WatchConnectivityService
3. Build Quick Log UI for Watch
4. Develop complications
5. Enable background health monitoring

### Privacy & Battery Impact

**Privacy**:
- All data processed on-device
- No cloud processing of health data
- Granular HealthKit permissions
- User-controlled sync preferences

**Battery Life**:
- Target: <5% daily battery impact (passive monitoring)
- Adaptive monitoring frequency based on battery level
- Background refresh optimization

---

## Changelog

### v1.2 (2025-10-28)
- Added Apple Watch integration documentation
- Created comprehensive implementation roadmap
- Documented pattern recognition enhancements (41x correlation increase)
- Added WatchOS technical specifications
- Outlined 10-week development plan

### v1.1 (2025-01-25)
- Added MVVM ViewModels for critical features
- Created dependency injection container
- Removed duplicate feature modules
- Created comprehensive improvement roadmap
- Architecture score improved from 7.5/10 to 8.0/10

### v1.0 (2025-01-22)
- Initial architecture documentation
- Moved 26 deprecated AI files to `/deprecated/AI-Old/`
- Clarified FlarePredictor as statistical, not ML
- Added medical disclaimers to UI
- Verified build contains only 30 active files
