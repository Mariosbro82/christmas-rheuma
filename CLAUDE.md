# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**InflamAI** (InflamAI) is a production-grade iOS application for Ankylosing Spondylitis (AS) management, built to Fortune 100 quality standards with privacy-first architecture and WCAG AA accessibility compliance.

- **Platform**: iOS 17.0+, Swift 5.9+
- **Architecture**: MVVM + Feature-based modular design
- **Persistence**: Core Data (8 entities) with optional CloudKit sync
- **Privacy**: 100% on-device processing, zero third-party SDKs
- **Quality**: 190K lines of Swift, medical-grade calculators, clinically validated

## Build & Development Commands

### Opening the Project
```bash
cd /Users/fabianharnisch/Documents/Rheuma-app
open InflamAI.xcodeproj
```

**Note**: The project.pbxproj file has been moved to backups. To restore:
```bash
cp InflamAI.xcodeproj/project-backups/project.pbxproj InflamAI.xcodeproj/project.pbxproj
```

### Build Commands
```bash
# Clean build
Cmd + Shift + K (in Xcode)

# Build project
Cmd + B (in Xcode)

# Run on simulator
Cmd + R (in Xcode)
# Recommended: iPhone 15 Pro (iOS 17.0+)

# Run on device (requires code signing for HealthKit)
# Select device in Xcode, then Cmd + R
```

### Testing
```bash
# Run unit tests (in Xcode)
Cmd + U

# Validate calculators (in DEBUG builds)
# These tests run automatically when BASDAICalculator or ASDAICalculator are initialized
```

## High-Level Architecture

### Core Architectural Pattern
**MVVM + Feature-based Modular Architecture** with dependency injection. Currently at 8.0/10 architecture score, targeting 9.5/10 (see ARCHITECTURE_IMPROVEMENT_ROADMAP.md).

### Project Structure
```
InflamAI/
├── InflamAIApp.swift           # Entry point with biometric lock
├── Core/                           # Shared infrastructure (36 subdirectories)
│   ├── Persistence/               # InflamAIPersistenceController
│   ├── Services/                  # HealthKit, WeatherKit, WatchConnectivity
│   ├── Security/                  # BiometricAuth, Encryption, Keychain
│   ├── Utilities/                 # BASDAI/ASDAS calculators, CorrelationEngine
│   ├── ML/                        # FlarePredictor (statistical, NOT ML)
│   ├── DependencyInjection/       # DIContainer for service locator pattern
│   ├── Export/                    # PDFExportService
│   └── Components/                # Reusable UI components
├── Features/                       # 15 self-contained feature modules
│   ├── Home/                      # Main dashboard
│   ├── BodyMap/                   # 47-region interactive anatomy
│   ├── CheckIn/                   # Daily BASDAI questionnaire
│   ├── Medication/                # Medication tracking & adherence
│   ├── Exercise/                  # 52-exercise library
│   ├── Trends/                    # Analytics with Swift Charts
│   ├── Flares/                    # Flare timeline & tracking
│   ├── AI/                        # Pattern analysis (statistical)
│   ├── QuickCapture/              # JointTap SOS emergency logging
│   ├── Coach/                     # Personalized routine generator
│   └── Onboarding/                # 12-page onboarding flow
├── InflamAI.xcdatamodeld/     # Core Data model (8 entities)
└── deprecated/                     # Legacy files (not compiled)
```

### Key Architectural Decisions

**1. Privacy-First Design**
- Zero third-party SDKs (no Firebase, Analytics, Facebook)
- All processing on-device (no cloud inference)
- Optional CloudKit sync (user-controlled)
- Biometric lock (Face ID/Touch ID) auto-locks on background
- GDPR-compliant data deletion

**2. Pattern Analysis is Statistical, NOT Machine Learning**
- File: `Core/ML/FlarePredictor.swift`
- Uses Pearson correlation + lag analysis
- Requires minimum 30 days of data
- NO CreateML, NO TabularData, NO synthetic data
- Transparent methodology displayed to users

**3. Medical-Grade Calculators**
- `Core/Utilities/BASDAICalculator.swift` - Validated against medical literature
- `Core/Utilities/ASDAICalculator.swift` - ASDAS-CRP formula
- Unit tests verify clinical accuracy
- Medical disclaimers displayed in UI

**4. MVVM Adoption (In Progress)**
- Current: 12 ViewModels (~60% coverage)
- Target: 80%+ coverage for business logic
- Use `@MainActor` for ViewModels
- Dependency injection via DIContainer
- See `ARCHITECTURE_IMPROVEMENT_ROADMAP.md` for migration plan

## Core Data Model

### 8 Primary Entities

**SymptomLog** - Daily symptom records
- Relationships: BodyRegionLog (1:N), ContextSnapshot (1:1)
- Key fields: basdaiScore, fatigueLevel, moodScore, morningStiffnessMinutes

**BodyRegionLog** - 47 anatomical regions (C1-L5 spine + peripheral joints)
- Fields: regionID, painLevel (0-10), stiffnessMinutes, swelling, warmth

**ContextSnapshot** - Environmental/biometric data
- Weather: barometricPressure, pressureChange12h, humidity, temperature
- Biometrics: hrvValue, restingHeartRate, stepCount, sleepEfficiency

**Medication** - Prescription records
- Relationships: MedicationLog (1:N)
- Fields: name, dosage, frequency, reminderTimes

**MedicationLog** - Adherence tracking

**ExerciseSession** - Workout history

**FlareEvent** - Acute flare episodes

**UserProfile** - Settings & preferences (singleton)

### Persistence Best Practices
```swift
// Use shared controller
let context = InflamAIPersistenceController.shared.container.viewContext

// Save with error handling
do {
    try context.save()
} catch {
    print("Core Data save error: \(error)")
}

// Background operations
let backgroundContext = InflamAIPersistenceController.shared.container.newBackgroundContext()
backgroundContext.perform {
    // Heavy queries here
}
```

## Critical Component Details

### 1. Interactive Body Map (47 Regions)
**Location**: `Features/BodyMap/`

Anatomically accurate body diagram with:
- Spine: C1-C7 (cervical), T1-T12 (thoracic), L1-L5 (lumbar), SI joints
- Peripheral: shoulders, elbows, wrists, hands, hips, knees, ankles, feet (bilateral)
- Real-time heatmap overlay (7/30/90-day averages)
- VoiceOver support with anatomical names
- 44pt minimum hit targets for accessibility

### 2. BASDAI Calculator
**Location**: `Core/Utilities/BASDAICalculator.swift`

```swift
// Formula (clinically validated):
// BASDAI = (Q1 + Q2 + Q3 + Q4 + ((Q5 + Q6scaled) / 2)) / 5

// Interpretation:
// 0-2: Remission (Green)
// 2-4: Low Activity (Yellow)
// 4-6: Moderate Activity (Orange)
// 6+: High Activity (Red)
```

**Critical**: Never modify formula without clinical validation. Unit tests verify against published medical literature.

### 3. Pattern Analysis / FlarePredictor
**Location**: `Core/ML/FlarePredictor.swift`

**What it does**:
- Pearson correlation analysis (NOT machine learning)
- Analyzes weather, biometrics, activity vs. symptoms
- Identifies triggers with statistical significance (p < 0.05, |r| > 0.4)
- Lag analysis (0h, 12h, 24h offsets)

**What it does NOT do**:
- ❌ Machine learning training
- ❌ Synthetic data generation
- ❌ Fabricated predictions
- ❌ Cloud inference

**Requirements**:
- Minimum 30 days of data
- Proper error handling for insufficient data
- Medical disclaimers in UI

### 4. Apple Watch Integration (Planned)
**Status**: Documentation complete, implementation pending
**Roadmap**: `docs/HEALTHKIT_ENHANCEMENT_ROADMAP.md` (10-week plan)

Will enable:
- Continuous biometric monitoring (HRV, HR, sleep stages)
- Quick symptom logging (3 taps)
- Pre-flare detection (12-24h warning)
- Watch complications
- 1,000x more data points for pattern analysis

### 5. HealthKit Integration
**Location**: `Core/Services/HealthKitService.swift`

**Reads** (read-only, no writes):
- Sleep duration & efficiency
- HRV (SDNN) - for stress/inflammation markers
- Resting heart rate
- Step count

**Privacy guarantees**:
- User-controlled permissions
- On-device only
- Transparent Info.plist strings

### 6. WeatherKit Integration
**Location**: `Core/Services/WeatherKitService.swift`

**Critical AS trigger**: Rapid barometric pressure drops (>5 mmHg in 12 hours)
- Caches data to reduce API calls
- Location used only for weather context
- Never stored or shared

## Code Quality Standards

### Swift Best Practices
```swift
// ✅ DO: Optional handling
guard let value = optionalValue else { return }

// ❌ DON'T: Force unwrap
let value = optionalValue!  // NEVER do this

// ✅ DO: Weak self in closures
someAsync { [weak self] in
    guard let self else { return }
}

// ✅ DO: Proper error handling
do {
    try riskyOperation()
} catch {
    print("Error: \(error.localizedDescription)")
}
```

### SwiftUI Patterns
```swift
// Use @StateObject for ViewModel initialization
@StateObject private var viewModel = FeatureViewModel()

// Use @ObservedObject when passed from parent
@ObservedObject var viewModel: FeatureViewModel

// Inject Core Data context
.environment(\.managedObjectContext, persistenceController.container.viewContext)

// Use @MainActor for ViewModels
@MainActor
class FeatureViewModel: ObservableObject {
    @Published var data: [Item] = []
}
```

### Git Commit Conventions
```bash
# Use conventional commits
feat: Add medication adherence calendar view
fix: Resolve BASDAI calculation rounding error
docs: Update CLAUDE.md with build instructions
refactor: Extract body map logic to ViewModel
test: Add unit tests for CorrelationEngine
```

## Common Development Tasks

### Adding a New Feature
1. Create feature directory: `Features/NewFeature/`
2. Create files:
   - `NewFeatureView.swift` - SwiftUI view
   - `NewFeatureViewModel.swift` - Business logic (if needed)
   - `Models/` - Data models (if needed)
3. Add to `DIContainer.swift` if using DI
4. Update navigation in `InflamAIApp.swift`
5. See `HOW_TO_ADD_FEATURES.md` for detailed guide

### Creating a New ViewModel
```swift
import Foundation
import Combine
import CoreData

@MainActor
class NewFeatureViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var items: [Item] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    // MARK: - Dependencies
    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization
    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    // MARK: - Public Methods
    func loadData() async {
        isLoading = true
        defer { isLoading = false }

        // Fetch from Core Data
        let context = persistenceController.container.viewContext
        // ... implementation
    }
}
```

### Adding a New Core Data Entity
1. Open `InflamAI.xcdatamodeld` in Xcode
2. Add entity with attributes
3. Set relationships
4. Generate NSManagedObject subclass (or use @FetchRequest)
5. Create lightweight migration if modifying existing schema
6. Test thoroughly before deploying

### Running Validation Tests
```swift
#if DEBUG
// In BASDAICalculator.swift:
BASDAICalculator.runValidationTests()
// Validates: Medical literature examples, edge cases, formula accuracy

// In CorrelationEngine.swift:
CorrelationEngine().runValidationTests()
// Validates: Pearson coefficient calculation, p-value accuracy
#endif
```

## Important Constraints & Limitations

### Medical/Clinical
- **Not a medical device** - Always display "⚠️ Not Medical Advice" disclaimers
- **No clinical claims** - Use "statistical pattern analysis", never "AI prediction"
- **Consult physicians** - Encourage users to discuss findings with rheumatologist
- **No treatment recommendations** - Display patterns only, never prescribe

### Technical
- **iOS 17.0+ only** - Cannot support iOS 16 (relies on Swift Charts, async/await)
- **HealthKit requires device** - Simulator has limited HealthKit data
- **WeatherKit requires Apple ID** - Needs proper entitlements
- **FlarePredictor requires 30+ days** - Gracefully handle insufficient data

### Architecture (Current State)
- **64 root-level files** - In process of moving to Features (see roadmap)
- **11 duplicate managers** - Consolidation pending (CloudSyncManager, ThemeManager, SecurityManager)
- **Test coverage minimal** - Target 40%+ coverage (see roadmap)
- **MVVM adoption incomplete** - Currently 60%, targeting 80%

## Known Issues & Workarounds

### Issue: project.pbxproj Missing
**Symptom**: Xcode says "missing its project.pbxproj file"
**Workaround**:
```bash
cp InflamAI.xcodeproj/project-backups/project.pbxproj InflamAI.xcodeproj/project.pbxproj
```

### Issue: Build Errors After File Moves
**Symptom**: "Cannot find 'ClassName' in scope"
**Solution**:
1. Clean build folder: `Cmd + Shift + K`
2. Delete derived data: `Cmd + Shift + Option + K`
3. Rebuild: `Cmd + B`
4. Verify imports reference correct module

### Issue: HealthKit Authorization Not Showing
**Symptom**: Permission prompts don't appear
**Solution**:
1. Must run on physical device (not simulator)
2. Verify Info.plist has HealthKit usage strings
3. Check entitlements file includes HealthKit

## File Locations Reference

### Critical Files
- **App Entry**: `InflamAIApp.swift`
- **Core Data Model**: `InflamAI.xcdatamodeld`
- **Persistence**: `Core/Persistence/InflamAIPersistenceController.swift`
- **BASDAI Calculator**: `Core/Utilities/BASDAICalculator.swift`
- **Pattern Analysis**: `Core/ML/FlarePredictor.swift`
- **DI Container**: `Core/DependencyInjection/DIContainer.swift`

### Documentation Files
- **Architecture Overview**: `ARCHITECTURE.md`
- **Improvement Roadmap**: `ARCHITECTURE_IMPROVEMENT_ROADMAP.md`
- **Feature Guide**: `HOW_TO_ADD_FEATURES.md`
- **README**: `README.md` (comprehensive feature overview)
- **Watch Integration**: `docs/APPLE_WATCH_INTEGRATION_ANALYSIS.md`
- **HealthKit Roadmap**: `docs/HEALTHKIT_ENHANCEMENT_ROADMAP.md`

### Configuration Files
- **Entitlements**: `InflamAI/InflamAI.entitlements`
- **Info.plist**: `InflamAI/Info.plist` (contains privacy strings)

## When Working on This Codebase

### DO:
- ✅ Follow MVVM pattern for new features
- ✅ Use dependency injection via DIContainer
- ✅ Add unit tests for business logic
- ✅ Include medical disclaimers for health-related features
- ✅ Verify accessibility (VoiceOver, Dynamic Type)
- ✅ Use proper error handling (no force unwraps)
- ✅ Respect privacy-first architecture (no third-party SDKs)
- ✅ Update documentation when making architectural changes

### DON'T:
- ❌ Add third-party analytics or tracking SDKs
- ❌ Modify BASDAI/ASDAS formulas without clinical validation
- ❌ Use force unwrapping (use guard/if let)
- ❌ Import CreateML or TabularData (iOS incompatible)
- ❌ Generate synthetic health data
- ❌ Make clinical claims or medical recommendations
- ❌ Skip accessibility testing
- ❌ Commit HealthKit data to version control

### Before Committing:
1. Build succeeds without warnings
2. Run unit tests (if available)
3. Test on simulator AND device (if HealthKit changes)
4. Verify accessibility with VoiceOver
5. Update relevant documentation
6. Follow conventional commit format

## Architecture Improvement Roadmap

**Current Score**: 8.0/10
**Target Score**: 9.5/10
**Timeline**: 10 weeks

See `ARCHITECTURE_IMPROVEMENT_ROADMAP.md` for detailed plan:

**Week 1**: Consolidate 11 duplicate manager files
**Weeks 2-3**: Move 64 root-level files to Features
**Week 4**: Reorganize MotherMode module
**Weeks 5-7**: Complete MVVM adoption (80%+ coverage)
**Weeks 8-9**: Establish test infrastructure (40%+ coverage)
**Week 10**: Documentation & polish

## Accessibility Requirements (WCAG AA)

All features must meet:
- **VoiceOver**: All interactive elements labeled with hints
- **Dynamic Type**: Support up to XXXL without clipping
- **Contrast**: 4.5:1 minimum for text, 3:1 for UI components
- **Hit Targets**: 44×44pt minimum for all buttons/tappable areas
- **Haptics**: Provide feedback for important milestones
- **Reduce Motion**: Respect accessibility preferences

## Privacy & Security Requirements

Every feature must:
- Process data on-device only (no cloud inference)
- Store sensitive data in Keychain (not UserDefaults)
- Encrypt Core Data if containing PHI
- Request minimal permissions
- Provide clear privacy strings in Info.plist
- Support biometric lock
- Enable GDPR-compliant data deletion
- Never log sensitive health data

## Additional Resources

- **Apple Watch Implementation**: See `docs/WATCH_APP_TECHNICAL_SPEC.md`
- **Pattern Recognition Enhancements**: See `docs/PATTERN_RECOGNITION_OPPORTUNITIES.md`
- **HealthKit Best Practices**: See `Core/Services/HealthKitService.swift` inline docs
- **Core Data Migration**: See `InflamAIPersistenceController.swift` comments

---

**Last Updated**: 2025-11-21
**Architecture Version**: 1.2
**For Questions**: Refer to inline code documentation and `ARCHITECTURE.md`
- only work in af older named andriod-app,if your project task involves a) about generating any code matched to anriod ecosystem or b) if you are prompted to plan / think /respond regarding andriod app