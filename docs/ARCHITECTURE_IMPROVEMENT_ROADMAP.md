# Architecture Improvement Roadmap
## InflamAI (InflamAI)

**Created**: 2025-01-25
**Status**: In Progress
**Priority**: High

---

## Executive Summary

This document outlines the architectural improvements needed to bring InflamAI from its current **7.5/10 architecture score** to a target **9.5/10** through systematic refactoring, consolidation, and standardization.

### Current State
- ✅ **Strong foundation**: Feature-based organization, comprehensive security
- ✅ **Well-documented**: Excellent ARCHITECTURE.md
- ⚠️ **Inconsistent patterns**: Mixed MVVM adoption, duplicate managers
- 🔴 **Organization issues**: 64 root-level files, scattered code

### Target State
- 100% feature-based organization
- 80%+ MVVM coverage for business logic
- Zero code duplication
- Comprehensive dependency injection
- 40%+ test coverage

---

## Completed Improvements ✅

### Phase 1: Initial Cleanup (Completed 2025-01-25)

1. **Removed Duplicate Feature Module**
   - ✅ Deleted empty `Features/Flare` directory
   - ✅ Consolidated into `Features/Flares`
   - Impact: Eliminated confusion, cleaner structure

2. **Created Critical ViewModels**
   - ✅ `FlareTimelineViewModel` - Flare history management
   - ✅ `AIInsightsViewModel` - Pattern analysis with FlarePredictor integration
   - ✅ `ExerciseLibraryViewModel` - Exercise tracking and library
   - Impact: +33% MVVM coverage for key features

3. **Established DI Foundation**
   - ✅ Created `Core/DependencyInjection/DIContainer.swift`
   - ✅ Factory methods for all ViewModels
   - ✅ SwiftUI Environment integration
   - Impact: Improved testability, standardized initialization

---

## Remaining Work (Priority Order)

### 🔥 Priority 1: Consolidate Duplicate Managers (Week 1)

**Problem**: Multiple instances of critical managers exist:
- CloudSyncManager: 4 copies
- ThemeManager: 3 copies
- SecurityManager: 4 copies

**Solution**:
1. Audit each manager to identify canonical version
2. Update all imports to use canonical version
3. Delete duplicate files
4. Update Xcode project references

**Files to Consolidate**:

```bash
# CloudSyncManager - KEEP Core/Cloud version (26KB, most complete)
✅ KEEP: InflamAI/Core/Cloud/CloudSyncManager.swift
❌ DELETE: InflamAI/Core/CloudSyncManager.swift
❌ DELETE: InflamAI/Managers/CloudSyncManager.swift
❌ DELETE: InflamAI/CloudSyncManager.swift

# ThemeManager - KEEP Core/Theme version
✅ KEEP: InflamAI/Core/Theme/ThemeManager.swift
❌ DELETE: InflamAI/Core/UI/ThemeManager.swift
❌ DELETE: InflamAI/Managers/ThemeManager.swift

# SecurityManager - KEEP Core/Security version
✅ KEEP: InflamAI/Core/Security/SecurityManager.swift
❌ DELETE: InflamAI/Security/SecurityManager.swift
❌ DELETE: InflamAI/Managers/SecurityManager.swift
❌ DELETE: InflamAI/SecurityManager.swift
```

**Validation**:
```bash
# After consolidation, verify:
find InflamAI -name "CloudSyncManager.swift" | wc -l  # Should be 1
find InflamAI -name "ThemeManager.swift" | wc -l      # Should be 1
find InflamAI -name "SecurityManager.swift" | wc -l   # Should be 1

# Ensure project builds
xcodebuild -scheme InflamAI clean build
```

**Effort**: 2-3 days
**Risk**: Medium (requires careful import updates)

---

### ⭐ Priority 2: Organize Root-Level Files (Week 2-3)

**Problem**: 64 Swift files at project root break feature isolation

**Solution**: Move files into appropriate feature directories

#### 2.1 Create New Feature Modules

```bash
mkdir -p InflamAI/Features/Dashboard
mkdir -p InflamAI/Features/Journal
mkdir -p InflamAI/Features/Analytics
mkdir -p InflamAI/Features/Social
mkdir -p InflamAI/Features/BASSDAI
```

#### 2.2 File Migration Plan

**Dashboard Feature** (5 files):
```
Move: DashboardView.swift → Features/Dashboard/
Move: TraeHomeView.swift → Features/Dashboard/
Create: Features/Dashboard/DashboardViewModel.swift
```

**Journal Feature** (3 files):
```
Move: JournalView.swift → Features/Journal/
Move: JournalHistoryView.swift → Features/Journal/
Create: Features/Journal/JournalViewModel.swift
```

**Analytics Feature** (4 files):
```
Move: AnalyticsView.swift → Features/Analytics/
Move: PainAnalyticsCharts.swift → Features/Analytics/
Move: PersonalizedInsightsView.swift → Features/Analytics/
Create: Features/Analytics/AnalyticsViewModel.swift
```

**BASSDAI Feature** (2 files):
```
Move: BASSDAIView.swift → Features/BASSDAI/
Move: BASSDAIPracticeView.swift → Features/BASSDAI/
Create: Features/BASSDAI/BASSDAIViewModel.swift
```

**Social Feature** (1 file):
```
Move: SocialSupportNetwork.swift → Features/Social/
Create: Features/Social/SocialViewModel.swift
```

**Medication Feature** (additions):
```
Move: MedicationCard.swift → Features/Medication/
Move: TodayMedicationCard.swift → Features/Medication/
Move: MedicationDetailView.swift → Features/Medication/
Move: MedicationView.swift → Features/Medication/
```

**Core Utilities** (move to Core):
```
Move: PerformanceManager.swift → Core/Monitoring/
Move: AdaptiveUIManager.swift → Core/UI/
Move: EmergencyResponseSystem.swift → Core/Emergency/
Move: MedicalPDFGenerator.swift → Core/Export/
```

**Effort**: 3-4 days
**Risk**: Low (mostly file moves, update imports)

---

### 🏗️ Priority 3: Move MotherMode to Features (Week 4)

**Problem**: 20-file feature set exists outside Features structure

**Solution**:
```bash
mkdir -p InflamAI/Features/MotherMode/ViewModels
mkdir -p InflamAI/Features/MotherMode/Views
mkdir -p InflamAI/Features/MotherMode/Services
mkdir -p InflamAI/Features/MotherMode/Models

# Move files
mv InflamAI/MotherMode/*ViewModel.swift → Features/MotherMode/ViewModels/
mv InflamAI/MotherMode/*View.swift → Features/MotherMode/Views/
mv InflamAI/MotherMode/*Manager.swift → Features/MotherMode/Services/
mv InflamAI/MotherMode/*Models.swift → Features/MotherMode/Models/
```

**Resulting Structure**:
```
Features/MotherMode/
├── ViewModels/
│   ├── MotherModeViewModel.swift
│   ├── DailyCheckInViewModel.swift
│   ├── QuickCheckInViewModel.swift
│   └── RoutinePlayerViewModel.swift
├── Views/
│   ├── CaregiverCardView.swift
│   ├── QuickCheckInView.swift
│   ├── DailyASCheckInView.swift
│   └── ... (12 more views)
├── Services/
│   ├── VoiceNoteManager.swift
│   ├── MotherModeNotificationScheduler.swift
│   └── ...
└── Models/
    ├── MotherModeModels.swift
    ├── SymptomEntry.swift
    └── ExerciseRoutine.swift
```

**Effort**: 1 week
**Risk**: Low (self-contained feature)

---

### 📱 Priority 4: Complete MVVM Adoption (Weeks 5-7)

**Current**: 9 ViewModels (~3.5% coverage)
**Target**: 25+ ViewModels (80%+ coverage for business logic views)

#### Week 5: Secondary Features

Create ViewModels for:
1. **QuickCapture Feature**
   ```swift
   Features/QuickCapture/JointTapSOSViewModel.swift
   ```

2. **Coach Feature**
   ```swift
   Features/Coach/CoachViewModel.swift
   ```

3. **Settings Feature**
   ```swift
   Features/Settings/SettingsViewModel.swift
   Features/Settings/NotificationSettingsViewModel.swift
   ```

#### Week 6: Dashboard & Analytics

Create ViewModels for:
1. **Dashboard** (created in Priority 2)
2. **Analytics** (created in Priority 2)
3. **Journal** (created in Priority 2)

#### Week 7: Remaining Views

Create ViewModels for:
1. **BASSDAI** (created in Priority 2)
2. **Social** (created in Priority 2)
3. **Questionnaires**
   ```swift
   Features/Questionnaires/QuestionnaireViewModel.swift
   ```

**ViewModel Template**:
```swift
import Foundation
import Combine
import CoreData

@MainActor
class FeatureViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var state: LoadingState = .idle
    @Published var data: [DataType] = []
    @Published var errorMessage: String?

    // MARK: - Dependencies
    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization
    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    // MARK: - Public Methods
    func load() async {
        state = .loading
        // Implementation
    }

    // MARK: - Private Methods
    private func performBusinessLogic() {
        // Implementation
    }
}

enum LoadingState: Equatable {
    case idle, loading, loaded, error(Error)

    static func == (lhs: LoadingState, rhs: LoadingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.loading, .loading), (.loaded, .loaded):
            return true
        case (.error, .error):
            return true
        default:
            return false
        }
    }
}
```

**Effort**: 3 weeks
**Risk**: Low (additive, non-breaking changes)

---

### 🧪 Priority 5: Establish Testing Infrastructure (Week 8-9)

**Current**: No visible test coverage
**Target**: 40%+ coverage for core business logic

#### Week 8: Setup & Core Tests

1. **Create Test Targets**
   ```bash
   # In Xcode:
   # File > New > Target > Unit Testing Bundle
   # Name: InflamAITests

   # File > New > Target > UI Testing Bundle
   # Name: InflamAIUITests
   ```

2. **Core Infrastructure Tests**
   ```
   InflamAITests/
   ├── Core/
   │   ├── FlarePredictor​Tests.swift
   │   ├── BASDAICalculatorTests.swift
   │   ├── CorrelationEngineTests.swift
   │   └── TimeRangeTests.swift
   ├── ViewModels/
   │   ├── FlareTimelineViewModelTests.swift
   │   ├── AIInsightsViewModelTests.swift
   │   └── ExerciseLibraryViewModelTests.swift
   └── Mocks/
       └── MockPersistenceController.swift
   ```

3. **Mock Persistence Controller**
   ```swift
   class MockPersistenceController: InflamAIPersistenceController {
       var stubSymptomLogs: [SymptomLog] = []
       var stubFlareEvents: [FlareEvent] = []

       override func fetch<T>(_ request: NSFetchRequest<T>) throws -> [T] {
           // Return stub data
       }
   }
   ```

#### Week 9: Feature Tests

1. **ViewModel Tests**
   ```swift
   import XCTest
   @testable import InflamAI_Swift

   class FlareTimelineViewModelTests: XCTestCase {
       var sut: FlareTimelineViewModel!
       var mockPersistence: MockPersistenceController!

       override func setUp() {
           super.setUp()
           mockPersistence = MockPersistenceController()
           sut = FlareTimelineViewModel(persistenceController: mockPersistence)
       }

       func test_loadFlares_withData_updatesState() async {
           // Given
           mockPersistence.stubFlareEvents = [/* test data */]

           // When
           await sut.loadFlares()

           // Then
           XCTAssertEqual(sut.loadingState, .loaded)
           XCTAssertFalse(sut.flares.isEmpty)
       }
   }
   ```

2. **UI Tests**
   ```swift
   class FlareTimelineUITests: XCTestCase {
       func testFlareTimelineNavigation() {
           let app = XCUIApplication()
           app.launch()

           app.tabBars.buttons["Flares"].tap()
           XCTAssertTrue(app.navigationBars["Flare Timeline"].exists)
       }
   }
   ```

**Effort**: 2 weeks
**Risk**: Low (tests shouldn't break existing functionality)

---

### 📚 Priority 6: Documentation & ADRs (Ongoing)

#### Architecture Decision Records

Create ADRs for major decisions:

1. **ADR-001: MVVM Pattern Adoption**
   ```markdown
   # ADR-001: MVVM Pattern Adoption

   ## Status
   Accepted

   ## Context
   Currently only 3.5% ViewModel adoption. Business logic scattered in views.

   ## Decision
   Adopt MVVM for all features with business logic.

   ## Consequences
   - **Positive**: Better testability, separation of concerns, easier maintenance
   - **Negative**: More boilerplate, learning curve for team

   ## Implementation
   See Priority 4 roadmap.

   ## Date
   2025-01-25
   ```

2. **ADR-002: Dependency Injection via DIContainer**
3. **ADR-003: Core Data as Single Source of Truth**
4. **ADR-004: Privacy-First Architecture**

#### Update ARCHITECTURE.md

Add sections:
- Dependency Injection patterns
- ViewModel guidelines
- Testing strategy
- File organization standards

**Effort**: Ongoing, 1-2 hours per ADR
**Risk**: None (documentation only)

---

## Success Metrics

### Before Improvements
| Metric | Current | Target |
|--------|---------|--------|
| Root-level Swift files | 64 | 0 |
| MVVM Coverage | 3.5% (9/256) | 80%+ |
| Duplicate managers | 11 files | 0 |
| Test coverage | 0% | 40%+ |
| Architecture score | 7.5/10 | 9.5/10 |

### After Improvements (Target: 10 weeks)
| Metric | Target Value |
|--------|--------------|
| Root-level Swift files | **0** ✅ |
| MVVM Coverage | **80%+** ✅ |
| Duplicate managers | **0** ✅ |
| Test coverage | **40%+** ✅ |
| Architecture score | **9.5/10** ✅ |

---

## Progress Tracking

### Week-by-Week Checklist

**Week 1**: Consolidate Duplicates
- [ ] Audit all manager files
- [ ] Choose canonical versions
- [ ] Update imports
- [ ] Delete duplicates
- [ ] Verify build

**Week 2**: Root Files Part 1
- [ ] Create new feature directories
- [ ] Move Dashboard files
- [ ] Move Journal files
- [ ] Move Analytics files
- [ ] Create ViewModels

**Week 3**: Root Files Part 2
- [ ] Move BASSDAI files
- [ ] Move Social files
- [ ] Move Medication additions
- [ ] Move Core utilities
- [ ] Verify build

**Week 4**: MotherMode
- [ ] Create MotherMode subdirectories
- [ ] Move ViewModels
- [ ] Move Views
- [ ] Move Services/Models
- [ ] Update imports
- [ ] Verify build

**Week 5**: MVVM Part 1
- [ ] JointTapSOSViewModel
- [ ] CoachViewModel
- [ ] SettingsViewModel
- [ ] Write unit tests

**Week 6**: MVVM Part 2
- [ ] DashboardViewModel
- [ ] AnalyticsViewModel
- [ ] JournalViewModel
- [ ] Write unit tests

**Week 7**: MVVM Part 3
- [ ] BASSDAIViewModel
- [ ] SocialViewModel
- [ ] QuestionnaireViewModel
- [ ] Write unit tests

**Week 8**: Testing Part 1
- [ ] Create test targets
- [ ] MockPersistenceController
- [ ] Core utility tests
- [ ] Initial ViewModel tests

**Week 9**: Testing Part 2
- [ ] Complete ViewModel tests
- [ ] UI tests for critical flows
- [ ] Measure coverage

**Week 10**: Documentation & Polish
- [ ] Write ADRs
- [ ] Update ARCHITECTURE.md
- [ ] Create ViewModel guide
- [ ] Final review

---

## Risk Mitigation

### High-Risk Activities
1. **Consolidating duplicates** - Extensive testing required
2. **Moving many files** - Careful import management needed

### Mitigation Strategies
1. **Git branches**: Create feature branch for each priority
2. **Incremental commits**: Commit after each file move
3. **Build verification**: Run build after every 5-10 file changes
4. **Automated tests**: Prevent regressions
5. **Code review**: Review all changes before merging

### Rollback Plan
- Each priority is a separate Git branch
- Can revert individual priorities if issues arise
- Keep main branch stable throughout

---

## Resources Required

### Development Time
- **Total**: ~10 weeks (1 developer full-time)
- **Or**: ~20 weeks (1 developer part-time)

### Tools Needed
- Xcode (latest version)
- Git (for branch management)
- xcodeproj gem (for project file manipulation)

### Knowledge Required
- Swift & SwiftUI
- MVVM pattern
- Core Data
- XCTest framework
- Dependency Injection concepts

---

## Maintenance Plan

### After Completion

1. **Establish Standards**
   - Document ViewModel creation process
   - Create file organization guidelines
   - Set up pre-commit hooks for linting

2. **Regular Reviews**
   - Monthly architecture reviews
   - Quarterly refactoring sprints
   - Annual comprehensive audit

3. **Prevent Regression**
   - Code review checklist
   - Automated linting
   - Architecture decision log

---

## Next Steps

### Immediate (This Week)
1. ✅ Review this roadmap with team
2. ⬜ Create GitHub project for tracking
3. ⬜ Set up development branch
4. ⬜ Begin Priority 1 (Consolidate duplicates)

### This Month
1. Complete Priorities 1-2
2. Begin Priority 3 (MotherMode)
3. Write first ADRs

### This Quarter
1. Complete all priorities
2. Achieve 40%+ test coverage
3. Update all documentation
4. Architecture score: 9.5/10

---

**Last Updated**: 2025-01-25
**Next Review**: 2025-02-01
**Owner**: Development Team
**Status**: ✅ Roadmap Complete, Ready for Implementation
