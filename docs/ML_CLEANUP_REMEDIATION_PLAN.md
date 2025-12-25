# ML Placeholder Cleanup - Remediation Plan

## Overview

This plan addresses all critical, high, and medium priority issues identified in the code review of the ML placeholder data cleanup. Issues are organized by category and priority.

---

## Phase 1: Critical Issues (This Week)

### 1.1 🏗️ Architecture: Service Proliferation (CRITICAL)

**Problem**: 4 redundant prediction services with overlapping functionality
- `UnifiedNeuralEngine.swift` (92 features, singleton)
- `FlarePredictor.swift` (delegates + weather)
- `BinaryFlarePredictionService.swift` (45 features)
- `NeuralEngineMLService.swift` (~34 features)

**Tasks**:

- [ ] **Task 1.1.1**: Audit all UI code to identify which prediction services are actually called
  - Search for imports: `BinaryFlarePredictionService`, `NeuralEngineMLService`
  - Document all call sites
  - File: Create `docs/PREDICTION_SERVICE_AUDIT.md`

- [ ] **Task 1.1.2**: Verify `BinaryFlarePredictionService` is unused
  - If unused: Delete `InflamAI/Core/ML/NeuralEngine/BinaryFlarePredictionService.swift`
  - If used: Migrate callers to `UnifiedNeuralEngine`

- [ ] **Task 1.1.3**: Verify `NeuralEngineMLService` is unused
  - If unused: Delete `InflamAI/Core/ML/NeuralEngine/NeuralEngineMLService.swift`
  - If used: Migrate callers to `UnifiedNeuralEngine`

- [ ] **Task 1.1.4**: Update `FlarePredictor` to be a thin wrapper
  - Remove any parallel prediction logic
  - Keep only weather enhancement layer
  - Ensure it delegates ALL prediction to `UnifiedNeuralEngine.shared`

- [ ] **Task 1.1.5**: Update `CLAUDE.md` with architecture diagram
  ```
  ML Prediction Architecture:
  - PRIMARY: UnifiedNeuralEngine.shared.predict()
  - SECONDARY: FlarePredictor (weather enhancement wrapper only)
  - DELETED: BinaryFlarePredictionService, NeuralEngineMLService
  ```

**Acceptance Criteria**:
- Only 2 prediction-related files remain (UnifiedNeuralEngine, FlarePredictor)
- All UI code uses `UnifiedNeuralEngine.shared` or `FlarePredictor`
- No duplicate feature extraction logic

---

### 1.2 💻 Code Quality: Duplicate Code in SymptomLog+MLExtensions (HIGH)

**Problem**: Two complete implementations of `populateMLProperties()` at lines 16-348 and 350-572

**Tasks**:

- [ ] **Task 1.2.1**: Verify which implementation is actually called
  - Search codebase for `populateMLProperties` calls
  - Determine if both are needed for different use cases

- [ ] **Task 1.2.2**: Delete duplicate implementation
  - Remove lines 350-572 (the legacy/backup implementation)
  - Keep lines 16-348 (the comprehensive FIXED version)
  - File: `InflamAI/Core/ML/SymptomLog+MLExtensions.swift`

- [ ] **Task 1.2.3**: Clean up duplicate helper methods
  - Remove duplicate `calculatePainMetrics` (lines 375-409)
  - Remove duplicate `calculateJointCounts` (lines 411-429)
  - Remove duplicate `calculateMentalHealthMetrics` (lines 433-454)
  - Remove duplicate `applyIntelligentDefaults` (lines 458-483)
  - Remove duplicate `calculateDerivedScores` (lines 487-511)

**Acceptance Criteria**:
- Single implementation of each method
- File reduced by ~220 lines
- Build succeeds with no duplicate symbol errors

---

### 1.3 📝 Documentation: Breaking API Changes (CRITICAL)

**Problem**: `predict()` now returns `nil` when data quality insufficient - undocumented breaking change

**Tasks**:

- [ ] **Task 1.3.1**: Document `predict()` method comprehensively
  - File: `InflamAI/Core/ML/UnifiedNeuralEngine.swift:153`
  - Add documentation for:
    - When returns nil (5 cases)
    - Data requirements (15% availability, 20 features, 7 days)
    - How callers should handle nil
    - Migration from previous behavior

  ```swift
  /// Get flare risk prediction
  ///
  /// - Returns: `FlareRiskPrediction` if sufficient REAL data exists, `nil` otherwise
  ///
  /// ## Returns Nil When
  /// 1. Engine not ready (`engineStatus != .ready`)
  /// 2. Data availability < 15% (fewer than 13.8/92 features)
  /// 3. Data quality score < 0.15
  /// 4. Fewer than 20 non-zero features per day average
  /// 5. Prediction execution failed (check `errorMessage`)
  ///
  /// ## Migration from Previous Behavior
  /// **BREAKING CHANGE**: Previously returned predictions even with fake placeholder data.
  /// Now returns `nil` to avoid misleading predictions. Update UI to handle `nil` gracefully.
  ///
  /// ## Example
  /// ```swift
  /// if let prediction = await engine.predict() {
  ///     showPrediction(prediction)
  /// } else {
  ///     showDataCollectionPrompt(engine.errorMessage)
  /// }
  /// ```
  public func predict() async -> FlareRiskPrediction?
  ```

- [ ] **Task 1.3.2**: Document `FeatureAvailability` struct
  - File: `InflamAI/Core/ML/FeatureExtractor.swift:17-86`
  - Add documentation for:
    - Availability score interpretation (0-14%, 15-29%, 30-49%, etc.)
    - Category breakdown explanation
    - Data source flags meaning
    - Usage examples

- [ ] **Task 1.3.3**: Create migration guide
  - File: `docs/ML_PLACEHOLDER_CLEANUP_MIGRATION.md`
  - Document:
    - Summary of changes
    - Breaking changes with code examples
    - Feature value changes (fake → 0)
    - Data quality thresholds
    - User-facing impact
    - Gradual onboarding recommendations

**Acceptance Criteria**:
- All public ML APIs have comprehensive documentation
- Migration guide exists with code examples
- Breaking changes clearly marked

---

### 1.4 📝 Documentation: Medical Disclaimers (HIGH)

**Problem**: Only `FlarePredictor.swift` has medical disclaimer, other services lack them

**Tasks**:

- [ ] **Task 1.4.1**: Add standardized disclaimer to `UnifiedNeuralEngine`
  - File: `InflamAI/Core/ML/UnifiedNeuralEngine.swift:18`
  ```swift
  /// Unified Neural Engine - The single ML service for the entire app
  ///
  /// ⚠️ MEDICAL DISCLAIMER ⚠️
  /// This is NOT a medical device. Predictions are for informational purposes only.
  /// - Do NOT use for diagnosis or treatment decisions
  /// - Do NOT replace medical advice from your rheumatologist
  /// - All predictions are statistical estimates based on user-logged data
  /// - Always consult your healthcare provider before acting on predictions
  ```

- [ ] **Task 1.4.2**: Add disclaimer to `FeatureExtractor`
  - File: `InflamAI/Core/ML/FeatureExtractor.swift`

- [ ] **Task 1.4.3**: Add disclaimer to any remaining prediction services
  - Verify all prediction-related files have consistent disclaimers

**Acceptance Criteria**:
- All prediction services have identical medical disclaimers
- Disclaimers visible in generated documentation

---

## Phase 2: High Priority Issues (Next 2 Weeks)

### 2.1 🏗️ Architecture: FeatureAvailability Not Propagated (HIGH)

**Problem**: `FlarePredictor` doesn't check `FeatureAvailability` before delegating to `UnifiedNeuralEngine`

**Tasks**:

- [ ] **Task 2.1.1**: Add FeatureAvailability validation to FlarePredictor
  - File: `InflamAI/Core/ML/FlarePredictor.swift`
  - Before calling `UnifiedNeuralEngine.predict()`, validate data quality
  ```swift
  guard let coreMLPrediction = await neuralEngine.predict() else {
      // Log why prediction failed
      let result = await neuralEngine.featureExtractor.extract30DayFeaturesWithMetrics()
      if !result.isUsableForPrediction {
          print("⚠️ Insufficient data: \(result.availability.summary)")
      }
      return
  }
  ```

- [ ] **Task 2.1.2**: Ensure consistent data quality checks across all entry points

**Acceptance Criteria**:
- All prediction entry points validate data quality
- Failed predictions log the reason

---

### 2.2 💻 Code Quality: Magic Numbers (MEDIUM)

**Problem**: Magic numbers (92, 30, 0.15, 20) appear 15+ times without constants

**Tasks**:

- [ ] **Task 2.2.1**: Create MLModelConfiguration struct
  - File: `InflamAI/Core/ML/MLModelConfiguration.swift` (new file)
  ```swift
  struct MLModelConfiguration {
      struct FeatureConfig {
          static let totalFeatures = 92
          static let sequenceLength = 30  // days
      }

      struct DataQuality {
          /// Minimum 15% feature availability for predictions
          /// Rationale: Model performance degrades below 15 features
          static let minimumAvailability: Float = 0.15

          /// Minimum 20 non-zero features per day
          /// Rationale: Core symptoms + context = 20 minimum viable features
          static let minimumNonZeroFeatures = 20
      }

      struct MedicalThresholds {
          static let significantPressureDrop = -5.0  // mmHg
          static let prolongedStiffnessMinutes = 45
      }
  }
  ```

- [ ] **Task 2.2.2**: Replace all magic numbers with configuration constants
  - Search for: `92`, `30`, `0.15`, `20`, `-5`
  - Replace with `MLModelConfiguration` references

**Acceptance Criteria**:
- No magic numbers in ML code
- All thresholds documented with rationale
- Single source of truth for configuration

---

### 2.3 📝 Documentation: FeatureIndex Enum (MEDIUM)

**Problem**: 92 feature indices with no documentation about ordering requirement

**Tasks**:

- [ ] **Task 2.3.1**: Document FeatureIndex enum comprehensively
  - File: `InflamAI/Core/ML/FeatureExtractor.swift:176-222`
  ```swift
  /// Feature indices matching the ML model's expected input order
  ///
  /// ⚠️ CRITICAL: Order MUST match `scaler_params.json` from training pipeline
  /// Changing order will break predictions silently (no compile-time error)
  ///
  /// ## Feature Grouping (92 total)
  /// 1. Demographics (0-5): Age, gender, HLA-B27, disease duration, BMI, smoking
  /// 2. Clinical (6-17): BASDAI, ASDAS, BASFI, joint counts, etc.
  /// 3. Pain (18-31): Pain characterization metrics
  /// 4. Activity (32-54): HealthKit biometrics (HRV, HR, steps)
  /// 5. Sleep (55-63): Sleep quality and stages
  /// 6. Mental Health (64-75): Mood, stress, cognitive
  /// 7. Environmental (76-82): Weather and season
  /// 8. Adherence (83-87): Medication/exercise tracking
  /// 9. Universal (88-91): Overall assessment
  ///
  /// ## Modifying This Enum
  /// 1. Retrain ML model with new feature set
  /// 2. Regenerate scaler_params.json
  /// 3. Update this enum to match
  /// 4. Update featureCount in UnifiedNeuralEngine
  enum FeatureIndex: Int {
  ```

- [ ] **Task 2.3.2**: Document scaler_params.json dependency
  - File: `InflamAI/Core/ML/UnifiedNeuralEngine.swift:341`
  - Add documentation about file format, location, regeneration

**Acceptance Criteria**:
- FeatureIndex enum fully documented
- scaler_params.json dependency documented
- Maintenance instructions clear

---

### 2.4 📝 Documentation: Data Quality Thresholds (MEDIUM)

**Problem**: 0.15 and 20 thresholds lack clinical rationale

**Tasks**:

- [ ] **Task 2.4.1**: Document threshold rationale
  - File: `InflamAI/Core/ML/UnifiedNeuralEngine.swift:146-150`
  ```swift
  /// Minimum data quality score required for predictions (0.0-1.0)
  ///
  /// Set to 0.15 (15% of 92 features = 13.8 features) based on:
  /// - Model ablation study: Performance degrades below 15 features
  /// - Clinical need: Must have basic symptom + context data
  /// - User experience: Prevents "0% risk" from empty data
  private let minimumDataQualityScore: Float = 0.15

  /// Minimum non-zero features required per day
  ///
  /// Set to 20 based on clinical minimum viable dataset:
  /// - Core symptoms: BASDAI, pain, stiffness (3)
  /// - Mental health: Mood, fatigue, stress (3)
  /// - Body regions: At least 5 logged (5)
  /// - Context: Time, location, season (3)
  /// - Biometrics: HRV, HR, steps (3)
  /// - Environmental: Weather pressure, temp, humidity (3)
  private let minimumNonZeroFeatures: Int = 20
  ```

**Acceptance Criteria**:
- All thresholds have documented rationale
- Clinical reasoning explained

---

## Phase 3: Medium Priority Issues (Next Month)

### 3.1 🏗️ Architecture: Dependency Injection Inconsistency (MEDIUM)

**Problem**: Some services use hardcoded singletons instead of DI

**Tasks**:

- [ ] **Task 3.1.1**: Add DI to remaining services (if not deleted)
  - Files: `BinaryFlarePredictionService.swift`, `NeuralEngineMLService.swift`
  - Pattern:
  ```swift
  init(persistenceController: InflamAIPersistenceController = .shared,
       healthKitService: HealthKitService? = nil)
  ```

**Note**: This may be N/A if services are deleted in Phase 1

---

### 3.2 🏗️ Architecture: Zero-as-Missing Documentation (MEDIUM)

**Problem**: 0 represents "missing data" but some features have legitimate zero values

**Tasks**:

- [ ] **Task 3.2.1**: Document the zero-as-missing pattern
  - File: `InflamAI/Core/ML/FeatureExtractor.swift` (header comment)
  ```swift
  // ARCHITECTURE DECISION: Zero-as-Missing with Explicit Tracking
  //
  // - Feature values: 0 can mean either "zero" or "missing"
  // - FeatureAvailability.featureHasRealData[i]: true = REAL data, false = missing
  // - ML model learns separate "missing" pattern from availability mask
  //
  // Special Cases:
  // - Age: 0 = unknown (separate from valid ages 18-100)
  // - Gender: 0.0 = unknown (ambiguous with female=0, requires flag)
  // - Days since flare: -1 = no previous flare (sentinel value)
  ```

- [ ] **Task 3.2.2**: Consider adding gender_known flag feature
  - Evaluate if gender=0 ambiguity is a real problem
  - If yes, add `gender_known` boolean feature

**Acceptance Criteria**:
- Zero-as-missing pattern documented
- Ambiguous cases identified and addressed

---

### 3.3 💻 Code Quality: Long Method Refactoring (MEDIUM)

**Problem**: `extract30DayFeaturesWithMetrics` is 60 lines with multiple responsibilities

**Tasks**:

- [ ] **Task 3.3.1**: Extract helper methods
  - File: `InflamAI/Core/ML/FeatureExtractor.swift:242-301`
  ```swift
  func extract30DayFeaturesWithMetrics(...) async -> FeatureExtractionResult {
      let healthKitAuthorized = await checkHealthKitAuthorization()
      let (features, metrics) = await extractFeaturesFor30Days(...)
      return buildExtractionResult(features, metrics, ...)
  }

  private func extractFeaturesFor30Days(...) async -> ([[Float]], FeatureExtractionMetrics)
  private func buildExtractionResult(...) -> FeatureExtractionResult
  ```

**Acceptance Criteria**:
- No method exceeds 30 lines
- Single responsibility per method
- Improved testability

---

### 3.4 📝 Documentation: User-Facing Error Messages (MEDIUM)

**Problem**: Error messages are developer-focused, not user-friendly

**Tasks**:

- [ ] **Task 3.4.1**: Create user-friendly error messages
  - File: `InflamAI/Core/ML/UnifiedNeuralEngine.swift:199-218`
  ```swift
  // Instead of:
  errorMessage = "Need more REAL data. No fake placeholders used."

  // Use:
  errorMessage = "Need More Data"
  errorDetails = """
  To get flare predictions:
  1. Log symptoms daily for at least 7 days
  2. Connect Apple Watch for biometric data
  3. Enable Location for weather tracking

  Progress: \(Int(availability.overallAvailability * 100))%
  """
  ```

- [ ] **Task 3.4.2**: Add `errorDetails` property for extended messages
  - Add to `UnifiedNeuralEngine` published properties

**Acceptance Criteria**:
- Error messages are user-actionable
- Progress shown where applicable
- No developer jargon in user-facing messages

---

## Implementation Checklist

### Week 1 (Critical)
- [ ] 1.1.1 - Audit prediction service usage
- [ ] 1.1.2 - Delete/migrate BinaryFlarePredictionService
- [ ] 1.1.3 - Delete/migrate NeuralEngineMLService
- [ ] 1.2.2 - Delete duplicate code in SymptomLog+MLExtensions
- [ ] 1.3.1 - Document predict() method
- [ ] 1.3.3 - Create migration guide
- [ ] 1.4.1 - Add medical disclaimers

### Week 2 (High)
- [ ] 1.1.4 - Refactor FlarePredictor as thin wrapper
- [ ] 1.1.5 - Update CLAUDE.md architecture
- [ ] 1.3.2 - Document FeatureAvailability
- [ ] 2.1.1 - Add FeatureAvailability validation to FlarePredictor

### Week 3-4 (Medium)
- [ ] 2.2.1 - Create MLModelConfiguration struct
- [ ] 2.2.2 - Replace magic numbers
- [ ] 2.3.1 - Document FeatureIndex enum
- [ ] 2.3.2 - Document scaler_params.json
- [ ] 2.4.1 - Document threshold rationale

### Month 2 (Polish)
- [ ] 3.1.1 - DI consistency (if applicable)
- [ ] 3.2.1 - Document zero-as-missing pattern
- [ ] 3.3.1 - Refactor long methods
- [ ] 3.4.1 - User-friendly error messages

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Prediction services | 4 | 2 |
| Duplicate code lines | ~220 | 0 |
| Documented public APIs | 40% | 100% |
| Magic number occurrences | 15+ | 0 |
| Medical disclaimers | 1/4 files | 4/4 files |
| Methods > 30 lines | 3 | 0 |

---

## Notes

- **Dependencies**: Task 1.1.2-1.1.3 may eliminate need for 3.1.1
- **Testing**: See separate `ML_TESTING_PLAN.md` for test coverage tasks
- **Performance**: See separate `ML_PERFORMANCE_PLAN.md` for optimization tasks
- **Security**: See separate `ML_SECURITY_PLAN.md` for security hardening

---

*Plan Created: 2024-12-02*
*Last Updated: 2024-12-02*
*Status: Ready for Implementation*
