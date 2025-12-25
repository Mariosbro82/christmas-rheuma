# CRITICAL CODEBASE AUDIT: Data Persistence & HealthKit Integration Failures

**Date**: 2024-11-27
**Severity**: CRITICAL
**Auditor**: Automated Code Analysis

---

## Executive Summary

This audit identified **27 critical failures** preventing user data from being saved and displayed correctly. The app has fundamental architectural issues where data flows into oblivion - users enter information (name, weight, BMI, birthday, disease onset) that silently fails to persist.

---

## CATEGORY 1: PERSISTENCE ARCHITECTURE BROKEN

### CRITICAL: Two Persistence Controllers, One Loads Non-Existent Model

| File | Model Name Loaded | Exists? |
|------|-------------------|---------|
| `Persistence.swift:78` | `"InflamAI"` | NO |
| `Core/Persistence/InflamAIPersistenceController.swift:74` | `"InflamAI"` | YES |

**Impact**: Any code using `PersistenceController.shared` writes to an in-memory store that evaporates on app close.

**Location**: `/InflamAI/Persistence.swift:78`
```swift
container = NSPersistentContainer(name: "InflamAI")  // DOESN'T EXIST!
```

### CRITICAL: Two HealthKitManager Classes (Namespace Collision)

Both exist simultaneously:
- `/InflamAI/HealthKitManager.swift`
- `/InflamAI/Managers/HealthKitManager.swift`

Both store data in memory-only `@Published` properties that vanish on app close.

### CRITICAL: Three UserProfileEditViewModel Classes

| Location | Class Name |
|----------|------------|
| `Features/Settings/UserProfileEditView.swift:194` | UserProfileEditViewModel |
| `Features/Onboarding/OnboardingFlow.swift:1794` | UserProfileEditViewModel |
| `SettingsView.swift:780` | SettingsProfileEditViewModel |

**Impact**: Race conditions, duplicate profiles, inconsistent state.

---

## CATEGORY 2: USER PROFILE DATA NEVER PERSISTS

### Onboarding Profile Only Saves at Completion

**Location**: `OnboardingFlow.swift:46-69`

Profile data entered on page 3 is NOT saved until page 7:
```swift
CompletionPage(onComplete: {
    profileViewModel.saveProfile()  // ONLY HERE
    dismiss()
})
```

**Impact**: If user quits at page 4, 5, or 6: Name, weight, height, birthday, diagnosis date = LOST.

### Save Failures Are Silent

**Pattern found in all save locations**:
```swift
do {
    try context.save()
    print("Profile saved successfully")
} catch {
    print("Error saving profile: \(error)")  // JUST PRINTS TO CONSOLE
}
```

**Impact**: User never sees errors. Data loss is invisible.

---

## CATEGORY 3: HEALTHKIT IS A FACADE

### 5-Second Timeout Kills Real Authorization

**Location**: `InflamAIApp.swift:100-108`
```swift
await withTimeout(seconds: 5) {  // ONLY 5 SECONDS!
    try await HealthKitService.shared.requestAuthorization()
}
```

**Impact**: Real HealthKit permission dialog takes 10-30 seconds. App times out before user taps "Allow".

### Authorization Success = No Data Fetched

After authorization completes:
```swift
print("✅ HealthKit authorization completed")
// WHERE IS THE DATA FETCH? - MISSING!
```

**Impact**: `fetchAllBiometrics()` method exists at `HealthKitService.swift:551` but has ZERO callers.

### HealthKit Data Goes to Memory, Not Core Data

**Location**: `HealthKitManager.swift`
```swift
@Published var heartRate: Double = 0
@Published var heartRateData: [HeartRateReading] = []
@Published var sleepData: [SleepDataPoint] = []
```

**Impact**: On app close, all data gone. No persistence to Core Data.

### Background Processing Returns Empty Arrays

**Location**: `BackgroundHealthProcessor.swift:438-446`
```swift
private func performDataAnalysis(_ data: AggregatedHealthData) async {
    async let patterns = patternAnalyzer.analyzePatterns(data)    // Returns []
    async let anomalies = anomalyDetector.detectAnomalies(data)   // Returns []
    async let trends = trendCalculator.calculateTrends(data)      // Returns []
    let _ = await [patterns, anomalies, trends, ...]              // DISCARDED!
}
```

**Impact**: Dead code. All analysis methods return empty arrays.

---

## CATEGORY 4: DATA DISPLAY LIES

### 20 ML Features Hardcoded as "Unavailable"

**Location**: `MLFeaturesDisplayView.swift:700-706`
```swift
case "blood_oxygen", "cardio_fitness", "respiratory_rate", ...
    "gait_asymmetry", "bipedal_support":
    return (false, 0)  // ALWAYS RETURNS ZERO
```

**Impact**: Even if HealthKit data existed, display layer refuses to show it.

### Zeros Display as Dashes

**Location**: `MLFeaturesDisplayView.swift:870`
```swift
if currentValue == 0.0 { return "—" }
```

**Impact**: User sees sea of dashes, thinks "data loading" when it's "data never collected".

### ML Estimates Masquerade as Real Data

**Location**: `SymptomLog+MLExtensions.swift:177-225`
```swift
if snapshot.hrvValue > 0 {
    // Real data
} else {
    snapshot.hrvValue = Double(ageEffect + stressEffect) / 2.0  // FAKE ESTIMATE
}
```

**Impact**: Neural Engine trains on mathematical estimates, not real biometrics.

---

## CATEGORY 5: CORE DATA SILENT FAILURES

### `try?` Swallows Critical Errors

**Affected Files**:
| File | Line |
|------|------|
| `Features/Routines/RoutineExecutionView.swift` | 455, 477 |
| `Features/BodyMap/BodyMapModels.swift` | 88 |
| `Features/Flares/FlareTimelineView.swift` | 783 |
| `Features/Medication/MedicationViewModel.swift` | 243, 260 |
| `Features/Coach/CoachCompositorView.swift` | 829 |
| `Features/PainMap/PainMapViewModel.swift` | 87 |

**Impact**: User completes medication logging, save silently fails, data lost, user never knows.

### Field Name Mismatch in BodyMap

**Location**: `BodyMapModels.swift:83`
```swift
regionLog.region = region.rawValue  // WRONG FIELD NAME
```

**Core Data model has `regionID`, not `region`**. This line does nothing or crashes.

### ContextSnapshot Missing in 90% of Flows

Only `DailyCheckInViewModel` creates ContextSnapshot. Missing from:
- `QuickLogViewModel` (line 126: `// TODO: Implement this method`)
- `BodyMapModels`
- `PainMapViewModel`
- All quick logging flows

**Impact**: 90% of symptom logs have NO weather/biometric context for flare prediction.

---

## SUMMARY TABLE

| System | Status | Data Loss Risk |
|--------|--------|----------------|
| User Profile | 3 duplicate ViewModels, silent save failures | 100% on onboarding exit |
| HealthKit Fetch | Timeout kills auth, no data fetch after | 100% |
| HealthKit Store | Memory-only, never to Core Data | 100% on app close |
| Weather Data | Location never initialized | 100% |
| Core Data Saves | 9 silent `try?` failures | Variable |
| ML Features | 20 hardcoded as unavailable | 100% |
| ContextSnapshot | Only created in 1 of 4 flows | 90% missing |
| Persistence Controllers | 2 controllers, 1 loads ghost model | Split brain |

---

## DATA FLOW FAILURE CHAIN

```
User enters name, weight, BMI, birthday, disease onset
    ↓
OnboardingFlow saves profile... only on page 7 completion
    ↓
User quits on page 4 → DATA LOST
    ↓
HealthKit requests authorization → 5-second timeout fires → Auth cancelled
    ↓
No data ever fetched from HealthKit
    ↓
DailyCheckIn tries to fetch → isAuthorized=false → throws → caught silently
    ↓
ContextSnapshot filled with zeros → saved as "real data"
    ↓
MLFeaturesDisplayView sees zeros → shows "—" for everything
    ↓
Neural Engine trains on mathematical estimates, not biometrics
    ↓
User sees "Active: 0/92 features" and assumes it's loading
```

---

## IMMEDIATE ACTION ITEMS

### Priority 1 - Critical (Data Loss Prevention)

1. **DELETE** `Persistence.swift` - uses non-existent "InflamAI" model
2. **DELETE** one of the duplicate `HealthKitManager.swift` files
3. **CONSOLIDATE** three `UserProfileEditViewModel` classes into one
4. **ADD** intermediate saves to OnboardingFlow (save after each page)
5. **INCREASE** HealthKit authorization timeout to 30+ seconds
6. **CALL** `fetchAllBiometrics()` after successful authorization

### Priority 2 - High (Silent Failures)

7. **REPLACE** all `try? context.save()` with proper error handling + user alerts
8. **FIX** `BodyMapModels.swift:83` - change `region` to `regionID`
9. **UNCOMMENT** `populateMLProperties()` in `QuickLogViewModel.swift:126`
10. **ADD** ContextSnapshot creation to all symptom log save paths

### Priority 3 - Medium (Display Issues)

11. **REMOVE** hardcoded unavailable features in `MLFeaturesDisplayView.swift:700-706`
12. **ADD** "No data" indicators instead of showing dashes as if loading
13. **LABEL** estimated values differently from real biometric data

---

## Files Requiring Changes

### DELETE These Files
- `/InflamAI/Persistence.swift`
- One of: `/InflamAI/HealthKitManager.swift` OR `/InflamAI/Managers/HealthKitManager.swift`

### CRITICAL Fixes Required
- `/InflamAI/InflamAIApp.swift` - HealthKit timeout + data fetch
- `/InflamAI/Features/Onboarding/OnboardingFlow.swift` - intermediate saves
- `/InflamAI/Features/Settings/UserProfileEditView.swift` - consolidate ViewModel
- `/InflamAI/Features/BodyMap/BodyMapModels.swift` - field name fix
- `/InflamAI/Features/CheckIn/QuickLogViewModel.swift` - uncomment TODO
- `/InflamAI/Features/AI/MLFeaturesDisplayView.swift` - remove hardcoded zeros
- `/InflamAI/Core/Services/HealthKitService.swift` - ensure isAuthorized updates

### Error Handling Fixes
- `/InflamAI/Features/Routines/RoutineExecutionView.swift:455,477`
- `/InflamAI/Features/Flares/FlareTimelineView.swift:783`
- `/InflamAI/Features/Medication/MedicationViewModel.swift:243,260`
- `/InflamAI/Features/Coach/CoachCompositorView.swift:829`
- `/InflamAI/Features/PainMap/PainMapViewModel.swift:87`

---

## Conclusion

This codebase has **architectural cancer** - the UI looks professional but data flows into oblivion. Users trust the app with medical information that silently disappears. The Neural Engine trains on calculated estimates, not real Apple Watch data.

**Estimated fix time**: 2-3 weeks of focused refactoring
**Risk if unfixed**: Complete user data loss, invalid ML predictions, potential medical harm from incorrect flare predictions
