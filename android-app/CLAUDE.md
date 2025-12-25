# CLAUDE.md - InflamAI Android

This file provides guidance to Claude Code when working with the Android version of InflamAI.

## Project Overview

**InflamAI Android** is a native Android port of the iOS InflamAI app - a production-grade application for Ankylosing Spondylitis (AS) management, built to Fortune 100 quality standards with privacy-first architecture and WCAG AA accessibility compliance.

- **Platform**: Android 9+ (API 28+), Kotlin 1.9+
- **Architecture**: MVVM + Multi-module + Clean Architecture
- **UI**: Jetpack Compose with Material Design 3
- **Persistence**: Room Database
- **DI**: Hilt (Dagger)
- **Health Data**: Health Connect API
- **Privacy**: 100% on-device processing, zero third-party SDKs

## Project Structure

```
android-app/
├── app/                           # Main application module
│   ├── src/main/java/com/inflamai/app/
│   │   ├── InflamAIApplication.kt # Hilt @HiltAndroidApp
│   │   ├── MainActivity.kt        # Single Activity with Compose
│   │   ├── di/                    # Hilt modules
│   │   ├── navigation/            # Navigation Compose
│   │   └── security/              # BiometricAuthManager
│   └── src/main/res/              # Resources
├── core/
│   ├── common/                    # Shared utilities
│   ├── data/                      # Room DB, repositories, services
│   │   ├── database/entity/       # 13 Room entities
│   │   ├── database/dao/          # Data Access Objects
│   │   └── service/               # Health Connect, Weather
│   ├── domain/                    # Business logic, use cases
│   │   └── calculator/            # BASDAI, ASDAS, Correlation
│   └── ui/                        # Shared Compose components, theme
├── feature/                       # Feature modules
│   ├── home/
│   ├── bodymap/
│   ├── checkin/
│   ├── medication/
│   ├── trends/
│   ├── flares/
│   ├── exercise/
│   ├── ai/
│   ├── settings/
│   ├── onboarding/
│   ├── meditation/
│   └── quickcapture/
├── wear/                          # Wear OS companion app
├── build.gradle.kts               # Root build file
├── settings.gradle.kts            # Module includes
└── gradle/libs.versions.toml      # Version catalog
```

## Build Commands

```bash
# Build project
./gradlew build

# Run on device/emulator
./gradlew installDebug

# Run tests
./gradlew test

# Clean build
./gradlew clean build
```

## Room Database (13 Entities)

Equivalent to iOS Core Data model:

| Entity | Purpose |
|--------|---------|
| SymptomLogEntity | Daily symptom tracking with BASDAI scores |
| BodyRegionLogEntity | 47-region pain tracking |
| ContextSnapshotEntity | Weather + biometric context |
| MedicationEntity | Prescription tracking |
| DoseLogEntity | Medication adherence |
| FlareEventEntity | Flare episodes |
| ExerciseSessionEntity | Workout history |
| ExerciseEntity | Exercise library (50+) |
| UserProfileEntity | User settings (singleton) |
| MeditationSessionEntity | Meditation tracking |
| MeditationStreakEntity | Meditation streaks |
| TriggerLogEntity | Pattern trigger events |
| TriggerAnalysisCacheEntity | Cached analysis results |

## Medical Calculators (CLINICALLY VALIDATED)

**CRITICAL: Do NOT modify formulas without clinical review**

### BASDAI Calculator
```kotlin
// Location: core/domain/calculator/BASDAICalculator.kt
// Formula: (Q1 + Q2 + Q3 + Q4 + ((Q5 + Q6) / 2)) / 5

BASDAICalculator.calculate(
    fatigue = 5.0,
    spinalPain = 6.0,
    peripheralPain = 3.0,
    enthesitisPain = 4.0,
    morningSeverity = 7.0,
    morningDuration = 5.0  // 0-10 scale
) // Returns: Double (0-10)
```

### ASDAS-CRP Calculator
```kotlin
// Location: core/domain/calculator/ASDACalculator.kt
// Formula: 0.12×BackPain + 0.06×Duration + 0.11×PatientGlobal
//          + 0.07×PeripheralPain + 0.58×Ln(CRP+1)

ASDACalculator.calculateWithCRP(
    backPain = 5.0,
    morningStiffnessDuration = 5.0,
    patientGlobalAssessment = 5.0,
    peripheralPain = 5.0,
    crpMgL = 10.0
) // Returns: Double
```

### Correlation Engine
```kotlin
// Location: core/domain/calculator/CorrelationEngine.kt
// Statistical analysis - NOT machine learning

CorrelationEngine().analyzeCorrelation(
    factorData = weatherPressureData,
    symptomData = painScoreData,
    factorName = "Barometric Pressure",
    factorCategory = "Weather"
)
```

## Health Connect Integration

```kotlin
// Location: core/data/service/health/HealthConnectService.kt

// Check availability
healthConnectService.isHealthConnectAvailable()

// Get daily snapshot
healthConnectService.getDailyHealthSnapshot(Instant.now())
// Returns: DailyHealthSnapshot with HR, HRV, steps, sleep, etc.
```

### Data Types Read (Read-Only)
- Heart Rate & HRV (RMSSD)
- Resting Heart Rate
- Steps & Distance
- Sleep Sessions & Stages
- Active Calories
- Oxygen Saturation
- Respiratory Rate
- Exercise Sessions

## Key Dependencies (libs.versions.toml)

| Library | Version | Purpose |
|---------|---------|---------|
| Kotlin | 1.9.22 | Language |
| Compose BOM | 2024.02.02 | UI |
| Room | 2.6.1 | Database |
| Hilt | 2.50 | DI |
| Health Connect | 1.1.0-alpha07 | Health data |
| Vico | 1.14.0 | Charts |
| Biometric | 1.2.0-alpha05 | Auth |

## Privacy & Security

### Zero Third-Party SDKs
- NO Firebase Analytics
- NO Amplitude
- NO Facebook SDK
- NO crash reporting services

### Data Protection
- Biometric lock (BiometricPrompt API)
- Room database encryption (optional SQLCipher)
- EncryptedSharedPreferences for sensitive prefs
- All processing on-device
- GDPR-compliant data deletion

### Health Connect Permissions
```xml
<uses-permission android:name="android.permission.health.READ_HEART_RATE" />
<uses-permission android:name="android.permission.health.READ_HEART_RATE_VARIABILITY" />
<uses-permission android:name="android.permission.health.READ_SLEEP" />
<!-- See AndroidManifest.xml for full list -->
```

## Accessibility (WCAG AA)

- **TalkBack**: All elements have contentDescription
- **Text Scaling**: Uses sp units, supports XXXL
- **Touch Targets**: 48dp minimum
- **Contrast**: 4.5:1 minimum for text
- **Motion**: Respects Animator duration scale
- **Haptics**: HapticFeedbackConstants for feedback

## When Working on This Codebase

### DO:
- ✅ Use Jetpack Compose for all UI
- ✅ Follow MVVM with StateFlow
- ✅ Use Hilt for dependency injection
- ✅ Add @Composable contentDescription for accessibility
- ✅ Use Room for persistence
- ✅ Run validation tests for calculators in DEBUG
- ✅ Keep all processing on-device

### DON'T:
- ❌ Add third-party analytics SDKs
- ❌ Modify BASDAI/ASDAS formulas without clinical review
- ❌ Use XML layouts (Compose only)
- ❌ Force unwrap nullables
- ❌ Skip accessibility testing
- ❌ Store health data in cloud without encryption

## Testing

```bash
# Unit tests
./gradlew test

# Calculator validation (runs in DEBUG automatically)
BASDAICalculator.runValidationTests()
ASDACalculator.runValidationTests()
CorrelationEngine().runValidationTests()

# UI tests
./gradlew connectedAndroidTest
```

## iOS to Android Mapping

| iOS | Android |
|-----|---------|
| Core Data | Room Database |
| SwiftUI | Jetpack Compose |
| @StateObject | ViewModel + StateFlow |
| Combine | Kotlin Coroutines + Flow |
| HealthKit | Health Connect |
| WeatherKit | Open-Meteo API |
| LocalAuthentication | BiometricPrompt |
| DIContainer | Hilt |

## Documentation Reference

- iOS CLAUDE.md: `/Users/fabianharnisch/Documents/inflamai-demo/CLAUDE.md`
- Architecture decisions align with iOS for feature parity

---

**Last Updated**: 2025-12-23
**Architecture Version**: 1.0 (Android Port)
