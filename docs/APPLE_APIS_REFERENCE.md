# Apple APIs Reference - InflamAI (InflamAI)

> Complete inventory of Apple frameworks and APIs used in this iOS health application.

**Last Updated**: 2025-11-26
**Total Frameworks**: 46
**Total Files Analyzed**: 322

---

## Executive Summary

InflamAI leverages 46 Apple frameworks for a privacy-first, medical-grade Ankylosing Spondylitis management app. Key integrations include HealthKit for biometrics, WeatherKit for barometric pressure tracking, CoreML for flare prediction, and comprehensive security via LocalAuthentication and CryptoKit.

**Zero third-party SDKs** - All processing on-device.

---

## Framework Overview

| Framework | Files | Primary Purpose |
|-----------|-------|-----------------|
| SwiftUI | 193 | All UI views and components |
| Foundation | 189 | Core utilities, Date, URL, JSON |
| Combine | 112 | Reactive state management |
| CoreData | 98 | Persistent storage (17 entities) |
| CoreML | 49 | ML model inference |
| HealthKit | 46 | Biometric data collection |
| UserNotifications | 34 | Medication/flare alerts |
| UIKit | 33 | PDF generation, haptics bridge |
| Charts | 30 | Swift Charts visualizations |
| CryptoKit | 25 | AES-256 encryption |
| AVFoundation | 25 | Audio/speech processing |
| CoreLocation | 22 | Weather context |
| os.log | 21 | System logging |
| Security | 13 | Keychain operations |
| WatchConnectivity | 12 | Apple Watch sync |
| Speech | 11 | Voice commands |
| LocalAuthentication | 10 | Face ID / Touch ID |
| CloudKit | 10 | Optional iCloud sync |
| Network | 9 | Connectivity monitoring |
| NaturalLanguage | 9 | Command parsing |
| CoreHaptics | 9 | Haptic feedback |
| ARKit | 8 | 3D body mapping |
| BackgroundTasks | 7 | Background processing |
| Vision | 7 | Image analysis |
| CallKit | 7 | Emergency calls |
| CoreMotion | 6 | Motion sensors |
| WeatherKit | 2 | Barometric pressure |

---

## Detailed API Usage

### 1. HealthKit

**Import Count**: 46 files
**Authorization**: Read-only (no writes except mindful sessions)

#### Data Types Read

| Type | Identifier | Unit | Usage |
|------|------------|------|-------|
| Heart Rate | `.heartRate` | BPM | Real-time monitoring |
| HRV (SDNN) | `.heartRateVariabilitySDNN` | ms | Stress/inflammation marker |
| Resting HR | `.restingHeartRate` | BPM | Daily baseline |
| Step Count | `.stepCount` | count | Activity tracking |
| Distance | `.distanceWalkingRunning` | m | Mobility metric |
| Active Energy | `.activeEnergyBurned` | kcal | Exercise tracking |
| Sleep Analysis | `.sleepAnalysis` | - | 5 sleep stages |
| Body Temp | `.bodyTemperature` | °C | Inflammation indicator |
| SpO2 | `.oxygenSaturation` | % | Vital sign |
| Respiratory Rate | `.respiratoryRate` | br/min | Vital sign |
| Blood Pressure | `.bloodPressure*` | mmHg | Cardiovascular |

#### Background Delivery

```swift
healthStore.enableBackgroundDelivery(for: sampleType, frequency: .hourly)
```

Types enabled: Heart Rate, Steps, Sleep, Distance, Active Energy, Blood Pressure

#### Key Files
- `Core/Services/HealthKitService.swift` (262 lines)
- `Core/BackgroundHealthProcessor.swift` (838 lines)
- `Managers/HealthKitManager.swift` (609 lines)

---

### 2. WeatherKit

**Import Count**: 2 dedicated files
**Total Weather Code**: ~2,900 lines

#### Implementation

| Service | Purpose | Lines |
|---------|---------|-------|
| `WeatherKitService.swift` | Apple WeatherKit integration | 230 |
| `WeatherService.swift` | OpenWeatherMap fallback | 544 |
| `ResilientWeatherService.swift` | Retry logic + caching | 218 |
| `WeatherCache.swift` | 24-hour disk cache | 250 |
| `WeatherNotificationService.swift` | Background alerts | 257 |
| `WeatherFlareRiskViewModel.swift` | Flare prediction | 448 |

#### Critical Feature: Barometric Pressure

```swift
// Pressure conversion
let mmHg = hPa * 0.750062

// Flare trigger threshold
if pressureChange12h < -5.0 { // 5+ mmHg drop
    triggerFlareRiskWarning()
}
```

#### Caching Strategy
- **Memory**: 10-minute TTL, 100 items, 10MB
- **Disk**: 24-hour TTL, 50MB max, LRU eviction

---

### 3. CoreML

**Import Count**: 49 files
**Model**: `ASFlarePredictor.mlpackage` (22.5 KB)

#### Model Architecture

| Property | Value |
|----------|-------|
| Type | LSTM (Bidirectional) |
| Input Shape | (1, 30, 92) |
| Sequence Length | 30 days |
| Features | 92 per timestep |
| Output | Flare probability (0-1) |
| Compute Units | Neural Engine preferred |

#### 92 Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Demographics | 6 | Age, gender, HLA-B27, BMI |
| Clinical | 15 | BASDAI, ASDAS-CRP, BASFI |
| Pain | 14 | Current, average, max, locations |
| Activity | 20 | HR, HRV, steps, distance |
| Sleep | 8 | Hours, REM, deep, efficiency |
| Mental Health | 11 | Mood, anxiety, stress |
| Environmental | 8 | Temp, humidity, pressure |
| Adherence | 5 | Medication, physio |
| Context | 4 | Season, daylight |

#### On-Device Learning

```swift
// MLUpdateTask for personalization
private func performOnDeviceUpdate() async throws {
    let updateTask = try MLUpdateTask(...)
    // Personalizes model to user's patterns
}
```

#### Key Files
- `Core/ML/UnifiedNeuralEngine.swift` (~800 lines)
- `Core/ML/FlarePredictor.swift` (~600 lines) - Statistical fallback
- `Core/ML/ExplainabilityEngine.swift` (~350 lines)

---

### 4. Core Data

**Entities**: 17
**Relationships**: 20+
**CloudKit Sync**: Optional

#### Entity Summary

| Entity | Purpose | Key Fields |
|--------|---------|------------|
| SymptomLog | Daily records | BASDAI, fatigue, mood, pain |
| BodyRegionLog | 47 body regions | painLevel, stiffness, swelling |
| ContextSnapshot | Environmental | pressure, temp, HRV, steps |
| Medication | Prescriptions | name, dosage, frequency |
| DoseLog | Adherence | timestamp, taken, skipReason |
| FlareEvent | Acute episodes | severity, triggers, duration |
| ExerciseSession | Workouts | type, duration, painBefore/After |
| UserProfile | Settings | preferences, sync flags |
| JointComfortProfile | ROM calibration | flexionCap, painThreshold |
| QuestionnaireResponse | Assessments | score, answers, version |

#### CloudKit Record Types
```
HealthData, PainEntry, MedicationRecord, JournalEntry,
WorkoutSession, VitalSigns, UserProfile, TreatmentPlan,
SymptomTracking, AppointmentRecord
```

---

### 5. LocalAuthentication (Face ID / Touch ID)

**Import Count**: 10 files

#### Implementation

```swift
let context = LAContext()
context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics,
                       localizedReason: "Unlock InflamAI") { success, error in
    // Handle authentication result
}
```

#### Security Features
- Auto-lock on app backgrounding
- Fallback to device passcode
- Keychain integration for credentials

#### Key Files
- `Core/Security/BiometricAuthenticationEngine.swift`
- `Core/Security/SecurityManager.swift`
- `Features/Settings/SettingsView.swift`

---

### 6. WatchConnectivity

**Import Count**: 12 files

#### Sync Capabilities

| Direction | Data |
|-----------|------|
| iPhone → Watch | Medication schedules, settings |
| Watch → iPhone | HRV, HR, sleep, quick pain logs |
| Bidirectional | Real-time health updates |

#### Key Features
- Background delivery enabled
- Complication support planned
- Emergency SOS integration

---

### 7. Charts (Swift Charts)

**Import Count**: 30 files

#### Visualizations

| Chart Type | Usage |
|------------|-------|
| Line Chart | BASDAI trends, pressure trends |
| Bar Chart | Symptom distribution |
| Area Chart | Sleep patterns |
| Point Chart | Correlation scatter plots |

#### Key Views
- `Features/Trends/TrendsView.swift`
- `Features/Weather/PressureTrendChart.swift`
- `Views/Analytics/*.swift`

---

### 8. UserNotifications

**Import Count**: 34 files

#### Notification Types

| Type | Trigger | Content |
|------|---------|---------|
| Medication Reminder | Scheduled | "Time to take [medication]" |
| Flare Warning | Pressure drop | "Weather change detected" |
| Check-in Prompt | Daily | "Log your symptoms" |
| Emergency Alert | User-triggered | SOS with location |

---

### 9. Security Stack

#### CryptoKit (25 files)
- AES-256-GCM encryption
- HKDF key derivation
- SHA-256 hashing

#### Security Framework (13 files)
- Keychain access
- Secure credential storage

#### Implementation
```swift
// Encryption example
let sealedBox = try AES.GCM.seal(data, using: key)
let encryptedData = sealedBox.combined
```

---

### 10. Additional Frameworks

#### AR/3D Body Mapping
- **ARKit** (8 files): AR body scanning
- **Vision** (7 files): Image analysis
- **SceneKit** (5 files): 3D visualization
- **RealityKit** (4 files): AR rendering

#### Voice Commands
- **Speech** (11 files): Recognition
- **NaturalLanguage** (9 files): Command parsing
- **AVFoundation** (25 files): Audio capture

#### Emergency Services
- **CallKit** (7 files): Emergency calls
- **Contacts** (4 files): Emergency contacts
- **MessageUI** (4 files): SMS/MMS alerts

#### Background Processing
- **BackgroundTasks** (7 files): BGProcessingTask
- Task ID: `com.inflamai.weatherMonitoring`
- Frequency: Every 12 hours

---

## Privacy Architecture

### On-Device Processing
- All ML inference local
- No cloud analytics
- No third-party SDKs

### Data Flow
```
User Input → Core Data → Local Analysis → UI
     ↓              ↓
HealthKit      WeatherKit
(read-only)   (location-based)
     ↓              ↓
     └── CloudKit (optional, encrypted) ──┘
```

### Permissions Required

| Permission | Usage String |
|------------|--------------|
| HealthKit | "Identify patterns affecting AS symptoms" |
| Location | "Fetch weather for barometric pressure" |
| Notifications | "Medication reminders and flare alerts" |
| Face ID | "Secure your health data" |

---

## File Locations

### Core Services
```
Core/Services/
├── HealthKitService.swift
├── WeatherKitService.swift
└── WeatherService.swift

Core/ML/
├── UnifiedNeuralEngine.swift
├── FlarePredictor.swift
├── ExplainabilityEngine.swift
└── NeuralEngine/
    └── models/ASFlarePredictor.mlpackage/

Core/Security/
├── BiometricAuthenticationEngine.swift
├── KeychainManager.swift
└── SecurityManager.swift

Core/Persistence/
└── InflamAIPersistenceController.swift
```

---

## Version Requirements

| Framework | Minimum iOS |
|-----------|-------------|
| SwiftUI | 13.0 |
| Charts | 16.0 |
| WeatherKit | 16.0 |
| WidgetKit | 14.0 |
| **App Target** | **17.0+** |

---

## References

- [HealthKit Documentation](https://developer.apple.com/documentation/healthkit)
- [WeatherKit Documentation](https://developer.apple.com/documentation/weatherkit)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [LocalAuthentication](https://developer.apple.com/documentation/localauthentication)

---

*This document was generated from codebase analysis on 2025-11-26.*
