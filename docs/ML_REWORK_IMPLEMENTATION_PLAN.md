# ML Model Rework & Implementation Plan
Based on the comparison between the "InflamAI" configuration screenshots and the provided extraction log, the following enabled data points are missing from your extracted data.
Mobility & Gait Metrics
These are enabled in settings but absent from the log:
 * Gehtempo (Walking Speed)
 * Schrittlänge im Gehen (Walking Step Length)
 * Asymmetrischer Gang (Walking Asymmetry)
 * Bipedale Abstützungsdauer (Double Support Time)
 * Sechs-Minuten-Gehtest (Six-Minute Walk Test)
 * Treppensteigen: Abwärts (Stair Descent Speed)
 * Treppensteigen: Aufwärts (Stair Ascent Speed)
Vitals & Respiratory
The log captures Resting Heart Rate and HRV, but misses these enabled metrics:
 * Herzfrequenz (General Heart Rate / Pulse samples outside of resting)
 * Atemfrequenz (Respiratory Rate)
 * Blutsauerstoff (Blood Oxygen / SpO2)
 * Cardiofitness (VO2 Max)
Body Measurements
The log lists "Gender" (from Demographics/CoreData), but ignores these enabled measurements:
 * Gewicht (Weight)
 * Größe (Height)
 * Body-Mass-Index (BMI)
Activity & Energy
The log captures Active Energy and Exercise Time, but misses:
 * Ruheenergie (Resting Energy / Basal Burn)
 * Stehminuten (Stand Minutes)
 * Minuten der Achtsamkeit (Mindfulness Minutes)
 * Trainings (HKWorkout sessions – distinct from appleExerciseTime)
 * Strecke (Fahrrad) (Cycling Distance)
 * Strecke (Schwimmen) (Swimming Distance)
Fact Check: Your log shows HKQuantityType.flightsClimbed (Treppensteigen), which matches the configuration. However, the velocity metrics for stairs (Aufwärts/Abwärts) are definitely missing.

---

## EASY IMPLEMENTATIONS (Already Authorized in HealthKit)

These features are already authorized but not extracted. Add to `FeatureExtractor.swift` in `extractActivityMetrics()`.

---

### 1. Walking Speed (Gehtempo)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierWalkingSpeed` |
| Unit | `m/s` |
| Aggregation | `DAILY_AVG` |
| Feature Index | 51 (walking_tempo) |
| Difficulty | ⭐ Easy |

```swift
// Walking Speed (m/s)
if let speed = await healthKit.fetchWalkingSpeed(for: date) {
    features[FeatureIndex.walking_tempo.rawValue] = Float(speed)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchWalkingSpeed(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .walkingSpeed)!
    return try await fetchDiscreteAverage(type: type, unit: HKUnit.meter().unitDivided(by: .second()), for: date)
}
```

---

### 2. Walking Step Length (Schrittlänge)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierWalkingStepLength` |
| Unit | `cm` |
| Aggregation | `DAILY_AVG` |
| Feature Index | 52 (step_length) |
| Difficulty | ⭐ Easy |

```swift
// Step Length (cm)
if let stepLength = await healthKit.fetchStepLength(for: date) {
    features[FeatureIndex.step_length.rawValue] = Float(stepLength * 100) // m to cm
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchStepLength(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .walkingStepLength)!
    return try await fetchDiscreteAverage(type: type, unit: .meter(), for: date)
}
```

---

### 3. Walking Asymmetry (Asymmetrischer Gang)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierWalkingAsymmetryPercentage` |
| Unit | `%` (0-100) |
| Aggregation | `DAILY_AVG` |
| Feature Index | 53 (gait_asymmetry) |
| Difficulty | ⭐ Easy |
| Note | Higher % = more asymmetric gait (potential AS indicator!) |

```swift
// Gait Asymmetry (%)
if let asymmetry = await healthKit.fetchWalkingAsymmetry(for: date) {
    features[FeatureIndex.gait_asymmetry.rawValue] = Float(asymmetry * 100)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchWalkingAsymmetry(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .walkingAsymmetryPercentage)!
    return try await fetchDiscreteAverage(type: type, unit: .percent(), for: date)
}
```

---

### 4. Double Support Time (Bipedale Abstützung)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierWalkingDoubleSupportPercentage` |
| Unit | `%` (0-100) |
| Aggregation | `DAILY_AVG` |
| Feature Index | 54 (bipedal_support) |
| Difficulty | ⭐ Easy |
| Note | Higher % = slower/more cautious gait (pain indicator!) |

```swift
// Double Support Time (%)
if let doubleSupport = await healthKit.fetchDoubleSupportTime(for: date) {
    features[FeatureIndex.bipedal_support.rawValue] = Float(doubleSupport * 100)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchDoubleSupportTime(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .walkingDoubleSupportPercentage)!
    return try await fetchDiscreteAverage(type: type, unit: .percent(), for: date)
}
```

---

### 5. Blood Oxygen / SpO2 (Blutsauerstoff)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierOxygenSaturation` |
| Unit | `%` (0-100) |
| Aggregation | `DAILY_AVG` |
| Feature Index | 32 (blood_oxygen) |
| Difficulty | ⭐ Easy |
| Note | Requires Apple Watch with SpO2 sensor |

```swift
// Blood Oxygen (%)
if let spo2 = await healthKit.fetchBloodOxygen(for: date) {
    features[FeatureIndex.blood_oxygen.rawValue] = Float(spo2 * 100)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchBloodOxygen(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .oxygenSaturation)!
    return try await fetchDiscreteAverage(type: type, unit: .percent(), for: date)
}
```

---

### 6. Respiratory Rate (Atemfrequenz)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierRespiratoryRate` |
| Unit | `breaths/min` |
| Aggregation | `DAILY_AVG` |
| Feature Index | 34 (respiratory_rate) |
| Difficulty | ⭐ Easy |

```swift
// Respiratory Rate (breaths/min)
if let respRate = await healthKit.fetchRespiratoryRate(for: date) {
    features[FeatureIndex.respiratory_rate.rawValue] = Float(respRate)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchRespiratoryRate(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!
    return try await fetchDiscreteAverage(type: type, unit: HKUnit.count().unitDivided(by: .minute()), for: date)
}
```

---

### 7. VO2 Max / Cardio Fitness (Cardiofitness)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierVO2Max` |
| Unit | `mL/kg/min` |
| Aggregation | `LAST_SAMPLE` |
| Feature Index | 33 (cardio_fitness) |
| Difficulty | ⭐ Easy |
| Note | Updated infrequently (after outdoor walks/runs) |

```swift
// VO2 Max (mL/kg/min)
if let vo2 = await healthKit.fetchVO2Max(for: date) {
    features[FeatureIndex.cardio_fitness.rawValue] = Float(vo2)
    count += 1
}
```

**HealthKitService already has this method!** Just call it in FeatureExtractor.

---

### 8. Resting Energy / Basal Burn (Ruheenergie)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierBasalEnergyBurned` |
| Unit | `kcal` |
| Aggregation | `DAILY_SUM` |
| Feature Index | 36 (resting_energy) |
| Difficulty | ⭐ Easy |

```swift
// Resting Energy (kcal)
if let basalEnergy = await healthKit.fetchBasalEnergy(for: date) {
    features[FeatureIndex.resting_energy.rawValue] = Float(basalEnergy)
    count += 1
}
```

**HealthKitService already has this method!** Just call it in FeatureExtractor.

---

### 9. Stand Minutes (Stehminuten)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierAppleStandTime` |
| Unit | `minutes` |
| Aggregation | `DAILY_SUM` |
| Feature Index | 45 (stand_minutes) |
| Difficulty | ⭐ Easy |

```swift
// Stand Time (minutes)
if let standTime = await healthKit.fetchStandTime(for: date) {
    features[FeatureIndex.stand_minutes.rawValue] = Float(standTime)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchStandTime(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .appleStandTime)!
    return try await fetchCumulativeSum(type: type, unit: .minute(), for: date)
}
```

---

### 10. Stair Ascent Speed (Treppensteigen Aufwärts)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierStairAscentSpeed` |
| Unit | `m/s` |
| Aggregation | `DAILY_AVG` |
| Feature Index | Need to add or use existing |
| Difficulty | ⭐ Easy |

```swift
// Stair Ascent Speed (m/s)
if let ascentSpeed = await healthKit.fetchStairAscentSpeed(for: date) {
    // Store in appropriate index
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchStairAscentSpeed(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .stairAscentSpeed)!
    return try await fetchDiscreteAverage(type: type, unit: HKUnit.meter().unitDivided(by: .second()), for: date)
}
```

---

### 11. Stair Descent Speed (Treppensteigen Abwärts)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierStairDescentSpeed` |
| Unit | `m/s` |
| Aggregation | `DAILY_AVG` |
| Feature Index | 44 (stairs_down) |
| Difficulty | ⭐ Easy |

```swift
// Stair Descent Speed (m/s)
if let descentSpeed = await healthKit.fetchStairDescentSpeed(for: date) {
    features[FeatureIndex.stairs_down.rawValue] = Float(descentSpeed)
    count += 1
}
```

**HealthKitService addition:**
```swift
func fetchStairDescentSpeed(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .stairDescentSpeed)!
    return try await fetchDiscreteAverage(type: type, unit: HKUnit.meter().unitDivided(by: .second()), for: date)
}
```

---

## SUMMARY: Easy Wins

| # | Feature | HK Type | Est. Time |
|---|---------|---------|-----------|
| 1 | Walking Speed | walkingSpeed | 5 min |
| 2 | Step Length | walkingStepLength | 5 min |
| 3 | Gait Asymmetry | walkingAsymmetryPercentage | 5 min |
| 4 | Double Support | walkingDoubleSupportPercentage | 5 min |
| 5 | Blood Oxygen | oxygenSaturation | 5 min |
| 6 | Respiratory Rate | respiratoryRate | 5 min |
| 7 | VO2 Max | vo2Max | 2 min (exists) |
| 8 | Resting Energy | basalEnergyBurned | 2 min (exists) |
| 9 | Stand Minutes | appleStandTime | 5 min |
| 10 | Stair Ascent | stairAscentSpeed | 5 min |
| 11 | Stair Descent | stairDescentSpeed | 5 min |

**Total: ~50 minutes to add 11 new features**
**Result: 13 → 24 HealthKit features (85% increase!)**

---

## MEDIUM DIFFICULTY - Additional HealthKit Features

These require more complex queries or have limitations.

---

### 12. Six-Minute Walk Test (Sechs-Minuten-Gehtest)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierSixMinuteWalkTestDistance` |
| Unit | `meters` |
| Aggregation | `LAST_SAMPLE` |
| Feature Index | 35 (walk_test_distance) |
| Difficulty | ⭐⭐ Medium |
| Note | Only available if user performs test via Apple Health |

```swift
func fetchSixMinuteWalkTest(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .sixMinuteWalkTestDistance)!
    let samples = try await fetchQuantitySamples(type: type, for: date, limit: 1)
    guard let sample = samples.first else { return 0 }
    return sample.quantity.doubleValue(for: .meter())
}
```

---

### 13. Walking Heart Rate (Herzfrequenz beim Gehen)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierHeartRate` + workout filter |
| Unit | `bpm` |
| Aggregation | `AVG during walks` |
| Feature Index | 39 (walking_hr) |
| Difficulty | ⭐⭐ Medium |
| Note | Requires filtering HR samples during walking workouts |

```swift
func fetchWalkingHeartRate(for date: Date) async throws -> Double {
    // Complex: Need to correlate HR samples with walking workout times
    // Or use HKQuantityType.heartRate with workout predicate
    let hrType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
    // Filter for samples during walking periods
    return try await fetchDiscreteAverage(type: hrType, unit: HKUnit.count().unitDivided(by: .minute()), for: date)
}
```

---

### 14. Workout Sessions Count (Trainings)
| Property | Value |
|----------|-------|
| HK Identifier | `HKWorkoutType` |
| Unit | `count` |
| Aggregation | `DAILY_COUNT` |
| Feature Index | 50 (training_sessions) |
| Difficulty | ⭐⭐ Medium |

```swift
func fetchWorkoutCount(for date: Date) async throws -> Int {
    let workoutType = HKWorkoutType.workoutType()
    let calendar = Calendar.current
    let startOfDay = calendar.startOfDay(for: date)
    let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!
    let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

    return try await withCheckedThrowingContinuation { continuation in
        let query = HKSampleQuery(sampleType: workoutType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            if let error = error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume(returning: samples?.count ?? 0)
            }
        }
        healthStore.execute(query)
    }
}
```

---

### 15. Mindfulness Minutes (Achtsamkeit)
| Property | Value |
|----------|-------|
| HK Identifier | `HKCategoryTypeIdentifierMindfulSession` |
| Unit | `minutes` |
| Aggregation | `DAILY_SUM` |
| Feature Index | Not in current model - consider adding |
| Difficulty | ⭐⭐ Medium |

```swift
func fetchMindfulnessMinutes(for date: Date) async throws -> Double {
    let mindfulType = HKCategoryType.categoryType(forIdentifier: .mindfulSession)!
    let calendar = Calendar.current
    let startOfDay = calendar.startOfDay(for: date)
    let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!
    let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay, options: .strictStartDate)

    return try await withCheckedThrowingContinuation { continuation in
        let query = HKSampleQuery(sampleType: mindfulType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            if let error = error {
                continuation.resume(throwing: error)
            } else {
                let totalSeconds = (samples as? [HKCategorySample])?.reduce(0.0) { sum, sample in
                    sum + sample.endDate.timeIntervalSince(sample.startDate)
                } ?? 0
                continuation.resume(returning: totalSeconds / 60.0)
            }
        }
        healthStore.execute(query)
    }
}
```

---

### 16. Cycling Distance (Strecke Fahrrad)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierDistanceCycling` |
| Unit | `km` |
| Aggregation | `DAILY_SUM` |
| Feature Index | Not in current model |
| Difficulty | ⭐ Easy |

```swift
func fetchCyclingDistance(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .distanceCycling)!
    return try await fetchCumulativeSum(type: type, unit: .meterUnit(with: .kilo), for: date)
}
```

---

### 17. Swimming Distance (Strecke Schwimmen)
| Property | Value |
|----------|-------|
| HK Identifier | `HKQuantityTypeIdentifierDistanceSwimming` |
| Unit | `meters` |
| Aggregation | `DAILY_SUM` |
| Feature Index | Not in current model |
| Difficulty | ⭐ Easy |

```swift
func fetchSwimmingDistance(for date: Date) async throws -> Double {
    let type = HKQuantityType.quantityType(forIdentifier: .distanceSwimming)!
    return try await fetchCumulativeSum(type: type, unit: .meter(), for: date)
}
```

---

### 18. Sleep Consistency
| Property | Value |
|----------|-------|
| Source | Calculated from sleep data |
| Unit | `%` (0-100) |
| Aggregation | `7-day variance` |
| Feature Index | 61 (sleep_consistency) |
| Difficulty | ⭐⭐ Medium |

```swift
func calculateSleepConsistency(last7Days: [Double]) -> Double {
    guard last7Days.count >= 3 else { return 0 }
    let mean = last7Days.reduce(0, +) / Double(last7Days.count)
    let variance = last7Days.map { pow($0 - mean, 2) }.reduce(0, +) / Double(last7Days.count)
    let stdDev = sqrt(variance)
    // Lower std dev = higher consistency
    return max(0, min(100, 100 - (stdDev * 20)))
}
```

---

## CALCULATED / DERIVED FEATURES

These don't need external data - calculated from existing features.

| Index | Feature | Calculation | Status |
|-------|---------|-------------|--------|
| 17 | disease_activity_composite | Weighted sum of clinical scores | ⚠️ Needs clinical inputs first |
| 19 | pain_avg_24h | Average of logged pain values | ⚠️ Needs pain logging |
| 20 | pain_max_24h | Max of logged pain values | ⚠️ Needs pain logging |
| 30 | pain_variability | Std dev of pain over time | ⚠️ Needs pain logging |
| 60 | sleep_score | ✅ Already calculated (efficiency) | ✅ Working |
| 66 | mood_stability | Variance of mood over 7 days | ⚠️ Needs mood logging |
| 69 | stress_resilience | Calculated from stress patterns | ⚠️ Needs stress logging |
| 75 | depression_risk | PHQ-2 score mapping | ⚠️ Needs questionnaire |
| 80 | pressure_change | pressure[today] - pressure[yesterday] | 🟡 Easy to add |
| 82 | weather_change_score | Composite weather delta | 🟡 Easy to add |
| 91 | season | Calendar.current.component(.month) | 🟡 Easy to add |

---

## ENVIRONMENTAL / WEATHER FEATURES (7 total)

| Index | Feature | Source | Status |
|-------|---------|--------|--------|
| 76 | daylight_time | Calculate from location/date | 🟡 Medium |
| 77 | temperature | ✅ OpenMeteo | ✅ Working |
| 78 | humidity | ✅ OpenMeteo | ✅ Working |
| 79 | pressure | ✅ OpenMeteo | ✅ Working |
| 80 | pressure_change | Calculate: today - yesterday | 🟡 Easy |
| 81 | air_quality | Needs AQI API (OpenMeteo has it) | 🟡 Medium |
| 82 | weather_change_score | Composite calculation | 🟡 Medium |

### Pressure Change Implementation
```swift
func calculatePressureChange(today: Double, yesterday: Double) -> Double {
    return today - yesterday  // Negative = pressure dropping (potential trigger)
}
```

### Daylight Time Implementation
```swift
func calculateDaylightHours(for date: Date, latitude: Double, longitude: Double) -> Double {
    // Use astronomical calculations or Solar framework
    // Simplified: estimate based on month and latitude
    let month = Calendar.current.component(.month, from: date)
    let baseDaylight: [Int: Double] = [1: 8, 2: 9, 3: 11, 4: 13, 5: 15, 6: 16, 7: 16, 8: 14, 9: 12, 10: 10, 11: 9, 12: 8]
    return baseDaylight[month] ?? 12
}
```

---

## ADHERENCE FEATURES (5 total)

| Index | Feature | Source | How to Get |
|-------|---------|--------|------------|
| 83 | med_adherence | Core Data | Track medication log vs scheduled |
| 84 | physio_adherence | Core Data | Track exercise completion |
| 85 | physio_effectiveness | Core Data | User rates exercise helpfulness |
| 86 | journal_mood | Core Data | Mood from journal entries |
| 87 | quick_log | Core Data | Count of quick logs submitted |

### Med Adherence Calculation
```swift
func calculateMedAdherence(for date: Date, context: NSManagedObjectContext) -> Float {
    let request = MedicationLog.fetchRequest()
    let calendar = Calendar.current
    let startOfWeek = calendar.date(byAdding: .day, value: -7, to: date)!
    request.predicate = NSPredicate(format: "date >= %@ AND date <= %@", startOfWeek as NSDate, date as NSDate)

    let logs = try? context.fetch(request)
    let taken = logs?.filter { $0.taken }.count ?? 0
    let scheduled = logs?.count ?? 1
    return Float(taken) / Float(max(1, scheduled)) * 100
}
```

---

## REQUIRES USER INPUT - FULL LIST

### Clinical Assessment (12 features, indices 6-17)

| Index | Feature | How to Get | Priority | Difficulty |
|-------|---------|------------|----------|------------|
| 6 | **basdai_score** | 6-question BASDAI form | 🔴 CRITICAL | ⭐⭐ Medium |
| 7 | asdas_crp | Lab CRP value input | Low | ⭐ Easy input |
| 8 | basfi | 10-question BASFI form | Medium | ⭐⭐ Medium |
| 9 | basmi | Clinical measurement | 🚫 Impossible | N/A |
| 10 | patient_global | Single 0-10 slider | Medium | ⭐ Easy |
| 11 | physician_global | Doctor input | 🚫 Impossible | N/A |
| 12 | tender_joint_count | Body map tap count | Medium | ⭐⭐ Medium |
| 13 | swollen_joint_count | Body map tap count | Medium | ⭐⭐ Medium |
| 14 | enthesitis | Body map specific points | Medium | ⭐⭐ Medium |
| 15 | dactylitis | Finger/toe selection | Low | ⭐⭐ Medium |
| 16 | spinal_mobility | Clinical measurement | 🚫 Impossible | N/A |
| 17 | disease_activity_composite | Auto-calculated | Auto | N/A |

---

### Pain Characteristics (14 features, indices 18-31)

| Index | Feature | How to Get | Priority | Difficulty |
|-------|---------|------------|----------|------------|
| 18 | **pain_current** | Quick log slider (0-10) | 🔴 CRITICAL | ⭐ Easy |
| 19 | pain_avg_24h | Auto-calculated from logs | Auto | N/A |
| 20 | pain_max_24h | Auto-calculated from logs | Auto | N/A |
| 21 | nocturnal_pain | Morning check-in question | 🟡 HIGH | ⭐ Easy |
| 22 | **morning_stiffness_duration** | Quick log (minutes) | 🔴 CRITICAL | ⭐ Easy |
| 23 | morning_stiffness_severity | Quick log slider (0-10) | 🟡 HIGH | ⭐ Easy |
| 24 | pain_location_count | Auto from body map | Auto | N/A |
| 25 | pain_burning | Pain type checkbox | Low | ⭐ Easy |
| 26 | pain_aching | Pain type checkbox | Low | ⭐ Easy |
| 27 | pain_sharp | Pain type checkbox | Low | ⭐ Easy |
| 28 | pain_interference_sleep | Single slider | Medium | ⭐ Easy |
| 29 | pain_interference_activity | Single slider | Medium | ⭐ Easy |
| 30 | pain_variability | Auto-calculated (7-day std dev) | Auto | N/A |
| 31 | breakthrough_pain | Binary yes/no | Low | ⭐ Easy |

---

### Mental Health (12 features, indices 64-75)

| Index | Feature | How to Get | Priority | Difficulty |
|-------|---------|------------|----------|------------|
| 64 | **mood_current** | Emoji picker (5 options) | 🟡 HIGH | ⭐ Easy |
| 65 | mood_valence | Derived from mood (-1 to +1) | Auto | N/A |
| 66 | mood_stability | 7-day mood variance | Auto | N/A |
| 67 | anxiety | Single slider (0-10) | Medium | ⭐ Easy |
| 68 | **stress_level** | Single slider (0-10) | 🟡 HIGH | ⭐ Easy |
| 69 | stress_resilience | Calculated from patterns | Auto | N/A |
| 70 | mental_fatigue | Single slider (0-10) | Medium | ⭐ Easy |
| 71 | cognitive_function | 🚫 No objective measure | 🚫 Impossible | N/A |
| 72 | emotional_regulation | 🚫 Psych construct | 🚫 Impossible | N/A |
| 73 | social_engagement | Single slider or count | Low | ⭐ Easy |
| 74 | mental_wellbeing | Single slider (0-10) | Low | ⭐ Easy |
| 75 | depression_risk | PHQ-2 (2 questions) | Medium | ⭐⭐ Medium |

---

### Universal/Context (4 features, indices 88-91)

| Index | Feature | How to Get | Status |
|-------|---------|------------|--------|
| 88 | universal_assessment | Overall "how do you feel" slider | 🟡 Easy to add |
| 89 | time_weighted_assessment | Weighted average of day | Auto-calculated |
| 90 | ambient_noise | Microphone measurement | 🚫 Privacy concern |
| 91 | season | Calendar.current.component(.month) / 4 | ✅ Trivial |

---

---

## 🔄 CORRECTIONS & REVISED PLANS (Dec 2024)

Based on deeper HealthKit investigation and implementation strategy review.

### ✅ NOT IMPOSSIBLE - Available in HealthKit

| Index | Feature | German Name | HealthKit Identifier | Status |
|-------|---------|-------------|---------------------|--------|
| 35 | walk_test_distance | Sechs-Minuten-Gehtest | `HKQuantityTypeIdentifierSixMinuteWalkTestDistance` | ✅ Available |
| 39 | walking_hr | Durchschnittliche Herzfrequenz Gehen | `HKQuantityTypeIdentifier.heartRate` (filtered) | ✅ Available |
| 40 | cardio_recovery | Cardioerholung | `HKQuantityTypeIdentifierHeartRateRecoveryOneMinute` | ✅ ~80% feasible |
| 90 | ambient_noise | Umgebungslautstärke | `HKQuantityTypeIdentifierEnvironmentalAudioExposure` | ✅ Warnings only |

### ✅ NOT IMPOSSIBLE - User Input (Simple)

| Index | Feature | Implementation |
|-------|---------|----------------|
| 7 | asdas_crp | User enters CRP value from lab results |
| 9 | basmi | User enters measurement (can guide with instructions) |
| 11 | physician_global | User enters after doctor visit |
| 16 | spinal_mobility | User enters self-measurement |

### ✅ NOT IMPOSSIBLE - Derivable with Backend Logic

| Index | Feature | Implementation Strategy |
|-------|---------|------------------------|
| 66 | mood_stability | **Derive from:** mood entries + standing hours + heart rate + sleep time. Research medical evidence for correlation factors. Consider working person patterns (kids → less sleep, work → low standing hours + high pulse if mood unstable) |
| 69 | stress_resilience | **Backend logic:** Define stress level baseline, track key stress metrics. If person reports stress AND stress index shows +30% higher stress shortly after event → lower resilience. Use medical research for thresholds |
| 71 | cognitive_function | **Small survey:** "Can I think clearly today?", "Can I concentrate well?", "Am I able to do tasks outside of basic life things?" - Niche implementation |
| 72 | emotional_regulation | **Very small backend logic + tiny quiz:** Quick self-assessment questions about emotional responses |
| 75 | depression_risk | **Apple provides depression screening** in HealthKit. If Apple's own questions → use those. If open-source questionnaire → use trusted healthcare org (PHQ-2/PHQ-9) |

### 🔄 ADJUSTED FEATURES

| Index | Feature | Change |
|-------|---------|--------|
| 62 | exertion_level | Similar to "Body Battery" concept (like Garmin Body State app). This IS the body battery variable we want |
| 63 | burned_calories | **May not need** - already have active_energy + basal_energy (resting_energy). Redundant? |
| 73 | social_engagement | **REMOVE COMPLETELY** - not valuable for AS prediction |
| 81 | air_quality | Use external API **only if** same behavior as OpenMeteo (consistent data source) |

### 📊 REVISED "TRULY IMPOSSIBLE" LIST

After corrections, only these remain truly impossible:

| Index | Feature | Why Still Impossible |
|-------|---------|---------------------|
| ~~35~~ | ~~walk_test_distance~~ | ✅ MOVED - Available in HealthKit |
| ~~40~~ | ~~cardio_recovery~~ | ✅ MOVED - "Cardioerholung" exists |
| 73 | social_engagement | ❌ REMOVED - Not valuable, skip entirely |

**New Recommendation: 92 - 1 (removed) = 91 features possible with proper implementation**

---

## 🚫 TRULY IMPOSSIBLE FEATURES (Remove from Model)

| Index | Feature | Why Impossible |
|-------|---------|----------------|
| 73 | social_engagement | Not valuable for AS prediction - SKIP |

**Updated: Only 1 feature to remove → 91 realistic features**

---

## PRIORITY IMPLEMENTATION ORDER

### Phase 1: Quick Wins (1-2 hours)
1. Add 11 easy HealthKit features (documented above)
2. Add pressure_change calculation
3. Add season calculation

**Result: 13 → 26 features automatically collected**

### Phase 2: User Input MVP (2-3 hours)
4. Add "Quick Morning Check-in" screen:
   - Pain current (slider)
   - Morning stiffness (minutes)
   - Mood (emoji)
   - Stress (slider)

**Result: +4 critical user input features**

### Phase 3: Complete User Input (4-6 hours)
5. Implement BASDAI questionnaire
6. Connect body map to joint counts
7. Add remaining sliders

**Result: Full clinical feature coverage**

### Phase 4: Model Retrain
8. Retrain on 84-feature model (remove 8 impossible)
9. Train with 40-60% missing data simulation
10. Test with real user data

---

## FINAL FEATURE COUNT (REVISED Dec 2024)

| Category | Total | Extractable | Removed | Notes |
|----------|-------|-------------|---------|-------|
| Demographics | 6 | 6 | 0 | All via user input |
| Clinical | 12 | 12 | 0 | User input + auto-calc |
| Pain | 14 | 14 | 0 | User input + body map |
| Activity (HK) | 23 | 23 | 0 | +walk_test, +cardio_recovery now possible |
| Sleep (HK) | 9 | 9 | 0 | Full coverage |
| Mental Health | 12 | 11 | 1 | -social_engagement (removed) |
| Environmental | 7 | 7 | 0 | OpenMeteo + calculations |
| Adherence | 5 | 5 | 0 | Core Data tracking |
| Universal | 4 | 4 | 0 | +ambient_noise warnings |
| **TOTAL** | **92** | **91** | **1** | Only social_engagement removed |

### Implementation Breakdown:

| Source | Count | Features |
|--------|-------|----------|
| **HealthKit (automatic)** | ~28 | Activity, sleep, vitals, mobility, noise |
| **User Input (simple)** | ~25 | Pain sliders, mood, clinical values |
| **Backend Logic (derived)** | ~15 | Stability scores, resilience, composites |
| **Small Surveys** | ~5 | Cognitive, emotional, depression screening |
| **Weather API** | ~7 | Temperature, humidity, pressure, AQI |
| **Core Data (tracked)** | ~11 | Adherence, journal, body map counts |

**Realistic Model: 91 features with proper implementation**
**Removed: 1 (social_engagement - not valuable)**
