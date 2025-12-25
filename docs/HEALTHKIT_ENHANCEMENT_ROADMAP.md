# HealthKit Enhancement Roadmap

**Document Version:** 1.0
**Last Updated:** 2025-10-28
**Timeline:** 10-week phased implementation

---

## Overview

This roadmap outlines the three-phase approach to implementing comprehensive Apple Watch and HealthKit integration for InflamAI/InflamAI. Each phase builds on the previous, allowing for iterative deployment and user feedback integration.

---

## Phase 1: Foundation & MVP (Weeks 1-4)

**Goal:** Deploy minimal viable Watch app with core pain tracking and establish data pipeline

### Week 1: WatchOS Project Setup

**Tasks:**
- [ ] Create WatchOS app target in Xcode
- [ ] Configure app groups for data sharing between iPhone and Watch
- [ ] Set up HealthKit entitlements for Watch target
- [ ] Create shared framework for common models (SymptomLog, BodyRegion, etc.)

**Deliverables:**
```
InflamAI/
├── InflamAI-Watch/                    (NEW)
│   ├── InflamAI_WatchApp.swift
│   ├── ContentView.swift
│   └── Info.plist
├── InflamAI-Watch Extension/          (NEW)
│   └── ComplicationController.swift
└── Shared/                             (NEW)
    └── Models/
        ├── SharedSymptomLog.swift
        └── SharedBodyRegion.swift
```

**Technical Requirements:**
- WatchOS 9.0+ deployment target
- App Groups: `group.com.inflamai.shared`
- HealthKit background delivery enabled

---

### Week 2: Quick Pain Logger Complication

**Priority Feature:** Frictionless pain logging from watch face

**Implementation:**

1. **Complication Families:**
   - Circular Small: Pain emoji (😊 😐 😫)
   - Rectangular: Pain level + last log time
   - Graphic Corner: Pain trend graph
   - Graphic Circular: Pain level ring

2. **Pain Logging Flow:**
   ```
   User taps complication
   → Full-screen pain slider (0-10)
   → Optional body region quick select (tap mannequin)
   → Haptic confirmation
   → Sync to iPhone via WCSession
   ```

3. **Watch View (SwiftUI):**
   ```swift
   // InflamAI-Watch/PainLoggerView.swift
   struct PainLoggerView: View {
       @State private var painLevel: Double = 5
       @EnvironmentObject var watchConnectivity: WatchConnectivityManager

       var body: some View {
           VStack {
               Text("Pain Level")
                   .font(.headline)

               Slider(value: $painLevel, in: 0...10, step: 0.5)
                   .tint(painColor)

               Text("\(Int(painLevel))/10")
                   .font(.system(size: 48, weight: .bold))
                   .foregroundColor(painColor)

               Button("Log Pain") {
                   logPain()
               }
               .buttonStyle(.borderedProminent)
           }
           .digitalCrownRotation($painLevel, from: 0, through: 10, by: 0.5)
       }

       private func logPain() {
           watchConnectivity.sendPainLog(level: painLevel, timestamp: Date())
           WKInterfaceDevice.current().play(.success)
       }
   }
   ```

**Acceptance Criteria:**
- Complication updates within 15 seconds of pain log
- Digital Crown can adjust pain level
- Haptic feedback on successful log
- Data syncs to iPhone Core Data within 30 seconds

---

### Week 3: Background Biometric Collection

**Goal:** Continuous HRV and heart rate monitoring for correlation engine

**Implementation:**

1. **HealthKit Background Delivery:**
   ```swift
   // Shared/Services/BackgroundHealthMonitor.swift
   class BackgroundHealthMonitor {
       func enableBackgroundDelivery() async throws {
           // HRV updates
           try await healthStore.enableBackgroundDelivery(
               for: HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
               frequency: .hourly
           )

           // Resting heart rate
           try await healthStore.enableBackgroundDelivery(
               for: HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
               frequency: .hourly
           )
       }
   }
   ```

2. **Data Storage Strategy:**
   - Store raw biometric samples in Core Data (ContextSnapshot entity)
   - Aggregate hourly for correlation analysis
   - Prune data older than 90 days

3. **Watch → iPhone Sync:**
   ```swift
   // Use WCSession.transferUserInfo for reliability
   func syncBiometricData(_ data: [BiometricReading]) {
       let dict = data.map { $0.toDictionary() }
       WCSession.default.transferUserInfo(["biometrics": dict])
   }
   ```

**Performance Targets:**
- Battery impact: < 5% additional drain per day
- Sync latency: < 2 minutes for non-urgent data
- Storage: < 10MB per week of biometric data

---

### Week 4: Medication Reminders

**Goal:** Wrist-based medication notifications with quick confirmation

**Features:**

1. **Local Notification on Watch:**
   ```swift
   let content = UNMutableNotificationContent()
   content.title = "💊 Medication Reminder"
   content.body = "Time to take Humira (40mg injection)"
   content.categoryIdentifier = "MEDICATION_REMINDER"
   content.sound = .default

   // Actions
   let takenAction = UNNotificationAction(
       identifier: "MEDICATION_TAKEN",
       title: "Mark Taken",
       options: .foreground
   )
   let snoozeAction = UNNotificationAction(
       identifier: "MEDICATION_SNOOZE",
       title: "Snooze 30min",
       options: []
   )
   ```

2. **Adherence Tracking:**
   - Record timestamp when "Mark Taken" tapped
   - Sync to iPhone DoseLog Core Data entity
   - Calculate weekly adherence percentage

3. **Smart Scheduling:**
   - Skip notifications if Watch detects sleep
   - Adjust timing based on wake time patterns
   - Escalate to phone if no response after 15 minutes

**Acceptance Criteria:**
- Notification appears on watch face
- One-tap confirmation without opening app
- Adherence data visible in iPhone trends view
- 90%+ notification delivery rate

---

### Phase 1 Deliverables Summary

✅ **By End of Week 4:**
- Working WatchOS app with pain logging
- Complications on watch face
- Background biometric monitoring (HRV, HR)
- Medication reminders with adherence tracking
- Basic watch ↔ phone sync

**User-Facing Value:**
- 25-second pain logging (vs 60+ seconds on phone)
- 2-3x more biometric data for correlations
- 65%+ medication adherence (up from 40%)

**Metrics to Track:**
- Watch app installations (target: 30% of iPhone users)
- Daily watch interactions (target: 5+ per user)
- Pain logs per week (target: 2x increase from manual)

---

## Phase 2: Advanced Monitoring (Weeks 5-7)

**Goal:** Enhanced correlation engine with predictive capabilities

### Week 5: Sleep Stage Analysis

**Enhancement to existing HealthKitService.swift:57**

**Current Implementation:**
```swift
func fetchSleepData(for date: Date) async throws -> SleepData {
    // Only gets total sleep duration and basic efficiency
}
```

**Enhanced Implementation:**
```swift
struct EnhancedSleepData {
    let totalDuration: TimeInterval
    let deepSleepDuration: TimeInterval      // NEW
    let remSleepDuration: TimeInterval       // NEW
    let coreSleepDuration: TimeInterval      // NEW
    let awakeDuration: TimeInterval          // NEW
    let efficiency: Double
    let quality: Int
    let sleepDebt: TimeInterval              // NEW - cumulative deficit
    let consistency: Double                   // NEW - bedtime variability
}
```

**Sleep Stages → AS Correlation:**
```swift
// Add to CorrelationEngine.swift
private func analyzeSleepStageTriggers(logs: [SymptomLog], pain: [Double]) -> [Trigger] {
    let deepSleepPercentages = logs.compactMap { log in
        guard let sleep = log.enhancedSleepData else { return nil }
        return (sleep.deepSleepDuration / sleep.totalDuration) * 100
    }

    if let r = pearsonCorrelation(deepSleepPercentages, pain) {
        // Expected: negative correlation (more deep sleep = less pain)
        return [Trigger(
            name: "Deep Sleep Deficiency",
            category: .biometric,
            correlation: -r,  // Invert so positive = bad
            pValue: calculatePValue(r: r, n: deepSleepPercentages.count),
            lag: 0,
            icon: "bed.double.fill"
        )]
    }
}
```

**Features:**
- Deep sleep % vs morning stiffness correlation
- REM sleep vs mood/fatigue correlation
- Sleep debt calculation (cumulative)
- Overnight heart rate dip analysis

**Expected Insights:**
- "You need 18% deep sleep (currently 12%) to minimize morning stiffness"
- "Your sleep debt is 3.5 hours - catch up this weekend"

---

### Week 6: Activity Pattern Correlation

**Intraday Activity Analysis:**

1. **Standing Hours Tracking:**
   ```swift
   // CorrelationEngine enhancement
   func analyzeStandingPatterns(logs: [SymptomLog]) -> [Trigger] {
       let standHours = logs.compactMap { $0.contextSnapshot?.standHours }
       let pain = logs.compactMap { $0.basdaiScore }

       // Hypothesis: More standing hours = less next-day stiffness
       if let r = pearsonCorrelation(standHours, pain) {
           return [Trigger(
               name: "Standing Hours",
               category: .activity,
               correlation: r,
               pValue: calculatePValue(r: r, n: standHours.count),
               lag: 24,  // Next-day effect
               icon: "figure.stand"
           )]
       }
   }
   ```

2. **Exercise Response Profiling:**
   ```swift
   struct ExerciseResponse {
       let workout: HKWorkout
       let immediateHeartRateElevation: Double
       let painLevel24hLater: Double
       let painLevel48hLater: Double

       var isWithinTolerance: Bool {
           // If pain increases > 2 points, exercise was too intense
           return painLevel24hLater - workout.prePainLevel < 2.0
       }
   }
   ```

3. **Movement Micro-Breaks:**
   - Detect prolonged sitting periods (> 60 min)
   - Correlate with increased stiffness
   - Send Watch "Time to move" reminders

**Acceptance Criteria:**
- Detect optimal standing hours for each user
- Identify exercise intensity threshold
- Recommend personalized movement breaks

---

### Week 7: Predictive Flare Alerts

**Goal:** 12-24 hour advance warning of AS flares

**Flare Prediction Algorithm:**

```swift
// Core/AI/FlarePredictor.swift
class FlarePredictor {
    func assessFlareRisk(biometrics: [BiometricSnapshot],
                         recentPain: [Double]) async -> FlareRisk {

        // Feature extraction
        let hrv24hTrend = calculate24HourTrend(biometrics.map { $0.hrvValue })
        let restingHRChange = calculateBaselineDeviation(biometrics.map { $0.restingHeartRate })
        let sleepQuality = biometrics.last?.sleepData?.quality ?? 5
        let activityReduction = calculateActivityDrop(biometrics)

        // Scoring model
        var riskScore = 0.0

        // HRV drop is strongest predictor
        if hrv24hTrend < -0.15 {  // 15% drop
            riskScore += 0.4
        }

        // Elevated resting heart rate
        if restingHRChange > 10 {  // +10 bpm above baseline
            riskScore += 0.3
        }

        // Poor sleep
        if sleepQuality < 5 {
            riskScore += 0.2
        }

        // Reduced activity (pain avoidance behavior)
        if activityReduction > 0.3 {  // 30% reduction
            riskScore += 0.1
        }

        // Classify risk
        if riskScore >= 0.7 {
            return .high(predictedOnset: Date().addingTimeInterval(12 * 3600))
        } else if riskScore >= 0.4 {
            return .moderate(predictedOnset: Date().addingTimeInterval(24 * 3600))
        } else {
            return .low
        }
    }
}

enum FlareRisk {
    case low
    case moderate(predictedOnset: Date)
    case high(predictedOnset: Date)
}
```

**User Notification:**
```
🔴 Flare Risk Alert
Based on your biometrics, you may experience increased pain in the next 12-24 hours.

Suggestions:
• Take preventive NSAID now if approved by doctor
• Reduce activity intensity tomorrow
• Prioritize 8+ hours sleep tonight
• Apply heat therapy before bed
```

**Validation Approach:**
- Track prediction accuracy over 30 days
- Target: 70% sensitivity, 80% specificity
- Improve model with user feedback ("Did you experience a flare?")

---

### Phase 2 Deliverables Summary

✅ **By End of Week 7:**
- Sleep stage correlation analysis
- Intraday activity pattern detection
- Predictive flare alerts (12-24h advance)
- Enhanced correlation engine with 15+ metrics

**User-Facing Value:**
- "Your deep sleep dropped to 10% - expect higher stiffness tomorrow"
- "Standing 10+ hours reduces your pain by 35%"
- "Flare predicted in 18 hours - take preventive action"

**Metrics to Track:**
- Flare prediction accuracy (target: 70%)
- User action on alerts (target: 60% take preventive measures)
- Correlation discoveries per user (target: 3+ statistically significant)

---

## Phase 3: Premium Features (Weeks 8-10)

**Goal:** Market-leading differentiation and research capabilities

### Week 8: Full Complication Suite

**Expand beyond pain logging to comprehensive watch face integration**

**Complication Types:**

1. **Modular Large:**
   ```
   ┌─────────────────────┐
   │ BASDAI Score: 4.2   │
   │ ↓ 0.8 from last week│
   │ Next med: 2h 15m    │
   └─────────────────────┘
   ```

2. **Graphic Bezel:**
   - Circular pain trend (7 days)
   - Color-coded by severity

3. **Graphic Rectangular:**
   ```
   ┌──────────────────────────┐
   │ 📊 Pain: 5/10            │
   │ 💊 Meds: ✅ 2/2 today    │
   │ 🏃 Steps: 6,234          │
   └──────────────────────────┘
   ```

4. **Graphic Corner:**
   - Medication countdown timer
   - Flare risk indicator (🟢🟡🔴)

**Implementation:**
- Support all iOS 16+ complication families
- Real-time updates via WidgetKit
- Background refresh every 15 minutes

---

### Week 9: Research Study Mode

**Goal:** Enable clinical research partnerships and publishable data

**IRB-Compliant Data Collection:**

```swift
// Core/Research/StudyProtocol.swift
struct StudyProtocol {
    let id: UUID
    let title: String
    let institution: String
    let piName: String
    let consentText: String

    let dataTypes: Set<StudyDataType>
    let duration: TimeInterval  // e.g., 90 days
    let minimumComplianceRate: Double  // e.g., 0.8 = 80% daily logs

    enum StudyDataType {
        case painLogs
        case biometrics(types: [HKQuantityTypeIdentifier])
        case medications
        case environmentalFactors
        case deidentifiedDemographics
    }
}
```

**Features:**
- Enhanced data export (CSV, JSON, HL7 FHIR)
- Participant compliance tracking
- Automated reminders for study requirements
- Secure data transmission to research portals

**Research Value:**
- Partner with rheumatology research centers
- Publish correlations in peer-reviewed journals
- Validate BASDAI/ASDAS calculators against clinical data
- Study medication effectiveness in real-world settings

**Privacy Protections:**
- Explicit research consent separate from app usage
- Data anonymization (strip PII)
- Right to withdraw from study anytime
- Local storage with optional cloud upload

---

### Week 10: Machine Learning Predictor (Advanced)

**Goal:** Replace correlation-based triggers with ML-powered predictions

**Current Approach (Pearson Correlation):**
- Linear relationships only
- Single-variable analysis
- Manual threshold setting

**ML Approach (Random Forest / Gradient Boosting):**
- Non-linear pattern detection
- Multi-variate feature interactions
- Adaptive threshold learning

**Implementation:**

```swift
// Core/AI/MLFlarePredictor.swift
import CoreML

class MLFlarePredictor {
    private let model: FlarePredictionModel  // CoreML model

    func predictFlareRisk(features: BiometricFeatures) async -> FlarePrediction {
        // Extract 50+ features
        let input = FlarePredictionModelInput(
            hrv_24h_mean: features.hrvMean,
            hrv_24h_std: features.hrvStd,
            resting_hr_change: features.restingHRChange,
            deep_sleep_pct: features.deepSleepPercentage,
            steps_daily: features.stepCount,
            barometric_pressure_delta: features.pressureChange,
            // ... 44 more features
        )

        let prediction = try? model.prediction(input: input)
        return FlarePrediction(
            probability: prediction?.flareProbability ?? 0,
            timeToFlare: prediction?.hoursUntilFlare ?? nil,
            topContributors: prediction?.featureImportance ?? []
        )
    }
}
```

**Training Data Requirements:**
- 500+ users with 30+ days of data each
- Labeled flare events (patient-reported)
- Feature engineering from existing CorrelationEngine

**Expected Performance:**
- 75-80% accuracy (vs 70% for correlation-based)
- Personalized per-user model adaptation
- Explain predictions: "Top factors: HRV drop (40%), poor sleep (30%)"

---

### Phase 3 Deliverables Summary

✅ **By End of Week 10:**
- Complete complication suite (all watch faces)
- Research study mode with IRB-compliant export
- ML-powered flare predictor (optional, based on data availability)
- Market-leading AS management platform

**User-Facing Value:**
- Watch face shows all health data at a glance
- Participate in cutting-edge AS research
- AI predictions improve over time with personal data

**Business Value:**
- Research partnerships with hospitals
- Peer-reviewed publications (credibility)
- Premium subscription tier ($9.99/month) justified
- FDA clearance pathway (if pursuing medical device status)

---

## Post-Launch: Continuous Improvement

### Month 2-3: User Feedback Integration

**Monitoring:**
- App Store reviews (target: maintain 4.5+ stars)
- In-app feedback (NPS surveys)
- Support tickets (common pain points)

**Common Requests to Anticipate:**
- Battery life concerns → optimize background tasks
- Sync delays → implement retry logic
- Complication not updating → debug WidgetKit refresh

### Month 4-6: Advanced Analytics

**New Features Based on Data:**
- Medication effectiveness analysis (compare BASDAI before/after biologic changes)
- Weather pattern clustering (identify local micro-climates)
- Social features (anonymized comparison to other AS patients)

### Month 7-12: Ecosystem Expansion

**Integration Opportunities:**
- Health Records (import lab results like CRP, ESR)
- Apple Health Medications (iOS 16+ medication tracking)
- Research Kit studies
- HealthKit sharing with healthcare providers

---

## Resource Requirements

### Development Team

| Role | Phase 1 | Phase 2 | Phase 3 | Total Weeks |
|------|---------|---------|---------|-------------|
| iOS Developer (Swift/SwiftUI) | 4 | 3 | 3 | 10 |
| WatchOS Specialist | 4 | 2 | 3 | 9 |
| Backend Engineer (sync/storage) | 2 | 2 | 2 | 6 |
| Data Scientist (ML, optional) | - | - | 3 | 3 |
| QA Engineer | 2 | 2 | 2 | 6 |
| **Total effort** | **12 weeks** | **9 weeks** | **13 weeks** | **34 weeks** |

**Note:** Overlapping work possible - total calendar time can be 10-12 weeks with parallel execution.

### Infrastructure Costs

| Component | Monthly Cost | Notes |
|-----------|-------------|-------|
| CloudKit storage | $50-200 | Scales with users |
| HealthKit (free) | $0 | Apple framework |
| CoreML hosting | $0 | On-device inference |
| Development certificates | $99/year | Apple Developer Program |
| **Total** | ~$50-200/month | Minimal incremental cost |

---

## Risk Mitigation Strategies

### Technical Risks

1. **Battery Drain Concerns**
   - Mitigation: Adaptive sampling rates, batch processing
   - Fallback: User-controlled monitoring intensity

2. **Sync Failures**
   - Mitigation: Queue-based retry logic, local-first architecture
   - Fallback: Manual sync button in settings

3. **HealthKit Permission Rejection**
   - Mitigation: Progressive disclosure, clear value prop
   - Fallback: App works without Watch (graceful degradation)

### Business Risks

1. **Low Watch Adoption**
   - Mitigation: Market research shows 40% of iPhone users own Watch
   - Fallback: Phone-only app still valuable

2. **Competitor Response**
   - Mitigation: File patents on correlation algorithms
   - Fallback: 6-12 month lead time advantage

3. **Regulatory Scrutiny**
   - Mitigation: Wellness app positioning, not medical device
   - Fallback: Obtain FDA clearance if needed (Digital Health Pre-Cert)

---

## Success Criteria

### Phase 1 Success (Week 4)
- ✅ 30% of iPhone users install Watch app
- ✅ 5+ daily Watch interactions per user
- ✅ 90% pain log data completeness
- ✅ < 5% battery impact reported

### Phase 2 Success (Week 7)
- ✅ 70% flare prediction accuracy
- ✅ 3+ statistically significant correlations discovered per user
- ✅ 4.5+ star App Store rating maintained

### Phase 3 Success (Week 10)
- ✅ Research partnership signed (1+ hospital)
- ✅ Premium subscription conversion > 20%
- ✅ Featured in App Store Health & Fitness section

---

## Next Steps

1. **Stakeholder Review** - Present roadmap to product/business teams
2. **Budget Approval** - Secure 10-week development budget
3. **Hire WatchOS Developer** - Critical path dependency
4. **Create Detailed Sprint Plan** - Break down each week into 2-day sprints
5. **Begin Phase 1 Week 1** - WatchOS project setup

---

**Document Status:** ✅ Ready for execution
**Owner:** Engineering Team
**Last Review:** 2025-10-28
