# Apple Watch App Technical Specification

**Version**: 1.0
**Last Updated**: October 28, 2025
**Target watchOS**: 10.0+
**Target iOS**: 17.0+

---

## Executive Summary

### 🎯 Watch App Purpose
A **standalone + companion** WatchOS app that enables:
1. **Quick symptom logging** (3 taps, <10 seconds)
2. **Medication reminders** with one-tap confirmation
3. **Real-time biometric monitoring** (HR, HRV, activity)
4. **Complication widgets** (medication countdown, pain level, activity rings)
5. **Background health data sync** to iPhone app

### 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      iPhone App (iOS 17+)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  SwiftUI Views                                         │ │
│  │  - HomeView, TrendsView, MedicationView                │ │
│  │  - SymptomLogDetailView, QuestionnaireView             │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ViewModels (@MainActor)                               │ │
│  │  - SymptomLogViewModel, MedicationViewModel            │ │
│  │  - HealthKitViewModel, CorrelationViewModel            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Services                                              │ │
│  │  - PersistenceController (Core Data + CloudKit)        │ │
│  │  - HealthKitService (continuous monitoring)            │ │
│  │  - WeatherService, NotificationService                 │ │
│  │  - WatchConnectivityService ← NEW                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                             ↕                               │
│              WatchConnectivity Framework                    │
│         (Application Context + Messages + Files)            │
└──────────────────────────────┬──────────────────────────────┘
                               ↕
┌─────────────────────────────────────────────────────────────┐
│                   Apple Watch App (watchOS 10+)              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Watch SwiftUI Views                                   │ │
│  │  - QuickLogView, MedicationTrackerView                 │ │
│  │  - BiometricsView, ComplicationsView                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Watch ViewModels                                      │ │
│  │  - WatchSymptomViewModel, WatchMedicationViewModel     │ │
│  │  - WatchHealthViewModel                                │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Watch Services                                        │ │
│  │  - WatchConnectivityManager (receives/sends data)      │ │
│  │  - WatchHealthKitService (local queries)               │ │
│  │  - WatchPersistenceController (local Core Data)        │ │
│  │  - ComplicationDataSource ← NEW                        │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Complications (Widget Extensions)                     │ │
│  │  - Circular, Modular, Graphic Circular                 │ │
│  │  - Shows: Medication countdown, Pain level, Activity   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Project Structure

### 1.1 Xcode Targets

**Current State**:
- ✅ `InflamAI` (iOS app)

**Required Additions**:
1. **WatchOS App Target**: `InflamAI Watch App`
2. **WatchOS Extension**: `InflamAI Watch App Extension` (if needed for complications)
3. **Shared Framework**: `InflamAICore` (shared models/services)

### 1.2 Directory Structure

```
InflamAI/
├── InflamAI/              # Existing iOS app
│   ├── Features/
│   ├── Core/
│   └── ...
│
├── InflamAI-Watch/        # NEW: Watch app
│   ├── Views/
│   │   ├── QuickLogView.swift
│   │   ├── MedicationTrackerView.swift
│   │   ├── BiometricsView.swift
│   │   └── SettingsView.swift
│   ├── ViewModels/
│   │   ├── WatchSymptomViewModel.swift
│   │   ├── WatchMedicationViewModel.swift
│   │   └── WatchHealthViewModel.swift
│   ├── Services/
│   │   ├── WatchConnectivityManager.swift
│   │   ├── WatchHealthKitService.swift
│   │   └── WatchPersistenceController.swift
│   ├── Complications/
│   │   ├── ComplicationController.swift
│   │   └── ComplicationViews.swift
│   └── Assets.xcassets
│
├── InflamAICore/                # NEW: Shared framework
│   ├── Models/
│   │   ├── SymptomLog.swift      # Move from iOS app
│   │   ├── Medication.swift
│   │   ├── HealthMetric.swift
│   │   └── WatchMessage.swift    # NEW
│   ├── Services/
│   │   └── ConnectivityProtocol.swift  # NEW
│   └── Utilities/
│       └── DateExtensions.swift
│
└── InflamAI.xcodeproj
```

### 1.3 Shared Core Data Model

**File**: [InflamAI/InflamAI.xcdatamodeld](../InflamAI/InflamAI.xcdatamodeld)

**Required Changes**:
1. **Enable CloudKit sync** (already configured with `NSPersistentCloudKitContainer`)
2. **Add Watch-specific entities**:

```swift
// NEW: WatchQuickLog (temporary cache before full sync)
entity WatchQuickLog {
    @NSManaged var id: UUID
    @NSManaged var timestamp: Date
    @NSManaged var painScore: Int16
    @NSManaged var stiffnessScore: Int16
    @NSManaged var fatigueScore: Int16
    @NSManaged var synced: Bool  // False until synced to iPhone
    @NSManaged var source: String  // "watch_quick_log"
}

// NEW: WatchMedicationLog (medication taken confirmations)
entity WatchMedicationLog {
    @NSManaged var id: UUID
    @NSManaged var medicationID: UUID
    @NSManaged var timestamp: Date
    @NSManaged var synced: Bool
}
```

**Sync Strategy**:
- **iPhone → Watch**: Application Context (every app launch)
- **Watch → iPhone**: Interactive Messages (immediate) + Background Transfer (files)
- **Conflict Resolution**: Server timestamp wins (iPhone is source of truth)

---

## 2. WatchOS App Features

### 2.1 Quick Symptom Logging

**UI Flow**:
```
[Watch Face] → Tap complication
    ↓
[Quick Log View]
 ┌────────────────────────┐
 │  Quick Log             │
 │                        │
 │  Pain:    ●●●○○ (3/5)  │
 │  Stiffness: ●●●●○ (4/5)│
 │  Fatigue:  ●●○○○ (2/5) │
 │                        │
 │  [Log] [Cancel]        │
 └────────────────────────┘
    ↓ Tap [Log]
[Confirmation]
 "Logged at 2:34 PM ✓"
    ↓ Auto-dismiss after 1.5s
[Watch Face]
```

**Implementation**:

```swift
// InflamAI-Watch/Views/QuickLogView.swift

import SwiftUI
import WatchKit

struct QuickLogView: View {
    @StateObject private var viewModel = WatchSymptomViewModel()
    @Environment(\.dismiss) private var dismiss

    @State private var painScore: Int = 0
    @State private var stiffnessScore: Int = 0
    @State private var fatigueScore: Int = 0
    @State private var showConfirmation = false

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                Text("Quick Log")
                    .font(.headline)
                    .padding(.top, 4)

                // Pain
                VStack(alignment: .leading, spacing: 4) {
                    Text("Pain")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        ForEach(0..<5) { index in
                            Circle()
                                .fill(index < painScore ? Color.red : Color.gray.opacity(0.3))
                                .frame(width: 24, height: 24)
                                .onTapGesture {
                                    painScore = index + 1
                                    WKInterfaceDevice.current().play(.click)
                                }
                        }
                    }
                }

                // Stiffness
                VStack(alignment: .leading, spacing: 4) {
                    Text("Stiffness")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        ForEach(0..<5) { index in
                            Circle()
                                .fill(index < stiffnessScore ? Color.orange : Color.gray.opacity(0.3))
                                .frame(width: 24, height: 24)
                                .onTapGesture {
                                    stiffnessScore = index + 1
                                    WKInterfaceDevice.current().play(.click)
                                }
                        }
                    }
                }

                // Fatigue
                VStack(alignment: .leading, spacing: 4) {
                    Text("Fatigue")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        ForEach(0..<5) { index in
                            Circle()
                                .fill(index < fatigueScore ? Color.blue : Color.gray.opacity(0.3))
                                .frame(width: 24, height: 24)
                                .onTapGesture {
                                    fatigueScore = index + 1
                                    WKInterfaceDevice.current().play(.click)
                                }
                        }
                    }
                }

                // Buttons
                HStack(spacing: 12) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .buttonStyle(.bordered)
                    .tint(.gray)

                    Button("Log") {
                        Task {
                            await viewModel.logSymptoms(
                                pain: painScore,
                                stiffness: stiffnessScore,
                                fatigue: fatigueScore
                            )
                            showConfirmation = true
                            WKInterfaceDevice.current().play(.success)

                            // Auto-dismiss after 1.5 seconds
                            try? await Task.sleep(for: .seconds(1.5))
                            dismiss()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(painScore == 0 && stiffnessScore == 0 && fatigueScore == 0)
                }
                .padding(.top, 8)
            }
            .padding()
        }
        .alert("Logged ✓", isPresented: $showConfirmation) {
            // No buttons - auto-dismisses
        }
    }
}

// ViewModel
@MainActor
class WatchSymptomViewModel: ObservableObject {
    private let connectivityManager = WatchConnectivityManager.shared
    private let persistence = WatchPersistenceController.shared

    func logSymptoms(pain: Int, stiffness: Int, fatigue: Int) async {
        // 1. Save locally
        let log = WatchQuickLog(context: persistence.container.viewContext)
        log.id = UUID()
        log.timestamp = Date()
        log.painScore = Int16(pain)
        log.stiffnessScore = Int16(stiffness)
        log.fatigueScore = Int16(fatigue)
        log.synced = false
        log.source = "watch_quick_log"

        try? persistence.container.viewContext.save()

        // 2. Send to iPhone immediately
        let message: [String: Any] = [
            "type": "symptom_log",
            "id": log.id.uuidString,
            "timestamp": log.timestamp.timeIntervalSince1970,
            "pain": pain,
            "stiffness": stiffness,
            "fatigue": fatigue
        ]

        await connectivityManager.sendMessage(message)
    }
}
```

**Performance Requirements**:
- **Launch time**: <1 second from complication tap
- **Log time**: <10 seconds from launch to confirmation
- **Total interaction**: <15 seconds
- **Offline support**: Queue logs locally, sync when phone reachable

---

### 2.2 Medication Tracker

**UI Flow**:
```
[Watch Face] → Medication due notification
    ↓ Tap notification
[Medication View]
 ┌────────────────────────┐
 │  Time for:             │
 │  💊 Humira (Adalimumab)│
 │                        │
 │  Scheduled: 8:00 AM    │
 │  Current:   8:05 AM    │
 │                        │
 │  [Taken] [Skip]        │
 │  [Remind me in 15min]  │
 └────────────────────────┘
```

**Implementation**:

```swift
// InflamAI-Watch/Views/MedicationTrackerView.swift

import SwiftUI
import UserNotifications

struct MedicationTrackerView: View {
    @StateObject private var viewModel = WatchMedicationViewModel()

    var body: some View {
        List {
            if viewModel.dueMedications.isEmpty {
                Text("No medications due")
                    .foregroundColor(.secondary)
            } else {
                ForEach(viewModel.dueMedications) { medication in
                    MedicationRowView(
                        medication: medication,
                        onTaken: {
                            Task {
                                await viewModel.markTaken(medication)
                            }
                        },
                        onSkip: {
                            Task {
                                await viewModel.markSkipped(medication)
                            }
                        },
                        onSnooze: {
                            Task {
                                await viewModel.snooze(medication, minutes: 15)
                            }
                        }
                    )
                }
            }

            Section("Upcoming") {
                ForEach(viewModel.upcomingMedications) { medication in
                    HStack {
                        Text(medication.name)
                            .font(.caption)
                        Spacer()
                        Text(medication.scheduledTime, style: .time)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Medications")
        .onAppear {
            Task {
                await viewModel.loadMedications()
            }
        }
    }
}

struct MedicationRowView: View {
    let medication: WatchMedication
    let onTaken: () -> Void
    let onSkip: () -> Void
    let onSnooze: () -> Void

    @State private var showActions = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(medication.name)
                .font(.headline)

            HStack {
                Text("Due: \(medication.scheduledTime, style: .time)")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                if medication.isOverdue {
                    Text("OVERDUE")
                        .font(.caption2)
                        .foregroundColor(.red)
                        .bold()
                }
            }

            HStack(spacing: 8) {
                Button {
                    onTaken()
                    WKInterfaceDevice.current().play(.success)
                } label: {
                    Label("Taken", systemImage: "checkmark.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)

                Button {
                    showActions = true
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.vertical, 4)
        .confirmationDialog("Medication Actions", isPresented: $showActions) {
            Button("Skip this dose") {
                onSkip()
            }
            Button("Remind in 15 min") {
                onSnooze()
            }
            Button("Cancel", role: .cancel) {}
        }
    }
}

// ViewModel
@MainActor
class WatchMedicationViewModel: ObservableObject {
    @Published var dueMedications: [WatchMedication] = []
    @Published var upcomingMedications: [WatchMedication] = []

    private let connectivityManager = WatchConnectivityManager.shared

    func loadMedications() async {
        // Fetch from local context synced from iPhone
        // TODO: Implement Core Data query
        dueMedications = []  // Placeholder
        upcomingMedications = []
    }

    func markTaken(_ medication: WatchMedication) async {
        // Send to iPhone
        let message: [String: Any] = [
            "type": "medication_taken",
            "id": medication.id.uuidString,
            "timestamp": Date().timeIntervalSince1970
        ]

        await connectivityManager.sendMessage(message)

        // Update local UI
        dueMedications.removeAll { $0.id == medication.id }
    }

    func markSkipped(_ medication: WatchMedication) async {
        let message: [String: Any] = [
            "type": "medication_skipped",
            "id": medication.id.uuidString,
            "timestamp": Date().timeIntervalSince1970
        ]

        await connectivityManager.sendMessage(message)
        dueMedications.removeAll { $0.id == medication.id }
    }

    func snooze(_ medication: WatchMedication, minutes: Int) async {
        // Schedule local notification
        let content = UNMutableNotificationContent()
        content.title = "Medication Reminder"
        content.body = medication.name
        content.sound = .default

        let trigger = UNTimeIntervalNotificationTrigger(
            timeInterval: TimeInterval(minutes * 60),
            repeats: false
        )

        let request = UNNotificationRequest(
            identifier: medication.id.uuidString,
            content: content,
            trigger: trigger
        )

        try? await UNUserNotificationCenter.current().add(request)

        // Remove from due list temporarily
        dueMedications.removeAll { $0.id == medication.id }
    }
}

struct WatchMedication: Identifiable {
    let id: UUID
    let name: String
    let scheduledTime: Date
    var isOverdue: Bool {
        Date() > scheduledTime.addingTimeInterval(15 * 60)  // 15 min grace period
    }
}
```

---

### 2.3 Biometrics Dashboard

**UI**:
```
[Biometrics View]
 ┌────────────────────────┐
 │  Current Metrics       │
 │                        │
 │  ❤️  72 bpm            │
 │  📊 HRV: 45 ms         │
 │  🚶 4,230 steps        │
 │  🔥 320 cal            │
 │                        │
 │  [Refresh]             │
 └────────────────────────┘
```

**Implementation**:

```swift
// InflamAI-Watch/Views/BiometricsView.swift

import SwiftUI
import HealthKit

struct BiometricsView: View {
    @StateObject private var viewModel = WatchHealthViewModel()

    var body: some View {
        List {
            Section("Current") {
                MetricRow(
                    icon: "heart.fill",
                    color: .red,
                    title: "Heart Rate",
                    value: viewModel.heartRate.map { "\(Int($0)) bpm" } ?? "--"
                )

                MetricRow(
                    icon: "waveform.path.ecg",
                    color: .blue,
                    title: "HRV",
                    value: viewModel.hrv.map { "\(Int($0)) ms" } ?? "--"
                )
            }

            Section("Today") {
                MetricRow(
                    icon: "figure.walk",
                    color: .green,
                    title: "Steps",
                    value: viewModel.steps.map { "\(Int($0))" } ?? "--"
                )

                MetricRow(
                    icon: "flame.fill",
                    color: .orange,
                    title: "Active Energy",
                    value: viewModel.activeEnergy.map { "\(Int($0)) cal" } ?? "--"
                )
            }

            Button {
                Task {
                    await viewModel.refresh()
                }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
        }
        .navigationTitle("Biometrics")
        .onAppear {
            Task {
                await viewModel.startMonitoring()
            }
        }
    }
}

struct MetricRow: View {
    let icon: String
    let color: Color
    let title: String
    let value: String

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24)

            VStack(alignment: .leading) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.body)
                    .bold()
            }

            Spacer()
        }
    }
}

// ViewModel
@MainActor
class WatchHealthViewModel: ObservableObject {
    @Published var heartRate: Double?
    @Published var hrv: Double?
    @Published var steps: Double?
    @Published var activeEnergy: Double?

    private let healthStore = HKHealthStore()

    func startMonitoring() async {
        await requestAuthorization()
        await refresh()

        // Start continuous HR monitoring
        startHeartRateQuery()
    }

    func refresh() async {
        async let hr = fetchLatestHeartRate()
        async let hrvValue = fetchLatestHRV()
        async let stepsValue = fetchTodaySteps()
        async let energyValue = fetchTodayActiveEnergy()

        heartRate = await hr
        hrv = await hrvValue
        steps = await stepsValue
        activeEnergy = await energyValue
    }

    private func requestAuthorization() async {
        let types: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!
        ]

        try? await healthStore.requestAuthorization(toShare: [], read: types)
    }

    private func fetchLatestHeartRate() async -> Double? {
        let type = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let predicate = HKQuery.predicateForSamples(
            withStart: Date().addingTimeInterval(-3600),  // Last hour
            end: Date()
        )

        let query = HKSampleQuery(
            sampleType: type,
            predicate: predicate,
            limit: 1,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)]
        ) { query, samples, error in
            // Handle in continuation
        }

        return await withCheckedContinuation { continuation in
            healthStore.execute(query)
            // TODO: Properly extract value from samples
            continuation.resume(returning: nil)
        }
    }

    private func startHeartRateQuery() {
        // Create anchor query for continuous updates
        let type = HKQuantityType.quantityType(forIdentifier: .heartRate)!

        let query = HKAnchoredObjectQuery(
            type: type,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample],
                  let latest = samples.last else { return }

            Task { @MainActor in
                self?.heartRate = latest.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }
        }

        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample],
                  let latest = samples.last else { return }

            Task { @MainActor in
                self?.heartRate = latest.quantity.doubleValue(for: HKUnit(from: "count/min"))
            }
        }

        healthStore.execute(query)
    }

    // Similar implementations for HRV, steps, energy...
}
```

---

### 2.4 Watch Complications

**Supported Families**:
1. **Circular Small** - Medication countdown or pain level
2. **Modular Small** - Medication countdown
3. **Graphic Circular** - Activity rings + pain indicator
4. **Graphic Rectangular** - Multi-metric dashboard

**Complication Data**:
```swift
// InflamAI-Watch/Complications/ComplicationController.swift

import ClockKit
import SwiftUI

class ComplicationController: NSObject, CLKComplicationDataSource {

    // MARK: - Timeline Configuration

    func getSupportedTimeTravelDirections(
        for complication: CLKComplication,
        withHandler handler: @escaping (CLKComplicationTimeTravelDirections) -> Void
    ) {
        handler([.forward])  // Show upcoming medication times
    }

    func getTimelineStartDate(
        for complication: CLKComplication,
        withHandler handler: @escaping (Date?) -> Void
    ) {
        handler(Date())
    }

    func getTimelineEndDate(
        for complication: CLKComplication,
        withHandler handler: @escaping (Date?) -> Void
    ) {
        handler(Date().addingTimeInterval(24 * 3600))  // 24 hours ahead
    }

    func getPrivacyBehavior(
        for complication: CLKComplication,
        withHandler handler: @escaping (CLKComplicationPrivacyBehavior) -> Void
    ) {
        handler(.showOnLockScreen)  // Show medication reminders even when locked
    }

    // MARK: - Timeline Population

    func getCurrentTimelineEntry(
        for complication: CLKComplication,
        withHandler handler: @escaping (CLKComplicationTimelineEntry?) -> Void
    ) {
        switch complication.family {
        case .circularSmall:
            handler(createCircularSmallEntry(for: Date()))
        case .modularSmall:
            handler(createModularSmallEntry(for: Date()))
        case .graphicCircular:
            handler(createGraphicCircularEntry(for: Date()))
        case .graphicRectangular:
            handler(createGraphicRectangularEntry(for: Date()))
        default:
            handler(nil)
        }
    }

    func getTimelineEntries(
        for complication: CLKComplication,
        after date: Date,
        limit: Int,
        withHandler handler: @escaping ([CLKComplicationTimelineEntry]?) -> Void
    ) {
        // Generate entries for next 24 hours
        let medications = getUpcomingMedications(after: date, limit: limit)

        let entries = medications.compactMap { med -> CLKComplicationTimelineEntry? in
            switch complication.family {
            case .circularSmall:
                return createCircularSmallEntry(for: med.time, medication: med)
            case .modularSmall:
                return createModularSmallEntry(for: med.time, medication: med)
            case .graphicCircular:
                return createGraphicCircularEntry(for: med.time, medication: med)
            case .graphicRectangular:
                return createGraphicRectangularEntry(for: med.time, medication: med)
            default:
                return nil
            }
        }

        handler(entries)
    }

    // MARK: - Entry Creators

    private func createCircularSmallEntry(
        for date: Date,
        medication: UpcomingMedication? = nil
    ) -> CLKComplicationTimelineEntry {
        let template = CLKComplicationTemplateCircularSmallStackImage()

        if let med = medication {
            // Show hours until medication
            let hours = Int(med.time.timeIntervalSince(date) / 3600)
            template.line1ImageProvider = CLKImageProvider(onePieceImage: UIImage(systemName: "pills.fill")!)
            template.line2TextProvider = CLKTextProvider(format: "%dh", hours)
        } else {
            // Show current pain level
            let painLevel = getCurrentPainLevel()
            template.line1ImageProvider = CLKImageProvider(onePieceImage: UIImage(systemName: "cross.circle.fill")!)
            template.line2TextProvider = CLKTextProvider(format: "%d/5", painLevel)
        }

        return CLKComplicationTimelineEntry(date: date, complicationTemplate: template)
    }

    private func createModularSmallEntry(
        for date: Date,
        medication: UpcomingMedication? = nil
    ) -> CLKComplicationTimelineEntry {
        let template = CLKComplicationTemplateModularSmallStackImage()

        if let med = medication {
            let hours = Int(med.time.timeIntervalSince(date) / 3600)
            let minutes = Int((med.time.timeIntervalSince(date).truncatingRemainder(dividingBy: 3600)) / 60)

            template.line1ImageProvider = CLKImageProvider(onePieceImage: UIImage(systemName: "pills.fill")!)
            template.line2TextProvider = CLKTextProvider(format: "%dh %dm", hours, minutes)
        } else {
            let painLevel = getCurrentPainLevel()
            template.line1ImageProvider = CLKImageProvider(onePieceImage: UIImage(systemName: "cross.circle.fill")!)
            template.line2TextProvider = CLKTextProvider(format: "Pain: %d", painLevel)
        }

        return CLKComplicationTimelineEntry(date: date, complicationTemplate: template)
    }

    private func createGraphicCircularEntry(
        for date: Date,
        medication: UpcomingMedication? = nil
    ) -> CLKComplicationTimelineEntry {
        let template = CLKComplicationTemplateGraphicCircularStackImage()

        if let med = medication {
            // Medication countdown with progress ring
            let progress = 1.0 - (med.time.timeIntervalSince(date) / (24 * 3600))
            let hours = Int(med.time.timeIntervalSince(date) / 3600)

            let gaugeProvider = CLKSimpleGaugeProvider(
                style: .ring,
                gaugeColor: .blue,
                fillFraction: Float(progress)
            )

            template.line1ImageProvider = CLKFullColorImageProvider(
                fullColorImage: UIImage(systemName: "pills.circle.fill")!
            )
            template.line2TextProvider = CLKTextProvider(format: "%dh", hours)

            // Wrap in gauge template
            let gaugeTemplate = CLKComplicationTemplateGraphicCircularClosedGaugeImage()
            gaugeTemplate.imageProvider = CLKFullColorImageProvider(
                fullColorImage: UIImage(systemName: "pills.fill")!
            )
            gaugeTemplate.gaugeProvider = gaugeProvider

            return CLKComplicationTimelineEntry(date: date, complicationTemplate: gaugeTemplate)
        } else {
            // Activity rings + pain indicator
            let painLevel = getCurrentPainLevel()
            let activityRings = getActivityRingsData()

            // Use activity summary gauge
            template.line1ImageProvider = CLKFullColorImageProvider(
                fullColorImage: createPainIndicatorImage(level: painLevel)
            )
            template.line2TextProvider = CLKTextProvider(format: "%d/5", painLevel)

            return CLKComplicationTimelineEntry(date: date, complicationTemplate: template)
        }
    }

    private func createGraphicRectangularEntry(
        for date: Date,
        medication: UpcomingMedication? = nil
    ) -> CLKComplicationTimelineEntry {
        let template = CLKComplicationTemplateGraphicRectangularStandardBody()

        if let med = medication {
            // Medication reminder
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            let timeString = formatter.string(from: med.time)

            template.headerImageProvider = CLKFullColorImageProvider(
                fullColorImage: UIImage(systemName: "pills.circle.fill")!
            )
            template.headerTextProvider = CLKTextProvider(format: "Medication Due")
            template.body1TextProvider = CLKTextProvider(format: med.name)
            template.body2TextProvider = CLKTextProvider(format: "at %@", timeString)
        } else {
            // Multi-metric dashboard
            let painLevel = getCurrentPainLevel()
            let steps = getTodaySteps()
            let hr = getCurrentHeartRate()

            template.headerImageProvider = CLKFullColorImageProvider(
                fullColorImage: UIImage(systemName: "heart.circle.fill")!
            )
            template.headerTextProvider = CLKTextProvider(format: "InflamAI")
            template.body1TextProvider = CLKTextProvider(format: "Pain: %d/5  Steps: %d", painLevel, steps)
            template.body2TextProvider = CLKTextProvider(format: "HR: %d bpm", hr)
        }

        return CLKComplicationTimelineEntry(date: date, complicationTemplate: template)
    }

    // MARK: - Helper Methods

    private func getUpcomingMedications(after date: Date, limit: Int) -> [UpcomingMedication] {
        // TODO: Fetch from Core Data
        return []
    }

    private func getCurrentPainLevel() -> Int {
        // TODO: Fetch latest log
        return 3
    }

    private func getTodaySteps() -> Int {
        // TODO: Query HealthKit
        return 4230
    }

    private func getCurrentHeartRate() -> Int {
        // TODO: Query HealthKit
        return 72
    }

    private func getActivityRingsData() -> (move: Double, exercise: Double, stand: Double) {
        // TODO: Query HealthKit activity summary
        return (move: 0.75, exercise: 0.5, stand: 0.8)
    }

    private func createPainIndicatorImage(level: Int) -> UIImage {
        // Generate colored circle based on pain level
        let size = CGSize(width: 40, height: 40)
        let color: UIColor = {
            switch level {
            case 0...1: return .green
            case 2...3: return .yellow
            case 4...5: return .red
            default: return .gray
            }
        }()

        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            color.setFill()
            context.cgContext.fillEllipse(in: CGRect(origin: .zero, size: size))
        }
    }

    // MARK: - Placeholder Data

    func getLocalizableSampleTemplate(
        for complication: CLKComplication,
        withHandler handler: @escaping (CLKComplicationTemplate?) -> Void
    ) {
        // Provide sample for watch face gallery
        switch complication.family {
        case .circularSmall:
            handler(createSampleCircularSmall())
        case .modularSmall:
            handler(createSampleModularSmall())
        case .graphicCircular:
            handler(createSampleGraphicCircular())
        case .graphicRectangular:
            handler(createSampleGraphicRectangular())
        default:
            handler(nil)
        }
    }

    private func createSampleCircularSmall() -> CLKComplicationTemplate {
        let template = CLKComplicationTemplateCircularSmallStackImage()
        template.line1ImageProvider = CLKImageProvider(onePieceImage: UIImage(systemName: "pills.fill")!)
        template.line2TextProvider = CLKTextProvider(format: "2h")
        return template
    }

    private func createSampleModularSmall() -> CLKComplicationTemplate {
        let template = CLKComplicationTemplateModularSmallStackImage()
        template.line1ImageProvider = CLKImageProvider(onePieceImage: UIImage(systemName: "heart.fill")!)
        template.line2TextProvider = CLKTextProvider(format: "72 bpm")
        return template
    }

    private func createSampleGraphicCircular() -> CLKComplicationTemplate {
        let template = CLKComplicationTemplateGraphicCircularClosedGaugeImage()
        template.imageProvider = CLKFullColorImageProvider(
            fullColorImage: UIImage(systemName: "pills.circle.fill")!
        )
        template.gaugeProvider = CLKSimpleGaugeProvider(
            style: .ring,
            gaugeColor: .blue,
            fillFraction: 0.75
        )
        return template
    }

    private func createSampleGraphicRectangular() -> CLKComplicationTemplate {
        let template = CLKComplicationTemplateGraphicRectangularStandardBody()
        template.headerImageProvider = CLKFullColorImageProvider(
            fullColorImage: UIImage(systemName: "heart.circle.fill")!
        )
        template.headerTextProvider = CLKTextProvider(format: "InflamAI")
        template.body1TextProvider = CLKTextProvider(format: "Pain: 3/5  Steps: 4230")
        template.body2TextProvider = CLKTextProvider(format: "HR: 72 bpm")
        return template
    }
}

struct UpcomingMedication {
    let name: String
    let time: Date
}
```

**Complication Update Strategy**:
```swift
// Update complications when data changes

extension WatchSymptomViewModel {
    func updateComplications() {
        let server = CLKComplicationServer.sharedInstance()

        for complication in server.activeComplications ?? [] {
            server.reloadTimeline(for: complication)
        }
    }
}
```

---

## 3. WatchConnectivity Service

### 3.1 Architecture

**Communication Channels**:

| Method | Use Case | Delivery | Requires Reachability |
|--------|----------|----------|----------------------|
| **Application Context** | Configuration sync (medications list, settings) | Background, guaranteed delivery | No |
| **User Info Transfer** | Large data (symptom history, trends) | Background, queued | No |
| **Interactive Messages** | Real-time logging (symptom logs, med confirmations) | Immediate | Yes (phone must be reachable) |
| **File Transfer** | Bulk export (PDF reports, CSV) | Background | No |

### 3.2 Implementation

**Shared Protocol**:
```swift
// InflamAICore/Services/ConnectivityProtocol.swift

import Foundation

enum WatchMessageType: String, Codable {
    case symptomLog = "symptom_log"
    case medicationTaken = "medication_taken"
    case medicationSkipped = "medication_skipped"
    case configUpdate = "config_update"
    case healthMetricRequest = "health_metric_request"
    case healthMetricResponse = "health_metric_response"
}

struct WatchMessage: Codable {
    let type: WatchMessageType
    let timestamp: Date
    let payload: [String: AnyCodable]  // Use AnyCodable for flexible payloads
}

// Helper for type-safe payload encoding
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        // Encode based on type
        // Implementation omitted for brevity
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Decode to appropriate type
        // Implementation omitted for brevity
        self.value = ""
    }
}
```

**iPhone Implementation**:
```swift
// InflamAI/Core/Services/WatchConnectivityService.swift

import WatchConnectivity
import Combine

@MainActor
class WatchConnectivityService: NSObject, ObservableObject {
    static let shared = WatchConnectivityService()

    @Published var isReachable = false
    @Published var isPaired = false
    @Published var isWatchAppInstalled = false

    private var session: WCSession?
    private let persistence: PersistenceController

    private override init() {
        self.persistence = PersistenceController.shared
        super.init()

        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
            session?.activate()
        }
    }

    // MARK: - Send Methods

    func updateApplicationContext(_ context: [String: Any]) async throws {
        guard let session = session else { return }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            do {
                try session.updateApplicationContext(context)
                continuation.resume()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    func sendMessage(_ message: [String: Any]) async throws -> [String: Any] {
        guard let session = session, session.isReachable else {
            throw WatchConnectivityError.notReachable
        }

        return try await withCheckedThrowingContinuation { continuation in
            session.sendMessage(message, replyHandler: { reply in
                continuation.resume(returning: reply)
            }, errorHandler: { error in
                continuation.resume(throwing: error)
            })
        }
    }

    func transferUserInfo(_ userInfo: [String: Any]) {
        session?.transferUserInfo(userInfo)
    }

    func transferFile(_ fileURL: URL, metadata: [String: Any]?) {
        session?.transferFile(fileURL, metadata: metadata)
    }

    // MARK: - Sync Methods

    func syncMedicationsToWatch() async {
        let medications = fetchAllMedications()

        let context: [String: Any] = [
            "medications": medications.map { med in
                [
                    "id": med.id?.uuidString ?? "",
                    "name": med.name ?? "",
                    "dosage": med.dosage ?? "",
                    "frequency": med.frequency,
                    "scheduledTimes": med.scheduledTimesArray.map { $0.timeIntervalSince1970 }
                ]
            }
        ]

        try? await updateApplicationContext(context)
    }

    func syncRecentLogsToWatch() async {
        let logs = fetchRecentLogs(days: 7)

        let userInfo: [String: Any] = [
            "logs": logs.map { log in
                [
                    "id": log.id?.uuidString ?? "",
                    "timestamp": log.timestamp?.timeIntervalSince1970 ?? 0,
                    "painScore": log.painScore,
                    "stiffnessScore": log.stiffnessScore,
                    "fatigueScore": log.fatigueScore
                ]
            }
        ]

        transferUserInfo(userInfo)
    }

    // MARK: - Core Data Helpers

    private func fetchAllMedications() -> [Medication] {
        let request = Medication.fetchRequest()
        return (try? persistence.container.viewContext.fetch(request)) ?? []
    }

    private func fetchRecentLogs(days: Int) -> [SymptomLog] {
        let request = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp >= %@", Date().addingTimeInterval(-TimeInterval(days * 86400)) as CVarArg)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
        return (try? persistence.container.viewContext.fetch(request)) ?? []
    }
}

// MARK: - WCSessionDelegate

extension WatchConnectivityService: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        Task { @MainActor in
            isPaired = session.isPaired
            isWatchAppInstalled = session.isWatchAppInstalled
            isReachable = session.isReachable

            if activationState == .activated && isWatchAppInstalled {
                // Initial sync
                await syncMedicationsToWatch()
                await syncRecentLogsToWatch()
            }
        }
    }

    func sessionDidBecomeInactive(_ session: WCSession) {
        Task { @MainActor in
            isReachable = false
        }
    }

    func sessionDidDeactivate(_ session: WCSession) {
        Task { @MainActor in
            isReachable = false
        }
        session.activate()
    }

    func sessionReachabilityDidChange(_ session: WCSession) {
        Task { @MainActor in
            isReachable = session.isReachable
        }
    }

    // MARK: - Receive Messages from Watch

    func session(_ session: WCSession, didReceiveMessage message: [String: Any], replyHandler: @escaping ([String: Any]) -> Void) {
        Task { @MainActor in
            do {
                let response = try await handleWatchMessage(message)
                replyHandler(response)
            } catch {
                replyHandler(["error": error.localizedDescription])
            }
        }
    }

    private func handleWatchMessage(_ message: [String: Any]) async throws -> [String: Any] {
        guard let typeString = message["type"] as? String else {
            throw WatchConnectivityError.invalidMessage
        }

        switch typeString {
        case "symptom_log":
            return try await handleSymptomLog(message)

        case "medication_taken":
            return try await handleMedicationTaken(message)

        case "medication_skipped":
            return try await handleMedicationSkipped(message)

        case "health_metric_request":
            return try await handleHealthMetricRequest(message)

        default:
            throw WatchConnectivityError.unknownMessageType
        }
    }

    private func handleSymptomLog(_ message: [String: Any]) async throws -> [String: Any] {
        guard let idString = message["id"] as? String,
              let id = UUID(uuidString: idString),
              let timestamp = message["timestamp"] as? TimeInterval,
              let pain = message["pain"] as? Int,
              let stiffness = message["stiffness"] as? Int,
              let fatigue = message["fatigue"] as? Int else {
            throw WatchConnectivityError.invalidMessage
        }

        // Create SymptomLog entity
        let context = persistence.container.viewContext
        let log = SymptomLog(context: context)
        log.id = id
        log.timestamp = Date(timeIntervalSince1970: timestamp)
        log.painScore = Int16(pain)
        log.stiffnessScore = Int16(stiffness)
        log.fatigueScore = Int16(fatigue)
        log.source = "watch"

        try context.save()

        // Trigger correlation re-calculation
        NotificationCenter.default.post(name: .symptomLogAdded, object: log)

        return ["success": true, "id": idString]
    }

    private func handleMedicationTaken(_ message: [String: Any]) async throws -> [String: Any] {
        guard let idString = message["id"] as? String,
              let id = UUID(uuidString: idString),
              let timestamp = message["timestamp"] as? TimeInterval else {
            throw WatchConnectivityError.invalidMessage
        }

        let context = persistence.container.viewContext

        // Find medication
        let request = Medication.fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        guard let medication = try context.fetch(request).first else {
            throw WatchConnectivityError.medicationNotFound
        }

        // Create medication log
        let log = MedicationLog(context: context)
        log.id = UUID()
        log.timestamp = Date(timeIntervalSince1970: timestamp)
        log.medication = medication
        log.taken = true
        log.source = "watch"

        try context.save()

        return ["success": true]
    }

    private func handleMedicationSkipped(_ message: [String: Any]) async throws -> [String: Any] {
        // Similar to handleMedicationTaken but with taken = false
        return ["success": true]
    }

    private func handleHealthMetricRequest(_ message: [String: Any]) async throws -> [String: Any] {
        guard let metric = message["metric"] as? String else {
            throw WatchConnectivityError.invalidMessage
        }

        // Fetch latest metric from HealthKit
        let healthKitService = HealthKitService.shared

        switch metric {
        case "steps":
            let steps = await healthKitService.fetchTodaySteps()
            return ["metric": "steps", "value": steps]

        case "heartRate":
            let hr = await healthKitService.fetchLatestHeartRate()
            return ["metric": "heartRate", "value": hr ?? 0]

        default:
            throw WatchConnectivityError.unknownMetric
        }
    }

    // MARK: - Receive Context Updates

    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
        Task { @MainActor in
            // Handle context updates from Watch (if any)
        }
    }
}

enum WatchConnectivityError: Error {
    case notReachable
    case invalidMessage
    case unknownMessageType
    case medicationNotFound
    case unknownMetric
}

extension Notification.Name {
    static let symptomLogAdded = Notification.Name("symptomLogAdded")
}
```

**Watch Implementation**:
```swift
// InflamAI-Watch/Services/WatchConnectivityManager.swift

import WatchConnectivity

@MainActor
class WatchConnectivityManager: NSObject, ObservableObject {
    static let shared = WatchConnectivityManager()

    @Published var isReachable = false

    private var session: WCSession?
    private let persistence = WatchPersistenceController.shared

    private override init() {
        super.init()

        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
            session?.activate()
        }
    }

    func sendMessage(_ message: [String: Any]) async {
        guard let session = session else { return }

        if session.isReachable {
            // Send immediately
            _ = try? await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[String: Any], Error>) in
                session.sendMessage(message, replyHandler: { reply in
                    continuation.resume(returning: reply)
                }, errorHandler: { error in
                    continuation.resume(throwing: error)
                })
            }
        } else {
            // Queue for later
            session.transferUserInfo(message)
        }
    }
}

extension WatchConnectivityManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        Task { @MainActor in
            isReachable = session.isReachable
        }
    }

    func sessionReachabilityDidChange(_ session: WCSession) {
        Task { @MainActor in
            isReachable = session.isReachable
        }
    }

    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
        Task { @MainActor in
            // Update local cache with medications
            if let medications = applicationContext["medications"] as? [[String: Any]] {
                await updateLocalMedications(medications)
            }
        }
    }

    private func updateLocalMedications(_ medicationsData: [[String: Any]]) async {
        // Update Core Data on Watch
        let context = persistence.container.viewContext

        for medData in medicationsData {
            guard let idString = medData["id"] as? String,
                  let id = UUID(uuidString: idString),
                  let name = medData["name"] as? String else { continue }

            // Find or create
            let request = WatchMedication.fetchRequest()
            request.predicate = NSPredicate(format: "id == %@", id as CVarArg)

            let medication: WatchMedication
            if let existing = try? context.fetch(request).first {
                medication = existing
            } else {
                medication = WatchMedication(context: context)
                medication.id = id
            }

            medication.name = name
            medication.dosage = medData["dosage"] as? String
            // ... update other fields
        }

        try? context.save()
    }
}
```

---

## 4. Background Health Monitoring

### 4.1 iPhone Background Monitoring

**File**: [Core/Services/HealthKitService.swift](../InflamAI/Core/Services/HealthKitService.swift)

**Enhancements**:
```swift
extension HealthKitService {

    func enableBackgroundMonitoring() async {
        await enableBackgroundDelivery(for: .heartRate)
        await enableBackgroundDelivery(for: .heartRateVariabilitySDNN)
        await enableBackgroundDelivery(for: .sleepAnalysis)
    }

    private func enableBackgroundDelivery(for type: HKQuantityTypeIdentifier) async {
        guard let quantityType = HKQuantityType.quantityType(forIdentifier: type) else { return }

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            healthStore.enableBackgroundDelivery(for: quantityType, frequency: .hourly) { success, error in
                if success {
                    print("✅ Enabled background delivery for \(type.rawValue)")
                } else {
                    print("❌ Failed to enable background delivery: \(error?.localizedDescription ?? "unknown")")
                }
                continuation.resume()
            }
        }
    }

    func setupObserverQueries() {
        setupHeartRateObserver()
        setupHRVObserver()
        setupSleepObserver()
    }

    private func setupHeartRateObserver() {
        let type = HKQuantityType.quantityType(forIdentifier: .heartRate)!

        let query = HKObserverQuery(sampleType: type, predicate: nil) { [weak self] query, completionHandler, error in
            guard error == nil else {
                completionHandler()
                return
            }

            Task { @MainActor in
                // New heart rate data available
                await self?.processNewHeartRateData()
                completionHandler()
            }
        }

        healthStore.execute(query)
    }

    private func processNewHeartRateData() async {
        // Fetch latest HR samples
        let samples = await fetchRecentHeartRate(hours: 1)

        // Check for anomalies (e.g., sustained high HR)
        let avgHR = samples.map { $0.quantity.doubleValue(for: HKUnit(from: "count/min")) }.average()

        if avgHR > 100 {  // Elevated HR
            // Notify user or update correlation engine
            NotificationCenter.default.post(name: .elevatedHeartRateDetected, object: avgHR)
        }
    }
}
```

### 4.2 Watch Background Monitoring

**Background Modes** (Watch target capabilities):
- ☑️ **Background Modes**
  - ☑️ Audio, AirPlay, and Picture in Picture
  - ☑️ Background fetch
  - ☑️ Remote notifications
- ☑️ **HealthKit** (background delivery)
- ☑️ **WorkoutKit** (workout session management)

**Implementation**:
```swift
// InflamAI-Watch/Services/WatchHealthKitService.swift

import HealthKit
import WatchKit

class WatchHealthKitService {
    static let shared = WatchHealthKitService()

    private let healthStore = HKHealthStore()

    func enableBackgroundWorkout() {
        // Register for background workout sessions
        let configuration = HKWorkoutConfiguration()
        configuration.activityType = .other
        configuration.locationType = .unknown

        // This allows continuous HR monitoring even when app is backgrounded
        // Note: Requires user to "start workout" to enable continuous monitoring
    }

    func scheduleBackgroundRefresh() {
        // Schedule WKRefreshBackgroundTask
        let targetDate = Date().addingTimeInterval(3600)  // 1 hour from now

        WKExtension.shared().scheduleBackgroundRefresh(
            withPreferredDate: targetDate,
            userInfo: nil
        ) { error in
            if let error = error {
                print("❌ Failed to schedule background refresh: \(error)")
            }
        }
    }

    func handleBackgroundRefresh(task: WKRefreshBackgroundTask) {
        Task {
            // Fetch latest health data
            await fetchLatestMetrics()

            // Sync to iPhone
            await WatchConnectivityManager.shared.syncHealthData()

            // Update complications
            CLKComplicationServer.sharedInstance().reloadTimeline(for: CLKComplicationServer.sharedInstance().activeComplications?.first)

            // Schedule next refresh
            scheduleBackgroundRefresh()

            task.setTaskCompletedWithSnapshot(false)
        }
    }

    private func fetchLatestMetrics() async {
        // Query HRV, HR, steps
        // Store in local Core Data
    }
}

// In Watch App's ExtensionDelegate:
class ExtensionDelegate: NSObject, WKExtensionDelegate {
    func handle(_ backgroundTasks: Set<WKRefreshBackgroundTask>) {
        for task in backgroundTasks {
            switch task {
            case let backgroundTask as WKApplicationRefreshBackgroundTask:
                WatchHealthKitService.shared.handleBackgroundRefresh(task: backgroundTask)

            case let snapshotTask as WKSnapshotRefreshBackgroundTask:
                snapshotTask.setTaskCompleted(
                    restoredDefaultState: true,
                    estimatedSnapshotExpiration: Date().addingTimeInterval(3600),
                    userInfo: nil
                )

            default:
                task.setTaskCompletedWithSnapshot(false)
            }
        }
    }
}
```

---

## 5. Data Model Sync

### 5.1 Core Data Configuration

**Shared Data Model**:
- Use **same .xcdatamodeld** file across iOS and watchOS targets
- Enable **CloudKit sync** for seamless multi-device sync
- **Watch-specific entities** for local caching

**CloudKit Configuration**:
```swift
// InflamAICore/Services/PersistenceController.swift

import CoreData

class PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentCloudKitContainer

    init(inMemory: Bool = false) {
        container = NSPersistentCloudKitContainer(name: "InflamAI")

        if inMemory {
            container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        } else {
            // Enable CloudKit sync
            container.persistentStoreDescriptions.first?.cloudKitContainerOptions = NSPersistentCloudKitContainerOptions(
                containerIdentifier: "iCloud.com.inflamai.InflamAI"
            )

            // Enable remote change notifications
            container.persistentStoreDescriptions.first?.setOption(
                true as NSNumber,
                forKey: NSPersistentStoreRemoteChangeNotificationPostOptionKey
            )
        }

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Core Data failed to load: \(error.localizedDescription)")
            }
        }

        // Auto-merge changes from CloudKit
        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy

        // Observe remote changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleRemoteChange),
            name: .NSPersistentStoreRemoteChange,
            object: container.persistentStoreCoordinator
        )
    }

    @objc private func handleRemoteChange(_ notification: Notification) {
        Task { @MainActor in
            // Refresh UI when CloudKit sync completes
            container.viewContext.refreshAllObjects()
        }
    }
}
```

**Watch-Specific Store** (optional, for offline caching):
```swift
// InflamAI-Watch/Services/WatchPersistenceController.swift

class WatchPersistenceController {
    static let shared = WatchPersistenceController()

    let container: NSPersistentContainer

    init() {
        // Use same model but separate store file
        container = NSPersistentContainer(name: "InflamAI")

        let storeURL = FileManager.default
            .containerURL(forSecurityApplicationGroupIdentifier: "group.com.inflamai")!
            .appendingPathComponent("WatchCache.sqlite")

        let description = NSPersistentStoreDescription(url: storeURL)
        description.shouldMigrateStoreAutomatically = true
        description.shouldInferMappingModelAutomatically = true

        container.persistentStoreDescriptions = [description]

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Watch Core Data failed: \(error)")
            }
        }

        container.viewContext.automaticallyMergesChangesFromParent = true
    }

    // Clean up old data (keep only last 7 days)
    func purgeOldData() {
        let context = container.viewContext
        let cutoffDate = Date().addingTimeInterval(-7 * 86400)

        let request = WatchQuickLog.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp < %@ AND synced == YES", cutoffDate as CVarArg)

        if let oldLogs = try? context.fetch(request) {
            for log in oldLogs {
                context.delete(log)
            }
            try? context.save()
        }
    }
}
```

### 5.2 Conflict Resolution

**Strategy**: **Last-write-wins** with source tracking

```swift
// When syncing from Watch → iPhone
func mergeWatchLog(_ watchLog: [String: Any]) async throws {
    let context = persistence.container.viewContext

    guard let idString = watchLog["id"] as? String,
          let id = UUID(uuidString: idString) else {
        throw SyncError.invalidID
    }

    // Check if log already exists
    let request = SymptomLog.fetchRequest()
    request.predicate = NSPredicate(format: "id == %@", id as CVarArg)

    if let existingLog = try context.fetch(request).first {
        // Conflict: Compare timestamps
        let watchTimestamp = watchLog["timestamp"] as? TimeInterval ?? 0
        let existingTimestamp = existingLog.timestamp?.timeIntervalSince1970 ?? 0

        if watchTimestamp > existingTimestamp {
            // Watch version is newer → update
            existingLog.painScore = Int16(watchLog["pain"] as? Int ?? 0)
            existingLog.stiffnessScore = Int16(watchLog["stiffness"] as? Int ?? 0)
            existingLog.fatigueScore = Int16(watchLog["fatigue"] as? Int ?? 0)
            existingLog.source = "watch"
        }
        // Else: Keep existing version
    } else {
        // No conflict: Create new log
        let newLog = SymptomLog(context: context)
        newLog.id = id
        newLog.timestamp = Date(timeIntervalSince1970: watchLog["timestamp"] as? TimeInterval ?? 0)
        newLog.painScore = Int16(watchLog["pain"] as? Int ?? 0)
        newLog.stiffnessScore = Int16(watchLog["stiffness"] as? Int ?? 0)
        newLog.fatigueScore = Int16(watchLog["fatigue"] as? Int ?? 0)
        newLog.source = "watch"
    }

    try context.save()
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Test Cases**:
1. **WatchConnectivity**:
   - Message serialization/deserialization
   - Reachability handling (offline queuing)
   - Conflict resolution

2. **Health Data**:
   - HealthKit query accuracy
   - Background delivery
   - Data binning/aggregation

3. **Complications**:
   - Timeline generation
   - Data freshness
   - Placeholder templates

**Example**:
```swift
// InflamAITests/WatchConnectivityTests.swift

import XCTest
@testable import InflamAI_Swift

class WatchConnectivityTests: XCTestCase {

    func testSymptomLogMessageSerialization() {
        let message: [String: Any] = [
            "type": "symptom_log",
            "id": UUID().uuidString,
            "timestamp": Date().timeIntervalSince1970,
            "pain": 3,
            "stiffness": 4,
            "fatigue": 2
        ]

        // Test that message can be serialized and deserialized
        XCTAssertNotNil(message["type"])
        XCTAssertNotNil(message["id"])
        XCTAssertEqual(message["pain"] as? Int, 3)
    }

    func testConflictResolution() async throws {
        let service = WatchConnectivityService.shared

        // Create older log on iPhone
        let olderLog = createSymptomLog(timestamp: Date().addingTimeInterval(-3600), pain: 3)

        // Receive newer log from Watch
        let newerMessage: [String: Any] = [
            "type": "symptom_log",
            "id": olderLog.id!.uuidString,
            "timestamp": Date().timeIntervalSince1970,  // Newer
            "pain": 4,  // Different value
            "stiffness": 4,
            "fatigue": 2
        ]

        let response = try await service.handleWatchMessage(newerMessage)

        XCTAssertTrue(response["success"] as? Bool ?? false)

        // Verify newer value was kept
        let fetchedLog = fetchSymptomLog(id: olderLog.id!)
        XCTAssertEqual(fetchedLog?.painScore, 4)  // Updated to newer value
    }
}
```

### 6.2 UI Tests

**Test Flows**:
1. **Quick Log**:
   - Launch from complication
   - Select scores
   - Verify confirmation
   - Check sync to iPhone

2. **Medication Reminder**:
   - Receive notification
   - Mark as taken
   - Verify sync

3. **Complications**:
   - Install complication
   - Verify data updates
   - Check tap navigation

**Example**:
```swift
// InflamAIWatchUITests/QuickLogUITests.swift

import XCTest

class QuickLogUITests: XCTestCase {

    func testQuickLogFlow() throws {
        let app = XCUIApplication()
        app.launch()

        // Navigate to Quick Log
        app.buttons["Quick Log"].tap()

        // Select pain level
        app.otherElements.containing(.staticText, identifier: "Pain").element
            .buttons.element(boundBy: 2).tap()  // 3/5

        // Select stiffness
        app.otherElements.containing(.staticText, identifier: "Stiffness").element
            .buttons.element(boundBy: 3).tap()  // 4/5

        // Select fatigue
        app.otherElements.containing(.staticText, identifier: "Fatigue").element
            .buttons.element(boundBy: 1).tap()  // 2/5

        // Submit
        app.buttons["Log"].tap()

        // Verify confirmation
        XCTAssertTrue(app.staticTexts["Logged ✓"].waitForExistence(timeout: 2))

        // Wait for auto-dismiss
        XCTAssertTrue(app.staticTexts["Logged ✓"].waitForNonExistence(timeout: 3))
    }
}
```

### 6.3 Integration Tests

**Test Scenarios**:
1. **End-to-End Sync**:
   - Log on Watch
   - Verify appears on iPhone
   - Verify triggers correlation update

2. **Background Monitoring**:
   - Enable background delivery
   - Simulate new HRV data
   - Verify flare detection triggers

3. **CloudKit Sync**:
   - Create log on iPhone
   - Verify syncs to Watch
   - Test offline queue

---

## 7. Performance Requirements

### 7.1 Battery Life

**Targets**:
| Scenario | Battery Impact | Monitoring Frequency |
|----------|---------------|---------------------|
| **Passive (no logging)** | <5% per day | HR every 5 min, HRV every 10 min |
| **Active Logging (3x/day)** | <8% per day | Same + 3 quick log sessions |
| **Always-On Display** | <12% per day | Same + complication updates every 15 min |
| **Workout Mode** | ~1% per min | HR every second, HRV every 30 sec |

**Optimization**:
- Use **HKAnchoredObjectQuery** to minimize redundant fetches
- **Batch complication updates** (max 4x/hour)
- **Defer non-critical syncs** to charging sessions
- Use **low-power mode** detection to reduce monitoring frequency

```swift
extension WatchHealthKitService {
    func adjustForBatteryLevel() {
        WKInterfaceDevice.current().isBatteryMonitoringEnabled = true
        let batteryLevel = WKInterfaceDevice.current().batteryLevel

        if batteryLevel < 0.2 {  // <20%
            // Reduce monitoring frequency
            monitoringInterval = 15 * 60  // Every 15 min
        } else {
            monitoringInterval = 5 * 60  // Every 5 min
        }
    }
}
```

### 7.2 Responsiveness

**Targets**:
| Action | Target Latency |
|--------|---------------|
| **Complication Tap → App Launch** | <1 second |
| **Quick Log → Confirmation** | <500ms |
| **Medication Taken → Sync** | <2 seconds (if reachable) |
| **Background Refresh** | <10 seconds |

### 7.3 Data Transfer

**Bandwidth Optimization**:
- **Compress large payloads** (use Gzip for JSON)
- **Delta sync** (only changes since last sync)
- **Binary encoding** for health metrics (Protobuf or MessagePack)

```swift
extension WatchConnectivityManager {
    func sendCompressedData(_ data: Data) async {
        let compressed = try? (data as NSData).compressed(using: .lzfse) as Data
        transferUserInfo(["data": compressed ?? data])
    }
}
```

---

## 8. Privacy & Security

### 8.1 Data Protection

**Encryption**:
- **At Rest**: All Core Data stores use `NSFileProtectionComplete`
- **In Transit**: WatchConnectivity uses TLS 1.3
- **CloudKit**: End-to-end encryption for sensitive fields

**Implementation**:
```swift
// Set file protection for Core Data store
let storeDescription = NSPersistentStoreDescription()
storeDescription.setOption(
    FileProtectionType.complete as NSObject,
    forKey: NSPersistentStoreFileProtectionKey
)
```

### 8.2 HealthKit Permissions

**Granular Permissions**:
- Request **only necessary types** (HR, HRV, sleep, steps, energy)
- **No write permissions** to HealthKit (read-only)
- **Re-prompt** if authorization revoked

**User Control**:
```
⚙️ Privacy Settings

HealthKit Data Access:
  ✅ Heart Rate (for inflammation tracking)
  ✅ HRV (for stress/recovery)
  ✅ Sleep Analysis (for sleep-inflammation correlation)
  ✅ Steps & Energy (for activity thresholds)

  ❌ Blood Glucose
  ❌ Blood Pressure

[Manage in Settings app]
```

### 8.3 Complication Privacy

**Sensitive Data**:
- **Pain levels**: Hide on lock screen (use generic icon)
- **Medication names**: Show only "Medication Due" (not drug names)
- **Allow user to disable** sensitive complications

```swift
func getPrivacyBehavior(for complication: CLKComplication, withHandler handler: @escaping (CLKComplicationPrivacyBehavior) -> Void) {
    let userPreference = UserDefaults.standard.bool(forKey: "hideHealthDataOnLockScreen")

    if userPreference {
        handler(.hideOnLockScreen)  // Show generic placeholder when locked
    } else {
        handler(.showOnLockScreen)
    }
}
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- ✅ Create WatchOS app target
- ✅ Set up shared Core Data model
- ✅ Implement WatchConnectivityService (iPhone + Watch)
- ✅ Build Quick Log UI
- ✅ Test basic symptom logging flow

### Phase 2: Medication Tracking (Week 3)
- ✅ Build Medication Tracker UI
- ✅ Implement local notifications
- ✅ Sync medication schedule from iPhone
- ✅ Test medication confirmation flow

### Phase 3: Complications (Week 4)
- ✅ Implement ComplicationController
- ✅ Create templates for all families
- ✅ Build timeline data sources
- ✅ Test complication updates

### Phase 4: Background Monitoring (Week 5)
- ✅ Enable background HealthKit queries
- ✅ Set up WKRefreshBackgroundTask
- ✅ Implement battery-aware monitoring
- ✅ Test background data collection

### Phase 5: Biometrics Dashboard (Week 6)
- ✅ Build BiometricsView UI
- ✅ Implement real-time HR/HRV display
- ✅ Add activity rings visualization
- ✅ Test continuous monitoring

### Phase 6: Polish & Testing (Weeks 7-8)
- ✅ UI/UX refinements
- ✅ Performance optimization
- ✅ Battery life testing
- ✅ Integration tests
- ✅ Beta testing with 10 users

### Phase 7: Advanced Features (Weeks 9-10)
- ✅ Sleep stage visualization
- ✅ Flare prediction alerts
- ✅ Medication timing optimization
- ✅ Export Watch data to PDF

---

## 10. Open Questions & Decisions

### 10.1 Technical Decisions

1. **Standalone vs Companion App**:
   - **Decision**: Companion (requires iPhone)
   - **Rationale**: Core Data sync, weather data, complex correlations require iPhone processing power

2. **CloudKit vs WatchConnectivity**:
   - **Decision**: Both
   - **Rationale**: CloudKit for long-term sync, WatchConnectivity for real-time messaging

3. **Complication Update Frequency**:
   - **Decision**: Every 15 minutes
   - **Rationale**: Balance freshness vs battery life

### 10.2 UX Decisions

1. **Haptic Feedback**:
   - **Play success haptic** on log submission
   - **Play notification haptic** for medication reminders
   - **Play warning haptic** for flare predictions

2. **Voice Input**:
   - **Future enhancement**: Dictate symptom notes
   - **Not MVP**: Requires NLP processing

3. **Glance UI**:
   - **Deprecated in watchOS 10**: Focus on complications + app

---

## Appendix: Code Checklist

### A. Xcode Project Setup
- [ ] Create WatchOS App target
- [ ] Add HealthKit entitlement (Watch target)
- [ ] Add App Groups entitlement (both targets)
- [ ] Configure WatchConnectivity capability
- [ ] Add background modes (Watch target)
- [ ] Link shared framework (InflamAICore)

### B. Core Files to Create
- [ ] `WatchConnectivityService.swift` (iPhone)
- [ ] `WatchConnectivityManager.swift` (Watch)
- [ ] `QuickLogView.swift` (Watch)
- [ ] `MedicationTrackerView.swift` (Watch)
- [ ] `BiometricsView.swift` (Watch)
- [ ] `ComplicationController.swift` (Watch)
- [ ] `WatchHealthKitService.swift` (Watch)
- [ ] `WatchPersistenceController.swift` (Watch)

### C. Core Data Changes
- [ ] Add `WatchQuickLog` entity
- [ ] Add `WatchMedicationLog` entity
- [ ] Enable CloudKit sync
- [ ] Test migration

### D. Testing Files
- [ ] `WatchConnectivityTests.swift`
- [ ] `QuickLogUITests.swift`
- [ ] `MedicationTrackerUITests.swift`
- [ ] `ComplicationTests.swift`

### E. Documentation
- [ ] Update README.md with Watch features
- [ ] Create Watch app user guide
- [ ] Document API contracts (WatchConnectivity messages)
- [ ] Privacy policy updates (HealthKit permissions)

---

**End of Technical Specification**

*For implementation guidance, see [HEALTHKIT_ENHANCEMENT_ROADMAP.md](./HEALTHKIT_ENHANCEMENT_ROADMAP.md)*
*For pattern recognition opportunities, see [PATTERN_RECOGNITION_OPPORTUNITIES.md](./PATTERN_RECOGNITION_OPPORTUNITIES.md)*
