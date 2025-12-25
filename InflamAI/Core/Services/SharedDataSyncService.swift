//
//  SharedDataSyncService.swift
//  InflamAI
//
//  Syncs Core Data to shared App Group container for widgets and Watch app
//  Observes Core Data changes and pushes updates automatically
//

import Foundation
import CoreData
import WidgetKit
import Combine

/// Service that syncs data from Core Data to the shared App Group container
/// Used by widgets (iOS + Watch) and WatchConnectivity
@MainActor
final class SharedDataSyncService: ObservableObject {

    // MARK: - Singleton

    static let shared = SharedDataSyncService()

    // MARK: - Published State

    @Published private(set) var lastSyncDate: Date?
    @Published private(set) var isSyncing = false

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private let widgetWriter: WidgetDataWriter
    private let defaults: UserDefaults?
    private var cancellables = Set<AnyCancellable>()
    private var notificationObserver: NSObjectProtocol?

    // MARK: - Rate Limiting (fixes message rate-limit exceeded warnings)

    /// Minimum time between sync operations (prevents >32hz message spam)
    private let syncDebounceInterval: TimeInterval = 2.0  // 2 seconds

    /// Last sync timestamp for debouncing
    private var lastSyncTimestamp: Date = .distantPast

    /// Pending sync task (for debounce coalescing)
    private var pendingSyncTask: Task<Void, Never>?

    // MARK: - Initialization

    private init(
        persistenceController: InflamAIPersistenceController = .shared,
        widgetWriter: WidgetDataWriter = .shared
    ) {
        self.persistenceController = persistenceController
        self.widgetWriter = widgetWriter
        self.defaults = AppGroupConfig.sharedDefaults

        setupCoreDataObserver()
    }

    deinit {
        if let observer = notificationObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - Core Data Observer

    private func setupCoreDataObserver() {
        // Observe Core Data saves to trigger sync
        notificationObserver = NotificationCenter.default.addObserver(
            forName: .NSManagedObjectContextDidSave,
            object: persistenceController.container.viewContext,
            queue: .main
        ) { [weak self] notification in
            Task { @MainActor in
                await self?.handleCoreDataSave(notification)
            }
        }
    }

    private func handleCoreDataSave(_ notification: Notification) async {
        // FIXED: Debounce rapid saves to prevent message rate-limit exceeded (>32hz)
        // Instead of syncing immediately, schedule a debounced sync

        // Check what was saved and sync relevant data
        guard let userInfo = notification.userInfo else { return }

        let insertedObjects = userInfo[NSInsertedObjectsKey] as? Set<NSManagedObject> ?? []
        let updatedObjects = userInfo[NSUpdatedObjectsKey] as? Set<NSManagedObject> ?? []
        let changedObjects = insertedObjects.union(updatedObjects)

        // Skip if no relevant changes
        guard !changedObjects.isEmpty else { return }

        // Determine what needs syncing based on changed entities
        var needsBASDAISync = false
        var needsMedicationSync = false
        var needsFlareSync = false
        var needsStreakSync = false

        for object in changedObjects {
            switch object {
            case is SymptomLog:
                needsBASDAISync = true
                needsStreakSync = true
                needsFlareSync = true
            case is Medication, is DoseLog:
                needsMedicationSync = true
            case is FlareEvent:
                needsFlareSync = true
            default:
                break
            }
        }

        // Skip if nothing relevant changed
        guard needsBASDAISync || needsMedicationSync || needsFlareSync || needsStreakSync else { return }

        // DEBOUNCE: Check if we've synced recently
        let now = Date()
        let timeSinceLastSync = now.timeIntervalSince(lastSyncTimestamp)

        if timeSinceLastSync < syncDebounceInterval {
            // Cancel any pending sync and schedule a new one
            pendingSyncTask?.cancel()

            pendingSyncTask = Task { @MainActor in
                // Wait for debounce interval
                let waitTime = UInt64((syncDebounceInterval - timeSinceLastSync) * 1_000_000_000)
                try? await Task.sleep(nanoseconds: waitTime)

                guard !Task.isCancelled else { return }

                // Perform the sync
                await performDebouncedSync(
                    needsBASDAISync: needsBASDAISync,
                    needsMedicationSync: needsMedicationSync,
                    needsFlareSync: needsFlareSync,
                    needsStreakSync: needsStreakSync
                )
            }
            return
        }

        // No recent sync - perform immediately
        await performDebouncedSync(
            needsBASDAISync: needsBASDAISync,
            needsMedicationSync: needsMedicationSync,
            needsFlareSync: needsFlareSync,
            needsStreakSync: needsStreakSync
        )
    }

    /// Performs sync operations with timestamp tracking
    private func performDebouncedSync(
        needsBASDAISync: Bool,
        needsMedicationSync: Bool,
        needsFlareSync: Bool,
        needsStreakSync: Bool
    ) async {
        lastSyncTimestamp = Date()

        #if DEBUG
        print("📤 [SharedDataSync] Performing debounced sync")
        #endif

        // Perform targeted syncs
        if needsBASDAISync {
            await syncBASDAIData()
        }
        if needsMedicationSync {
            await syncMedicationData()
        }
        if needsFlareSync {
            await syncFlareData()
        }
        if needsStreakSync {
            await syncStreakData()
        }
    }

    // MARK: - Full Sync

    /// Perform a complete sync of all data to shared container
    func performFullSync() async {
        guard !isSyncing else { return }

        isSyncing = true
        defer {
            isSyncing = false
            lastSyncDate = Date()
        }

        await syncBASDAIData()
        await syncMedicationData()
        await syncFlareData()
        await syncStreakData()
        await syncTodaySummary()
        await syncHealthData()

        // Reload all widgets
        WidgetCenter.shared.reloadAllTimelines()

        print("SharedDataSyncService: Full sync completed")
    }

    // MARK: - BASDAI Sync

    func syncBASDAIData() async {
        let context = persistenceController.container.viewContext

        do {
            // Get recent BASDAI scores
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.predicate = NSPredicate(format: "basdaiScore > 0")
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
            request.fetchLimit = 30

            let logs = try context.fetch(request)

            guard let latestLog = logs.first else {
                widgetWriter.updateBASDAI(score: 0, category: "No Data", trend: "stable")
                return
            }

            let score = latestLog.basdaiScore
            let category = basdaiCategory(for: score)
            let trend = calculateBASDAITrend(from: logs)

            widgetWriter.updateBASDAI(score: score, category: category, trend: trend)

        } catch {
            print("SharedDataSyncService: Failed to sync BASDAI - \(error)")
        }
    }

    private func basdaiCategory(for score: Double) -> String {
        switch score {
        case 0..<2: return "Remission"
        case 2..<4: return "Low"
        case 4..<6: return "Moderate"
        case 6..<8: return "High"
        default: return "Very High"
        }
    }

    private func calculateBASDAITrend(from logs: [SymptomLog]) -> String {
        guard logs.count >= 3 else { return "stable" }

        // Compare average of last 3 vs previous 3
        let recent = Array(logs.prefix(3))
        let previous = Array(logs.dropFirst(3).prefix(3))

        guard !previous.isEmpty else { return "stable" }

        let recentAvg = recent.reduce(0.0) { $0 + $1.basdaiScore } / Double(recent.count)
        let previousAvg = previous.reduce(0.0) { $0 + $1.basdaiScore } / Double(previous.count)

        let difference = recentAvg - previousAvg

        if difference < -0.5 {
            return "improving"
        } else if difference > 0.5 {
            return "worsening"
        } else {
            return "stable"
        }
    }

    // MARK: - Medication Sync

    func syncMedicationData() async {
        let context = persistenceController.container.viewContext

        do {
            let request: NSFetchRequest<Medication> = Medication.fetchRequest()
            request.predicate = NSPredicate(format: "isActive == YES")
            request.sortDescriptors = [NSSortDescriptor(keyPath: \Medication.name, ascending: true)]

            let medications = try context.fetch(request)

            // Convert to widget-compatible format
            let reminders: [WidgetMedicationData.MedicationReminder] = medications.compactMap { med -> WidgetMedicationData.MedicationReminder? in
                guard let name = med.name else { return nil }
                let nextDose = calculateNextDoseTime(for: med) ?? Calendar.current.date(byAdding: .hour, value: 1, to: Date())!

                let dosageString = "\(med.dosage)\(med.dosageUnit ?? "mg")"
                return WidgetMedicationData.MedicationReminder(
                    id: med.id ?? UUID(),
                    name: name,
                    dosage: dosageString,
                    nextDoseTime: nextDose,
                    frequency: med.frequency ?? "Daily"
                )
            }

            widgetWriter.updateMedications(reminders)

        } catch {
            print("SharedDataSyncService: Failed to sync medications - \(error)")
        }
    }

    private func calculateNextDoseTime(for medication: Medication) -> Date? {
        // Simple next dose calculation - in a real app this would be more sophisticated
        guard let reminderTimes = medication.reminderTimes as? [Date] else {
            return Calendar.current.date(byAdding: .hour, value: 1, to: Date())
        }

        let now = Date()
        let calendar = Calendar.current

        // Find next reminder time today or tomorrow
        for time in reminderTimes {
            let components = calendar.dateComponents([.hour, .minute], from: time)
            if let todayTime = calendar.date(bySettingHour: components.hour ?? 0,
                                             minute: components.minute ?? 0,
                                             second: 0,
                                             of: now) {
                if todayTime > now {
                    return todayTime
                }

                // Try tomorrow
                if let tomorrowTime = calendar.date(byAdding: .day, value: 1, to: todayTime) {
                    return tomorrowTime
                }
            }
        }

        return Calendar.current.date(byAdding: .day, value: 1, to: now)
    }

    // MARK: - Flare Sync

    func syncFlareData() async {
        let context = persistenceController.container.viewContext

        do {
            // Check for active flares
            let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            request.predicate = NSPredicate(format: "isResolved == NO")
            request.sortDescriptors = [NSSortDescriptor(keyPath: \FlareEvent.startDate, ascending: false)]
            request.fetchLimit = 1

            let activeFlares = try context.fetch(request)
            let hasActiveFlare = !activeFlares.isEmpty

            // Calculate flare risk (simplified - in production use FlarePredictor)
            let riskPercentage = await calculateFlareRisk()
            let riskLevel = flareRiskLevel(for: riskPercentage)
            let factors = await identifyRiskFactors()

            widgetWriter.updateFlareRisk(percentage: riskPercentage, level: riskLevel, factors: factors)

            // Update today's summary with flare status
            defaults?.set(hasActiveFlare, forKey: WidgetDataKeys.hasActiveFlare)
            if let flare = activeFlares.first {
                defaults?.set(flare.startDate, forKey: WidgetDataKeys.activeFlareStartDate)
            }

        } catch {
            print("SharedDataSyncService: Failed to sync flare data - \(error)")
        }
    }

    private func calculateFlareRisk() async -> Int {
        // Use Neural Engine prediction if available
        if let prediction = UnifiedNeuralEngine.shared.currentPrediction {
            return prediction.riskPercentage
        }

        // Fallback: simplified risk calculation from Core Data
        let context = persistenceController.container.viewContext

        do {
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            let threeDaysAgo = Calendar.current.date(byAdding: .day, value: -3, to: Date()) ?? Date()
            request.predicate = NSPredicate(format: "timestamp >= %@", threeDaysAgo as NSDate)

            let recentLogs = try context.fetch(request)

            guard !recentLogs.isEmpty else { return 25 }

            let avgBASDAI = recentLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(recentLogs.count)
            let avgFatigue = recentLogs.reduce(0.0) { $0 + Double($1.fatigueLevel) } / Double(recentLogs.count)

            // Simple risk formula
            let risk = min(100, Int((avgBASDAI * 8) + (avgFatigue * 2)))
            return risk

        } catch {
            return 25
        }
    }

    private func flareRiskLevel(for percentage: Int) -> String {
        switch percentage {
        case 0..<25: return "low"
        case 25..<50: return "moderate"
        case 50..<75: return "high"
        default: return "veryHigh"
        }
    }

    private func identifyRiskFactors() async -> [String] {
        // Use Neural Engine's contributing factors if available
        if let prediction = UnifiedNeuralEngine.shared.currentPrediction {
            // Map ContributingFactor objects to their string names
            return prediction.topFactors.prefix(3).map { $0.name }
        }

        // Fallback: simplified factor analysis
        var factors: [String] = []

        // Check weather (would use WeatherKit in production)
        if let pressureChange = defaults?.double(forKey: "weather.pressureChange12h"),
           pressureChange > 5 {
            factors.append("Pressure drop")
        }

        // Check sleep (would use HealthKit in production)
        if let sleepHours = defaults?.double(forKey: WidgetDataKeys.healthSleepHours),
           sleepHours < 6 {
            factors.append("Low sleep")
        }

        return Array(factors.prefix(3))
    }

    // MARK: - Streak Sync

    func syncStreakData() async {
        let context = persistenceController.container.viewContext

        do {
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]

            let logs = try context.fetch(request)

            let streak = calculateLoggingStreak(from: logs)
            widgetWriter.updateStreak(days: streak)

        } catch {
            print("SharedDataSyncService: Failed to sync streak - \(error)")
        }
    }

    private func calculateLoggingStreak(from logs: [SymptomLog]) -> Int {
        guard !logs.isEmpty else { return 0 }

        let calendar = Calendar.current
        var streak = 0
        var currentDate = calendar.startOfDay(for: Date())

        // Check if logged today
        let todayLogs = logs.filter { log in
            guard let timestamp = log.timestamp else { return false }
            return calendar.isDate(timestamp, inSameDayAs: currentDate)
        }

        if todayLogs.isEmpty {
            // Check yesterday - if no log yesterday, streak is 0
            currentDate = calendar.date(byAdding: .day, value: -1, to: currentDate) ?? currentDate
        }

        // Count consecutive days
        for log in logs {
            guard let timestamp = log.timestamp else { continue }
            let logDay = calendar.startOfDay(for: timestamp)

            if calendar.isDate(logDay, inSameDayAs: currentDate) {
                streak += 1
                currentDate = calendar.date(byAdding: .day, value: -1, to: currentDate) ?? currentDate
            } else if logDay < currentDate {
                break
            }
        }

        return streak
    }

    // MARK: - Today Summary Sync

    func syncTodaySummary() async {
        let context = persistenceController.container.viewContext
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())

        do {
            // Count today's symptom logs
            let symptomRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            symptomRequest.predicate = NSPredicate(format: "timestamp >= %@", startOfDay as NSDate)
            let todayLogs = try context.fetch(symptomRequest)

            // Count pain entries (body region logs from today's symptom logs)
            var painEntries = 0
            for log in todayLogs {
                if let regions = log.bodyRegionLogs as? Set<BodyRegionLog> {
                    painEntries += regions.count
                }
            }

            // Count assessments (BASDAI scores today)
            let assessments = todayLogs.filter { $0.basdaiScore > 0 }.count

            let hasLoggedToday = !todayLogs.isEmpty

            // Check active flare
            let flareRequest: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            flareRequest.predicate = NSPredicate(format: "isResolved == NO")
            let hasActiveFlare = try context.fetch(flareRequest).count > 0

            widgetWriter.updateTodaySummary(
                painEntries: painEntries,
                assessments: assessments,
                hasLogged: hasLoggedToday,
                hasActiveFlare: hasActiveFlare
            )

        } catch {
            print("SharedDataSyncService: Failed to sync today summary - \(error)")
        }
    }

    // MARK: - Health Data Sync

    func syncHealthData() async {
        // This would be populated by HealthKitService
        // For now, just ensure the keys exist with default values
        if defaults?.object(forKey: WidgetDataKeys.healthSteps) == nil {
            defaults?.set(0, forKey: WidgetDataKeys.healthSteps)
        }
        if defaults?.object(forKey: WidgetDataKeys.healthHRV) == nil {
            defaults?.set(0.0, forKey: WidgetDataKeys.healthHRV)
        }
        if defaults?.object(forKey: WidgetDataKeys.healthUpdated) == nil {
            defaults?.set(Date(), forKey: WidgetDataKeys.healthUpdated)
        }
    }

    /// Update health data from HealthKit (called by HealthKitService)
    func updateHealthData(steps: Int, hrv: Double, restingHR: Int, sleepHours: Double) {
        defaults?.set(steps, forKey: WidgetDataKeys.healthSteps)
        defaults?.set(hrv, forKey: WidgetDataKeys.healthHRV)
        defaults?.set(restingHR, forKey: WidgetDataKeys.healthRestingHR)
        defaults?.set(sleepHours, forKey: WidgetDataKeys.healthSleepHours)
        defaults?.set(Date(), forKey: WidgetDataKeys.healthUpdated)

        WidgetCenter.shared.reloadAllTimelines()
    }

    // MARK: - Watch Connectivity Support

    /// Get all widget data as dictionary for Watch transfer
    func getDataForWatch() -> [String: Any] {
        guard let defaults = defaults else { return [:] }

        return [
            "basdai": [
                "score": defaults.double(forKey: WidgetDataKeys.basdaiScore),
                "category": defaults.string(forKey: WidgetDataKeys.basdaiCategory) ?? "Unknown",
                "trend": defaults.string(forKey: WidgetDataKeys.basdaiTrend) ?? "stable"
            ],
            "flareRisk": [
                "percentage": defaults.integer(forKey: WidgetDataKeys.flareRiskPercentage),
                "level": defaults.string(forKey: WidgetDataKeys.flareRiskLevel) ?? "low"
            ],
            "streak": defaults.integer(forKey: WidgetDataKeys.loggingStreak),
            "hasActiveFlare": defaults.bool(forKey: WidgetDataKeys.hasActiveFlare),
            "lastSync": Date().timeIntervalSince1970
        ]
    }
}
