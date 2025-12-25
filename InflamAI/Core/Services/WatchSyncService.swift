//
//  WatchSyncService.swift
//  InflamAI
//
//  iOS-side WatchConnectivity service that communicates with the Watch app
//  Handles bidirectional sync of medications, symptoms, and activity data
//

import Foundation
import WatchConnectivity
import CoreData
import Combine

/// Service for syncing data between iOS app and Apple Watch
@MainActor
final class WatchSyncService: NSObject, ObservableObject {

    // MARK: - Singleton

    static let shared = WatchSyncService()

    // MARK: - Published State

    @Published private(set) var isWatchReachable = false
    @Published private(set) var isWatchPaired = false
    @Published private(set) var isWatchAppInstalled = false
    @Published private(set) var lastSyncDate: Date?
    @Published private(set) var pendingTransfers = 0

    // MARK: - Dependencies

    private var session: WCSession?
    private let persistenceController: InflamAIPersistenceController
    private let sharedDataSync: SharedDataSyncService

    // MARK: - Initialization

    private override init() {
        self.persistenceController = .shared
        self.sharedDataSync = .shared
        super.init()

        setupWatchConnectivity()
    }

    // MARK: - Setup

    private func setupWatchConnectivity() {
        guard WCSession.isSupported() else {
            print("WatchSyncService: WatchConnectivity not supported on this device")
            return
        }

        session = WCSession.default
        session?.delegate = self
        session?.activate()
    }

    // MARK: - Send Data to Watch

    /// Send medications to Watch (immediate if reachable, queued otherwise)
    func sendMedicationsToWatch() async {
        guard let session = session, session.activationState == .activated else { return }

        let medications = await fetchMedicationsForWatch()

        let payload: [String: Any] = [
            "payloadType": "medications",
            "medications": medications,
            "generatedAt": Date().timeIntervalSince1970
        ]

        if session.isReachable {
            // Send immediately
            session.sendMessage(payload, replyHandler: nil) { error in
                print("WatchSyncService: Failed to send medications - \(error.localizedDescription)")
            }
        } else {
            // Queue for later
            do {
                try session.updateApplicationContext(payload)
                print("WatchSyncService: Queued medications via application context")
            } catch {
                print("WatchSyncService: Failed to queue medications - \(error.localizedDescription)")
            }
        }
    }

    /// Send activity summary to Watch
    func sendActivitySummaryToWatch() async {
        guard let session = session, session.activationState == .activated else { return }

        let summary = await buildActivitySummary()

        let payload: [String: Any] = [
            "payloadType": "activity_summary",
            "summary": summary,
            "generatedAt": Date().timeIntervalSince1970
        ]

        if session.isReachable {
            session.sendMessage(payload, replyHandler: nil) { error in
                print("WatchSyncService: Failed to send activity summary - \(error.localizedDescription)")
            }
        } else {
            do {
                try session.updateApplicationContext(payload)
            } catch {
                print("WatchSyncService: Failed to queue activity summary - \(error.localizedDescription)")
            }
        }
    }

    /// Send all widget data to Watch for complications/widgets
    func sendWidgetDataToWatch() async {
        guard let session = session, session.activationState == .activated else { return }

        let widgetData = sharedDataSync.getDataForWatch()

        let payload: [String: Any] = [
            "payloadType": "widget_data",
            "data": widgetData,
            "generatedAt": Date().timeIntervalSince1970
        ]

        // Use transferUserInfo for guaranteed delivery
        session.transferUserInfo(payload)
        pendingTransfers = session.outstandingUserInfoTransfers.count
    }

    /// Send flare risk prediction from ML engine to Watch
    func sendFlareRiskToWatch(
        riskPercentage: Int,
        riskLevel: String,
        confidence: String,
        topFactors: [String]
    ) async {
        guard let session = session, session.activationState == .activated else { return }

        let payload: [String: Any] = [
            "payloadType": "flare_risk",
            "prediction": [
                "riskPercentage": riskPercentage,
                "riskLevel": riskLevel,
                "confidence": confidence,
                "topFactors": topFactors,
                "updatedAt": Date().timeIntervalSince1970
            ]
        ]

        if session.isReachable {
            session.sendMessage(payload, replyHandler: nil) { error in
                print("WatchSyncService: Failed to send flare risk - \(error.localizedDescription)")
            }
        } else {
            do {
                try session.updateApplicationContext(payload)
                print("WatchSyncService: Queued flare risk via application context")
            } catch {
                print("WatchSyncService: Failed to queue flare risk - \(error.localizedDescription)")
            }
        }
    }

    /// Perform full sync to Watch
    func performFullSync() async {
        await sendMedicationsToWatch()
        await sendActivitySummaryToWatch()
        await sendWidgetDataToWatch()
        lastSyncDate = Date()

        // Update shared defaults for widgets
        AppGroupConfig.sharedDefaults?.set(Date(), forKey: WidgetDataKeys.lastWatchSync)
        AppGroupConfig.sharedDefaults?.set(isWatchReachable, forKey: WidgetDataKeys.watchConnected)
    }

    // MARK: - Data Fetching

    private func fetchMedicationsForWatch() async -> [[String: Any]] {
        let context = persistenceController.container.viewContext

        return await context.perform {
            let request: NSFetchRequest<Medication> = Medication.fetchRequest()
            request.predicate = NSPredicate(format: "isActive == YES")
            request.sortDescriptors = [NSSortDescriptor(keyPath: \Medication.name, ascending: true)]

            guard let medications = try? context.fetch(request) else { return [] }

            return medications.compactMap { med -> [String: Any]? in
                guard let id = med.id,
                      let name = med.name else { return nil }

                // Calculate next dose time
                let nextDoseTime = self.calculateNextDoseTime(for: med)
                let status = self.medicationStatus(nextDoseTime: nextDoseTime)

                // Get upcoming times
                let upcomingTimes: [TimeInterval] = (med.reminderTimes as? [Date] ?? [])
                    .map { $0.timeIntervalSince1970 }

                return [
                    "id": id.uuidString,
                    "name": name,
                    "dosage": med.dosage ?? "",
                    "frequency": med.frequency ?? "Daily",
                    "scheduledTime": nextDoseTime.timeIntervalSince1970,
                    "status": status,
                    "upcomingTimes": upcomingTimes
                ]
            }
        }
    }

    private func calculateNextDoseTime(for medication: Medication) -> Date {
        guard let reminderTimes = medication.reminderTimes as? [Date] else {
            return Calendar.current.date(byAdding: .hour, value: 1, to: Date()) ?? Date()
        }

        let now = Date()
        let calendar = Calendar.current

        for time in reminderTimes {
            let components = calendar.dateComponents([.hour, .minute], from: time)
            if let todayTime = calendar.date(bySettingHour: components.hour ?? 0,
                                             minute: components.minute ?? 0,
                                             second: 0,
                                             of: now) {
                if todayTime > now {
                    return todayTime
                }
            }
        }

        // All times passed today, return first time tomorrow
        if let firstTime = reminderTimes.first {
            let components = calendar.dateComponents([.hour, .minute], from: firstTime)
            let tomorrow = calendar.date(byAdding: .day, value: 1, to: now) ?? now
            return calendar.date(bySettingHour: components.hour ?? 8,
                                minute: components.minute ?? 0,
                                second: 0,
                                of: tomorrow) ?? tomorrow
        }

        return calendar.date(byAdding: .day, value: 1, to: now) ?? now
    }

    private func medicationStatus(nextDoseTime: Date) -> String {
        let now = Date()
        let fifteenMinutesAgo = now.addingTimeInterval(-15 * 60)

        if nextDoseTime < fifteenMinutesAgo {
            return "overdue"
        } else if nextDoseTime < now.addingTimeInterval(30 * 60) {
            return "due"
        } else {
            return "upcoming"
        }
    }

    private func buildActivitySummary() async -> [String: Any] {
        let context = persistenceController.container.viewContext

        return await context.perform {
            var summary: [String: Any] = [:]

            // Last medication event
            let medLogRequest: NSFetchRequest<DoseLog> = DoseLog.fetchRequest()
            medLogRequest.sortDescriptors = [NSSortDescriptor(keyPath: \DoseLog.timestamp, ascending: false)]
            medLogRequest.fetchLimit = 1

            if let lastDoseLog = try? context.fetch(medLogRequest).first,
               let id = lastDoseLog.id,
               let timestamp = lastDoseLog.timestamp {
                summary["lastMedication"] = [
                    "id": id.uuidString,
                    "medicationId": lastDoseLog.medication?.id?.uuidString ?? "",
                    "name": lastDoseLog.medication?.name ?? "Medication",
                    "timestamp": timestamp.timeIntervalSince1970,
                    "taken": lastDoseLog.taken,
                    "skipped": lastDoseLog.skipped,
                    "source": lastDoseLog.source ?? "ios"
                ]
            }

            // Last flare event
            let flareRequest: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            flareRequest.sortDescriptors = [NSSortDescriptor(keyPath: \FlareEvent.startDate, ascending: false)]
            flareRequest.fetchLimit = 1

            if let lastFlare = try? context.fetch(flareRequest).first,
               let id = lastFlare.id,
               let startDate = lastFlare.startDate {
                summary["lastFlare"] = [
                    "id": id.uuidString,
                    "timestamp": startDate.timeIntervalSince1970,
                    "severity": lastFlare.severity,
                    "resolved": lastFlare.isResolved
                ]
            }

            // Last symptom log (quick log equivalent)
            let symptomRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            symptomRequest.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]
            symptomRequest.fetchLimit = 1

            if let lastLog = try? context.fetch(symptomRequest).first,
               let id = lastLog.id,
               let timestamp = lastLog.timestamp {
                summary["lastQuickLog"] = [
                    "id": id.uuidString,
                    "timestamp": timestamp.timeIntervalSince1970,
                    "painScore": Int(lastLog.basdaiScore),
                    "stiffnessScore": Int(lastLog.morningStiffnessMinutes / 10),
                    "fatigueScore": Int(lastLog.fatigueLevel),
                    "isFlare": lastLog.isFlareEvent
                ]
            }

            return summary
        }
    }

    // MARK: - Handle Watch Requests

    private func handleMedicationSyncRequest(replyHandler: @escaping ([String: Any]) -> Void) {
        Task {
            let medications = await fetchMedicationsForWatch()
            replyHandler([
                "medications": medications,
                "generatedAt": Date().timeIntervalSince1970
            ])
        }
    }

    private func handleStatusPing(replyHandler: @escaping ([String: Any]) -> Void) {
        Task {
            let summary = await buildActivitySummary()
            replyHandler([
                "summary": summary,
                "generatedAt": Date().timeIntervalSince1970
            ])
        }
    }

    // MARK: - Handle Data from Watch

    private func handleQuickLogFromWatch(_ data: [String: Any]) {
        // Support both naming conventions: pain/stiffness/fatigue and painScore/stiffnessScore/fatigueScore
        let painScore = (data["pain"] as? Int) ?? (data["painScore"] as? Int) ?? 0
        let stiffnessScore = (data["stiffness"] as? Int) ?? (data["stiffnessScore"] as? Int) ?? 0
        let fatigueScore = (data["fatigue"] as? Int) ?? (data["fatigueScore"] as? Int) ?? 0

        // Require at least one value
        guard painScore > 0 || stiffnessScore > 0 || fatigueScore > 0 else {
            print("WatchSyncService: Quick log has no symptom data")
            return
        }

        let isFlare = data["isFlare"] as? Bool ?? false

        // Save to Core Data
        let context = persistenceController.container.viewContext
        let log = SymptomLog(context: context)
        log.id = UUID()
        log.timestamp = Date()
        log.basdaiScore = Double(painScore)
        log.fatigueLevel = Int16(fatigueScore)
        log.morningStiffnessMinutes = Int16(stiffnessScore * 10)
        log.isFlareEvent = isFlare
        log.source = "watch"

        do {
            try context.save()
            print("WatchSyncService: Saved quick log from Watch - pain: \(painScore), stiffness: \(stiffnessScore), fatigue: \(fatigueScore)")

            // Trigger widget update
            Task {
                await sharedDataSync.performFullSync()
            }
        } catch {
            print("WatchSyncService: Failed to save quick log - \(error)")
        }
    }

    private func handleMedicationLogFromWatch(_ data: [String: Any]) {
        guard let medicationIdString = data["medicationId"] as? String,
              let medicationId = UUID(uuidString: medicationIdString),
              let taken = data["taken"] as? Bool else {
            return
        }

        let context = persistenceController.container.viewContext

        // Find the medication
        let request: NSFetchRequest<Medication> = Medication.fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", medicationId as CVarArg)
        request.fetchLimit = 1

        guard let medication = try? context.fetch(request).first else {
            print("WatchSyncService: Medication not found for ID \(medicationId)")
            return
        }

        // Create dose log
        let doseLog = DoseLog(context: context)
        doseLog.id = UUID()
        doseLog.timestamp = Date()
        doseLog.taken = taken
        doseLog.skipped = !taken
        doseLog.medication = medication
        doseLog.source = "watch"

        do {
            try context.save()
            print("WatchSyncService: Saved medication log from Watch")

            // Trigger widget update
            Task {
                await sharedDataSync.performFullSync()
            }
        } catch {
            print("WatchSyncService: Failed to save medication log - \(error)")
        }
    }

    private func handleFlareLogFromWatch(_ data: [String: Any]) {
        guard let severity = data["severity"] as? Int else { return }

        let context = persistenceController.container.viewContext
        let flare = FlareEvent(context: context)
        flare.id = UUID()
        flare.startDate = Date()
        flare.severity = Int16(severity)
        flare.isResolved = false
        flare.notes = data["notes"] as? String

        do {
            try context.save()
            print("WatchSyncService: Saved flare event from Watch")

            Task {
                await sharedDataSync.performFullSync()
            }
        } catch {
            print("WatchSyncService: Failed to save flare event - \(error)")
        }
    }
}

// MARK: - WCSessionDelegate

extension WatchSyncService: WCSessionDelegate {

    nonisolated func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        Task { @MainActor in
            if let error = error {
                print("WatchSyncService: Activation failed - \(error.localizedDescription)")
                return
            }

            isWatchPaired = session.isPaired
            isWatchAppInstalled = session.isWatchAppInstalled
            isWatchReachable = session.isReachable

            print("WatchSyncService: Activated - paired: \(session.isPaired), installed: \(session.isWatchAppInstalled), reachable: \(session.isReachable)")

            if activationState == .activated && session.isReachable {
                await performFullSync()
            }
        }
    }

    nonisolated func sessionDidBecomeInactive(_ session: WCSession) {
        print("WatchSyncService: Session became inactive")
    }

    nonisolated func sessionDidDeactivate(_ session: WCSession) {
        print("WatchSyncService: Session deactivated, reactivating...")
        session.activate()
    }

    nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
        Task { @MainActor in
            isWatchReachable = session.isReachable
            print("WatchSyncService: Reachability changed - \(session.isReachable)")

            if session.isReachable {
                await performFullSync()
            }
        }
    }

    nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any], replyHandler: @escaping ([String: Any]) -> Void) {
        guard let type = message["type"] as? String else {
            replyHandler(["error": "Missing message type"])
            return
        }

        Task { @MainActor in
            switch type {
            case "request_medication_sync":
                handleMedicationSyncRequest(replyHandler: replyHandler)

            case "status_ping":
                handleStatusPing(replyHandler: replyHandler)

            case "symptom_log", "quick_log":
                handleQuickLogFromWatch(message)
                replyHandler(["success": true, "message": "Symptom log saved"])

            case "medication_log", "medication_taken":
                handleMedicationLogFromWatch(message)
                replyHandler(["success": true, "message": "Medication log saved"])

            case "flare_log", "flare_event":
                handleFlareLogFromWatch(message)
                replyHandler(["success": true, "message": "Flare event saved"])

            default:
                replyHandler(["status": "unknown_type", "success": false])
            }
        }
    }

    nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        guard let type = message["type"] as? String else { return }

        Task { @MainActor in
            switch type {
            case "quick_log", "symptom_log":
                handleQuickLogFromWatch(message)

            case "medication_log", "medication_taken":
                handleMedicationLogFromWatch(message)

            case "flare_log", "flare_event":
                handleFlareLogFromWatch(message)

            default:
                print("WatchSyncService: Unknown message type - \(type)")
            }
        }
    }

    nonisolated func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
        guard let type = userInfo["type"] as? String else { return }

        Task { @MainActor in
            switch type {
            case "quick_log":
                handleQuickLogFromWatch(userInfo)

            case "medication_log":
                handleMedicationLogFromWatch(userInfo)

            case "flare_log":
                handleFlareLogFromWatch(userInfo)

            default:
                print("WatchSyncService: Unknown userInfo type - \(type)")
            }
        }
    }

    nonisolated func session(_ session: WCSession, didFinish userInfoTransfer: WCSessionUserInfoTransfer, error: Error?) {
        Task { @MainActor in
            pendingTransfers = session.outstandingUserInfoTransfers.count

            if let error = error {
                print("WatchSyncService: User info transfer failed - \(error.localizedDescription)")
            } else {
                print("WatchSyncService: User info transfer completed")
            }
        }
    }
}
