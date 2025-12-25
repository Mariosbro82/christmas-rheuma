//
//  WatchConnectivityManager.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import Foundation
import Combine
import WatchConnectivity

struct WatchMedicationSummary: Identifiable, Equatable {
    enum Status: String {
        case due
        case overdue
        case upcoming
    }

    let id: UUID
    let name: String
    let dosage: String
    let frequency: String
    let scheduledTime: Date
    let status: Status
    let upcomingTimes: [Date]

    var isOverdue: Bool {
        status == .overdue || Date() > scheduledTime.addingTimeInterval(15 * 60)
    }
}

/// Flare risk prediction data from Neural Engine
struct WatchFlareRiskPrediction: Equatable {
    let riskPercentage: Int
    let riskLevel: String
    let confidence: String
    let topFactors: [String]
    let updatedAt: Date

    var riskColor: String {
        switch riskPercentage {
        case 0..<25: return "green"
        case 25..<50: return "yellow"
        case 50..<75: return "orange"
        default: return "red"
        }
    }

    init?(from dict: [String: Any]) {
        guard let percentage = dict["riskPercentage"] as? Int,
              let level = dict["riskLevel"] as? String,
              let timestamp = dict["updatedAt"] as? TimeInterval else {
            return nil
        }

        self.riskPercentage = percentage
        self.riskLevel = level
        self.confidence = dict["confidence"] as? String ?? "Low"
        self.topFactors = dict["topFactors"] as? [String] ?? []
        self.updatedAt = Date(timeIntervalSince1970: timestamp)
    }
}

struct WatchActivitySummary: Equatable {
    struct MedicationEvent: Equatable {
        let id: UUID
        let medicationId: UUID?
        let name: String
        let timestamp: Date
        let taken: Bool
        let skipped: Bool
        let source: String
    }

    struct FlareEvent: Equatable {
        let id: UUID
        let timestamp: Date
        let severity: Int
        let resolved: Bool
    }

    struct QuickLogEvent: Equatable {
        let id: UUID
        let timestamp: Date
        let painScore: Int
        let stiffnessScore: Int
        let fatigueScore: Int
        let isFlare: Bool
    }

    let generatedAt: Date
    let lastMedication: MedicationEvent?
    let lastFlare: FlareEvent?
    let lastQuickLog: QuickLogEvent?

    init?(payload: [String: Any], generatedAt: TimeInterval?) {
        let generatedDate = generatedAt.map { Date(timeIntervalSince1970: $0) } ?? Date()

        var medicationEvent: MedicationEvent?
        if let medicationDict = payload["lastMedication"] as? [String: Any],
           let idString = medicationDict["id"] as? String,
           let eventId = UUID(uuidString: idString),
           let timestamp = medicationDict["timestamp"] as? TimeInterval {
            let medicationId = (medicationDict["medicationId"] as? String).flatMap(UUID.init(uuidString:))
            let name = medicationDict["name"] as? String ?? "Medication"
            let taken = medicationDict["taken"] as? Bool ?? false
            let skipped = medicationDict["skipped"] as? Bool ?? false
            let source = medicationDict["source"] as? String ?? "unknown"
            medicationEvent = MedicationEvent(
                id: eventId,
                medicationId: medicationId,
                name: name,
                timestamp: Date(timeIntervalSince1970: timestamp),
                taken: taken,
                skipped: skipped,
                source: source
            )
        }

        var flareEvent: FlareEvent?
        if let flareDict = payload["lastFlare"] as? [String: Any],
           let idString = flareDict["id"] as? String,
           let flareId = UUID(uuidString: idString),
           let timestamp = flareDict["timestamp"] as? TimeInterval {
            let severity = flareDict["severity"] as? Int ?? 0
            let resolved = flareDict["resolved"] as? Bool ?? false
            flareEvent = FlareEvent(
                id: flareId,
                timestamp: Date(timeIntervalSince1970: timestamp),
                severity: severity,
                resolved: resolved
            )
        }

        var quickLogEvent: QuickLogEvent?
        if let quickLogDict = payload["lastQuickLog"] as? [String: Any],
           let idString = quickLogDict["id"] as? String,
           let quickLogId = UUID(uuidString: idString),
           let timestamp = quickLogDict["timestamp"] as? TimeInterval {
            let painScore = quickLogDict["painScore"] as? Int ?? 0
            let stiffness = quickLogDict["stiffnessScore"] as? Int ?? 0
            let fatigue = quickLogDict["fatigueScore"] as? Int ?? 0
            let isFlare = quickLogDict["isFlare"] as? Bool ?? false

            quickLogEvent = QuickLogEvent(
                id: quickLogId,
                timestamp: Date(timeIntervalSince1970: timestamp),
                painScore: painScore,
                stiffnessScore: stiffness,
                fatigueScore: fatigue,
                isFlare: isFlare
            )
        }

        if medicationEvent == nil && flareEvent == nil && quickLogEvent == nil {
            return nil
        }

        self.generatedAt = generatedDate
        self.lastMedication = medicationEvent
        self.lastFlare = flareEvent
        self.lastQuickLog = quickLogEvent
    }
}

@MainActor
class WatchConnectivityManager: NSObject, ObservableObject {
    static let shared = WatchConnectivityManager()

    @Published var isReachable = false
    @Published var medications: [WatchMedicationSummary] = []
    @Published var activitySummary: WatchActivitySummary?
    @Published var flareRiskPrediction: WatchFlareRiskPrediction?

    private var session: WCSession?
    private var refreshTimer: Timer?
    private let refreshInterval: TimeInterval = 15 * 60

    private override init() {
        super.init()

        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
            session?.activate()
        }

        startAutoRefresh()
    }

    deinit {
        refreshTimer?.invalidate()
    }

    @discardableResult
    func sendMessage(_ message: [String: Any]) async -> [String: Any]? {
        guard let session = session else { return nil }

        if session.isReachable {
            do {
                return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[String: Any], Error>) in
                    session.sendMessage(
                        message,
                        replyHandler: { reply in
                            continuation.resume(returning: reply)
                        },
                        errorHandler: { error in
                            continuation.resume(throwing: error)
                        }
                    )
                }
            } catch {
                print("❌ Failed to send message: \(error.localizedDescription)")
                session.transferUserInfo(message)
                return ["queued": true]
            }
        }

        session.transferUserInfo(message)
        return ["queued": true]
    }

    private func startAutoRefresh() {
        refreshTimer?.invalidate()

        refreshTimer = Timer.scheduledTimer(withTimeInterval: refreshInterval, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { await self.performAutoRefresh(triggeredByTimer: true) }
        }

        Task { await performAutoRefresh(triggeredByTimer: false) }
    }

    private func performAutoRefresh(triggeredByTimer: Bool) async {
        await requestMedicationSync()
        await refreshActivitySummary()

        if triggeredByTimer {
            print("⏱️ Watch performed scheduled refresh")
        } else {
            print("🚀 Watch performed initial refresh")
        }
    }

    private func refreshActivitySummary() async {
        let reply = await sendMessage(["type": "status_ping"])

        if let queued = reply?["queued"] as? Bool, queued {
            return
        }

        if let summaryDict = reply?["summary"] as? [String: Any] {
            let generatedAt = reply?["generatedAt"] as? TimeInterval
            await updateActivitySummary(summaryDict, generatedAt: generatedAt)
        }
    }

    func requestMedicationSync() async {
        let reply = await sendMessage(["type": "request_medication_sync"])

        if let queued = reply?["queued"] as? Bool, queued {
            return
        }

        if let medicationsData = reply?["medications"] as? [[String: Any]] {
            await updateLocalMedications(medicationsData)
        }
    }

    private func updateActivitySummary(_ summary: [String: Any], generatedAt: TimeInterval?) async {
        guard !summary.isEmpty else {
            activitySummary = nil
            return
        }

        guard let parsed = WatchActivitySummary(payload: summary, generatedAt: generatedAt) else {
            return
        }

        activitySummary = parsed
        print("📝 Updated activity summary on Watch")
    }
}

// MARK: - WCSessionDelegate

extension WatchConnectivityManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        Task { @MainActor in
            isReachable = session.isReachable

            if let error = error {
                print("❌ Watch session activation error: \(error.localizedDescription)")
            } else {
                print("✅ Watch session activated: \(activationState.rawValue)")
            }

            if activationState == .activated {
                await self.performAutoRefresh(triggeredByTimer: false)
            }
        }
    }

    func sessionReachabilityDidChange(_ session: WCSession) {
        Task { @MainActor in
            isReachable = session.isReachable
            print("📡 Watch reachability changed: \(isReachable)")

            if session.isReachable {
                await self.performAutoRefresh(triggeredByTimer: false)
            }
        }
    }

    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
        Task { @MainActor in
            print("📥 Watch received application context: \(applicationContext.keys)")

            if let payloadType = applicationContext["payloadType"] as? String {
                switch payloadType {
                case "medications":
                    if let medications = applicationContext["medications"] as? [[String: Any]] {
                        await updateLocalMedications(medications)
                    }
                case "activity_summary":
                    if let summary = applicationContext["summary"] as? [String: Any] {
                        let generatedAt = applicationContext["generatedAt"] as? TimeInterval
                        await updateActivitySummary(summary, generatedAt: generatedAt)
                    }
                case "flare_risk":
                    if let riskData = applicationContext["prediction"] as? [String: Any] {
                        await updateFlareRiskPrediction(riskData)
                    }
                default:
                    break
                }
            }
        }
    }

    func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
        Task { @MainActor in
            print("📥 Watch received user info: \(userInfo.keys)")

            if let payloadType = userInfo["payloadType"] as? String {
                switch payloadType {
                case "medications":
                    if let medications = userInfo["medications"] as? [[String: Any]] {
                        await updateLocalMedications(medications)
                    }
                case "activity_summary":
                    if let summary = userInfo["summary"] as? [String: Any] {
                        let generatedAt = userInfo["generatedAt"] as? TimeInterval
                        await updateActivitySummary(summary, generatedAt: generatedAt)
                    }
                case "flare_risk":
                    if let riskData = userInfo["prediction"] as? [String: Any] {
                        await updateFlareRiskPrediction(riskData)
                    }
                default:
                    break
                }
            }
        }
    }

    private func updateFlareRiskPrediction(_ riskData: [String: Any]) async {
        guard let prediction = WatchFlareRiskPrediction(from: riskData) else {
            print("⚠️ Failed to parse flare risk prediction")
            return
        }

        flareRiskPrediction = prediction
        print("📊 Updated flare risk prediction on Watch: \(prediction.riskPercentage)%")
    }

    private func updateLocalMedications(_ medicationsData: [[String: Any]]) async {
        let parsed: [WatchMedicationSummary] = medicationsData.compactMap { item in
            guard
                let idString = item["id"] as? String,
                let id = UUID(uuidString: idString),
                let name = item["name"] as? String,
                let dosage = item["dosage"] as? String,
                let frequency = item["frequency"] as? String,
                let timestamp = item["scheduledTime"] as? TimeInterval,
                let statusString = item["status"] as? String,
                let status = WatchMedicationSummary.Status(rawValue: statusString)
            else {
                return nil
            }

            let upcomingTimes: [Date] = (item["upcomingTimes"] as? [TimeInterval] ?? []).map {
                Date(timeIntervalSince1970: $0)
            }

            return WatchMedicationSummary(
                id: id,
                name: name,
                dosage: dosage,
                frequency: frequency,
                scheduledTime: Date(timeIntervalSince1970: timestamp),
                status: status,
                upcomingTimes: upcomingTimes
            )
        }

        medications = parsed.sorted { lhs, rhs in
            lhs.scheduledTime < rhs.scheduledTime
        }

        print("📝 Updated \(parsed.count) medications on Watch")
    }
}
