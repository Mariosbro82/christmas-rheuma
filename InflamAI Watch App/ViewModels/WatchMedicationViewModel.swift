//
//  WatchMedicationViewModel.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import Foundation
import Combine
import UserNotifications

@MainActor
class WatchMedicationViewModel: ObservableObject {
    @Published var dueMedications: [WatchMedicationSummary] = []
    @Published var upcomingMedications: [WatchMedicationSummary] = []

    private let connectivityManager: WatchConnectivityManager
    private var cancellables = Set<AnyCancellable>()

    init(connectivityManager: WatchConnectivityManager? = nil) {
        let resolvedManager = connectivityManager ?? WatchConnectivityManager.shared
        self.connectivityManager = resolvedManager

        resolvedManager.$medications
            .receive(on: DispatchQueue.main)
            .sink { [weak self] medications in
                self?.updateSections(with: medications)
            }
            .store(in: &cancellables)
    }

    func loadMedications() async {
        updateSections(with: connectivityManager.medications)

        if connectivityManager.medications.isEmpty {
            await connectivityManager.requestMedicationSync()
        }
    }

    func markTaken(_ medication: WatchMedicationSummary) async {
        let message: [String: Any] = [
            "type": "medication_taken",
            "id": medication.id.uuidString,
            "timestamp": Date().timeIntervalSince1970
        ]

        let reply = await connectivityManager.sendMessage(message)
        let success = (reply?["success"] as? Bool) ?? (reply?["queued"] as? Bool) ?? false

        if success {
            dueMedications.removeAll { $0.id == medication.id }
            await connectivityManager.requestMedicationSync()
        }
    }

    func markSkipped(_ medication: WatchMedicationSummary) async {
        let message: [String: Any] = [
            "type": "medication_skipped",
            "id": medication.id.uuidString,
            "timestamp": Date().timeIntervalSince1970
        ]

        let reply = await connectivityManager.sendMessage(message)
        let success = (reply?["success"] as? Bool) ?? (reply?["queued"] as? Bool) ?? false

        if success {
            dueMedications.removeAll { $0.id == medication.id }
            await connectivityManager.requestMedicationSync()
        }
    }

    func snooze(_ medication: WatchMedicationSummary, minutes: Int) async {
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

        dueMedications.removeAll { $0.id == medication.id }
    }

    private func updateSections(with medications: [WatchMedicationSummary]) {
        dueMedications = medications.filter { $0.status != .upcoming }
        upcomingMedications = medications.filter { $0.status == .upcoming }
    }
}
