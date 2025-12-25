//
//  WatchSymptomViewModel.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import Foundation
import Combine

@MainActor
class WatchSymptomViewModel: ObservableObject {
    private let connectivityManager = WatchConnectivityManager.shared

    func logSymptoms(
        pain: Int,
        stiffness: Int,
        fatigue: Int,
        markAsFlare: Bool
    ) async -> Bool {
        let logId = UUID()
        let timestamp = Date()

        let message: [String: Any] = [
            "type": "symptom_log",
            "id": logId.uuidString,
            "timestamp": timestamp.timeIntervalSince1970,
            "pain": pain,
            "stiffness": stiffness,
            "fatigue": fatigue,
            "isFlare": markAsFlare
        ]

        let reply = await connectivityManager.sendMessage(message)

        if let success = reply?["success"] as? Bool {
            return success
        }

        if let queued = reply?["queued"] as? Bool {
            return queued
        }

        return false
    }
}
