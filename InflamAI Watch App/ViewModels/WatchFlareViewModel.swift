//
//  WatchFlareViewModel.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import Foundation
import Combine

@MainActor
class WatchFlareViewModel: ObservableObject {
    @Published var isSubmitting = false
    @Published var errorMessage: String?

    private let connectivityManager = WatchConnectivityManager.shared

    func logFlare(
        severity: Int,
        symptoms: [String],
        triggers: [String],
        note: String?
    ) async -> Bool {
        isSubmitting = true
        errorMessage = nil

        var message: [String: Any] = [
            "type": "flare_quick_log",
            "timestamp": Date().timeIntervalSince1970,
            "severity": severity,
            "symptoms": symptoms,
            "triggers": triggers
        ]

        if let note, !note.isEmpty {
            message["note"] = note
        }

        let reply = await connectivityManager.sendMessage(message)

        isSubmitting = false

        if let success = reply?["success"] as? Bool {
            return success
        }

        if let queued = reply?["queued"] as? Bool {
            return queued
        }

        errorMessage = reply?["error"] as? String
        return false
    }
}
