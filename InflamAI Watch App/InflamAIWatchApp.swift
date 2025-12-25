//
//  InflamAIWatchApp.swift
//  InflamAI Watch App
//
//  Created by Claude Code on 2025-10-28.
//

import SwiftUI

@main
struct InflamAIWatchApp: App {
    @StateObject private var connectivityManager = WatchConnectivityManager.shared
    @StateObject private var healthViewModel = WatchHealthViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(connectivityManager)
                .environmentObject(healthViewModel)
        }
    }
}
