//
//  TraeAmKochenApp.swift
//  TraeAmKochen
//
//  Created by Codex on 2024-05-29.
//
//  DEPRECATED: Use InflamAIApp.swift instead

import SwiftUI

// @main - REMOVED: Using InflamAIApp as main entry point
struct TraeAmKochenApp_Deprecated: App {
    @StateObject private var environment = TraeAppEnvironment()
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @State private var showOnboarding: Bool
    @Environment(\.scenePhase) private var scenePhase

    init() {
        _showOnboarding = State(initialValue: !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding"))
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(environment)
                .task {
                    environment.cacheOfflineAssets()
                }
                .onAppear(perform: synchronizeOnboardingState)
                .onChange(of: hasCompletedOnboarding) { _ in
                    synchronizeOnboardingState()
                }
                .onChange(of: scenePhase) { phase in
                    if phase == .active {
                        synchronizeOnboardingState()
                    }
                }
                .fullScreenCover(isPresented: $showOnboarding, onDismiss: synchronizeOnboardingState) {
                    OnboardingFlow()
                        .interactiveDismissDisabled()
                }
        }
    }

    private func synchronizeOnboardingState() {
        showOnboarding = !hasCompletedOnboarding
    }
}
