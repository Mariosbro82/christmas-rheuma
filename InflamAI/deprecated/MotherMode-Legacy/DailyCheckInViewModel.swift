//
//  DailyCheckInViewModel.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation

@MainActor
final class DailyCheckInViewModel: ObservableObject {
    @Published var pain: Double = 4
    @Published var stiffnessMinutes: Int = 30
    @Published var fatigue: Double = 5
    @Published var sleepQuality: Double = 6
    @Published var mobilityCompleted: Bool = false
    @Published var notes: String = ""
    
    private weak var environment: TraeAppEnvironment?
    
    init(environment: TraeAppEnvironment) {
        self.environment = environment
    }
    
    func saveEntry() {
        let entry = SymptomEntry(
            id: UUID(),
            loggedAt: Date(),
            pain: pain,
            stiffnessMinutes: stiffnessMinutes,
            fatigue: fatigue,
            sleepQuality: sleepQuality,
            mobilityCompleted: mobilityCompleted,
            notes: notes
        )
        environment?.logSymptomEntry(entry)
    }
}
