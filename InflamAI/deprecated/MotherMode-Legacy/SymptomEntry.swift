//
//  SymptomEntry.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation

struct SymptomEntry: Identifiable, Equatable {
    let id: UUID
    let loggedAt: Date
    let pain: Double
    let stiffnessMinutes: Int
    let fatigue: Double
    let sleepQuality: Double
    let mobilityCompleted: Bool
    let notes: String
}
