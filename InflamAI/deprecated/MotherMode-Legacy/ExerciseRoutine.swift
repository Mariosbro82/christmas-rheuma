//
//  ExerciseRoutine.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import SwiftUI

struct ExerciseRoutine: Identifiable, Equatable {
    struct Step: Identifiable, Equatable {
        let id = UUID()
        let title: String
        let instruction: String
        let durationSeconds: Int
        let modification: String?
    }
    
    let id: UUID
    let titleKey: LocalizedStringResource
    let subtitleKey: LocalizedStringResource
    let estimatedMinutes: Int
    let steps: [Step]
    
    static func morningMobility() -> ExerciseRoutine {
        ExerciseRoutine(
            id: UUID(),
            titleKey: "routine.morning_mobility.title",
            subtitleKey: "routine.morning_mobility.subtitle",
            estimatedMinutes: 6,
            steps: [
                Step(title: "Gentle spinal roll-down", instruction: "Roll down vertebra by vertebra, soften knees.", durationSeconds: 60, modification: "Use wall support if balance is limited."),
                Step(title: "Low lunge opener", instruction: "Step into low lunge, breathe calmly for 30s each side.", durationSeconds: 90, modification: "Use chair for support."),
                Step(title: "Thoracic rotation", instruction: "Kneel, hand behind head, rotate gently through upper spine.", durationSeconds: 90, modification: "Reduce range if discomfort."),
                Step(title: "Diaphragmatic breathing", instruction: "Inhale through nose, exhale slowly to calm nervous system.", durationSeconds: 120, modification: nil)
            ])
    }
    
    static func deskUnwind() -> ExerciseRoutine {
        ExerciseRoutine(
            id: UUID(),
            titleKey: "routine.desk_unwind.title",
            subtitleKey: "routine.desk_unwind.subtitle",
            estimatedMinutes: 3,
            steps: [
                Step(title: "Neck reset", instruction: "Slow nods and gentle turns within comfort.", durationSeconds: 60, modification: "Keep movements small if dizzy."),
                Step(title: "Shoulder rolls", instruction: "Roll shoulders forward and back with relaxed breathing.", durationSeconds: 60, modification: nil),
                Step(title: "Seated twist", instruction: "Rotate torso gently, hold for three calm breaths each side.", durationSeconds: 60, modification: "Hold chair back for support.")
            ])
    }
}
