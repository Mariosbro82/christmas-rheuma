//
//  ExerciseStep.swift
//  InflamAI-Swift
//
//  Exercise step model for step-by-step coaching
//

import Foundation

/// Type of exercise step
enum StepType: String, Codable {
    case info          // Information only (setup, positioning)
    case timer         // Time-based (hold for X seconds)
    case repetitions   // Rep-based (do X repetitions)
}

/// Individual step within an exercise
struct ExerciseStep: Identifiable, Codable {
    let id: UUID
    let instruction: String
    let type: StepType
    let duration: Int?        // Seconds for timer-based steps
    let repetitions: Int?     // Count for rep-based steps
    let imageHint: String?    // SF Symbol name for visual aid

    init(
        id: UUID = UUID(),
        instruction: String,
        type: StepType,
        duration: Int? = nil,
        repetitions: Int? = nil,
        imageHint: String? = nil
    ) {
        self.id = id
        self.instruction = instruction
        self.type = type
        self.duration = duration
        self.repetitions = repetitions
        self.imageHint = imageHint
    }

    /// Creates an info step (no timer or reps)
    static func info(_ instruction: String, imageHint: String? = nil) -> ExerciseStep {
        ExerciseStep(instruction: instruction, type: .info, imageHint: imageHint)
    }

    /// Creates a timer-based step
    static func timer(_ instruction: String, duration: Int, imageHint: String? = nil) -> ExerciseStep {
        ExerciseStep(instruction: instruction, type: .timer, duration: duration, imageHint: imageHint)
    }

    /// Creates a repetition-based step
    static func reps(_ instruction: String, repetitions: Int, imageHint: String? = nil) -> ExerciseStep {
        ExerciseStep(instruction: instruction, type: .repetitions, repetitions: repetitions, imageHint: imageHint)
    }
}

/// Exercise feedback from user after completion
enum ExerciseFeedback: String, Codable {
    case easy           // 😊 Easy / No problem
    case manageable     // 😐 Manageable
    case difficult      // 😰 Difficult but doable
    case unbearable     // ⚠️ Unbearable / Had to stop

    var emoji: String {
        switch self {
        case .easy: return "😊"
        case .manageable: return "😐"
        case .difficult: return "😰"
        case .unbearable: return "⚠️"
        }
    }

    var description: String {
        switch self {
        case .easy: return "Easy / No problem"
        case .manageable: return "Manageable"
        case .difficult: return "Difficult but doable"
        case .unbearable: return "Unbearable / Had to stop"
        }
    }
}
