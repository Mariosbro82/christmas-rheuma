//
//  BreathingPattern.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import Foundation

// MARK: - Breathing Technique

enum BreathingTechnique: String, CaseIterable, Codable {
    case fourSevenEight = "fourSevenEight"
    case boxBreathing = "boxBreathing"
    case triangleBreathing = "triangleBreathing"
    case coherentBreathing = "coherentBreathing"
    case bellowsBreath = "bellowsBreath"
    case alternateNostril = "alternateNostril"
    case deepBelly = "deepBelly"
    case pursedLip = "pursedLip"
    case resonantBreathing = "resonantBreathing"
    case custom = "custom"

    var displayName: String {
        switch self {
        case .fourSevenEight: return "4-7-8 Breathing"
        case .boxBreathing: return "Box Breathing"
        case .triangleBreathing: return "Triangle Breathing"
        case .coherentBreathing: return "Coherent Breathing"
        case .bellowsBreath: return "Bellows Breath"
        case .alternateNostril: return "Alternate Nostril"
        case .deepBelly: return "Deep Belly Breathing"
        case .pursedLip: return "Pursed Lip Breathing"
        case .resonantBreathing: return "Resonant Breathing"
        case .custom: return "Custom Pattern"
        }
    }

    var description: String {
        switch self {
        case .fourSevenEight:
            return "Inhale for 4, hold for 7, exhale for 8. Calms nervous system and reduces anxiety."
        case .boxBreathing:
            return "Equal parts inhale, hold, exhale, hold (4-4-4-4). Used by Navy SEALs for stress management."
        case .triangleBreathing:
            return "Inhale for 4, hold for 4, exhale for 4. Simple and effective for beginners."
        case .coherentBreathing:
            return "Inhale for 5, exhale for 5. Optimal for heart rate variability and relaxation."
        case .bellowsBreath:
            return "Rapid forceful breathing for 1 second each. Energizing and increases alertness."
        case .alternateNostril:
            return "Breathe through alternating nostrils. Balances left and right brain hemispheres."
        case .deepBelly:
            return "Long inhale (6), hold (2), longer exhale (8). Activates parasympathetic nervous system."
        case .pursedLip:
            return "Inhale through nose, exhale through pursed lips. Helps with breathlessness."
        case .resonantBreathing:
            return "Breathe at 6 seconds in, 6 seconds out. Maximizes heart rate variability."
        case .custom:
            return "Create your own breathing pattern based on your needs."
        }
    }

    /// Default breathing pattern (inhale, hold, exhale, pause) in seconds
    var defaultPattern: (inhale: Int, hold: Int, exhale: Int, pause: Int) {
        switch self {
        case .fourSevenEight: return (4, 7, 8, 0)
        case .boxBreathing: return (4, 4, 4, 4)
        case .triangleBreathing: return (4, 4, 4, 0)
        case .coherentBreathing: return (5, 0, 5, 0)
        case .bellowsBreath: return (1, 0, 1, 0)
        case .alternateNostril: return (4, 2, 4, 2)
        case .deepBelly: return (6, 2, 8, 2)
        case .pursedLip: return (2, 0, 4, 0)
        case .resonantBreathing: return (6, 0, 6, 0)
        case .custom: return (4, 0, 4, 0)
        }
    }

    /// Recommended for specific AS symptoms
    var recommendedFor: [TargetSymptom] {
        switch self {
        case .fourSevenEight:
            return [.anxiety, .stress, .insomnia, .flare]
        case .boxBreathing:
            return [.anxiety, .stress, .chronicPain]
        case .triangleBreathing:
            return [.stress, .anxiety]
        case .coherentBreathing:
            return [.inflammation, .spinalPain, .chronicPain]
        case .bellowsBreath:
            return [.fatigue, .brainfog, .lowMood]
        case .alternateNostril:
            return [.stress, .anxiety, .headaches]
        case .deepBelly:
            return [.spinalPain, .inflammation, .morningStiffness]
        case .pursedLip:
            return [.fatigue, .muscleTension]
        case .resonantBreathing:
            return [.inflammation, .chronicPain, .stress]
        case .custom:
            return []
        }
    }

    /// Difficulty level
    var difficulty: DifficultyLevel {
        switch self {
        case .triangleBreathing, .coherentBreathing, .deepBelly:
            return .beginner
        case .fourSevenEight, .boxBreathing, .pursedLip, .resonantBreathing:
            return .intermediate
        case .bellowsBreath, .alternateNostril, .custom:
            return .advanced
        }
    }
}

// MARK: - Breathing Pattern Model

struct BreathingPattern: Codable, Identifiable {
    let id: UUID
    let technique: BreathingTechnique
    let inhaleCount: Int
    let holdCount: Int
    let exhaleCount: Int
    let pauseCount: Int
    let totalCycles: Int
    let duration: TimeInterval

    init(
        id: UUID = UUID(),
        technique: BreathingTechnique,
        inhaleCount: Int? = nil,
        holdCount: Int? = nil,
        exhaleCount: Int? = nil,
        pauseCount: Int? = nil,
        totalCycles: Int = 10,
        duration: TimeInterval? = nil
    ) {
        self.id = id
        self.technique = technique

        let defaultPattern = technique.defaultPattern
        self.inhaleCount = inhaleCount ?? defaultPattern.inhale
        self.holdCount = holdCount ?? defaultPattern.hold
        self.exhaleCount = exhaleCount ?? defaultPattern.exhale
        self.pauseCount = pauseCount ?? defaultPattern.pause
        self.totalCycles = totalCycles

        // Calculate duration if not provided
        let cycleDuration = Double(self.inhaleCount + self.holdCount + self.exhaleCount + self.pauseCount)
        self.duration = duration ?? (cycleDuration * Double(totalCycles))
    }

    var cyclesPerMinute: Double {
        let cycleDuration = Double(inhaleCount + holdCount + exhaleCount + pauseCount)
        return 60.0 / cycleDuration
    }

    var displayName: String {
        technique.displayName
    }

    var description: String {
        technique.description
    }

    /// Create pattern from technique using defaults
    static func from(technique: BreathingTechnique, cycles: Int = 10) -> BreathingPattern {
        return BreathingPattern(technique: technique, totalCycles: cycles)
    }

    /// AS-specific recommended breathing patterns
    static let asRecommended: [BreathingPattern] = [
        BreathingPattern.from(technique: .deepBelly, cycles: 8),
        BreathingPattern.from(technique: .coherentBreathing, cycles: 10),
        BreathingPattern.from(technique: .fourSevenEight, cycles: 6),
        BreathingPattern.from(technique: .boxBreathing, cycles: 8)
    ]
}
