//
//  MeditationAssets.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-09.
//

import SwiftUI

/// Maps meditation categories, types, and states to custom dino character assets
enum MeditationAssets {

    // MARK: - Category Images

    /// Get the dino character image for a meditation category
    static func image(for category: MeditationCategory) -> String {
        switch category {
        case .painManagement:
            return "dino-hearth 1" // Heart for pain/health management
        case .stressReduction:
            return "dino-meditating" // Meditating dino for stress
        case .sleepImprovement:
            return "dino-sleeping 1" // Sleeping dino
        case .anxietyRelief:
            return "dino-sweat-little-scared" // Anxious/scared dino
        case .focusConcentration:
            return "dino-showing-whiteboard" // Teaching/focused dino
        case .emotionalWellbeing:
            return "dino-happy 1" // Happy dino
        case .bodyAwareness:
            return "dino-spine-showing" // Dino showing spine/body
        case .breathwork:
            return "dino-meditating" // Meditating dino (breathing focus)
        case .movementMeditation:
            return "dino-walking 1" // Walking dino
        case .visualization:
            return "dino-showing-whiteboard" // Dino with whiteboard (visualization)
        case .mindfulness:
            return "dino-meditating" // Classic meditation pose
        case .compassion:
            return "dino-hearth 1" // Heart for compassion
        case .gratitude:
            return "dino-happy 1" // Happy/grateful dino
        case .energyBoost:
            return "dino-walking-fast" // Energetic dino
        case .recovery:
            return "dino-tired 1" // Resting/recovering dino
        }
    }

    // MARK: - Mood State Images

    /// Get dino image for mood/stress level (0-10 scale)
    static func image(forMoodLevel level: Int) -> String {
        switch level {
        case 0...2:
            return "dino-happy 1" // Very happy
        case 3...4:
            return "dino-stading-normal" // Casual/neutral
        case 5...6:
            return "dino-sad" // Slightly sad
        case 7...8:
            return "dino-tired 1" // Tired/stressed
        case 9...10:
            return "dino-sweat-little-scared" // Very stressed/anxious
        default:
            return "dino-stading-normal"
        }
    }

    /// Get dino image for pain level (0-10 scale)
    static func image(forPainLevel level: Int) -> String {
        switch level {
        case 0...2:
            return "dino-happy 1" // Little to no pain
        case 3...4:
            return "dino-stading-normal" // Mild pain
        case 5...6:
            return "dino-sad" // Moderate pain
        case 7...8:
            return "dino-tired 1" // Significant pain
        case 9...10:
            return "dino-sweat-little-scared" // Severe pain
        default:
            return "dino-stading-normal"
        }
    }

    // MARK: - Session State Images

    /// Get dino image for session player states
    static var meditatingImage: String { "dino-meditating" }
    static var completedImage: String { "dino-happy 1" }
    static var startingImage: String { "dino-stading-normal" }
    static var pausedImage: String { "dino-stop-hand" }

    // MARK: - Progress & Achievement Images

    static var streakActiveImage: String { "dino-strong-mussel" } // Strong/consistent
    static var streakBrokenImage: String { "dino-sad" }
    static var achievementImage: String { "dino-happy 1" }
    static var progressImage: String { "dino-showing-whiteboard" }

    // MARK: - Time of Day Images

    static func image(forTimeOfDay time: TimeOfDay) -> String {
        switch time {
        case .earlyMorning, .morning:
            return "dino-tired 1" // Waking up
        case .midday, .afternoon:
            return "dino-stading-normal" // Active day
        case .evening:
            return "dino-showing-whiteboard" // Winding down
        case .night:
            return "dino-sleeping 1" // Nighttime
        case .anytime:
            return "dino-meditating" // Default meditation
        }
    }

    // MARK: - Symptom-Specific Images

    static func image(forSymptom symptom: TargetSymptom) -> String {
        switch symptom {
        case .spinalPain, .lowerBackPain:
            return "dino-spine-showing"
        case .jointPain, .hipPain, .neckPain:
            return "dino-hearth 1"
        case .morningStiffness, .muscleTension:
            return "dino-tired 1"
        case .inflammation, .flare:
            return "dino-sweat-little-scared"
        case .insomnia:
            return "dino-sleeping 1"
        case .anxiety:
            return "dino-sweat-little-scared"
        case .depression, .lowMood:
            return "dino-sad"
        case .stress, .irritability:
            return "dino-sweat-little-scared"
        case .brainfog:
            return "dino-sad"
        case .fatigue:
            return "dino-tired 1"
        case .chronicPain:
            return "dino-hearth 1"
        case .headaches:
            return "dino-sad"
        }
    }

    // MARK: - Breathing Pattern Images

    static func image(forBreathingTechnique technique: BreathingTechnique) -> String {
        switch technique {
        case .fourSevenEight, .boxBreathing, .triangleBreathing,
             .coherentBreathing, .resonantBreathing, .deepBelly:
            return "dino-meditating" // Calm breathing
        case .bellowsBreath:
            return "dino-walking-fast" // Energetic breathing
        case .alternateNostril, .pursedLip:
            return "dino-stading-normal" // Focused breathing
        case .custom:
            return "dino-meditating"
        }
    }

    // MARK: - General App Images

    static var lockScreenImage: String { "dino-lock" }
    static var medicationImage: String { "dino-medications" }
    static var doctorImage: String { "dino-stading-normal" }
    static var exerciseImage: String { "dino-walking-fast" }
    static var bodyMapImage: String { "dino-spine-showing" }

    // MARK: - Helper Function

    /// Create an Image view from asset name
    static func imageView(_ assetName: String) -> some View {
        Image(assetName)
            .resizable()
            .aspectRatio(contentMode: .fit)
    }
}

// MARK: - View Extension for Easy Access

extension View {
    /// Apply meditation asset image with consistent styling
    func meditationAsset(_ assetName: String, size: CGFloat = 60) -> some View {
        Image(assetName)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(width: size, height: size)
    }
}
