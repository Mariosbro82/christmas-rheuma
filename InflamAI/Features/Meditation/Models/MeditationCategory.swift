//
//  MeditationCategory.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import Foundation

// MARK: - Meditation Categories

enum MeditationCategory: String, CaseIterable, Codable {
    case painManagement = "painManagement"
    case stressReduction = "stressReduction"
    case sleepImprovement = "sleepImprovement"
    case anxietyRelief = "anxietyRelief"
    case focusConcentration = "focusConcentration"
    case emotionalWellbeing = "emotionalWellbeing"
    case bodyAwareness = "bodyAwareness"
    case breathwork = "breathwork"
    case movementMeditation = "movementMeditation"
    case visualization = "visualization"
    case mindfulness = "mindfulness"
    case compassion = "compassion"
    case gratitude = "gratitude"
    case energyBoost = "energyBoost"
    case recovery = "recovery"

    var displayName: String {
        switch self {
        case .painManagement: return "Pain Management"
        case .stressReduction: return "Stress Reduction"
        case .sleepImprovement: return "Sleep Improvement"
        case .anxietyRelief: return "Anxiety Relief"
        case .focusConcentration: return "Focus & Concentration"
        case .emotionalWellbeing: return "Emotional Wellbeing"
        case .bodyAwareness: return "Body Awareness"
        case .breathwork: return "Breathwork"
        case .movementMeditation: return "Movement Meditation"
        case .visualization: return "Visualization"
        case .mindfulness: return "Mindfulness"
        case .compassion: return "Compassion"
        case .gratitude: return "Gratitude"
        case .energyBoost: return "Energy Boost"
        case .recovery: return "Recovery"
        }
    }

    /// Custom dino character image for this category
    var dinoImage: String {
        return MeditationAssets.image(for: self)
    }

    /// Fallback SF Symbol icon (for backwards compatibility)
    var icon: String {
        switch self {
        case .painManagement: return "heart.circle"
        case .stressReduction: return "leaf.circle"
        case .sleepImprovement: return "moon.circle"
        case .anxietyRelief: return "wind.circle"
        case .focusConcentration: return "target"
        case .emotionalWellbeing: return "heart.text.square"
        case .bodyAwareness: return "figure.mind.and.body"
        case .breathwork: return "lungs"
        case .movementMeditation: return "figure.walk"
        case .visualization: return "eye.circle"
        case .mindfulness: return "brain.head.profile"
        case .compassion: return "hands.sparkles"
        case .gratitude: return "star.circle"
        case .energyBoost: return "bolt.circle"
        case .recovery: return "bandage"
        }
    }
}

// MARK: - Meditation Types

enum MeditationType: String, CaseIterable, Codable {
    case guided = "guided"
    case unguided = "unguided"
    case timer = "timer"
    case breathingExercise = "breathingExercise"
    case bodyScan = "bodyScan"
    case walkingMeditation = "walkingMeditation"
    case lovingKindness = "lovingKindness"
    case visualization = "visualization"
    case mantra = "mantra"
    case soundBath = "soundBath"
    case progressiveMuscleRelaxation = "progressiveMuscleRelaxation"
    case mindfulEating = "mindfulEating"
    case natureSounds = "natureSounds"
    case binaural = "binaural"
    case custom = "custom"

    var displayName: String {
        switch self {
        case .guided: return "Guided Meditation"
        case .unguided: return "Unguided Meditation"
        case .timer: return "Meditation Timer"
        case .breathingExercise: return "Breathing Exercise"
        case .bodyScan: return "Body Scan"
        case .walkingMeditation: return "Walking Meditation"
        case .lovingKindness: return "Loving Kindness"
        case .visualization: return "Visualization"
        case .mantra: return "Mantra Meditation"
        case .soundBath: return "Sound Bath"
        case .progressiveMuscleRelaxation: return "Progressive Muscle Relaxation"
        case .mindfulEating: return "Mindful Eating"
        case .natureSounds: return "Nature Sounds"
        case .binaural: return "Binaural Beats"
        case .custom: return "Custom Session"
        }
    }
}

// MARK: - Difficulty Levels

enum DifficultyLevel: String, CaseIterable, Codable {
    case beginner = "beginner"
    case intermediate = "intermediate"
    case advanced = "advanced"

    var displayName: String {
        switch self {
        case .beginner: return "Beginner"
        case .intermediate: return "Intermediate"
        case .advanced: return "Advanced"
        }
    }
}

// MARK: - Target Symptoms (AS-Specific)

enum TargetSymptom: String, CaseIterable, Codable {
    // AS-Specific Symptoms
    case spinalPain = "spinalPain"
    case jointPain = "jointPain"
    case morningStiffness = "morningStiffness"
    case inflammation = "inflammation"
    case flare = "flare"
    case neckPain = "neckPain"
    case lowerBackPain = "lowerBackPain"
    case hipPain = "hipPain"

    // General Symptoms
    case muscleTension = "muscleTension"
    case fatigue = "fatigue"
    case insomnia = "insomnia"
    case anxiety = "anxiety"
    case depression = "depression"
    case stress = "stress"
    case brainfog = "brainfog"
    case irritability = "irritability"
    case lowMood = "lowMood"
    case chronicPain = "chronicPain"
    case headaches = "headaches"

    var displayName: String {
        switch self {
        case .spinalPain: return "Spinal Pain"
        case .jointPain: return "Joint Pain"
        case .morningStiffness: return "Morning Stiffness"
        case .inflammation: return "Inflammation"
        case .flare: return "Flare Episode"
        case .neckPain: return "Neck Pain"
        case .lowerBackPain: return "Lower Back Pain"
        case .hipPain: return "Hip Pain"
        case .muscleTension: return "Muscle Tension"
        case .fatigue: return "Fatigue"
        case .insomnia: return "Insomnia"
        case .anxiety: return "Anxiety"
        case .depression: return "Depression"
        case .stress: return "Stress"
        case .brainfog: return "Brain Fog"
        case .irritability: return "Irritability"
        case .lowMood: return "Low Mood"
        case .chronicPain: return "Chronic Pain"
        case .headaches: return "Headaches"
        }
    }
}

// MARK: - Time of Day

enum TimeOfDay: String, CaseIterable, Codable {
    case earlyMorning = "earlyMorning"
    case morning = "morning"
    case midday = "midday"
    case afternoon = "afternoon"
    case evening = "evening"
    case night = "night"
    case anytime = "anytime"

    var displayName: String {
        switch self {
        case .earlyMorning: return "Early Morning (5-7 AM)"
        case .morning: return "Morning (7-11 AM)"
        case .midday: return "Midday (11 AM-2 PM)"
        case .afternoon: return "Afternoon (2-6 PM)"
        case .evening: return "Evening (6-9 PM)"
        case .night: return "Night (9 PM-12 AM)"
        case .anytime: return "Anytime"
        }
    }

    /// Get current time of day
    static var current: TimeOfDay {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 5..<7: return .earlyMorning
        case 7..<11: return .morning
        case 11..<14: return .midday
        case 14..<18: return .afternoon
        case 18..<21: return .evening
        case 21..<24, 0..<5: return .night
        default: return .anytime
        }
    }
}
