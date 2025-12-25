//
//  FlareTriggers.swift
//  InflamAI
//
//  Comprehensive list of potential AS flare triggers
//

import Foundation
import SwiftUI

/// Common triggers for Ankylosing Spondylitis flares
enum FlareTrigger: String, CaseIterable, Codable, Identifiable {
    var id: String { rawValue }

    // MARK: - Physical Activity
    case overexertion = "overexertion"
    case prolongedSitting = "prolonged_sitting"
    case prolongedStanding = "prolonged_standing"
    case heavyLifting = "heavy_lifting"
    case lackOfExercise = "lack_of_exercise"
    case newExerciseRoutine = "new_exercise_routine"
    case poorPosture = "poor_posture"
    case repetitiveMotion = "repetitive_motion"

    // MARK: - Sleep & Rest
    case poorSleep = "poor_sleep"
    case sleepDeprivation = "sleep_deprivation"
    case irregularSleepSchedule = "irregular_sleep_schedule"
    case uncomfortableMattress = "uncomfortable_mattress"

    // MARK: - Stress & Mental Health
    case emotionalStress = "emotional_stress"
    case workStress = "work_stress"
    case anxiety = "anxiety"
    case depression = "depression"
    case majorLifeEvent = "major_life_event"

    // MARK: - Diet & Nutrition
    case missedMeal = "missed_meal"
    case highSugarIntake = "high_sugar_intake"
    case processedFoods = "processed_foods"
    case alcohol = "alcohol"
    case caffeine = "caffeine"
    case dairyProducts = "dairy_products"
    case glutenFoods = "gluten_foods"
    case nightshadeFoods = "nightshade_foods"
    case dehydration = "dehydration"

    // MARK: - Medication
    case missedMedication = "missed_medication"
    case medicationChange = "medication_change"
    case nsaidReduction = "nsaid_reduction"

    // MARK: - Environmental
    case coldWeather = "cold_weather"
    case hotWeather = "hot_weather"
    case humidWeather = "humid_weather"
    case barometricPressureChange = "barometric_pressure_change"
    case airQuality = "air_quality"
    case seasonalChange = "seasonal_change"

    // MARK: - Infections & Illness
    case coldFlu = "cold_flu"
    case infection = "infection"
    case gutIssues = "gut_issues"
    case immuneSystemFlare = "immune_system_flare"

    // MARK: - Lifestyle
    case smoking = "smoking"
    case travel = "travel"
    case jetLag = "jet_lag"
    case allergyFlareUp = "allergy_flare_up"

    // MARK: - Hormonal
    case hormonalChanges = "hormonal_changes"
    case menstrualCycle = "menstrual_cycle"

    // MARK: - Other
    case unknown = "unknown"
    case other = "other"

    // MARK: - Display Information

    var displayName: String {
        switch self {
        // Physical Activity
        case .overexertion: return "Overexertion"
        case .prolongedSitting: return "Prolonged Sitting"
        case .prolongedStanding: return "Prolonged Standing"
        case .heavyLifting: return "Heavy Lifting"
        case .lackOfExercise: return "Lack of Exercise"
        case .newExerciseRoutine: return "New Exercise Routine"
        case .poorPosture: return "Poor Posture"
        case .repetitiveMotion: return "Repetitive Motion"

        // Sleep & Rest
        case .poorSleep: return "Poor Sleep Quality"
        case .sleepDeprivation: return "Sleep Deprivation"
        case .irregularSleepSchedule: return "Irregular Sleep Schedule"
        case .uncomfortableMattress: return "Uncomfortable Mattress"

        // Stress & Mental Health
        case .emotionalStress: return "Emotional Stress"
        case .workStress: return "Work Stress"
        case .anxiety: return "Anxiety"
        case .depression: return "Depression"
        case .majorLifeEvent: return "Major Life Event"

        // Diet & Nutrition
        case .missedMeal: return "Missed Meal"
        case .highSugarIntake: return "High Sugar Intake"
        case .processedFoods: return "Processed Foods"
        case .alcohol: return "Alcohol"
        case .caffeine: return "Excessive Caffeine"
        case .dairyProducts: return "Dairy Products"
        case .glutenFoods: return "Gluten Foods"
        case .nightshadeFoods: return "Nightshade Foods"
        case .dehydration: return "Dehydration"

        // Medication
        case .missedMedication: return "Missed Medication"
        case .medicationChange: return "Medication Change"
        case .nsaidReduction: return "NSAID Reduction"

        // Environmental
        case .coldWeather: return "Cold Weather"
        case .hotWeather: return "Hot Weather"
        case .humidWeather: return "Humid Weather"
        case .barometricPressureChange: return "Barometric Pressure Change"
        case .airQuality: return "Poor Air Quality"
        case .seasonalChange: return "Seasonal Change"

        // Infections & Illness
        case .coldFlu: return "Cold/Flu"
        case .infection: return "Infection"
        case .gutIssues: return "Gut Issues"
        case .immuneSystemFlare: return "Immune System Flare"

        // Lifestyle
        case .smoking: return "Smoking"
        case .travel: return "Travel"
        case .jetLag: return "Jet Lag"
        case .allergyFlareUp: return "Allergy Flare-up"

        // Hormonal
        case .hormonalChanges: return "Hormonal Changes"
        case .menstrualCycle: return "Menstrual Cycle"

        // Other
        case .unknown: return "Unknown"
        case .other: return "Other"
        }
    }

    var icon: String {
        switch self {
        // Physical Activity
        case .overexertion, .heavyLifting: return "figure.strengthtraining.traditional"
        case .prolongedSitting: return "figure.seated.side"
        case .prolongedStanding: return "figure.stand"
        case .lackOfExercise: return "figure.walk.circle"
        case .newExerciseRoutine: return "figure.run"
        case .poorPosture: return "figure.stand.line.dotted.figure.stand"
        case .repetitiveMotion: return "arrow.clockwise"

        // Sleep & Rest
        case .poorSleep, .sleepDeprivation, .irregularSleepSchedule: return "bed.double.fill"
        case .uncomfortableMattress: return "bed.double"

        // Stress & Mental Health
        case .emotionalStress, .workStress: return "brain.head.profile"
        case .anxiety: return "wind"
        case .depression: return "cloud.rain.fill"
        case .majorLifeEvent: return "exclamationmark.triangle.fill"

        // Diet & Nutrition
        case .missedMeal: return "fork.knife.circle"
        case .highSugarIntake: return "birthday.cake.fill"
        case .processedFoods: return "takeoutbag.and.cup.and.straw.fill"
        case .alcohol: return "wineglass.fill"
        case .caffeine: return "cup.and.saucer.fill"
        case .dairyProducts: return "carton.fill"
        case .glutenFoods, .nightshadeFoods: return "leaf.fill"
        case .dehydration: return "drop.fill"

        // Medication
        case .missedMedication, .medicationChange, .nsaidReduction: return "pills.fill"

        // Environmental
        case .coldWeather: return "snowflake"
        case .hotWeather: return "sun.max.fill"
        case .humidWeather: return "humidity.fill"
        case .barometricPressureChange: return "gauge.with.dots.needle.bottom.50percent"
        case .airQuality: return "aqi.medium"
        case .seasonalChange: return "leaf.circle.fill"

        // Infections & Illness
        case .coldFlu: return "thermometer.medium"
        case .infection: return "bandage.fill"
        case .gutIssues: return "cross.case.fill"
        case .immuneSystemFlare: return "shield.lefthalf.filled"

        // Lifestyle
        case .smoking: return "smoke.fill"
        case .travel: return "airplane"
        case .jetLag: return "clock.badge.exclamationmark.fill"
        case .allergyFlareUp: return "allergens"

        // Hormonal
        case .hormonalChanges, .menstrualCycle: return "waveform.path.ecg"

        // Other
        case .unknown: return "questionmark.circle.fill"
        case .other: return "ellipsis.circle.fill"
        }
    }

    var category: FlareTriggerCategory {
        switch self {
        case .overexertion, .prolongedSitting, .prolongedStanding, .heavyLifting,
             .lackOfExercise, .newExerciseRoutine, .poorPosture, .repetitiveMotion:
            return .physicalActivity

        case .poorSleep, .sleepDeprivation, .irregularSleepSchedule, .uncomfortableMattress:
            return .sleep

        case .emotionalStress, .workStress, .anxiety, .depression, .majorLifeEvent:
            return .stress

        case .missedMeal, .highSugarIntake, .processedFoods, .alcohol, .caffeine,
             .dairyProducts, .glutenFoods, .nightshadeFoods, .dehydration:
            return .diet

        case .missedMedication, .medicationChange, .nsaidReduction:
            return .medication

        case .coldWeather, .hotWeather, .humidWeather, .barometricPressureChange,
             .airQuality, .seasonalChange:
            return .environmental

        case .coldFlu, .infection, .gutIssues, .immuneSystemFlare:
            return .illness

        case .smoking, .travel, .jetLag, .allergyFlareUp:
            return .lifestyle

        case .hormonalChanges, .menstrualCycle:
            return .hormonal

        case .unknown, .other:
            return .other
        }
    }
}

/// Categories for organizing triggers
enum FlareTriggerCategory: String, CaseIterable, Identifiable {
    case physicalActivity = "Physical Activity"
    case sleep = "Sleep & Rest"
    case stress = "Stress & Mental Health"
    case diet = "Diet & Nutrition"
    case medication = "Medication"
    case environmental = "Environmental"
    case illness = "Infections & Illness"
    case lifestyle = "Lifestyle"
    case hormonal = "Hormonal"
    case other = "Other"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .physicalActivity: return "figure.walk"
        case .sleep: return "bed.double.fill"
        case .stress: return "brain.head.profile"
        case .diet: return "fork.knife"
        case .medication: return "pills.fill"
        case .environmental: return "cloud.sun.fill"
        case .illness: return "cross.case.fill"
        case .lifestyle: return "person.fill"
        case .hormonal: return "waveform.path.ecg"
        case .other: return "ellipsis.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .physicalActivity: return .blue
        case .sleep: return .indigo
        case .stress: return .orange
        case .diet: return .green
        case .medication: return .red
        case .environmental: return .cyan
        case .illness: return .purple
        case .lifestyle: return .pink
        case .hormonal: return .mint
        case .other: return .gray
        }
    }

    var triggers: [FlareTrigger] {
        FlareTrigger.allCases.filter { $0.category == self }
    }
}
