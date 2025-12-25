//
//  MorningCheckInViewModel.swift
//  InflamAI
//
//  ViewModel for Quick Morning Check-in
//  Handles data collection and Core Data persistence
//
//  ML Features collected:
//  - pain_current (index 18)
//  - nocturnal_pain (index 21)
//  - morning_stiffness_duration (index 22)
//  - morning_stiffness_severity (index 23)
//  - pain_burning (index 25)
//  - pain_aching (index 26)
//  - pain_sharp (index 27)
//  - pain_interference_sleep (index 28)
//  - pain_interference_activity (index 29)
//  - mood_current (index 64)
//  - stress_level (index 68)
//  - universal_assessment (index 88)
//

import Foundation
import CoreData
import Combine
import UIKit

@MainActor
class MorningCheckInViewModel: ObservableObject {

    // MARK: - Published Properties (User Inputs)

    /// Universal assessment: "How do you feel overall?" (0-10)
    /// Maps to ML feature index 88
    @Published var overallFeeling: Double = 5.0

    /// Current pain level (0-10)
    /// Maps to ML feature index 18
    @Published var painCurrent: Double = 0.0

    /// Morning stiffness duration in minutes (0-180)
    /// Maps to ML feature index 22
    @Published var stiffnessDuration: Double = 0.0

    /// Morning stiffness severity (0-10)
    /// Maps to ML feature index 23
    @Published var stiffnessSeverity: Double = 0.0

    /// Current mood (0-10)
    /// Maps to ML feature index 64
    @Published var moodCurrent: Double = 5.0

    /// Stress level (0-10)
    /// Maps to ML feature index 68
    @Published var stressLevel: Double = 0.0

    // MARK: - Pain Characteristics (Phase 2 additions)

    /// Nocturnal pain - did you have pain at night?
    /// Maps to ML feature index 21
    @Published var nocturnalPain: Bool = false

    /// Pain type: burning sensation
    /// Maps to ML feature index 25
    @Published var painBurning: Bool = false

    /// Pain type: aching/dull pain
    /// Maps to ML feature index 26
    @Published var painAching: Bool = false

    /// Pain type: sharp/stabbing pain
    /// Maps to ML feature index 27
    @Published var painSharp: Bool = false

    /// How much does pain interfere with sleep? (0-10)
    /// Maps to ML feature index 28
    @Published var painInterferenceSleep: Double = 0.0

    /// How much does pain interfere with daily activities? (0-10)
    /// Maps to ML feature index 29
    @Published var painInterferenceActivity: Double = 0.0

    /// Breakthrough pain - sudden severe pain despite treatment
    /// Maps to ML feature index 31
    @Published var breakthroughPain: Bool = false

    // MARK: - UI State

    @Published var isSaving = false
    @Published var showingSaveConfirmation = false
    @Published var showingError = false
    @Published var errorMessage = ""

    // MARK: - Computed Properties

    var overallFeelingEmoji: String {
        switch overallFeeling {
        case 0..<2: return "😫"
        case 2..<4: return "😔"
        case 4..<6: return "😐"
        case 6..<8: return "🙂"
        case 8..<10: return "😊"
        default: return "🌟"
        }
    }

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - Save Check-In

    func saveCheckIn() async {
        isSaving = true
        defer { isSaving = false }

        do {
            // Create new SymptomLog
            let log = SymptomLog(context: context)
            log.id = UUID()
            log.timestamp = Date()
            log.source = "morning_checkin"

            // === Map user inputs to Core Data fields ===

            // Universal assessment (index 88)
            log.overallFeeling = Float(overallFeeling)

            // Pain current (index 18)
            // Note: Using painAverage24h as proxy since we don't have a dedicated "painCurrent" field
            log.painAverage24h = Float(painCurrent)

            // Nocturnal pain (index 21) - convert Bool to Float (0-10 scale)
            log.nocturnalPain = nocturnalPain ? 10.0 : 0.0

            // Pain types (indices 25-27) - convert Bool to Float (0-10 scale)
            log.painBurning = painBurning ? 10.0 : 0.0
            log.painAching = painAching ? 10.0 : 0.0
            log.painSharp = painSharp ? 10.0 : 0.0

            // Pain interference (indices 28-29)
            log.painInterferenceSleep = Float(painInterferenceSleep)
            log.painInterferenceActivity = Float(painInterferenceActivity)

            // Breakthrough pain (index 31) - sudden severe pain despite treatment
            log.breakthroughPainCount = breakthroughPain ? 1 : 0

            // Morning stiffness duration (index 22)
            log.morningStiffnessMinutes = Int16(stiffnessDuration)

            // Morning stiffness severity (index 23)
            log.morningStiffnessSeverity = Float(stiffnessSeverity)

            // Mood current (index 64)
            log.moodScore = Int16(moodCurrent)

            // Stress level (index 68)
            log.stressLevel = Float(stressLevel)

            // === Derived values ===

            // Energy level - inverse correlation with fatigue/pain/stiffness
            let energyEstimate = max(0, 10 - (painCurrent * 0.3) - (stiffnessSeverity * 0.2) - (stressLevel * 0.2))
            log.energyLevel = Float(energyEstimate)

            // Patient global - derived from overall feeling (inverted: 10=feeling great = 0 disease activity)
            log.patientGlobal = Float(10 - overallFeeling)

            // Mood valence - convert 0-10 mood to -10 to +10 scale
            log.moodValence = Float((moodCurrent - 5.0) * 2.0)

            // Day quality - same as overall feeling
            log.dayQuality = Float(overallFeeling)

            // Anxiety estimate - correlated with stress (rough estimate, user should report explicitly)
            log.anxietyLevel = Float(stressLevel * 0.7)

            // === Attach context data (weather + HealthKit) ===
            await attachContextData(to: log)

            // === Auto-populate remaining ML properties ===
            log.populateMLProperties(context: context)

            // === Save to Core Data ===
            try context.save()

            print("✅ Morning check-in saved successfully")
            print("   - Overall feeling: \(overallFeeling)")
            print("   - Pain: \(painCurrent), nocturnal: \(nocturnalPain), breakthrough: \(breakthroughPain)")
            print("   - Pain types: burning=\(painBurning), aching=\(painAching), sharp=\(painSharp)")
            print("   - Pain interference: sleep=\(painInterferenceSleep), activity=\(painInterferenceActivity)")
            print("   - Stiffness: \(stiffnessDuration) min, severity \(stiffnessSeverity)")
            print("   - Mood: \(moodCurrent)")
            print("   - Stress: \(stressLevel)")

            // === ML Integration ===

            // Record to Neural Engine for prediction updates
            // Using pain + stiffness + stress as composite indicator
            let compositeScore = (painCurrent + stiffnessSeverity + stressLevel) / 3.0
            UnifiedNeuralEngine.shared.recordScore(compositeScore)

            // Check if this might indicate a flare (high pain + high stiffness)
            let isFlareSignal = painCurrent >= 7.0 || (stiffnessDuration >= 60 && stiffnessSeverity >= 6.0)
            if isFlareSignal {
                print("⚠️ [MorningCheckIn] Potential flare signal detected")
                MLIntegrationService.shared.recordTrainingSample(basdaiScore: compositeScore, isHighRisk: true)
                MLIntegrationService.shared.autoValidatePredictions(flareOccurred: true)
            }

            // Success feedback
            UINotificationFeedbackGenerator().notificationOccurred(.success)

            showingSaveConfirmation = true

        } catch {
            print("❌ Failed to save morning check-in: \(error)")
            errorMessage = "Failed to save: \(error.localizedDescription)"
            showingError = true
            UINotificationFeedbackGenerator().notificationOccurred(.error)
        }
    }

    // MARK: - Context Data

    private func attachContextData(to log: SymptomLog) async {
        // Create context snapshot for weather + biometrics
        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = Date()

        let today = Date()

        // === Fetch weather data ===
        do {
            let weather = try await OpenMeteoService.shared.fetchCurrentWeather()
            snapshot.barometricPressure = weather.pressure
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.pressureChange12h = weather.pressureChange12h
            print("✅ Weather attached: \(weather.pressure) mmHg, \(weather.humidity)%, \(weather.temperature)°C")
        } catch {
            print("⚠️ Weather fetch failed: \(error.localizedDescription)")
            // Leave as 0 - indicates "no weather data"
        }

        // === Fetch HealthKit data ===
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: today)

            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency

            log.sleepDurationHours = biometrics.sleep.durationHours
            log.sleepQuality = Int16(biometrics.sleep.quality)
            log.exerciseMinutesToday = Int16(biometrics.exerciseMinutes)

            print("✅ HealthKit attached: HR=\(biometrics.restingHeartRate), HRV=\(String(format: "%.1f", biometrics.hrvValue))ms")
            print("   Sleep: \(String(format: "%.1f", biometrics.sleep.durationHours))h, Steps: \(biometrics.stepCount)")

        } catch {
            print("⚠️ HealthKit fetch failed: \(error.localizedDescription)")
            // Leave as 0 - indicates "no HealthKit data"
        }

        log.contextSnapshot = snapshot
    }
}
