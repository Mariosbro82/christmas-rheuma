//
//  DailyCheckInViewModel.swift
//  InflamAI
//
//  ViewModel for BASDAI check-in flow
//

import Foundation
import CoreData
import Combine
import UIKit

@MainActor
class DailyCheckInViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var currentIndex = 0
    @Published var answers: [Double] = Array(repeating: 0.0, count: 12)  // 6 BASDAI + 6 ML
    @Published var showingResults = false
    @Published var showingError = false
    @Published var errorMessage = ""

    // Total questions: 6 BASDAI + 6 ML = 12
    static let totalQuestions = 12
    static let basadaiQuestionCount = 6

    // MARK: - ML Questions (indices 6-11)
    static let mlQuestions: [MLQuestion] = [
        MLQuestion(
            number: 7,
            text: "How would you rate your OVERALL FEELING right now?",
            subtitle: "Your general sense of wellbeing",
            minLabel: "Very poor",
            maxLabel: "Excellent"
        ),
        MLQuestion(
            number: 8,
            text: "How would you rate your ENERGY LEVEL today?",
            subtitle: "Physical and mental energy",
            minLabel: "No energy",
            maxLabel: "Full of energy"
        ),
        MLQuestion(
            number: 9,
            text: "How would you rate your STRESS LEVEL today?",
            subtitle: "Work, life, health related stress",
            minLabel: "No stress",
            maxLabel: "Extremely stressed"
        ),
        MLQuestion(
            number: 10,
            text: "How would you rate your ANXIETY LEVEL today?",
            subtitle: "Worry, nervousness, or unease",
            minLabel: "No anxiety",
            maxLabel: "Severe anxiety"
        ),
        MLQuestion(
            number: 11,
            text: "How has your PAIN been on average over the past 24 hours?",
            subtitle: "Overall pain, not just AS-specific",
            minLabel: "No pain",
            maxLabel: "Worst imaginable"
        ),
        MLQuestion(
            number: 12,
            text: "How would you rate your OVERALL HEALTH today?",
            subtitle: "Patient Global Assessment",
            minLabel: "Very good",
            maxLabel: "Very poor"
        )
    ]

    // MARK: - Computed Properties

    var progress: Double {
        Double(currentIndex + 1) / Double(Self.totalQuestions)
    }

    var isLastQuestion: Bool {
        currentIndex == Self.totalQuestions - 1
    }

    var isBASDAIQuestion: Bool {
        currentIndex < Self.basadaiQuestionCount
    }

    var currentQuestion: BASDAIQuestion {
        guard isBASDAIQuestion else {
            // Return a placeholder - the view will use currentMLQuestion instead
            return BASDAICalculator.questions[0]
        }
        return BASDAICalculator.questions[currentIndex]
    }

    var currentMLQuestion: MLQuestion? {
        guard !isBASDAIQuestion else { return nil }
        let mlIndex = currentIndex - Self.basadaiQuestionCount
        guard mlIndex < Self.mlQuestions.count else { return nil }
        return Self.mlQuestions[mlIndex]
    }

    var basdaiScore: Double {
        // Only use first 6 answers (BASDAI questions)
        let basdaiAnswers = Array(answers.prefix(Self.basadaiQuestionCount))
        return BASDAICalculator.calculate(answers: basdaiAnswers) ?? 0.0
    }

    // ML feature values extracted from answers array
    var overallFeeling: Double { answers[6] }
    var energyLevel: Double { answers[7] }
    var stressLevel: Double { answers[8] }
    var anxietyLevel: Double { answers[9] }
    var painAverage24h: Double { answers[10] }
    var patientGlobal: Double { answers[11] }

    var asdasScore: Double? {
        // ASDAS requires CRP value - would fetch from latest lab result
        // For now, return nil if no CRP available
        return nil
    }

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - Navigation

    func nextQuestion() {
        guard currentIndex < Self.totalQuestions - 1 else { return }
        currentIndex += 1

        // Light haptic feedback
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func previousQuestion() {
        guard currentIndex > 0 else { return }
        currentIndex -= 1

        // Light haptic feedback
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    // MARK: - Completion

    func completeCheckIn() {
        Task {
            do {
                // Extract BASDAI answers (first 6)
                let basdaiAnswers = Array(answers.prefix(Self.basadaiQuestionCount))

                // Validate BASDAI answers
                guard let score = BASDAICalculator.calculate(answers: basdaiAnswers) else {
                    errorMessage = "Invalid answers. Please check all questions."
                    showingError = true
                    return
                }

                // Create symptom log
                let log = SymptomLog(context: context)
                log.id = UUID()
                log.timestamp = Date()
                log.basdaiScore = score
                log.source = "daily_checkin"

                #if DEBUG
                let timestampStr = {
                    let df = DateFormatter()
                    df.dateFormat = "dd.MM.yyyy HH:mm:ss"
                    return df.string(from: log.timestamp ?? Date())
                }()
                print("💾 [DailyCheckIn] Creating SymptomLog: timestamp=\(timestampStr), basdaiScore=\(score), source=daily_checkin")
                #endif

                // Encode BASDAI answers as JSON (only first 6)
                if let answersData = try? JSONEncoder().encode(basdaiAnswers) {
                    log.basdaiAnswers = answersData
                }

                // Map individual BASDAI answers to specific fields
                log.fatigueLevel = Int16(answers[0])
                log.moodScore = Int16(10 - answers[0]) // Inverse of fatigue
                log.morningStiffnessMinutes = Int16(answers[5])

                // === SAVE ML FEATURES FROM USER INPUT (Questions 7-12) ===
                // These are now REAL user-provided values, not defaults!
                log.overallFeeling = Float(overallFeeling)           // Q7
                log.energyLevel = Float(energyLevel)                 // Q8
                log.stressLevel = Float(stressLevel)                 // Q9
                log.anxietyLevel = Float(anxietyLevel)               // Q10
                log.painAverage24h = Float(painAverage24h)           // Q11
                log.patientGlobal = Float(patientGlobal)             // Q12

                // Derived values from user input
                log.dayQuality = Float(overallFeeling)
                log.mentalFatigueLevel = Float(answers[0]) // From fatigue question

                // Activity limitation derived from pain and fatigue
                log.activityLimitationScore = Float((painAverage24h + answers[0]) / 2.0)

                // Social engagement derived from energy and overall feeling
                log.socialEngagement = Float((energyLevel + overallFeeling) / 2.0)

                // Cognitive function - inverse of fatigue/stress combo
                log.cognitiveFunction = Float(10.0 - ((answers[0] + stressLevel) / 2.0))

                // Coping ability - inverse of stress/anxiety
                log.copingAbility = Float(10.0 - ((stressLevel + anxietyLevel) / 2.0))

                // NOTE: mentalWellbeing and depressionRisk still require validated assessment
                // log.mentalWellbeing = 0  // Requires validated assessment (WEMWBS)
                // log.depressionRisk = 0   // Requires PHQ-9 or BDI screening

                print("✅ ML Features saved from check-in:")
                print("   - overallFeeling: \(overallFeeling)")
                print("   - energyLevel: \(energyLevel)")
                print("   - stressLevel: \(stressLevel)")
                print("   - anxietyLevel: \(anxietyLevel)")
                print("   - painAverage24h: \(painAverage24h)")
                print("   - patientGlobal: \(patientGlobal)")

                // Fetch context data (weather, health)
                await attachContextData(to: log)
                
                // Auto-populate remaining ML properties from available data
                log.populateMLProperties(context: context)

                // Save
                try context.save()

                #if DEBUG
                print("✅ [DailyCheckIn] SymptomLog SAVED successfully to Core Data")
                #endif

                // Success haptic
                UINotificationFeedbackGenerator().notificationOccurred(.success)

                // === ML Integration (Phase 3) ===
                // Determine if this is a flare signal (high BASDAI)
                let isFlareSignal = score >= 6.0

                if isFlareSignal {
                    print("🔥 [CheckIn] High BASDAI score (\(String(format: "%.1f", score))) - recording as flare signal for ML training")
                }

                // Record training sample for ML personalization
                MLIntegrationService.shared.recordTrainingSample(basdaiScore: score, isHighRisk: isFlareSignal)

                // Record score to Neural Engine for prediction updates
                UnifiedNeuralEngine.shared.recordScore(score)

                // Auto-validate previous predictions if this is a flare
                if isFlareSignal {
                    MLIntegrationService.shared.autoValidatePredictions(flareOccurred: true)
                }

                // Show results
                showingResults = true

            } catch {
                errorMessage = "Failed to save check-in: \(error.localizedDescription)"
                showingError = true
            }
        }
    }

    // MARK: - Context Data

    private func attachContextData(to log: SymptomLog) async {
        // Create context snapshot
        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = Date()

        let today = Date()

        // ===== FETCH REAL WEATHER DATA (Open-Meteo - FREE, no API key) =====
        // Uses fallback cache if API fails (up to 24h old data)
        if let weather = await OpenMeteoService.shared.fetchCurrentWeatherWithFallback() {
            snapshot.barometricPressure = weather.pressure
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.pressureChange12h = weather.pressureChange12h
            print("✅ Weather data attached: \(weather.pressure) mmHg, \(weather.humidity)%, \(weather.temperature)°C")
        } else {
            // Neither API nor fallback cache available
            print("⚠️ Weather unavailable: No API response or cached data")
            // Values stay at Core Data defaults (0) - UI should show "unavailable"
        }

        // ===== FETCH COMPREHENSIVE HEALTHKIT DATA =====
        // CRITICAL FIX: Fetch ALL available biometric data using the comprehensive method
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: today)

            // Store in ContextSnapshot
            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency

            // Store sleep data in SymptomLog
            log.sleepDurationHours = biometrics.sleep.durationHours
            log.sleepQuality = Int16(biometrics.sleep.quality)

            // Store exercise data in SymptomLog (using existing field)
            log.exerciseMinutesToday = Int16(biometrics.exerciseMinutes)

            print("✅ COMPREHENSIVE HealthKit data attached:")
            print("   - Heart: RHR=\(biometrics.restingHeartRate)bpm, HRV=\(String(format: "%.1f", biometrics.hrvValue))ms")
            print("   - Activity: Steps=\(biometrics.stepCount), Exercise=\(biometrics.exerciseMinutes)min, Distance=\(String(format: "%.2f", biometrics.distanceKm))km")
            print("   - Sleep: \(String(format: "%.1f", biometrics.sleep.durationHours))h, Efficiency=\(String(format: "%.1f", biometrics.sleep.efficiency))%")
            print("   - Mobility: Speed=\(String(format: "%.2f", biometrics.walkingSpeedMps))m/s, StepLength=\(String(format: "%.1f", biometrics.walkingStepLengthCm))cm")

            // Mark that we have valid HealthKit data
            UserDefaults.standard.set(true, forKey: "hasValidHealthKitData")
            UserDefaults.standard.set(Date(), forKey: "lastHealthKitFetch")
        } catch {
            // NO FAKE DATA - leave as 0 to indicate unavailable
            print("⚠️ HealthKit fetch failed: \(error.localizedDescription) - no health data available")
            // Values stay at Core Data defaults (0) - UI should show "unavailable"
            UserDefaults.standard.set(false, forKey: "hasValidHealthKitData")
        }

        log.contextSnapshot = snapshot
    }
    
    // REMOVED: calculateDepressionRisk - Depression screening requires validated assessments (PHQ-9)
    // Fabricating depression scores from mood/fatigue is medically inappropriate
}

// MARK: - ML Question Model

/// Question model for ML feature collection (similar to BASDAIQuestion)
struct MLQuestion {
    let number: Int
    let text: String
    let subtitle: String
    let minLabel: String
    let maxLabel: String

    init(number: Int, text: String, subtitle: String, minLabel: String, maxLabel: String) {
        self.number = number
        self.text = text
        self.subtitle = subtitle
        self.minLabel = minLabel
        self.maxLabel = maxLabel
    }
}
