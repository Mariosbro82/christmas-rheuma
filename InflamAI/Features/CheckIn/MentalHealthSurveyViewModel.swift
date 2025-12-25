//
//  MentalHealthSurveyViewModel.swift
//  InflamAI
//
//  ViewModel for Mental Health Assessment
//  Includes Cognitive Function, Emotional Regulation, and PHQ-2
//
//  ML Features:
//  - cognitive_function (index 71) - 3 questions
//  - emotional_regulation (index 72) - 3 questions
//  - depression_risk (index 75) - PHQ-2 (2 questions)
//  - mental_wellbeing (index 74) - 1 question
//

import Foundation
import CoreData
import SwiftUI

// MARK: - Mental Health Question Model

struct MentalHealthQuestion: Identifiable {
    let id: Int
    let section: MentalHealthSection
    let text: String
    let subtitle: String?
    let icon: String
    let isFrequencyScale: Bool  // PHQ-2 uses 0-3 frequency scale
    let minLabel: String?
    let maxLabel: String?
}

enum MentalHealthSection: String {
    case cognitive = "Cognitive Function"
    case emotional = "Emotional Regulation"
    case depression = "Depression Screening"
    case wellbeing = "Mental Wellbeing"

    var color: Color {
        switch self {
        case .cognitive: return .blue
        case .emotional: return .pink
        case .depression: return .purple
        case .wellbeing: return .green
        }
    }
}

// MARK: - Questions Definition

struct MentalHealthQuestions {

    /// All questions in the mental health survey
    static let all: [MentalHealthQuestion] = [
        // === COGNITIVE FUNCTION (3 questions) ===
        MentalHealthQuestion(
            id: 0,
            section: .cognitive,
            text: "How difficult is it to concentrate on tasks today?",
            subtitle: "Reading, working, or following conversations",
            icon: "brain.head.profile",
            isFrequencyScale: false,
            minLabel: "No difficulty",
            maxLabel: "Extremely difficult"
        ),
        MentalHealthQuestion(
            id: 1,
            section: .cognitive,
            text: "How would you rate your mental clarity right now?",
            subtitle: "Ability to think clearly and make decisions",
            icon: "lightbulb.fill",
            isFrequencyScale: false,
            minLabel: "Very clear",
            maxLabel: "Very foggy"
        ),
        MentalHealthQuestion(
            id: 2,
            section: .cognitive,
            text: "How much is brain fog affecting you today?",
            subtitle: "Feeling mentally sluggish or confused",
            icon: "cloud.fill",
            isFrequencyScale: false,
            minLabel: "Not at all",
            maxLabel: "Severely"
        ),

        // === EMOTIONAL REGULATION (3 questions) ===
        MentalHealthQuestion(
            id: 3,
            section: .emotional,
            text: "How easily do you feel overwhelmed by emotions?",
            subtitle: "Difficulty managing strong feelings",
            icon: "heart.circle",
            isFrequencyScale: false,
            minLabel: "Not at all",
            maxLabel: "Very easily"
        ),
        MentalHealthQuestion(
            id: 4,
            section: .emotional,
            text: "How irritable or easily frustrated do you feel?",
            subtitle: "Quick to anger or annoyance",
            icon: "bolt.heart",
            isFrequencyScale: false,
            minLabel: "Not irritable",
            maxLabel: "Very irritable"
        ),
        MentalHealthQuestion(
            id: 5,
            section: .emotional,
            text: "How well can you calm yourself when upset?",
            subtitle: "Ability to self-soothe",
            icon: "leaf.fill",
            isFrequencyScale: false,
            minLabel: "Very well",
            maxLabel: "Cannot calm down"
        ),

        // === PHQ-2 DEPRESSION SCREENING (2 questions) ===
        // Validated PHQ-2 questions - do not modify wording
        MentalHealthQuestion(
            id: 6,
            section: .depression,
            text: "Over the past 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
            subtitle: nil,
            icon: "heart.slash",
            isFrequencyScale: true,
            minLabel: nil,
            maxLabel: nil
        ),
        MentalHealthQuestion(
            id: 7,
            section: .depression,
            text: "Over the past 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
            subtitle: nil,
            icon: "cloud.rain",
            isFrequencyScale: true,
            minLabel: nil,
            maxLabel: nil
        ),

        // === MENTAL WELLBEING (1 question) ===
        MentalHealthQuestion(
            id: 8,
            section: .wellbeing,
            text: "Overall, how would you rate your mental wellbeing right now?",
            subtitle: "General sense of mental health",
            icon: "sparkles",
            isFrequencyScale: false,
            minLabel: "Very poor",
            maxLabel: "Excellent"
        )
    ]

    static var totalQuestions: Int { all.count }
}

// MARK: - ViewModel

@MainActor
class MentalHealthSurveyViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var currentIndex = 0
    @Published var answers: [Double]
    @Published var showingResults = false
    @Published var showingError = false
    @Published var errorMessage = ""

    // MARK: - Computed Properties

    var progress: Double {
        Double(currentIndex + 1) / Double(MentalHealthQuestions.totalQuestions)
    }

    var isLastQuestion: Bool {
        currentIndex == MentalHealthQuestions.totalQuestions - 1
    }

    var currentQuestion: MentalHealthQuestion {
        MentalHealthQuestions.all[currentIndex]
    }

    var currentAnswer: Double {
        answers[currentIndex]
    }

    var currentSectionTitle: String {
        "\(currentQuestion.section.rawValue) - Question \(currentIndex + 1) of \(MentalHealthQuestions.totalQuestions)"
    }

    var currentSectionColor: Color {
        currentQuestion.section.color
    }

    /// Cognitive function score (0-10) - mean of first 3 questions
    var cognitiveScore: Double {
        let cognitiveAnswers = Array(answers[0..<3])
        return cognitiveAnswers.reduce(0, +) / 3.0
    }

    /// Emotional regulation score (0-10) - mean of questions 4-6
    var emotionalScore: Double {
        let emotionalAnswers = Array(answers[3..<6])
        return emotionalAnswers.reduce(0, +) / 3.0
    }

    /// PHQ-2 score (0-6) - sum of questions 7-8
    var phq2Score: Int {
        Int(answers[6]) + Int(answers[7])
    }

    /// Mental wellbeing score (0-10) - question 9 (inverted: 10 = excellent)
    var wellbeingScore: Double {
        10 - answers[8]  // Invert so higher = better
    }

    /// Depression risk (0-10) derived from PHQ-2
    var depressionRisk: Double {
        // PHQ-2 ranges 0-6, convert to 0-10 scale
        return Double(phq2Score) / 6.0 * 10.0
    }

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
        // Initialize answers array with defaults
        // Cognitive/Emotional/Wellbeing: 5 (middle), PHQ-2: 0 (not at all)
        self.answers = [5, 5, 5, 5, 5, 5, 0, 0, 5]
    }

    // MARK: - Actions

    func setAnswer(_ value: Double) {
        answers[currentIndex] = value
    }

    func nextQuestion() {
        guard currentIndex < MentalHealthQuestions.totalQuestions - 1 else { return }
        currentIndex += 1
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func previousQuestion() {
        guard currentIndex > 0 else { return }
        currentIndex -= 1
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    // MARK: - Completion

    func completeSurvey() {
        Task {
            do {
                // Create symptom log with mental health scores
                let log = SymptomLog(context: context)
                log.id = UUID()
                log.timestamp = Date()
                log.source = "mental_health_survey"

                // Store calculated scores
                log.cognitiveFunction = Float(cognitiveScore)
                log.emotionalRegulation = Float(emotionalScore)
                log.depressionRisk = Float(depressionRisk)
                log.mentalWellbeing = Float(wellbeingScore)

                // Store PHQ-2 raw score in notes for reference
                log.notes = "PHQ-2 score: \(phq2Score)/6. " +
                           "Cognitive: \(String(format: "%.1f", cognitiveScore))/10. " +
                           "Emotional: \(String(format: "%.1f", emotionalScore))/10. " +
                           "Wellbeing: \(String(format: "%.1f", wellbeingScore))/10."

                // Derive related scores
                log.mentalFatigueLevel = Float(cognitiveScore)  // Brain fog = mental fatigue
                log.anxietyLevel = Float(emotionalScore * 0.7)  // Emotional dysregulation correlates with anxiety

                // Fetch context data
                await attachContextData(to: log)

                // Auto-populate ML properties
                log.populateMLProperties(context: context)

                // Save
                try context.save()

                print("✅ Mental health survey saved")
                print("   - Cognitive function: \(String(format: "%.1f", cognitiveScore))")
                print("   - Emotional regulation: \(String(format: "%.1f", emotionalScore))")
                print("   - PHQ-2 score: \(phq2Score)/6")
                print("   - Mental wellbeing: \(String(format: "%.1f", wellbeingScore))")

                // Success feedback
                UINotificationFeedbackGenerator().notificationOccurred(.success)

                showingResults = true

            } catch {
                errorMessage = "Failed to save: \(error.localizedDescription)"
                showingError = true
            }
        }
    }

    // MARK: - Context Data

    private func attachContextData(to log: SymptomLog) async {
        let snapshot = ContextSnapshot(context: context)
        snapshot.id = UUID()
        snapshot.timestamp = Date()

        // Fetch weather
        do {
            let weather = try await OpenMeteoService.shared.fetchCurrentWeather()
            snapshot.barometricPressure = weather.pressure
            snapshot.humidity = Int16(weather.humidity)
            snapshot.temperature = weather.temperature
            snapshot.pressureChange12h = weather.pressureChange12h
        } catch {
            print("⚠️ Weather fetch failed: \(error.localizedDescription)")
        }

        // Fetch HealthKit data
        do {
            let biometrics = try await HealthKitService.shared.fetchAllBiometrics(for: Date())
            snapshot.restingHeartRate = Int16(biometrics.restingHeartRate)
            snapshot.stepCount = Int32(biometrics.stepCount)
            snapshot.hrvValue = biometrics.hrvValue
            snapshot.sleepEfficiency = biometrics.sleep.efficiency
        } catch {
            print("⚠️ HealthKit fetch failed: \(error.localizedDescription)")
        }

        log.contextSnapshot = snapshot
    }
}
