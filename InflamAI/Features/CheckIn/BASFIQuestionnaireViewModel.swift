//
//  BASFIQuestionnaireViewModel.swift
//  InflamAI
//
//  ViewModel for BASFI (Bath Ankylosing Spondylitis Functional Index)
//  Implements the validated 10-question functional assessment
//
//  ML Feature: basfi (index 8)
//  Scoring: Mean of 10 questions (0-10 each)
//

import Foundation
import CoreData
import Combine
import UIKit

// MARK: - BASFI Question Model

struct BASFIQuestion: Identifiable {
    let id: Int
    let text: String
    let subtitle: String
    let icon: String
}

// MARK: - BASFI Calculator

struct BASFICalculator {
    /// The 10 validated BASFI questions
    /// Reference: Calin A, et al. J Rheumatol. 1994;21(12):2281-5
    static let questions: [BASFIQuestion] = [
        BASFIQuestion(
            id: 0,
            text: "Putting on your socks or tights without help or aids",
            subtitle: "e.g., sock aid",
            icon: "🧦"
        ),
        BASFIQuestion(
            id: 1,
            text: "Bending forward from the waist to pick up a pen from the floor without an aid",
            subtitle: "Without bending your knees",
            icon: "✏️"
        ),
        BASFIQuestion(
            id: 2,
            text: "Reaching up to a high shelf without help or aids",
            subtitle: "e.g., grabber tool",
            icon: "📚"
        ),
        BASFIQuestion(
            id: 3,
            text: "Getting up out of an armless chair without using your hands or any other help",
            subtitle: "From a dining-type chair",
            icon: "🪑"
        ),
        BASFIQuestion(
            id: 4,
            text: "Getting up off the floor from lying on your back without help",
            subtitle: "Without assistance",
            icon: "🛏️"
        ),
        BASFIQuestion(
            id: 5,
            text: "Standing unsupported for 10 minutes without discomfort",
            subtitle: "Without leaning",
            icon: "🧍"
        ),
        BASFIQuestion(
            id: 6,
            text: "Climbing 12-15 steps without using a handrail or walking aid",
            subtitle: "One foot on each step",
            icon: "🪜"
        ),
        BASFIQuestion(
            id: 7,
            text: "Looking over your shoulder without turning your body",
            subtitle: "While standing",
            icon: "👀"
        ),
        BASFIQuestion(
            id: 8,
            text: "Doing physically demanding activities",
            subtitle: "e.g., physiotherapy, gardening, sports",
            icon: "🏃"
        ),
        BASFIQuestion(
            id: 9,
            text: "Doing a full day's activities at home or work",
            subtitle: "Whether at home or at work",
            icon: "💼"
        )
    ]

    /// Calculate BASFI score (mean of 10 questions)
    static func calculate(answers: [Double]) -> Double? {
        guard answers.count == 10 else { return nil }
        guard answers.allSatisfy({ $0 >= 0 && $0 <= 10 }) else { return nil }

        let sum = answers.reduce(0, +)
        return sum / 10.0
    }

    /// Interpret BASFI score
    static func interpretation(score: Double) -> (category: String, advice: String, color: String) {
        switch score {
        case 0..<2:
            return (
                "Minimal Limitation",
                "Good functional capacity. Continue with regular exercise.",
                "green"
            )
        case 2..<4:
            return (
                "Mild Limitation",
                "Minor functional impact. Regular stretching is recommended.",
                "yellow"
            )
        case 4..<6:
            return (
                "Moderate Limitation",
                "Discuss mobility options with your rheumatologist.",
                "orange"
            )
        case 6..<8:
            return (
                "Significant Limitation",
                "Discuss treatment optimization with your healthcare provider.",
                "red"
            )
        default:
            return (
                "Severe Limitation",
                "Please urgently discuss treatment options with your doctor.",
                "red"
            )
        }
    }
}

// MARK: - ViewModel

@MainActor
class BASFIQuestionnaireViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var currentIndex = 0
    @Published var answers: [Double] = Array(repeating: 5.0, count: 10)
    @Published var showingResults = false
    @Published var showingError = false
    @Published var errorMessage = ""

    // MARK: - Computed Properties

    var progress: Double {
        Double(currentIndex + 1) / 10.0
    }

    var isLastQuestion: Bool {
        currentIndex == 9
    }

    var currentQuestion: BASFIQuestion {
        BASFICalculator.questions[currentIndex]
    }

    var basfiScore: Double {
        BASFICalculator.calculate(answers: answers) ?? 0.0
    }

    // MARK: - Dependencies

    private let context: NSManagedObjectContext

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - Navigation

    func nextQuestion() {
        guard currentIndex < 9 else { return }
        currentIndex += 1
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func previousQuestion() {
        guard currentIndex > 0 else { return }
        currentIndex -= 1
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    // MARK: - Completion

    func completeQuestionnaire() {
        Task {
            do {
                guard let score = BASFICalculator.calculate(answers: answers) else {
                    errorMessage = "Invalid answers. Please check all questions."
                    showingError = true
                    return
                }

                // Create symptom log with BASFI score
                let log = SymptomLog(context: context)
                log.id = UUID()
                log.timestamp = Date()
                log.source = "basfi_questionnaire"
                log.basfi = Float(score)

                // Store individual answers in notes for reference
                let answersString = answers.enumerated().map { "Q\($0.offset + 1): \(Int($0.element))" }.joined(separator: ", ")
                log.notes = "BASFI answers: \(answersString)"

                // Fetch context data
                await attachContextData(to: log)

                // Auto-populate ML properties
                log.populateMLProperties(context: context)

                // Save
                try context.save()

                print("✅ BASFI questionnaire saved")
                print("   - Score: \(String(format: "%.1f", score))")
                print("   - Answers: \(answers.map { String(format: "%.0f", $0) })")

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
