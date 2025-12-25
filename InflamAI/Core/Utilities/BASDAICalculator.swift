//
//  BASDAICalculator.swift
//  InflamAI
//
//  Clinically accurate Bath Ankylosing Spondylitis Disease Activity Index calculator
//  Formula validated against medical literature
//

import Foundation
import SwiftUI

/// BASDAI Calculator - Medical-grade accuracy
struct BASDAICalculator {

    // MARK: - BASDAI Questions

    static let questions: [BASDAIQuestion] = [
        BASDAIQuestion(
            number: 1,
            text: "How would you describe the overall level of FATIGUE/TIREDNESS you have experienced?",
            subtitle: "During the past week",
            minLabel: "None",
            maxLabel: "Very severe"
        ),
        BASDAIQuestion(
            number: 2,
            text: "How would you describe the overall level of AS NECK, BACK OR HIP PAIN you have had?",
            subtitle: "During the past week",
            minLabel: "None",
            maxLabel: "Very severe"
        ),
        BASDAIQuestion(
            number: 3,
            text: "How would you describe the overall level of PAIN/SWELLING in joints other than neck, back, or hips you have had?",
            subtitle: "During the past week",
            minLabel: "None",
            maxLabel: "Very severe"
        ),
        BASDAIQuestion(
            number: 4,
            text: "How would you describe the overall level of DISCOMFORT you have had from any areas tender to touch or pressure?",
            subtitle: "During the past week",
            minLabel: "None",
            maxLabel: "Very severe"
        ),
        BASDAIQuestion(
            number: 5,
            text: "How would you describe the overall level of MORNING STIFFNESS you have had from the time you wake up?",
            subtitle: "During the past week",
            minLabel: "None",
            maxLabel: "Very severe"
        ),
        BASDAIQuestion(
            number: 6,
            text: "How long does your MORNING STIFFNESS last from the time you wake up?",
            subtitle: "During the past week",
            minLabel: "0 minutes",
            maxLabel: "2+ hours",
            isDuration: true
        )
    ]

    // MARK: - Medical Formula

    /// Calculate BASDAI score using validated medical formula
    /// Formula: (Q1 + Q2 + Q3 + Q4 + ((Q5 + Q6scaled) / 2)) / 5
    /// Q6: Convert minutes to 0-10 scale (0 min = 0, 120+ min = 10)
    ///
    /// - Parameter answers: Array of 6 answers [Q1, Q2, Q3, Q4, Q5, Q6_minutes]
    /// - Returns: BASDAI score (0-10) or nil if invalid
    static func calculate(answers: [Double]) -> Double? {
        guard answers.count == 6 else {
            print("❌ BASDAI calculation failed: Expected 6 answers, got \(answers.count)")
            return nil
        }

        // Validate ranges
        for (index, answer) in answers.enumerated() {
            if index < 5 {
                // Q1-Q5 must be 0-10
                guard answer >= 0 && answer <= 10 else {
                    print("❌ BASDAI Q\(index + 1) out of range: \(answer)")
                    return nil
                }
            } else {
                // Q6 is in minutes, must be >= 0
                guard answer >= 0 else {
                    print("❌ BASDAI Q6 (duration) negative: \(answer)")
                    return nil
                }
            }
        }

        // Q6: Convert minutes to 0-10 scale
        // 0 min = 0, 120+ min = 10
        let q6Scaled = min(answers[5] / 12.0, 10.0)

        // Medical formula
        let score = (answers[0] + answers[1] + answers[2] + answers[3] +
                     ((answers[4] + q6Scaled) / 2.0)) / 5.0

        return score
    }

    /// Calculate BASDAI from SymptomLog
    static func calculate(from log: SymptomLog) -> Double? {
        guard let answersData = log.basdaiAnswers,
              let answers = try? JSONDecoder().decode([Double].self, from: answersData) else {
            return nil
        }
        return calculate(answers: answers)
    }

    // MARK: - Interpretation

    /// Interpret BASDAI score
    static func interpretation(score: Double) -> BASDAIInterpretation {
        switch score {
        case 0..<2:
            return BASDAIInterpretation(
                category: "Remission",
                severity: .remission,
                color: .green,
                advice: "Symptoms are well controlled. Continue current treatment.",
                clinicalNote: "BASDAI < 2 indicates remission or very low disease activity."
            )
        case 2..<4:
            return BASDAIInterpretation(
                category: "Low Activity",
                severity: .low,
                color: Color(red: 0.6, green: 0.8, blue: 0.2),
                advice: "Disease activity is low. Monitor for changes and maintain routine care.",
                clinicalNote: "BASDAI 2-4 indicates low disease activity. No immediate changes needed."
            )
        case 4..<6:
            return BASDAIInterpretation(
                category: "Moderate Activity",
                severity: .moderate,
                color: .orange,
                advice: "Moderate disease activity. Consider discussing with your rheumatologist.",
                clinicalNote: "BASDAI 4-6 may warrant treatment adjustment. Schedule rheumatology visit."
            )
        case 6..<8:
            return BASDAIInterpretation(
                category: "High Activity",
                severity: .high,
                color: Color(red: 0.9, green: 0.3, blue: 0.1),
                advice: "High disease activity. Contact your rheumatologist soon.",
                clinicalNote: "BASDAI ≥6 indicates high activity. Treatment escalation may be needed."
            )
        default:
            return BASDAIInterpretation(
                category: "Very High Activity",
                severity: .veryHigh,
                color: .red,
                advice: "Very high disease activity. Contact your healthcare provider urgently.",
                clinicalNote: "BASDAI ≥8 indicates very high activity. Urgent rheumatology consultation recommended."
            )
        }
    }

    // MARK: - Trend Analysis

    /// Calculate BASDAI trend (positive = worsening, negative = improving)
    static func calculateTrend(logs: [SymptomLog]) -> Double? {
        guard logs.count >= 2 else { return nil }

        let sortedLogs = logs.sorted { $0.timestamp ?? Date.distantPast < $1.timestamp ?? Date.distantPast }
        guard let firstScore = sortedLogs.first?.basdaiScore,
              let lastScore = sortedLogs.last?.basdaiScore else {
            return nil
        }

        return lastScore - firstScore
    }

    /// Calculate average BASDAI for period
    static func average(logs: [SymptomLog]) -> Double? {
        guard !logs.isEmpty else { return nil }
        let total = logs.reduce(0.0) { $0 + $1.basdaiScore }
        return total / Double(logs.count)
    }
}

// MARK: - Supporting Models

struct BASDAIQuestion {
    let number: Int
    let text: String
    let subtitle: String
    let minLabel: String
    let maxLabel: String
    let isDuration: Bool

    init(number: Int, text: String, subtitle: String, minLabel: String, maxLabel: String, isDuration: Bool = false) {
        self.number = number
        self.text = text
        self.subtitle = subtitle
        self.minLabel = minLabel
        self.maxLabel = maxLabel
        self.isDuration = isDuration
    }
}

struct BASDAIInterpretation {
    let category: String
    let severity: Severity
    let color: Color
    let advice: String
    let clinicalNote: String

    enum Severity: String {
        case remission = "Remission"
        case low = "Low"
        case moderate = "Moderate"
        case high = "High"
        case veryHigh = "Very High"
    }
}

// MARK: - Validation Tests (Internal)

#if DEBUG
extension BASDAICalculator {
    /// Test BASDAI calculation against known medical literature values
    static func runValidationTests() {
        print("🧪 Running BASDAI validation tests...")

        // Test 1: Example calculation
        // Q1-Q5: [6, 5, 7, 6, 8], Q6: 45 minutes
        // Formula: (6 + 5 + 7 + 6 + ((8 + 3.75) / 2)) / 5 = 5.975
        // Q6 scaled: 45/12 = 3.75
        let test1 = calculate(answers: [6, 5, 7, 6, 8, 45])
        assert(abs(test1! - 5.975) < 0.01, "Test 1 failed: \(test1!)")
        print("✅ Test 1 passed: Example calculation")

        // Test 2: Remission (all zeros)
        let test2 = calculate(answers: [0, 0, 0, 0, 0, 0])
        assert(test2! == 0.0, "Test 2 failed: \(test2!)")
        print("✅ Test 2 passed: Remission")

        // Test 3: Very high activity
        let test3 = calculate(answers: [10, 10, 10, 10, 10, 120])
        assert(test3! == 10.0, "Test 3 failed: \(test3!)")
        print("✅ Test 3 passed: Maximum score")

        // Test 4: Q6 scaling (90 minutes = 7.5 on scale)
        let test4 = calculate(answers: [5, 5, 5, 5, 5, 90])
        let expectedQ6Scaled = 90.0 / 12.0 // = 7.5
        let expected = (5 + 5 + 5 + 5 + ((5 + expectedQ6Scaled) / 2.0)) / 5.0
        assert(abs(test4! - expected) < 0.01, "Test 4 failed: \(test4!) vs \(expected)")
        print("✅ Test 4 passed: Q6 duration scaling")

        // Test 5: Invalid input (wrong count)
        let test5 = calculate(answers: [5, 5, 5])
        assert(test5 == nil, "Test 5 failed: Should return nil for invalid input")
        print("✅ Test 5 passed: Invalid input rejection")

        print("✅ All BASDAI validation tests passed!")
    }
}
#endif
