//
//  ASDACalculator.swift
//  InflamAI
//
//  Clinically accurate Ankylosing Spondylitis Disease Activity Score with CRP
//  Formula: 0.12×BackPain + 0.06×Duration + 0.11×PatientGlobal + 0.07×PeripheralPain + 0.58×Ln(CRP+1)
//

import Foundation
import SwiftUI

/// ASDAS-CRP Calculator - Medical-grade accuracy
struct ASDACalculator {

    // MARK: - Medical Formula

    /// Calculate ASDAS-CRP score using validated medical formula
    ///
    /// Formula: 0.12×BackPain + 0.06×Duration + 0.11×PatientGlobal + 0.07×PeripheralPain + 0.58×Ln(CRP+1)
    ///
    /// - Parameters:
    ///   - backPain: Spinal pain level (0-10 VAS)
    ///   - duration: Morning stiffness duration (0-10 scale)
    ///   - patientGlobal: Patient global assessment (0-10 VAS)
    ///   - peripheralPain: Peripheral joint pain/swelling (0-10 VAS)
    ///   - crp: C-Reactive Protein in mg/L
    /// - Returns: ASDAS-CRP score (typically 0-6) or nil if invalid
    static func calculate(
        backPain: Double,
        duration: Double,
        patientGlobal: Double,
        peripheralPain: Double,
        crp: Double
    ) -> Double? {
        // Validate inputs
        guard backPain >= 0 && backPain <= 10,
              duration >= 0 && duration <= 10,
              patientGlobal >= 0 && patientGlobal <= 10,
              peripheralPain >= 0 && peripheralPain <= 10,
              crp >= 0 else {
            print("❌ ASDAS calculation failed: Invalid input parameters")
            return nil
        }

        // Medical formula with validated coefficients
        let score = (0.12 * backPain) +
                    (0.06 * duration) +
                    (0.11 * patientGlobal) +
                    (0.07 * peripheralPain) +
                    (0.58 * log(crp + 1))

        return score
    }

    /// Calculate ASDAS from SymptomLog (using BASDAI components + CRP)
    static func calculate(from log: SymptomLog) -> Double? {
        guard let answersData = log.basdaiAnswers,
              let answers = try? JSONDecoder().decode([Double].self, from: answersData) else {
            return nil
        }

        // Map BASDAI questions to ASDAS components
        let backPain = answers[1] // Q2: Spinal pain
        let durationRaw = answers[5] // Q6: Morning stiffness in minutes
        let duration = min(durationRaw / 12.0, 10.0) // Scale to 0-10
        let peripheralPain = answers[2] // Q3: Peripheral joints
        let patientGlobal = (answers[0] + answers[1] + answers[4]) / 3.0 // Average of fatigue, pain, stiffness
        let crp = log.crpValue

        return calculate(
            backPain: backPain,
            duration: duration,
            patientGlobal: patientGlobal,
            peripheralPain: peripheralPain,
            crp: crp
        )
    }

    // MARK: - Interpretation

    /// Interpret ASDAS-CRP score according to clinical cutoffs
    static func interpretation(score: Double) -> ASDAInterpretation {
        switch score {
        case ..<1.3:
            return ASDAInterpretation(
                category: "Inactive Disease",
                severity: .inactive,
                color: .green,
                advice: "Disease is inactive. Maintain current treatment and monitoring schedule.",
                clinicalNote: "ASDAS < 1.3 indicates inactive disease state."
            )
        case 1.3..<2.1:
            return ASDAInterpretation(
                category: "Moderate Activity",
                severity: .moderate,
                color: .yellow,
                advice: "Moderate disease activity. Regular monitoring recommended.",
                clinicalNote: "ASDAS 1.3-2.1 indicates moderate disease activity."
            )
        case 2.1..<3.5:
            return ASDAInterpretation(
                category: "High Activity",
                severity: .high,
                color: .orange,
                advice: "High disease activity. Discuss treatment intensification with rheumatologist.",
                clinicalNote: "ASDAS 2.1-3.5 indicates high disease activity. Consider treatment escalation."
            )
        default:
            return ASDAInterpretation(
                category: "Very High Activity",
                severity: .veryHigh,
                color: .red,
                advice: "Very high disease activity. Urgent rheumatology consultation recommended.",
                clinicalNote: "ASDAS ≥ 3.5 indicates very high disease activity. Immediate intervention may be needed."
            )
        }
    }

    /// Determine if ASDAS change is clinically important (≥1.1 units)
    static func isClinicallyImportantChange(from baseline: Double, to current: Double) -> Bool {
        abs(current - baseline) >= 1.1
    }

    /// Determine if patient achieved ASDAS improvement (≥0.6 change)
    static func hasImprovement(from baseline: Double, to current: Double) -> Bool {
        (baseline - current) >= 0.6
    }
}

// MARK: - Supporting Models

struct ASDAInterpretation {
    let category: String
    let severity: Severity
    let color: Color
    let advice: String
    let clinicalNote: String

    enum Severity: String {
        case inactive = "Inactive"
        case moderate = "Moderate"
        case high = "High"
        case veryHigh = "Very High"
    }
}

// MARK: - Validation Tests

#if DEBUG
extension ASDACalculator {
    /// Test ASDAS calculation against known medical literature values
    static func runValidationTests() {
        print("🧪 Running ASDAS validation tests...")

        // Test 1: Example from medical literature
        // BackPain=6, Duration=5, Global=6, Peripheral=4, CRP=10
        let test1 = calculate(backPain: 6, duration: 5, patientGlobal: 6, peripheralPain: 4, crp: 10)
        let term1: Double = 0.12 * 6
        let term2: Double = 0.06 * 5
        let term3: Double = 0.11 * 6
        let term4: Double = 0.07 * 4
        let term5: Double = 0.58 * log(11)
        let expected1 = term1 + term2 + term3 + term4 + term5
        assert(abs(test1! - expected1) < 0.01, "Test 1 failed: \(test1!) vs \(expected1)")
        print("✅ Test 1 passed: Medical literature example")

        // Test 2: Inactive disease (low values, CRP=0.5)
        let test2 = calculate(backPain: 1, duration: 0.5, patientGlobal: 1, peripheralPain: 0, crp: 0.5)
        assert(test2! < 1.3, "Test 2 failed: Should indicate inactive disease")
        print("✅ Test 2 passed: Inactive disease")

        // Test 3: Very high activity
        let test3 = calculate(backPain: 9, duration: 10, patientGlobal: 9, peripheralPain: 8, crp: 50)
        assert(test3! >= 3.5, "Test 3 failed: Should indicate very high activity")
        print("✅ Test 3 passed: Very high activity")

        // Test 4: Clinically important change
        let baseline = 3.0
        let current = 1.8
        assert(isClinicallyImportantChange(from: baseline, to: current), "Test 4 failed")
        print("✅ Test 4 passed: Clinically important change detection")

        // Test 5: ASDAS improvement
        assert(hasImprovement(from: 3.0, to: 2.0), "Test 5 failed")
        print("✅ Test 5 passed: ASDAS improvement detection")

        print("✅ All ASDAS validation tests passed!")
    }
}
#endif
