//
//  ExplainabilityEngine.swift
//  InflamAI
//
//  Swift-native feature importance computation (SHAP-like perturbation method)
//  Explains which features drove each prediction
//

import Foundation
import CoreML

// Type alias for compatibility
typealias ExplainabilityFeatureScaler = CalibrationFeatureScaler

@available(iOS 17.0, *)
@MainActor
class ExplainabilityEngine {

    private let model: ASFlarePredictor
    private let scaler: ExplainabilityFeatureScaler
    private let featureNames: [String]

    struct FeatureImportance: Identifiable {
        let id = UUID()
        let featureName: String
        let importance: Float  // -1 to 1 (negative = decreases risk, positive = increases risk)
        let currentValue: Float
        let normalizedValue: Float

        var impactDescription: String {
            if abs(importance) < 0.1 {
                return "Minimal impact"
            } else if importance > 0.3 {
                return "Strong risk increase"
            } else if importance > 0.1 {
                return "Moderate risk increase"
            } else if importance < -0.3 {
                return "Strong risk decrease"
            } else {
                return "Moderate risk decrease"
            }
        }

        var icon: String {
            importance > 0.1 ? "arrow.up.circle.fill" :
            importance < -0.1 ? "arrow.down.circle.fill" :
            "minus.circle.fill"
        }

        var color: String {
            importance > 0.1 ? "red" :
            importance < -0.1 ? "green" :
            "gray"
        }
    }

    init(model: ASFlarePredictor, scaler: ExplainabilityFeatureScaler, featureNames: [String]) {
        self.model = model
        self.scaler = scaler
        self.featureNames = featureNames
    }

    // MARK: - Feature Importance Computation

    /// Compute SHAP-like feature importance using perturbation method
    /// This approximates SHAP values by perturbing each feature and measuring impact
    func computeFeatureImportance(
        for features: [[Float]],
        topK: Int = 5
    ) async throws -> [FeatureImportance] {

        // Get baseline prediction
        let baselineProbability = try await getPredictionProbability(features: features)

        // Compute importance for each feature
        var importanceScores: [(index: Int, importance: Float, value: Float)] = []

        // We'll sample features rather than testing all 92 (too slow)
        // Focus on most variable features from the latest day
        let latestDay = features.last ?? Array(repeating: 0.0, count: 92)
        let significantFeatures = getSignificantFeatureIndices(from: latestDay)

        for featureIndex in significantFeatures {
            // Create perturbed version (set feature to neutral/zero)
            var perturbedFeatures = features
            for dayIndex in 0..<perturbedFeatures.count {
                perturbedFeatures[dayIndex][featureIndex] = 0.0  // Neutral value (already normalized)
            }

            // Get perturbed prediction
            let perturbedProbability = try await getPredictionProbability(features: perturbedFeatures)

            // Importance = how much prediction changed when feature removed
            let importance = baselineProbability - perturbedProbability

            importanceScores.append((
                index: featureIndex,
                importance: importance,
                value: latestDay[featureIndex]
            ))
        }

        // Sort by absolute importance
        importanceScores.sort { abs($0.importance) > abs($1.importance) }

        // Convert to FeatureImportance objects
        let topFeatures = importanceScores.prefix(topK).map { score -> FeatureImportance in
            let featureName = score.index < featureNames.count ?
                featureNames[score.index] : "Feature \(score.index)"

            return FeatureImportance(
                featureName: humanReadableName(featureName),
                importance: score.importance,
                currentValue: score.value,  // Normalized value
                normalizedValue: score.value
            )
        }

        return topFeatures
    }

    // MARK: - Helper Methods

    private func getPredictionProbability(features: [[Float]]) async throws -> Float {
        // Normalize features
        let normalizedFeatures = scaler.transform(features)

        // Create MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
        for (i, timestep) in normalizedFeatures.enumerated() {
            for (j, value) in timestep.enumerated() {
                inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
            }
        }

        // Run prediction
        let input = ASFlarePredictorInput(features: inputArray)
        let output = try await model.prediction(input: input)

        // Extract flare probability
        let probabilities = output.probabilities
        return probabilities[1].floatValue  // P(flare = 1)
    }

    private func getSignificantFeatureIndices(from features: [Float]) -> [Int] {
        // Return indices of features with non-zero values
        // This reduces computation while focusing on relevant features
        var significant: [Int] = []

        for (index, value) in features.enumerated() {
            if abs(value) > 0.01 {  // Threshold for significance
                significant.append(index)
            }
        }

        // If too few, sample random features
        if significant.count < 10 {
            significant = Array(0..<min(30, features.count))
        }

        return significant
    }

    private func humanReadableName(_ technicalName: String) -> String {
        // Convert snake_case to readable names
        let readable = technicalName
            .replacingOccurrences(of: "_", with: " ")
            .capitalized

        // Add specific translations
        let translations: [String: String] = [
            "Basdai Score": "Disease Activity",
            "Pain Current": "Current Pain Level",
            "Sleep Hours": "Sleep Duration",
            "Hrv": "Heart Rate Variability",
            "Resting Hr": "Resting Heart Rate",
            "Pressure Change": "Barometric Pressure Change",
            "Mood Current": "Current Mood",
            "Stress Level": "Stress Level",
            "Med Adherence": "Medication Adherence",
            "Morning Stiffness Duration": "Morning Stiffness",
            "Steps": "Daily Steps"
        ]

        return translations[readable] ?? readable
    }

    // MARK: - Counterfactual Explanations ("What-If" Scenarios)

    /// Generate actionable recommendations by finding which features could be changed
    func generateRecommendations(
        for features: [[Float]],
        currentProbability: Float
    ) async throws -> [Recommendation] {

        var recommendations: [Recommendation] = []

        // Test modifiable features
        let modifiableFeatures: [(index: Int, name: String, change: String)] = [
            (54, "sleep_hours", "Sleep 1 more hour"),
            (67, "stress_level", "Reduce stress (meditation, rest)"),
            (83, "med_adherence", "Take medication on time"),
            (46, "training_minutes", "Light exercise (15 minutes)"),
            (21, "pain_current", "Apply pain management techniques")
        ]

        for (featureIndex, featureName, suggestion) in modifiableFeatures {
            var modifiedFeatures = features

            // Simulate improvement
            let improvementFactor: Float = featureName.contains("adherence") ? 1.0 :
                                           featureName.contains("stress") ? -2.0 :
                                           featureName.contains("sleep") ? 1.0 :
                                           featureName.contains("pain") ? -2.0 : 0.5

            // Apply to most recent days
            for dayIndex in max(0, features.count - 3)..<features.count {
                modifiedFeatures[dayIndex][featureIndex] += improvementFactor
            }

            // Get new prediction
            let newProbability = try await getPredictionProbability(features: modifiedFeatures)
            let reduction = currentProbability - newProbability

            if reduction > 0.05 {  // At least 5% risk reduction
                recommendations.append(Recommendation(
                    action: suggestion,
                    expectedReduction: reduction,
                    confidence: min(1.0, reduction * 2.0),  // Scale confidence
                    category: categorizeFeature(featureName)
                ))
            }
        }

        // Sort by expected reduction
        recommendations.sort { $0.expectedReduction > $1.expectedReduction }

        return Array(recommendations.prefix(5))  // Top 5 recommendations
    }

    struct Recommendation: Identifiable {
        let id = UUID()
        let action: String
        let expectedReduction: Float  // 0-1 (percentage point reduction in flare risk)
        let confidence: Float  // 0-1
        let category: String

        var impactDescription: String {
            let percentage = Int(expectedReduction * 100)
            return "Could reduce risk by \(percentage)%"
        }

        var icon: String {
            switch category {
            case "Sleep": return "bed.double.fill"
            case "Stress": return "brain.head.profile"
            case "Medication": return "pills.fill"
            case "Exercise": return "figure.walk"
            case "Pain": return "bandage.fill"
            default: return "checkmark.circle.fill"
            }
        }
    }

    private func categorizeFeature(_ featureName: String) -> String {
        if featureName.contains("sleep") { return "Sleep" }
        if featureName.contains("stress") { return "Stress" }
        if featureName.contains("med") || featureName.contains("adherence") { return "Medication" }
        if featureName.contains("training") || featureName.contains("exercise") { return "Exercise" }
        if featureName.contains("pain") { return "Pain" }
        return "General"
    }
}
