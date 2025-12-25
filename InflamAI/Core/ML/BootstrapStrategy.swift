//
//  BootstrapStrategy.swift
//  InflamAI
//
//  Hybrid prediction strategy that progressively blends synthetic baseline
//  with personalized model as user collects data
//

import Foundation
import CoreML

@MainActor
class BootstrapStrategy: ObservableObject {

    // MARK: - Published Properties

    @Published var currentPhase: BootstrapPhase = .initializing
    @Published var daysOfData: Int = 0
    @Published var syntheticWeight: Float = 1.0
    @Published var personalWeight: Float = 0.0
    @Published var confidenceModifier: Float = 0.5  // Reduces confidence early on

    enum BootstrapPhase: String, Comparable {
        case initializing = "Initializing"           // Day 0
        case fullySynthetic = "Baseline"            // Days 1-7
        case earlyBlending = "Early Learning"       // Days 8-14
        case midBlending = "Adapting"               // Days 15-21
        case lateBlending = "Almost There"          // Days 22-27
        case personalized = "Personalized"          // Days 28+

        var dayRange: ClosedRange<Int> {
            switch self {
            case .initializing: return 0...0
            case .fullySynthetic: return 1...7
            case .earlyBlending: return 8...14
            case .midBlending: return 15...21
            case .lateBlending: return 22...27
            case .personalized: return 28...Int.max
            }
        }

        var description: String {
            switch self {
            case .initializing:
                return "Setting up your model"
            case .fullySynthetic:
                return "Using baseline patterns (collecting your data)"
            case .earlyBlending:
                return "Starting to learn your patterns (30% personalized)"
            case .midBlending:
                return "Model adapting to you (60% personalized)"
            case .lateBlending:
                return "Almost fully personalized (90% personalized)"
            case .personalized:
                return "Fully personalized to your unique patterns"
            }
        }

        var icon: String {
            switch self {
            case .initializing: return "gearshape.fill"
            case .fullySynthetic: return "chart.bar.fill"
            case .earlyBlending: return "arrow.triangle.2.circlepath"
            case .midBlending: return "sparkles"
            case .lateBlending: return "checkmark.circle.fill"
            case .personalized: return "star.fill"
            }
        }

        var color: String {
            switch self {
            case .initializing: return "gray"
            case .fullySynthetic: return "orange"
            case .earlyBlending: return "yellow"
            case .midBlending: return "blue"
            case .lateBlending: return "green"
            case .personalized: return "purple"
            }
        }

        static func < (lhs: BootstrapPhase, rhs: BootstrapPhase) -> Bool {
            return lhs.dayRange.lowerBound < rhs.dayRange.lowerBound
        }
    }

    // MARK: - Dependencies

    private let syntheticModel: ASFlarePredictor
    private var personalizedModel: ASFlarePredictor?
    private let scaler: FeatureScaler

    // MARK: - Configuration

    struct BlendingConfig {
        let syntheticWeight: Float
        let personalWeight: Float
        let confidenceModifier: Float  // Multiplier for confidence (0.5 = reduce by 50%)

        static let fullySynthetic = BlendingConfig(
            syntheticWeight: 1.0,
            personalWeight: 0.0,
            confidenceModifier: 0.6  // Moderate confidence
        )

        static let earlyBlending = BlendingConfig(
            syntheticWeight: 0.7,
            personalWeight: 0.3,
            confidenceModifier: 0.7
        )

        static let midBlending = BlendingConfig(
            syntheticWeight: 0.4,
            personalWeight: 0.6,
            confidenceModifier: 0.85
        )

        static let lateBlending = BlendingConfig(
            syntheticWeight: 0.1,
            personalWeight: 0.9,
            confidenceModifier: 0.95
        )

        static let personalized = BlendingConfig(
            syntheticWeight: 0.0,
            personalWeight: 1.0,
            confidenceModifier: 1.0  // Full confidence
        )
    }

    // MARK: - Initialization

    init(syntheticModel: ASFlarePredictor, scaler: FeatureScaler) {
        self.syntheticModel = syntheticModel
        self.scaler = scaler
        updatePhase()
    }

    // MARK: - Public API

    /// Make hybrid prediction that blends synthetic + personalized models
    func predict(features: [[Float]]) async throws -> BootstrapHybridPrediction {
        // Normalize features (transform each day's features separately)
        let normalizedFeatures = features.map { dayFeatures in
            scaler.transform(dayFeatures)
        }

        // Get synthetic prediction
        let syntheticProb = try await getPrediction(
            from: syntheticModel,
            features: normalizedFeatures
        )

        // Get personalized prediction (if available)
        var personalProb: Float?
        if let personalModel = personalizedModel {
            personalProb = try await getPrediction(
                from: personalModel,
                features: normalizedFeatures
            )
        }

        // Blend predictions based on current phase
        let blendedProb = blendPredictions(
            synthetic: syntheticProb,
            personal: personalProb
        )

        // Compute confidence adjustment
        let rawConfidence = computeRawConfidence(probability: blendedProb)
        let adjustedConfidence = rawConfidence * confidenceModifier

        return BootstrapHybridPrediction(
            probability: blendedProb,
            confidence: adjustedConfidence,
            syntheticProbability: syntheticProb,
            personalProbability: personalProb,
            phase: currentPhase,
            syntheticWeight: syntheticWeight,
            personalWeight: personalWeight
        )
    }

    /// Update bootstrap phase based on days of data
    func updatePhase(daysOfData: Int? = nil) {
        if let days = daysOfData {
            self.daysOfData = days
        }

        // Determine phase
        let newPhase: BootstrapPhase
        switch self.daysOfData {
        case 0:
            newPhase = .initializing
        case 1...7:
            newPhase = .fullySynthetic
        case 8...14:
            newPhase = .earlyBlending
        case 15...21:
            newPhase = .midBlending
        case 22...27:
            newPhase = .lateBlending
        default:
            newPhase = .personalized
        }

        // Update phase if changed
        if newPhase != currentPhase {
            currentPhase = newPhase
            updateBlendingWeights()
        }
    }

    /// Load personalized model when available
    func loadPersonalizedModel(from url: URL) throws {
        personalizedModel = try ASFlarePredictor(contentsOf: url)
        print("✅ Loaded personalized model from \(url.lastPathComponent)")
    }

    /// Check if personalized model is available
    var hasPersonalizedModel: Bool {
        return personalizedModel != nil
    }

    // MARK: - Prediction Blending

    private func blendPredictions(
        synthetic: Float,
        personal: Float?
    ) -> Float {
        guard let personal = personal else {
            // No personalized model yet - use synthetic
            return synthetic
        }

        // Weighted average
        let blended = syntheticWeight * synthetic + personalWeight * personal

        return blended
    }

    private func updateBlendingWeights() {
        let config = getBlendingConfig(for: currentPhase)

        syntheticWeight = config.syntheticWeight
        personalWeight = config.personalWeight
        confidenceModifier = config.confidenceModifier

        print("📊 Bootstrap phase: \(currentPhase.rawValue)")
        print("   Weights: \(String(format: "%.0f%%", syntheticWeight * 100)) synthetic, \(String(format: "%.0f%%", personalWeight * 100)) personal")
    }

    private func getBlendingConfig(for phase: BootstrapPhase) -> BlendingConfig {
        switch phase {
        case .initializing, .fullySynthetic:
            return .fullySynthetic
        case .earlyBlending:
            return .earlyBlending
        case .midBlending:
            return .midBlending
        case .lateBlending:
            return .lateBlending
        case .personalized:
            return .personalized
        }
    }

    // MARK: - Helper Methods

    private func getPrediction(
        from model: ASFlarePredictor,
        features: [[Float]]
    ) async throws -> Float {
        // Create MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
        for (i, timestep) in features.enumerated() {
            for (j, value) in timestep.enumerated() {
                inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
            }
        }

        // Run prediction
        let input = ASFlarePredictorInput(features: inputArray)
        let output = try await model.prediction(input: input)

        // Extract probability
        let probabilities = output.probabilities
        return probabilities[1].floatValue  // P(flare = 1)
    }

    private func computeRawConfidence(probability: Float) -> Float {
        // Confidence increases as probability moves away from 0.5
        // P = 0.5 → confidence = 0
        // P = 0.0 or 1.0 → confidence = 1.0
        return abs(probability - 0.5) * 2.0
    }

    // MARK: - Progress Tracking

    /// Get detailed bootstrap progress for UI
    func getBootstrapProgress() -> BootstrapProgress {
        let currentPhaseEnd = currentPhase.dayRange.upperBound
        let progressInPhase: Float

        if currentPhase == .personalized {
            progressInPhase = 1.0
        } else {
            let phaseStart = currentPhase.dayRange.lowerBound
            let phaseLength = currentPhaseEnd - phaseStart + 1
            let dayInPhase = daysOfData - phaseStart + 1
            progressInPhase = Float(dayInPhase) / Float(phaseLength)
        }

        // Overall progress (to day 28)
        let overallProgress = min(1.0, Float(daysOfData) / 28.0)

        return BootstrapProgress(
            phase: currentPhase,
            daysOfData: daysOfData,
            progressInPhase: progressInPhase,
            overallProgress: overallProgress,
            daysUntilPersonalized: max(0, 28 - daysOfData),
            syntheticWeight: syntheticWeight,
            personalWeight: personalWeight,
            hasPersonalizedModel: hasPersonalizedModel
        )
    }

    // MARK: - Transition Recommendations

    /// Get actionable recommendations for current phase
    func getPhaseRecommendations() -> [String] {
        switch currentPhase {
        case .initializing:
            return [
                "Complete your onboarding to start collecting data",
                "Enable HealthKit for automatic biometric tracking"
            ]

        case .fullySynthetic:
            return [
                "Log daily symptoms to build your baseline",
                "Aim for 1-2 check-ins per day",
                "The model will start personalizing after 7 days"
            ]

        case .earlyBlending:
            return [
                "Great progress! Model is learning your patterns",
                "Continue daily logging for best accuracy",
                "Personalization: \(String(format: "%.0f%%", personalWeight * 100))"
            ]

        case .midBlending:
            return [
                "Model is well-adapted to your patterns",
                "Predictions are now 60% personalized to you",
                "Keep logging to reach full personalization"
            ]

        case .lateBlending:
            return [
                "Almost there! Model is 90% personalized",
                "Just \(28 - daysOfData) more days until full personalization",
                "Your unique patterns are well-understood"
            ]

        case .personalized:
            return [
                "Fully personalized! Model knows your unique triggers",
                "Continue daily logging to maintain accuracy",
                "Model updates weekly with your latest data"
            ]
        }
    }
}

// MARK: - Result Types

struct BootstrapHybridPrediction {
    let probability: Float               // Blended prediction
    let confidence: Float                // Adjusted confidence
    let syntheticProbability: Float      // Raw synthetic prediction
    let personalProbability: Float?      // Raw personal prediction (if available)
    let phase: BootstrapStrategy.BootstrapPhase
    let syntheticWeight: Float
    let personalWeight: Float

    var willFlare: Bool {
        return probability >= 0.5
    }

    var riskLevel: String {
        if probability >= 0.7 {
            return "High"
        } else if probability >= 0.5 {
            return "Moderate"
        } else if probability >= 0.3 {
            return "Low"
        } else {
            return "Very Low"
        }
    }

    var blendingDescription: String {
        if personalWeight == 0 {
            return "Using baseline patterns (no personal data yet)"
        } else if personalWeight == 1.0 {
            return "100% personalized to your unique patterns"
        } else {
            return "\(String(format: "%.0f%%", personalWeight * 100)) personalized to you"
        }
    }
}

struct BootstrapProgress {
    let phase: BootstrapStrategy.BootstrapPhase
    let daysOfData: Int
    let progressInPhase: Float        // 0-1 progress within current phase
    let overallProgress: Float        // 0-1 progress to day 28
    let daysUntilPersonalized: Int
    let syntheticWeight: Float
    let personalWeight: Float
    let hasPersonalizedModel: Bool

    var statusMessage: String {
        if phase == .personalized {
            return "Fully personalized (\(daysOfData) days of data)"
        } else {
            return "\(daysOfData)/28 days logged - \(phase.description)"
        }
    }

    var nextMilestone: String {
        switch phase {
        case .initializing, .fullySynthetic:
            return "First personalization at day 8"
        case .earlyBlending:
            return "60% personalized at day 15"
        case .midBlending:
            return "90% personalized at day 22"
        case .lateBlending:
            return "Fully personalized at day 28"
        case .personalized:
            return "Continuous learning active"
        }
    }
}
