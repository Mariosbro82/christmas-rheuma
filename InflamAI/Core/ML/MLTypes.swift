//
//  MLTypes.swift
//  InflamAI
//
//  Shared type definitions for ML components
//  This file consolidates all common types to avoid duplication
//
//  ============================================================================
//  TYPE CONSOLIDATION GUIDE
//  ============================================================================
//
//  This file contains the canonical ML-prefixed type definitions that should
//  be used for all NEW code. Legacy types exist in other files for backwards
//  compatibility but new code should use these types.
//
//  Preferred Types (use these):
//  ┌─────────────────────────┬─────────────────────────────────────────────┐
//  │ MLRiskLevel             │ Risk level for predictions (low → critical) │
//  │ MLConfidenceLevel       │ Prediction confidence (low → veryHigh)      │
//  │ MLEngineStatus          │ ML engine operational status                │
//  │ MLPersonalizationPhase  │ Model personalization progress              │
//  │ MLContributingFactor    │ Factor contributing to flare risk           │
//  │ MLRecommendedAction     │ Action based on risk level                  │
//  │ MLPredictionSource      │ Source of prediction (neural/statistical)   │
//  │ MLCalibrationMetrics    │ Model calibration quality                   │
//  │ MLHybridPrediction      │ Combined prediction result                  │
//  └─────────────────────────┴─────────────────────────────────────────────┘
//
//  Legacy Types (for backwards compatibility - avoid in new code):
//  - RiskLevel (UnifiedNeuralEngine.swift) → Use MLRiskLevel
//  - HybridRiskLevel (MLPredictionService.swift) → Use MLRiskLevel
//  - FlarePredictorRiskLevel (FlarePredictor.swift) → Use MLRiskLevel
//  - ConfidenceLevel (UnifiedNeuralEngine.swift) → Use MLConfidenceLevel
//  - ContributingFactor (UnifiedNeuralEngine.swift) → Use MLContributingFactor
//  - PredictionFactor (MLPredictionService.swift) → Use MLContributingFactor
//
//  Migration: When refactoring legacy code, replace legacy types with
//  ML-prefixed types and use the conversion helpers provided below.
//
//  ============================================================================

import Foundation
import SwiftUI

// MARK: - Risk Levels

/// Unified risk level for flare predictions
public enum MLRiskLevel: String, CaseIterable, Codable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case veryHigh = "Very High"
    case critical = "Critical"
    case unknown = "Unknown"

    public var icon: String {
        switch self {
        case .low: return "checkmark.circle.fill"
        case .moderate: return "exclamationmark.circle.fill"
        case .high: return "exclamationmark.triangle.fill"
        case .veryHigh, .critical: return "flame.fill"
        case .unknown: return "questionmark.circle.fill"
        }
    }

    public var color: Color {
        switch self {
        case .low: return .green
        case .moderate: return .yellow
        case .high: return .orange
        case .veryHigh, .critical: return .red
        case .unknown: return .gray
        }
    }

    public var emoji: String {
        switch self {
        case .low: return "✅"
        case .moderate: return "⚠️"
        case .high: return "🔶"
        case .veryHigh, .critical: return "🚨"
        case .unknown: return "❓"
        }
    }

    public static func from(percentage: Double) -> MLRiskLevel {
        switch percentage {
        case 0..<20: return .low
        case 20..<40: return .moderate
        case 40..<70: return .high
        case 70..<90: return .veryHigh
        case 90...100: return .critical
        default: return .unknown
        }
    }

    public static func from(probability: Float) -> MLRiskLevel {
        return from(percentage: Double(probability * 100))
    }
}

// MARK: - Confidence Levels

/// Confidence level for predictions
public enum MLConfidenceLevel: String, CaseIterable, Codable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case veryHigh = "Very High"

    public var color: Color {
        switch self {
        case .low: return .red
        case .moderate: return .orange
        case .high: return .yellow
        case .veryHigh: return .green
        }
    }

    public static func from(probability: Float) -> MLConfidenceLevel {
        let distance = abs(probability - 0.5)
        switch distance {
        case 0..<0.15: return .low
        case 0.15..<0.25: return .moderate
        case 0.25..<0.35: return .high
        default: return .veryHigh
        }
    }

    public static func from(dataQuality: Float) -> MLConfidenceLevel {
        switch dataQuality {
        case 0..<0.25: return .low
        case 0.25..<0.5: return .moderate
        case 0.5..<0.75: return .high
        default: return .veryHigh
        }
    }
}

// MARK: - Engine Status

/// Status of ML engine
public enum MLEngineStatus: Equatable {
    case initializing
    case ready
    case predicting
    case learning
    case error(String)

    public var displayMessage: String {
        switch self {
        case .initializing: return "Initializing Neural Engine..."
        case .ready: return "Neural Engine Ready"
        case .predicting: return "Making prediction..."
        case .learning: return "Learning your patterns..."
        case .error(let msg): return "Error: \(msg)"
        }
    }

    public var isOperational: Bool {
        switch self {
        case .ready, .predicting, .learning: return true
        default: return false
        }
    }
}

// MARK: - Personalization Phase

/// Phase of model personalization
public enum MLPersonalizationPhase: String, CaseIterable, Codable {
    case bootstrap = "Bootstrap"
    case earlyAdaptation = "Early Adaptation"
    case personalized = "Personalized"
    case expert = "Expert"

    public var daysRequired: Int {
        switch self {
        case .bootstrap: return 0
        case .earlyAdaptation: return 8
        case .personalized: return 22
        case .expert: return 90
        }
    }

    public var description: String {
        switch self {
        case .bootstrap:
            return "Collecting your baseline patterns"
        case .earlyAdaptation:
            return "Model is adapting to your unique patterns"
        case .personalized:
            return "Model knows your personal triggers"
        case .expert:
            return "Deep understanding of your condition"
        }
    }

    public static func from(daysOfData: Int) -> MLPersonalizationPhase {
        switch daysOfData {
        case 0..<8: return .bootstrap
        case 8..<22: return .earlyAdaptation
        case 22..<90: return .personalized
        default: return .expert
        }
    }
}

// MARK: - Contributing Factor

/// A factor that contributes to flare risk
public struct MLContributingFactor: Identifiable, Codable {
    public let id: UUID
    public let name: String
    public let impact: MLFactorImpact
    public let value: Double
    public let recommendation: String
    public let trend: MLFactorTrend
    public let category: MLFactorCategory

    public init(id: UUID = UUID(), name: String, impact: MLFactorImpact, value: Double, recommendation: String, trend: MLFactorTrend = .stable, category: MLFactorCategory = .other) {
        self.id = id
        self.name = name
        self.impact = impact
        self.value = value
        self.recommendation = recommendation
        self.trend = trend
        self.category = category
    }

    public enum MLFactorCategory: String, Codable, CaseIterable {
        case weather = "Weather"
        case activity = "Activity"
        case sleep = "Sleep"
        case medication = "Medication"
        case stress = "Stress"
        case other = "Other"
    }

    public enum MLFactorTrend: String, Codable, CaseIterable {
        case increasing
        case stable
        case decreasing
    }

    public enum MLFactorImpact: Int, Codable, CaseIterable {
        case low = 1
        case medium = 2
        case high = 3

        public var color: Color {
            switch self {
            case .low: return .blue
            case .medium: return .orange
            case .high: return .red
            }
        }

        public var label: String {
            switch self {
            case .low: return "Low"
            case .medium: return "Medium"
            case .high: return "High"
            }
        }
    }
}

// MARK: - Recommended Action

/// Recommended action based on prediction
public enum MLRecommendedAction: String, CaseIterable, Codable {
    case continueRoutine = "Continue your current routine"
    case monitorSymptoms = "Monitor symptoms closely"
    case preventiveMeasures = "Consider preventive measures"
    case immediatePrecautions = "Take immediate precautions"

    public static func from(riskLevel: MLRiskLevel) -> MLRecommendedAction {
        switch riskLevel {
        case .low, .unknown: return .continueRoutine
        case .moderate: return .monitorSymptoms
        case .high: return .preventiveMeasures
        case .veryHigh, .critical: return .immediatePrecautions
        }
    }
}

// MARK: - Service Status

/// Status for ML services
public enum MLServiceStatus: Equatable {
    case idle
    case loading
    case ready
    case predicting
    case error(String)

    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

// MARK: - Prediction Source

/// Source of a prediction
public enum MLPredictionSource: String, CaseIterable, Codable {
    case neuralEngine = "Neural Engine"
    case statistical = "Statistical"
    case hybrid = "Hybrid"
    case unknown = "Unknown"
}

// MARK: - Calibration Metrics

/// Metrics for model calibration quality
public struct MLCalibrationMetrics: Codable {
    public let expectedCalibrationError: Float
    public let maxCalibrationError: Float
    public let brierScore: Float
    public let reliability: Float
    public let resolution: Float
    public let timestamp: Date

    public init(
        expectedCalibrationError: Float,
        maxCalibrationError: Float,
        brierScore: Float,
        reliability: Float,
        resolution: Float,
        timestamp: Date = Date()
    ) {
        self.expectedCalibrationError = expectedCalibrationError
        self.maxCalibrationError = maxCalibrationError
        self.brierScore = brierScore
        self.reliability = reliability
        self.resolution = resolution
        self.timestamp = timestamp
    }

    public var isWellCalibrated: Bool {
        expectedCalibrationError < 0.1 && brierScore < 0.25
    }

    public var qualityRating: String {
        if expectedCalibrationError < 0.05 { return "Excellent" }
        if expectedCalibrationError < 0.1 { return "Good" }
        if expectedCalibrationError < 0.15 { return "Fair" }
        return "Needs Improvement"
    }
}

// MARK: - Hybrid Prediction

/// Combined prediction from multiple sources
public struct MLHybridPrediction: Codable {
    public let probability: Float
    public let confidence: MLConfidenceLevel
    public let riskLevel: MLRiskLevel
    public let primarySource: MLPredictionSource
    public let neuralProbability: Float?
    public let statisticalProbability: Float?
    public let dataQuality: Float
    public let timestamp: Date
    public let contributingFactors: [MLContributingFactor]

    public init(
        probability: Float,
        confidence: MLConfidenceLevel,
        riskLevel: MLRiskLevel,
        primarySource: MLPredictionSource,
        neuralProbability: Float? = nil,
        statisticalProbability: Float? = nil,
        dataQuality: Float,
        timestamp: Date = Date(),
        contributingFactors: [MLContributingFactor] = []
    ) {
        self.probability = probability
        self.confidence = confidence
        self.riskLevel = riskLevel
        self.primarySource = primarySource
        self.neuralProbability = neuralProbability
        self.statisticalProbability = statisticalProbability
        self.dataQuality = dataQuality
        self.timestamp = timestamp
        self.contributingFactors = contributingFactors
    }

    public var willFlare: Bool {
        probability >= 0.5
    }

    public var percentageRisk: Int {
        Int(probability * 100)
    }
}

// MARK: - Data Readiness

/// Status of data readiness for ML
public struct MLDataReadinessStatus: Codable {
    public let isReady: Bool
    public let daysAvailable: Int
    public let daysRequired: Int
    public let message: String
    public let healthKitAvailable: Bool
    public let coreDataAvailable: Bool

    public init(
        isReady: Bool,
        daysAvailable: Int,
        daysRequired: Int,
        message: String,
        healthKitAvailable: Bool = true,
        coreDataAvailable: Bool = true
    ) {
        self.isReady = isReady
        self.daysAvailable = daysAvailable
        self.daysRequired = daysRequired
        self.message = message
        self.healthKitAvailable = healthKitAvailable
        self.coreDataAvailable = coreDataAvailable
    }

    public var progress: Float {
        guard daysRequired > 0 else { return 1.0 }
        return min(1.0, Float(daysAvailable) / Float(daysRequired))
    }
}

// MARK: - Learning Pipeline Status

/// Status of the continuous learning pipeline
public struct MLLearningPipelineStatus: Codable {
    public let phase: MLPersonalizationPhase
    public let progress: Float
    public let totalSamples: Int
    public let lastUpdateDate: Date?
    public let modelVersion: Int
    public let isPersonalized: Bool

    public init(
        phase: MLPersonalizationPhase,
        progress: Float,
        totalSamples: Int,
        lastUpdateDate: Date?,
        modelVersion: Int,
        isPersonalized: Bool
    ) {
        self.phase = phase
        self.progress = progress
        self.totalSamples = totalSamples
        self.lastUpdateDate = lastUpdateDate
        self.modelVersion = modelVersion
        self.isPersonalized = isPersonalized
    }
}

// MARK: - Backtest Result

/// Result of backtesting predictions against outcomes
public struct MLBacktestResult: Codable {
    public let accuracy: Float
    public let precision: Float
    public let recall: Float
    public let f1Score: Float
    public let areaUnderROC: Float
    public let brierScore: Float
    public let totalPredictions: Int
    public let truePositives: Int
    public let falsePositives: Int
    public let trueNegatives: Int
    public let falseNegatives: Int
    public let timestamp: Date

    public init(
        accuracy: Float,
        precision: Float,
        recall: Float,
        f1Score: Float,
        areaUnderROC: Float,
        brierScore: Float,
        totalPredictions: Int,
        truePositives: Int,
        falsePositives: Int,
        trueNegatives: Int,
        falseNegatives: Int,
        timestamp: Date = Date()
    ) {
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1Score = f1Score
        self.areaUnderROC = areaUnderROC
        self.brierScore = brierScore
        self.totalPredictions = totalPredictions
        self.truePositives = truePositives
        self.falsePositives = falsePositives
        self.trueNegatives = trueNegatives
        self.falseNegatives = falseNegatives
        self.timestamp = timestamp
    }

    public var isReliable: Bool {
        accuracy >= 0.7 && totalPredictions >= 10
    }
}

// MARK: - Type Aliases for Compatibility

// NOTE: Aliases commented out - use ML-prefixed types directly in new code:
// - MLRiskLevel, MLConfidenceLevel, MLEngineStatus, etc.
// - MLContributingFactor, MLPredictionSource, MLCalibrationMetrics, etc.
// These aliases conflict with types defined in FlarePredictor.swift and other files.
// public typealias ContributingFactor = MLContributingFactor
// public typealias PredictionSourceType = MLPredictionSource
// public typealias CalibrationMetrics = MLCalibrationMetrics
// public typealias HybridPrediction = MLHybridPrediction
// public typealias DataReadinessStatus = MLDataReadinessStatus
// public typealias LearningPipelineStatus = MLLearningPipelineStatus
// public typealias BacktestResult = MLBacktestResult
