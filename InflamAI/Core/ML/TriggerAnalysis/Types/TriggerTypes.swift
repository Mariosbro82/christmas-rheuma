//
//  TriggerTypes.swift
//  InflamAI
//
//  Core type definitions for the Hybrid Trigger Detection System
//  Supports statistical, k-NN, and neural network analysis
//

import Foundation
import SwiftUI

// MARK: - Trigger Categories

/// Categories of triggers that can affect AS symptoms
public enum TriggerCategory: String, CaseIterable, Codable, Identifiable {
    case food = "food"
    case sleep = "sleep"
    case activity = "activity"
    case weather = "weather"
    case stress = "stress"
    case medication = "medication"
    case other = "other"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .food: return "Food & Drink"
        case .sleep: return "Sleep"
        case .activity: return "Physical Activity"
        case .weather: return "Weather"
        case .stress: return "Stress & Mental"
        case .medication: return "Medication"
        case .other: return "Other"
        }
    }

    public var icon: String {
        switch self {
        case .food: return "fork.knife"
        case .sleep: return "bed.double.fill"
        case .activity: return "figure.run"
        case .weather: return "cloud.sun.fill"
        case .stress: return "brain.head.profile"
        case .medication: return "pills.fill"
        case .other: return "ellipsis.circle"
        }
    }

    public var color: Color {
        switch self {
        case .food: return .orange
        case .sleep: return .indigo
        case .activity: return .green
        case .weather: return .cyan
        case .stress: return .purple
        case .medication: return .blue
        case .other: return .gray
        }
    }
}

// MARK: - Trigger Confidence Levels

/// Confidence level in a trigger's effect
public enum TriggerConfidence: String, Codable, CaseIterable {
    case high = "high"           // n >= 60, p < 0.01, |r| > 0.5, |d| > 0.5
    case medium = "medium"       // n >= 30, p < 0.05, |r| > 0.3
    case low = "low"             // Significant but weak
    case insufficient = "insufficient"  // Not enough data

    public var displayName: String {
        switch self {
        case .high: return "High Confidence"
        case .medium: return "Moderate Confidence"
        case .low: return "Low Confidence"
        case .insufficient: return "Insufficient Data"
        }
    }

    public var color: Color {
        switch self {
        case .high: return .green
        case .medium: return .yellow
        case .low: return .orange
        case .insufficient: return .gray
        }
    }

    public var icon: String {
        switch self {
        case .high: return "checkmark.seal.fill"
        case .medium: return "exclamationmark.triangle.fill"
        case .low: return "questionmark.circle.fill"
        case .insufficient: return "ellipsis.circle"
        }
    }

    public var sortOrder: Int {
        switch self {
        case .high: return 0
        case .medium: return 1
        case .low: return 2
        case .insufficient: return 3
        }
    }
}

// MARK: - Data Source

/// Source of trigger data
public enum TriggerDataSource: String, Codable {
    case manual = "manual"           // User-logged
    case healthKit = "healthkit"     // From HealthKit
    case weather = "weather"         // From OpenMeteo
    case medication = "medication"   // From medication tracking
    case inferred = "inferred"       // Derived from other data
}

// MARK: - Trigger Definition

/// Definition of a trackable trigger
public struct TriggerDefinition: Identifiable, Codable, Hashable {
    public let id: String
    public let name: String
    public let category: TriggerCategory
    public let icon: String
    public let unit: String?
    public let isBinary: Bool
    public let minValue: Double
    public let maxValue: Double
    public let defaultValue: Double
    public let expectedLagHours: Int
    public let dataSource: TriggerDataSource
    public let clinicalRelevance: String
    public let trackingPrompt: String

    public init(
        id: String,
        name: String,
        category: TriggerCategory,
        icon: String,
        unit: String? = nil,
        isBinary: Bool = false,
        minValue: Double = 0,
        maxValue: Double = 10,
        defaultValue: Double = 0,
        expectedLagHours: Int = 24,
        dataSource: TriggerDataSource = .manual,
        clinicalRelevance: String,
        trackingPrompt: String = ""
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.icon = icon
        self.unit = unit
        self.isBinary = isBinary
        self.minValue = minValue
        self.maxValue = maxValue
        self.defaultValue = defaultValue
        self.expectedLagHours = expectedLagHours
        self.dataSource = dataSource
        self.clinicalRelevance = clinicalRelevance
        self.trackingPrompt = trackingPrompt.isEmpty ? "How much \(name.lowercased()) today?" : trackingPrompt
    }

    public var expectedLagDescription: String {
        switch expectedLagHours {
        case 0: return "Same day effect"
        case 1..<12: return "Within hours"
        case 12..<24: return "Next day effect"
        case 24..<48: return "1-2 day delay"
        case 48..<72: return "2-3 day delay"
        default: return "\(expectedLagHours / 24)+ day delay"
        }
    }
}

// MARK: - Trigger Value (logged instance)

/// A logged trigger value
public struct TriggerValue: Codable, Identifiable, Hashable {
    public let id: UUID
    public let name: String
    public let category: TriggerCategory
    public let value: Double
    public let unit: String?
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        name: String,
        category: TriggerCategory,
        value: Double,
        unit: String? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.value = value
        self.unit = unit
        self.timestamp = timestamp
    }

    public var isPresent: Bool {
        value > 0
    }

    public var displayValue: String {
        if let unit = unit {
            return "\(Int(value)) \(unit)"
        } else {
            return String(format: "%.1f", value)
        }
    }
}

// MARK: - Lagged Correlation Result

/// Result of correlation analysis at a specific lag
public struct LaggedCorrelationResult: Codable, Identifiable {
    public let id: UUID
    public let lag: Int  // Days
    public let correlation: Double
    public let pValue: Double
    public let sampleSize: Int

    public init(
        id: UUID = UUID(),
        lag: Int,
        correlation: Double,
        pValue: Double,
        sampleSize: Int
    ) {
        self.id = id
        self.lag = lag
        self.correlation = correlation
        self.pValue = pValue
        self.sampleSize = sampleSize
    }

    public var lagDescription: String {
        switch lag {
        case 0: return "Same day"
        case 1: return "Next day"
        case 2: return "2 days later"
        case 3: return "3 days later"
        default: return "\(lag) days later"
        }
    }

    public var isSignificant: Bool {
        pValue < 0.05
    }

    public var strengthDescription: String {
        let absR = abs(correlation)
        if absR >= 0.7 { return "Strong" }
        if absR >= 0.5 { return "Moderate" }
        if absR >= 0.3 { return "Weak" }
        return "Negligible"
    }

    public var direction: String {
        correlation > 0 ? "increases" : "decreases"
    }
}

// MARK: - Effect Size

/// Clinical effect size metrics
public struct EffectSize: Codable {
    public let meanWithTrigger: Double
    public let meanWithoutTrigger: Double
    public let meanDifference: Double
    public let pooledStandardDeviation: Double
    public let cohenD: Double
    public let percentChange: Double

    public init(
        meanWithTrigger: Double,
        meanWithoutTrigger: Double,
        meanDifference: Double,
        pooledStandardDeviation: Double,
        cohenD: Double,
        percentChange: Double
    ) {
        self.meanWithTrigger = meanWithTrigger
        self.meanWithoutTrigger = meanWithoutTrigger
        self.meanDifference = meanDifference
        self.pooledStandardDeviation = pooledStandardDeviation
        self.cohenD = cohenD
        self.percentChange = percentChange
    }

    public var clinicallySignificant: Bool {
        abs(cohenD) >= 0.5 && abs(meanDifference) >= 1.0
    }

    public var cohenDInterpretation: String {
        let absD = abs(cohenD)
        if absD >= 0.8 { return "Large effect" }
        if absD >= 0.5 { return "Medium effect" }
        if absD >= 0.2 { return "Small effect" }
        return "Negligible effect"
    }

    public var impactDescription: String {
        let direction = meanDifference > 0 ? "increases" : "decreases"
        return "\(direction) pain by \(String(format: "%.1f", abs(meanDifference))) points"
    }
}

// MARK: - Statistical Trigger Result

/// Complete statistical analysis result for a trigger
public struct StatisticalTriggerResult: Identifiable, Codable {
    public let id: UUID
    public let triggerName: String
    public let triggerCategory: TriggerCategory
    public let icon: String

    // Sample info
    public let totalDays: Int
    public let triggerDays: Int
    public let nonTriggerDays: Int

    // Lagged correlations
    public let laggedResults: [LaggedCorrelationResult]
    public let bestLag: LaggedCorrelationResult?

    // Effect size
    public let effectSize: EffectSize

    // Statistical significance
    public let rawPValue: Double
    public let correctedPValue: Double
    public let isSignificant: Bool

    // Classification
    public let confidence: TriggerConfidence

    // Timestamp
    public let analysisDate: Date

    public init(
        id: UUID = UUID(),
        triggerName: String,
        triggerCategory: TriggerCategory,
        icon: String,
        totalDays: Int,
        triggerDays: Int,
        nonTriggerDays: Int,
        laggedResults: [LaggedCorrelationResult],
        bestLag: LaggedCorrelationResult?,
        effectSize: EffectSize,
        rawPValue: Double,
        correctedPValue: Double,
        isSignificant: Bool,
        confidence: TriggerConfidence,
        analysisDate: Date = Date()
    ) {
        self.id = id
        self.triggerName = triggerName
        self.triggerCategory = triggerCategory
        self.icon = icon
        self.totalDays = totalDays
        self.triggerDays = triggerDays
        self.nonTriggerDays = nonTriggerDays
        self.laggedResults = laggedResults
        self.bestLag = bestLag
        self.effectSize = effectSize
        self.rawPValue = rawPValue
        self.correctedPValue = correctedPValue
        self.isSignificant = isSignificant
        self.confidence = confidence
        self.analysisDate = analysisDate
    }

    public var impactDescription: String {
        guard let bestLag = bestLag else {
            return "Insufficient data to determine effect"
        }

        let direction = effectSize.meanDifference > 0 ? "increases" : "decreases"
        let timing = bestLag.lagDescription.lowercased()

        return "\(triggerName) \(direction) pain by \(String(format: "%.1f", abs(effectSize.meanDifference))) points (\(timing))"
    }

    public var confidenceExplanation: String {
        switch confidence {
        case .high:
            return "Strong evidence from \(totalDays) days of data (p=\(String(format: "%.3f", correctedPValue)))"
        case .medium:
            return "Moderate evidence - track \(max(0, 60 - totalDays)) more days to strengthen"
        case .low:
            return "Weak evidence - early indication only"
        case .insufficient:
            return "Need \(max(0, 14 - triggerDays)) more days with this trigger"
        }
    }
}

// MARK: - Engine Types

/// Analysis engine types
public enum EngineType: String, CaseIterable, Codable {
    case statistical = "statistical"
    case knn = "knn"
    case neural = "neural"

    public var displayName: String {
        switch self {
        case .statistical: return "Statistical Analysis"
        case .knn: return "Similar Days (k-NN)"
        case .neural: return "Neural Network"
        }
    }

    public var icon: String {
        switch self {
        case .statistical: return "chart.bar.xaxis"
        case .knn: return "person.3.fill"
        case .neural: return "brain.head.profile"
        }
    }

    public var minimumDays: Int {
        switch self {
        case .statistical: return 7
        case .knn: return 30
        case .neural: return 90
        }
    }

    public var description: String {
        switch self {
        case .statistical:
            return "Correlation analysis with statistical significance testing"
        case .knn:
            return "Finds similar historical days to predict outcomes"
        case .neural:
            return "Deep learning for complex pattern detection"
        }
    }
}

// MARK: - Activation Phase

/// Progressive engine activation phases based on active engines
public enum ActivationPhase: String, CaseIterable, Codable {
    case statistical = "statistical"       // 7-29 days: Statistical only
    case knn = "knn"                       // 30-89 days: Statistical + k-NN
    case neural = "neural"                 // 90+ days: All engines (opt-in)

    public var displayName: String {
        switch self {
        case .statistical: return "Statistical Analysis"
        case .knn: return "Pattern Matching"
        case .neural: return "Neural Network"
        }
    }

    public var description: String {
        switch self {
        case .statistical:
            return "Identifying correlations with statistical testing"
        case .knn:
            return "Finding similar days to predict outcomes"
        case .neural:
            return "Deep pattern analysis with neural networks"
        }
    }

    public var minimumDays: Int {
        switch self {
        case .statistical: return 7
        case .knn: return 30
        case .neural: return 90
        }
    }

    public var activeEngines: [EngineType] {
        switch self {
        case .statistical: return [.statistical]
        case .knn: return [.statistical, .knn]
        case .neural: return [.statistical, .knn, .neural]
        }
    }

    public static func forDays(_ days: Int, neuralOptIn: Bool = false) -> ActivationPhase {
        if neuralOptIn && days >= 90 {
            return .neural
        } else if days >= 30 {
            return .knn
        } else {
            return .statistical
        }
    }
}

// MARK: - Similar Day (for k-NN)

/// A similar historical day from k-NN analysis
public struct SimilarDay: Identifiable, Codable {
    public let id: UUID
    public let date: Date
    public let painLevel: Double
    public let distance: Double
    public let triggers: [TriggerValue]
    public let keyFeatures: [String: Double]

    public init(
        id: UUID = UUID(),
        date: Date,
        painLevel: Double,
        distance: Double,
        triggers: [TriggerValue],
        keyFeatures: [String: Double]
    ) {
        self.id = id
        self.date = date
        self.painLevel = painLevel
        self.distance = distance
        self.triggers = triggers
        self.keyFeatures = keyFeatures
    }

    public var similarityScore: Double {
        max(0, 100 - distance * 10)
    }

    public var similarityDescription: String {
        String(format: "%.0f%% similar", similarityScore)
    }
}

// MARK: - Common Trigger (from k-NN)

/// A trigger commonly found among similar days
public struct CommonTrigger: Identifiable, Codable {
    public let id: UUID
    public let name: String
    public let frequency: Double  // 0-1
    public let averageValue: Double

    public init(
        id: UUID = UUID(),
        name: String,
        frequency: Double,
        averageValue: Double
    ) {
        self.id = id
        self.name = name
        self.frequency = frequency
        self.averageValue = averageValue
    }

    public var frequencyDescription: String {
        "\(Int(frequency * 100))% of similar days"
    }
}

// MARK: - Feature Attribution (for Neural)

/// Feature importance from neural network
public struct FeatureAttribution: Identifiable, Codable {
    public let id: UUID
    public let feature: String
    public let attribution: Double
    public let direction: AttributionDirection

    public enum AttributionDirection: String, Codable {
        case increases
        case decreases
    }

    public init(
        id: UUID = UUID(),
        feature: String,
        attribution: Double,
        direction: AttributionDirection
    ) {
        self.id = id
        self.feature = feature
        self.attribution = attribution
        self.direction = direction
    }

    public var importance: Double {
        abs(attribution)
    }

    public var description: String {
        let verb = direction == .increases ? "increased" : "decreased"
        return "\(feature) \(verb) prediction by \(String(format: "%.1f", abs(attribution))) points"
    }
}

// MARK: - Recommendation

/// Recommendation type
public enum RecommendationType: String, Codable, CaseIterable {
    case avoid      // Reduce this trigger
    case encourage  // Continue/increase this
    case track      // Need more data
    case monitor    // Watch for changes

    public var displayName: String {
        switch self {
        case .avoid: return "Avoid"
        case .encourage: return "Encourage"
        case .track: return "Track"
        case .monitor: return "Monitor"
        }
    }

    public var icon: String {
        switch self {
        case .avoid: return "xmark.circle.fill"
        case .encourage: return "checkmark.circle.fill"
        case .track: return "chart.line.uptrend.xyaxis"
        case .monitor: return "eye.fill"
        }
    }

    public var color: Color {
        switch self {
        case .avoid: return .red
        case .encourage: return .green
        case .track: return .blue
        case .monitor: return .orange
        }
    }
}

/// Impact level of a recommendation
public enum RecommendationImpact: String, Codable, CaseIterable {
    case high
    case medium
    case low

    public var displayName: String {
        switch self {
        case .high: return "High Impact"
        case .medium: return "Medium Impact"
        case .low: return "Low Impact"
        }
    }

    public var sortOrder: Int {
        switch self {
        case .high: return 0
        case .medium: return 1
        case .low: return 2
        }
    }
}

/// Actionable recommendation based on trigger analysis
public struct TriggerRecommendation: Identifiable, Codable {
    public let id: UUID
    public let triggerName: String
    public let triggerCategory: TriggerCategory
    public let icon: String
    public let type: RecommendationType
    public let title: String
    public let description: String
    public let impact: RecommendationImpact
    public let confidence: TriggerConfidence
    public let actionable: Bool
    public let evidenceSummary: String

    public init(
        id: UUID = UUID(),
        triggerName: String,
        triggerCategory: TriggerCategory,
        icon: String,
        type: RecommendationType,
        title: String,
        description: String,
        impact: RecommendationImpact,
        confidence: TriggerConfidence,
        actionable: Bool,
        evidenceSummary: String
    ) {
        self.id = id
        self.triggerName = triggerName
        self.triggerCategory = triggerCategory
        self.icon = icon
        self.type = type
        self.title = title
        self.description = description
        self.impact = impact
        self.confidence = confidence
        self.actionable = actionable
        self.evidenceSummary = evidenceSummary
    }

    // Legacy initializer for backward compatibility
    public init(
        id: UUID = UUID(),
        trigger: String,
        action: String,
        expectedImpact: String,
        confidence: TriggerConfidence,
        timing: String
    ) {
        self.id = id
        self.triggerName = trigger
        self.triggerCategory = .other
        self.icon = "exclamationmark.triangle"
        self.type = .monitor
        self.title = action
        self.description = expectedImpact
        self.impact = .medium
        self.confidence = confidence
        self.actionable = true
        self.evidenceSummary = timing
    }
}

// MARK: - Array Extensions

extension Array where Element == Double {
    /// Calculate mean of array
    public func mean() -> Double {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / Double(count)
    }

    /// Calculate variance of array
    public func variance() -> Double {
        guard count > 1 else { return 0 }
        let m = mean()
        return map { pow($0 - m, 2) }.reduce(0, +) / Double(count - 1)
    }

    /// Calculate standard deviation of array
    public func standardDeviation() -> Double {
        sqrt(variance())
    }
}
