//
//  TriggerAnalysisService.swift
//  InflamAI
//
//  Main orchestrator for the Hybrid Trigger Detection System
//  Combines Statistical, k-NN, and Neural engines with ensemble weighting
//
//  Progressive activation:
//  - 7+ days:  Statistical only (transparent, explainable)
//  - 30+ days: Statistical + k-NN (personalized pattern matching)
//  - 90+ days: All engines (neural network for complex patterns)
//

import Foundation
import CoreData
import Combine

// MARK: - TriggerAnalysisService

@MainActor
public final class TriggerAnalysisService: ObservableObject {

    // MARK: - Singleton

    public static let shared = TriggerAnalysisService()

    // MARK: - Published State

    @Published public private(set) var currentPhase: ActivationPhase = .statistical
    @Published public private(set) var recommendations: [TriggerRecommendation] = []
    @Published public private(set) var topTriggers: [UnifiedTriggerResult] = []
    @Published public private(set) var isAnalyzing: Bool = false
    @Published public private(set) var lastAnalysisDate: Date?
    @Published public private(set) var daysOfData: Int = 0
    @Published public private(set) var errorMessage: String?

    // MARK: - Engine References

    private let statisticalEngine: StatisticalTriggerEngine
    private let knnEngine: KNNTriggerEngine
    private let neuralEngine: NeuralTriggerEngine
    private let triggerDataService: TriggerDataService

    // MARK: - Configuration

    public struct Configuration {
        /// Minimum days for statistical analysis
        public var statisticalMinDays: Int = 7

        /// Minimum days for k-NN activation
        public var knnMinDays: Int = 30

        /// Minimum days for neural network activation
        public var neuralMinDays: Int = 90

        /// Whether user opted into neural engine
        public var neuralOptIn: Bool = false

        /// Ensemble weights for each engine
        public var statisticalWeight: Double = 0.4
        public var knnWeight: Double = 0.35
        public var neuralWeight: Double = 0.25

        public static let `default` = Configuration()
    }

    public var configuration: Configuration

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    private init(
        persistenceController: InflamAIPersistenceController = .shared,
        statisticalEngine: StatisticalTriggerEngine = .shared,
        knnEngine: KNNTriggerEngine = .shared,
        neuralEngine: NeuralTriggerEngine = .shared,
        triggerDataService: TriggerDataService = .shared,
        configuration: Configuration = .default
    ) {
        self.persistenceController = persistenceController
        self.statisticalEngine = statisticalEngine
        self.knnEngine = knnEngine
        self.neuralEngine = neuralEngine
        self.triggerDataService = triggerDataService
        self.configuration = configuration

        // Initial phase determination
        Task {
            await updatePhase()
        }
    }

    // MARK: - Context

    private var viewContext: NSManagedObjectContext {
        persistenceController.container.viewContext
    }

    // MARK: - Phase Management

    /// Update current activation phase based on available data
    public func updatePhase() async {
        daysOfData = calculateDaysOfData()

        if configuration.neuralOptIn && daysOfData >= configuration.neuralMinDays {
            currentPhase = .neural
        } else if daysOfData >= configuration.knnMinDays {
            currentPhase = .knn
        } else if daysOfData >= configuration.statisticalMinDays {
            currentPhase = .statistical
        } else {
            currentPhase = .statistical // Will show "need more data" message
        }
    }

    /// Calculate total days with symptom data
    private func calculateDaysOfData() -> Int {
        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        let logs = (try? viewContext.fetch(request)) ?? []

        let calendar = Calendar.current
        let uniqueDays = Set(logs.compactMap { log in
            log.timestamp.map { calendar.startOfDay(for: $0) }
        })

        return uniqueDays.count
    }

    // MARK: - Main Analysis

    /// Run full trigger analysis with all active engines
    public func analyzeAllTriggers() async -> [UnifiedTriggerResult] {
        isAnalyzing = true
        errorMessage = nil
        defer { isAnalyzing = false }

        await updatePhase()

        // Check minimum data requirements
        guard daysOfData >= configuration.statisticalMinDays else {
            errorMessage = "Need at least \(configuration.statisticalMinDays) days of data (have \(daysOfData))"
            return []
        }

        // Run statistical analysis (always active)
        let statisticalResults = await statisticalEngine.analyzeAllTriggers(days: min(daysOfData, 90))

        // Train engines if needed
        if currentPhase == .knn || currentPhase == .neural {
            if !knnEngine.isReady {
                await knnEngine.train()
            }
        }

        if currentPhase == .neural {
            if !neuralEngine.isReady {
                await neuralEngine.train()
            }
        }

        // Convert to unified results with all engine outputs
        var unifiedResults = statisticalResults.map { result -> UnifiedTriggerResult in
            var knnResult: KNNTriggerResult? = nil
            var neuralResult: NeuralTriggerResult? = nil
            var activeEngines: [EngineType] = [.statistical]

            // Add k-NN analysis if active
            if (currentPhase == .knn || currentPhase == .neural) && knnEngine.isReady {
                if let knnAnalysis = knnEngine.analyzeTrigger(named: result.triggerName) {
                    knnResult = KNNTriggerResult(
                        triggerName: result.triggerName,
                        similarDays: knnAnalysis.similarHighPainDays,
                        predictedEffect: knnAnalysis.predictedImpact,
                        confidence: knnAnalysis.confidence
                    )
                    activeEngines.append(.knn)
                }
            }

            // Add neural analysis if active
            if currentPhase == .neural && neuralEngine.isReady {
                // Neural provides overall prediction, not per-trigger analysis
                // Use feature attribution to estimate trigger impact
                activeEngines.append(.neural)
            }

            // Calculate ensemble score
            let ensembleScore = calculateEnsembleScore(
                statistical: result,
                knn: knnResult,
                neural: neuralResult
            )

            // Determine confidence from best engine
            let ensembleConfidence = determineEnsembleConfidence(
                statistical: result.confidence,
                knn: knnResult?.confidence,
                neural: neuralResult?.confidence
            )

            // Determine primary engine
            let primaryEngine = determinePrimaryEngine(
                statistical: result,
                knn: knnResult,
                neural: neuralResult
            )

            return UnifiedTriggerResult(
                triggerName: result.triggerName,
                triggerCategory: result.triggerCategory,
                icon: result.icon,
                statisticalResult: result,
                knnResult: knnResult,
                neuralResult: neuralResult,
                ensembleScore: ensembleScore,
                ensembleConfidence: ensembleConfidence,
                primaryEngine: primaryEngine,
                activeEngines: activeEngines
            )
        }

        // Sort by ensemble score
        unifiedResults.sort { abs($0.ensembleScore) > abs($1.ensembleScore) }

        topTriggers = unifiedResults
        recommendations = generateRecommendations(from: unifiedResults)
        lastAnalysisDate = Date()

        return unifiedResults
    }

    /// Analyze a specific trigger
    public func analyzeTrigger(named triggerName: String) async -> UnifiedTriggerResult? {
        guard let statisticalResult = await statisticalEngine.analyzeTrigger(name: triggerName) else {
            return nil
        }

        return UnifiedTriggerResult(
            triggerName: statisticalResult.triggerName,
            triggerCategory: statisticalResult.triggerCategory,
            icon: statisticalResult.icon,
            statisticalResult: statisticalResult,
            knnResult: nil,
            neuralResult: nil,
            ensembleScore: statisticalResult.effectSize.cohenD,
            ensembleConfidence: statisticalResult.confidence,
            primaryEngine: .statistical,
            activeEngines: [.statistical]
        )
    }

    // MARK: - Recommendations

    /// Generate actionable recommendations from analysis results
    private func generateRecommendations(from results: [UnifiedTriggerResult]) -> [TriggerRecommendation] {
        var recommendations: [TriggerRecommendation] = []

        // Get significant positive triggers (worsen symptoms)
        let significantHarmful = results
            .filter { $0.statisticalResult?.isSignificant == true }
            .filter { ($0.statisticalResult?.effectSize.meanDifference ?? 0) > 0 }
            .prefix(3)

        for result in significantHarmful {
            if let statistical = result.statisticalResult {
                let explanation = statisticalEngine.generateExplanation(for: statistical)

                recommendations.append(TriggerRecommendation(
                    id: UUID(),
                    triggerName: result.triggerName,
                    triggerCategory: result.triggerCategory,
                    icon: result.icon,
                    type: .avoid,
                    title: "Consider reducing \(result.triggerName)",
                    description: explanation.summary,
                    impact: RecommendationImpact.high,
                    confidence: result.ensembleConfidence,
                    actionable: true,
                    evidenceSummary: "Based on \(statistical.totalDays) days of data"
                ))
            }
        }

        // Get significant negative triggers (protective)
        let significantProtective = results
            .filter { $0.statisticalResult?.isSignificant == true }
            .filter { ($0.statisticalResult?.effectSize.meanDifference ?? 0) < 0 }
            .prefix(2)

        for result in significantProtective {
            if let statistical = result.statisticalResult {
                recommendations.append(TriggerRecommendation(
                    id: UUID(),
                    triggerName: result.triggerName,
                    triggerCategory: result.triggerCategory,
                    icon: result.icon,
                    type: .encourage,
                    title: "Continue \(result.triggerName)",
                    description: "This appears to help reduce your symptoms.",
                    impact: RecommendationImpact.medium,
                    confidence: result.ensembleConfidence,
                    actionable: true,
                    evidenceSummary: "Associated with \(String(format: "%.1f", abs(statistical.effectSize.meanDifference))) point lower pain"
                ))
            }
        }

        // Add data collection recommendations for low-data triggers
        let needsMoreData = results
            .filter { $0.ensembleConfidence == .insufficient || $0.ensembleConfidence == .low }
            .prefix(2)

        for result in needsMoreData {
            recommendations.append(TriggerRecommendation(
                id: UUID(),
                triggerName: result.triggerName,
                triggerCategory: result.triggerCategory,
                icon: result.icon,
                type: .track,
                title: "Keep tracking \(result.triggerName)",
                description: "More data needed for reliable analysis.",
                impact: RecommendationImpact.low,
                confidence: .insufficient,
                actionable: false,
                evidenceSummary: "Currently \(result.statisticalResult?.triggerDays ?? 0) days tracked"
            ))
        }

        return recommendations
    }

    // MARK: - Quick Insights

    /// Get a quick summary of trigger status
    public func getQuickInsights() async -> TriggerInsightsSummary {
        await updatePhase()

        let significantCount = topTriggers.filter { $0.statisticalResult?.isSignificant == true }.count
        let harmfulCount = topTriggers.filter {
            ($0.statisticalResult?.isSignificant == true) &&
            ($0.statisticalResult?.effectSize.meanDifference ?? 0) > 0
        }.count
        let protectiveCount = topTriggers.filter {
            ($0.statisticalResult?.isSignificant == true) &&
            ($0.statisticalResult?.effectSize.meanDifference ?? 0) < 0
        }.count

        let topHarmful = topTriggers
            .filter { ($0.statisticalResult?.effectSize.meanDifference ?? 0) > 0 }
            .first

        let topProtective = topTriggers
            .filter { ($0.statisticalResult?.effectSize.meanDifference ?? 0) < 0 }
            .first

        return TriggerInsightsSummary(
            totalTriggersTracked: triggerDataService.getLoggedTriggerNames().count,
            daysOfData: daysOfData,
            significantTriggers: significantCount,
            harmfulTriggers: harmfulCount,
            protectiveTriggers: protectiveCount,
            currentPhase: currentPhase,
            topHarmfulTrigger: topHarmful?.triggerName,
            topProtectiveTrigger: topProtective?.triggerName,
            lastAnalysisDate: lastAnalysisDate,
            nextPhaseAt: getNextPhaseThreshold(),
            daysUntilNextPhase: getDaysUntilNextPhase()
        )
    }

    private func getNextPhaseThreshold() -> Int? {
        switch currentPhase {
        case .statistical:
            return configuration.knnMinDays
        case .knn:
            return configuration.neuralOptIn ? configuration.neuralMinDays : nil
        case .neural:
            return nil
        }
    }

    private func getDaysUntilNextPhase() -> Int? {
        guard let threshold = getNextPhaseThreshold() else { return nil }
        let remaining = threshold - daysOfData
        return remaining > 0 ? remaining : 0
    }

    // MARK: - Engine-Specific Queries

    /// Get explanation for a specific trigger
    public func getExplanation(for triggerName: String) async -> TriggerExplanation? {
        guard let result = await statisticalEngine.analyzeTrigger(name: triggerName) else {
            return nil
        }
        return statisticalEngine.generateExplanation(for: result)
    }

    /// Get triggers by category
    public func getTriggers(category: TriggerCategory) -> [UnifiedTriggerResult] {
        topTriggers.filter { $0.triggerCategory == category }
    }

    /// Get only high-confidence triggers
    public func getHighConfidenceTriggers() -> [UnifiedTriggerResult] {
        topTriggers.filter { $0.ensembleConfidence == .high }
    }

    // MARK: - Neural Opt-In

    /// Enable neural network engine (requires user consent)
    public func enableNeuralEngine() async {
        configuration.neuralOptIn = true
        await updatePhase()
    }

    /// Disable neural network engine
    public func disableNeuralEngine() async {
        configuration.neuralOptIn = false
        await updatePhase()
    }

    // MARK: - Cache Management

    /// Invalidate all caches and force fresh analysis
    public func invalidateCaches() async {
        await triggerDataService.invalidateAllAnalysisCaches()
    }

    // MARK: - Ensemble Methods

    /// Calculate weighted ensemble score from all engines
    private func calculateEnsembleScore(
        statistical: StatisticalTriggerResult,
        knn: KNNTriggerResult?,
        neural: NeuralTriggerResult?
    ) -> Double {
        var weightedSum: Double = 0
        var totalWeight: Double = 0

        // Statistical contribution (always available)
        let statWeight = configuration.statisticalWeight
        weightedSum += statistical.effectSize.cohenD * statWeight
        totalWeight += statWeight

        // k-NN contribution (if available)
        if let knn = knn {
            let knnWeight = configuration.knnWeight
            // Normalize predictedImpact to roughly same scale as Cohen's d
            let normalizedImpact = knn.predictedEffect / 3.0  // Assuming max impact ~3 points
            weightedSum += normalizedImpact * knnWeight
            totalWeight += knnWeight
        }

        // Neural contribution (if available)
        if let neural = neural {
            let neuralWeight = configuration.neuralWeight
            let normalizedEffect = neural.predictedEffect / 3.0
            weightedSum += normalizedEffect * neuralWeight
            totalWeight += neuralWeight
        }

        return totalWeight > 0 ? weightedSum / totalWeight : statistical.effectSize.cohenD
    }

    /// Determine ensemble confidence from individual engine confidences
    private func determineEnsembleConfidence(
        statistical: TriggerConfidence,
        knn: TriggerConfidence?,
        neural: TriggerConfidence?
    ) -> TriggerConfidence {
        // Collect all available confidences
        var confidences = [statistical]
        if let k = knn { confidences.append(k) }
        if let n = neural { confidences.append(n) }

        // If any engine has high confidence, use high
        if confidences.contains(.high) {
            return .high
        }

        // If majority have medium or better, use medium
        let mediumOrBetter = confidences.filter { $0 == .high || $0 == .medium }.count
        if mediumOrBetter > confidences.count / 2 {
            return .medium
        }

        // If any has low confidence, use low
        if confidences.contains(.low) {
            return .low
        }

        return .insufficient
    }

    /// Determine which engine should be primary for explanations
    private func determinePrimaryEngine(
        statistical: StatisticalTriggerResult,
        knn: KNNTriggerResult?,
        neural: NeuralTriggerResult?
    ) -> EngineType {
        // Prefer neural if available with high confidence
        if let neural = neural, neural.confidence == .high {
            return .neural
        }

        // Prefer k-NN if available with high confidence
        if let knn = knn, knn.confidence == .high {
            return .knn
        }

        // Fall back to statistical
        return .statistical
    }

    // MARK: - Tomorrow Prediction

    /// Get prediction for tomorrow using all active engines
    public func predictTomorrow() async -> TomorrowPrediction {
        await updatePhase()

        var predictions: [String: (value: Double, confidence: TriggerConfidence)] = [:]

        // Statistical doesn't predict, it analyzes historical patterns

        // k-NN prediction
        if (currentPhase == .knn || currentPhase == .neural) && knnEngine.isReady {
            let knnPred = await knnEngine.predictTomorrow()
            predictions["knn"] = (knnPred.predictedPain, knnPred.confidence)
        }

        // Neural prediction
        if currentPhase == .neural && neuralEngine.isReady {
            let neuralPred = await neuralEngine.predictTomorrow()
            predictions["neural"] = (neuralPred.predictedPain, neuralPred.confidence)
        }

        // Ensemble prediction
        var weightedSum: Double = 0
        var totalWeight: Double = 0

        if let knn = predictions["knn"] {
            let weight = confidenceToWeight(knn.confidence) * configuration.knnWeight
            weightedSum += knn.value * weight
            totalWeight += weight
        }

        if let neural = predictions["neural"] {
            let weight = confidenceToWeight(neural.confidence) * configuration.neuralWeight
            weightedSum += neural.value * weight
            totalWeight += weight
        }

        let ensemblePrediction = totalWeight > 0 ? weightedSum / totalWeight : nil

        return TomorrowPrediction(
            knnPrediction: predictions["knn"]?.value,
            knnConfidence: predictions["knn"]?.confidence,
            neuralPrediction: predictions["neural"]?.value,
            neuralConfidence: predictions["neural"]?.confidence,
            ensemblePrediction: ensemblePrediction,
            currentPhase: currentPhase
        )
    }

    /// Convert confidence to weight
    private func confidenceToWeight(_ confidence: TriggerConfidence) -> Double {
        switch confidence {
        case .high: return 1.0
        case .medium: return 0.7
        case .low: return 0.4
        case .insufficient: return 0.1
        }
    }
}

// MARK: - Tomorrow Prediction

/// Prediction for tomorrow's pain level
public struct TomorrowPrediction {
    public let knnPrediction: Double?
    public let knnConfidence: TriggerConfidence?
    public let neuralPrediction: Double?
    public let neuralConfidence: TriggerConfidence?
    public let ensemblePrediction: Double?
    public let currentPhase: ActivationPhase

    public var hasPrediction: Bool {
        ensemblePrediction != nil
    }

    public var predictionLevel: String? {
        guard let pred = ensemblePrediction else { return nil }
        switch pred {
        case 0..<2: return "Low"
        case 2..<4: return "Mild"
        case 4..<6: return "Moderate"
        case 6..<8: return "High"
        default: return "Severe"
        }
    }

    public var explanation: String {
        if !hasPrediction {
            return "Need more data for predictions. Keep tracking!"
        }

        let predStr = String(format: "%.1f", ensemblePrediction ?? 0)
        return "Predicted pain level: \(predStr)/10 (\(predictionLevel ?? "Unknown"))"
    }
}

// MARK: - Unified Trigger Result

/// Combined result from all active engines
public struct UnifiedTriggerResult: Identifiable {
    public let id: UUID
    public let triggerName: String
    public let triggerCategory: TriggerCategory
    public let icon: String

    // Engine-specific results
    public let statisticalResult: StatisticalTriggerResult?
    public let knnResult: KNNTriggerResult?
    public let neuralResult: NeuralTriggerResult?

    // Ensemble outputs
    public let ensembleScore: Double
    public let ensembleConfidence: TriggerConfidence
    public let primaryEngine: EngineType
    public let activeEngines: [EngineType]

    public init(
        id: UUID = UUID(),
        triggerName: String,
        triggerCategory: TriggerCategory,
        icon: String,
        statisticalResult: StatisticalTriggerResult?,
        knnResult: KNNTriggerResult?,
        neuralResult: NeuralTriggerResult?,
        ensembleScore: Double,
        ensembleConfidence: TriggerConfidence,
        primaryEngine: EngineType,
        activeEngines: [EngineType]
    ) {
        self.id = id
        self.triggerName = triggerName
        self.triggerCategory = triggerCategory
        self.icon = icon
        self.statisticalResult = statisticalResult
        self.knnResult = knnResult
        self.neuralResult = neuralResult
        self.ensembleScore = ensembleScore
        self.ensembleConfidence = ensembleConfidence
        self.primaryEngine = primaryEngine
        self.activeEngines = activeEngines
    }

    /// User-friendly effect description
    public var effectDescription: String {
        if ensembleScore > 0.2 {
            return "Worsens symptoms"
        } else if ensembleScore < -0.2 {
            return "May help symptoms"
        } else {
            return "Minimal effect"
        }
    }

    /// Whether this trigger is significant
    public var isSignificant: Bool {
        statisticalResult?.isSignificant ?? false
    }
}

// MARK: - k-NN Result Placeholder

/// Placeholder for Phase 2 k-NN results
public struct KNNTriggerResult {
    public let triggerName: String
    public let similarDays: [SimilarDay]
    public let predictedEffect: Double
    public let confidence: TriggerConfidence
}

// MARK: - Neural Result Placeholder

/// Placeholder for Phase 3 Neural results
public struct NeuralTriggerResult {
    public let triggerName: String
    public let predictedEffect: Double
    public let featureAttribution: [FeatureAttribution]
    public let uncertainty: Double
    public let confidence: TriggerConfidence
}

// MARK: - Insights Summary

/// Summary of trigger analysis for dashboard display
public struct TriggerInsightsSummary {
    public let totalTriggersTracked: Int
    public let daysOfData: Int
    public let significantTriggers: Int
    public let harmfulTriggers: Int
    public let protectiveTriggers: Int
    public let currentPhase: ActivationPhase
    public let topHarmfulTrigger: String?
    public let topProtectiveTrigger: String?
    public let lastAnalysisDate: Date?
    public let nextPhaseAt: Int?
    public let daysUntilNextPhase: Int?

    public var phaseDescription: String {
        switch currentPhase {
        case .statistical:
            return "Statistical Analysis"
        case .knn:
            return "Pattern Matching Active"
        case .neural:
            return "Neural Network Active"
        }
    }

    public var progressToNextPhase: Double? {
        guard let nextPhase = nextPhaseAt, nextPhase > 0 else { return nil }
        return min(1.0, Double(daysOfData) / Double(nextPhase))
    }
}

