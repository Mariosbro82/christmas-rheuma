//
//  MLPredictionService.swift
//  InflamAI
//
//  Unified ML Prediction Service
//  Single source of truth for all flare predictions
//  ALL ROADS LEAD TO ASFlarePredictor.mlpackage
//
//  Architecture:
//  - UnifiedNeuralEngine → ASFlarePredictor.mlpackage (CoreML)
//  - FlarePredictor → UnifiedNeuralEngine → ASFlarePredictor.mlpackage (CoreML + Weather Enhancement)
//
//  Phase 3: Unified CoreML Architecture
//

import Foundation
import Combine
import CoreData
import CoreML

/// Unified prediction service that wraps both Neural Engine and Statistical predictor
/// Implements hybrid approach showing both predictions with clear source labeling
@MainActor
public final class MLPredictionService: ObservableObject {

    // MARK: - Singleton

    public static let shared = MLPredictionService()

    // MARK: - Published State

    /// Current hybrid prediction result
    @Published public private(set) var currentPrediction: HybridPrediction?

    /// Which prediction source is currently primary
    @Published public private(set) var primarySource: PredictionSource = .neuralEngine

    /// Service status
    @Published public private(set) var status: MLServiceStatus = .loading

    /// Data quality report from last prediction
    @Published public private(set) var dataQuality: DataQualityReport?

    /// Error message if any
    @Published public private(set) var errorMessage: String?

    /// Last prediction timestamp
    @Published public private(set) var lastPredictionTime: Date?

    // MARK: - Underlying Engines

    /// Neural Engine (CoreML-based)
    private let neuralEngine: UnifiedNeuralEngine

    /// Statistical Predictor (Pearson correlation-based)
    private let statisticalPredictor: FlarePredictor

    /// Outcome Tracker for prediction validation
    private let outcomeTracker: OutcomeTracker

    /// Continuous Learning Pipeline for on-device model updates
    private let learningPipeline: ContinuousLearningPipeline

    /// Calibration Engine for confidence intervals
    private let calibrationEngine: CalibrationEngine

    /// Persistence controller
    private let persistenceController: InflamAIPersistenceController

    private var cancellables = Set<AnyCancellable>()

    /// Auto-personalization enabled (user can opt-out in Settings)
    @Published public var autoPersonalizationEnabled: Bool {
        didSet {
            UserDefaults.standard.set(autoPersonalizationEnabled, forKey: "mlAutoPersonalizationEnabled")
        }
    }

    // MARK: - Configuration

    /// Minimum days required for neural engine
    private let neuralEngineMinDays = 7

    /// Minimum days required for statistical predictor
    private let statisticalMinDays = 30

    // MARK: - Initialization

    private init(
        neuralEngine: UnifiedNeuralEngine = .shared,
        persistenceController: InflamAIPersistenceController = .shared,
        outcomeTracker: OutcomeTracker = .shared
    ) {
        self.neuralEngine = neuralEngine
        self.persistenceController = persistenceController
        self.outcomeTracker = outcomeTracker

        // Create statistical predictor with context
        self.statisticalPredictor = FlarePredictor(
            context: persistenceController.container.viewContext
        )

        // Load auto-personalization preference (default: enabled)
        self.autoPersonalizationEnabled = UserDefaults.standard.object(forKey: "mlAutoPersonalizationEnabled") as? Bool ?? true

        // Create calibration engine with model and scaler
        // Use default scaler values (will be updated during calibration)
        let defaultMean = [Float](repeating: 0.0, count: 92)
        let defaultStd = [Float](repeating: 1.0, count: 92)
        let scaler = CalibrationFeatureScaler(mean: defaultMean, std: defaultStd)

        // Initialize calibration engine (will use the neural engine's model internally)
        do {
            let model = try ASFlarePredictor(configuration: MLModelConfiguration())
            self.calibrationEngine = CalibrationEngine(model: model, scaler: scaler)
        } catch {
            print("❌ CRITICAL: Failed to load ASFlarePredictor model for calibration: \(error)")
            // Create a placeholder calibration engine with minimal functionality
            // Note: This is a critical failure - the app should still work but predictions will be disabled
            fatalError("Unable to initialize ML prediction service - ASFlarePredictor model not found. Please reinstall the app.")
        }

        // Create continuous learning pipeline
        self.learningPipeline = ContinuousLearningPipeline(
            persistenceController: persistenceController,
            featureExtractor: neuralEngine.featureExtractor,
            calibrationEngine: calibrationEngine
        )

        // Subscribe to engine state changes
        setupSubscriptions()

        // Initialize asynchronously
        Task {
            await initialize()
        }
    }

    // MARK: - Public API

    /// Get unified prediction - ALL ROADS LEAD TO ASFlarePredictor.mlpackage
    /// Both Neural Engine and FlarePredictor now use the same CoreML model
    /// Returns nil if insufficient data quality (prevents showing meaningless predictions)
    public func getPrediction() async -> HybridPrediction? {
        status = .predicting

        // Update data quality first
        await updateDataQuality()

        // Check if we have enough data before attempting prediction
        if let quality = dataQuality {
            if quality.overallScore < 0.15 {
                print("⚠️ [MLPredictionService] Insufficient data quality (\(Int(quality.overallScore * 100))%) - skipping prediction")
                self.errorMessage = "Need more data for predictions. Current data quality: \(Int(quality.overallScore * 100))%"
                self.currentPrediction = nil
                self.status = .ready
                return nil
            }
        }

        // PRIMARY: Get CoreML prediction from Neural Engine
        let neural = await getNeuralEnginePrediction()

        // ENHANCEMENT: Get weather-enhanced prediction from FlarePredictor
        // Note: FlarePredictor now also calls CoreML internally, then adds weather analysis
        let statistical = await getStatisticalPrediction()

        // If neither source has a prediction, don't create a hybrid
        guard neural != nil || statistical != nil else {
            print("⚠️ [MLPredictionService] No predictions available from either source")
            self.errorMessage = "Not enough data for predictions. Keep logging symptoms daily."
            self.currentPrediction = nil
            self.status = .ready
            return nil
        }

        // Determine primary source based on what's available
        let source = determinePrimarySource(neural: neural, statistical: statistical)
        self.primarySource = source

        // Create unified prediction
        let hybrid = HybridPrediction(
            neuralEnginePrediction: neural,
            statisticalPrediction: statistical,  // Now includes weather enhancement on top of CoreML
            primarySource: source,
            combinedRiskScore: calculateCombinedRisk(neural: neural, statistical: statistical),
            combinedConfidence: calculateCombinedConfidence(neural: neural, statistical: statistical),
            timestamp: Date(),
            dataQuality: dataQuality
        )

        self.currentPrediction = hybrid
        self.lastPredictionTime = Date()
        self.status = .ready

        // Log prediction
        logPrediction(hybrid)

        // Track prediction for outcome validation
        trackPredictionForValidation(hybrid)

        print("✅ [MLPredictionService] Prediction complete via \(source.rawValue)")

        return hybrid
    }

    /// Refresh all predictions
    public func refresh() async {
        print("🔄 [MLPredictionService] Refreshing predictions...")

        // Refresh underlying engines
        await neuralEngine.refresh()
        await statisticalPredictor.updatePrediction()

        // Get new hybrid prediction
        _ = await getPrediction()
    }

    /// Train statistical model (requires 30+ days)
    public func trainStatisticalModel() async throws {
        try await statisticalPredictor.trainModel()
    }

    /// Trigger neural engine personalization
    public func triggerPersonalization() async throws {
        try await neuralEngine.triggerPersonalization()
    }

    /// Get data readiness info
    public func getDataReadiness() -> DataReadinessReport {
        let daysOfData = neuralEngine.daysOfUserData

        return DataReadinessReport(
            totalDays: daysOfData,
            neuralEngineReady: daysOfData >= neuralEngineMinDays,
            neuralEngineDaysNeeded: max(0, neuralEngineMinDays - daysOfData),
            statisticalReady: statisticalPredictor.isModelTrained,
            statisticalDaysNeeded: statisticalPredictor.isModelTrained ? 0 : max(0, statisticalMinDays - daysOfData),
            personalizationReady: daysOfData >= 37,
            personalizationDaysNeeded: max(0, 37 - daysOfData)
        )
    }

    // MARK: - Private Methods

    private func initialize() async {
        print("🚀 [MLPredictionService] Initializing...")

        // Wait for neural engine to be ready
        while neuralEngine.engineStatus != .ready {
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
        }

        // Update data quality
        await updateDataQuality()

        // Get initial prediction if possible
        if neuralEngine.daysOfUserData >= neuralEngineMinDays {
            _ = await getPrediction()
        }

        status = .ready
        print("✅ [MLPredictionService] Initialized")
    }

    private func setupSubscriptions() {
        // Subscribe to neural engine changes
        neuralEngine.$currentPrediction
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                Task { @MainActor in
                    await self?.updateDataQuality()
                }
            }
            .store(in: &cancellables)
    }

    private func getNeuralEnginePrediction() async -> NeuralEnginePredictionResult? {
        guard neuralEngine.engineStatus == .ready,
              neuralEngine.daysOfUserData >= neuralEngineMinDays else {
            return nil
        }

        guard let prediction = await neuralEngine.predict() else {
            return nil
        }

        return NeuralEnginePredictionResult(
            willFlare: prediction.willFlare,
            probability: prediction.probability,
            confidence: prediction.confidence,
            riskLevel: mapRiskLevel(prediction.riskLevel),
            isPersonalized: prediction.isPersonalized,
            modelVersion: prediction.modelVersion,
            topFactors: prediction.topFactors.map { factor in
                PredictionFactor(
                    name: factor.name,
                    impact: mapImpact(factor.impact),
                    value: factor.value,
                    recommendation: factor.recommendation
                )
            },
            dataQualityScore: neuralEngine.featureExtractor.lastExtractionMetrics?.dataQualityScore ?? 0
        )
    }

    private func getStatisticalPrediction() async -> StatisticalPredictionResult? {
        guard statisticalPredictor.isModelTrained else {
            return nil
        }

        await statisticalPredictor.updatePrediction()

        return StatisticalPredictionResult(
            riskPercentage: statisticalPredictor.riskPercentage,
            riskLevel: mapStatisticalRiskLevel(statisticalPredictor.flareRiskLevel),
            daysUntilLikelyFlare: statisticalPredictor.daysUntilLikelyFlare,
            contributingFactors: statisticalPredictor.contributingFactors.map { factor in
                PredictionFactor(
                    name: factor.name,
                    impact: mapStatisticalImpact(factor.impact),
                    value: factor.value,
                    recommendation: factor.recommendation
                )
            },
            weatherRisk: statisticalPredictor.weatherForecastRisk.map { weather in
                WeatherRiskSummary(
                    overallRisk: weather.overallRisk.rawValue,
                    riskScore: weather.riskScore,
                    summary: weather.summaryText
                )
            }
        )
    }

    private func determinePrimarySource(
        neural: NeuralEnginePredictionResult?,
        statistical: StatisticalPredictionResult?
    ) -> PredictionSource {
        // If only one is available, use that
        if neural != nil && statistical == nil { return .neuralEngine }
        if neural == nil && statistical != nil { return .statistical }
        if neural == nil && statistical == nil { return .none }

        // If both available, prefer neural engine if personalized or has good data quality
        if let neural = neural {
            if neural.isPersonalized { return .neuralEngine }
            if neural.dataQualityScore >= 0.6 { return .neuralEngine }
        }

        // If statistical has been trained and neural isn't personalized, use statistical
        if statistical != nil && !(neural?.isPersonalized ?? false) {
            return .statistical
        }

        return .neuralEngine
    }

    private func calculateCombinedRisk(
        neural: NeuralEnginePredictionResult?,
        statistical: StatisticalPredictionResult?
    ) -> Float {
        let neuralRisk = neural?.probability ?? 0
        let statisticalRisk = Float(statistical?.riskPercentage ?? 0) / 100.0

        // If both available, weighted average (neural 60%, statistical 40%)
        if neural != nil && statistical != nil {
            let neuralWeight: Float = neural?.isPersonalized == true ? 0.7 : 0.5
            let statisticalWeight: Float = 1.0 - neuralWeight
            return (neuralRisk * neuralWeight) + (statisticalRisk * statisticalWeight)
        }

        // Otherwise return whichever is available
        if neural != nil { return neuralRisk }
        if statistical != nil { return statisticalRisk }

        return 0
    }

    private func calculateCombinedConfidence(
        neural: NeuralEnginePredictionResult?,
        statistical: StatisticalPredictionResult?
    ) -> HybridConfidence {
        let hasNeural = neural != nil
        let hasStatistical = statistical != nil
        let neuralPersonalized = neural?.isPersonalized ?? false
        let neuralQuality = neural?.dataQualityScore ?? 0

        if hasNeural && hasStatistical {
            if neuralPersonalized && neuralQuality >= 0.7 {
                return .high
            }
            return .moderate
        }

        if hasNeural && neuralPersonalized {
            return .moderate
        }

        if hasNeural || hasStatistical {
            return .low
        }

        return .veryLow
    }

    private func updateDataQuality() async {
        guard let metrics = neuralEngine.featureExtractor.lastExtractionMetrics else {
            return
        }

        self.dataQuality = DataQualityReport(
            overallScore: metrics.dataQualityScore,
            healthKitAvailable: metrics.healthKitFeatures,
            healthKitTotal: metrics.healthKitExpected,
            coreDataAvailable: metrics.coreDataFeatures,
            weatherAvailable: metrics.weatherFeatures,
            weatherTotal: metrics.weatherExpected,
            missingFeatures: metrics.missingFeatureNames
        )
    }

    private func logPrediction(_ prediction: HybridPrediction) {
        #if DEBUG
        print("📊 [MLPredictionService] Hybrid Prediction:")
        print("   Primary Source: \(prediction.primarySource)")
        print("   Combined Risk: \(String(format: "%.1f%%", prediction.combinedRiskScore * 100))")
        print("   Confidence: \(prediction.combinedConfidence)")

        if let neural = prediction.neuralEnginePrediction {
            print("   Neural Engine: \(String(format: "%.1f%%", neural.probability * 100)) (\(neural.isPersonalized ? "Personalized" : "Baseline"))")
        }

        if let statistical = prediction.statisticalPrediction {
            print("   Statistical: \(String(format: "%.1f%%", statistical.riskPercentage))")
        }
        #endif
    }

    /// Track prediction with OutcomeTracker for later validation
    private func trackPredictionForValidation(_ prediction: HybridPrediction) {
        // Track hybrid prediction
        outcomeTracker.recordPrediction(
            source: .hybrid,
            probability: prediction.combinedRiskScore,
            willFlare: prediction.combinedRiskScore >= 0.5,
            confidence: confidenceToFloat(prediction.combinedConfidence)
        )

        // Also track individual source predictions for comparison
        if let neural = prediction.neuralEnginePrediction {
            outcomeTracker.recordPrediction(
                source: .neuralEngine,
                probability: neural.probability,
                willFlare: neural.willFlare,
                confidence: confidenceToFloat(neural.confidence)
            )
        }

        if let statistical = prediction.statisticalPrediction {
            outcomeTracker.recordPrediction(
                source: .statistical,
                probability: Float(statistical.riskPercentage / 100.0),
                willFlare: statistical.riskPercentage >= 50,
                confidence: 0.5  // Statistical has fixed confidence
            )
        }
    }

    private func confidenceToFloat(_ confidence: HybridConfidence) -> Float {
        switch confidence {
        case .veryLow: return 0.25
        case .low: return 0.5
        case .moderate: return 0.75
        case .high: return 0.95
        }
    }

    private func confidenceToFloat(_ confidence: ConfidenceLevel) -> Float {
        switch confidence {
        case .low: return 0.5
        case .moderate: return 0.75
        case .high: return 0.90
        case .veryHigh: return 0.95
        }
    }

    /// Run backtest on historical data
    public func runBacktest() async -> BacktestResult {
        return await outcomeTracker.runBacktest()
    }

    /// Get accuracy metrics by source
    public func getAccuracyBySource() -> [PredictionSourceType: SourceAccuracy] {
        return outcomeTracker.getAccuracyBySource()
    }

    /// Get overall accuracy metrics
    public func getAccuracyMetrics() -> AccuracyMetrics? {
        return outcomeTracker.accuracyMetrics
    }

    /// Get calibration metrics
    func getCalibrationMetrics() -> OutcomeCalibrationMetrics? {
        return outcomeTracker.calibrationMetrics
    }

    /// Record flare outcome for validation
    public func recordFlareOutcome(flareOccurred: Bool, flareDate: Date? = nil) {
        outcomeTracker.recordOutcome(flareOccurred: flareOccurred, flareDate: flareDate)
    }

    /// Auto-validate predictions against flare history
    public func autoValidatePredictions() async {
        await outcomeTracker.autoValidatePredictions()
    }

    // MARK: - Continuous Learning Pipeline

    /// Add training sample after user completes check-in
    /// Call this when a user logs symptoms to build personal training data
    public func addTrainingSample(flareOccurred: Bool) async {
        guard autoPersonalizationEnabled else {
            print("ℹ️ [MLPredictionService] Auto-personalization disabled - skipping training sample")
            return
        }

        do {
            // Extract current features
            let extractionResult = await neuralEngine.featureExtractor.extract30DayFeaturesWithMetrics()
            let features = extractionResult.features

            // Add to learning pipeline
            try await learningPipeline.addTrainingSample(
                features: features,
                actualOutcome: flareOccurred
            )

            print("✅ [MLPredictionService] Added training sample (flare: \(flareOccurred))")
        } catch {
            print("❌ [MLPredictionService] Failed to add training sample: \(error)")
        }
    }

    /// Get learning pipeline status
    func getLearningStatus() -> LearningPipelineStatus {
        return learningPipeline.getPersonalizationStatus()
    }

    /// Check if user data is ready for personalization
    func checkDataReadiness() async -> DataReadinessStatus {
        return await learningPipeline.checkDataReadiness()
    }

    /// Manually trigger model update (for testing or user-initiated)
    public func triggerModelUpdate() async throws {
        try await learningPipeline.triggerManualUpdate()
    }

    // MARK: - Calibration Engine

    /// Get prediction with calibrated confidence intervals
    func getPredictionWithCalibration() async -> CalibratedPredictionResult? {
        do {
            // Extract features
            let extractionResult = await neuralEngine.featureExtractor.extract30DayFeaturesWithMetrics()
            let features = extractionResult.features

            // Get calibrated prediction with uncertainty
            let calibrated = try await calibrationEngine.predictWithUncertainty(features: features)

            // Get uncertainty decomposition
            let uncertainty = calibrationEngine.decomposeUncertainty(calibratedPrediction: calibrated)

            return CalibratedPredictionResult(
                probability: calibrated.probability,
                confidence: calibrated.confidence,
                uncertaintyScore: calibrated.uncertaintyScore,
                predictionInterval: calibrated.predictionInterval,
                aleatoricUncertainty: uncertainty.aleatoric,
                epistemicUncertainty: uncertainty.epistemic,
                interpretation: uncertainty.interpretation
            )
        } catch {
            print("❌ [MLPredictionService] Calibrated prediction failed: \(error)")
            return nil
        }
    }

    /// Update calibration with new outcome data
    public func updateCalibration(prediction: Float, actualOutcome: Bool) {
        // Get historical data from outcome tracker
        let history = outcomeTracker.getPredictionHistory(source: .neuralEngine)
        let predictions = history.map { $0.predictedProbability }
        let outcomes = history.compactMap { $0.actualFlare }

        calibrationEngine.updateCalibration(
            newPrediction: prediction,
            actualOutcome: actualOutcome,
            historicalPredictions: predictions,
            historicalOutcomes: outcomes
        )
    }

    /// Get expected calibration error (ECE)
    public func getExpectedCalibrationError() -> Float {
        let history = outcomeTracker.getPredictionHistory(source: .neuralEngine)
        let predictions = history.map { $0.predictedProbability }
        let outcomes = history.compactMap { $0.actualFlare }

        guard !predictions.isEmpty, !outcomes.isEmpty else { return 0 }

        return calibrationEngine.computeECE(predictions: predictions, outcomes: outcomes)
    }

    /// Get calibration plot data for visualization
    public func getCalibrationPlotData() -> [(predicted: Float, actual: Float, count: Int)] {
        let history = outcomeTracker.getPredictionHistory(source: .neuralEngine)
        let predictions = history.map { $0.predictedProbability }
        let outcomes = history.compactMap { $0.actualFlare }

        guard !predictions.isEmpty, !outcomes.isEmpty else { return [] }

        return calibrationEngine.generateCalibrationPlotData(predictions: predictions, outcomes: outcomes)
    }

    // MARK: - Mapping Helpers

    private func mapRiskLevel(_ level: RiskLevel) -> HybridRiskLevel {
        switch level {
        case .low: return .low
        case .moderate: return .moderate
        case .high: return .high
        case .critical: return .critical
        }
    }

    private func mapStatisticalRiskLevel(_ level: FlarePredictorRiskLevel) -> HybridRiskLevel {
        switch level {
        case .unknown: return .unknown
        case .low: return .low
        case .moderate: return .moderate
        case .high: return .high
        case .critical: return .critical
        }
    }

    private func mapImpact(_ impact: ContributingFactor.ImpactLevel) -> PredictionFactor.Impact {
        switch impact {
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        }
    }

    private func mapStatisticalImpact(_ impact: FlarePredictorFactor.Impact) -> PredictionFactor.Impact {
        switch impact {
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        }
    }
}

// MARK: - Data Types

/// Hybrid prediction combining both engines
/// Note: This is the service-specific HybridPrediction used by MLPredictionService
/// MLTypes.swift has MLHybridPrediction with a different structure for general use
public struct HybridPrediction {
    /// Neural Engine prediction (CoreML-based)
    public let neuralEnginePrediction: NeuralEnginePredictionResult?

    /// Statistical prediction (Pearson correlation-based)
    public let statisticalPrediction: StatisticalPredictionResult?

    /// Which source is considered primary
    public let primarySource: PredictionSource

    /// Combined risk score (0.0 - 1.0)
    public let combinedRiskScore: Float

    /// Combined confidence level
    public let combinedConfidence: HybridConfidence

    /// Timestamp
    public let timestamp: Date

    /// Data quality report
    public let dataQuality: DataQualityReport?

    /// Human-readable summary
    public var summary: String {
        let riskPercent = Int(combinedRiskScore * 100)
        let riskText = riskPercent >= 50 ? "ELEVATED" : "LOW"
        return "\(riskText) RISK: \(riskPercent)% chance of flare in next 7 days"
    }

    /// Risk level for UI
    public var riskLevel: HybridRiskLevel {
        switch combinedRiskScore {
        case 0..<0.25: return .low
        case 0.25..<0.50: return .moderate
        case 0.50..<0.75: return .high
        default: return .critical
        }
    }

    /// All contributing factors from both sources
    public var allFactors: [PredictionFactor] {
        var factors: [PredictionFactor] = []

        if let neural = neuralEnginePrediction {
            factors.append(contentsOf: neural.topFactors)
        }

        if let statistical = statisticalPrediction {
            // Add statistical factors that aren't duplicates
            for factor in statistical.contributingFactors {
                if !factors.contains(where: { $0.name == factor.name }) {
                    factors.append(factor)
                }
            }
        }

        return factors.sorted { $0.impact.rawValue > $1.impact.rawValue }
    }
}

/// Neural Engine prediction result
public struct NeuralEnginePredictionResult {
    public let willFlare: Bool
    public let probability: Float
    public let confidence: ConfidenceLevel
    public let riskLevel: HybridRiskLevel
    public let isPersonalized: Bool
    public let modelVersion: Int
    public let topFactors: [PredictionFactor]
    public let dataQualityScore: Float
}

/// Statistical prediction result
public struct StatisticalPredictionResult {
    public let riskPercentage: Double
    public let riskLevel: HybridRiskLevel
    public let daysUntilLikelyFlare: Int?
    public let contributingFactors: [PredictionFactor]
    public let weatherRisk: WeatherRiskSummary?
}

/// Weather risk summary
public struct WeatherRiskSummary {
    public let overallRisk: String
    public let riskScore: Double
    public let summary: String
}

/// Prediction source
public enum PredictionSource: String {
    case neuralEngine = "Neural Engine"
    case statistical = "Statistical Analysis"
    case hybrid = "Hybrid (Both)"
    case none = "No Prediction"

    public var icon: String {
        switch self {
        case .neuralEngine: return "brain.head.profile"
        case .statistical: return "chart.xyaxis.line"
        case .hybrid: return "arrow.triangle.branch"
        case .none: return "questionmark.circle"
        }
    }

    public var description: String {
        switch self {
        case .neuralEngine: return "CoreML neural network trained on your patterns"
        case .statistical: return "Statistical correlation analysis of your history"
        case .hybrid: return "Combined prediction from both engines"
        case .none: return "Insufficient data for prediction"
        }
    }
}

// Note: MLServiceStatus is defined in MLTypes.swift

/// Hybrid risk level
public enum HybridRiskLevel: String, CaseIterable {
    case unknown = "Unknown"
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case critical = "Critical"

    public var color: String {
        switch self {
        case .unknown: return "gray"
        case .low: return "green"
        case .moderate: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }

    public var icon: String {
        switch self {
        case .unknown: return "questionmark.circle"
        case .low: return "checkmark.shield.fill"
        case .moderate: return "exclamationmark.triangle"
        case .high: return "exclamationmark.triangle.fill"
        case .critical: return "xmark.octagon.fill"
        }
    }
}

/// Hybrid confidence level
public enum HybridConfidence: String, CaseIterable {
    case veryLow = "Very Low"
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"

    public var description: String {
        switch self {
        case .veryLow: return "Limited data, prediction unreliable"
        case .low: return "Single source prediction"
        case .moderate: return "Dual source, building confidence"
        case .high: return "Personalized model with good data"
        }
    }
}

/// Contributing factor from either engine
public struct PredictionFactor: Identifiable {
    public let id = UUID()
    public let name: String
    public let impact: Impact
    public let value: Double
    public let recommendation: String

    public enum Impact: Int {
        case low = 1
        case medium = 2
        case high = 3

        public var color: String {
            switch self {
            case .low: return "blue"
            case .medium: return "orange"
            case .high: return "red"
            }
        }
    }
}

/// Data quality report
public struct DataQualityReport {
    public let overallScore: Float
    public let healthKitAvailable: Int
    public let healthKitTotal: Int
    public let coreDataAvailable: Int
    public let weatherAvailable: Int
    public let weatherTotal: Int
    public let missingFeatures: [String]

    public var healthKitPercentage: Float {
        guard healthKitTotal > 0 else { return 0 }
        return Float(healthKitAvailable) / Float(healthKitTotal)
    }

    public var isHealthKitConnected: Bool {
        healthKitAvailable > 0
    }

    public var qualityLevel: String {
        switch overallScore {
        case 0..<0.3: return "Poor"
        case 0.3..<0.5: return "Fair"
        case 0.5..<0.7: return "Good"
        case 0.7..<0.9: return "Very Good"
        default: return "Excellent"
        }
    }
}

/// Data readiness report
public struct DataReadinessReport {
    public let totalDays: Int
    public let neuralEngineReady: Bool
    public let neuralEngineDaysNeeded: Int
    public let statisticalReady: Bool
    public let statisticalDaysNeeded: Int
    public let personalizationReady: Bool
    public let personalizationDaysNeeded: Int

    public var overallReadiness: Float {
        var readiness: Float = 0

        if neuralEngineReady { readiness += 0.3 }
        if statisticalReady { readiness += 0.3 }
        if personalizationReady { readiness += 0.4 }

        return readiness
    }

    public var nextMilestone: String {
        if !neuralEngineReady {
            return "Need \(neuralEngineDaysNeeded) more days for AI predictions"
        }
        if !statisticalReady {
            return "Need \(statisticalDaysNeeded) more days for statistical analysis"
        }
        if !personalizationReady {
            return "Need \(personalizationDaysNeeded) more days for personalization"
        }
        return "All milestones reached!"
    }
}

/// Calibrated prediction result with confidence intervals
struct CalibratedPredictionResult {
    let probability: Float
    let confidence: CalibrationEngine.CalibratedPrediction.ConfidenceLevel
    let uncertaintyScore: Float
    let predictionInterval: (lower: Float, upper: Float)
    let aleatoricUncertainty: Float
    let epistemicUncertainty: Float
    let interpretation: String

    var confidenceIntervalText: String {
        let lower = Int(predictionInterval.lower * 100)
        let upper = Int(predictionInterval.upper * 100)
        return "\(lower)% - \(upper)%"
    }

    var uncertaintyLevel: String {
        if uncertaintyScore < 0.15 {
            return "Low"
        } else if uncertaintyScore < 0.30 {
            return "Moderate"
        } else {
            return "High"
        }
    }
}

// MARK: - FlarePredictor uses FlarePredictorRiskLevel defined in FlarePredictor.swift
