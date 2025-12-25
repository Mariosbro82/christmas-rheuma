//
//  OutcomeTracker.swift
//  InflamAI
//
//  Tracks ML predictions vs actual outcomes for:
//  - Real-world accuracy measurement
//  - Backtesting historical predictions
//  - Model performance validation
//  - Calibration improvement
//
//  Phase 3 of Production-Ready ML Plan
//

import Foundation
import CoreData

/// Tracks predictions and their actual outcomes for accuracy measurement
@MainActor
public final class OutcomeTracker: ObservableObject {

    // MARK: - Singleton

    public static let shared = OutcomeTracker()

    // MARK: - Published State

    /// All tracked predictions (most recent first)
    @Published public private(set) var trackedPredictions: [TrackedPrediction] = []

    /// Overall accuracy metrics
    @Published public private(set) var accuracyMetrics: AccuracyMetrics?

    /// Calibration metrics
    @Published public private(set) var calibrationMetrics: OutcomeCalibrationMetrics?

    /// Last backtest results
    @Published public private(set) var lastBacktestResult: BacktestResult?

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private let userDefaults: UserDefaults

    // MARK: - Configuration

    private let predictionHorizonDays: Int = 7  // Predictions are for 7-day window
    private let maxStoredPredictions: Int = 500
    private let storageKey = "outcome_tracker_predictions"

    // MARK: - Initialization

    private init(
        persistenceController: InflamAIPersistenceController = .shared,
        userDefaults: UserDefaults = .standard
    ) {
        self.persistenceController = persistenceController
        self.userDefaults = userDefaults

        loadStoredPredictions()
        updateMetrics()
    }

    // MARK: - Public API

    /// Record a new prediction for future validation
    public func recordPrediction(
        source: PredictionSourceType,
        probability: Float,
        willFlare: Bool,
        confidence: Float,
        features: [String: Float]? = nil
    ) {
        let prediction = TrackedPrediction(
            id: UUID(),
            predictionDate: Date(),
            source: source,
            predictedProbability: probability,
            predictedWillFlare: willFlare,
            confidence: confidence,
            outcomeDate: nil,
            actualFlare: nil,
            actualFlareDate: nil,
            features: features,
            wasValidated: false
        )

        trackedPredictions.insert(prediction, at: 0)

        // Trim old predictions
        if trackedPredictions.count > maxStoredPredictions {
            trackedPredictions = Array(trackedPredictions.prefix(maxStoredPredictions))
        }

        savePredictions()

        #if DEBUG
        print("📊 [OutcomeTracker] Recorded prediction: \(Int(probability * 100))% (\(source.rawValue))")
        #endif
    }

    /// Record the actual outcome for a prediction
    public func recordOutcome(
        flareOccurred: Bool,
        flareDate: Date? = nil
    ) {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -predictionHorizonDays, to: Date()) ?? Date()

        // Find predictions that need outcomes (made 7+ days ago, not yet validated)
        var updated = false
        for i in 0..<trackedPredictions.count {
            var prediction = trackedPredictions[i]

            // Skip if already validated
            if prediction.wasValidated { continue }

            // Check if prediction is old enough to validate
            if prediction.predictionDate <= cutoffDate {
                prediction.actualFlare = flareOccurred
                prediction.actualFlareDate = flareDate
                prediction.outcomeDate = Date()
                prediction.wasValidated = true

                trackedPredictions[i] = prediction
                updated = true

                #if DEBUG
                print("✅ [OutcomeTracker] Validated prediction from \(prediction.predictionDate.formatted()): actual=\(flareOccurred)")
                #endif
            }
        }

        if updated {
            savePredictions()
            updateMetrics()
        }
    }

    /// Auto-validate predictions by checking Core Data for flare events
    public func autoValidatePredictions() async {
        let context = persistenceController.container.viewContext
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -predictionHorizonDays, to: Date()) ?? Date()

        // Fetch recent flare events
        let flareEvents = await context.perform {
            let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            request.predicate = NSPredicate(
                format: "startDate >= %@",
                Calendar.current.date(byAdding: .day, value: -30, to: Date())! as NSDate
            )
            return (try? context.fetch(request)) ?? []
        }

        var validatedCount = 0

        for i in 0..<trackedPredictions.count {
            var prediction = trackedPredictions[i]

            // Skip if already validated or too recent
            if prediction.wasValidated { continue }
            if prediction.predictionDate > cutoffDate { continue }

            // Define the prediction window
            let windowStart = prediction.predictionDate
            let windowEnd = Calendar.current.date(byAdding: .day, value: predictionHorizonDays, to: windowStart)!

            // Check if a flare occurred in this window
            let flareInWindow = flareEvents.first { flare in
                guard let flareStart = flare.startDate else { return false }
                return flareStart >= windowStart && flareStart <= windowEnd
            }

            prediction.actualFlare = flareInWindow != nil
            prediction.actualFlareDate = flareInWindow?.startDate
            prediction.outcomeDate = Date()
            prediction.wasValidated = true

            trackedPredictions[i] = prediction
            validatedCount += 1
        }

        if validatedCount > 0 {
            savePredictions()
            updateMetrics()
            #if DEBUG
            print("✅ [OutcomeTracker] Auto-validated \(validatedCount) predictions")
            #endif
        }
    }

    /// Run backtest on historical data
    public func runBacktest() async -> BacktestResult {
        #if DEBUG
        print("🔬 [OutcomeTracker] Starting backtest...")
        #endif

        let context = persistenceController.container.viewContext

        // Fetch all symptom logs
        let symptomLogs = await context.perform {
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]
            return (try? context.fetch(request)) ?? []
        }

        // Fetch all flare events
        let flareEvents = await context.perform {
            let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            return (try? context.fetch(request)) ?? []
        }

        guard symptomLogs.count >= 37 else {
            let result = BacktestResult(
                totalPredictions: 0,
                correctPredictions: 0,
                truePositives: 0,
                trueNegatives: 0,
                falsePositives: 0,
                falseNegatives: 0,
                accuracy: 0,
                precision: 0,
                recall: 0,
                f1Score: 0,
                brierScore: 0,
                auc: 0,
                testPeriodStart: Date(),
                testPeriodEnd: Date(),
                predictionDetails: []
            )
            lastBacktestResult = result
            return result
        }

        // Generate predictions for historical dates and compare to actual outcomes
        var predictionDetails: [BacktestPredictionDetail] = []

        // Start from day 37 (need 30 days of history + 7 days to see outcome)
        for i in 36..<(symptomLogs.count - 7) {
            guard let predictionDate = symptomLogs[i].timestamp else { continue }

            // Calculate a simple risk score based on historical patterns
            let riskScore = calculateHistoricalRiskScore(logs: symptomLogs, currentIndex: i)

            // Check if flare occurred in next 7 days
            let windowEnd = Calendar.current.date(byAdding: .day, value: 7, to: predictionDate)!
            let actualFlare = flareEvents.contains { flare in
                guard let flareStart = flare.startDate else { return false }
                return flareStart > predictionDate && flareStart <= windowEnd
            }

            let predicted = riskScore >= 0.5

            predictionDetails.append(BacktestPredictionDetail(
                date: predictionDate,
                predictedProbability: riskScore,
                predictedFlare: predicted,
                actualFlare: actualFlare,
                wasCorrect: predicted == actualFlare
            ))
        }

        // Calculate metrics
        let tp = predictionDetails.filter { $0.predictedFlare && $0.actualFlare }.count
        let tn = predictionDetails.filter { !$0.predictedFlare && !$0.actualFlare }.count
        let fp = predictionDetails.filter { $0.predictedFlare && !$0.actualFlare }.count
        let fn = predictionDetails.filter { !$0.predictedFlare && $0.actualFlare }.count

        let total = predictionDetails.count
        let correct = tp + tn

        let accuracy = total > 0 ? Float(correct) / Float(total) : 0
        let precision = (tp + fp) > 0 ? Float(tp) / Float(tp + fp) : 0
        let recall = (tp + fn) > 0 ? Float(tp) / Float(tp + fn) : 0
        let f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0

        // Brier score (lower is better)
        let brierScore = predictionDetails.reduce(0.0) { sum, detail in
            let actual: Float = detail.actualFlare ? 1.0 : 0.0
            let diff = detail.predictedProbability - actual
            return sum + (diff * diff)
        } / Float(max(total, 1))

        // Simple AUC approximation
        let auc = calculateAUC(predictions: predictionDetails)

        let result = BacktestResult(
            totalPredictions: total,
            correctPredictions: correct,
            truePositives: tp,
            trueNegatives: tn,
            falsePositives: fp,
            falseNegatives: fn,
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1Score: f1,
            brierScore: brierScore,
            auc: auc,
            testPeriodStart: predictionDetails.first?.date ?? Date(),
            testPeriodEnd: predictionDetails.last?.date ?? Date(),
            predictionDetails: predictionDetails
        )

        lastBacktestResult = result

        #if DEBUG
        print("📊 [OutcomeTracker] Backtest complete:")
        print("   Predictions: \(total)")
        print("   Accuracy: \(String(format: "%.1f%%", accuracy * 100))")
        print("   Precision: \(String(format: "%.1f%%", precision * 100))")
        print("   Recall: \(String(format: "%.1f%%", recall * 100))")
        print("   F1 Score: \(String(format: "%.2f", f1))")
        print("   Brier Score: \(String(format: "%.3f", brierScore))")
        print("   AUC: \(String(format: "%.2f", auc))")
        #endif

        return result
    }

    /// Get accuracy by prediction source
    public func getAccuracyBySource() -> [PredictionSourceType: SourceAccuracy] {
        var sourceMetrics: [PredictionSourceType: SourceAccuracy] = [:]

        let validatedPredictions = trackedPredictions.filter { $0.wasValidated }

        for source in PredictionSourceType.allCases {
            let sourcePredictions = validatedPredictions.filter { $0.source == source }
            guard !sourcePredictions.isEmpty else { continue }

            let correct = sourcePredictions.filter { $0.predictedWillFlare == $0.actualFlare }.count
            let accuracy = Float(correct) / Float(sourcePredictions.count)

            // Brier score for this source
            let brier = sourcePredictions.reduce(0.0) { sum, pred in
                let actual: Float = pred.actualFlare == true ? 1.0 : 0.0
                let diff = pred.predictedProbability - actual
                return sum + (diff * diff)
            } / Float(sourcePredictions.count)

            sourceMetrics[source] = SourceAccuracy(
                source: source,
                totalPredictions: sourcePredictions.count,
                correctPredictions: correct,
                accuracy: accuracy,
                brierScore: brier
            )
        }

        return sourceMetrics
    }

    /// Clear all tracked predictions
    public func clearAllPredictions() {
        trackedPredictions = []
        accuracyMetrics = nil
        calibrationMetrics = nil
        lastBacktestResult = nil
        savePredictions()
    }

    /// Get prediction history for a specific source (for calibration)
    public func getPredictionHistory(source: PredictionSourceType) -> [TrackedPrediction] {
        return trackedPredictions.filter { $0.source == source && $0.wasValidated }
    }

    /// Get all validated predictions
    public func getValidatedPredictions() -> [TrackedPrediction] {
        return trackedPredictions.filter { $0.wasValidated }
    }

    // MARK: - Private Methods

    private func calculateHistoricalRiskScore(logs: [SymptomLog], currentIndex: Int) -> Float {
        // Look at last 7 days of data
        let startIndex = max(0, currentIndex - 6)
        let recentLogs = Array(logs[startIndex...currentIndex])

        // Calculate average BASDAI
        let avgBASDAI = recentLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(recentLogs.count)

        // Calculate trend
        var trend: Double = 0
        if recentLogs.count >= 3 {
            let first3 = recentLogs.prefix(3).reduce(0.0) { $0 + $1.basdaiScore } / 3.0
            let last3 = recentLogs.suffix(3).reduce(0.0) { $0 + $1.basdaiScore } / 3.0
            trend = last3 - first3
        }

        // Simple risk formula: normalize BASDAI (0-10) to 0-1, add trend factor
        var risk = Float(avgBASDAI / 10.0)
        if trend > 0.5 { risk += 0.15 }  // Increasing trend
        if trend > 1.0 { risk += 0.15 }  // Strong increasing trend

        return min(1.0, max(0.0, risk))
    }

    private func calculateAUC(predictions: [BacktestPredictionDetail]) -> Float {
        // Simple AUC calculation using trapezoidal rule
        guard !predictions.isEmpty else { return 0.5 }

        let sorted = predictions.sorted { $0.predictedProbability > $1.predictedProbability }

        var tpr: [Float] = [0]  // True positive rate
        var fpr: [Float] = [0]  // False positive rate

        let totalPositives = Float(sorted.filter { $0.actualFlare }.count)
        let totalNegatives = Float(sorted.filter { !$0.actualFlare }.count)

        guard totalPositives > 0 && totalNegatives > 0 else { return 0.5 }

        var cumTP: Float = 0
        var cumFP: Float = 0

        for pred in sorted {
            if pred.actualFlare {
                cumTP += 1
            } else {
                cumFP += 1
            }
            tpr.append(cumTP / totalPositives)
            fpr.append(cumFP / totalNegatives)
        }

        // Calculate area using trapezoidal rule
        var auc: Float = 0
        for i in 1..<tpr.count {
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        }

        return auc
    }

    private func updateMetrics() {
        let validated = trackedPredictions.filter { $0.wasValidated }
        guard !validated.isEmpty else {
            accuracyMetrics = nil
            calibrationMetrics = nil
            return
        }

        // Calculate accuracy metrics
        let correct = validated.filter { $0.predictedWillFlare == $0.actualFlare }.count
        let accuracy = Float(correct) / Float(validated.count)

        let tp = validated.filter { $0.predictedWillFlare && $0.actualFlare == true }.count
        let _ = validated.filter { !$0.predictedWillFlare && $0.actualFlare == false }.count // tn - kept for future metrics expansion
        let fp = validated.filter { $0.predictedWillFlare && $0.actualFlare == false }.count
        let fn = validated.filter { !$0.predictedWillFlare && $0.actualFlare == true }.count

        let precision = (tp + fp) > 0 ? Float(tp) / Float(tp + fp) : 0
        let recall = (tp + fn) > 0 ? Float(tp) / Float(tp + fn) : 0
        let f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0

        accuracyMetrics = AccuracyMetrics(
            totalPredictions: validated.count,
            correctPredictions: correct,
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1Score: f1,
            lastUpdated: Date()
        )

        // Calculate calibration metrics (by probability buckets)
        calibrationMetrics = calculateCalibration(predictions: validated)
    }

    private func calculateCalibration(predictions: [TrackedPrediction]) -> OutcomeCalibrationMetrics {
        // Bucket predictions by probability
        var buckets: [[TrackedPrediction]] = Array(repeating: [], count: 10)

        for pred in predictions {
            let bucketIndex = min(9, Int(pred.predictedProbability * 10))
            buckets[bucketIndex].append(pred)
        }

        var calibrationBuckets: [CalibrationBucket] = []
        var ece: Float = 0  // Expected Calibration Error

        for (index, bucket) in buckets.enumerated() {
            guard !bucket.isEmpty else { continue }

            let avgPredicted = bucket.reduce(0.0) { $0 + $1.predictedProbability } / Float(bucket.count)
            let actualRate = Float(bucket.filter { $0.actualFlare == true }.count) / Float(bucket.count)

            let calibrationError = abs(avgPredicted - actualRate)
            ece += calibrationError * Float(bucket.count) / Float(predictions.count)

            calibrationBuckets.append(CalibrationBucket(
                bucketIndex: index,
                minProbability: Float(index) * 0.1,
                maxProbability: Float(index + 1) * 0.1,
                predictedProbability: avgPredicted,
                actualRate: actualRate,
                count: bucket.count,
                calibrationError: calibrationError
            ))
        }

        return OutcomeCalibrationMetrics(
            expectedCalibrationError: ece,
            buckets: calibrationBuckets,
            isWellCalibrated: ece < 0.1
        )
    }

    // MARK: - Persistence

    // MARK: - Secure Storage

    /// Get secure storage URL for predictions (encrypted at rest)
    private func getPredictionsFileURL() -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent("outcome_predictions.encrypted.json")
    }

    private func savePredictions() {
        do {
            let data = try JSONEncoder().encode(trackedPredictions)
            let fileURL = getPredictionsFileURL()
            // Use completeFileProtection for PHI data encryption at rest
            try data.write(to: fileURL, options: [.completeFileProtection, .atomic])
        } catch {
            #if DEBUG
            print("❌ [OutcomeTracker] Failed to save predictions: \(error)")
            #endif
        }
    }

    private func loadStoredPredictions() {
        let fileURL = getPredictionsFileURL()

        // Migration: Try loading from encrypted file first, then fall back to UserDefaults
        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                trackedPredictions = try JSONDecoder().decode([TrackedPrediction].self, from: data)
                #if DEBUG
                print("✅ [OutcomeTracker] Loaded \(trackedPredictions.count) tracked predictions from secure storage")
                #endif
                return
            } catch {
                #if DEBUG
                print("❌ [OutcomeTracker] Failed to load from secure storage: \(error)")
                #endif
            }
        }

        // Fall back to UserDefaults for migration (one-time)
        if let data = userDefaults.data(forKey: storageKey) {
            do {
                trackedPredictions = try JSONDecoder().decode([TrackedPrediction].self, from: data)
                #if DEBUG
                print("✅ [OutcomeTracker] Migrated \(trackedPredictions.count) predictions from UserDefaults")
                #endif
                // Migrate to secure storage
                savePredictions()
                // Remove from UserDefaults after successful migration
                userDefaults.removeObject(forKey: storageKey)
            } catch {
                #if DEBUG
                print("❌ [OutcomeTracker] Failed to load predictions: \(error)")
                #endif
            }
        }
    }
}

// MARK: - Data Types

/// A tracked prediction with its outcome
public struct TrackedPrediction: Codable, Identifiable {
    public let id: UUID
    public let predictionDate: Date
    public let source: PredictionSourceType
    public let predictedProbability: Float
    public let predictedWillFlare: Bool
    public let confidence: Float
    public var outcomeDate: Date?
    public var actualFlare: Bool?
    public var actualFlareDate: Date?
    public let features: [String: Float]?
    public var wasValidated: Bool

    /// Whether the prediction was correct
    public var wasCorrect: Bool? {
        guard let actual = actualFlare else { return nil }
        return predictedWillFlare == actual
    }

    /// Absolute error between predicted probability and actual outcome
    public var absoluteError: Float? {
        guard let actual = actualFlare else { return nil }
        let actualValue: Float = actual ? 1.0 : 0.0
        return abs(predictedProbability - actualValue)
    }
}

/// Source of prediction
public enum PredictionSourceType: String, Codable, CaseIterable {
    case neuralEngine = "Neural Engine"
    case statistical = "Statistical"
    case hybrid = "Hybrid"
}

/// Overall accuracy metrics
public struct AccuracyMetrics {
    public let totalPredictions: Int
    public let correctPredictions: Int
    public let accuracy: Float
    public let precision: Float
    public let recall: Float
    public let f1Score: Float
    public let lastUpdated: Date

    public var accuracyPercentage: String {
        String(format: "%.1f%%", accuracy * 100)
    }
}

/// Calibration metrics with detailed bucket information
public struct OutcomeCalibrationMetrics {
    public let expectedCalibrationError: Float  // ECE - lower is better
    public let buckets: [CalibrationBucket]
    public let isWellCalibrated: Bool  // ECE < 0.1

    public var ecePercentage: String {
        String(format: "%.1f%%", expectedCalibrationError * 100)
    }
}

/// Calibration bucket for reliability diagram
public struct CalibrationBucket {
    public let bucketIndex: Int
    public let minProbability: Float
    public let maxProbability: Float
    public let predictedProbability: Float
    public let actualRate: Float
    public let count: Int
    public let calibrationError: Float
}

/// Accuracy by prediction source
public struct SourceAccuracy {
    public let source: PredictionSourceType
    public let totalPredictions: Int
    public let correctPredictions: Int
    public let accuracy: Float
    public let brierScore: Float
}

/// Backtest results
public struct BacktestResult {
    public let totalPredictions: Int
    public let correctPredictions: Int
    public let truePositives: Int
    public let trueNegatives: Int
    public let falsePositives: Int
    public let falseNegatives: Int
    public let accuracy: Float
    public let precision: Float
    public let recall: Float
    public let f1Score: Float
    public let brierScore: Float
    public let auc: Float
    public let testPeriodStart: Date
    public let testPeriodEnd: Date
    public let predictionDetails: [BacktestPredictionDetail]

    public var confusionMatrix: [[Int]] {
        [[truePositives, falseNegatives],
         [falsePositives, trueNegatives]]
    }

    public var summary: String {
        """
        Backtest Results (\(testPeriodStart.formatted(date: .abbreviated, time: .omitted)) - \(testPeriodEnd.formatted(date: .abbreviated, time: .omitted)))
        Predictions: \(totalPredictions)
        Accuracy: \(String(format: "%.1f%%", accuracy * 100))
        Precision: \(String(format: "%.1f%%", precision * 100))
        Recall: \(String(format: "%.1f%%", recall * 100))
        F1 Score: \(String(format: "%.2f", f1Score))
        AUC: \(String(format: "%.2f", auc))
        Brier Score: \(String(format: "%.3f", brierScore))
        """
    }
}

/// Individual backtest prediction detail
public struct BacktestPredictionDetail {
    public let date: Date
    public let predictedProbability: Float
    public let predictedFlare: Bool
    public let actualFlare: Bool
    public let wasCorrect: Bool
}
