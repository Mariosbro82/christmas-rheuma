//
//  ValidationFramework.swift
//  InflamAI
//
//  Comprehensive model validation and performance tracking
//  Monitors accuracy, calibration, and trends over time
//

import Foundation
import CoreData

@MainActor
class ValidationFramework: ObservableObject {

    // MARK: - Published Properties

    @Published var overallMetrics: PerformanceMetrics?
    @Published var weeklyMetrics: [WeeklyPerformance] = []
    @Published var recentPredictions: [PredictionRecord] = []
    @Published var calibrationQuality: CalibrationMetrics?

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController

    // MARK: - Configuration

    private let maxRecords: Int = 1000  // Keep last 1000 predictions
    private let minSamplesForMetrics: Int = 10

    // MARK: - Initialization

    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
        loadValidationData()
        Task {
            await computeMetrics()
        }
    }

    // MARK: - Public API

    /// Record a new prediction and actual outcome
    func recordPrediction(
        prediction: Float,
        actualOutcome: Bool,
        features: [[Float]],
        timestamp: Date = Date()
    ) async {
        let record = PredictionRecord(
            id: UUID(),
            timestamp: timestamp,
            prediction: prediction,
            predictedClass: prediction >= 0.5,
            actualOutcome: actualOutcome,
            features: features,
            wasCorrect: (prediction >= 0.5) == actualOutcome
        )

        recentPredictions.append(record)

        // Trim if needed
        if recentPredictions.count > maxRecords {
            recentPredictions.removeFirst(recentPredictions.count - maxRecords)
        }

        // Persist
        saveValidationData()

        // Recompute metrics
        await computeMetrics()
    }

    /// Get current performance metrics
    func getCurrentMetrics() -> PerformanceMetrics? {
        return overallMetrics
    }

    /// Get performance trend (improving, stable, declining)
    func getPerformanceTrend() -> PerformanceTrend {
        guard weeklyMetrics.count >= 2 else {
            return .insufficient
        }

        let recent = weeklyMetrics.suffix(2)
        let oldAccuracy = recent.first!.metrics.accuracy
        let newAccuracy = recent.last!.metrics.accuracy

        let change = newAccuracy - oldAccuracy

        if change > 0.05 {
            return .improving
        } else if change < -0.05 {
            return .declining
        } else {
            return .stable
        }
    }

    /// Generate comprehensive validation report
    func generateReport() async -> ValidationReport {
        let metrics = overallMetrics ?? PerformanceMetrics.empty
        let trend = getPerformanceTrend()
        let calibration = calibrationQuality ?? CalibrationMetrics.empty

        // Compute benchmarks
        let benchmarks = computeBenchmarks()

        return ValidationReport(
            metrics: metrics,
            trend: trend,
            calibration: calibration,
            weeklyPerformance: weeklyMetrics,
            totalPredictions: recentPredictions.count,
            benchmarks: benchmarks,
            generatedAt: Date()
        )
    }

    // MARK: - Metrics Computation

    private func computeMetrics() async {
        guard recentPredictions.count >= minSamplesForMetrics else {
            overallMetrics = nil
            return
        }

        // Overall metrics
        overallMetrics = computePerformanceMetrics(for: recentPredictions)

        // Weekly breakdown
        weeklyMetrics = computeWeeklyMetrics()

        // Calibration
        calibrationQuality = computeCalibrationMetrics()
    }

    private func computePerformanceMetrics(for records: [PredictionRecord]) -> PerformanceMetrics {
        guard !records.isEmpty else {
            return PerformanceMetrics.empty
        }

        // Confusion matrix
        var truePositives = 0
        var trueNegatives = 0
        var falsePositives = 0
        var falseNegatives = 0

        for record in records {
            switch (record.predictedClass, record.actualOutcome) {
            case (true, true):
                truePositives += 1
            case (false, false):
                trueNegatives += 1
            case (true, false):
                falsePositives += 1
            case (false, true):
                falseNegatives += 1
            }
        }

        // Basic metrics
        let total = records.count
        let correct = truePositives + trueNegatives
        let accuracy = Float(correct) / Float(total)

        // Precision: TP / (TP + FP)
        let precision = truePositives + falsePositives > 0 ?
            Float(truePositives) / Float(truePositives + falsePositives) : 0.0

        // Recall (Sensitivity): TP / (TP + FN)
        let recall = truePositives + falseNegatives > 0 ?
            Float(truePositives) / Float(truePositives + falseNegatives) : 0.0

        // F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        let f1Score = precision + recall > 0 ?
            2.0 * (precision * recall) / (precision + recall) : 0.0

        // Specificity: TN / (TN + FP)
        let specificity = trueNegatives + falsePositives > 0 ?
            Float(trueNegatives) / Float(trueNegatives + falsePositives) : 0.0

        // ROC-AUC (approximate using Mann-Whitney U)
        let rocAuc = computeROCAUC(records: records)

        return PerformanceMetrics(
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1Score: f1Score,
            specificity: specificity,
            rocAuc: rocAuc,
            totalSamples: total,
            truePositives: truePositives,
            trueNegatives: trueNegatives,
            falsePositives: falsePositives,
            falseNegatives: falseNegatives
        )
    }

    private func computeWeeklyMetrics() -> [WeeklyPerformance] {
        guard !recentPredictions.isEmpty else { return [] }

        let calendar = Calendar.current
        let now = Date()

        // Group by week
        var weeklyGroups: [Int: [PredictionRecord]] = [:]

        for record in recentPredictions {
            let weekOffset = calendar.dateComponents([.weekOfYear], from: record.timestamp, to: now).weekOfYear ?? 0
            weeklyGroups[weekOffset, default: []].append(record)
        }

        // Compute metrics for each week
        var weeklyPerformance: [WeeklyPerformance] = []

        for (weekOffset, records) in weeklyGroups.sorted(by: { $0.key > $1.key }) {
            let metrics = computePerformanceMetrics(for: records)
            let weekStart = calendar.date(byAdding: .weekOfYear, value: -weekOffset, to: now)!

            weeklyPerformance.append(WeeklyPerformance(
                weekStart: weekStart,
                metrics: metrics,
                sampleCount: records.count
            ))
        }

        return weeklyPerformance.sorted { $0.weekStart < $1.weekStart }
    }

    private func computeCalibrationMetrics() -> CalibrationMetrics {
        guard recentPredictions.count >= minSamplesForMetrics else {
            return CalibrationMetrics.empty
        }

        let predictions = recentPredictions.map { $0.prediction }
        let outcomes = recentPredictions.map { $0.actualOutcome }

        // Expected Calibration Error (ECE)
        let ece = computeECE(predictions: predictions, outcomes: outcomes)

        // Brier Score (mean squared error of probabilities)
        var brierScore: Float = 0.0
        for (pred, outcome) in zip(predictions, outcomes) {
            let target: Float = outcome ? 1.0 : 0.0
            brierScore += pow(pred - target, 2)
        }
        brierScore /= Float(predictions.count)

        // Log Loss (cross-entropy)
        var logLoss: Float = 0.0
        let epsilon: Float = 1e-7
        for (pred, outcome) in zip(predictions, outcomes) {
            let clampedPred = max(epsilon, min(1.0 - epsilon, pred))
            let target: Float = outcome ? 1.0 : 0.0
            logLoss += -target * log(clampedPred) - (1 - target) * log(1 - clampedPred)
        }
        logLoss /= Float(predictions.count)

        return CalibrationMetrics(
            expectedCalibrationError: ece,
            brierScore: brierScore,
            logLoss: logLoss
        )
    }

    private func computeECE(predictions: [Float], outcomes: [Bool], numBins: Int = 10) -> Float {
        var bins: [[Int]] = Array(repeating: [], count: numBins)

        for (i, prob) in predictions.enumerated() {
            let binIndex = min(Int(prob * Float(numBins)), numBins - 1)
            bins[binIndex].append(i)
        }

        var ece: Float = 0.0
        let totalSamples = Float(predictions.count)

        for binIndices in bins where !binIndices.isEmpty {
            let binSize = Float(binIndices.count)
            let avgPrediction = binIndices.map { predictions[$0] }.reduce(0, +) / binSize
            let avgOutcome = binIndices.map { outcomes[$0] ? 1.0 : 0.0 }.reduce(0, +) / binSize
            ece += (binSize / totalSamples) * abs(avgPrediction - avgOutcome)
        }

        return ece
    }

    private func computeROCAUC(records: [PredictionRecord]) -> Float {
        // Mann-Whitney U statistic approximation
        let positives = records.filter { $0.actualOutcome }.map { $0.prediction }
        let negatives = records.filter { !$0.actualOutcome }.map { $0.prediction }

        guard !positives.isEmpty, !negatives.isEmpty else { return 0.5 }

        var sumRanks: Float = 0.0
        for posProb in positives {
            for negProb in negatives {
                if posProb > negProb {
                    sumRanks += 1.0
                } else if posProb == negProb {
                    sumRanks += 0.5
                }
            }
        }

        return sumRanks / Float(positives.count * negatives.count)
    }

    // MARK: - Benchmarks

    private func computeBenchmarks() -> Benchmarks {
        guard let metrics = overallMetrics else {
            return Benchmarks.empty
        }

        // Target benchmarks for 10/10 rating
        let targetAccuracy: Float = 0.75
        let targetF1: Float = 0.75
        let targetCalibration: Float = 0.10  // ECE < 0.10

        // Compare to targets
        let accuracyGap = metrics.accuracy - targetAccuracy
        let f1Gap = metrics.f1Score - targetF1
        let calibrationGap = (calibrationQuality?.expectedCalibrationError ?? 0.5) - targetCalibration

        return Benchmarks(
            targetAccuracy: targetAccuracy,
            currentAccuracy: metrics.accuracy,
            accuracyGap: accuracyGap,
            targetF1: targetF1,
            currentF1: metrics.f1Score,
            f1Gap: f1Gap,
            targetCalibration: targetCalibration,
            currentCalibration: calibrationQuality?.expectedCalibrationError ?? 0.5,
            calibrationGap: calibrationGap
        )
    }

    // MARK: - Persistence

    private func saveValidationData() {
        let cacheURL = getValidationCacheURL()

        do {
            let data = try JSONEncoder().encode(recentPredictions)
            try data.write(to: cacheURL)
        } catch {
            print("❌ Failed to save validation data: \(error)")
        }
    }

    private func loadValidationData() {
        let cacheURL = getValidationCacheURL()

        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            return
        }

        do {
            let data = try Data(contentsOf: cacheURL)
            recentPredictions = try JSONDecoder().decode([PredictionRecord].self, from: data)
            print("✅ Loaded \(recentPredictions.count) validation records")
        } catch {
            print("❌ Failed to load validation data: \(error)")
        }
    }

    private func getValidationCacheURL() -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent("validation_records.json")
    }
}

// MARK: - Data Types

struct PredictionRecord: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let prediction: Float              // Raw probability
    let predictedClass: Bool           // Predicted flare (yes/no)
    let actualOutcome: Bool            // Actual flare (yes/no)
    let features: [[Float]]            // Input features (for debugging)
    let wasCorrect: Bool
}

struct PerformanceMetrics {
    let accuracy: Float                // (TP + TN) / Total
    let precision: Float               // TP / (TP + FP)
    let recall: Float                  // TP / (TP + FN) - Sensitivity
    let f1Score: Float                 // Harmonic mean of precision/recall
    let specificity: Float             // TN / (TN + FP)
    let rocAuc: Float                  // Area under ROC curve
    let totalSamples: Int

    // Confusion matrix
    let truePositives: Int
    let trueNegatives: Int
    let falsePositives: Int
    let falseNegatives: Int

    static let empty = PerformanceMetrics(
        accuracy: 0, precision: 0, recall: 0, f1Score: 0,
        specificity: 0, rocAuc: 0, totalSamples: 0,
        truePositives: 0, trueNegatives: 0, falsePositives: 0, falseNegatives: 0
    )

    var rating: String {
        if accuracy >= 0.80 { return "Excellent" }
        if accuracy >= 0.70 { return "Good" }
        if accuracy >= 0.60 { return "Fair" }
        return "Needs Improvement"
    }

    var summary: String {
        """
        Accuracy: \(String(format: "%.1f%%", accuracy * 100))
        Precision: \(String(format: "%.1f%%", precision * 100))
        Recall: \(String(format: "%.1f%%", recall * 100))
        F1 Score: \(String(format: "%.3f", f1Score))
        ROC-AUC: \(String(format: "%.3f", rocAuc))
        Samples: \(totalSamples)
        """
    }
}

struct CalibrationMetrics {
    let expectedCalibrationError: Float  // Lower is better (< 0.10 = well calibrated)
    let brierScore: Float                // Lower is better (mean squared error)
    let logLoss: Float                   // Lower is better (cross-entropy)

    static let empty = CalibrationMetrics(
        expectedCalibrationError: 0.5,
        brierScore: 0.5,
        logLoss: 1.0
    )

    var isWellCalibrated: Bool {
        return expectedCalibrationError < 0.10
    }

    var calibrationRating: String {
        if expectedCalibrationError < 0.05 { return "Excellent" }
        if expectedCalibrationError < 0.10 { return "Good" }
        if expectedCalibrationError < 0.15 { return "Fair" }
        return "Poor"
    }
}

struct WeeklyPerformance {
    let weekStart: Date
    let metrics: PerformanceMetrics
    let sampleCount: Int

    var weekLabel: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: weekStart)
    }
}

enum PerformanceTrend {
    case improving      // Accuracy increasing
    case stable         // Accuracy stable
    case declining      // Accuracy decreasing
    case insufficient   // Not enough data

    var icon: String {
        switch self {
        case .improving: return "arrow.up.circle.fill"
        case .stable: return "minus.circle.fill"
        case .declining: return "arrow.down.circle.fill"
        case .insufficient: return "questionmark.circle.fill"
        }
    }

    var color: String {
        switch self {
        case .improving: return "green"
        case .stable: return "blue"
        case .declining: return "red"
        case .insufficient: return "gray"
        }
    }

    var description: String {
        switch self {
        case .improving: return "Performance improving over time"
        case .stable: return "Performance stable"
        case .declining: return "Performance declining - check data quality"
        case .insufficient: return "Collect more data to assess trends"
        }
    }
}

struct Benchmarks {
    let targetAccuracy: Float
    let currentAccuracy: Float
    let accuracyGap: Float

    let targetF1: Float
    let currentF1: Float
    let f1Gap: Float

    let targetCalibration: Float
    let currentCalibration: Float
    let calibrationGap: Float

    static let empty = Benchmarks(
        targetAccuracy: 0.75, currentAccuracy: 0, accuracyGap: -0.75,
        targetF1: 0.75, currentF1: 0, f1Gap: -0.75,
        targetCalibration: 0.10, currentCalibration: 0.5, calibrationGap: 0.4
    )

    var meetsAccuracyTarget: Bool { currentAccuracy >= targetAccuracy }
    var meetsF1Target: Bool { currentF1 >= targetF1 }
    var meetsCalibrationTarget: Bool { currentCalibration <= targetCalibration }

    var overallScore: Float {
        // 10/10 rating requires all targets met
        let accuracyScore = min(Float(1.0), currentAccuracy / targetAccuracy)
        let f1ScoreNorm = min(Float(1.0), currentF1 / targetF1)
        let calibrationScore: Float = currentCalibration <= targetCalibration ? 1.0 : 0.5

        return (accuracyScore + f1ScoreNorm + calibrationScore) / 3.0
    }

    var rating: Float {
        return overallScore * 10.0  // 0-10 scale
    }
}

struct ValidationReport {
    let metrics: PerformanceMetrics
    let trend: PerformanceTrend
    let calibration: CalibrationMetrics
    let weeklyPerformance: [WeeklyPerformance]
    let totalPredictions: Int
    let benchmarks: Benchmarks
    let generatedAt: Date

    var summary: String {
        """
        === MODEL VALIDATION REPORT ===
        Generated: \(generatedAt)

        OVERALL PERFORMANCE:
        \(metrics.summary)

        CALIBRATION:
        ECE: \(String(format: "%.3f", calibration.expectedCalibrationError)) (\(calibration.calibrationRating))
        Brier Score: \(String(format: "%.3f", calibration.brierScore))

        TREND: \(trend.description)

        BENCHMARKS:
        Accuracy: \(benchmarks.currentAccuracy >= benchmarks.targetAccuracy ? "✅" : "❌") \(String(format: "%.1f%%", benchmarks.currentAccuracy * 100)) (Target: \(String(format: "%.1f%%", benchmarks.targetAccuracy * 100)))
        F1 Score: \(benchmarks.currentF1 >= benchmarks.targetF1 ? "✅" : "❌") \(String(format: "%.3f", benchmarks.currentF1)) (Target: \(String(format: "%.3f", benchmarks.targetF1)))
        Calibration: \(benchmarks.meetsCalibrationTarget ? "✅" : "❌") ECE \(String(format: "%.3f", benchmarks.currentCalibration)) (Target: <\(String(format: "%.2f", benchmarks.targetCalibration)))

        OVERALL RATING: \(String(format: "%.1f", benchmarks.rating))/10
        """
    }
}
