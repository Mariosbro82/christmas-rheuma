//
//  MLIntegration.swift
//  InflamAI
//
//  Lightweight ML integration layer for DailyCheckIn
//  Works with existing NeuralEngineStub without breaking changes
//

import Foundation
import Combine

/// Lightweight ML integration service
/// Adds training data collection and auto-validation without replacing existing infrastructure
@MainActor
public final class MLIntegrationService: ObservableObject {

    // MARK: - Singleton

    public static let shared = MLIntegrationService()

    // MARK: - Published Properties

    @Published public var autoPersonalizationEnabled: Bool {
        didSet {
            UserDefaults.standard.set(autoPersonalizationEnabled, forKey: "mlAutoPersonalizationEnabled")
        }
    }

    @Published public private(set) var trainingSamplesCollected: Int = 0
    @Published public private(set) var lastTrainingSampleDate: Date?
    @Published public private(set) var predictionsTracked: Int = 0

    // MARK: - Private Properties

    private var trackedPredictions: [TrackedPrediction] = []
    private let userDefaultsKey = "mlTrackedPredictions"

    // MARK: - Data Types

    struct TrackedPrediction: Codable {
        let id: UUID
        let timestamp: Date
        let predictedProbability: Float
        let predictedFlare: Bool
        let basdaiScore: Double?
        var actualFlare: Bool?
        var validatedAt: Date?

        var isValidated: Bool {
            actualFlare != nil
        }
    }

    // MARK: - Initialization

    private init() {
        self.autoPersonalizationEnabled = UserDefaults.standard.object(forKey: "mlAutoPersonalizationEnabled") as? Bool ?? true
        loadTrackedPredictions()
    }

    // MARK: - Public API

    /// Record a training sample after check-in
    public func recordTrainingSample(basdaiScore: Double, isHighRisk: Bool) {
        guard autoPersonalizationEnabled else {
            print("ℹ️ [MLIntegration] Auto-personalization disabled - skipping sample")
            return
        }

        let prediction = TrackedPrediction(
            id: UUID(),
            timestamp: Date(),
            predictedProbability: Float(basdaiScore / 10.0),  // Normalize BASDAI to 0-1
            predictedFlare: isHighRisk,
            basdaiScore: basdaiScore,
            actualFlare: nil,
            validatedAt: nil
        )

        trackedPredictions.append(prediction)
        trainingSamplesCollected = trackedPredictions.count
        lastTrainingSampleDate = Date()

        saveTrackedPredictions()

        print("✅ [MLIntegration] Recorded training sample #\(trainingSamplesCollected) (BASDAI: \(String(format: "%.1f", basdaiScore)), High Risk: \(isHighRisk))")
    }

    /// Auto-validate recent predictions against flare events
    public func autoValidatePredictions(flareOccurred: Bool, flareDate: Date? = nil) {
        let validationDate = flareDate ?? Date()
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: validationDate)!

        var validatedCount = 0

        for i in trackedPredictions.indices {
            let prediction = trackedPredictions[i]

            // Skip already validated
            guard !prediction.isValidated else { continue }

            // Check if prediction was within 7 days before the flare/validation date
            if prediction.timestamp >= sevenDaysAgo && prediction.timestamp <= validationDate {
                trackedPredictions[i].actualFlare = flareOccurred
                trackedPredictions[i].validatedAt = Date()
                validatedCount += 1
            }
        }

        if validatedCount > 0 {
            saveTrackedPredictions()
            print("✅ [MLIntegration] Validated \(validatedCount) predictions (flare: \(flareOccurred))")
        }
    }

    /// Get accuracy metrics
    public func getAccuracyMetrics() -> AccuracyMetrics {
        let validated = trackedPredictions.filter { $0.isValidated }

        guard !validated.isEmpty else {
            return AccuracyMetrics(
                accuracy: 0,
                precision: 0,
                recall: 0,
                totalPredictions: 0,
                validatedPredictions: 0
            )
        }

        var truePositives = 0
        var falsePositives = 0
        var trueNegatives = 0
        var falseNegatives = 0

        for prediction in validated {
            let predicted = prediction.predictedFlare
            let actual = prediction.actualFlare ?? false

            if predicted && actual { truePositives += 1 }
            else if predicted && !actual { falsePositives += 1 }
            else if !predicted && !actual { trueNegatives += 1 }
            else { falseNegatives += 1 }
        }

        let total = validated.count
        let accuracy = Float(truePositives + trueNegatives) / Float(total)
        let precision = truePositives + falsePositives > 0 ? Float(truePositives) / Float(truePositives + falsePositives) : 0
        let recall = truePositives + falseNegatives > 0 ? Float(truePositives) / Float(truePositives + falseNegatives) : 0

        return AccuracyMetrics(
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            totalPredictions: trackedPredictions.count,
            validatedPredictions: validated.count
        )
    }

    /// Clear all tracked data
    public func clearAllData() {
        trackedPredictions = []
        trainingSamplesCollected = 0
        lastTrainingSampleDate = nil
        predictionsTracked = 0
        UserDefaults.standard.removeObject(forKey: userDefaultsKey)
        print("🗑️ [MLIntegration] Cleared all tracked data")
    }

    // MARK: - Persistence

    private func saveTrackedPredictions() {
        if let data = try? JSONEncoder().encode(trackedPredictions) {
            UserDefaults.standard.set(data, forKey: userDefaultsKey)
        }
    }

    private func loadTrackedPredictions() {
        if let data = UserDefaults.standard.data(forKey: userDefaultsKey),
           let predictions = try? JSONDecoder().decode([TrackedPrediction].self, from: data) {
            self.trackedPredictions = predictions
            self.trainingSamplesCollected = predictions.count
            self.lastTrainingSampleDate = predictions.last?.timestamp
        }
    }

    // MARK: - Data Types

    public struct AccuracyMetrics {
        public let accuracy: Float
        public let precision: Float
        public let recall: Float
        public let totalPredictions: Int
        public let validatedPredictions: Int

        public var f1Score: Float {
            guard precision + recall > 0 else { return 0 }
            return 2 * (precision * recall) / (precision + recall)
        }

        public var hasEnoughData: Bool {
            validatedPredictions >= 10
        }

        public var summary: String {
            if !hasEnoughData {
                return "Need \(10 - validatedPredictions) more validated predictions"
            }
            let accuracyPercent = Int(accuracy * 100)
            return "Accuracy: \(accuracyPercent)% (\(validatedPredictions) validated)"
        }
    }
}
