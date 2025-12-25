//
//  KNNTriggerEngine.swift
//  InflamAI
//
//  k-Nearest Neighbors engine for personalized trigger detection
//  Finds similar historical days to predict outcomes and identify triggers
//
//  Activates at 30+ days of data
//  Non-parametric, handles non-linear relationships
//  Fully explainable through similar day visualization
//

import Foundation
import CoreData
import Combine
import Accelerate

// MARK: - KNNTriggerEngine

@MainActor
public final class KNNTriggerEngine: ObservableObject {

    // MARK: - Singleton

    public static let shared = KNNTriggerEngine()

    // MARK: - Published State

    @Published public private(set) var isReady: Bool = false
    @Published public private(set) var trainingDaysCount: Int = 0
    @Published public private(set) var lastTrainingDate: Date?
    @Published public private(set) var isProcessing: Bool = false
    @Published public private(set) var errorMessage: String?

    // MARK: - Configuration

    public struct Configuration {
        /// Number of neighbors to consider
        public var k: Int = 5

        /// Minimum days required for k-NN
        public var minimumDays: Int = 30

        /// Maximum distance for a day to be considered "similar"
        public var maxDistance: Double = 3.0

        /// Feature weights
        public var featureWeights: [String: Double] = [:]

        /// Whether to use adaptive k (based on local density)
        public var adaptiveK: Bool = true

        /// Distance metric
        public var distanceMetric: DistanceMetric = .euclidean

        public enum DistanceMetric {
            case euclidean
            case manhattan
            case cosine
        }

        public static let `default` = Configuration()
    }

    public var configuration: Configuration

    // MARK: - Feature Names

    /// Standard features used for similarity calculation
    private let standardFeatures: [String] = [
        // Sleep
        "sleep_duration", "sleep_quality",
        // Activity
        "steps", "exercise",
        // Stress
        "stress", "anxiety",
        // Weather
        "pressure_drop", "high_humidity", "cold_temperature",
        // Food/Drink
        "coffee", "alcohol"
    ]

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private let triggerDataService: TriggerDataService
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Training Data

    private var trainingData: [DayFeatureVector] = []
    private var featureMeans: [String: Double] = [:]
    private var featureStdDevs: [String: Double] = [:]

    // MARK: - Initialization

    private init(
        persistenceController: InflamAIPersistenceController = .shared,
        triggerDataService: TriggerDataService = .shared,
        configuration: Configuration = .default
    ) {
        self.persistenceController = persistenceController
        self.triggerDataService = triggerDataService
        self.configuration = configuration
    }

    // MARK: - Context

    private var viewContext: NSManagedObjectContext {
        persistenceController.container.viewContext
    }

    // MARK: - Training

    /// Build k-NN model from historical data
    public func train() async {
        isProcessing = true
        errorMessage = nil
        defer { isProcessing = false }

        // Fetch all symptom logs
        let symptomLogs = fetchAllSymptomLogs()

        guard symptomLogs.count >= configuration.minimumDays else {
            errorMessage = "Need at least \(configuration.minimumDays) days of data"
            isReady = false
            return
        }

        // Build feature vectors for each day
        var vectors: [DayFeatureVector] = []

        let calendar = Calendar.current
        let groupedLogs = Dictionary(grouping: symptomLogs) { log -> Date in
            calendar.startOfDay(for: log.timestamp ?? Date())
        }

        for (date, logs) in groupedLogs {
            // Get the average pain for this day
            let avgPain = logs.map { $0.basdaiScore }.reduce(0, +) / Double(logs.count)

            // Get trigger values for this day
            let triggers = triggerDataService.getTriggersAsDict(for: date)

            // Build feature vector
            var features: [String: Double] = [:]

            for featureName in standardFeatures {
                features[featureName] = triggers[featureName] ?? 0
            }

            // Add context from ContextSnapshot if available
            if let log = logs.first, let context = log.contextSnapshot {
                features["pressure_drop"] = context.pressureChange12h
                features["high_humidity"] = Double(context.humidity)
                features["cold_temperature"] = context.temperature < 10 ? 1 : 0
                features["steps"] = Double(context.stepCount)
                features["sleep_duration"] = context.sleepDuration
            }

            vectors.append(DayFeatureVector(
                date: date,
                painLevel: avgPain,
                features: features,
                triggerLogs: logs.first?.triggerLogs?.allObjects as? [TriggerLog] ?? []
            ))
        }

        // Normalize features
        normalizeFeatures(&vectors)

        trainingData = vectors.sorted { $0.date < $1.date }
        trainingDaysCount = trainingData.count
        lastTrainingDate = Date()
        isReady = trainingData.count >= configuration.minimumDays
    }

    /// Normalize features using z-score normalization
    private func normalizeFeatures(_ vectors: inout [DayFeatureVector]) {
        // Calculate means and standard deviations
        for feature in standardFeatures {
            let values = vectors.compactMap { $0.features[feature] }
            guard !values.isEmpty else { continue }

            let mean = values.reduce(0, +) / Double(values.count)
            let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count)
            let stdDev = sqrt(variance)

            featureMeans[feature] = mean
            featureStdDevs[feature] = stdDev > 0 ? stdDev : 1.0
        }

        // Apply normalization
        for i in 0..<vectors.count {
            var normalizedFeatures: [String: Double] = [:]

            for (feature, value) in vectors[i].features {
                let mean = featureMeans[feature] ?? 0
                let stdDev = featureStdDevs[feature] ?? 1

                normalizedFeatures[feature] = (value - mean) / stdDev
            }

            vectors[i] = DayFeatureVector(
                date: vectors[i].date,
                painLevel: vectors[i].painLevel,
                features: normalizedFeatures,
                rawFeatures: vectors[i].features,
                triggerLogs: vectors[i].triggerLogs
            )
        }
    }

    // MARK: - Prediction

    /// Find k nearest neighbors for a query day
    public func findSimilarDays(
        for queryFeatures: [String: Double],
        k: Int? = nil
    ) -> [SimilarDay] {
        guard isReady else { return [] }

        let neighborCount = k ?? configuration.k

        // Normalize query features
        var normalizedQuery: [String: Double] = [:]
        for (feature, value) in queryFeatures {
            let mean = featureMeans[feature] ?? 0
            let stdDev = featureStdDevs[feature] ?? 1
            normalizedQuery[feature] = (value - mean) / stdDev
        }

        // Calculate distances
        var distances: [(index: Int, distance: Double)] = []

        for (index, vector) in trainingData.enumerated() {
            let distance = calculateDistance(normalizedQuery, vector.features)
            distances.append((index, distance))
        }

        // Sort by distance and take k nearest
        distances.sort { $0.distance < $1.distance }
        let nearest = distances.prefix(neighborCount)

        // Convert to SimilarDay
        return nearest.map { item in
            let vector = trainingData[item.index]

            return SimilarDay(
                date: vector.date,
                painLevel: vector.painLevel,
                distance: item.distance,
                triggers: vector.triggerLogs.map { $0.toTriggerValue() },
                keyFeatures: vector.rawFeatures ?? vector.features
            )
        }
    }

    /// Predict pain level for query features
    public func predictPain(for queryFeatures: [String: Double]) -> KNNPrediction {
        let similarDays = findSimilarDays(for: queryFeatures)

        guard !similarDays.isEmpty else {
            return KNNPrediction(
                predictedPain: 0,
                confidence: .insufficient,
                similarDays: [],
                commonTriggers: []
            )
        }

        // Weight by inverse distance
        var weightedSum: Double = 0
        var totalWeight: Double = 0

        for day in similarDays {
            let weight = 1.0 / max(0.1, day.distance)
            weightedSum += day.painLevel * weight
            totalWeight += weight
        }

        let predictedPain = totalWeight > 0 ? weightedSum / totalWeight : 0

        // Calculate confidence based on neighbor consistency
        let painValues = similarDays.map { $0.painLevel }
        let stdDev = painValues.standardDeviation()
        let avgDistance = similarDays.map { $0.distance }.reduce(0, +) / Double(similarDays.count)

        let confidence: TriggerConfidence
        if avgDistance < 1.0 && stdDev < 1.0 {
            confidence = .high
        } else if avgDistance < 2.0 && stdDev < 2.0 {
            confidence = .medium
        } else if similarDays.count >= 3 {
            confidence = .low
        } else {
            confidence = .insufficient
        }

        // Find common triggers among similar high-pain days
        let commonTriggers = findCommonTriggers(in: similarDays)

        return KNNPrediction(
            predictedPain: predictedPain,
            confidence: confidence,
            similarDays: similarDays,
            commonTriggers: commonTriggers
        )
    }

    /// Predict pain for tomorrow based on today's logged triggers
    public func predictTomorrow() async -> KNNPrediction {
        // Get today's triggers
        let todaysTriggers = await triggerDataService.todaysTriggers
        var features: [String: Double] = [:]

        for trigger in todaysTriggers {
            if let name = trigger.triggerName {
                features[name] = trigger.triggerValue
            }
        }

        return predictPain(for: features)
    }

    // MARK: - Distance Calculation

    /// Calculate distance between two feature vectors
    private func calculateDistance(
        _ a: [String: Double],
        _ b: [String: Double]
    ) -> Double {
        switch configuration.distanceMetric {
        case .euclidean:
            return euclideanDistance(a, b)
        case .manhattan:
            return manhattanDistance(a, b)
        case .cosine:
            return cosineDistance(a, b)
        }
    }

    private func euclideanDistance(
        _ a: [String: Double],
        _ b: [String: Double]
    ) -> Double {
        var sumSquared: Double = 0

        for feature in standardFeatures {
            let aVal = a[feature] ?? 0
            let bVal = b[feature] ?? 0
            let weight = configuration.featureWeights[feature] ?? 1.0
            let diff = (aVal - bVal) * weight
            sumSquared += diff * diff
        }

        return sqrt(sumSquared)
    }

    private func manhattanDistance(
        _ a: [String: Double],
        _ b: [String: Double]
    ) -> Double {
        var sum: Double = 0

        for feature in standardFeatures {
            let aVal = a[feature] ?? 0
            let bVal = b[feature] ?? 0
            let weight = configuration.featureWeights[feature] ?? 1.0
            sum += abs(aVal - bVal) * weight
        }

        return sum
    }

    private func cosineDistance(
        _ a: [String: Double],
        _ b: [String: Double]
    ) -> Double {
        var dotProduct: Double = 0
        var normA: Double = 0
        var normB: Double = 0

        for feature in standardFeatures {
            let aVal = a[feature] ?? 0
            let bVal = b[feature] ?? 0
            let weight = configuration.featureWeights[feature] ?? 1.0

            dotProduct += aVal * bVal * weight
            normA += aVal * aVal * weight
            normB += bVal * bVal * weight
        }

        let denominator = sqrt(normA) * sqrt(normB)
        if denominator == 0 { return 1.0 }

        let similarity = dotProduct / denominator
        return 1.0 - similarity  // Convert to distance
    }

    // MARK: - Trigger Analysis

    /// Find triggers that commonly appear among similar high-pain days
    private func findCommonTriggers(in similarDays: [SimilarDay]) -> [CommonTrigger] {
        let highPainDays = similarDays.filter { $0.painLevel > 5 }
        guard !highPainDays.isEmpty else { return [] }

        // Count trigger occurrences
        var triggerCounts: [String: (count: Int, totalValue: Double)] = [:]

        for day in highPainDays {
            for trigger in day.triggers {
                if trigger.isPresent {
                    let current = triggerCounts[trigger.name] ?? (0, 0)
                    triggerCounts[trigger.name] = (current.count + 1, current.totalValue + trigger.value)
                }
            }
        }

        // Convert to CommonTrigger
        return triggerCounts.map { (name, data) in
            CommonTrigger(
                name: name,
                frequency: Double(data.count) / Double(highPainDays.count),
                averageValue: data.totalValue / Double(data.count)
            )
        }.sorted { $0.frequency > $1.frequency }
    }

    /// Analyze a specific trigger using k-NN approach
    public func analyzeTrigger(named triggerName: String) -> KNNTriggerAnalysis? {
        guard isReady else { return nil }

        // Split days by trigger presence
        var daysWithTrigger: [DayFeatureVector] = []
        var daysWithoutTrigger: [DayFeatureVector] = []

        for vector in trainingData {
            let triggerValue = vector.features[triggerName] ?? vector.rawFeatures?[triggerName] ?? 0
            if triggerValue > 0 {
                daysWithTrigger.append(vector)
            } else {
                daysWithoutTrigger.append(vector)
            }
        }

        guard !daysWithTrigger.isEmpty else {
            return KNNTriggerAnalysis(
                triggerName: triggerName,
                daysWithTrigger: 0,
                daysWithoutTrigger: daysWithoutTrigger.count,
                avgPainWithTrigger: 0,
                avgPainWithoutTrigger: daysWithoutTrigger.map { $0.painLevel }.mean(),
                predictedImpact: 0,
                confidence: .insufficient,
                similarHighPainDays: [],
                coOccurringTriggers: [:]
            )
        }

        let avgPainWith = daysWithTrigger.map { $0.painLevel }.mean()
        let avgPainWithout = daysWithoutTrigger.isEmpty ? 0 : daysWithoutTrigger.map { $0.painLevel }.mean()
        let impact = avgPainWith - avgPainWithout

        // Find high-pain days with this trigger
        let highPainDays = daysWithTrigger
            .filter { $0.painLevel > 5 }
            .sorted { $0.painLevel > $1.painLevel }
            .prefix(5)

        let similarDays = highPainDays.map { vector in
            SimilarDay(
                date: vector.date,
                painLevel: vector.painLevel,
                distance: 0,
                triggers: vector.triggerLogs.map { $0.toTriggerValue() },
                keyFeatures: vector.rawFeatures ?? vector.features
            )
        }

        // Find co-occurring triggers
        let coOccurring = findCoOccurringTriggers(
            with: triggerName,
            in: daysWithTrigger
        )

        // Determine confidence
        let confidence: TriggerConfidence
        if daysWithTrigger.count >= 15 && daysWithoutTrigger.count >= 15 {
            confidence = .high
        } else if daysWithTrigger.count >= 7 && daysWithoutTrigger.count >= 7 {
            confidence = .medium
        } else if daysWithTrigger.count >= 3 {
            confidence = .low
        } else {
            confidence = .insufficient
        }

        return KNNTriggerAnalysis(
            triggerName: triggerName,
            daysWithTrigger: daysWithTrigger.count,
            daysWithoutTrigger: daysWithoutTrigger.count,
            avgPainWithTrigger: avgPainWith,
            avgPainWithoutTrigger: avgPainWithout,
            predictedImpact: impact,
            confidence: confidence,
            similarHighPainDays: Array(similarDays),
            coOccurringTriggers: coOccurring
        )
    }

    /// Find triggers that often occur together with a given trigger
    private func findCoOccurringTriggers(
        with targetTrigger: String,
        in days: [DayFeatureVector]
    ) -> [String: Double] {
        var coOccurrence: [String: Int] = [:]
        let totalDays = days.count

        for day in days {
            for (feature, value) in (day.rawFeatures ?? day.features) {
                if feature != targetTrigger && value > 0 {
                    coOccurrence[feature, default: 0] += 1
                }
            }
        }

        // Convert to frequencies
        var frequencies: [String: Double] = [:]
        for (trigger, count) in coOccurrence {
            frequencies[trigger] = Double(count) / Double(totalDays)
        }

        return frequencies
    }

    // MARK: - Data Fetching

    private func fetchAllSymptomLogs() -> [SymptomLog] {
        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]
        return (try? viewContext.fetch(request)) ?? []
    }

    // MARK: - Feature Weight Learning

    /// Learn optimal feature weights from data
    public func learnFeatureWeights() async {
        guard isReady else { return }

        // Use leave-one-out cross-validation to optimize weights
        // This is a simplified version - a full implementation would use gradient descent

        for feature in standardFeatures {
            var bestWeight: Double = 1.0
            var bestError: Double = .infinity

            for weight in stride(from: 0.5, through: 2.0, by: 0.25) {
                configuration.featureWeights[feature] = weight

                // Calculate leave-one-out error
                var totalError: Double = 0
                for i in 0..<trainingData.count {
                    let query = trainingData[i].features
                    let actualPain = trainingData[i].painLevel

                    // Temporarily exclude this day
                    let originalData = trainingData
                    trainingData.remove(at: i)

                    let prediction = predictPain(for: query)
                    totalError += pow(prediction.predictedPain - actualPain, 2)

                    trainingData = originalData
                }

                if totalError < bestError {
                    bestError = totalError
                    bestWeight = weight
                }
            }

            configuration.featureWeights[feature] = bestWeight
        }
    }
}

// MARK: - Day Feature Vector

/// Internal representation of a day's feature vector
struct DayFeatureVector {
    let date: Date
    let painLevel: Double
    let features: [String: Double]  // Normalized
    let rawFeatures: [String: Double]?  // Original values
    let triggerLogs: [TriggerLog]

    init(
        date: Date,
        painLevel: Double,
        features: [String: Double],
        rawFeatures: [String: Double]? = nil,
        triggerLogs: [TriggerLog]
    ) {
        self.date = date
        self.painLevel = painLevel
        self.features = features
        self.rawFeatures = rawFeatures ?? features
        self.triggerLogs = triggerLogs
    }
}

// MARK: - k-NN Prediction

/// Result of k-NN pain prediction
public struct KNNPrediction {
    public let predictedPain: Double
    public let confidence: TriggerConfidence
    public let similarDays: [SimilarDay]
    public let commonTriggers: [CommonTrigger]

    public var predictedLevel: String {
        switch predictedPain {
        case 0..<2: return "Low"
        case 2..<4: return "Mild"
        case 4..<6: return "Moderate"
        case 6..<8: return "High"
        default: return "Severe"
        }
    }

    public var explanation: String {
        if similarDays.isEmpty {
            return "Not enough historical data for prediction"
        }

        let avgPain = String(format: "%.1f", predictedPain)
        return "Based on \(similarDays.count) similar days, predicted pain level: \(avgPain)"
    }
}

// MARK: - k-NN Trigger Analysis

/// Detailed k-NN analysis of a specific trigger
public struct KNNTriggerAnalysis {
    public let triggerName: String
    public let daysWithTrigger: Int
    public let daysWithoutTrigger: Int
    public let avgPainWithTrigger: Double
    public let avgPainWithoutTrigger: Double
    public let predictedImpact: Double
    public let confidence: TriggerConfidence
    public let similarHighPainDays: [SimilarDay]
    public let coOccurringTriggers: [String: Double]

    public var impactDescription: String {
        if abs(predictedImpact) < 0.5 {
            return "Minimal impact on symptoms"
        } else if predictedImpact > 0 {
            return "Increases symptoms by \(String(format: "%.1f", predictedImpact)) points"
        } else {
            return "Decreases symptoms by \(String(format: "%.1f", abs(predictedImpact))) points"
        }
    }

    public var topCoOccurring: [(trigger: String, frequency: Double)] {
        coOccurringTriggers
            .sorted { $0.value > $1.value }
            .prefix(3)
            .map { ($0.key, $0.value) }
    }
}
