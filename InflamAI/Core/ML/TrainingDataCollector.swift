//
//  TrainingDataCollector.swift
//  InflamAI
//
//  Shared utility for collecting training data from Core Data
//  Used by both MLUpdateService and NeuralFlarePredictionService
//

import Foundation
import CoreData

/// Centralized training data collection logic
class TrainingDataCollector {

    // MARK: - Configuration

    static let minimumDaysRequired = 37 // 30 for features + 7 for outcome window
    static let minimumHistoryDays = 30  // Days of history needed before each sample
    static let minimumDataQualityThreshold = 0.7 // 70% of days must have real data
    static let outcomeWindowStart = 3   // Days after feature extraction to start looking for flares
    static let outcomeWindowEnd = 7     // Days after feature extraction to stop looking for flares

    // MARK: - Public Methods

    /// Collect training data from Core Data
    /// Returns array of (30×92 features, flare label) tuples
    static func collectTrainingData(
        context: NSManagedObjectContext,
        featureExtractor: FeatureExtractor
    ) async throws -> [(features: [[Float]], label: Int)] {

        // 1. Fetch all symptom logs synchronously on the Core Data context
        let allLogs: [SymptomLog] = try await withCheckedThrowingContinuation { continuation in
            context.perform {
                let logsRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
                logsRequest.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]
                logsRequest.predicate = NSPredicate(format: "timestamp != nil")

                do {
                    let logs = try context.fetch(logsRequest)
                    if logs.isEmpty {
                        continuation.resume(throwing: CollectionError.noDataAvailable)
                    } else {
                        continuation.resume(returning: logs)
                    }
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }

        #if DEBUG
        print("📊 [TrainingDataCollector] Found \(allLogs.count) symptom logs")
        #endif

        // 2. Check minimum data requirement
        guard allLogs.count >= minimumDaysRequired else {
            #if DEBUG
            print("⚠️ [TrainingDataCollector] Insufficient data: \(allLogs.count)/\(minimumDaysRequired) days")
            #endif
            throw CollectionError.insufficientData(
                required: minimumDaysRequired,
                available: allLogs.count
            )
        }

        // 3. Collect samples (async feature extraction done outside Core Data perform block)
        var trainingSamples: [(features: [[Float]], label: Int)] = []
        let calendar = Calendar.current

        // Need to exclude last 7 days (for outcome verification)
        let logsForTraining = Array(allLogs.dropLast(outcomeWindowEnd))

        for (index, log) in logsForTraining.enumerated() {
            guard let currentDate = log.timestamp else { continue }

            // Skip if insufficient history
            if index < minimumHistoryDays { continue }

            // Extract features (async, done outside perform block)
            let features = await featureExtractor.extract30DayFeatures(endingOn: currentDate)

            // Validate feature shape
            guard validateFeatureShape(features) else {
                #if DEBUG
                print("⚠️ [TrainingDataCollector] Invalid shape for \(currentDate)")
                #endif
                continue
            }

            // Check data quality
            guard validateDataQuality(features) else {
                continue
            }

            // Determine outcome
            let label = determineFlareOutcome(
                from: currentDate,
                allLogs: allLogs,
                calendar: calendar
            )

            trainingSamples.append((features: features, label: label))
        }

        // 4. Log results
        #if DEBUG
        let flareCount = trainingSamples.filter { $0.label == 1 }.count
        let nonFlareCount = trainingSamples.filter { $0.label == 0 }.count

        print("✅ [TrainingDataCollector] Collected \(trainingSamples.count) samples")
        if trainingSamples.count > 0 {
            print("   Flare samples: \(flareCount) (\(String(format: "%.1f%%", Float(flareCount) / Float(trainingSamples.count) * 100)))")
        }
        print("   Non-flare samples: \(nonFlareCount)")

        // Warn if class imbalance is severe
        if trainingSamples.count > 0 {
            let flareRatio = Float(flareCount) / Float(trainingSamples.count)
            if flareRatio < 0.1 || flareRatio > 0.9 {
                print("⚠️ [TrainingDataCollector] Severe class imbalance detected!")
            }
        }
        #endif

        return trainingSamples
    }

    /// Check if user has sufficient data for training
    static func checkDataReadiness(context: NSManagedObjectContext) async -> DataReadinessInfo {
        return await context.perform {
            let logsRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            logsRequest.predicate = NSPredicate(format: "timestamp != nil")

            guard let logCount = try? context.count(for: logsRequest) else {
                return DataReadinessInfo(
                    isReady: false,
                    daysAvailable: 0,
                    daysRequired: minimumDaysRequired,
                    message: "Unable to check data availability",
                    estimatedSamplesAvailable: 0
                )
            }

            let isReady = logCount >= minimumDaysRequired
            let estimatedSamples = max(0, logCount - minimumDaysRequired)

            let message: String
            if isReady {
                message = "✅ Ready! \(logCount) days logged. Can create ~\(estimatedSamples) training samples."
            } else {
                let remaining = minimumDaysRequired - logCount
                message = "Keep logging! Need \(remaining) more days (\(logCount)/\(minimumDaysRequired))"
            }

            return DataReadinessInfo(
                isReady: isReady,
                daysAvailable: logCount,
                daysRequired: minimumDaysRequired,
                message: message,
                estimatedSamplesAvailable: estimatedSamples
            )
        }
    }

    // MARK: - Private Validation Methods

    /// Validate that features have correct shape (30 days × 92 features)
    private static func validateFeatureShape(_ features: [[Float]]) -> Bool {
        guard features.count == 30 else { return false }
        return features.allSatisfy { $0.count == 92 }
    }

    /// Validate that features have sufficient real data (not just padding)
    private static func validateDataQuality(_ features: [[Float]]) -> Bool {
        let nonZeroDays = features.filter { day in
            day.contains { $0 != 0.0 }
        }.count

        let requiredDays = Int(30.0 * minimumDataQualityThreshold)
        return nonZeroDays >= requiredDays
    }

    /// Determine if a flare occurred in the outcome window (3-7 days after date)
    private static func determineFlareOutcome(
        from date: Date,
        allLogs: [SymptomLog],
        calendar: Calendar
    ) -> Int {
        guard let windowStart = calendar.date(byAdding: .day, value: outcomeWindowStart, to: date),
              let windowEnd = calendar.date(byAdding: .day, value: outcomeWindowEnd, to: date) else {
            return 0
        }

        let flareInWindow = allLogs.contains { log in
            guard let logDate = log.timestamp else { return false }
            return logDate >= windowStart &&
                   logDate <= windowEnd &&
                   log.isFlareEvent
        }

        return flareInWindow ? 1 : 0
    }

    // MARK: - Errors

    enum CollectionError: LocalizedError {
        case noDataAvailable
        case insufficientData(required: Int, available: Int)
        case invalidFeatureShape

        var errorDescription: String? {
            switch self {
            case .noDataAvailable:
                return "No symptom logs found in database"
            case .insufficientData(let required, let available):
                return "Need \(required) days of data, only have \(available)"
            case .invalidFeatureShape:
                return "Features have invalid shape (expected 30×92)"
            }
        }
    }
}

// MARK: - Data Structures

public struct DataReadinessInfo {
    public let isReady: Bool
    public let daysAvailable: Int
    public let daysRequired: Int
    public let message: String
    public let estimatedSamplesAvailable: Int

    public var progressPercentage: Float {
        return min(1.0, Float(daysAvailable) / Float(daysRequired))
    }
}
