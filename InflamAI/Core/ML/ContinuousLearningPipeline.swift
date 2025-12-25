//
//  ContinuousLearningPipeline.swift
//  InflamAI
//
//  On-device continuous learning using MLUpdateTask
//  Personalizes the model to user's unique patterns over time
//

import Foundation
import CoreML
import CoreData

@available(iOS 17.0, *)
@MainActor
class ContinuousLearningPipeline: ObservableObject {

    // MARK: - Published Properties

    @Published var personalizationProgress: Float = 0.0  // 0-1
    @Published var totalPersonalSamples: Int = 0
    @Published var lastUpdateDate: Date?
    @Published var modelVersion: Int = 0  // Increments with each update
    @Published var personalizationPhase: PersonalizationPhase = .bootstrap

    enum PersonalizationPhase: String {
        case bootstrap = "Learning"          // Days 1-7: Collecting initial data
        case earlyAdaptation = "Adapting"   // Days 8-21: First updates
        case personalized = "Personalized"   // Days 22+: Fully customized
        case expert = "Expert"               // 90+ days: Deep expertise

        var daysRequired: Int {
            switch self {
            case .bootstrap: return 0
            case .earlyAdaptation: return 8
            case .personalized: return 22
            case .expert: return 90
            }
        }

        var description: String {
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
    }

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private let featureExtractor: FeatureExtractor
    private let calibrationEngine: CalibrationEngine

    // MARK: - Configuration

    private let minSamplesForUpdate: Int = 7          // Minimum samples before first update
    private let updateFrequency: TimeInterval = 7 * 24 * 3600  // Weekly updates
    private let maxTrainingTime: TimeInterval = 300   // 5 minutes max
    private let validationSplit: Float = 0.2          // 20% for validation

    // MARK: - Training Data Management

    struct TrainingSample: Codable {
        let features: [[Float]]  // 30 days × 92 features
        let label: Bool          // Flare outcome
        let timestamp: Date
        let sampleID: UUID
    }

    private var trainingDataCache: [TrainingSample] = []
    private let maxCacheSize: Int = 1000  // Retain last 1000 samples

    // MARK: - Initialization

    init(
        persistenceController: InflamAIPersistenceController = .shared,
        featureExtractor: FeatureExtractor,
        calibrationEngine: CalibrationEngine
    ) {
        self.persistenceController = persistenceController
        self.featureExtractor = featureExtractor
        self.calibrationEngine = calibrationEngine

        // Load cached training data
        loadTrainingDataCache()
        updatePersonalizationMetrics()
    }

    // MARK: - Public API

    /// Check if user has sufficient data for model training
    func checkDataReadiness() async -> DataReadinessStatus {
        let context = persistenceController.container.viewContext

        let logsRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        logsRequest.predicate = NSPredicate(format: "timestamp != nil")

        guard let logCount = try? context.count(for: logsRequest) else {
            return DataReadinessStatus(
                isReady: false,
                daysAvailable: 0,
                daysRequired: 37,
                message: "Unable to check data availability"
            )
        }

        let isReady = logCount >= 37
        let message: String

        if isReady {
            message = "Ready for personalization! \(logCount) days of data available."
        } else {
            let remaining = 37 - logCount
            message = "Keep logging! Need \(remaining) more days (\(logCount)/37)"
        }

        return DataReadinessStatus(
            isReady: isReady,
            daysAvailable: logCount,
            daysRequired: 37,
            message: message
        )
    }

    /// Add a new training sample after user logs symptoms
    func addTrainingSample(
        features: [[Float]],
        actualOutcome: Bool
    ) async throws {
        let sample = TrainingSample(
            features: features,
            label: actualOutcome,
            timestamp: Date(),
            sampleID: UUID()
        )

        trainingDataCache.append(sample)
        totalPersonalSamples = trainingDataCache.count

        // Trim cache if needed
        if trainingDataCache.count > maxCacheSize {
            trainingDataCache.removeFirst(trainingDataCache.count - maxCacheSize)
        }

        // Persist to disk
        saveTrainingDataCache()

        // Update metrics
        updatePersonalizationMetrics()

        // Check if we should trigger an update
        if shouldTriggerUpdate() {
            try await performModelUpdate()
        }
    }

    /// Manually trigger model update (for testing or user-initiated)
    func triggerManualUpdate() async throws {
        guard trainingDataCache.count >= minSamplesForUpdate else {
            throw LearningError.insufficientData(
                required: minSamplesForUpdate,
                available: trainingDataCache.count
            )
        }

        try await performModelUpdate()
    }

    /// Get personalization status for UI display
    func getPersonalizationStatus() -> LearningPipelineStatus {
        let daysOfData = totalPersonalSamples > 0 ?
            Calendar.current.dateComponents(
                [.day],
                from: trainingDataCache.first?.timestamp ?? Date(),
                to: Date()
            ).day ?? 0 : 0

        return LearningPipelineStatus(
            phase: personalizationPhase,
            progress: personalizationProgress,
            totalSamples: totalPersonalSamples,
            daysOfData: daysOfData,
            lastUpdate: lastUpdateDate,
            modelVersion: modelVersion,
            nextUpdateIn: calculateNextUpdateInterval()
        )
    }

    // MARK: - Model Update Logic

    private func performModelUpdate() async throws {
        print("🔄 Starting on-device model update...")
        let startTime = Date()

        // 1. Prepare training data
        let (trainingSamples, validationSamples) = splitTrainingData()

        guard !trainingSamples.isEmpty else {
            throw LearningError.insufficientData(
                required: minSamplesForUpdate,
                available: trainingDataCache.count
            )
        }

        // 2. Convert to MLBatchProvider
        let trainingBatch = try createBatchProvider(from: trainingSamples)
        let validationBatch = try createBatchProvider(from: validationSamples)

        // 3. Configure update task
        let updateTask = try MLUpdateTask(
            forModelAt: getModelURL(),
            trainingData: trainingBatch,
            configuration: createUpdateConfiguration(),
            completionHandler: { context in
                Task { @MainActor in
                    await self.handleUpdateCompletion(
                        context: context,
                        validationData: validationBatch,
                        startTime: startTime
                    )
                }
            }
        )

        // 4. Start update
        updateTask.resume()
    }

    private func handleUpdateCompletion(
        context: MLUpdateContext,
        validationData: MLBatchProvider,
        startTime: Date
    ) async {
        let duration = Date().timeIntervalSince(startTime)

        switch context.task.state {
        case .completed:
            // Validate updated model
            do {
                // Note: After MLUpdateTask, the updated model is saved to a temp location
                // We need to write it to our app's model storage location
                let updatedModelURL = getUpdatedModelStorageURL()
                try context.model.write(to: updatedModelURL)

                let isValid = try await validateUpdatedModel(
                    modelURL: updatedModelURL,
                    validationData: validationData
                )

                if isValid {
                    // Deploy updated model
                    try deployUpdatedModel(from: updatedModelURL)

                    modelVersion += 1
                    lastUpdateDate = Date()
                    personalizationProgress = calculateProgress()

                    print("✅ Model update successful! Version \(modelVersion)")
                    print("⏱️ Update took \(String(format: "%.1f", duration))s")
                } else {
                    print("⚠️ Updated model failed validation - keeping current model")
                }
            } catch {
                print("❌ Model update failed: \(error.localizedDescription)")
            }

        case .failed:
            print("❌ Model update task failed")

        default:
            break
        }
    }

    // MARK: - Training Data Preparation

    private func splitTrainingData() -> (training: [TrainingSample], validation: [TrainingSample]) {
        let shuffled = trainingDataCache.shuffled()
        let splitIndex = Int(Float(shuffled.count) * (1.0 - validationSplit))

        let training = Array(shuffled[..<splitIndex])
        let validation = Array(shuffled[splitIndex...])

        return (training, validation)
    }

    private func createBatchProvider(from samples: [TrainingSample]) throws -> MLBatchProvider {
        var featureProviders: [MLFeatureProvider] = []

        for sample in samples {
            // Create input MLMultiArray
            let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
            for (i, timestep) in sample.features.enumerated() {
                for (j, value) in timestep.enumerated() {
                    inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
                }
            }

            // Create label array
            let labelArray = try MLMultiArray(shape: [1], dataType: .int32)
            labelArray[0] = NSNumber(value: sample.label ? 1 : 0)

            // Create feature provider
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "features": MLFeatureValue(multiArray: inputArray),
                "flare_label": MLFeatureValue(multiArray: labelArray)
            ])

            featureProviders.append(provider)
        }

        return MLArrayBatchProvider(array: featureProviders)
    }

    private func createUpdateConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine + GPU
        config.allowLowPrecisionAccumulationOnGPU = true  // Faster training

        // Update task parameters (these would be set if MLUpdateTask supported it)
        // In practice, these are configured in the .mlmodel file during export
        // - Learning rate: 0.0001
        // - Epochs: 10
        // - Batch size: 32

        return config
    }

    // MARK: - Model Validation

    private func validateUpdatedModel(
        modelURL: URL,
        validationData: MLBatchProvider
    ) async throws -> Bool {
        // Load updated model
        let updatedModel = try ASFlarePredictor(contentsOf: modelURL)

        var correct = 0
        var total = 0

        // Evaluate on validation set
        for i in 0..<validationData.count {
            let features = validationData.features(at: i)

            guard let inputFeatures = features.featureValue(for: "features")?.multiArrayValue,
                  let labelValue = features.featureValue(for: "flare_label")?.multiArrayValue else {
                continue
            }

            // Make prediction
            let input = ASFlarePredictorInput(features: inputFeatures)
            let output = try await updatedModel.prediction(input: input)

            // Check if correct - use probabilities to determine predicted class
            let probabilities = output.probabilities
            let predictedFlare = probabilities[1].floatValue > 0.5
            let actualFlare = labelValue[0].int32Value == 1
            if predictedFlare == actualFlare {
                correct += 1
            }
            total += 1
        }

        let accuracy = total > 0 ? Float(correct) / Float(total) : 0.0
        print("📊 Updated model validation accuracy: \(String(format: "%.1f%%", accuracy * 100))")

        // Accept if accuracy >= 60% (reasonable for personalized model)
        return accuracy >= 0.60
    }

    private func deployUpdatedModel(from url: URL) throws {
        let deploymentURL = getModelURL()

        // Backup current model
        let backupURL = deploymentURL.deletingLastPathComponent()
            .appendingPathComponent("ASFlarePredictor.backup.mlpackage")

        if FileManager.default.fileExists(atPath: deploymentURL.path) {
            try? FileManager.default.removeItem(at: backupURL)
            try FileManager.default.copyItem(at: deploymentURL, to: backupURL)
        }

        // Deploy updated model
        try? FileManager.default.removeItem(at: deploymentURL)
        try FileManager.default.copyItem(at: url, to: deploymentURL)

        print("✅ Updated model deployed to \(deploymentURL.path)")
    }

    // MARK: - Helper Methods

    private func getUpdatedModelStorageURL() -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documentsURL.appendingPathComponent("PersonalizedFlarePredictor_v\(modelVersion + 1).mlmodelc")
    }

    private func shouldTriggerUpdate() -> Bool {
        // Trigger if:
        // 1. We have enough samples
        guard trainingDataCache.count >= minSamplesForUpdate else { return false }

        // 2. Enough time has passed since last update
        if let lastUpdate = lastUpdateDate {
            let timeSinceUpdate = Date().timeIntervalSince(lastUpdate)
            if timeSinceUpdate < updateFrequency {
                return false
            }
        }

        return true
    }

    private func calculateNextUpdateInterval() -> TimeInterval? {
        guard let lastUpdate = lastUpdateDate else {
            return nil  // Never updated yet
        }

        let timeSinceUpdate = Date().timeIntervalSince(lastUpdate)
        let timeUntilNext = max(0, updateFrequency - timeSinceUpdate)
        return timeUntilNext
    }

    private func updatePersonalizationMetrics() {
        // Calculate days of data
        guard let firstSample = trainingDataCache.first else {
            personalizationPhase = .bootstrap
            personalizationProgress = 0.0
            return
        }

        let daysOfData = Calendar.current.dateComponents(
            [.day],
            from: firstSample.timestamp,
            to: Date()
        ).day ?? 0

        // Update phase
        if daysOfData >= PersonalizationPhase.expert.daysRequired {
            personalizationPhase = .expert
        } else if daysOfData >= PersonalizationPhase.personalized.daysRequired {
            personalizationPhase = .personalized
        } else if daysOfData >= PersonalizationPhase.earlyAdaptation.daysRequired {
            personalizationPhase = .earlyAdaptation
        } else {
            personalizationPhase = .bootstrap
        }

        // Update progress
        personalizationProgress = calculateProgress()
    }

    private func calculateProgress() -> Float {
        let targetSamples: Float = 90.0  // 90 days of data = 100%
        return min(1.0, Float(totalPersonalSamples) / targetSamples)
    }

    private func getModelURL() -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent("ASFlarePredictor.mlpackage")
    }

    // MARK: - Persistence

    private func saveTrainingDataCache() {
        let cacheURL = getTrainingDataCacheURL()

        do {
            let data = try JSONEncoder().encode(trainingDataCache)
            try data.write(to: cacheURL)
        } catch {
            print("❌ Failed to save training data cache: \(error)")
        }
    }

    private func loadTrainingDataCache() {
        let cacheURL = getTrainingDataCacheURL()

        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            return
        }

        do {
            let data = try Data(contentsOf: cacheURL)
            trainingDataCache = try JSONDecoder().decode([TrainingSample].self, from: data)
            totalPersonalSamples = trainingDataCache.count
            print("✅ Loaded \(trainingDataCache.count) training samples from cache")
        } catch {
            print("❌ Failed to load training data cache: \(error)")
        }
    }

    private func getTrainingDataCacheURL() -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent("neural_engine_training_cache.json")
    }

    // MARK: - Errors

    enum LearningError: LocalizedError {
        case insufficientData(required: Int, available: Int)
        case updateFailed(String)
        case validationFailed

        var errorDescription: String? {
            switch self {
            case .insufficientData(let required, let available):
                return "Need \(required) samples for update, only have \(available)"
            case .updateFailed(let message):
                return "Model update failed: \(message)"
            case .validationFailed:
                return "Updated model failed validation"
            }
        }
    }
}

// MARK: - Status Types

struct DataReadinessStatus {
    let isReady: Bool
    let daysAvailable: Int
    let daysRequired: Int
    let message: String

    var progressPercentage: Float {
        return min(1.0, Float(daysAvailable) / Float(daysRequired))
    }
}

@available(iOS 17.0, *)
struct LearningPipelineStatus {
    let phase: ContinuousLearningPipeline.PersonalizationPhase
    let progress: Float  // 0-1
    let totalSamples: Int
    let daysOfData: Int
    let lastUpdate: Date?
    let modelVersion: Int
    let nextUpdateIn: TimeInterval?

    var statusMessage: String {
        switch phase {
        case .bootstrap:
            return "Collecting baseline data (\(totalSamples)/7 samples)"
        case .earlyAdaptation:
            return "Early personalization - \(daysOfData) days logged"
        case .personalized:
            return "Personalized to your patterns - \(modelVersion) updates"
        case .expert:
            return "Expert model - \(daysOfData) days of deep learning"
        }
    }

    var nextUpdateMessage: String {
        guard let interval = nextUpdateIn else {
            return "First update available after 7 days of data"
        }

        if interval == 0 {
            return "Update available now"
        }

        let days = Int(interval / 86400)
        return "Next update in \(days) days"
    }
}
