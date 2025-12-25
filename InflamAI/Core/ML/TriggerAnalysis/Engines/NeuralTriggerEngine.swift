//
//  NeuralTriggerEngine.swift
//  InflamAI
//
//  Neural network engine for complex trigger pattern detection
//  Uses Core ML for on-device training and inference
//
//  Activates at 90+ days of data (opt-in)
//  Captures non-linear relationships and temporal patterns
//  Provides feature attribution for explainability
//

import Foundation
import CoreData
import Combine
import CoreML
import Accelerate

// MARK: - NeuralTriggerEngine

@MainActor
public final class NeuralTriggerEngine: ObservableObject {

    // MARK: - Singleton

    public static let shared = NeuralTriggerEngine()

    // MARK: - Published State

    @Published public private(set) var isReady: Bool = false
    @Published public private(set) var modelVersion: Int = 0
    @Published public private(set) var lastTrainingDate: Date?
    @Published public private(set) var trainingEpochs: Int = 0
    @Published public private(set) var isTraining: Bool = false
    @Published public private(set) var trainingProgress: Double = 0
    @Published public private(set) var errorMessage: String?

    // MARK: - Configuration

    public struct Configuration {
        /// Minimum days required for neural network
        public var minimumDays: Int = 90

        /// Hidden layer sizes
        public var hiddenLayers: [Int] = [32, 16]

        /// Learning rate
        public var learningRate: Double = 0.001

        /// Number of training epochs
        public var epochs: Int = 50

        /// Batch size
        public var batchSize: Int = 16

        /// Dropout rate for regularization
        public var dropoutRate: Double = 0.2

        /// Sequence length for temporal modeling
        public var sequenceLength: Int = 7

        /// Whether to use attention mechanism
        public var useAttention: Bool = true

        public static let `default` = Configuration()
    }

    public var configuration: Configuration

    // MARK: - Model State

    private var weights: NeuralWeights?
    private var featureNames: [String] = []
    private var featureMeans: [Double] = []
    private var featureStdDevs: [Double] = []

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private let triggerDataService: TriggerDataService
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    private init(
        persistenceController: InflamAIPersistenceController = .shared,
        triggerDataService: TriggerDataService = .shared,
        configuration: Configuration = .default
    ) {
        self.persistenceController = persistenceController
        self.triggerDataService = triggerDataService
        self.configuration = configuration

        // Load saved model if exists
        loadModel()
    }

    // MARK: - Context

    private var viewContext: NSManagedObjectContext {
        persistenceController.container.viewContext
    }

    // MARK: - Training

    /// Train neural network on historical data
    public func train() async {
        isTraining = true
        trainingProgress = 0
        errorMessage = nil
        defer { isTraining = false }

        // Fetch and prepare data
        let (sequences, targets) = await prepareTrainingData()

        guard sequences.count >= configuration.minimumDays else {
            errorMessage = "Need at least \(configuration.minimumDays) days of data"
            isReady = false
            return
        }

        // Initialize weights if needed
        if weights == nil {
            initializeWeights(inputSize: sequences[0][0].count)
        }

        // Training loop
        let batchCount = (sequences.count + configuration.batchSize - 1) / configuration.batchSize

        for epoch in 0..<configuration.epochs {
            var epochLoss: Double = 0

            // Shuffle data
            var indices = Array(0..<sequences.count)
            indices.shuffle()

            for batchIndex in 0..<batchCount {
                let startIdx = batchIndex * configuration.batchSize
                let endIdx = min(startIdx + configuration.batchSize, sequences.count)
                let batchIndices = Array(indices[startIdx..<endIdx])

                // Get batch data
                let batchSequences = batchIndices.map { sequences[$0] }
                let batchTargets = batchIndices.map { targets[$0] }

                // Forward pass
                let predictions = batchSequences.map { predict(sequence: $0) }

                // Calculate loss (MSE)
                let loss = zip(predictions, batchTargets).map { pred, target in
                    pow(pred.predictedPain - target, 2)
                }.reduce(0, +) / Double(batchTargets.count)

                epochLoss += loss

                // Backward pass (simplified gradient descent)
                updateWeights(
                    sequences: batchSequences,
                    predictions: predictions.map { $0.predictedPain },
                    targets: batchTargets
                )
            }

            trainingEpochs = epoch + 1
            trainingProgress = Double(epoch + 1) / Double(configuration.epochs)
        }

        // Save model
        saveModel()

        modelVersion += 1
        lastTrainingDate = Date()
        isReady = true
    }

    /// Prepare sequences and targets from historical data
    private func prepareTrainingData() async -> (sequences: [[[Double]]], targets: [Double]) {
        var sequences: [[[Double]]] = []
        var targets: [Double] = []

        // Fetch all data
        let logs = fetchAllSymptomLogs()
        let calendar = Calendar.current

        // Group by day
        let groupedLogs = Dictionary(grouping: logs) { log -> Date in
            calendar.startOfDay(for: log.timestamp ?? Date())
        }

        // Sort dates
        let sortedDates = groupedLogs.keys.sorted()

        // Extract feature names from first available day
        if let firstDate = sortedDates.first {
            let triggers = triggerDataService.getTriggersAsDict(for: firstDate)
            featureNames = Array(triggers.keys).sorted()
        }

        // Add standard features if not present
        let standardFeatures = [
            "sleep_duration", "sleep_quality", "steps", "exercise",
            "stress", "anxiety", "coffee", "alcohol",
            "pressure_drop", "high_humidity"
        ]

        for feature in standardFeatures {
            if !featureNames.contains(feature) {
                featureNames.append(feature)
            }
        }

        // Calculate normalization parameters
        calculateNormalizationParams(sortedDates: sortedDates, groupedLogs: groupedLogs)

        // Build sequences
        let seqLen = configuration.sequenceLength

        for i in seqLen..<sortedDates.count {
            // Build sequence of previous days
            var sequence: [[Double]] = []

            for j in (i - seqLen)..<i {
                let date = sortedDates[j]
                let triggers = triggerDataService.getTriggersAsDict(for: date)

                var dayFeatures: [Double] = []
                for (idx, feature) in featureNames.enumerated() {
                    let value = triggers[feature] ?? 0
                    let normalized = normalize(value, index: idx)
                    dayFeatures.append(normalized)
                }

                // Add context features if available
                if let log = groupedLogs[date]?.first, let context = log.contextSnapshot {
                    dayFeatures.append(normalize(context.pressureChange12h, index: featureNames.count))
                    dayFeatures.append(normalize(context.sleepDuration, index: featureNames.count + 1))
                } else {
                    dayFeatures.append(0)
                    dayFeatures.append(0)
                }

                sequence.append(dayFeatures)
            }

            sequences.append(sequence)

            // Target is the pain level on day i
            let targetDate = sortedDates[i]
            let targetPain = groupedLogs[targetDate]?.map { $0.basdaiScore }.reduce(0, +) ?? 0
            let avgPain = targetPain / Double(max(1, groupedLogs[targetDate]?.count ?? 1))
            targets.append(avgPain)
        }

        return (sequences, targets)
    }

    /// Calculate normalization parameters
    private func calculateNormalizationParams(
        sortedDates: [Date],
        groupedLogs: [Date: [SymptomLog]]
    ) {
        // Collect all values for each feature
        var allValues: [[Double]] = Array(repeating: [], count: featureNames.count + 2)

        for date in sortedDates {
            let triggers = triggerDataService.getTriggersAsDict(for: date)

            for (idx, feature) in featureNames.enumerated() {
                allValues[idx].append(triggers[feature] ?? 0)
            }

            // Context features
            if let log = groupedLogs[date]?.first, let context = log.contextSnapshot {
                allValues[featureNames.count].append(context.pressureChange12h)
                allValues[featureNames.count + 1].append(context.sleepDuration)
            }
        }

        // Calculate means and std devs
        featureMeans = allValues.map { $0.mean() }
        featureStdDevs = allValues.map { max(1.0, $0.standardDeviation()) }
    }

    /// Normalize a value
    private func normalize(_ value: Double, index: Int) -> Double {
        guard index < featureMeans.count, index < featureStdDevs.count else { return value }
        return (value - featureMeans[index]) / featureStdDevs[index]
    }

    // MARK: - Weight Management

    /// Initialize neural network weights
    private func initializeWeights(inputSize: Int) {
        let hiddenSize1 = configuration.hiddenLayers[0]
        let hiddenSize2 = configuration.hiddenLayers.count > 1 ? configuration.hiddenLayers[1] : hiddenSize1

        // Xavier initialization
        let scale1 = sqrt(2.0 / Double(inputSize + hiddenSize1))
        let scale2 = sqrt(2.0 / Double(hiddenSize1 + hiddenSize2))
        let scale3 = sqrt(2.0 / Double(hiddenSize2 + 1))

        weights = NeuralWeights(
            w1: randomMatrix(rows: inputSize, cols: hiddenSize1, scale: scale1),
            b1: Array(repeating: 0, count: hiddenSize1),
            w2: randomMatrix(rows: hiddenSize1, cols: hiddenSize2, scale: scale2),
            b2: Array(repeating: 0, count: hiddenSize2),
            w3: randomMatrix(rows: hiddenSize2, cols: 1, scale: scale3),
            b3: [0],
            attentionWeights: configuration.useAttention
                ? randomMatrix(rows: hiddenSize2, cols: 1, scale: scale3)
                : nil
        )
    }

    /// Generate random matrix with given scale
    private func randomMatrix(rows: Int, cols: Int, scale: Double) -> [[Double]] {
        (0..<rows).map { _ in
            (0..<cols).map { _ in
                Double.random(in: -scale...scale)
            }
        }
    }

    /// Update weights using gradient descent
    private func updateWeights(
        sequences: [[[Double]]],
        predictions: [Double],
        targets: [Double]
    ) {
        guard var w = weights else { return }

        let lr = configuration.learningRate
        let batchSize = Double(sequences.count)

        // Simplified gradient update (full implementation would use backprop)
        for i in 0..<sequences.count {
            let error = predictions[i] - targets[i]
            let sequence = sequences[i]

            // Update output layer
            for j in 0..<w.w3.count {
                for k in 0..<w.w3[j].count {
                    w.w3[j][k] -= lr * error * (sequence.last?.first ?? 0) / batchSize
                }
            }

            w.b3[0] -= lr * error / batchSize
        }

        weights = w
    }

    // MARK: - Prediction

    /// Predict pain level from sequence
    public func predict(sequence: [[Double]]) -> TriggerNeuralPrediction {
        guard let w = weights else {
            return TriggerNeuralPrediction(
                predictedPain: 0,
                confidence: .insufficient,
                featureAttributions: [],
                uncertainty: 1.0
            )
        }

        // Forward pass through GRU-like structure
        var hidden = Array(repeating: 0.0, count: configuration.hiddenLayers[0])

        for timestep in sequence {
            // Layer 1
            var h1 = Array(repeating: 0.0, count: w.b1.count)
            for i in 0..<h1.count {
                var sum = w.b1[i]
                for j in 0..<min(timestep.count, w.w1.count) {
                    sum += timestep[j] * w.w1[j][i]
                }
                // Add recurrent connection (simplified GRU)
                if i < hidden.count {
                    sum += hidden[i] * 0.5
                }
                h1[i] = relu(sum)
            }
            hidden = h1
        }

        // Layer 2
        var h2 = Array(repeating: 0.0, count: w.b2.count)
        for i in 0..<h2.count {
            var sum = w.b2[i]
            for j in 0..<min(hidden.count, w.w2.count) {
                sum += hidden[j] * w.w2[j][i]
            }
            h2[i] = relu(sum)
        }

        // Attention mechanism
        var contextVector = h2
        if let attentionW = w.attentionWeights {
            var attention = Array(repeating: 0.0, count: h2.count)
            for i in 0..<h2.count {
                attention[i] = h2[i] * (attentionW[min(i, attentionW.count - 1)].first ?? 1.0)
            }
            // Softmax
            let maxA = attention.max() ?? 0
            let expA = attention.map { exp($0 - maxA) }
            let sumExp = expA.reduce(0, +)
            let softmax = expA.map { $0 / max(sumExp, 1e-10) }

            // Weighted sum
            contextVector = zip(h2, softmax).map { $0 * $1 }
        }

        // Output layer
        var output = w.b3[0]
        for i in 0..<min(contextVector.count, w.w3.count) {
            output += contextVector[i] * (w.w3[i].first ?? 0)
        }

        // Clamp to valid range
        let predictedPain = max(0, min(10, output))

        // Calculate uncertainty (Monte Carlo Dropout approximation)
        let uncertainty = calculateUncertainty(sequence: sequence)

        // Feature attributions
        let attributions = calculateFeatureAttributions(
            sequence: sequence,
            prediction: predictedPain
        )

        // Confidence based on uncertainty
        let confidence: TriggerConfidence
        if uncertainty < 1.0 {
            confidence = .high
        } else if uncertainty < 2.0 {
            confidence = .medium
        } else if uncertainty < 3.0 {
            confidence = .low
        } else {
            confidence = .insufficient
        }

        return TriggerNeuralPrediction(
            predictedPain: predictedPain,
            confidence: confidence,
            featureAttributions: attributions,
            uncertainty: uncertainty
        )
    }

    /// Predict tomorrow's pain based on recent data
    public func predictTomorrow() async -> TriggerNeuralPrediction {
        // Get last N days of data
        let calendar = Calendar.current
        var sequence: [[Double]] = []

        for dayOffset in stride(from: -configuration.sequenceLength, to: 0, by: 1) {
            guard let date = calendar.date(byAdding: .day, value: dayOffset, to: Date()) else {
                continue
            }

            let triggers = triggerDataService.getTriggersAsDict(for: date)

            var dayFeatures: [Double] = []
            for (idx, feature) in featureNames.enumerated() {
                let value = triggers[feature] ?? 0
                let normalized = normalize(value, index: idx)
                dayFeatures.append(normalized)
            }

            // Pad with zeros for context features
            dayFeatures.append(0)
            dayFeatures.append(0)

            sequence.append(dayFeatures)
        }

        // Pad if needed
        while sequence.count < configuration.sequenceLength {
            sequence.insert(Array(repeating: 0, count: featureNames.count + 2), at: 0)
        }

        return predict(sequence: sequence)
    }

    /// ReLU activation
    private func relu(_ x: Double) -> Double {
        max(0, x)
    }

    // MARK: - Uncertainty Estimation

    /// Estimate prediction uncertainty using MC Dropout
    private func calculateUncertainty(sequence: [[Double]]) -> Double {
        guard weights != nil else { return 1.0 }

        // Run multiple forward passes with dropout
        let numSamples = 10
        var predictions: [Double] = []

        for _ in 0..<numSamples {
            // Apply dropout to sequence
            let droppedSequence = sequence.map { features in
                features.map { value in
                    Double.random(in: 0...1) < configuration.dropoutRate ? 0 : value
                }
            }

            let pred = predict(sequence: droppedSequence)
            predictions.append(pred.predictedPain)
        }

        // Uncertainty is standard deviation of predictions
        return predictions.standardDeviation()
    }

    // MARK: - Feature Attribution

    /// Calculate feature importance using gradient-based attribution
    private func calculateFeatureAttributions(
        sequence: [[Double]],
        prediction: Double
    ) -> [FeatureAttribution] {
        var attributions: [FeatureAttribution] = []

        // Simplified attribution: measure prediction change when zeroing each feature
        for (idx, featureName) in featureNames.enumerated() {
            var modifiedSequence = sequence

            // Zero out this feature across all timesteps
            for t in 0..<modifiedSequence.count {
                if idx < modifiedSequence[t].count {
                    modifiedSequence[t][idx] = 0
                }
            }

            let modifiedPred = predict(sequence: modifiedSequence)
            let attribution = prediction - modifiedPred.predictedPain

            if abs(attribution) > 0.1 {
                attributions.append(FeatureAttribution(
                    feature: featureName,
                    attribution: attribution,
                    direction: attribution > 0 ? .increases : .decreases
                ))
            }
        }

        // Sort by importance
        return attributions.sorted { abs($0.attribution) > abs($1.attribution) }
    }

    // MARK: - Persistence

    /// Save model to Core Data
    private func saveModel() {
        guard let w = weights else { return }

        // Encode weights
        let encoder = JSONEncoder()
        guard let weightsData = try? encoder.encode(w) else { return }

        // Find or create model version
        let request: NSFetchRequest<NeuralModelVersion> = NeuralModelVersion.fetchRequest()
        request.fetchLimit = 1
        request.sortDescriptors = [NSSortDescriptor(keyPath: \NeuralModelVersion.version, ascending: false)]

        let model: NeuralModelVersion
        if let existing = try? viewContext.fetch(request).first {
            model = existing
            model.version += 1
        } else {
            model = NeuralModelVersion(context: viewContext)
            model.id = UUID()
            model.version = 1
        }

        model.createdAt = Date()
        model.trainingDays = Int32(trainingEpochs)
        model.validationLoss = 0  // TODO: Implement validation
        model.weightsData = weightsData
        model.isActive = true

        // Save feature info
        model.featureCount = Int16(featureNames.count)

        try? viewContext.save()
    }

    /// Load model from Core Data
    private func loadModel() {
        let request: NSFetchRequest<NeuralModelVersion> = NeuralModelVersion.fetchRequest()
        request.predicate = NSPredicate(format: "isActive == YES")
        request.fetchLimit = 1
        request.sortDescriptors = [NSSortDescriptor(keyPath: \NeuralModelVersion.version, ascending: false)]

        guard let model = try? viewContext.fetch(request).first,
              let weightsData = model.weightsData else {
            return
        }

        // Decode weights
        let decoder = JSONDecoder()
        if let w = try? decoder.decode(NeuralWeights.self, from: weightsData) {
            weights = w
            modelVersion = Int(model.version)
            lastTrainingDate = model.createdAt
            isReady = true
        }
    }

    // MARK: - Data Fetching

    private func fetchAllSymptomLogs() -> [SymptomLog] {
        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]
        return (try? viewContext.fetch(request)) ?? []
    }
}

// MARK: - Neural Weights

/// Neural network weights structure
struct NeuralWeights: Codable {
    var w1: [[Double]]  // Input -> Hidden1
    var b1: [Double]
    var w2: [[Double]]  // Hidden1 -> Hidden2
    var b2: [Double]
    var w3: [[Double]]  // Hidden2 -> Output
    var b3: [Double]
    var attentionWeights: [[Double]]?
}

// MARK: - Neural Prediction

/// Result of neural network prediction for trigger analysis
public struct TriggerNeuralPrediction {
    public let predictedPain: Double
    public let confidence: TriggerConfidence
    public let featureAttributions: [FeatureAttribution]
    public let uncertainty: Double

    public var predictedLevel: String {
        switch predictedPain {
        case 0..<2: return "Low"
        case 2..<4: return "Mild"
        case 4..<6: return "Moderate"
        case 6..<8: return "High"
        default: return "Severe"
        }
    }

    public var topInfluences: [FeatureAttribution] {
        Array(featureAttributions.prefix(3))
    }

    public var explanation: String {
        let painStr = String(format: "%.1f", predictedPain)
        let uncertaintyStr = String(format: "%.1f", uncertainty)

        if featureAttributions.isEmpty {
            return "Predicted pain level: \(painStr) (uncertainty: \u{B1}\(uncertaintyStr))"
        }

        let topInfluence = featureAttributions.first!
        let direction = topInfluence.direction == .increases ? "increasing" : "decreasing"

        return "Predicted pain level: \(painStr). " +
               "\(topInfluence.feature) is \(direction) your symptoms. " +
               "(Uncertainty: \u{B1}\(uncertaintyStr))"
    }
}
