//
//  NeuralEnginePredictionService.swift
//  InflamAI
//
//  Neural Engine prediction service with 92-feature support
//  Created by Enhanced CoreML Exporter
//

import Foundation
import CoreML
import Combine

@available(iOS 17.0, *)
@MainActor
class NeuralEnginePredictionService: ObservableObject {

    // MARK: - Published Properties
    @Published var isModelLoaded = false
    @Published var lastPrediction: FlarePrediction?
    @Published var errorMessage: String?
    @Published var bootstrapProgress: Double = 0.0  // 0.0 to 1.0
    @Published var daysOfData: Int = 0
    @Published var isUsingSyntheticModel: Bool = true  // Model trained on synthetic data

    // MARK: - Private Properties
    private var model: ASFlarePredictor?
    private var scaler: NeuralEngineFeatureScaler?
    private var metadata: ModelMetadata?

    // MARK: - Data Structures
    struct FlarePrediction {
        let willFlare: Bool
        let probability: Float  // P(flare) 0-1
        let riskScore: Float    // 0-1 continuous
        let confidence: ConfidenceLevel
        let timestamp: Date
        let daysOfDataUsed: Int
        let bootstrapPhase: BootstrapPhase

        enum ConfidenceLevel: String {
            case low = "Low"
            case medium = "Medium"
            case high = "High"

            init(probability: Float) {
                let distance = abs(probability - 0.5)
                if distance < 0.15 { self = .low }
                else if distance < 0.3 { self = .medium }
                else { self = .high }
            }

            var color: String {
                switch self {
                case .low: return "gray"
                case .medium: return "orange"
                case .high: return "green"
                }
            }
        }

        enum BootstrapPhase: String {
            case learning = "Learning your patterns..."
            case earlyPersonalization = "Early personalization (30%)"
            case majorityPersonal = "Majority personal (60%)"
            case fullyPersonalized = "Fully personalized (90%)"

            init(daysOfData: Int) {
                switch daysOfData {
                case 0..<7: self = .learning
                case 7..<14: self = .earlyPersonalization
                case 14..<21: self = .majorityPersonal
                default: self = .fullyPersonalized
                }
            }

            var description: String {
                switch self {
                case .learning: return "Building baseline with synthetic data"
                case .earlyPersonalization: return "Blending your data (30%) with baseline"
                case .majorityPersonal: return "Mostly your patterns (60%)"
                case .fullyPersonalized: return "Trained on your unique patterns"
                }
            }
        }
    }

    struct ModelMetadata {
        let architecture: String
        let inputFeatures: Int
        let baselineAccuracy: Float
        let baselineF1: Float
        let featureNames: [String]
        let scalerMeans: [Float]
        let scalerStds: [Float]
    }

    // MARK: - Initialization
    init() {
        Task {
            await loadModel()
        }
    }

    // MARK: - Model Loading
    private func loadModel() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU

            self.model = try ASFlarePredictor(configuration: config)

            // Parse metadata
            if let modelDescription = model?.model.modelDescription,
               let userDefined = modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String] {
                self.metadata = parseMetadata(userDefined)
                self.scaler = NeuralEngineFeatureScaler(
                    means: metadata?.scalerMeans ?? [],
                    stds: metadata?.scalerStds ?? []
                )
                
                // Check if model was trained on synthetic data
                if let trainingDataSource = userDefined["training_data_source"],
                   trainingDataSource == "real_patient_data" {
                    self.isUsingSyntheticModel = false
                }
                // Default is true (synthetic) unless explicitly marked as real
            }

            self.isModelLoaded = true
            print("✅ Neural Engine loaded successfully")
            print("   Architecture: \(metadata?.architecture ?? "unknown")")
            print("   Features: \(metadata?.inputFeatures ?? 0)")
            print("   Baseline Accuracy: \(String(format: "%.1f%%", (metadata?.baselineAccuracy ?? 0) * 100))")

        } catch {
            self.errorMessage = "Failed to load neural engine: \(error.localizedDescription)"
            print("❌ Model loading failed: \(error)")
        }
    }

    private func parseMetadata(_ userDefined: [String: String]) -> ModelMetadata {
        return ModelMetadata(
            architecture: userDefined["architecture"] ?? "LSTM",
            inputFeatures: Int(userDefined["input_features"] ?? "92") ?? 92,
            baselineAccuracy: Float(userDefined["baseline_accuracy"] ?? "0") ?? 0,
            baselineF1: Float(userDefined["baseline_f1"] ?? "0") ?? 0,
            featureNames: parseJSONStringArray(userDefined["feature_names"]) ?? [],
            scalerMeans: parseJSONFloatArray(userDefined["scaler_means"]) ?? [],
            scalerStds: parseJSONFloatArray(userDefined["scaler_stds"]) ?? []
        )
    }

    private func parseJSONStringArray(_ json: String?) -> [String]? {
        guard let json = json,
              let data = json.data(using: .utf8),
              let array = try? JSONDecoder().decode([String].self, from: data) else {
            return nil
        }
        return array
    }

    private func parseJSONFloatArray(_ json: String?) -> [Float]? {
        guard let json = json,
              let data = json.data(using: .utf8),
              let array = try? JSONDecoder().decode([Float].self, from: data) else {
            return nil
        }
        return array
    }

    // MARK: - Prediction
    func predict(features: [[Float]]) async throws -> FlarePrediction {
        guard let model = model else {
            throw PredictionError.modelNotLoaded
        }

        guard features.count == 30 && features.first?.count == 92 else {
            throw PredictionError.invalidInputShape(
                expected: "(30, 92)",
                got: "(\(features.count), \(features.first?.count ?? 0))"
            )
        }

        // Normalize features
        let normalizedFeatures = scaler?.transform(features) ?? features

        // Create MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
        for (i, timestep) in normalizedFeatures.enumerated() {
            for (j, value) in timestep.enumerated() {
                inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
            }
        }

        // Create input
        let input = ASFlarePredictorInput(features: inputArray)

        // Run prediction
        let output = try await model.prediction(input: input)

        // Parse outputs
        let probabilities = output.probabilities
        let willFlareProb = probabilities[1].floatValue  // P(flare=1)
        let willFlare = willFlareProb > 0.5
        let riskScore = output.risk_score[0].floatValue
        let confidence = FlarePrediction.ConfidenceLevel(probability: willFlareProb)

        // Count non-zero days (actual data vs padding)
        let daysOfData = features.filter { timestep in
            timestep.contains { $0 != 0.0 }
        }.count

        self.daysOfData = daysOfData
        self.bootstrapProgress = min(1.0, Double(daysOfData) / 28.0)  // 28 days = full bootstrap

        let bootstrapPhase = FlarePrediction.BootstrapPhase(daysOfData: daysOfData)

        let prediction = FlarePrediction(
            willFlare: willFlare,
            probability: willFlareProb,
            riskScore: riskScore,
            confidence: confidence,
            timestamp: Date(),
            daysOfDataUsed: daysOfData,
            bootstrapPhase: bootstrapPhase
        )

        self.lastPrediction = prediction
        return prediction
    }

    // MARK: - Errors
    enum PredictionError: LocalizedError {
        case modelNotLoaded
        case invalidInputShape(expected: String, got: String)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "Neural engine not loaded. Please wait for initialization."
            case .invalidInputShape(let expected, let got):
                return "Invalid input shape. Expected \(expected), got \(got)"
            }
        }
    }
}

// MARK: - NeuralEngineFeatureScaler (local to NeuralEnginePredictionService)
class NeuralEngineFeatureScaler {
    private let means: [Float]
    private let stds: [Float]

    init(means: [Float], stds: [Float]) {
        self.means = means
        self.stds = stds
    }

    func transform(_ features: [[Float]]) -> [[Float]] {
        guard !means.isEmpty && !stds.isEmpty else {
            return features  // No scaling
        }

        return features.map { timestep in
            timestep.enumerated().map { (index, value) in
                guard index < means.count && index < stds.count else {
                    return value
                }
                let mean = means[index]
                let std = stds[index]
                return std > 0 ? (value - mean) / std : value
            }
        }
    }
}
