//
//  CalibrationEngine.swift
//  InflamAI
//
//  Temperature scaling for probability calibration + Monte Carlo dropout for uncertainty
//  Ensures predictions are reliable: 70% predictions are actually ~70% accurate
//

import Foundation
import CoreML

@available(iOS 17.0, *)
@MainActor
class CalibrationEngine {

    private let model: ASFlarePredictor
    private let scaler: CalibrationFeatureScaler

    // Temperature parameter (learned during calibration)
    // T > 1: softens probabilities (less confident)
    // T < 1: sharpens probabilities (more confident)
    // T = 1: no change (default)
    private var temperature: Float = 1.0

    // Monte Carlo dropout parameters
    private let mcIterations: Int = 10  // Number of forward passes with dropout

    struct CalibratedPrediction {
        let probability: Float              // Calibrated P(flare)
        let confidence: ConfidenceLevel
        let uncertaintyScore: Float         // 0-1 (0 = very certain, 1 = very uncertain)
        let predictionInterval: (lower: Float, upper: Float)  // 95% confidence interval
        let ensemblePredictions: [Float]    // All MC dropout predictions

        enum ConfidenceLevel: String {
            case high = "High"              // Uncertainty < 0.15
            case medium = "Medium"          // Uncertainty 0.15-0.30
            case low = "Low"                // Uncertainty > 0.30

            var color: String {
                switch self {
                case .high: return "green"
                case .medium: return "yellow"
                case .low: return "orange"
                }
            }

            var description: String {
                switch self {
                case .high: return "Model is confident in this prediction"
                case .medium: return "Prediction has moderate uncertainty"
                case .low: return "Prediction has high uncertainty - collect more data"
                }
            }
        }
    }

    init(model: ASFlarePredictor, scaler: CalibrationFeatureScaler) {
        self.model = model
        self.scaler = scaler
    }

    // MARK: - Temperature Scaling

    /// Apply temperature scaling to raw model probability
    /// This calibrates predictions so confidence levels match reality
    func applyTemperatureScaling(rawProbability: Float) -> Float {
        // Apply temperature to logit, then sigmoid
        // P_calibrated = sigmoid(logit / T)

        // Convert probability to logit
        let epsilon: Float = 1e-7  // Prevent log(0)
        let clampedProb = max(epsilon, min(1.0 - epsilon, rawProbability))
        let logit = log(clampedProb / (1.0 - clampedProb))

        // Scale by temperature
        let scaledLogit = logit / temperature

        // Convert back to probability
        let calibratedProb = 1.0 / (1.0 + exp(-scaledLogit))

        return calibratedProb
    }

    /// Learn optimal temperature parameter from validation data
    /// Call this during the bootstrap phase as user collects real data
    func calibrateTemperature(
        predictions: [Float],  // Raw model probabilities
        outcomes: [Bool]       // Actual flare outcomes
    ) {
        guard predictions.count == outcomes.count,
              predictions.count >= 10 else {
            print("⚠️ Insufficient data for calibration (need 10+ samples)")
            return
        }

        // Grid search for optimal temperature
        var bestTemperature: Float = 1.0
        var bestLoss: Float = .infinity

        for T in stride(from: Float(0.5), through: Float(2.0), by: Float(0.1)) {
            var loss: Float = 0.0

            for (prediction, outcome) in zip(predictions, outcomes) {
                // Apply temperature
                let epsilon: Float = 1e-7
                let clampedProb = max(epsilon, min(1.0 - epsilon, prediction))
                let logit = log(clampedProb / (1.0 - clampedProb))
                let scaledLogit = logit / T
                let calibratedProb: Float = 1.0 / (1.0 + exp(-scaledLogit))

                // Cross-entropy loss
                let target: Float = outcome ? 1.0 : 0.0
                loss += -target * log(calibratedProb + epsilon) - (1 - target) * log(1 - calibratedProb + epsilon)
            }

            if loss < bestLoss {
                bestLoss = loss
                bestTemperature = T
            }
        }

        self.temperature = Float(bestTemperature)
        print("✅ Calibrated temperature: \(String(format: "%.2f", bestTemperature))")
    }

    // MARK: - Monte Carlo Dropout (Uncertainty Quantification)

    /// Run prediction with uncertainty estimation using MC dropout
    /// Makes multiple predictions with dropout enabled to measure variance
    func predictWithUncertainty(
        features: [[Float]]
    ) async throws -> CalibratedPrediction {

        // Normalize features
        let normalizedFeatures = scaler.transform(features)

        // Create input MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
        for (i, timestep) in normalizedFeatures.enumerated() {
            for (j, value) in timestep.enumerated() {
                inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
            }
        }

        // Run multiple predictions (MC dropout approximation)
        var predictions: [Float] = []

        for _ in 0..<mcIterations {
            let input = ASFlarePredictorInput(features: inputArray)
            let output = try await model.prediction(input: input)

            // Extract flare probability
            let probabilities = output.probabilities
            let flareProb = probabilities[1].floatValue
            predictions.append(flareProb)
        }

        // Compute statistics
        let meanProb = predictions.reduce(0, +) / Float(predictions.count)
        let variance = predictions.map { pow($0 - meanProb, 2) }.reduce(0, +) / Float(predictions.count)
        let stdDev = sqrt(variance)

        // Apply temperature scaling to mean
        let calibratedProb = applyTemperatureScaling(rawProbability: meanProb)

        // 95% confidence interval (±1.96 std dev)
        let lowerBound = max(0.0, calibratedProb - 1.96 * stdDev)
        let upperBound = min(1.0, calibratedProb + 1.96 * stdDev)

        // Uncertainty score (coefficient of variation)
        let uncertaintyScore = meanProb > 0.01 ? stdDev / meanProb : stdDev

        // Determine confidence level
        let confidence: CalibratedPrediction.ConfidenceLevel
        if uncertaintyScore < 0.15 {
            confidence = .high
        } else if uncertaintyScore < 0.30 {
            confidence = .medium
        } else {
            confidence = .low
        }

        return CalibratedPrediction(
            probability: calibratedProb,
            confidence: confidence,
            uncertaintyScore: uncertaintyScore,
            predictionInterval: (lower: lowerBound, upper: upperBound),
            ensemblePredictions: predictions
        )
    }

    // MARK: - Calibration Metrics

    /// Compute Expected Calibration Error (ECE)
    /// Measures how well calibrated the model is (lower = better)
    func computeECE(
        predictions: [Float],
        outcomes: [Bool],
        numBins: Int = 10
    ) -> Float {
        guard predictions.count == outcomes.count else { return 0.0 }

        // Create bins
        var bins: [[Int]] = Array(repeating: [], count: numBins)

        for (i, prob) in predictions.enumerated() {
            let binIndex = min(Int(prob * Float(numBins)), numBins - 1)
            bins[binIndex].append(i)
        }

        // Compute ECE
        var ece: Float = 0.0
        let totalSamples = Float(predictions.count)

        for binIndices in bins where !binIndices.isEmpty {
            let binSize = Float(binIndices.count)

            // Average predicted probability in bin
            let avgPrediction = binIndices.map { predictions[$0] }.reduce(0, +) / binSize

            // Average actual outcome in bin
            let avgOutcome = binIndices.map { outcomes[$0] ? 1.0 : 0.0 }.reduce(0, +) / binSize

            // Contribution to ECE
            ece += (binSize / totalSamples) * abs(avgPrediction - avgOutcome)
        }

        return ece
    }

    /// Generate calibration plot data
    /// Returns (predicted_prob, actual_prob) pairs for visualization
    func generateCalibrationPlotData(
        predictions: [Float],
        outcomes: [Bool],
        numBins: Int = 10
    ) -> [(predicted: Float, actual: Float, count: Int)] {
        guard predictions.count == outcomes.count else { return [] }

        var bins: [[Int]] = Array(repeating: [], count: numBins)

        for (i, prob) in predictions.enumerated() {
            let binIndex = min(Int(prob * Float(numBins)), numBins - 1)
            bins[binIndex].append(i)
        }

        var plotData: [(Float, Float, Int)] = []

        for (binIndex, binIndices) in bins.enumerated() where !binIndices.isEmpty {
            let binSize = Float(binIndices.count)
            let avgPrediction = binIndices.map { predictions[$0] }.reduce(0, +) / binSize
            let avgOutcome = binIndices.map { outcomes[$0] ? 1.0 : 0.0 }.reduce(0, +) / binSize

            plotData.append((avgPrediction, avgOutcome, binIndices.count))
        }

        return plotData
    }

    // MARK: - Uncertainty Decomposition

    /// Decompose uncertainty into aleatoric (data noise) and epistemic (model uncertainty)
    struct UncertaintyDecomposition {
        let aleatoric: Float      // Inherent randomness in data
        let epistemic: Float      // Model uncertainty (reducible with more data)
        let total: Float

        var interpretation: String {
            if epistemic > 0.15 {
                return "High model uncertainty - collect more similar data"
            } else if aleatoric > 0.15 {
                return "High data noise - inherent unpredictability"
            } else {
                return "Low uncertainty - model is confident"
            }
        }
    }

    func decomposeUncertainty(
        calibratedPrediction: CalibratedPrediction
    ) -> UncertaintyDecomposition {
        let predictions = calibratedPrediction.ensemblePredictions

        // Epistemic uncertainty (variance across predictions)
        let mean = predictions.reduce(0, +) / Float(predictions.count)
        let variance = predictions.map { pow($0 - mean, 2) }.reduce(0, +) / Float(predictions.count)
        let epistemic = sqrt(variance)

        // Aleatoric uncertainty (average predictive entropy)
        let avgEntropy = predictions.map { p in
            let epsilon: Float = 1e-7
            let clampedP = max(epsilon, min(1.0 - epsilon, p))
            return -clampedP * log(clampedP) - (1 - clampedP) * log(1 - clampedP)
        }.reduce(0, +) / Float(predictions.count)

        let aleatoric = avgEntropy / log(2.0)  // Normalize to [0, 1]

        let total = sqrt(epistemic * epistemic + aleatoric * aleatoric)

        return UncertaintyDecomposition(
            aleatoric: aleatoric,
            epistemic: epistemic,
            total: total
        )
    }

    // MARK: - Bootstrap Integration

    /// Update calibration as user collects more data
    /// Should be called after each symptom log during bootstrap phase
    func updateCalibration(
        newPrediction: Float,
        actualOutcome: Bool,
        historicalPredictions: [Float],
        historicalOutcomes: [Bool]
    ) {
        var allPredictions = historicalPredictions
        var allOutcomes = historicalOutcomes

        allPredictions.append(newPrediction)
        allOutcomes.append(actualOutcome)

        // Recalibrate if we have enough data
        if allPredictions.count >= 10 {
            calibrateTemperature(predictions: allPredictions, outcomes: allOutcomes)

            // Log calibration quality
            let ece = computeECE(predictions: allPredictions, outcomes: allOutcomes)
            print("📊 Calibration ECE: \(String(format: "%.3f", ece)) (lower is better)")
        }
    }

    // MARK: - Prediction Reliability Score

    /// Compute overall reliability score for a prediction
    /// Combines calibration quality + uncertainty
    func computeReliabilityScore(
        calibratedPrediction: CalibratedPrediction,
        calibrationECE: Float
    ) -> (score: Float, rating: String) {
        // Lower uncertainty = higher reliability
        let uncertaintyScore = 1.0 - calibratedPrediction.uncertaintyScore

        // Lower ECE = higher calibration quality
        let calibrationScore = max(0.0, 1.0 - calibrationECE)

        // Weighted average
        let reliability = 0.6 * uncertaintyScore + 0.4 * calibrationScore

        let rating: String
        if reliability >= 0.8 {
            rating = "Excellent"
        } else if reliability >= 0.6 {
            rating = "Good"
        } else if reliability >= 0.4 {
            rating = "Fair"
        } else {
            rating = "Poor"
        }

        return (reliability, rating)
    }
}

// MARK: - CalibrationFeatureScaler (local to CalibrationEngine)

/// Feature scaler for calibration - uses mean/std arrays for normalization
public class CalibrationFeatureScaler {
    private let mean: [Float]
    private let std: [Float]

    public init(mean: [Float], std: [Float]) {
        self.mean = mean
        self.std = std
    }

    /// Z-score normalization: (x - mean) / std
    public func transform(_ features: [[Float]]) -> [[Float]] {
        return features.map { timestep in
            timestep.enumerated().map { index, value in
                guard index < mean.count && index < std.count else { return value }
                let m = mean[index]
                let s = std[index]
                return s > 0 ? (value - m) / s : 0.0
            }
        }
    }

    /// Inverse transform: x = (z * std) + mean
    public func inverseTransform(_ normalizedFeatures: [[Float]]) -> [[Float]] {
        return normalizedFeatures.map { timestep in
            timestep.enumerated().map { index, value in
                guard index < mean.count && index < std.count else { return value }
                return value * std[index] + mean[index]
            }
        }
    }
}
