//
//  FeatureScaler.swift
//  InflamAI
//
//  Feature normalization for ML prediction inputs
//  Supports both min-max and standard scaling
//

import Foundation

/// Feature scaler for normalizing ML inputs to [-1, 1] or [0, 1] range
public class FeatureScaler {

    // MARK: - Properties

    private let mins: [Float]
    private let maxs: [Float]
    private let means: [Float]?
    private let stds: [Float]?
    private let useStandardScaler: Bool
    public let featureCount: Int

    // MARK: - Initialization

    /// Initialize with min-max scaling parameters
    public init(mins: [Float], maxs: [Float]) {
        self.mins = mins
        self.maxs = maxs
        self.means = nil
        self.stds = nil
        self.useStandardScaler = false
        self.featureCount = mins.count
    }

    /// Initialize with standard scaling parameters (mean/std)
    public init(means: [Float], stds: [Float]) {
        self.mins = []
        self.maxs = []
        self.means = means
        self.stds = stds
        self.useStandardScaler = true
        self.featureCount = means.count
    }

    // MARK: - Transformation

    /// Transform a single feature array to normalized values
    public func transform(_ features: [Float]) -> [Float] {
        if useStandardScaler {
            return standardTransform(features)
        } else {
            return minMaxTransform(features)
        }
    }

    /// Transform a 2D feature array (days x features)
    public func transform2D(_ features: [[Float]]) -> [[Float]] {
        return features.map { transform($0) }
    }

    /// Inverse transform normalized values back to original scale
    public func inverseTransform(_ normalized: [Float]) -> [Float] {
        if useStandardScaler {
            return inverseStandardTransform(normalized)
        } else {
            return inverseMinMaxTransform(normalized)
        }
    }

    // MARK: - Private Helpers

    private func minMaxTransform(_ features: [Float]) -> [Float] {
        guard features.count == mins.count else {
            print("⚠️ [FeatureScaler] Feature count mismatch: got \(features.count), expected \(mins.count)")
            return features
        }

        return zip(features, zip(mins, maxs)).map { feature, bounds in
            let (minVal, maxVal) = bounds
            let range = maxVal - minVal
            guard range > 0 else { return 0.0 }
            // Scale to [0, 1]
            return (feature - minVal) / range
        }
    }

    private func standardTransform(_ features: [Float]) -> [Float] {
        guard let means = means, let stds = stds else { return features }
        guard features.count == means.count else {
            print("⚠️ [FeatureScaler] Feature count mismatch: got \(features.count), expected \(means.count)")
            return features
        }

        return zip(features, zip(means, stds)).map { feature, params in
            let (mean, std) = params
            guard std > 0 else { return 0.0 }
            return (feature - mean) / std
        }
    }

    private func inverseMinMaxTransform(_ normalized: [Float]) -> [Float] {
        guard normalized.count == mins.count else { return normalized }

        return zip(normalized, zip(mins, maxs)).map { value, bounds in
            let (minVal, maxVal) = bounds
            return value * (maxVal - minVal) + minVal
        }
    }

    private func inverseStandardTransform(_ normalized: [Float]) -> [Float] {
        guard let means = means, let stds = stds else { return normalized }
        guard normalized.count == means.count else { return normalized }

        return zip(normalized, zip(means, stds)).map { value, params in
            let (mean, std) = params
            return value * std + mean
        }
    }
}

// MARK: - Default Scaler Factory

extension FeatureScaler {

    /// Create a default min-max scaler for 92 features
    /// Uses reasonable defaults until proper scaling parameters are loaded
    static func createDefault() -> FeatureScaler {
        // Default ranges for 92 features (reasonable medical/health data ranges)
        let defaultMins = [Float](repeating: 0.0, count: 92)
        let defaultMaxs = [Float](repeating: 10.0, count: 92)
        return FeatureScaler(mins: defaultMins, maxs: defaultMaxs)
    }

    /// Load scaler from JSON parameters file
    static func load(from jsonPath: String) -> FeatureScaler? {
        guard let data = FileManager.default.contents(atPath: jsonPath),
              let params = try? JSONDecoder().decode(ScalerParameters.self, from: data) else {
            print("⚠️ [FeatureScaler] Failed to load parameters from \(jsonPath)")
            return nil
        }

        if let mins = params.mins, let maxs = params.maxs {
            return FeatureScaler(mins: mins, maxs: maxs)
        } else if let means = params.means, let stds = params.stds {
            return FeatureScaler(means: means, stds: stds)
        }

        return nil
    }
}

// MARK: - Parameter Structure

struct ScalerParameters: Codable {
    let mins: [Float]?
    let maxs: [Float]?
    let means: [Float]?
    let stds: [Float]?

    enum CodingKeys: String, CodingKey {
        case mins = "feature_mins"
        case maxs = "feature_maxs"
        case means = "feature_means"
        case stds = "feature_stds"
    }
}
