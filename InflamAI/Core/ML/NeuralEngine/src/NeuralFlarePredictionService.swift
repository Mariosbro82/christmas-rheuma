//
//  NeuralFlarePredictionService.swift
//  InflamAI
//
//  Created by OVERLORD Neural Engine
//

import Foundation
import Combine
import CoreML
import CoreData

/// High-level service for flare prediction using the updatable neural network
@MainActor
class NeuralFlarePredictionService: ObservableObject {
    static let shared = NeuralFlarePredictionService()
    
    // MARK: - Published Properties
    @Published var prediction: UpdatableFlarePredictor.FlarePrediction?
    @Published var isModelReady = false
    @Published var lastUpdateDate: Date?
    @Published var error: String?
    
    // MARK: - Private Properties
    private let predictor = UpdatableFlarePredictor()
    private var cancellables = Set<AnyCancellable>()
    
    private init() {
        setupBindings()
    }
    
    private func setupBindings() {
        // Bind to predictor state
        predictor.$isModelLoaded
            .receive(on: DispatchQueue.main)
            .assign(to: \.isModelReady, on: self)
            .store(in: &cancellables)
            
        predictor.$lastPrediction
            .receive(on: DispatchQueue.main)
            .assign(to: \.prediction, on: self)
            .store(in: &cancellables)
            
        predictor.$errorMessage
            .receive(on: DispatchQueue.main)
            .assign(to: \.error, on: self)
            .store(in: &cancellables)
    }
    
    /// Run prediction with current user data
    func predict() async {
        guard isModelReady else {
            error = "Model not ready"
            return
        }
        
        do {
            // 1. Fetch features (placeholder for now)
            // In real app, this would fetch from Core Data and normalize
            let features = try await fetchCurrentFeatures()
            
            // 2. Run prediction
            _ = await predictor.predict(features: features)
            
        } catch {
            self.error = "Prediction failed: \(error.localizedDescription)"
        }
    }
    
    /// Fetch and prepare features for prediction
    private func fetchCurrentFeatures() async throws -> [[Float]] {
        // Use FeatureExtractor to get real symptom data
        let extractor = FeatureExtractor()
        let features = await extractor.extract30DayFeatures(endingOn: Date())
        
        guard features.count == 30, features.first?.count == 92 else {
            throw PredictionError.invalidInputShape(
                expected: "(30, 92)",
                got: "(\(features.count), \(features.first?.count ?? 0))"
            )
        }
        
        return features
    }
    
    /// Trigger manual model update (e.g. for testing)
    func forceUpdate() async {
        do {
            // Fetch training data
            let trainingData = try await fetchTrainingData()
            
            // Run update
            try await predictor.updateModel(with: trainingData)
            
            lastUpdateDate = Date()
            
        } catch {
            self.error = "Update failed: \(error.localizedDescription)"
        }
    }
    
    /// Fetch training data from Core Data for model updates
    /// Uses shared TrainingDataCollector for consistency
    private func fetchTrainingData() async throws -> [(features: [[Float]], label: Int)] {
        let context = InflamAIPersistenceController.shared.container.viewContext
        let featureExtractor = FeatureExtractor()

        print("🔄 [NeuralFlarePredictionService] Fetching training data...")

        return try await TrainingDataCollector.collectTrainingData(
            context: context,
            featureExtractor: featureExtractor
        )
    }
}
