//
//  AdherencePredictionModel.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreML
import CreateML
import os.log

// MARK: - Adherence Prediction Model Implementation
class AdherencePredictionModelImpl: AdherencePredictionModel {
    
    private let logger = Logger(subsystem: "InflamAI", category: "AdherencePredictionModel")
    private var model: MLModel?
    private var trainingData: [AdherenceRecord] = []
    private let modelName = "AdherencePredictionModel"
    
    // Feature extraction parameters
    private let featureWindow = 14 // Days to look back for patterns
    private let minimumTrainingData = 50 // Minimum records needed for training
    
    required init() async throws {
        logger.info("Initializing AdherencePredictionModel")
        
        // Try to load existing model
        await loadExistingModel()
        
        // If no model exists, create a basic rule-based predictor
        if model == nil {
            logger.info("No existing model found, using rule-based predictor")
        }
    }
    
    func predict(features: [String: Double]) async -> (riskLevel: RiskLevel, confidence: Double) {
        logger.info("Predicting adherence risk")
        
        // If we have a trained ML model, use it
        if let mlModel = model {
            return await predictWithMLModel(mlModel, features: features)
        } else {
            // Use rule-based prediction
            return predictWithRules(features: features)
        }
    }
    
    func updateWithNewData(_ data: [AdherenceRecord]) async {
        logger.info("Updating model with \(data.count) new records")
        
        // Add new data to training set
        trainingData.append(contentsOf: data)
        
        // Keep only recent data (last 6 months)
        let sixMonthsAgo = Date().addingTimeInterval(-6 * 30 * 24 * 60 * 60)
        trainingData = trainingData.filter { $0.scheduledTime >= sixMonthsAgo }
        
        // Retrain model if we have enough data
        if trainingData.count >= minimumTrainingData {
            await retrainModel()
        }
    }
    
    // MARK: - Private Methods
    
    private func loadExistingModel() async {
        do {
            // Try to load from app bundle or documents directory
            if let modelURL = getModelURL() {
                model = try MLModel(contentsOf: modelURL)
                logger.info("Loaded existing ML model")
            }
        } catch {
            logger.error("Failed to load existing model: \(error.localizedDescription)")
        }
    }
    
    private func predictWithMLModel(_ mlModel: MLModel, features: [String: Double]) async -> (riskLevel: RiskLevel, confidence: Double) {
        do {
            // Convert features to MLFeatureProvider
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            
            // Make prediction
            let prediction = try mlModel.prediction(from: input)
            
            // Extract risk level and confidence
            let riskScore = prediction.featureValue(for: "riskScore")?.doubleValue ?? 0.5
            let confidence = prediction.featureValue(for: "confidence")?.doubleValue ?? 0.5
            
            let riskLevel = convertScoreToRiskLevel(riskScore)
            
            logger.info("ML prediction: risk=\(riskLevel.rawValue), confidence=\(confidence)")
            return (riskLevel, confidence)
            
        } catch {
            logger.error("ML prediction failed: \(error.localizedDescription)")
            // Fallback to rule-based prediction
            return predictWithRules(features: features)
        }
    }
    
    private func predictWithRules(features: [String: Double]) -> (riskLevel: RiskLevel, confidence: Double) {
        logger.info("Using rule-based prediction")
        
        var riskScore: Double = 0.0
        var confidence: Double = 0.7 // Lower confidence for rule-based
        
        // Recent adherence rate (most important factor)
        let recentAdherenceRate = features["recent_adherence_rate"] ?? 1.0
        riskScore += (1.0 - recentAdherenceRate) * 0.4
        
        // Missed doses in last week
        let recentMissedDoses = features["recent_missed_doses"] ?? 0.0
        riskScore += min(recentMissedDoses / 7.0, 1.0) * 0.3
        
        // Time since last dose
        let timeSinceLastDose = features["time_since_last_dose_hours"] ?? 0.0
        if timeSinceLastDose > 48 { // More than 2 days
            riskScore += 0.2
        } else if timeSinceLastDose > 24 { // More than 1 day
            riskScore += 0.1
        }
        
        // Medication complexity
        let medicationComplexity = features["medication_complexity"] ?? 1.0
        riskScore += (medicationComplexity - 1.0) * 0.1
        
        // Day of week pattern (weekends typically higher risk)
        let dayOfWeek = features["day_of_week"] ?? 1.0
        if dayOfWeek == 1.0 || dayOfWeek == 7.0 { // Sunday or Saturday
            riskScore += 0.05
        }
        
        // Side effects reported
        let sideEffectsReported = features["side_effects_reported"] ?? 0.0
        riskScore += sideEffectsReported * 0.15
        
        // Clamp risk score between 0 and 1
        riskScore = max(0.0, min(1.0, riskScore))
        
        let riskLevel = convertScoreToRiskLevel(riskScore)
        
        logger.info("Rule-based prediction: risk=\(riskLevel.rawValue), score=\(riskScore), confidence=\(confidence)")
        return (riskLevel, confidence)
    }
    
    private func convertScoreToRiskLevel(_ score: Double) -> RiskLevel {
        switch score {
        case 0.0..<0.25:
            return .low
        case 0.25..<0.6:
            return .moderate
        case 0.6...1.0:
            return .high
        default:
            return .unknown
        }
    }
    
    private func retrainModel() async {
        logger.info("Retraining adherence prediction model with \(trainingData.count) records")
        
        do {
            // Prepare training data
            let trainingTable = prepareTrainingData()
            
            // Create and train the model
            let classifier = try MLClassifier(trainingData: trainingTable, targetColumn: "risk_level")
            
            // Save the model
            let modelURL = getModelURL() ?? getDocumentsDirectory().appendingPathComponent("\(modelName).mlmodel")
            try classifier.write(to: modelURL)
            
            // Load the new model
            model = try MLModel(contentsOf: modelURL)
            
            logger.info("Model retrained and saved successfully")
            
        } catch {
            logger.error("Failed to retrain model: \(error.localizedDescription)")
        }
    }
    
    private func prepareTrainingData() -> MLDataTable {
        var data: [String: [Any]] = [
            "recent_adherence_rate": [],
            "recent_missed_doses": [],
            "time_since_last_dose_hours": [],
            "medication_complexity": [],
            "day_of_week": [],
            "side_effects_reported": [],
            "risk_level": []
        ]
        
        for (index, record) in trainingData.enumerated() {
            // Calculate features for this record
            let features = calculateFeaturesForRecord(record, index: index)
            
            // Add to training data
            data["recent_adherence_rate"]?.append(features["recent_adherence_rate"] ?? 1.0)
            data["recent_missed_doses"]?.append(features["recent_missed_doses"] ?? 0.0)
            data["time_since_last_dose_hours"]?.append(features["time_since_last_dose_hours"] ?? 0.0)
            data["medication_complexity"]?.append(features["medication_complexity"] ?? 1.0)
            data["day_of_week"]?.append(features["day_of_week"] ?? 1.0)
            data["side_effects_reported"]?.append(features["side_effects_reported"] ?? 0.0)
            
            // Determine target (risk level based on actual outcome)
            let riskLevel = determineActualRiskLevel(for: record)
            data["risk_level"]?.append(riskLevel.rawValue)
        }
        
        return try! MLDataTable(dictionary: data)
    }
    
    private func calculateFeaturesForRecord(_ record: AdherenceRecord, index: Int) -> [String: Double] {
        var features: [String: Double] = [:]
        
        // Calculate recent adherence rate (last 7 days)
        let recentRecords = Array(trainingData[max(0, index - 7)..<index])
        let adherentCount = recentRecords.filter { $0.wasOnTime }.count
        features["recent_adherence_rate"] = recentRecords.isEmpty ? 1.0 : Double(adherentCount) / Double(recentRecords.count)
        
        // Count recent missed doses
        let missedCount = recentRecords.filter { !$0.wasOnTime }.count
        features["recent_missed_doses"] = Double(missedCount)
        
        // Time since last dose (simulated)
        if index > 0 {
            let timeDiff = record.scheduledTime.timeIntervalSince(trainingData[index - 1].scheduledTime)
            features["time_since_last_dose_hours"] = timeDiff / 3600.0
        } else {
            features["time_since_last_dose_hours"] = 24.0 // Default
        }
        
        // Medication complexity (simulated based on medication name)
        features["medication_complexity"] = calculateMedicationComplexity(record.medicationName)
        
        // Day of week
        let dayOfWeek = Calendar.current.component(.weekday, from: record.scheduledTime)
        features["day_of_week"] = Double(dayOfWeek)
        
        // Side effects (simulated)
        features["side_effects_reported"] = Double.random(in: 0...1) > 0.8 ? 1.0 : 0.0
        
        return features
    }
    
    private func determineActualRiskLevel(for record: AdherenceRecord) -> RiskLevel {
        // Determine risk level based on actual adherence outcome
        if record.wasOnTime {
            return .low
        } else {
            let delay = record.actualTime.timeIntervalSince(record.scheduledTime)
            if delay > 24 * 60 * 60 { // More than 24 hours late
                return .high
            } else if delay > 2 * 60 * 60 { // More than 2 hours late
                return .moderate
            } else {
                return .low
            }
        }
    }
    
    private func calculateMedicationComplexity(_ medicationName: String) -> Double {
        // Simple heuristic for medication complexity
        let complexMedications = ["methotrexate", "biologics", "dmards"]
        let lowercaseName = medicationName.lowercased()
        
        for complex in complexMedications {
            if lowercaseName.contains(complex) {
                return 3.0 // High complexity
            }
        }
        
        return 1.0 // Low complexity
    }
    
    private func getModelURL() -> URL? {
        // Try app bundle first
        if let bundleURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") {
            return bundleURL
        }
        
        // Try documents directory
        let documentsURL = getDocumentsDirectory().appendingPathComponent("\(modelName).mlmodel")
        if FileManager.default.fileExists(atPath: documentsURL.path) {
            return documentsURL
        }
        
        return nil
    }
    
    private func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}

// MARK: - Feature Extraction Utilities
extension AdherencePredictionModelImpl {
    
    func extractFeatures(for medication: Medication, timeframe: TimeInterval, adherenceHistory: [AdherenceRecord]) -> [String: Double] {
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-timeframe)
        
        // Filter relevant records
        let relevantRecords = adherenceHistory.filter {
            $0.medicationName == medication.name &&
            $0.scheduledTime >= startDate &&
            $0.scheduledTime <= endDate
        }
        
        var features: [String: Double] = [:]
        
        // Basic adherence metrics
        let totalRecords = relevantRecords.count
        let adherentRecords = relevantRecords.filter { $0.wasOnTime }.count
        
        features["recent_adherence_rate"] = totalRecords > 0 ? Double(adherentRecords) / Double(totalRecords) : 1.0
        features["recent_missed_doses"] = Double(totalRecords - adherentRecords)
        features["total_doses_scheduled"] = Double(totalRecords)
        
        // Timing patterns
        if let lastRecord = relevantRecords.last {
            features["time_since_last_dose_hours"] = endDate.timeIntervalSince(lastRecord.actualTime) / 3600.0
        } else {
            features["time_since_last_dose_hours"] = 48.0 // Default to 48 hours if no recent data
        }
        
        // Calculate average delay for non-adherent doses
        let nonAdherentRecords = relevantRecords.filter { !$0.wasOnTime }
        if !nonAdherentRecords.isEmpty {
            let totalDelay = nonAdherentRecords.reduce(0.0) { sum, record in
                sum + record.actualTime.timeIntervalSince(record.scheduledTime)
            }
            features["average_delay_hours"] = (totalDelay / Double(nonAdherentRecords.count)) / 3600.0
        } else {
            features["average_delay_hours"] = 0.0
        }
        
        // Medication-specific features
        features["medication_complexity"] = calculateMedicationComplexity(medication.name)
        features["doses_per_day"] = Double(medication.dosesPerDay ?? 1)
        
        // Temporal features
        let now = Date()
        let calendar = Calendar.current
        features["day_of_week"] = Double(calendar.component(.weekday, from: now))
        features["hour_of_day"] = Double(calendar.component(.hour, from: now))
        features["is_weekend"] = [1, 7].contains(calendar.component(.weekday, from: now)) ? 1.0 : 0.0
        
        // Trend analysis
        if relevantRecords.count >= 7 {
            let recentWeek = Array(relevantRecords.suffix(7))
            let previousWeek = Array(relevantRecords.dropLast(7).suffix(7))
            
            let recentAdherence = Double(recentWeek.filter { $0.wasOnTime }.count) / Double(recentWeek.count)
            let previousAdherence = previousWeek.isEmpty ? recentAdherence : Double(previousWeek.filter { $0.wasOnTime }.count) / Double(previousWeek.count)
            
            features["adherence_trend"] = recentAdherence - previousAdherence
        } else {
            features["adherence_trend"] = 0.0
        }
        
        // Risk factors (simulated - in real app, these would come from user data)
        features["side_effects_reported"] = Double.random(in: 0...1) > 0.8 ? 1.0 : 0.0
        features["stress_level"] = Double.random(in: 0...10) / 10.0
        features["sleep_quality"] = Double.random(in: 0...10) / 10.0
        
        logger.info("Extracted \(features.count) features for \(medication.name)")
        return features
    }
}