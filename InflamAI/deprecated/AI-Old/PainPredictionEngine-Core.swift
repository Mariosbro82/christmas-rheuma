//
//  PainPredictionEngine.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Combine
import CoreML
import CreateML

// MARK: - Pain Prediction Engine
@MainActor
class PainPredictionEngine: ObservableObject {
    
    // MARK: - Published Properties
    @Published var isTraining = false
    @Published var predictionAccuracy: Double = 0.0
    @Published var lastPrediction: PainPrediction?
    @Published var modelVersion: String = "1.0.0"
    @Published var trainingProgress: Double = 0.0
    
    // MARK: - Private Properties
    private var mlModel: MLModel?
    private var trainingData: [PainDataPoint] = []
    private var correlationEngine: CorrelationAnalysisEngine
    private var weatherService: WeatherService
    private var activityTracker: ActivityTracker
    private let modelUpdateInterval: TimeInterval = 86400 * 7 // 7 days
    private var lastModelUpdate: Date?
    
    // MARK: - Initialization
    init() {
        self.correlationEngine = CorrelationAnalysisEngine()
        self.weatherService = WeatherService()
        self.activityTracker = ActivityTracker()
        
        Task {
            await loadModel()
            await startPeriodicModelUpdates()
        }
    }
    
    // MARK: - Model Management
    
    private func loadModel() async {
        do {
            if let modelURL = getModelURL() {
                mlModel = try MLModel(contentsOf: modelURL)
                print("Pain prediction model loaded successfully")
            } else {
                await trainInitialModel()
            }
        } catch {
            print("Failed to load pain prediction model: \(error)")
            await trainInitialModel()
        }
    }
    
    private func getModelURL() -> URL? {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        return documentsPath.appendingPathComponent("PainPredictionModel.mlmodel")
    }
    
    private func trainInitialModel() async {
        isTraining = true
        trainingProgress = 0.0
        
        // Generate initial training data if none exists
        if trainingData.isEmpty {
            trainingData = generateSyntheticTrainingData()
        }
        
        await trainModel(with: trainingData)
        
        isTraining = false
        lastModelUpdate = Date()
    }
    
    private func trainModel(with data: [PainDataPoint]) async {
        do {
            // Convert data to CreateML format
            let trainingTable = try createTrainingTable(from: data)
            
            // Create and train the model
            trainingProgress = 0.3
            let regressor = try MLRegressor(trainingData: trainingTable, targetColumn: "painLevel")
            
            trainingProgress = 0.7
            
            // Save the model
            if let modelURL = getModelURL() {
                try regressor.write(to: modelURL)
                mlModel = try MLModel(contentsOf: modelURL)
            }
            
            trainingProgress = 1.0
            
            // Evaluate model accuracy
            predictionAccuracy = await evaluateModel(with: data)
            
            print("Model training completed with accuracy: \(predictionAccuracy)")
            
        } catch {
            print("Model training failed: \(error)")
        }
    }
    
    private func createTrainingTable(from data: [PainDataPoint]) throws -> MLDataTable {
        var painLevels: [Double] = []
        var weatherPressures: [Double] = []
        var temperatures: [Double] = []
        var humidities: [Double] = []
        var sleepQualities: [Double] = []
        var stressLevels: [Double] = []
        var activityLevels: [Double] = []
        var medicationAdherence: [Double] = []
        var timeOfDay: [Double] = []
        var dayOfWeek: [Double] = []
        
        for dataPoint in data {
            painLevels.append(dataPoint.painLevel)
            weatherPressures.append(dataPoint.weatherPressure)
            temperatures.append(dataPoint.temperature)
            humidities.append(dataPoint.humidity)
            sleepQualities.append(dataPoint.sleepQuality)
            stressLevels.append(dataPoint.stressLevel)
            activityLevels.append(dataPoint.activityLevel)
            medicationAdherence.append(dataPoint.medicationAdherence)
            timeOfDay.append(dataPoint.timeOfDay)
            dayOfWeek.append(dataPoint.dayOfWeek)
        }
        
        let dataTable = try MLDataTable(dictionary: [
            "painLevel": painLevels,
            "weatherPressure": weatherPressures,
            "temperature": temperatures,
            "humidity": humidities,
            "sleepQuality": sleepQualities,
            "stressLevel": stressLevels,
            "activityLevel": activityLevels,
            "medicationAdherence": medicationAdherence,
            "timeOfDay": timeOfDay,
            "dayOfWeek": dayOfWeek
        ])
        
        return dataTable
    }
    
    private func evaluateModel(with testData: [PainDataPoint]) async -> Double {
        guard let model = mlModel else { return 0.0 }
        
        var totalError: Double = 0.0
        var validPredictions = 0
        
        for dataPoint in testData {
            if let prediction = await makePrediction(for: dataPoint) {
                let error = abs(prediction.predictedPainLevel - dataPoint.painLevel)
                totalError += error
                validPredictions += 1
            }
        }
        
        guard validPredictions > 0 else { return 0.0 }
        
        let meanAbsoluteError = totalError / Double(validPredictions)
        let accuracy = max(0.0, 1.0 - (meanAbsoluteError / 10.0)) // Normalize to 0-1 scale
        
        return accuracy
    }
    
    // MARK: - Prediction
    
    func predictPain(for date: Date = Date()) async -> PainPrediction? {
        guard let model = mlModel else {
            print("No model available for prediction")
            return nil
        }
        
        // Gather current data
        let currentData = await gatherCurrentData(for: date)
        
        // Make prediction
        let prediction = await makePrediction(for: currentData)
        
        lastPrediction = prediction
        return prediction
    }
    
    private func makePrediction(for dataPoint: PainDataPoint) async -> PainPrediction? {
        guard let model = mlModel else { return nil }
        
        do {
            let input = try createMLInput(from: dataPoint)
            let output = try model.prediction(from: input)
            
            if let painLevel = output.featureValue(for: "painLevel")?.doubleValue {
                let confidence = calculateConfidence(for: dataPoint)
                let factors = await identifyContributingFactors(for: dataPoint)
                
                return PainPrediction(
                    predictedPainLevel: max(0, min(10, painLevel)),
                    confidence: confidence,
                    predictionDate: dataPoint.timestamp,
                    contributingFactors: factors,
                    recommendations: generateRecommendations(for: painLevel, factors: factors)
                )
            }
        } catch {
            print("Prediction failed: \(error)")
        }
        
        return nil
    }
    
    private func createMLInput(from dataPoint: PainDataPoint) throws -> MLFeatureProvider {
        let inputDict: [String: Any] = [
            "weatherPressure": dataPoint.weatherPressure,
            "temperature": dataPoint.temperature,
            "humidity": dataPoint.humidity,
            "sleepQuality": dataPoint.sleepQuality,
            "stressLevel": dataPoint.stressLevel,
            "activityLevel": dataPoint.activityLevel,
            "medicationAdherence": dataPoint.medicationAdherence,
            "timeOfDay": dataPoint.timeOfDay,
            "dayOfWeek": dataPoint.dayOfWeek
        ]
        
        return try MLDictionaryFeatureProvider(dictionary: inputDict)
    }
    
    private func gatherCurrentData(for date: Date) async -> PainDataPoint {
        let weather = await weatherService.getCurrentWeather()
        let activity = await activityTracker.getCurrentActivity()
        let calendar = Calendar.current
        
        return PainDataPoint(
            timestamp: date,
            painLevel: 0.0, // Will be predicted
            weatherPressure: weather?.pressure ?? 1013.25,
            temperature: weather?.temperature ?? 20.0,
            humidity: weather?.humidity ?? 50.0,
            sleepQuality: await getSleepQuality(),
            stressLevel: await getStressLevel(),
            activityLevel: activity?.level ?? 5.0,
            medicationAdherence: await getMedicationAdherence(),
            timeOfDay: Double(calendar.component(.hour, from: date)),
            dayOfWeek: Double(calendar.component(.weekday, from: date))
        )
    }
    
    // MARK: - Data Collection
    
    func addTrainingData(_ dataPoint: PainDataPoint) async {
        trainingData.append(dataPoint)
        
        // Retrain model if we have enough new data
        if trainingData.count % 50 == 0 {
            await retrainModel()
        }
    }
    
    private func retrainModel() async {
        guard !isTraining else { return }
        
        isTraining = true
        await trainModel(with: trainingData)
        isTraining = false
    }
    
    // MARK: - Helper Methods
    
    private func calculateConfidence(for dataPoint: PainDataPoint) -> Double {
        // Calculate confidence based on data quality and model certainty
        var confidence: Double = 0.8
        
        // Adjust based on data completeness
        if dataPoint.weatherPressure == 0 { confidence -= 0.1 }
        if dataPoint.sleepQuality == 0 { confidence -= 0.1 }
        if dataPoint.stressLevel == 0 { confidence -= 0.1 }
        
        return max(0.0, min(1.0, confidence))
    }
    
    private func identifyContributingFactors(for dataPoint: PainDataPoint) async -> [ContributingFactor] {
        var factors: [ContributingFactor] = []
        
        // Weather factors
        if dataPoint.weatherPressure < 1000 {
            factors.append(ContributingFactor(
                name: "Low Atmospheric Pressure",
                impact: 0.7,
                description: "Low pressure systems may increase joint pain"
            ))
        }
        
        if dataPoint.humidity > 70 {
            factors.append(ContributingFactor(
                name: "High Humidity",
                impact: 0.5,
                description: "High humidity can worsen inflammation"
            ))
        }
        
        // Sleep factors
        if dataPoint.sleepQuality < 5 {
            factors.append(ContributingFactor(
                name: "Poor Sleep Quality",
                impact: 0.8,
                description: "Poor sleep can increase pain sensitivity"
            ))
        }
        
        // Stress factors
        if dataPoint.stressLevel > 7 {
            factors.append(ContributingFactor(
                name: "High Stress Level",
                impact: 0.6,
                description: "Stress can trigger inflammatory responses"
            ))
        }
        
        // Activity factors
        if dataPoint.activityLevel < 3 {
            factors.append(ContributingFactor(
                name: "Low Activity Level",
                impact: 0.4,
                description: "Lack of movement can increase stiffness"
            ))
        } else if dataPoint.activityLevel > 8 {
            factors.append(ContributingFactor(
                name: "High Activity Level",
                impact: 0.5,
                description: "Overexertion may trigger flare-ups"
            ))
        }
        
        // Medication factors
        if dataPoint.medicationAdherence < 0.8 {
            factors.append(ContributingFactor(
                name: "Poor Medication Adherence",
                impact: 0.9,
                description: "Missing medications can lead to increased symptoms"
            ))
        }
        
        return factors
    }
    
    private func generateRecommendations(for painLevel: Double, factors: [ContributingFactor]) -> [String] {
        var recommendations: [String] = []
        
        if painLevel > 7 {
            recommendations.append("Consider taking prescribed pain medication")
            recommendations.append("Apply heat or cold therapy as recommended")
            recommendations.append("Practice gentle stretching or relaxation techniques")
        }
        
        for factor in factors {
            switch factor.name {
            case "Low Atmospheric Pressure":
                recommendations.append("Stay warm and consider indoor activities")
            case "High Humidity":
                recommendations.append("Use a dehumidifier if possible")
            case "Poor Sleep Quality":
                recommendations.append("Focus on improving sleep hygiene tonight")
            case "High Stress Level":
                recommendations.append("Practice stress reduction techniques")
            case "Low Activity Level":
                recommendations.append("Try gentle movement or stretching")
            case "High Activity Level":
                recommendations.append("Consider reducing activity intensity")
            case "Poor Medication Adherence":
                recommendations.append("Take your medications as prescribed")
            default:
                break
            }
        }
        
        return recommendations
    }
    
    // MARK: - Periodic Updates
    
    private func startPeriodicModelUpdates() async {
        Timer.scheduledTimer(withTimeInterval: modelUpdateInterval, repeats: true) { _ in
            Task {
                await self.checkForModelUpdate()
            }
        }
    }
    
    private func checkForModelUpdate() async {
        guard let lastUpdate = lastModelUpdate else {
            await retrainModel()
            return
        }
        
        if Date().timeIntervalSince(lastUpdate) > modelUpdateInterval {
            await retrainModel()
        }
    }
    
    // MARK: - Data Helpers
    
    private func getSleepQuality() async -> Double {
        // In a real implementation, this would integrate with HealthKit
        return Double.random(in: 3...9)
    }
    
    private func getStressLevel() async -> Double {
        // In a real implementation, this would use HRV or other stress indicators
        return Double.random(in: 1...8)
    }
    
    private func getMedicationAdherence() async -> Double {
        // In a real implementation, this would track actual medication taking
        return Double.random(in: 0.7...1.0)
    }
    
    // MARK: - Synthetic Data Generation
    
    private func generateSyntheticTrainingData() -> [PainDataPoint] {
        var data: [PainDataPoint] = []
        let calendar = Calendar.current
        
        for i in 0..<1000 {
            let date = Date().addingTimeInterval(-Double(i) * 3600) // Hourly data for last 1000 hours
            
            let pressure = Double.random(in: 980...1040)
            let temperature = Double.random(in: -10...35)
            let humidity = Double.random(in: 20...90)
            let sleepQuality = Double.random(in: 1...10)
            let stressLevel = Double.random(in: 1...10)
            let activityLevel = Double.random(in: 0...10)
            let medicationAdherence = Double.random(in: 0.5...1.0)
            
            // Generate pain level based on factors
            var painLevel = 5.0
            painLevel += (1013.25 - pressure) * 0.01 // Pressure effect
            painLevel += max(0, 10 - sleepQuality) * 0.3 // Sleep effect
            painLevel += (stressLevel - 5) * 0.2 // Stress effect
            painLevel += (1.0 - medicationAdherence) * 5 // Medication effect
            painLevel += Double.random(in: -1...1) // Random noise
            
            painLevel = max(0, min(10, painLevel))
            
            let dataPoint = PainDataPoint(
                timestamp: date,
                painLevel: painLevel,
                weatherPressure: pressure,
                temperature: temperature,
                humidity: humidity,
                sleepQuality: sleepQuality,
                stressLevel: stressLevel,
                activityLevel: activityLevel,
                medicationAdherence: medicationAdherence,
                timeOfDay: Double(calendar.component(.hour, from: date)),
                dayOfWeek: Double(calendar.component(.weekday, from: date))
            )
            
            data.append(dataPoint)
        }
        
        return data
    }
}

// MARK: - Supporting Types

struct PainDataPoint {
    let timestamp: Date
    let painLevel: Double
    let weatherPressure: Double
    let temperature: Double
    let humidity: Double
    let sleepQuality: Double
    let stressLevel: Double
    let activityLevel: Double
    let medicationAdherence: Double
    let timeOfDay: Double
    let dayOfWeek: Double
}

struct PainPrediction {
    let predictedPainLevel: Double
    let confidence: Double
    let predictionDate: Date
    let contributingFactors: [ContributingFactor]
    let recommendations: [String]
    
    var severityLevel: PainSeverity {
        switch predictedPainLevel {
        case 0..<3:
            return .mild
        case 3..<6:
            return .moderate
        case 6..<8:
            return .severe
        default:
            return .extreme
        }
    }
}

struct ContributingFactor {
    let name: String
    let impact: Double // 0.0 to 1.0
    let description: String
}

enum PainSeverity: String, CaseIterable {
    case mild = "Mild"
    case moderate = "Moderate"
    case severe = "Severe"
    case extreme = "Extreme"
    
    var color: String {
        switch self {
        case .mild:
            return "green"
        case .moderate:
            return "yellow"
        case .severe:
            return "orange"
        case .extreme:
            return "red"
        }
    }
}