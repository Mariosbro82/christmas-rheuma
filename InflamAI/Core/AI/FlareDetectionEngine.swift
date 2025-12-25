//
//  FlareDetectionEngine.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Combine
import CoreML
import HealthKit

// MARK: - Flare Detection Engine
@MainActor
class FlareDetectionEngine: ObservableObject {
    
    // MARK: - Published Properties
    @Published var currentRiskLevel: FlareRiskLevel = .low
    @Published var riskScore: Double = 0.0
    @Published var predictedFlareDate: Date?
    @Published var earlyWarnings: [EarlyWarning] = []
    @Published var isAnalyzing = false
    @Published var lastAnalysis: Date?
    @Published var flareHistory: [FlareEvent] = []
    @Published var riskFactors: [RiskFactor] = []
    
    // MARK: - Private Properties
    private var healthData: [HealthDataPoint] = []
    private var environmentalData: [EnvironmentalDataPoint] = []
    private var behaviorData: [BehaviorDataPoint] = []
    private var mlModel: MLModel?
    private var patternRecognizer: PatternRecognizer
    private var riskCalculator: RiskCalculator
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Configuration
    private let analysisInterval: TimeInterval = 3600 // 1 hour
    private let predictionWindow: TimeInterval = 7 * 24 * 3600 // 7 days
    private let minimumDataPoints = 50
    private let highRiskThreshold = 0.7
    private let moderateRiskThreshold = 0.4
    
    // MARK: - Initialization
    init() {
        self.patternRecognizer = PatternRecognizer()
        self.riskCalculator = RiskCalculator()
        
        setupPeriodicAnalysis()
        loadHistoricalData()
        loadMLModel()
    }
    
    // MARK: - Public Methods
    
    func addHealthDataPoint(_ dataPoint: HealthDataPoint) async {
        healthData.append(dataPoint)
        
        // Trigger analysis if significant change detected
        if isSignificantChange(dataPoint) {
            await performFlareAnalysis()
        }
    }
    
    func addEnvironmentalData(_ dataPoint: EnvironmentalDataPoint) async {
        environmentalData.append(dataPoint)
        
        // Check for environmental triggers
        if isEnvironmentalTrigger(dataPoint) {
            await performFlareAnalysis()
        }
    }
    
    func addBehaviorData(_ dataPoint: BehaviorDataPoint) async {
        behaviorData.append(dataPoint)
        
        // Analyze behavior patterns
        if isBehaviorRiskFactor(dataPoint) {
            await performFlareAnalysis()
        }
    }
    
    func performFlareAnalysis() async {
        guard healthData.count >= minimumDataPoints else {
            print("Insufficient data for flare analysis")
            return
        }
        
        isAnalyzing = true
        
        await withCheckedContinuation { continuation in
            Task {
                await self.runFlareAnalysis()
                continuation.resume()
            }
        }
        
        isAnalyzing = false
        lastAnalysis = Date()
    }
    
    func recordFlareEvent(_ event: FlareEvent) async {
        flareHistory.append(event)
        
        // Update ML model with new flare data
        await updateMLModel(with: event)
        
        // Recalibrate risk assessment
        await recalibrateRiskAssessment()
    }
    
    func getPredictionConfidence() -> Double {
        guard let lastAnalysis = lastAnalysis else { return 0.0 }
        
        let timeSinceAnalysis = Date().timeIntervalSince(lastAnalysis)
        let maxAge: TimeInterval = 24 * 3600 // 24 hours
        
        return max(0.0, 1.0 - (timeSinceAnalysis / maxAge))
    }
    
    func getDetailedRiskAssessment() async -> DetailedRiskAssessment {
        let patterns = await patternRecognizer.analyzePatterns(
            healthData: healthData,
            environmentalData: environmentalData,
            behaviorData: behaviorData
        )
        
        let riskFactors = await identifyRiskFactors(patterns: patterns)
        let protectiveFactors = await identifyProtectiveFactors(patterns: patterns)
        
        return DetailedRiskAssessment(
            overallRisk: riskScore,
            riskLevel: currentRiskLevel,
            riskFactors: riskFactors,
            protectiveFactors: protectiveFactors,
            confidence: getPredictionConfidence(),
            recommendations: generateRecommendations(riskFactors: riskFactors)
        )
    }
    
    // MARK: - Private Analysis Methods
    
    private func runFlareAnalysis() async {
        // Step 1: Pattern Recognition
        let patterns = await patternRecognizer.analyzePatterns(
            healthData: healthData,
            environmentalData: environmentalData,
            behaviorData: behaviorData
        )
        
        // Step 2: Risk Calculation
        let risk = await riskCalculator.calculateRisk(
            patterns: patterns,
            flareHistory: flareHistory
        )
        
        // Step 3: ML Prediction (if model available)
        let mlPrediction = await performMLPrediction(patterns: patterns)
        
        // Step 4: Combine results
        let combinedRisk = combineRiskScores(traditional: risk, ml: mlPrediction)
        
        // Step 5: Generate warnings and recommendations
        let warnings = await generateEarlyWarnings(risk: combinedRisk, patterns: patterns)
        
        await MainActor.run {
            self.riskScore = combinedRisk.score
            self.currentRiskLevel = combinedRisk.level
            self.predictedFlareDate = combinedRisk.predictedDate
            self.earlyWarnings = warnings
            self.riskFactors = combinedRisk.factors
        }
    }
    
    private func isSignificantChange(_ dataPoint: HealthDataPoint) -> Bool {
        guard let lastDataPoint = healthData.last else { return false }
        
        let painIncrease = dataPoint.painLevel - lastDataPoint.painLevel
        let stiffnessIncrease = dataPoint.jointStiffness - lastDataPoint.jointStiffness
        let fatigueIncrease = dataPoint.fatigueLevel - lastDataPoint.fatigueLevel
        
        return painIncrease > 2.0 || stiffnessIncrease > 2.0 || fatigueIncrease > 2.0
    }
    
    private func isEnvironmentalTrigger(_ dataPoint: EnvironmentalDataPoint) -> Bool {
        // Check for known environmental triggers
        let pressureDrop = dataPoint.barometricPressure < 1000 // Low pressure
        let highHumidity = dataPoint.humidity > 80
        let temperatureChange = abs(dataPoint.temperature - (environmentalData.last?.temperature ?? dataPoint.temperature)) > 10
        
        return pressureDrop || highHumidity || temperatureChange
    }
    
    private func isBehaviorRiskFactor(_ dataPoint: BehaviorDataPoint) -> Bool {
        return dataPoint.sleepQuality < 5.0 || dataPoint.stressLevel > 7.0 || dataPoint.medicationAdherence < 0.8
    }
    
    private func performMLPrediction(patterns: PatternAnalysisResult) async -> MLPredictionResult? {
        guard let model = mlModel else { return nil }
        
        // In a real implementation, this would use the actual ML model
        // For now, we'll simulate ML prediction
        
        let features = extractMLFeatures(from: patterns)
        let prediction = simulateMLPrediction(features: features)
        
        return MLPredictionResult(
            riskScore: prediction.riskScore,
            confidence: prediction.confidence,
            predictedDays: prediction.predictedDays
        )
    }
    
    private func extractMLFeatures(from patterns: PatternAnalysisResult) -> [Double] {
        return [
            patterns.painTrend,
            patterns.stiffnessTrend,
            patterns.fatigueTrend,
            patterns.sleepQualityTrend,
            patterns.stressTrend,
            patterns.weatherCorrelation,
            patterns.medicationAdherence,
            patterns.activityLevel
        ]
    }
    
    private func simulateMLPrediction(features: [Double]) -> (riskScore: Double, confidence: Double, predictedDays: Int) {
        // Simplified ML simulation
        let weightedSum = features.enumerated().reduce(0.0) { sum, element in
            let weight = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.1, 0.05][element.offset]
            return sum + element.element * weight
        }
        
        let riskScore = min(1.0, max(0.0, weightedSum / 10.0))
        let confidence = 0.8 // Simulated confidence
        let predictedDays = Int(7 * (1 - riskScore)) // Higher risk = sooner prediction
        
        return (riskScore, confidence, predictedDays)
    }
    
    private func combineRiskScores(traditional: RiskAssessmentResult, ml: MLPredictionResult?) -> CombinedRiskResult {
        let traditionalWeight = 0.6
        let mlWeight = 0.4
        
        let combinedScore: Double
        let predictedDate: Date?
        
        if let ml = ml {
            combinedScore = traditional.score * traditionalWeight + ml.riskScore * mlWeight
            let avgDays = (traditional.predictedDays + ml.predictedDays) / 2
            predictedDate = Date().addingTimeInterval(TimeInterval(avgDays * 24 * 3600))
        } else {
            combinedScore = traditional.score
            predictedDate = Date().addingTimeInterval(TimeInterval(traditional.predictedDays * 24 * 3600))
        }
        
        let level: FlareRiskLevel
        if combinedScore >= highRiskThreshold {
            level = .high
        } else if combinedScore >= moderateRiskThreshold {
            level = .moderate
        } else {
            level = .low
        }
        
        return CombinedRiskResult(
            score: combinedScore,
            level: level,
            predictedDate: predictedDate,
            factors: traditional.factors
        )
    }
    
    private func generateEarlyWarnings(risk: CombinedRiskResult, patterns: PatternAnalysisResult) async -> [EarlyWarning] {
        var warnings: [EarlyWarning] = []
        
        if risk.level == .high {
            warnings.append(EarlyWarning(
                id: UUID().uuidString,
                type: .highRisk,
                severity: .critical,
                message: "High flare risk detected. Consider contacting your healthcare provider.",
                timestamp: Date(),
                actionItems: [
                    "Review medication adherence",
                    "Implement stress reduction techniques",
                    "Monitor symptoms closely",
                    "Consider preventive measures"
                ]
            ))
        }
        
        if patterns.painTrend > 1.5 {
            warnings.append(EarlyWarning(
                id: UUID().uuidString,
                type: .increasingPain,
                severity: .warning,
                message: "Pain levels have been increasing over the past few days.",
                timestamp: Date(),
                actionItems: [
                    "Apply heat/cold therapy",
                    "Practice gentle exercises",
                    "Consider pain management techniques"
                ]
            ))
        }
        
        if patterns.sleepQualityTrend < -1.0 {
            warnings.append(EarlyWarning(
                id: UUID().uuidString,
                type: .poorSleep,
                severity: .warning,
                message: "Sleep quality has been declining, which may increase flare risk.",
                timestamp: Date(),
                actionItems: [
                    "Establish consistent sleep schedule",
                    "Create relaxing bedtime routine",
                    "Limit screen time before bed"
                ]
            ))
        }
        
        if patterns.stressTrend > 1.5 {
            warnings.append(EarlyWarning(
                id: UUID().uuidString,
                type: .highStress,
                severity: .warning,
                message: "Stress levels have been elevated, which may trigger a flare.",
                timestamp: Date(),
                actionItems: [
                    "Practice meditation or deep breathing",
                    "Engage in stress-reducing activities",
                    "Consider talking to a counselor"
                ]
            ))
        }
        
        return warnings
    }
    
    private func identifyRiskFactors(patterns: PatternAnalysisResult) async -> [RiskFactor] {
        var factors: [RiskFactor] = []
        
        if patterns.painTrend > 1.0 {
            factors.append(RiskFactor(
                name: "Increasing Pain",
                impact: .high,
                description: "Pain levels have been trending upward",
                value: patterns.painTrend
            ))
        }
        
        if patterns.sleepQualityTrend < -0.5 {
            factors.append(RiskFactor(
                name: "Poor Sleep Quality",
                impact: .moderate,
                description: "Sleep quality has been declining",
                value: abs(patterns.sleepQualityTrend)
            ))
        }
        
        if patterns.stressTrend > 1.0 {
            factors.append(RiskFactor(
                name: "Elevated Stress",
                impact: .high,
                description: "Stress levels have been increasing",
                value: patterns.stressTrend
            ))
        }
        
        if patterns.medicationAdherence < 0.8 {
            factors.append(RiskFactor(
                name: "Poor Medication Adherence",
                impact: .high,
                description: "Medication adherence below optimal level",
                value: 1.0 - patterns.medicationAdherence
            ))
        }
        
        if patterns.weatherCorrelation > 0.5 {
            factors.append(RiskFactor(
                name: "Weather Sensitivity",
                impact: .moderate,
                description: "Strong correlation with weather changes",
                value: patterns.weatherCorrelation
            ))
        }
        
        return factors.sorted { $0.impact.rawValue > $1.impact.rawValue }
    }
    
    private func identifyProtectiveFactors(patterns: PatternAnalysisResult) async -> [ProtectiveFactor] {
        var factors: [ProtectiveFactor] = []
        
        if patterns.medicationAdherence > 0.9 {
            factors.append(ProtectiveFactor(
                name: "Excellent Medication Adherence",
                benefit: .high,
                description: "Consistently taking medications as prescribed",
                value: patterns.medicationAdherence
            ))
        }
        
        if patterns.activityLevel > 7.0 {
            factors.append(ProtectiveFactor(
                name: "Regular Physical Activity",
                benefit: .moderate,
                description: "Maintaining good activity levels",
                value: patterns.activityLevel / 10.0
            ))
        }
        
        if patterns.sleepQualityTrend > 0.5 {
            factors.append(ProtectiveFactor(
                name: "Improving Sleep Quality",
                benefit: .moderate,
                description: "Sleep quality has been improving",
                value: patterns.sleepQualityTrend
            ))
        }
        
        return factors
    }
    
    private func generateRecommendations(riskFactors: [RiskFactor]) -> [String] {
        var recommendations: [String] = []
        
        for factor in riskFactors {
            switch factor.name {
            case "Increasing Pain":
                recommendations.append("Consider applying heat/cold therapy and gentle stretching")
            case "Poor Sleep Quality":
                recommendations.append("Establish a consistent sleep schedule and bedtime routine")
            case "Elevated Stress":
                recommendations.append("Practice stress reduction techniques like meditation or deep breathing")
            case "Poor Medication Adherence":
                recommendations.append("Set up medication reminders and discuss barriers with your healthcare provider")
            case "Weather Sensitivity":
                recommendations.append("Monitor weather forecasts and prepare for weather-related symptom changes")
            default:
                break
            }
        }
        
        return recommendations
    }
    
    // MARK: - Setup and Configuration
    
    private func setupPeriodicAnalysis() {
        Timer.publish(every: analysisInterval, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.performFlareAnalysis()
                }
            }
            .store(in: &cancellables)
    }
    
    private func loadHistoricalData() {
        // In a real implementation, load from Core Data
        generateSampleData()
    }
    
    private func loadMLModel() {
        // In a real implementation, load trained Core ML model
        // For now, we'll use rule-based prediction
    }
    
    private func updateMLModel(with event: FlareEvent) async {
        // In a real implementation, retrain or update the ML model
        // with new flare event data
    }
    
    private func recalibrateRiskAssessment() async {
        // Adjust risk thresholds based on new flare data
        await performFlareAnalysis()
    }
    
    private func generateSampleData() {
        let calendar = Calendar.current
        
        for i in 0..<100 {
            let date = Date().addingTimeInterval(-Double(i) * 86400)
            
            let healthPoint = HealthDataPoint(
                timestamp: date,
                painLevel: Double.random(in: 1...10),
                jointStiffness: Double.random(in: 1...10),
                fatigueLevel: Double.random(in: 1...10),
                swellingLevel: Double.random(in: 0...10),
                moodScore: Double.random(in: 1...10)
            )
            
            let envPoint = EnvironmentalDataPoint(
                timestamp: date,
                temperature: Double.random(in: -10...35),
                humidity: Double.random(in: 20...90),
                barometricPressure: Double.random(in: 980...1040),
                uvIndex: Double.random(in: 0...11)
            )
            
            let behaviorPoint = BehaviorDataPoint(
                timestamp: date,
                sleepQuality: Double.random(in: 1...10),
                stressLevel: Double.random(in: 1...10),
                medicationAdherence: Double.random(in: 0.5...1.0),
                exerciseMinutes: Int.random(in: 0...120)
            )
            
            healthData.append(healthPoint)
            environmentalData.append(envPoint)
            behaviorData.append(behaviorPoint)
        }
    }
}

// MARK: - Supporting Classes

class PatternRecognizer {
    
    func analyzePatterns(
        healthData: [HealthDataPoint],
        environmentalData: [EnvironmentalDataPoint],
        behaviorData: [BehaviorDataPoint]
    ) async -> PatternAnalysisResult {
        
        let painTrend = calculateTrend(values: healthData.map { $0.painLevel })
        let stiffnessTrend = calculateTrend(values: healthData.map { $0.jointStiffness })
        let fatigueTrend = calculateTrend(values: healthData.map { $0.fatigueLevel })
        let sleepQualityTrend = calculateTrend(values: behaviorData.map { $0.sleepQuality })
        let stressTrend = calculateTrend(values: behaviorData.map { $0.stressLevel })
        
        let weatherCorrelation = calculateWeatherCorrelation(
            painData: healthData.map { $0.painLevel },
            pressureData: environmentalData.map { $0.barometricPressure }
        )
        
        let medicationAdherence = behaviorData.map { $0.medicationAdherence }.reduce(0, +) / Double(behaviorData.count)
        let activityLevel = Double(behaviorData.map { $0.exerciseMinutes }.reduce(0, +)) / Double(behaviorData.count) / 12.0 // Normalize to 0-10
        
        return PatternAnalysisResult(
            painTrend: painTrend,
            stiffnessTrend: stiffnessTrend,
            fatigueTrend: fatigueTrend,
            sleepQualityTrend: sleepQualityTrend,
            stressTrend: stressTrend,
            weatherCorrelation: weatherCorrelation,
            medicationAdherence: medicationAdherence,
            activityLevel: activityLevel
        )
    }
    
    private func calculateTrend(values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        
        let firstHalf = Array(values.prefix(values.count / 2))
        let secondHalf = Array(values.suffix(values.count / 2))
        
        let firstAvg = firstHalf.reduce(0, +) / Double(firstHalf.count)
        let secondAvg = secondHalf.reduce(0, +) / Double(secondHalf.count)
        
        return secondAvg - firstAvg
    }
    
    private func calculateWeatherCorrelation(painData: [Double], pressureData: [Double]) -> Double {
        guard painData.count == pressureData.count && painData.count > 1 else { return 0.0 }
        
        let n = Double(painData.count)
        let sumPain = painData.reduce(0, +)
        let sumPressure = pressureData.reduce(0, +)
        let sumPainPressure = zip(painData, pressureData).map(*).reduce(0, +)
        let sumPain2 = painData.map { $0 * $0 }.reduce(0, +)
        let sumPressure2 = pressureData.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumPainPressure - sumPain * sumPressure
        let denominator = sqrt((n * sumPain2 - sumPain * sumPain) * (n * sumPressure2 - sumPressure * sumPressure))
        
        guard denominator != 0 else { return 0.0 }
        
        return abs(numerator / denominator) // Return absolute correlation
    }
}

class RiskCalculator {
    
    func calculateRisk(
        patterns: PatternAnalysisResult,
        flareHistory: [FlareEvent]
    ) async -> RiskAssessmentResult {
        
        var riskScore = 0.0
        var factors: [RiskFactor] = []
        
        // Pain trend contribution
        if patterns.painTrend > 1.0 {
            riskScore += 0.3 * (patterns.painTrend / 5.0)
            factors.append(RiskFactor(name: "Increasing Pain", impact: .high, description: "Pain trending upward", value: patterns.painTrend))
        }
        
        // Stress contribution
        if patterns.stressTrend > 1.0 {
            riskScore += 0.2 * (patterns.stressTrend / 5.0)
            factors.append(RiskFactor(name: "Elevated Stress", impact: .moderate, description: "Stress levels increasing", value: patterns.stressTrend))
        }
        
        // Sleep quality contribution
        if patterns.sleepQualityTrend < -0.5 {
            riskScore += 0.15 * abs(patterns.sleepQualityTrend / 5.0)
            factors.append(RiskFactor(name: "Poor Sleep", impact: .moderate, description: "Sleep quality declining", value: abs(patterns.sleepQualityTrend)))
        }
        
        // Medication adherence contribution
        if patterns.medicationAdherence < 0.8 {
            riskScore += 0.25 * (1.0 - patterns.medicationAdherence)
            factors.append(RiskFactor(name: "Poor Adherence", impact: .high, description: "Low medication adherence", value: 1.0 - patterns.medicationAdherence))
        }
        
        // Weather sensitivity contribution
        if patterns.weatherCorrelation > 0.5 {
            riskScore += 0.1 * patterns.weatherCorrelation
            factors.append(RiskFactor(name: "Weather Sensitivity", impact: .low, description: "Weather correlation detected", value: patterns.weatherCorrelation))
        }
        
        // Historical flare frequency
        let recentFlares = flareHistory.filter { $0.startDate > Date().addingTimeInterval(-30 * 24 * 3600) }
        if recentFlares.count > 1 {
            riskScore += 0.2
            factors.append(RiskFactor(name: "Recent Flares", impact: .high, description: "Multiple recent flares", value: Double(recentFlares.count)))
        }
        
        riskScore = min(1.0, riskScore)
        
        let predictedDays = Int(7 * (1 - riskScore))
        
        return RiskAssessmentResult(
            score: riskScore,
            predictedDays: predictedDays,
            factors: factors
        )
    }
}

// MARK: - Supporting Types

struct HealthDataPoint {
    let timestamp: Date
    let painLevel: Double
    let jointStiffness: Double
    let fatigueLevel: Double
    let swellingLevel: Double
    let moodScore: Double
}

struct EnvironmentalDataPoint {
    let timestamp: Date
    let temperature: Double
    let humidity: Double
    let barometricPressure: Double
    let uvIndex: Double
}

struct BehaviorDataPoint {
    let timestamp: Date
    let sleepQuality: Double
    let stressLevel: Double
    let medicationAdherence: Double
    let exerciseMinutes: Int
}

struct FlareEvent {
    let id: String
    let startDate: Date
    let endDate: Date?
    let severity: FlareSeverity
    let triggers: [String]
    let symptoms: [String]
}

enum FlareSeverity: String, CaseIterable {
    case mild = "Mild"
    case moderate = "Moderate"
    case severe = "Severe"
}

enum FlareRiskLevel: String, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
}

struct EarlyWarning {
    let id: String
    let type: WarningType
    let severity: WarningSeverity
    let message: String
    let timestamp: Date
    let actionItems: [String]
}

enum WarningType: String, CaseIterable {
    case highRisk = "High Risk"
    case increasingPain = "Increasing Pain"
    case poorSleep = "Poor Sleep"
    case highStress = "High Stress"
    case weatherAlert = "Weather Alert"
    case medicationMissed = "Medication Missed"
}

enum WarningSeverity: String, CaseIterable {
    case info = "Info"
    case warning = "Warning"
    case critical = "Critical"
}

struct RiskFactor {
    let name: String
    let impact: RiskImpact
    let description: String
    let value: Double
}

enum RiskImpact: String, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    
    var rawValue: Int {
        switch self {
        case .low: return 1
        case .moderate: return 2
        case .high: return 3
        }
    }
}

struct ProtectiveFactor {
    let name: String
    let benefit: ProtectiveBenefit
    let description: String
    let value: Double
}

enum ProtectiveBenefit: String, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
}

struct PatternAnalysisResult {
    let painTrend: Double
    let stiffnessTrend: Double
    let fatigueTrend: Double
    let sleepQualityTrend: Double
    let stressTrend: Double
    let weatherCorrelation: Double
    let medicationAdherence: Double
    let activityLevel: Double
}

struct RiskAssessmentResult {
    let score: Double
    let predictedDays: Int
    let factors: [RiskFactor]
}

struct MLPredictionResult {
    let riskScore: Double
    let confidence: Double
    let predictedDays: Int
}

struct CombinedRiskResult {
    let score: Double
    let level: FlareRiskLevel
    let predictedDate: Date?
    let factors: [RiskFactor]
}

struct DetailedRiskAssessment {
    let overallRisk: Double
    let riskLevel: FlareRiskLevel
    let riskFactors: [RiskFactor]
    let protectiveFactors: [ProtectiveFactor]
    let confidence: Double
    let recommendations: [String]
}