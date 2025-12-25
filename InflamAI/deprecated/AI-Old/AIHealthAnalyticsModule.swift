//
//  AIHealthAnalyticsModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import SwiftUI
import CoreML
import HealthKit
import Charts
import Combine
import CreateML
import TabularData
import NaturalLanguage

// MARK: - AI Models

struct HealthPredictionModel {
    let id = UUID()
    let name: String
    let version: String
    let accuracy: Double
    let lastTrained: Date
    let modelType: ModelType
    let inputFeatures: [String]
    let outputClasses: [String]
    
    enum ModelType: String, CaseIterable {
        case flareUpPrediction = "flare_up_prediction"
        case painIntensityForecast = "pain_intensity_forecast"
        case medicationEffectiveness = "medication_effectiveness"
        case symptomClassification = "symptom_classification"
        case treatmentRecommendation = "treatment_recommendation"
        case riskAssessment = "risk_assessment"
        case progressPrediction = "progress_prediction"
        
        var displayName: String {
            switch self {
            case .flareUpPrediction: return "Flare-up Prediction"
            case .painIntensityForecast: return "Pain Intensity Forecast"
            case .medicationEffectiveness: return "Medication Effectiveness"
            case .symptomClassification: return "Symptom Classification"
            case .treatmentRecommendation: return "Treatment Recommendation"
            case .riskAssessment: return "Risk Assessment"
            case .progressPrediction: return "Progress Prediction"
            }
        }
    }
}

struct PredictionResult {
    let id = UUID()
    let modelType: HealthPredictionModel.ModelType
    let prediction: String
    let confidence: Double
    let probability: [String: Double]
    let features: [String: Any]
    let timestamp: Date
    let timeframe: PredictionTimeframe
    let recommendations: [AIRecommendation]
    
    enum PredictionTimeframe: String, CaseIterable {
        case immediate = "immediate"
        case shortTerm = "short_term"
        case mediumTerm = "medium_term"
        case longTerm = "long_term"
        
        var displayName: String {
            switch self {
            case .immediate: return "Next 24 hours"
            case .shortTerm: return "Next 3 days"
            case .mediumTerm: return "Next week"
            case .longTerm: return "Next month"
            }
        }
        
        var hours: Int {
            switch self {
            case .immediate: return 24
            case .shortTerm: return 72
            case .mediumTerm: return 168
            case .longTerm: return 720
            }
        }
    }
}

struct AIRecommendation {
    let id = UUID()
    let title: String
    let description: String
    let category: RecommendationCategory
    let priority: RecommendationPriority
    let confidence: Double
    let evidence: [String]
    let actionItems: [ActionItem]
    let expectedOutcome: String
    
    enum RecommendationCategory: String, CaseIterable {
        case medication = "medication"
        case lifestyle = "lifestyle"
        case exercise = "exercise"
        case diet = "diet"
        case stress = "stress"
        case sleep = "sleep"
        case medical = "medical"
        case prevention = "prevention"
        
        var displayName: String {
            switch self {
            case .medication: return "Medication"
            case .lifestyle: return "Lifestyle"
            case .exercise: return "Exercise"
            case .diet: return "Diet"
            case .stress: return "Stress Management"
            case .sleep: return "Sleep"
            case .medical: return "Medical Care"
            case .prevention: return "Prevention"
            }
        }
        
        var icon: String {
            switch self {
            case .medication: return "pills.fill"
            case .lifestyle: return "heart.fill"
            case .exercise: return "figure.walk"
            case .diet: return "leaf.fill"
            case .stress: return "brain.head.profile"
            case .sleep: return "bed.double.fill"
            case .medical: return "stethoscope"
            case .prevention: return "shield.fill"
            }
        }
    }
    
    enum RecommendationPriority: String, CaseIterable {
        case critical = "critical"
        case high = "high"
        case medium = "medium"
        case low = "low"
        
        var displayName: String {
            switch self {
            case .critical: return "Critical"
            case .high: return "High"
            case .medium: return "Medium"
            case .low: return "Low"
            }
        }
        
        var color: Color {
            switch self {
            case .critical: return .red
            case .high: return .orange
            case .medium: return .yellow
            case .low: return .green
            }
        }
    }
}

struct ActionItem {
    let id = UUID()
    let title: String
    let description: String
    let dueDate: Date?
    let isCompleted: Bool
    let category: String
    let estimatedDuration: TimeInterval
}

// MARK: - Analytics Models

struct HealthAnalytics {
    let id = UUID()
    let userId: String
    let generatedDate: Date
    let timeRange: AnalyticsTimeRange
    let insights: [HealthInsight]
    let trends: [HealthTrend]
    let correlations: [HealthCorrelation]
    let predictions: [PredictionResult]
    let riskFactors: [RiskFactor]
    let progressMetrics: [ProgressMetric]
    
    enum AnalyticsTimeRange: String, CaseIterable {
        case week = "week"
        case month = "month"
        case quarter = "quarter"
        case year = "year"
        case all = "all"
        
        var displayName: String {
            switch self {
            case .week: return "Past Week"
            case .month: return "Past Month"
            case .quarter: return "Past 3 Months"
            case .year: return "Past Year"
            case .all: return "All Time"
            }
        }
        
        var days: Int {
            switch self {
            case .week: return 7
            case .month: return 30
            case .quarter: return 90
            case .year: return 365
            case .all: return Int.max
            }
        }
    }
}

struct HealthInsight {
    let id = UUID()
    let title: String
    let description: String
    let category: InsightCategory
    let importance: InsightImportance
    let confidence: Double
    let dataPoints: [String]
    let visualizationType: VisualizationType
    let actionable: Bool
    
    enum InsightCategory: String, CaseIterable {
        case symptom = "symptom"
        case medication = "medication"
        case lifestyle = "lifestyle"
        case environmental = "environmental"
        case behavioral = "behavioral"
        case physiological = "physiological"
        
        var displayName: String {
            switch self {
            case .symptom: return "Symptoms"
            case .medication: return "Medications"
            case .lifestyle: return "Lifestyle"
            case .environmental: return "Environmental"
            case .behavioral: return "Behavioral"
            case .physiological: return "Physiological"
            }
        }
    }
    
    enum InsightImportance: String, CaseIterable {
        case critical = "critical"
        case high = "high"
        case medium = "medium"
        case low = "low"
        
        var color: Color {
            switch self {
            case .critical: return .red
            case .high: return .orange
            case .medium: return .blue
            case .low: return .gray
            }
        }
    }
    
    enum VisualizationType: String, CaseIterable {
        case lineChart = "line_chart"
        case barChart = "bar_chart"
        case scatterPlot = "scatter_plot"
        case heatmap = "heatmap"
        case pieChart = "pie_chart"
        case timeline = "timeline"
        case correlation = "correlation"
        
        var displayName: String {
            switch self {
            case .lineChart: return "Line Chart"
            case .barChart: return "Bar Chart"
            case .scatterPlot: return "Scatter Plot"
            case .heatmap: return "Heat Map"
            case .pieChart: return "Pie Chart"
            case .timeline: return "Timeline"
            case .correlation: return "Correlation Matrix"
            }
        }
    }
}

struct HealthTrend {
    let id = UUID()
    let metric: String
    let direction: TrendDirection
    let magnitude: Double
    let significance: Double
    let timeframe: String
    let dataPoints: [TrendDataPoint]
    
    enum TrendDirection: String, CaseIterable {
        case increasing = "increasing"
        case decreasing = "decreasing"
        case stable = "stable"
        case volatile = "volatile"
        
        var displayName: String {
            switch self {
            case .increasing: return "Increasing"
            case .decreasing: return "Decreasing"
            case .stable: return "Stable"
            case .volatile: return "Volatile"
            }
        }
        
        var icon: String {
            switch self {
            case .increasing: return "arrow.up.right"
            case .decreasing: return "arrow.down.right"
            case .stable: return "arrow.right"
            case .volatile: return "waveform"
            }
        }
        
        var color: Color {
            switch self {
            case .increasing: return .red
            case .decreasing: return .green
            case .stable: return .blue
            case .volatile: return .orange
            }
        }
    }
}

struct TrendDataPoint {
    let date: Date
    let value: Double
    let confidence: Double
}

struct HealthCorrelation {
    let id = UUID()
    let variable1: String
    let variable2: String
    let correlation: Double
    let significance: Double
    let strength: CorrelationStrength
    let type: CorrelationType
    let description: String
    
    enum CorrelationStrength: String, CaseIterable {
        case veryWeak = "very_weak"
        case weak = "weak"
        case moderate = "moderate"
        case strong = "strong"
        case veryStrong = "very_strong"
        
        var displayName: String {
            switch self {
            case .veryWeak: return "Very Weak"
            case .weak: return "Weak"
            case .moderate: return "Moderate"
            case .strong: return "Strong"
            case .veryStrong: return "Very Strong"
            }
        }
        
        static func from(correlation: Double) -> CorrelationStrength {
            let abs = Swift.abs(correlation)
            if abs < 0.2 { return .veryWeak }
            else if abs < 0.4 { return .weak }
            else if abs < 0.6 { return .moderate }
            else if abs < 0.8 { return .strong }
            else { return .veryStrong }
        }
    }
    
    enum CorrelationType: String, CaseIterable {
        case positive = "positive"
        case negative = "negative"
        
        var displayName: String {
            switch self {
            case .positive: return "Positive"
            case .negative: return "Negative"
            }
        }
        
        var color: Color {
            switch self {
            case .positive: return .green
            case .negative: return .red
            }
        }
    }
}

struct RiskFactor {
    let id = UUID()
    let name: String
    let description: String
    let riskLevel: RiskLevel
    let probability: Double
    let impact: RiskImpact
    let category: RiskCategory
    let mitigationStrategies: [String]
    let timeframe: String
    
    enum RiskLevel: String, CaseIterable {
        case low = "low"
        case moderate = "moderate"
        case high = "high"
        case critical = "critical"
        
        var displayName: String {
            switch self {
            case .low: return "Low"
            case .moderate: return "Moderate"
            case .high: return "High"
            case .critical: return "Critical"
            }
        }
        
        var color: Color {
            switch self {
            case .low: return .green
            case .moderate: return .yellow
            case .high: return .orange
            case .critical: return .red
            }
        }
    }
    
    enum RiskImpact: String, CaseIterable {
        case minimal = "minimal"
        case minor = "minor"
        case moderate = "moderate"
        case major = "major"
        case severe = "severe"
        
        var displayName: String {
            switch self {
            case .minimal: return "Minimal"
            case .minor: return "Minor"
            case .moderate: return "Moderate"
            case .major: return "Major"
            case .severe: return "Severe"
            }
        }
    }
    
    enum RiskCategory: String, CaseIterable {
        case disease = "disease"
        case medication = "medication"
        case lifestyle = "lifestyle"
        case environmental = "environmental"
        case genetic = "genetic"
        case behavioral = "behavioral"
        
        var displayName: String {
            switch self {
            case .disease: return "Disease Progression"
            case .medication: return "Medication"
            case .lifestyle: return "Lifestyle"
            case .environmental: return "Environmental"
            case .genetic: return "Genetic"
            case .behavioral: return "Behavioral"
            }
        }
    }
}

struct ProgressMetric {
    let id = UUID()
    let name: String
    let currentValue: Double
    let previousValue: Double
    let targetValue: Double?
    let unit: String
    let changePercentage: Double
    let trend: HealthTrend.TrendDirection
    let category: MetricCategory
    
    enum MetricCategory: String, CaseIterable {
        case symptoms = "symptoms"
        case functionality = "functionality"
        case quality = "quality"
        case medication = "medication"
        case exercise = "exercise"
        case sleep = "sleep"
        case mood = "mood"
        
        var displayName: String {
            switch self {
            case .symptoms: return "Symptoms"
            case .functionality: return "Functionality"
            case .quality: return "Quality of Life"
            case .medication: return "Medication Adherence"
            case .exercise: return "Exercise"
            case .sleep: return "Sleep Quality"
            case .mood: return "Mood"
            }
        }
    }
}

// MARK: - AI Health Analytics Manager

@MainActor
class AIHealthAnalyticsManager: ObservableObject {
    @Published var isLoading = false
    @Published var currentAnalytics: HealthAnalytics?
    @Published var predictions: [PredictionResult] = []
    @Published var insights: [HealthInsight] = []
    @Published var trends: [HealthTrend] = []
    @Published var correlations: [HealthCorrelation] = []
    @Published var riskFactors: [RiskFactor] = []
    @Published var recommendations: [AIRecommendation] = []
    @Published var models: [HealthPredictionModel] = []
    @Published var selectedTimeRange: HealthAnalytics.AnalyticsTimeRange = .month
    @Published var lastAnalysisDate: Date?
    @Published var analysisProgress: Double = 0.0
    
    private let healthStore = HKHealthStore()
    private let mlModelManager = MLModelManager()
    private let dataProcessor = HealthDataProcessor()
    private let insightGenerator = InsightGenerator()
    private let predictionEngine = PredictionEngine()
    private let correlationAnalyzer = CorrelationAnalyzer()
    private let riskAssessment = RiskAssessmentEngine()
    private let recommendationEngine = RecommendationEngine()
    private let nlProcessor = NLProcessor()
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        setupHealthKit()
        loadModels()
        setupPeriodicAnalysis()
    }
    
    // MARK: - Setup Methods
    
    private func setupHealthKit() {
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
            HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!
        ]
        
        healthStore.requestAuthorization(toShare: [], read: typesToRead) { success, error in
            if let error = error {
                print("HealthKit authorization failed: \(error.localizedDescription)")
            }
        }
    }
    
    private func loadModels() {
        Task {
            do {
                models = try await mlModelManager.loadAllModels()
            } catch {
                print("Failed to load ML models: \(error)")
            }
        }
    }
    
    private func setupPeriodicAnalysis() {
        Timer.publish(every: 3600, on: .main, in: .common) // Every hour
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.performIncrementalAnalysis()
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Public API
    
    func generateComprehensiveAnalysis(timeRange: HealthAnalytics.AnalyticsTimeRange = .month) async throws {
        isLoading = true
        analysisProgress = 0.0
        selectedTimeRange = timeRange
        
        do {
            // Step 1: Collect and process health data
            analysisProgress = 0.1
            let healthData = try await dataProcessor.collectHealthData(timeRange: timeRange)
            
            // Step 2: Generate insights
            analysisProgress = 0.3
            insights = try await insightGenerator.generateInsights(from: healthData)
            
            // Step 3: Analyze trends
            analysisProgress = 0.5
            trends = try await dataProcessor.analyzeTrends(in: healthData)
            
            // Step 4: Find correlations
            analysisProgress = 0.6
            correlations = try await correlationAnalyzer.findCorrelations(in: healthData)
            
            // Step 5: Generate predictions
            analysisProgress = 0.7
            predictions = try await predictionEngine.generatePredictions(from: healthData)
            
            // Step 6: Assess risks
            analysisProgress = 0.8
            riskFactors = try await riskAssessment.assessRisks(from: healthData, insights: insights)
            
            // Step 7: Generate recommendations
            analysisProgress = 0.9
            recommendations = try await recommendationEngine.generateRecommendations(
                from: healthData,
                insights: insights,
                predictions: predictions,
                risks: riskFactors
            )
            
            // Step 8: Create comprehensive analytics
            analysisProgress = 1.0
            currentAnalytics = HealthAnalytics(
                userId: getCurrentUserId(),
                generatedDate: Date(),
                timeRange: timeRange,
                insights: insights,
                trends: trends,
                correlations: correlations,
                predictions: predictions,
                riskFactors: riskFactors,
                progressMetrics: calculateProgressMetrics(from: healthData)
            )
            
            lastAnalysisDate = Date()
            
        } catch {
            print("Analysis failed: \(error)")
            throw error
        }
        
        isLoading = false
    }
    
    func generatePrediction(for modelType: HealthPredictionModel.ModelType, timeframe: PredictionResult.PredictionTimeframe) async throws -> PredictionResult {
        guard let model = models.first(where: { $0.modelType == modelType }) else {
            throw AIAnalyticsError.modelNotFound(modelType.rawValue)
        }
        
        let healthData = try await dataProcessor.collectRecentHealthData()
        return try await predictionEngine.generatePrediction(using: model, data: healthData, timeframe: timeframe)
    }
    
    func getInsights(for category: HealthInsight.InsightCategory) -> [HealthInsight] {
        return insights.filter { $0.category == category }
    }
    
    func getRecommendations(for category: AIRecommendation.RecommendationCategory) -> [AIRecommendation] {
        return recommendations.filter { $0.category == category }
    }
    
    func getRiskFactors(above level: RiskFactor.RiskLevel) -> [RiskFactor] {
        let levelValue = getRiskLevelValue(level)
        return riskFactors.filter { getRiskLevelValue($0.riskLevel) >= levelValue }
    }
    
    func exportAnalytics() async throws -> Data {
        guard let analytics = currentAnalytics else {
            throw AIAnalyticsError.noAnalyticsAvailable
        }
        
        return try JSONEncoder().encode(analytics)
    }
    
    func trainCustomModel(for modelType: HealthPredictionModel.ModelType, with data: [String: Any]) async throws {
        try await mlModelManager.trainModel(type: modelType, data: data)
        models = try await mlModelManager.loadAllModels()
    }
    
    // MARK: - Private Methods
    
    private func performIncrementalAnalysis() async {
        guard !isLoading else { return }
        
        do {
            let recentData = try await dataProcessor.collectRecentHealthData()
            let newInsights = try await insightGenerator.generateIncrementalInsights(from: recentData)
            
            await MainActor.run {
                insights.append(contentsOf: newInsights)
                insights = Array(insights.suffix(100)) // Keep only recent insights
            }
        } catch {
            print("Incremental analysis failed: \(error)")
        }
    }
    
    private func calculateProgressMetrics(from healthData: HealthDataCollection) -> [ProgressMetric] {
        // Implementation would calculate various progress metrics
        return []
    }
    
    private func getCurrentUserId() -> String {
        return "current_user_id" // Implementation would get actual user ID
    }
    
    private func getRiskLevelValue(_ level: RiskFactor.RiskLevel) -> Int {
        switch level {
        case .low: return 1
        case .moderate: return 2
        case .high: return 3
        case .critical: return 4
        }
    }
}

// MARK: - Supporting Classes

class MLModelManager {
    private var loadedModels: [HealthPredictionModel.ModelType: MLModel] = [:]
    
    func loadAllModels() async throws -> [HealthPredictionModel] {
        var models: [HealthPredictionModel] = []
        
        for modelType in HealthPredictionModel.ModelType.allCases {
            if let model = try? await loadModel(type: modelType) {
                models.append(model)
            }
        }
        
        return models
    }
    
    func loadModel(type: HealthPredictionModel.ModelType) async throws -> HealthPredictionModel {
        // Implementation would load actual ML model
        return HealthPredictionModel(
            name: type.displayName,
            version: "1.0",
            accuracy: 0.85,
            lastTrained: Date(),
            modelType: type,
            inputFeatures: getInputFeatures(for: type),
            outputClasses: getOutputClasses(for: type)
        )
    }
    
    func trainModel(type: HealthPredictionModel.ModelType, data: [String: Any]) async throws {
        // Implementation would train ML model using CreateML
    }
    
    private func getInputFeatures(for type: HealthPredictionModel.ModelType) -> [String] {
        switch type {
        case .flareUpPrediction:
            return ["pain_level", "fatigue", "stress", "sleep_quality", "weather", "medication_adherence"]
        case .painIntensityForecast:
            return ["current_pain", "activity_level", "medication_timing", "sleep_hours", "mood"]
        case .medicationEffectiveness:
            return ["medication_type", "dosage", "timing", "adherence", "side_effects", "symptoms"]
        case .symptomClassification:
            return ["description", "severity", "duration", "location", "triggers"]
        case .treatmentRecommendation:
            return ["symptoms", "medical_history", "current_treatments", "preferences", "lifestyle"]
        case .riskAssessment:
            return ["age", "disease_duration", "severity", "comorbidities", "lifestyle_factors"]
        case .progressPrediction:
            return ["baseline_metrics", "treatment_response", "adherence", "lifestyle_changes"]
        }
    }
    
    private func getOutputClasses(for type: HealthPredictionModel.ModelType) -> [String] {
        switch type {
        case .flareUpPrediction:
            return ["no_flare", "mild_flare", "moderate_flare", "severe_flare"]
        case .painIntensityForecast:
            return ["low", "moderate", "high", "severe"]
        case .medicationEffectiveness:
            return ["highly_effective", "moderately_effective", "minimally_effective", "ineffective"]
        case .symptomClassification:
            return ["joint_pain", "fatigue", "stiffness", "swelling", "other"]
        case .treatmentRecommendation:
            return ["medication_adjustment", "lifestyle_change", "therapy", "specialist_referral"]
        case .riskAssessment:
            return ["low_risk", "moderate_risk", "high_risk", "critical_risk"]
        case .progressPrediction:
            return ["excellent", "good", "fair", "poor"]
        }
    }
}

class HealthDataProcessor {
    func collectHealthData(timeRange: HealthAnalytics.AnalyticsTimeRange) async throws -> HealthDataCollection {
        // Implementation would collect comprehensive health data
        return HealthDataCollection()
    }
    
    func collectRecentHealthData() async throws -> HealthDataCollection {
        // Implementation would collect recent health data
        return HealthDataCollection()
    }
    
    func analyzeTrends(in data: HealthDataCollection) async throws -> [HealthTrend] {
        // Implementation would analyze trends in health data
        return []
    }
}

class InsightGenerator {
    func generateInsights(from data: HealthDataCollection) async throws -> [HealthInsight] {
        // Implementation would generate health insights
        return []
    }
    
    func generateIncrementalInsights(from data: HealthDataCollection) async throws -> [HealthInsight] {
        // Implementation would generate incremental insights
        return []
    }
}

class PredictionEngine {
    func generatePredictions(from data: HealthDataCollection) async throws -> [PredictionResult] {
        // Implementation would generate predictions
        return []
    }
    
    func generatePrediction(using model: HealthPredictionModel, data: HealthDataCollection, timeframe: PredictionResult.PredictionTimeframe) async throws -> PredictionResult {
        // Implementation would generate specific prediction
        return PredictionResult(
            modelType: model.modelType,
            prediction: "Sample prediction",
            confidence: 0.85,
            probability: [:],
            features: [:],
            timestamp: Date(),
            timeframe: timeframe,
            recommendations: []
        )
    }
}

class CorrelationAnalyzer {
    func findCorrelations(in data: HealthDataCollection) async throws -> [HealthCorrelation] {
        // Implementation would find correlations in health data
        return []
    }
}

class RiskAssessmentEngine {
    func assessRisks(from data: HealthDataCollection, insights: [HealthInsight]) async throws -> [RiskFactor] {
        // Implementation would assess health risks
        return []
    }
}

class RecommendationEngine {
    func generateRecommendations(from data: HealthDataCollection, insights: [HealthInsight], predictions: [PredictionResult], risks: [RiskFactor]) async throws -> [AIRecommendation] {
        // Implementation would generate AI recommendations
        return []
    }
}

class NLProcessor {
    private let tagger = NLTagger(tagSchemes: [.sentimentScore, .language])
    
    func analyzeSentiment(text: String) -> Double {
        tagger.string = text
        let (sentiment, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
        return Double(sentiment?.rawValue ?? "0") ?? 0.0
    }
    
    func extractKeywords(from text: String) -> [String] {
        // Implementation would extract keywords
        return []
    }
}

struct HealthDataCollection {
    let symptoms: [SymptomEntry] = []
    let medications: [MedicationEntry] = []
    let vitals: [VitalSign] = []
    let activities: [ActivityEntry] = []
    let mood: [MoodEntry] = []
    let sleep: [SleepEntry] = []
    let environmental: [EnvironmentalFactor] = []
}

struct SymptomEntry {
    let date: Date
    let type: String
    let severity: Int
    let location: String?
    let notes: String?
}

struct MedicationEntry {
    let date: Date
    let name: String
    let dosage: String
    let taken: Bool
    let sideEffects: [String]
}

struct VitalSign {
    let date: Date
    let type: String
    let value: Double
    let unit: String
}

struct ActivityEntry {
    let date: Date
    let type: String
    let duration: TimeInterval
    let intensity: String
}

struct MoodEntry {
    let date: Date
    let mood: String
    let score: Int
    let notes: String?
}

struct SleepEntry {
    let date: Date
    let duration: TimeInterval
    let quality: Int
    let notes: String?
}

struct EnvironmentalFactor {
    let date: Date
    let type: String
    let value: String
    let impact: String?
}

// MARK: - Errors

enum AIAnalyticsError: Error, LocalizedError {
    case modelNotFound(String)
    case noAnalyticsAvailable
    case dataProcessingFailed
    case predictionFailed
    case insufficientData
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let model):
            return "ML model not found: \(model)"
        case .noAnalyticsAvailable:
            return "No analytics data available"
        case .dataProcessingFailed:
            return "Failed to process health data"
        case .predictionFailed:
            return "Failed to generate prediction"
        case .insufficientData:
            return "Insufficient data for analysis"
        }
    }
}