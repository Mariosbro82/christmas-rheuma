//
//  AIAnalyticsManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreML
import Combine
import HealthKit

// MARK: - AI Analytics Manager

@MainActor
class AIAnalyticsManager: ObservableObject {
    // MARK: - Published Properties
    @Published var isAnalyzing = false
    @Published var lastAnalysisDate: Date?
    @Published var painPredictions: [PainPrediction] = []
    @Published var correlationInsights: [CorrelationInsight] = []
    @Published var flareRiskLevel: FlareRiskLevel = .low
    @Published var medicationOptimizations: [MedicationOptimization] = []
    @Published var personalizedRecommendations: [PersonalizedRecommendation] = []
    
    // MARK: - Private Properties
    private let dataManager: DataManager
    private let healthKitManager: HealthKitManager
    private let weatherService: WeatherService
    private let mlModelManager: MLModelManager
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    init(dataManager: DataManager, healthKitManager: HealthKitManager) {
        self.dataManager = dataManager
        self.healthKitManager = healthKitManager
        self.weatherService = WeatherService()
        self.mlModelManager = MLModelManager()
        
        setupAnalysisScheduler()
    }
    
    // MARK: - Public Methods
    
    func performComprehensiveAnalysis() async {
        isAnalyzing = true
        defer { isAnalyzing = false }
        
        do {
            // Gather all necessary data
            let painData = await dataManager.getAllPainEntries()
            let medicationData = await dataManager.getAllMedicationEntries()
            let activityData = await healthKitManager.getActivityData()
            let sleepData = await healthKitManager.getSleepData()
            let weatherData = await weatherService.getHistoricalWeatherData()
            
            // Perform various analyses
            await performPainPredictionAnalysis(painData: painData, contextData: AnalysisContext(
                medications: medicationData,
                activities: activityData,
                sleep: sleepData,
                weather: weatherData
            ))
            
            await performCorrelationAnalysis(painData: painData, contextData: AnalysisContext(
                medications: medicationData,
                activities: activityData,
                sleep: sleepData,
                weather: weatherData
            ))
            
            await assessFlareRisk(painData: painData)
            await generateMedicationOptimizations(painData: painData, medicationData: medicationData)
            await generatePersonalizedRecommendations()
            
            lastAnalysisDate = Date()
            
        } catch {
            print("Analysis failed: \(error)")
        }
    }
    
    func predictPainForDate(_ date: Date) async -> PainPrediction? {
        do {
            let features = await gatherPredictionFeatures(for: date)
            return try await mlModelManager.predictPain(features: features)
        } catch {
            print("Pain prediction failed: \(error)")
            return nil
        }
    }
    
    func analyzeSymptomCorrelations() async -> [CorrelationInsight] {
        let painData = await dataManager.getAllPainEntries()
        let medicationData = await dataManager.getAllMedicationEntries()
        
        var insights: [CorrelationInsight] = []
        
        // Analyze pain-medication correlations
        let medicationCorrelations = analyzePainMedicationCorrelations(painData: painData, medicationData: medicationData)
        insights.append(contentsOf: medicationCorrelations)
        
        // Analyze weather correlations
        if let weatherCorrelations = await analyzeWeatherCorrelations(painData: painData) {
            insights.append(contentsOf: weatherCorrelations)
        }
        
        // Analyze activity correlations
        if let activityCorrelations = await analyzeActivityCorrelations(painData: painData) {
            insights.append(contentsOf: activityCorrelations)
        }
        
        return insights
    }
    
    func detectFlarePatterns() async -> [FlarePattern] {
        let painData = await dataManager.getAllPainEntries()
        return await mlModelManager.detectFlarePatterns(from: painData)
    }
    
    func optimizeMedicationSchedule() async -> [MedicationOptimization] {
        let painData = await dataManager.getAllPainEntries()
        let medicationData = await dataManager.getAllMedicationEntries()
        
        return await mlModelManager.optimizeMedicationSchedule(
            painData: painData,
            medicationData: medicationData
        )
    }
    
    // MARK: - Private Methods
    
    private func setupAnalysisScheduler() {
        // Schedule daily analysis
        Timer.publish(every: 24 * 60 * 60, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.performComprehensiveAnalysis()
                }
            }
            .store(in: &cancellables)
    }
    
    private func performPainPredictionAnalysis(painData: [PainEntry], contextData: AnalysisContext) async {
        let predictions = await mlModelManager.generatePainPredictions(
            historicalPain: painData,
            context: contextData
        )
        
        await MainActor.run {
            self.painPredictions = predictions
        }
    }
    
    private func performCorrelationAnalysis(painData: [PainEntry], contextData: AnalysisContext) async {
        let insights = await mlModelManager.analyzeCorrelations(
            painData: painData,
            context: contextData
        )
        
        await MainActor.run {
            self.correlationInsights = insights
        }
    }
    
    private func assessFlareRisk(painData: [PainEntry]) async {
        let riskLevel = await mlModelManager.assessFlareRisk(from: painData)
        
        await MainActor.run {
            self.flareRiskLevel = riskLevel
        }
    }
    
    private func generateMedicationOptimizations(painData: [PainEntry], medicationData: [MedicationEntry]) async {
        let optimizations = await mlModelManager.generateMedicationOptimizations(
            painData: painData,
            medicationData: medicationData
        )
        
        await MainActor.run {
            self.medicationOptimizations = optimizations
        }
    }
    
    private func generatePersonalizedRecommendations() async {
        let recommendations = await mlModelManager.generatePersonalizedRecommendations(
            userProfile: await dataManager.getUserProfile(),
            recentData: await gatherRecentUserData()
        )
        
        await MainActor.run {
            self.personalizedRecommendations = recommendations
        }
    }
    
    private func gatherPredictionFeatures(for date: Date) async -> PredictionFeatures {
        let recentPain = await dataManager.getPainEntries(from: Calendar.current.date(byAdding: .day, value: -7, to: date)!, to: date)
        let recentMedications = await dataManager.getMedicationEntries(from: Calendar.current.date(byAdding: .day, value: -7, to: date)!, to: date)
        let weather = await weatherService.getWeatherForecast(for: date)
        let sleepData = await healthKitManager.getSleepData(for: date)
        
        return PredictionFeatures(
            date: date,
            recentPainLevels: recentPain.map { $0.painLevel },
            recentMedications: recentMedications,
            weatherConditions: weather,
            sleepQuality: sleepData?.quality ?? 0.0,  // FIXED: 0 = no data, not fake 0.5
            dayOfWeek: Calendar.current.component(.weekday, from: date),
            timeOfYear: Calendar.current.dayOfYear(for: date) ?? 1
        )
    }
    
    private func analyzePainMedicationCorrelations(painData: [PainEntry], medicationData: [MedicationEntry]) -> [CorrelationInsight] {
        var insights: [CorrelationInsight] = []
        
        // Group medications by type
        let medicationsByType = Dictionary(grouping: medicationData) { $0.medicationType }
        
        for (medicationType, medications) in medicationsByType {
            let correlation = calculatePainMedicationCorrelation(painData: painData, medications: medications)
            
            if abs(correlation) > 0.3 { // Significant correlation threshold
                insights.append(CorrelationInsight(
                    id: UUID(),
                    type: .medicationEffectiveness,
                    title: "\(medicationType) Effectiveness",
                    description: correlation > 0 ? "\(medicationType) appears to reduce pain levels" : "\(medicationType) may not be effectively managing pain",
                    correlationStrength: abs(correlation),
                    confidence: 0.8,
                    actionable: true,
                    recommendation: correlation > 0 ? "Continue current \(medicationType) regimen" : "Consider discussing \(medicationType) effectiveness with your doctor"
                ))
            }
        }
        
        return insights
    }
    
    private func analyzeWeatherCorrelations(painData: [PainEntry]) async -> [CorrelationInsight]? {
        guard let weatherData = await weatherService.getHistoricalWeatherData() else {
            return nil
        }
        
        var insights: [CorrelationInsight] = []
        
        // Analyze barometric pressure correlation
        let pressureCorrelation = calculateWeatherCorrelation(
            painData: painData,
            weatherData: weatherData,
            weatherProperty: \.barometricPressure
        )
        
        if abs(pressureCorrelation) > 0.3 {
            insights.append(CorrelationInsight(
                id: UUID(),
                type: .weatherPattern,
                title: "Barometric Pressure Impact",
                description: pressureCorrelation > 0 ? "Higher barometric pressure correlates with increased pain" : "Lower barometric pressure correlates with increased pain",
                correlationStrength: abs(pressureCorrelation),
                confidence: 0.7,
                actionable: true,
                recommendation: "Monitor weather forecasts and prepare for pressure changes"
            ))
        }
        
        // Analyze humidity correlation
        let humidityCorrelation = calculateWeatherCorrelation(
            painData: painData,
            weatherData: weatherData,
            weatherProperty: \.humidity
        )
        
        if abs(humidityCorrelation) > 0.3 {
            insights.append(CorrelationInsight(
                id: UUID(),
                type: .weatherPattern,
                title: "Humidity Impact",
                description: humidityCorrelation > 0 ? "Higher humidity correlates with increased pain" : "Lower humidity correlates with increased pain",
                correlationStrength: abs(humidityCorrelation),
                confidence: 0.7,
                actionable: true,
                recommendation: "Consider using a dehumidifier/humidifier to maintain optimal humidity levels"
            ))
        }
        
        return insights
    }
    
    private func analyzeActivityCorrelations(painData: [PainEntry]) async -> [CorrelationInsight]? {
        guard let activityData = await healthKitManager.getActivityData() else {
            return nil
        }
        
        var insights: [CorrelationInsight] = []
        
        // Analyze step count correlation
        let stepCorrelation = calculateActivityCorrelation(
            painData: painData,
            activityData: activityData,
            activityProperty: \.stepCount
        )
        
        if abs(stepCorrelation) > 0.3 {
            insights.append(CorrelationInsight(
                id: UUID(),
                type: .activityPattern,
                title: "Physical Activity Impact",
                description: stepCorrelation > 0 ? "Increased activity correlates with higher pain" : "Increased activity correlates with lower pain",
                correlationStrength: abs(stepCorrelation),
                confidence: 0.8,
                actionable: true,
                recommendation: stepCorrelation > 0 ? "Consider gentle, low-impact exercises" : "Maintain regular physical activity for pain management"
            ))
        }
        
        return insights
    }
    
    private func calculatePainMedicationCorrelation(painData: [PainEntry], medications: [MedicationEntry]) -> Double {
        // Simplified correlation calculation
        // In a real implementation, this would use more sophisticated statistical methods
        
        let medicationDates = Set(medications.map { Calendar.current.startOfDay(for: $0.dateTaken) })
        let painByDate = Dictionary(grouping: painData) { Calendar.current.startOfDay(for: $0.date) }
        
        var medicatedPainLevels: [Double] = []
        var nonMedicatedPainLevels: [Double] = []
        
        for (date, painEntries) in painByDate {
            let avgPain = painEntries.map { $0.painLevel }.reduce(0, +) / Double(painEntries.count)
            
            if medicationDates.contains(date) {
                medicatedPainLevels.append(avgPain)
            } else {
                nonMedicatedPainLevels.append(avgPain)
            }
        }
        
        guard !medicatedPainLevels.isEmpty && !nonMedicatedPainLevels.isEmpty else {
            return 0.0
        }
        
        let medicatedAvg = medicatedPainLevels.reduce(0, +) / Double(medicatedPainLevels.count)
        let nonMedicatedAvg = nonMedicatedPainLevels.reduce(0, +) / Double(nonMedicatedPainLevels.count)
        
        // Return negative correlation if medication reduces pain
        return (medicatedAvg - nonMedicatedAvg) / 10.0 // Normalize to -1 to 1 range
    }
    
    private func calculateWeatherCorrelation(painData: [PainEntry], weatherData: [WeatherData], weatherProperty: KeyPath<WeatherData, Double>) -> Double {
        // Simplified correlation calculation
        let weatherByDate = Dictionary(uniqueKeysWithValues: weatherData.map { (Calendar.current.startOfDay(for: $0.date), $0) })
        
        var correlationPairs: [(pain: Double, weather: Double)] = []
        
        for painEntry in painData {
            let date = Calendar.current.startOfDay(for: painEntry.date)
            if let weather = weatherByDate[date] {
                correlationPairs.append((pain: painEntry.painLevel, weather: weather[keyPath: weatherProperty]))
            }
        }
        
        guard correlationPairs.count > 10 else { return 0.0 } // Need sufficient data
        
        return calculatePearsonCorrelation(correlationPairs)
    }
    
    private func calculateActivityCorrelation(painData: [PainEntry], activityData: [ActivityData], activityProperty: KeyPath<ActivityData, Double>) -> Double {
        let activityByDate = Dictionary(uniqueKeysWithValues: activityData.map { (Calendar.current.startOfDay(for: $0.date), $0) })
        
        var correlationPairs: [(pain: Double, activity: Double)] = []
        
        for painEntry in painData {
            let date = Calendar.current.startOfDay(for: painEntry.date)
            if let activity = activityByDate[date] {
                correlationPairs.append((pain: painEntry.painLevel, activity: activity[keyPath: activityProperty]))
            }
        }
        
        guard correlationPairs.count > 10 else { return 0.0 }
        
        return calculatePearsonCorrelation(correlationPairs)
    }
    
    private func calculatePearsonCorrelation(_ pairs: [(pain: Double, weather: Double)]) -> Double {
        let n = Double(pairs.count)
        guard n > 1 else { return 0.0 }
        
        let sumX = pairs.map { $0.pain }.reduce(0, +)
        let sumY = pairs.map { $0.weather }.reduce(0, +)
        let sumXY = pairs.map { $0.pain * $0.weather }.reduce(0, +)
        let sumX2 = pairs.map { $0.pain * $0.pain }.reduce(0, +)
        let sumY2 = pairs.map { $0.weather * $0.weather }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        guard denominator != 0 else { return 0.0 }
        
        return numerator / denominator
    }
    
    private func gatherRecentUserData() async -> RecentUserData {
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -30, to: endDate)!
        
        return RecentUserData(
            painEntries: await dataManager.getPainEntries(from: startDate, to: endDate),
            medicationEntries: await dataManager.getMedicationEntries(from: startDate, to: endDate),
            activityData: await healthKitManager.getActivityData(from: startDate, to: endDate),
            sleepData: await healthKitManager.getSleepData(from: startDate, to: endDate)
        )
    }
}

// MARK: - Supporting Types

struct AnalysisContext {
    let medications: [MedicationEntry]
    let activities: [ActivityData]?
    let sleep: [SleepData]?
    let weather: [WeatherData]?
}

struct PainPrediction: Identifiable, Codable {
    let id = UUID()
    let date: Date
    let predictedPainLevel: Double
    let confidence: Double
    let factors: [PredictionFactor]
    let recommendation: String?
}

struct PredictionFactor: Codable {
    let name: String
    let impact: Double // -1.0 to 1.0
    let description: String
}

struct CorrelationInsight: Identifiable, Codable {
    let id: UUID
    let type: CorrelationType
    let title: String
    let description: String
    let correlationStrength: Double
    let confidence: Double
    let actionable: Bool
    let recommendation: String?
}

enum CorrelationType: String, Codable {
    case medicationEffectiveness = "medication_effectiveness"
    case weatherPattern = "weather_pattern"
    case activityPattern = "activity_pattern"
    case sleepPattern = "sleep_pattern"
    case stressPattern = "stress_pattern"
    case dietPattern = "diet_pattern"
}

enum FlareRiskLevel: String, Codable, CaseIterable {
    case low = "low"
    case moderate = "moderate"
    case high = "high"
    case critical = "critical"
    
    var color: String {
        switch self {
        case .low: return "green"
        case .moderate: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }
    
    var description: String {
        switch self {
        case .low: return "Low risk of flare in the next 7 days"
        case .moderate: return "Moderate risk of flare - monitor symptoms closely"
        case .high: return "High risk of flare - consider preventive measures"
        case .critical: return "Critical risk - contact healthcare provider"
        }
    }
}

struct MedicationOptimization: Identifiable, Codable {
    let id = UUID()
    let medicationType: String
    let currentSchedule: String
    let suggestedSchedule: String
    let expectedImprovement: Double
    let confidence: Double
    let reasoning: String
}

struct PersonalizedRecommendation: Identifiable, Codable {
    let id = UUID()
    let category: RecommendationCategory
    let title: String
    let description: String
    let priority: RecommendationPriority
    let actionSteps: [String]
    let expectedBenefit: String
    let timeframe: String
}

enum RecommendationCategory: String, Codable {
    case medication = "medication"
    case exercise = "exercise"
    case lifestyle = "lifestyle"
    case diet = "diet"
    case sleep = "sleep"
    case stress = "stress"
    case weather = "weather"
}

enum RecommendationPriority: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case urgent = "urgent"
}

struct FlarePattern: Identifiable, Codable {
    let id = UUID()
    let patternType: FlarePatternType
    let frequency: String
    let triggers: [String]
    let duration: String
    let severity: String
    let confidence: Double
}

enum FlarePatternType: String, Codable {
    case seasonal = "seasonal"
    case weekly = "weekly"
    case monthly = "monthly"
    case stressBased = "stress_based"
    case weatherBased = "weather_based"
    case medicationBased = "medication_based"
}

struct PredictionFeatures {
    let date: Date
    let recentPainLevels: [Double]
    let recentMedications: [MedicationEntry]
    let weatherConditions: WeatherData?
    let sleepQuality: Double
    let dayOfWeek: Int
    let timeOfYear: Int
}

struct RecentUserData {
    let painEntries: [PainEntry]
    let medicationEntries: [MedicationEntry]
    let activityData: [ActivityData]?
    let sleepData: [SleepData]?
}

// MARK: - Extensions

extension Calendar {
    func dayOfYear(for date: Date) -> Int? {
        return ordinality(of: .day, in: .year, for: date)
    }
}