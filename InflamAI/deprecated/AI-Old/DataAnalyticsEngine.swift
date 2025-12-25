//
//  DataAnalyticsEngine.swift
//  InflamAI-Swift
//
//  Advanced data analytics and insights engine for health pattern analysis
//

import Foundation
import Combine
import HealthKit
import CoreML
import CreateML
import Charts
import SwiftUI
import CloudKit
import CoreData
import Accelerate
import simd

// MARK: - Analytics Models

struct HealthMetric: Codable, Identifiable {
    let id: String
    let type: MetricType
    let value: Double
    let unit: String
    let timestamp: Date
    let source: DataSource
    let confidence: Double
    let tags: [String]
    let metadata: [String: Any]?
    let correlationId: String?
    let deviceId: String?
    let accuracy: Double?
    let isOutlier: Bool
    let normalizedValue: Double?
    
    enum CodingKeys: String, CodingKey {
        case id, type, value, unit, timestamp, source, confidence, tags, correlationId, deviceId, accuracy, isOutlier, normalizedValue
    }
}

enum MetricType: String, Codable, CaseIterable {
    case painLevel = "pain_level"
    case stiffness = "stiffness"
    case fatigue = "fatigue"
    case mood = "mood"
    case sleepQuality = "sleep_quality"
    case exerciseIntensity = "exercise_intensity"
    case medicationAdherence = "medication_adherence"
    case heartRate = "heart_rate"
    case bloodPressure = "blood_pressure"
    case temperature = "temperature"
    case steps = "steps"
    case weight = "weight"
    case inflammation = "inflammation"
    case jointSwelling = "joint_swelling"
    case rangeOfMotion = "range_of_motion"
    case weatherPressure = "weather_pressure"
    case humidity = "humidity"
    case temperature_ambient = "temperature_ambient"
    case stress = "stress"
    case socialActivity = "social_activity"
    case workload = "workload"
    
    var displayName: String {
        switch self {
        case .painLevel: return "Pain Level"
        case .stiffness: return "Stiffness"
        case .fatigue: return "Fatigue"
        case .mood: return "Mood"
        case .sleepQuality: return "Sleep Quality"
        case .exerciseIntensity: return "Exercise Intensity"
        case .medicationAdherence: return "Medication Adherence"
        case .heartRate: return "Heart Rate"
        case .bloodPressure: return "Blood Pressure"
        case .temperature: return "Body Temperature"
        case .steps: return "Steps"
        case .weight: return "Weight"
        case .inflammation: return "Inflammation"
        case .jointSwelling: return "Joint Swelling"
        case .rangeOfMotion: return "Range of Motion"
        case .weatherPressure: return "Barometric Pressure"
        case .humidity: return "Humidity"
        case .temperature_ambient: return "Ambient Temperature"
        case .stress: return "Stress Level"
        case .socialActivity: return "Social Activity"
        case .workload: return "Work Load"
        }
    }
    
    var unit: String {
        switch self {
        case .painLevel, .stiffness, .fatigue, .mood, .sleepQuality, .stress: return "scale"
        case .exerciseIntensity, .medicationAdherence, .inflammation: return "percentage"
        case .heartRate: return "bpm"
        case .bloodPressure: return "mmHg"
        case .temperature: return "°F"
        case .steps: return "count"
        case .weight: return "lbs"
        case .jointSwelling: return "mm"
        case .rangeOfMotion: return "degrees"
        case .weatherPressure: return "inHg"
        case .humidity: return "%"
        case .temperature_ambient: return "°F"
        case .socialActivity, .workload: return "hours"
        }
    }
    
    var normalRange: ClosedRange<Double> {
        switch self {
        case .painLevel, .stiffness, .fatigue, .stress: return 0...10
        case .mood, .sleepQuality: return 1...10
        case .exerciseIntensity, .medicationAdherence, .inflammation, .humidity: return 0...100
        case .heartRate: return 60...100
        case .bloodPressure: return 80...120
        case .temperature: return 97...99
        case .steps: return 0...20000
        case .weight: return 100...300
        case .jointSwelling: return 0...50
        case .rangeOfMotion: return 0...180
        case .weatherPressure: return 28...32
        case .temperature_ambient: return 32...100
        case .socialActivity, .workload: return 0...16
        }
    }
    
    var isHigherBetter: Bool {
        switch self {
        case .mood, .sleepQuality, .exerciseIntensity, .medicationAdherence, .steps, .rangeOfMotion, .socialActivity:
            return true
        case .painLevel, .stiffness, .fatigue, .stress, .inflammation, .jointSwelling:
            return false
        default:
            return false
        }
    }
}

enum DataSource: String, Codable {
    case userInput = "user_input"
    case healthKit = "health_kit"
    case appleWatch = "apple_watch"
    case weatherAPI = "weather_api"
    case thirdPartyDevice = "third_party_device"
    case calculated = "calculated"
    case predicted = "predicted"
    case imported = "imported"
}

struct AnalyticsInsight: Codable, Identifiable {
    let id: String
    let type: InsightType
    let title: String
    let description: String
    let severity: InsightSeverity
    let confidence: Double
    let relevantMetrics: [MetricType]
    let timeframe: DateInterval
    let actionable: Bool
    let recommendations: [String]
    let supportingData: [HealthMetric]
    let correlations: [Correlation]
    let trends: [Trend]
    let predictions: [Prediction]
    let createdAt: Date
    let expiresAt: Date?
    let category: InsightCategory
    let tags: [String]
    let isPersonalized: Bool
    let evidenceLevel: EvidenceLevel
}

enum InsightType: String, Codable {
    case correlation = "correlation"
    case trend = "trend"
    case anomaly = "anomaly"
    case prediction = "prediction"
    case pattern = "pattern"
    case recommendation = "recommendation"
    case warning = "warning"
    case achievement = "achievement"
    case optimization = "optimization"
}

enum InsightSeverity: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
    
    var color: String {
        switch self {
        case .low: return "green"
        case .medium: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }
}

enum InsightCategory: String, Codable {
    case symptoms = "symptoms"
    case medication = "medication"
    case lifestyle = "lifestyle"
    case environmental = "environmental"
    case sleep = "sleep"
    case exercise = "exercise"
    case nutrition = "nutrition"
    case stress = "stress"
    case social = "social"
    case general = "general"
}

enum EvidenceLevel: String, Codable {
    case low = "low"
    case moderate = "moderate"
    case strong = "strong"
    case veryStrong = "very_strong"
    
    var description: String {
        switch self {
        case .low: return "Limited data available"
        case .moderate: return "Some supporting evidence"
        case .strong: return "Strong supporting evidence"
        case .veryStrong: return "Very strong evidence base"
        }
    }
}

struct Correlation: Codable, Identifiable {
    let id: String
    let metric1: MetricType
    let metric2: MetricType
    let coefficient: Double
    let pValue: Double
    let significance: CorrelationSignificance
    let direction: CorrelationDirection
    let timelag: TimeInterval?
    let strength: CorrelationStrength
    let sampleSize: Int
    let timeframe: DateInterval
    let isStatisticallySignificant: Bool
    let confidenceInterval: ClosedRange<Double>
}

enum CorrelationSignificance: String, Codable {
    case notSignificant = "not_significant"
    case marginal = "marginal"
    case significant = "significant"
    case highlySignificant = "highly_significant"
}

enum CorrelationDirection: String, Codable {
    case positive = "positive"
    case negative = "negative"
    case neutral = "neutral"
}

enum CorrelationStrength: String, Codable {
    case negligible = "negligible"
    case weak = "weak"
    case moderate = "moderate"
    case strong = "strong"
    case veryStrong = "very_strong"
    
    static func from(coefficient: Double) -> CorrelationStrength {
        let abs_coeff = abs(coefficient)
        switch abs_coeff {
        case 0.0..<0.1: return .negligible
        case 0.1..<0.3: return .weak
        case 0.3..<0.5: return .moderate
        case 0.5..<0.7: return .strong
        default: return .veryStrong
        }
    }
}

struct Trend: Codable, Identifiable {
    let id: String
    let metric: MetricType
    let direction: TrendDirection
    let magnitude: Double
    let duration: TimeInterval
    let confidence: Double
    let isSignificant: Bool
    let startDate: Date
    let endDate: Date
    let slope: Double
    let rSquared: Double
    let seasonality: SeasonalityInfo?
    let changePoints: [Date]
    let forecast: [ForecastPoint]
}

enum TrendDirection: String, Codable {
    case increasing = "increasing"
    case decreasing = "decreasing"
    case stable = "stable"
    case volatile = "volatile"
    case cyclical = "cyclical"
}

struct SeasonalityInfo: Codable {
    let period: TimeInterval
    let amplitude: Double
    let phase: Double
    let strength: Double
}

struct ForecastPoint: Codable {
    let date: Date
    let value: Double
    let confidence: Double
    let lowerBound: Double
    let upperBound: Double
}

struct Prediction: Codable, Identifiable {
    let id: String
    let type: PredictionType
    let targetMetric: MetricType
    let predictedValue: Double
    let confidence: Double
    let timeframe: DateInterval
    let factors: [PredictionFactor]
    let accuracy: Double?
    let modelVersion: String
    let createdAt: Date
    let isActionable: Bool
    let recommendations: [String]
}

enum PredictionType: String, Codable {
    case flareUp = "flare_up"
    case symptomSeverity = "symptom_severity"
    case medicationEffectiveness = "medication_effectiveness"
    case sleepQuality = "sleep_quality"
    case exercisePerformance = "exercise_performance"
    case moodChange = "mood_change"
    case painLevel = "pain_level"
    case overallWellbeing = "overall_wellbeing"
}

struct PredictionFactor: Codable {
    let metric: MetricType
    let importance: Double
    let direction: FactorDirection
    let confidence: Double
}

enum FactorDirection: String, Codable {
    case positive = "positive"
    case negative = "negative"
    case neutral = "neutral"
}

// MARK: - Statistical Models

struct StatisticalSummary: Codable {
    let mean: Double
    let median: Double
    let mode: Double?
    let standardDeviation: Double
    let variance: Double
    let minimum: Double
    let maximum: Double
    let range: Double
    let quartiles: Quartiles
    let skewness: Double
    let kurtosis: Double
    let outliers: [Double]
    let sampleSize: Int
    let confidenceInterval95: ClosedRange<Double>
}

struct Quartiles: Codable {
    let q1: Double
    let q2: Double // median
    let q3: Double
    let iqr: Double // interquartile range
}

struct TimeSeriesAnalysis: Codable {
    let metric: MetricType
    let timeframe: DateInterval
    let dataPoints: [TimeSeriesPoint]
    let trend: Trend
    let seasonality: SeasonalityInfo?
    let autocorrelation: [Double]
    let stationarity: StationarityTest
    let changePoints: [ChangePoint]
    let anomalies: [Anomaly]
    let forecast: [ForecastPoint]
    let modelFit: ModelFitStatistics
}

struct TimeSeriesPoint: Codable {
    let timestamp: Date
    let value: Double
    let isInterpolated: Bool
    let confidence: Double
}

struct StationarityTest: Codable {
    let isStationary: Bool
    let pValue: Double
    let testStatistic: Double
    let criticalValues: [Double]
    let testType: String
}

struct ChangePoint: Codable {
    let date: Date
    let magnitude: Double
    let confidence: Double
    let type: ChangePointType
}

enum ChangePointType: String, Codable {
    case mean = "mean"
    case variance = "variance"
    case trend = "trend"
    case seasonal = "seasonal"
}

struct Anomaly: Codable, Identifiable {
    let id: String
    let timestamp: Date
    let value: Double
    let expectedValue: Double
    let deviation: Double
    let severity: AnomalySeverity
    let type: AnomalyType
    let confidence: Double
    let possibleCauses: [String]
}

enum AnomalySeverity: String, Codable {
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    case extreme = "extreme"
}

enum AnomalyType: String, Codable {
    case pointAnomaly = "point_anomaly"
    case contextualAnomaly = "contextual_anomaly"
    case collectiveAnomaly = "collective_anomaly"
}

struct ModelFitStatistics: Codable {
    let rSquared: Double
    let adjustedRSquared: Double
    let rmse: Double
    let mae: Double
    let mape: Double
    let aic: Double
    let bic: Double
    let logLikelihood: Double
}

// MARK: - Data Analytics Engine

class DataAnalyticsEngine: ObservableObject {
    // Core Services
    private let healthStore = HKHealthStore()
    private let cloudKitContainer = CKContainer.default()
    
    // Published Properties
    @Published var insights: [AnalyticsInsight] = []
    @Published var correlations: [Correlation] = []
    @Published var trends: [Trend] = []
    @Published var predictions: [Prediction] = []
    @Published var anomalies: [Anomaly] = []
    @Published var isAnalyzing = false
    @Published var lastAnalysisDate: Date?
    @Published var analysisProgress: Double = 0.0
    
    // Internal State
    private var cancellables = Set<AnyCancellable>()
    private var healthMetrics: [HealthMetric] = []
    private var analysisQueue = DispatchQueue(label: "analytics.queue", qos: .userInitiated)
    private var mlModels: [String: MLModel] = [:]
    
    // Configuration
    private let minDataPointsForAnalysis = 7
    private let maxAnalysisTimeframe: TimeInterval = 365 * 24 * 60 * 60 // 1 year
    private let correlationThreshold = 0.3
    private let significanceLevel = 0.05
    
    init() {
        setupAnalyticsEngine()
        loadMLModels()
    }
    
    // MARK: - Setup
    
    private func setupAnalyticsEngine() {
        // Setup periodic analysis
        Timer.publish(every: 3600, on: .main, in: .common) // Every hour
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.performPeriodicAnalysis()
                }
            }
            .store(in: &cancellables)
    }
    
    private func loadMLModels() {
        // Load pre-trained ML models
        loadFlareUpPredictionModel()
        loadSymptomSeverityModel()
        loadMoodPredictionModel()
    }
    
    private func loadFlareUpPredictionModel() {
        // Load flare-up prediction model
    }
    
    private func loadSymptomSeverityModel() {
        // Load symptom severity prediction model
    }
    
    private func loadMoodPredictionModel() {
        // Load mood prediction model
    }
    
    // MARK: - Data Collection
    
    func collectHealthData(timeframe: DateInterval? = nil) async {
        let endDate = Date()
        let startDate = timeframe?.start ?? Calendar.current.date(byAdding: .month, value: -3, to: endDate)!
        
        await withTaskGroup(of: Void.self) { group in
            // Collect from HealthKit
            group.addTask {
                await self.collectHealthKitData(from: startDate, to: endDate)
            }
            
            // Collect user input data
            group.addTask {
                await self.collectUserInputData(from: startDate, to: endDate)
            }
            
            // Collect environmental data
            group.addTask {
                await self.collectEnvironmentalData(from: startDate, to: endDate)
            }
        }
    }
    
    private func collectHealthKitData(from startDate: Date, to endDate: Date) async {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        let metrics: [HKQuantityTypeIdentifier] = [
            .heartRate,
            .bloodPressureSystolic,
            .bloodPressureDiastolic,
            .bodyTemperature,
            .stepCount,
            .bodyMass
        ]
        
        for metricId in metrics {
            guard let quantityType = HKQuantityType.quantityType(forIdentifier: metricId) else { continue }
            
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            
            await withCheckedContinuation { continuation in
                let query = HKSampleQuery(
                    sampleType: quantityType,
                    predicate: predicate,
                    limit: HKObjectQueryNoLimit,
                    sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]
                ) { [weak self] query, samples, error in
                    defer { continuation.resume() }
                    
                    guard let samples = samples as? [HKQuantitySample] else { return }
                    
                    for sample in samples {
                        let metric = self?.convertHKSampleToHealthMetric(sample, type: metricId)
                        if let metric = metric {
                            self?.healthMetrics.append(metric)
                        }
                    }
                }
                
                healthStore.execute(query)
            }
        }
    }
    
    private func collectUserInputData(from startDate: Date, to endDate: Date) async {
        // Collect user input data from Core Data or other storage
    }
    
    private func collectEnvironmentalData(from startDate: Date, to endDate: Date) async {
        // Collect weather and environmental data
    }
    
    private func convertHKSampleToHealthMetric(_ sample: HKQuantitySample, type: HKQuantityTypeIdentifier) -> HealthMetric? {
        let metricType: MetricType
        let unit: HKUnit
        
        switch type {
        case .heartRate:
            metricType = .heartRate
            unit = HKUnit.count().unitDivided(by: .minute())
        case .bloodPressureSystolic:
            metricType = .bloodPressure
            unit = .millimeterOfMercury()
        case .bodyTemperature:
            metricType = .temperature
            unit = .degreeFahrenheit()
        case .stepCount:
            metricType = .steps
            unit = .count()
        case .bodyMass:
            metricType = .weight
            unit = .pound()
        default:
            return nil
        }
        
        return HealthMetric(
            id: sample.uuid.uuidString,
            type: metricType,
            value: sample.quantity.doubleValue(for: unit),
            unit: metricType.unit,
            timestamp: sample.startDate,
            source: .healthKit,
            confidence: 1.0,
            tags: [],
            metadata: nil,
            correlationId: nil,
            deviceId: sample.device?.name,
            accuracy: nil,
            isOutlier: false,
            normalizedValue: nil
        )
    }
    
    // MARK: - Analysis Methods
    
    func performComprehensiveAnalysis() async {
        DispatchQueue.main.async {
            self.isAnalyzing = true
            self.analysisProgress = 0.0
        }
        
        // Collect latest data
        await collectHealthData()
        
        DispatchQueue.main.async {
            self.analysisProgress = 0.2
        }
        
        // Perform different types of analysis
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await self.performCorrelationAnalysis()
            }
            
            group.addTask {
                await self.performTrendAnalysis()
            }
            
            group.addTask {
                await self.performAnomalyDetection()
            }
            
            group.addTask {
                await self.performPredictiveAnalysis()
            }
            
            group.addTask {
                await self.generateInsights()
            }
        }
        
        DispatchQueue.main.async {
            self.analysisProgress = 1.0
            self.isAnalyzing = false
            self.lastAnalysisDate = Date()
        }
    }
    
    private func performPeriodicAnalysis() async {
        // Perform lightweight analysis periodically
        await collectHealthData(timeframe: DateInterval(start: Calendar.current.date(byAdding: .day, value: -1, to: Date())!, end: Date()))
        await performTrendAnalysis()
        await performAnomalyDetection()
    }
    
    // MARK: - Correlation Analysis
    
    private func performCorrelationAnalysis() async {
        let metricTypes = Set(healthMetrics.map { $0.type })
        var newCorrelations: [Correlation] = []
        
        for metric1 in metricTypes {
            for metric2 in metricTypes {
                guard metric1 != metric2 else { continue }
                
                let data1 = getMetricData(for: metric1)
                let data2 = getMetricData(for: metric2)
                
                guard data1.count >= minDataPointsForAnalysis,
                      data2.count >= minDataPointsForAnalysis else { continue }
                
                if let correlation = calculateCorrelation(between: data1, and: data2, metric1: metric1, metric2: metric2) {
                    newCorrelations.append(correlation)
                }
            }
        }
        
        DispatchQueue.main.async {
            self.correlations = newCorrelations.filter { abs($0.coefficient) >= self.correlationThreshold }
        }
    }
    
    private func getMetricData(for type: MetricType) -> [(Date, Double)] {
        return healthMetrics
            .filter { $0.type == type }
            .sorted { $0.timestamp < $1.timestamp }
            .map { ($0.timestamp, $0.value) }
    }
    
    private func calculateCorrelation(between data1: [(Date, Double)], and data2: [(Date, Double)], metric1: MetricType, metric2: MetricType) -> Correlation? {
        // Align data points by time
        let alignedData = alignTimeSeriesData(data1, data2)
        guard alignedData.count >= minDataPointsForAnalysis else { return nil }
        
        let values1 = alignedData.map { $0.0 }
        let values2 = alignedData.map { $0.1 }
        
        // Calculate Pearson correlation coefficient
        let coefficient = pearsonCorrelation(values1, values2)
        
        // Calculate p-value and significance
        let pValue = calculatePValue(coefficient: coefficient, sampleSize: alignedData.count)
        let significance = determineSignificance(pValue: pValue)
        
        // Determine direction and strength
        let direction: CorrelationDirection = coefficient > 0 ? .positive : (coefficient < 0 ? .negative : .neutral)
        let strength = CorrelationStrength.from(coefficient: coefficient)
        
        // Calculate confidence interval
        let confidenceInterval = calculateConfidenceInterval(coefficient: coefficient, sampleSize: alignedData.count)
        
        return Correlation(
            id: UUID().uuidString,
            metric1: metric1,
            metric2: metric2,
            coefficient: coefficient,
            pValue: pValue,
            significance: significance,
            direction: direction,
            timelag: nil,
            strength: strength,
            sampleSize: alignedData.count,
            timeframe: DateInterval(start: data1.first?.0 ?? Date(), end: data1.last?.0 ?? Date()),
            isStatisticallySignificant: pValue < significanceLevel,
            confidenceInterval: confidenceInterval
        )
    }
    
    private func alignTimeSeriesData(_ data1: [(Date, Double)], _ data2: [(Date, Double)]) -> [(Double, Double)] {
        var aligned: [(Double, Double)] = []
        let tolerance: TimeInterval = 3600 // 1 hour tolerance
        
        for (date1, value1) in data1 {
            if let closest = data2.min(by: { abs($0.0.timeIntervalSince(date1)) < abs($1.0.timeIntervalSince(date1)) }),
               abs(closest.0.timeIntervalSince(date1)) <= tolerance {
                aligned.append((value1, closest.1))
            }
        }
        
        return aligned
    }
    
    private func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count > 1 else { return 0 }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        return denominator == 0 ? 0 : numerator / denominator
    }
    
    private func calculatePValue(coefficient: Double, sampleSize: Int) -> Double {
        // Simplified p-value calculation for correlation
        let t = coefficient * sqrt(Double(sampleSize - 2) / (1 - coefficient * coefficient))
        return 2 * (1 - tDistributionCDF(t: abs(t), df: sampleSize - 2))
    }
    
    private func tDistributionCDF(t: Double, df: Int) -> Double {
        // Simplified t-distribution CDF approximation
        return 0.5 + 0.5 * tanh(t / sqrt(Double(df)))
    }
    
    private func determineSignificance(pValue: Double) -> CorrelationSignificance {
        switch pValue {
        case 0..<0.001: return .highlySignificant
        case 0.001..<0.01: return .significant
        case 0.01..<0.1: return .marginal
        default: return .notSignificant
        }
    }
    
    private func calculateConfidenceInterval(coefficient: Double, sampleSize: Int) -> ClosedRange<Double> {
        // Fisher's z-transformation for confidence interval
        let z = 0.5 * log((1 + coefficient) / (1 - coefficient))
        let se = 1.0 / sqrt(Double(sampleSize - 3))
        let margin = 1.96 * se // 95% confidence interval
        
        let lowerZ = z - margin
        let upperZ = z + margin
        
        let lower = (exp(2 * lowerZ) - 1) / (exp(2 * lowerZ) + 1)
        let upper = (exp(2 * upperZ) - 1) / (exp(2 * upperZ) + 1)
        
        return lower...upper
    }
    
    // MARK: - Trend Analysis
    
    private func performTrendAnalysis() async {
        let metricTypes = Set(healthMetrics.map { $0.type })
        var newTrends: [Trend] = []
        
        for metricType in metricTypes {
            let data = getMetricData(for: metricType)
            guard data.count >= minDataPointsForAnalysis else { continue }
            
            if let trend = analyzeTrend(for: metricType, data: data) {
                newTrends.append(trend)
            }
        }
        
        DispatchQueue.main.async {
            self.trends = newTrends
        }
    }
    
    private func analyzeTrend(for metricType: MetricType, data: [(Date, Double)]) -> Trend? {
        guard data.count >= minDataPointsForAnalysis else { return nil }
        
        // Convert dates to numeric values for regression
        let baseDate = data.first!.0
        let x = data.map { $0.0.timeIntervalSince(baseDate) / 86400 } // Days since start
        let y = data.map { $0.1 }
        
        // Perform linear regression
        let (slope, intercept, rSquared) = linearRegression(x: x, y: y)
        
        // Determine trend direction
        let direction = determineTrendDirection(slope: slope, rSquared: rSquared)
        
        // Calculate magnitude and significance
        let magnitude = abs(slope)
        let isSignificant = rSquared > 0.5 && magnitude > 0.1
        
        // Generate forecast
        let forecast = generateForecast(slope: slope, intercept: intercept, baseDate: baseDate, data: data)
        
        return Trend(
            id: UUID().uuidString,
            metric: metricType,
            direction: direction,
            magnitude: magnitude,
            duration: data.last!.0.timeIntervalSince(data.first!.0),
            confidence: rSquared,
            isSignificant: isSignificant,
            startDate: data.first!.0,
            endDate: data.last!.0,
            slope: slope,
            rSquared: rSquared,
            seasonality: nil,
            changePoints: [],
            forecast: forecast
        )
    }
    
    private func linearRegression(x: [Double], y: [Double]) -> (slope: Double, intercept: Double, rSquared: Double) {
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        let intercept = (sumY - slope * sumX) / n
        
        // Calculate R-squared
        let yMean = sumY / n
        let ssTotal = y.map { pow($0 - yMean, 2) }.reduce(0, +)
        let ssResidual = zip(x, y).map { pow($1 - (slope * $0 + intercept), 2) }.reduce(0, +)
        let rSquared = 1 - (ssResidual / ssTotal)
        
        return (slope, intercept, rSquared)
    }
    
    private func determineTrendDirection(slope: Double, rSquared: Double) -> TrendDirection {
        if rSquared < 0.3 {
            return .volatile
        } else if abs(slope) < 0.01 {
            return .stable
        } else if slope > 0 {
            return .increasing
        } else {
            return .decreasing
        }
    }
    
    private func generateForecast(slope: Double, intercept: Double, baseDate: Date, data: [(Date, Double)]) -> [ForecastPoint] {
        var forecast: [ForecastPoint] = []
        let lastDate = data.last!.0
        
        for i in 1...7 { // 7-day forecast
            let futureDate = Calendar.current.date(byAdding: .day, value: i, to: lastDate)!
            let x = futureDate.timeIntervalSince(baseDate) / 86400
            let predictedValue = slope * x + intercept
            
            // Calculate confidence based on distance from last data point
            let confidence = max(0.1, 1.0 - Double(i) * 0.1)
            
            // Calculate bounds (simplified)
            let margin = abs(predictedValue) * 0.2 * Double(i)
            
            forecast.append(ForecastPoint(
                date: futureDate,
                value: predictedValue,
                confidence: confidence,
                lowerBound: predictedValue - margin,
                upperBound: predictedValue + margin
            ))
        }
        
        return forecast
    }
    
    // MARK: - Anomaly Detection
    
    private func performAnomalyDetection() async {
        let metricTypes = Set(healthMetrics.map { $0.type })
        var newAnomalies: [Anomaly] = []
        
        for metricType in metricTypes {
            let data = getMetricData(for: metricType)
            guard data.count >= minDataPointsForAnalysis else { continue }
            
            let anomalies = detectAnomalies(for: metricType, data: data)
            newAnomalies.append(contentsOf: anomalies)
        }
        
        DispatchQueue.main.async {
            self.anomalies = newAnomalies
        }
    }
    
    private func detectAnomalies(for metricType: MetricType, data: [(Date, Double)]) -> [Anomaly] {
        let values = data.map { $0.1 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count)
        let standardDeviation = sqrt(variance)
        
        var anomalies: [Anomaly] = []
        
        for (index, (date, value)) in data.enumerated() {
            let zScore = abs(value - mean) / standardDeviation
            
            if zScore > 2.5 { // Outlier threshold
                let severity: AnomalySeverity
                switch zScore {
                case 2.5..<3.0: severity = .mild
                case 3.0..<3.5: severity = .moderate
                case 3.5..<4.0: severity = .severe
                default: severity = .extreme
                }
                
                anomalies.append(Anomaly(
                    id: UUID().uuidString,
                    timestamp: date,
                    value: value,
                    expectedValue: mean,
                    deviation: value - mean,
                    severity: severity,
                    type: .pointAnomaly,
                    confidence: min(1.0, zScore / 4.0),
                    possibleCauses: generatePossibleCauses(for: metricType, value: value, mean: mean)
                ))
            }
        }
        
        return anomalies
    }
    
    private func generatePossibleCauses(for metricType: MetricType, value: Double, mean: Double) -> [String] {
        var causes: [String] = []
        
        switch metricType {
        case .painLevel:
            if value > mean {
                causes = ["Flare-up", "Medication change", "Weather change", "Stress", "Poor sleep"]
            } else {
                causes = ["Effective treatment", "Good day", "Medication working well"]
            }
        case .heartRate:
            if value > mean {
                causes = ["Exercise", "Stress", "Caffeine", "Medication side effect", "Illness"]
            } else {
                causes = ["Rest", "Medication effect", "Improved fitness"]
            }
        case .sleepQuality:
            if value < mean {
                causes = ["Pain", "Stress", "Medication", "Environmental factors"]
            } else {
                causes = ["Good pain management", "Relaxation", "Routine improvement"]
            }
        default:
            causes = ["Data entry error", "Unusual circumstances", "Measurement error"]
        }
        
        return causes
    }
    
    // MARK: - Predictive Analysis
    
    private func performPredictiveAnalysis() async {
        var newPredictions: [Prediction] = []
        
        // Predict flare-ups
        if let flareUpPrediction = await predictFlareUp() {
            newPredictions.append(flareUpPrediction)
        }
        
        // Predict symptom severity
        if let symptomPrediction = await predictSymptomSeverity() {
            newPredictions.append(symptomPrediction)
        }
        
        // Predict mood changes
        if let moodPrediction = await predictMoodChange() {
            newPredictions.append(moodPrediction)
        }
        
        DispatchQueue.main.async {
            self.predictions = newPredictions
        }
    }
    
    private func predictFlareUp() async -> Prediction? {
        // Implement flare-up prediction using ML model
        let painData = getMetricData(for: .painLevel)
        let stiffnessData = getMetricData(for: .stiffness)
        let fatigueData = getMetricData(for: .fatigue)
        
        guard painData.count >= minDataPointsForAnalysis else { return nil }
        
        // Simple heuristic-based prediction (replace with ML model)
        let recentPain = painData.suffix(7).map { $0.1 }
        let avgRecentPain = recentPain.reduce(0, +) / Double(recentPain.count)
        let painTrend = recentPain.last! - recentPain.first!
        
        let flareUpProbability = min(1.0, (avgRecentPain / 10.0) + (painTrend > 0 ? 0.3 : 0.0))
        
        if flareUpProbability > 0.6 {
            return Prediction(
                id: UUID().uuidString,
                type: .flareUp,
                targetMetric: .painLevel,
                predictedValue: flareUpProbability,
                confidence: 0.7,
                timeframe: DateInterval(start: Date(), end: Calendar.current.date(byAdding: .day, value: 3, to: Date())!),
                factors: [
                    PredictionFactor(metric: .painLevel, importance: 0.8, direction: .positive, confidence: 0.9),
                    PredictionFactor(metric: .stiffness, importance: 0.6, direction: .positive, confidence: 0.7),
                    PredictionFactor(metric: .fatigue, importance: 0.5, direction: .positive, confidence: 0.6)
                ],
                accuracy: nil,
                modelVersion: "1.0",
                createdAt: Date(),
                isActionable: true,
                recommendations: [
                    "Consider adjusting medication timing",
                    "Increase rest and stress management",
                    "Monitor symptoms closely",
                    "Contact healthcare provider if symptoms worsen"
                ]
            )
        }
        
        return nil
    }
    
    private func predictSymptomSeverity() async -> Prediction? {
        // Implement symptom severity prediction
        return nil
    }
    
    private func predictMoodChange() async -> Prediction? {
        // Implement mood change prediction
        return nil
    }
    
    // MARK: - Insight Generation
    
    private func generateInsights() async {
        var newInsights: [AnalyticsInsight] = []
        
        // Generate correlation insights
        for correlation in correlations {
            if let insight = generateCorrelationInsight(correlation) {
                newInsights.append(insight)
            }
        }
        
        // Generate trend insights
        for trend in trends {
            if let insight = generateTrendInsight(trend) {
                newInsights.append(insight)
            }
        }
        
        // Generate anomaly insights
        for anomaly in anomalies {
            if let insight = generateAnomalyInsight(anomaly) {
                newInsights.append(insight)
            }
        }
        
        // Generate prediction insights
        for prediction in predictions {
            if let insight = generatePredictionInsight(prediction) {
                newInsights.append(insight)
            }
        }
        
        DispatchQueue.main.async {
            self.insights = newInsights.sorted { $0.severity.rawValue > $1.severity.rawValue }
        }
    }
    
    private func generateCorrelationInsight(_ correlation: Correlation) -> AnalyticsInsight? {
        guard correlation.isStatisticallySignificant && correlation.strength != .negligible else { return nil }
        
        let title = "\(correlation.metric1.displayName) and \(correlation.metric2.displayName) Connection"
        let description = generateCorrelationDescription(correlation)
        let severity: InsightSeverity = correlation.strength == .strong || correlation.strength == .veryStrong ? .medium : .low
        
        return AnalyticsInsight(
            id: UUID().uuidString,
            type: .correlation,
            title: title,
            description: description,
            severity: severity,
            confidence: 1.0 - correlation.pValue,
            relevantMetrics: [correlation.metric1, correlation.metric2],
            timeframe: correlation.timeframe,
            actionable: true,
            recommendations: generateCorrelationRecommendations(correlation),
            supportingData: [],
            correlations: [correlation],
            trends: [],
            predictions: [],
            createdAt: Date(),
            expiresAt: Calendar.current.date(byAdding: .month, value: 1, to: Date()),
            category: determineInsightCategory(for: correlation.metric1),
            tags: ["correlation", correlation.strength.rawValue],
            isPersonalized: true,
            evidenceLevel: .strong
        )
    }
    
    private func generateCorrelationDescription(_ correlation: Correlation) -> String {
        let strength = correlation.strength.rawValue.capitalized
        let direction = correlation.direction == .positive ? "positive" : "negative"
        
        return "\(strength) \(direction) correlation found between \(correlation.metric1.displayName) and \(correlation.metric2.displayName). When one increases, the other tends to \(correlation.direction == .positive ? "increase" : "decrease") as well."
    }
    
    private func generateCorrelationRecommendations(_ correlation: Correlation) -> [String] {
        var recommendations: [String] = []
        
        if correlation.metric1 == .painLevel || correlation.metric2 == .painLevel {
            recommendations.append("Monitor both metrics closely to understand pain patterns")
            recommendations.append("Consider lifestyle adjustments that may affect both metrics")
        }
        
        if correlation.metric1 == .sleepQuality || correlation.metric2 == .sleepQuality {
            recommendations.append("Focus on sleep hygiene to potentially improve both metrics")
        }
        
        if correlation.metric1 == .stress || correlation.metric2 == .stress {
            recommendations.append("Implement stress management techniques")
        }
        
        return recommendations
    }
    
    private func generateTrendInsight(_ trend: Trend) -> AnalyticsInsight? {
        guard trend.isSignificant else { return nil }
        
        let title = "\(trend.metric.displayName) Trend"
        let description = generateTrendDescription(trend)
        let severity = determineTrendSeverity(trend)
        
        return AnalyticsInsight(
            id: UUID().uuidString,
            type: .trend,
            title: title,
            description: description,
            severity: severity,
            confidence: trend.confidence,
            relevantMetrics: [trend.metric],
            timeframe: DateInterval(start: trend.startDate, end: trend.endDate),
            actionable: true,
            recommendations: generateTrendRecommendations(trend),
            supportingData: [],
            correlations: [],
            trends: [trend],
            predictions: [],
            createdAt: Date(),
            expiresAt: Calendar.current.date(byAdding: .week, value: 2, to: Date()),
            category: determineInsightCategory(for: trend.metric),
            tags: ["trend", trend.direction.rawValue],
            isPersonalized: true,
            evidenceLevel: .moderate
        )
    }
    
    private func generateTrendDescription(_ trend: Trend) -> String {
        let direction = trend.direction.rawValue.capitalized
        let duration = Int(trend.duration / 86400) // Days
        
        return "\(trend.metric.displayName) has been \(direction.lowercased()) over the past \(duration) days."
    }
    
    private func determineTrendSeverity(_ trend: Trend) -> InsightSeverity {
        if trend.metric == .painLevel && trend.direction == .increasing {
            return .high
        } else if trend.metric == .sleepQuality && trend.direction == .decreasing {
            return .medium
        } else {
            return .low
        }
    }
    
    private func generateTrendRecommendations(_ trend: Trend) -> [String] {
        var recommendations: [String] = []
        
        switch (trend.metric, trend.direction) {
        case (.painLevel, .increasing):
            recommendations = [
                "Consider consulting your healthcare provider",
                "Review recent changes in medication or lifestyle",
                "Implement additional pain management strategies"
            ]
        case (.sleepQuality, .decreasing):
            recommendations = [
                "Review sleep hygiene practices",
                "Consider factors affecting sleep quality",
                "Track sleep environment changes"
            ]
        case (.mood, .decreasing):
            recommendations = [
                "Consider stress management techniques",
                "Engage in mood-boosting activities",
                "Consider speaking with a mental health professional"
            ]
        default:
            recommendations = ["Continue monitoring this trend"]
        }
        
        return recommendations
    }
    
    private func generateAnomalyInsight(_ anomaly: Anomaly) -> AnalyticsInsight? {
        guard anomaly.severity != .mild else { return nil }
        
        let title = "Unusual \(MetricType.allCases.first { _ in true }?.displayName ?? "Reading")"
        let description = "An unusual reading was detected that differs significantly from your typical pattern."
        
        return AnalyticsInsight(
            id: UUID().uuidString,
            type: .anomaly,
            title: title,
            description: description,
            severity: anomaly.severity == .extreme ? .high : .medium,
            confidence: anomaly.confidence,
            relevantMetrics: [],
            timeframe: DateInterval(start: anomaly.timestamp, end: anomaly.timestamp),
            actionable: true,
            recommendations: anomaly.possibleCauses.map { "Consider if \($0.lowercased()) might be a factor" },
            supportingData: [],
            correlations: [],
            trends: [],
            predictions: [],
            createdAt: Date(),
            expiresAt: Calendar.current.date(byAdding: .day, value: 7, to: Date()),
            category: .general,
            tags: ["anomaly", anomaly.severity.rawValue],
            isPersonalized: true,
            evidenceLevel: .moderate
        )
    }
    
    private func generatePredictionInsight(_ prediction: Prediction) -> AnalyticsInsight? {
        guard prediction.confidence > 0.6 else { return nil }
        
        let title = "\(prediction.type.rawValue.capitalized.replacingOccurrences(of: "_", with: " ")) Prediction"
        let description = "Based on recent patterns, there's a \(Int(prediction.confidence * 100))% chance of \(prediction.type.rawValue.replacingOccurrences(of: "_", with: " "))."
        
        return AnalyticsInsight(
            id: UUID().uuidString,
            type: .prediction,
            title: title,
            description: description,
            severity: prediction.confidence > 0.8 ? .high : .medium,
            confidence: prediction.confidence,
            relevantMetrics: [prediction.targetMetric],
            timeframe: prediction.timeframe,
            actionable: prediction.isActionable,
            recommendations: prediction.recommendations,
            supportingData: [],
            correlations: [],
            trends: [],
            predictions: [prediction],
            createdAt: Date(),
            expiresAt: prediction.timeframe.end,
            category: determineInsightCategory(for: prediction.targetMetric),
            tags: ["prediction", prediction.type.rawValue],
            isPersonalized: true,
            evidenceLevel: .strong
        )
    }
    
    private func determineInsightCategory(for metric: MetricType) -> InsightCategory {
        switch metric {
        case .painLevel, .stiffness, .inflammation, .jointSwelling:
            return .symptoms
        case .medicationAdherence:
            return .medication
        case .sleepQuality:
            return .sleep
        case .exerciseIntensity, .steps:
            return .exercise
        case .mood, .stress:
            return .stress
        case .socialActivity:
            return .social
        case .weatherPressure, .humidity, .temperature_ambient:
            return .environmental
        default:
            return .general
        }
    }
    
    // MARK: - Public Methods
    
    func getStatisticalSummary(for metricType: MetricType, timeframe: DateInterval? = nil) -> StatisticalSummary? {
        let data = healthMetrics
            .filter { $0.type == metricType }
            .filter { timeframe?.contains($0.timestamp) ?? true }
            .map { $0.value }
        
        guard data.count >= minDataPointsForAnalysis else { return nil }
        
        let sortedData = data.sorted()
        let mean = data.reduce(0, +) / Double(data.count)
        let variance = data.map { pow($0 - mean, 2) }.reduce(0, +) / Double(data.count)
        let standardDeviation = sqrt(variance)
        
        let q1Index = data.count / 4
        let q2Index = data.count / 2
        let q3Index = 3 * data.count / 4
        
        let quartiles = Quartiles(
            q1: sortedData[q1Index],
            q2: sortedData[q2Index],
            q3: sortedData[q3Index],
            iqr: sortedData[q3Index] - sortedData[q1Index]
        )
        
        // Calculate outliers using IQR method
        let lowerBound = quartiles.q1 - 1.5 * quartiles.iqr
        let upperBound = quartiles.q3 + 1.5 * quartiles.iqr
        let outliers = data.filter { $0 < lowerBound || $0 > upperBound }
        
        // Calculate confidence interval
        let marginOfError = 1.96 * standardDeviation / sqrt(Double(data.count))
        let confidenceInterval = (mean - marginOfError)...(mean + marginOfError)
        
        return StatisticalSummary(
            mean: mean,
            median: quartiles.q2,
            mode: calculateMode(data),
            standardDeviation: standardDeviation,
            variance: variance,
            minimum: sortedData.first!,
            maximum: sortedData.last!,
            range: sortedData.last! - sortedData.first!,
            quartiles: quartiles,
            skewness: calculateSkewness(data, mean: mean, standardDeviation: standardDeviation),
            kurtosis: calculateKurtosis(data, mean: mean, standardDeviation: standardDeviation),
            outliers: outliers,
            sampleSize: data.count,
            confidenceInterval95: confidenceInterval
        )
    }
    
    private func calculateMode(_ data: [Double]) -> Double? {
        let counts = Dictionary(data.map { ($0, 1) }, uniquingKeysWith: +)
        return counts.max { $0.value < $1.value }?.key
    }
    
    private func calculateSkewness(_ data: [Double], mean: Double, standardDeviation: Double) -> Double {
        let n = Double(data.count)
        let skewness = data.map { pow(($0 - mean) / standardDeviation, 3) }.reduce(0, +) / n
        return skewness
    }
    
    private func calculateKurtosis(_ data: [Double], mean: Double, standardDeviation: Double) -> Double {
        let n = Double(data.count)
        let kurtosis = data.map { pow(($0 - mean) / standardDeviation, 4) }.reduce(0, +) / n - 3
        return kurtosis
    }
    
    func exportAnalyticsData() -> [String: Any] {
        return [
            "insights": insights.map { try? JSONEncoder().encode($0) }.compactMap { $0 },
            "correlations": correlations.map { try? JSONEncoder().encode($0) }.compactMap { $0 },
            "trends": trends.map { try? JSONEncoder().encode($0) }.compactMap { $0 },
            "predictions": predictions.map { try? JSONEncoder().encode($0) }.compactMap { $0 },
            "anomalies": anomalies.map { try? JSONEncoder().encode($0) }.compactMap { $0 },
            "lastAnalysis": lastAnalysisDate?.timeIntervalSince1970 ?? 0
        ]
    }
}