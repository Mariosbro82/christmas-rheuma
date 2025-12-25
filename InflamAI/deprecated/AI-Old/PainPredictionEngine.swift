//
//  PainPredictionEngine.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreML
import CreateML
import HealthKit
import CoreLocation
import WeatherKit
import os.log

// MARK: - Pain Prediction Engine
@MainActor
class PainPredictionEngine: ObservableObject {
    
    private let logger = Logger(subsystem: "InflamAI", category: "PainPredictionEngine")
    
    // Published properties
    @Published var currentPrediction: PainPrediction?
    @Published var weeklyForecast: [PainPrediction] = []
    @Published var riskFactors: [RiskFactor] = []
    @Published var isAnalyzing = false
    @Published var lastAnalysisDate: Date?
    
    // ML Models
    private var painPredictionModel: MLModel?
    private var flareDetectionModel: MLModel?
    private var patternAnalysisModel: MLModel?
    
    // Data sources
    private let healthStore = HKHealthStore()
    private let locationManager = CLLocationManager()
    private var weatherService: WeatherService?
    
    // Historical data
    private var painHistory: [PainEntry] = []
    private var weatherHistory: [WeatherData] = []
    private var activityHistory: [ActivityData] = []
    private var sleepHistory: [SleepData] = []
    private var stressHistory: [StressData] = []
    
    // Analysis parameters
    private let predictionWindow = 7 // Days to predict ahead
    private let analysisWindow = 90 // Days of historical data to analyze
    private let minimumDataPoints = 30
    private let modelUpdateInterval: TimeInterval = 24 * 60 * 60 // 24 hours
    
    init() {
        setupWeatherService()
        loadHistoricalData()
        loadMLModels()
        
        // Start periodic analysis
        startPeriodicAnalysis()
    }
    
    // MARK: - Public Methods
    
    func generatePainPrediction() async {
        logger.info("Generating pain prediction")
        isAnalyzing = true
        
        do {
            // Collect current data
            let currentData = await collectCurrentData()
            
            // Generate prediction using ML model or fallback to pattern analysis
            if let mlModel = painPredictionModel {
                currentPrediction = await predictWithMLModel(mlModel, data: currentData)
            } else {
                currentPrediction = await predictWithPatternAnalysis(data: currentData)
            }
            
            // Generate weekly forecast
            weeklyForecast = await generateWeeklyForecast(baseData: currentData)
            
            // Analyze risk factors
            riskFactors = await analyzeRiskFactors(data: currentData)
            
            lastAnalysisDate = Date()
            
        } catch {
            logger.error("Pain prediction failed: \(error.localizedDescription)")
        }
        
        isAnalyzing = false
    }
    
    func addPainEntry(_ entry: PainEntry) async {
        painHistory.append(entry)
        
        // Keep only recent history
        let cutoffDate = Date().addingTimeInterval(-Double(analysisWindow) * 24 * 60 * 60)
        painHistory = painHistory.filter { $0.date >= cutoffDate }
        
        // Update models if we have enough data
        if painHistory.count >= minimumDataPoints {
            await updateMLModels()
        }
        
        // Regenerate prediction
        await generatePainPrediction()
        
        // Save data
        savePainHistory()
    }
    
    func detectFlareRisk() async -> FlareRiskAssessment {
        logger.info("Detecting flare risk")
        
        let currentData = await collectCurrentData()
        
        if let flareModel = flareDetectionModel {
            return await detectFlareWithMLModel(flareModel, data: currentData)
        } else {
            return await detectFlareWithRules(data: currentData)
        }
    }
    
    func analyzePatterns() async -> PatternAnalysisResult {
        logger.info("Analyzing pain patterns")
        
        // Analyze temporal patterns
        let temporalPatterns = analyzeTemporalPatterns()
        
        // Analyze weather correlations
        let weatherCorrelations = analyzeWeatherCorrelations()
        
        // Analyze activity correlations
        let activityCorrelations = analyzeActivityCorrelations()
        
        // Analyze sleep correlations
        let sleepCorrelations = analyzeSleepCorrelations()
        
        // Analyze stress correlations
        let stressCorrelations = analyzeStressCorrelations()
        
        return PatternAnalysisResult(
            temporalPatterns: temporalPatterns,
            weatherCorrelations: weatherCorrelations,
            activityCorrelations: activityCorrelations,
            sleepCorrelations: sleepCorrelations,
            stressCorrelations: stressCorrelations,
            analysisDate: Date()
        )
    }
    
    // MARK: - Private Methods
    
    private func setupWeatherService() {
        weatherService = WeatherService()
        locationManager.requestWhenInUseAuthorization()
    }
    
    private func loadHistoricalData() {
        // Load pain history
        if let data = UserDefaults.standard.data(forKey: "pain_history"),
           let history = try? JSONDecoder().decode([PainEntry].self, from: data) {
            painHistory = history
            logger.info("Loaded \(history.count) pain entries")
        }
        
        // Load weather history
        if let data = UserDefaults.standard.data(forKey: "weather_history"),
           let history = try? JSONDecoder().decode([WeatherData].self, from: data) {
            weatherHistory = history
            logger.info("Loaded \(history.count) weather entries")
        }
        
        // Load other historical data
        loadActivityHistory()
        loadSleepHistory()
        loadStressHistory()
    }
    
    private func loadActivityHistory() {
        if let data = UserDefaults.standard.data(forKey: "activity_history"),
           let history = try? JSONDecoder().decode([ActivityData].self, from: data) {
            activityHistory = history
        }
    }
    
    private func loadSleepHistory() {
        if let data = UserDefaults.standard.data(forKey: "sleep_history"),
           let history = try? JSONDecoder().decode([SleepData].self, from: data) {
            sleepHistory = history
        }
    }
    
    private func loadStressHistory() {
        if let data = UserDefaults.standard.data(forKey: "stress_history"),
           let history = try? JSONDecoder().decode([StressData].self, from: data) {
            stressHistory = history
        }
    }
    
    private func savePainHistory() {
        if let data = try? JSONEncoder().encode(painHistory) {
            UserDefaults.standard.set(data, forKey: "pain_history")
        }
    }
    
    private func loadMLModels() {
        Task {
            // Load pain prediction model
            if let modelURL = Bundle.main.url(forResource: "PainPredictionModel", withExtension: "mlmodel") {
                do {
                    painPredictionModel = try MLModel(contentsOf: modelURL)
                    logger.info("Loaded pain prediction model")
                } catch {
                    logger.error("Failed to load pain prediction model: \(error.localizedDescription)")
                }
            }
            
            // Load flare detection model
            if let modelURL = Bundle.main.url(forResource: "FlareDetectionModel", withExtension: "mlmodel") {
                do {
                    flareDetectionModel = try MLModel(contentsOf: modelURL)
                    logger.info("Loaded flare detection model")
                } catch {
                    logger.error("Failed to load flare detection model: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func collectCurrentData() async -> PredictionInputData {
        // Get current weather
        let weather = await getCurrentWeather()
        
        // Get recent health data
        let recentActivity = await getRecentActivityData()
        let recentSleep = await getRecentSleepData()
        let recentStress = await getRecentStressData()
        
        // Get recent pain data
        let recentPain = getRecentPainData()
        
        return PredictionInputData(
            currentWeather: weather,
            recentPainLevels: recentPain,
            recentActivity: recentActivity,
            recentSleep: recentSleep,
            recentStress: recentStress,
            timeOfDay: Calendar.current.component(.hour, from: Date()),
            dayOfWeek: Calendar.current.component(.weekday, from: Date()),
            seasonalFactor: getSeasonalFactor()
        )
    }
    
    private func getCurrentWeather() async -> WeatherData? {
        guard let location = locationManager.location,
              let weatherService = weatherService else {
            return nil
        }
        
        do {
            let weather = try await weatherService.weather(for: location)
            
            let weatherData = WeatherData(
                date: Date(),
                temperature: weather.currentWeather.temperature.value,
                humidity: weather.currentWeather.humidity,
                pressure: weather.currentWeather.pressure.value,
                windSpeed: weather.currentWeather.wind.speed.value,
                precipitationChance: weather.dailyForecast.first?.precipitationChance ?? 0.0,
                uvIndex: weather.currentWeather.uvIndex.value
            )
            
            // Add to history
            weatherHistory.append(weatherData)
            
            return weatherData
            
        } catch {
            logger.error("Failed to get weather data: \(error.localizedDescription)")
            return nil
        }
    }
    
    private func getRecentActivityData() async -> [ActivityData] {
        // Get activity data from HealthKit
        let calendar = Calendar.current
        let endDate = Date()
        let startDate = calendar.date(byAdding: .day, value: -7, to: endDate)!
        
        return await withCheckedContinuation { continuation in
            let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            
            let query = HKStatisticsCollectionQuery(
                quantityType: stepType,
                quantitySamplePredicate: predicate,
                options: .cumulativeSum,
                anchorDate: startDate,
                intervalComponents: DateComponents(day: 1)
            )
            
            query.initialResultsHandler = { _, results, error in
                guard let results = results else {
                    continuation.resume(returning: [])
                    return
                }
                
                var activityData: [ActivityData] = []
                
                results.enumerateStatistics(from: startDate, to: endDate) { statistics, _ in
                    let steps = statistics.sumQuantity()?.doubleValue(for: .count()) ?? 0
                    
                    activityData.append(ActivityData(
                        date: statistics.startDate,
                        stepCount: Int(steps),
                        activeEnergyBurned: 0, // Would need separate query
                        exerciseMinutes: 0 // Would need separate query
                    ))
                }
                
                continuation.resume(returning: activityData)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getRecentSleepData() async -> [SleepData] {
        // Get sleep data from HealthKit
        let calendar = Calendar.current
        let endDate = Date()
        let startDate = calendar.date(byAdding: .day, value: -7, to: endDate)!
        
        return await withCheckedContinuation { continuation in
            let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
            
            let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                guard let samples = samples as? [HKCategorySample] else {
                    continuation.resume(returning: [])
                    return
                }
                
                var sleepData: [SleepData] = []
                
                for sample in samples {
                    let duration = sample.endDate.timeIntervalSince(sample.startDate)
                    let quality = sample.value == HKCategoryValueSleepAnalysis.asleepDeep.rawValue ? 0.8 : 0.6
                    
                    sleepData.append(SleepData(
                        date: sample.startDate,
                        duration: duration,
                        quality: quality,
                        efficiency: 0.85 // Would calculate from actual data
                    ))
                }
                
                continuation.resume(returning: sleepData)
            }
            
            healthStore.execute(query)
        }
    }
    
    private func getRecentStressData() async -> [StressData] {
        // This would integrate with heart rate variability data from HealthKit
        // For now, return mock data
        let calendar = Calendar.current
        var stressData: [StressData] = []
        
        for i in 0..<7 {
            let date = calendar.date(byAdding: .day, value: -i, to: Date())!
            stressData.append(StressData(
                date: date,
                level: Double.random(in: 0.2...0.8),
                heartRateVariability: Double.random(in: 20...60)
            ))
        }
        
        return stressData
    }
    
    private func getRecentPainData() -> [Double] {
        let calendar = Calendar.current
        let sevenDaysAgo = calendar.date(byAdding: .day, value: -7, to: Date())!
        
        return painHistory
            .filter { $0.date >= sevenDaysAgo }
            .map { $0.painLevel }
    }
    
    private func getSeasonalFactor() -> Double {
        let month = Calendar.current.component(.month, from: Date())
        
        // Higher values for months typically associated with increased arthritis symptoms
        switch month {
        case 12, 1, 2: return 0.8 // Winter
        case 3, 4, 5: return 0.4 // Spring
        case 6, 7, 8: return 0.2 // Summer
        case 9, 10, 11: return 0.6 // Fall
        default: return 0.5
        }
    }
    
    private func predictWithMLModel(_ model: MLModel, data: PredictionInputData) async -> PainPrediction {
        do {
            // Prepare features for ML model
            let features = preparePredictionFeatures(data: data)
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            
            // Make prediction
            let prediction = try model.prediction(from: input)
            
            // Extract prediction results
            let predictedPainLevel = prediction.featureValue(for: "predicted_pain_level")?.doubleValue ?? 5.0
            let confidence = prediction.featureValue(for: "confidence")?.doubleValue ?? 0.5
            let flareRisk = prediction.featureValue(for: "flare_risk")?.doubleValue ?? 0.3
            
            logger.info("ML prediction: pain=\(predictedPainLevel), confidence=\(confidence)")
            
            return PainPrediction(
                date: Date(),
                predictedPainLevel: predictedPainLevel,
                confidence: confidence,
                flareRisk: flareRisk,
                contributingFactors: identifyContributingFactors(data: data),
                recommendations: generateRecommendations(painLevel: predictedPainLevel, data: data)
            )
            
        } catch {
            logger.error("ML prediction failed: \(error.localizedDescription)")
            return await predictWithPatternAnalysis(data: data)
        }
    }
    
    private func predictWithPatternAnalysis(data: PredictionInputData) async -> PainPrediction {
        logger.info("Using pattern-based prediction")
        
        var painScore = 5.0 // Base pain level
        var confidence = 0.6
        var flareRisk = 0.3
        
        // Weather factors
        if let weather = data.currentWeather {
            // Barometric pressure effect
            if weather.pressure < 1013.25 { // Below standard atmospheric pressure
                painScore += 1.0
                flareRisk += 0.2
            }
            
            // Humidity effect
            if weather.humidity > 0.7 {
                painScore += 0.5
                flareRisk += 0.1
            }
            
            // Temperature extremes
            if weather.temperature < 10 || weather.temperature > 30 {
                painScore += 0.5
            }
        }
        
        // Recent pain trend
        if !data.recentPainLevels.isEmpty {
            let averageRecentPain = data.recentPainLevels.reduce(0, +) / Double(data.recentPainLevels.count)
            let trend = calculatePainTrend(data.recentPainLevels)
            
            painScore = (painScore + averageRecentPain) / 2.0
            
            if trend > 0.5 { // Increasing trend
                flareRisk += 0.3
            }
            
            confidence += 0.2 // More confident with recent data
        }
        
        // Sleep quality effect
        if let recentSleep = data.recentSleep.last {
            if recentSleep.quality < 0.6 {
                painScore += 1.0
                flareRisk += 0.2
            }
        }
        
        // Stress effect
        if let recentStress = data.recentStress.last {
            if recentStress.level > 0.7 {
                painScore += 0.8
                flareRisk += 0.15
            }
        }
        
        // Seasonal adjustment
        painScore += data.seasonalFactor
        
        // Clamp values
        painScore = max(1.0, min(10.0, painScore))
        confidence = max(0.0, min(1.0, confidence))
        flareRisk = max(0.0, min(1.0, flareRisk))
        
        return PainPrediction(
            date: Date(),
            predictedPainLevel: painScore,
            confidence: confidence,
            flareRisk: flareRisk,
            contributingFactors: identifyContributingFactors(data: data),
            recommendations: generateRecommendations(painLevel: painScore, data: data)
        )
    }
    
    private func generateWeeklyForecast(baseData: PredictionInputData) async -> [PainPrediction] {
        var forecast: [PainPrediction] = []
        let calendar = Calendar.current
        
        for day in 1...predictionWindow {
            let forecastDate = calendar.date(byAdding: .day, value: day, to: Date())!
            
            // Simulate future data based on patterns
            var futureData = baseData
            
            // Adjust for day of week patterns
            let dayOfWeek = calendar.component(.weekday, from: forecastDate)
            let weekendFactor = [1, 7].contains(dayOfWeek) ? 0.8 : 1.0 // Lower pain on weekends
            
            // Generate prediction
            let basePrediction = await predictWithPatternAnalysis(data: futureData)
            
            let adjustedPrediction = PainPrediction(
                date: forecastDate,
                predictedPainLevel: basePrediction.predictedPainLevel * weekendFactor,
                confidence: basePrediction.confidence * 0.8, // Lower confidence for future predictions
                flareRisk: basePrediction.flareRisk,
                contributingFactors: basePrediction.contributingFactors,
                recommendations: basePrediction.recommendations
            )
            
            forecast.append(adjustedPrediction)
        }
        
        return forecast
    }
    
    private func analyzeRiskFactors(data: PredictionInputData) async -> [RiskFactor] {
        var factors: [RiskFactor] = []
        
        // Weather risk factors
        if let weather = data.currentWeather {
            if weather.pressure < 1013.25 {
                factors.append(RiskFactor(
                    type: .weather,
                    severity: .moderate,
                    description: "Low barometric pressure (\(Int(weather.pressure)) hPa)",
                    impact: 0.6
                ))
            }
            
            if weather.humidity > 0.7 {
                factors.append(RiskFactor(
                    type: .weather,
                    severity: .mild,
                    description: "High humidity (\(Int(weather.humidity * 100))%)",
                    impact: 0.4
                ))
            }
        }
        
        // Sleep risk factors
        if let recentSleep = data.recentSleep.last, recentSleep.quality < 0.6 {
            factors.append(RiskFactor(
                type: .sleep,
                severity: .moderate,
                description: "Poor sleep quality (\(Int(recentSleep.quality * 100))%)",
                impact: 0.7
            ))
        }
        
        // Stress risk factors
        if let recentStress = data.recentStress.last, recentStress.level > 0.7 {
            factors.append(RiskFactor(
                type: .stress,
                severity: .high,
                description: "High stress level (\(Int(recentStress.level * 100))%)",
                impact: 0.8
            ))
        }
        
        // Activity risk factors
        if let recentActivity = data.recentActivity.last {
            if recentActivity.stepCount < 2000 {
                factors.append(RiskFactor(
                    type: .activity,
                    severity: .mild,
                    description: "Low activity level (\(recentActivity.stepCount) steps)",
                    impact: 0.3
                ))
            } else if recentActivity.stepCount > 15000 {
                factors.append(RiskFactor(
                    type: .activity,
                    severity: .moderate,
                    description: "High activity level (\(recentActivity.stepCount) steps)",
                    impact: 0.5
                ))
            }
        }
        
        return factors
    }
    
    private func identifyContributingFactors(data: PredictionInputData) -> [String] {
        var factors: [String] = []
        
        if let weather = data.currentWeather {
            if weather.pressure < 1013.25 {
                factors.append("Low barometric pressure")
            }
            if weather.humidity > 0.7 {
                factors.append("High humidity")
            }
        }
        
        if let recentSleep = data.recentSleep.last, recentSleep.quality < 0.6 {
            factors.append("Poor sleep quality")
        }
        
        if let recentStress = data.recentStress.last, recentStress.level > 0.7 {
            factors.append("High stress levels")
        }
        
        if data.seasonalFactor > 0.6 {
            factors.append("Seasonal effects")
        }
        
        return factors
    }
    
    private func generateRecommendations(painLevel: Double, data: PredictionInputData) -> [String] {
        var recommendations: [String] = []
        
        if painLevel > 7.0 {
            recommendations.append("Consider taking prescribed pain medication")
            recommendations.append("Apply heat or cold therapy")
            recommendations.append("Practice gentle stretching or yoga")
        } else if painLevel > 5.0 {
            recommendations.append("Stay active with light exercise")
            recommendations.append("Practice stress reduction techniques")
        }
        
        if let weather = data.currentWeather, weather.pressure < 1013.25 {
            recommendations.append("Weather may affect symptoms - stay warm and dry")
        }
        
        if let recentSleep = data.recentSleep.last, recentSleep.quality < 0.6 {
            recommendations.append("Focus on improving sleep quality tonight")
        }
        
        return recommendations
    }
    
    private func calculatePainTrend(_ painLevels: [Double]) -> Double {
        guard painLevels.count >= 2 else { return 0.0 }
        
        // Simple linear regression to calculate trend
        let n = Double(painLevels.count)
        let x = Array(0..<painLevels.count).map(Double.init)
        let y = painLevels
        
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumXX = x.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        
        return slope
    }
    
    private func preparePredictionFeatures(data: PredictionInputData) -> [String: Double] {
        var features: [String: Double] = [:]
        
        // Weather features
        if let weather = data.currentWeather {
            features["temperature"] = weather.temperature
            features["humidity"] = weather.humidity
            features["pressure"] = weather.pressure
            features["wind_speed"] = weather.windSpeed
            features["precipitation_chance"] = weather.precipitationChance
        }
        
        // Recent pain features
        if !data.recentPainLevels.isEmpty {
            features["avg_recent_pain"] = data.recentPainLevels.reduce(0, +) / Double(data.recentPainLevels.count)
            features["pain_trend"] = calculatePainTrend(data.recentPainLevels)
            features["max_recent_pain"] = data.recentPainLevels.max() ?? 5.0
        }
        
        // Sleep features
        if let recentSleep = data.recentSleep.last {
            features["sleep_quality"] = recentSleep.quality
            features["sleep_duration"] = recentSleep.duration / 3600.0 // Convert to hours
        }
        
        // Stress features
        if let recentStress = data.recentStress.last {
            features["stress_level"] = recentStress.level
            features["heart_rate_variability"] = recentStress.heartRateVariability
        }
        
        // Activity features
        if let recentActivity = data.recentActivity.last {
            features["step_count"] = Double(recentActivity.stepCount)
            features["active_energy"] = Double(recentActivity.activeEnergyBurned)
        }
        
        // Temporal features
        features["time_of_day"] = Double(data.timeOfDay)
        features["day_of_week"] = Double(data.dayOfWeek)
        features["seasonal_factor"] = data.seasonalFactor
        
        return features
    }
    
    private func detectFlareWithMLModel(_ model: MLModel, data: PredictionInputData) async -> FlareRiskAssessment {
        // Implementation similar to pain prediction but focused on flare detection
        return FlareRiskAssessment(
            riskLevel: .moderate,
            confidence: 0.7,
            timeToFlare: 48 * 60 * 60, // 48 hours
            riskFactors: await analyzeRiskFactors(data: data),
            preventiveActions: generatePreventiveActions()
        )
    }
    
    private func detectFlareWithRules(data: PredictionInputData) async -> FlareRiskAssessment {
        var riskScore = 0.0
        let riskFactors = await analyzeRiskFactors(data: data)
        
        // Calculate risk based on factors
        for factor in riskFactors {
            riskScore += factor.impact
        }
        
        let riskLevel: FlareRiskLevel
        if riskScore < 0.3 {
            riskLevel = .low
        } else if riskScore < 0.7 {
            riskLevel = .moderate
        } else {
            riskLevel = .high
        }
        
        return FlareRiskAssessment(
            riskLevel: riskLevel,
            confidence: 0.6,
            timeToFlare: riskLevel == .high ? 24 * 60 * 60 : 72 * 60 * 60,
            riskFactors: riskFactors,
            preventiveActions: generatePreventiveActions()
        )
    }
    
    private func generatePreventiveActions() -> [String] {
        return [
            "Ensure adequate rest and sleep",
            "Take medications as prescribed",
            "Practice stress reduction techniques",
            "Maintain gentle, regular exercise",
            "Stay hydrated and eat anti-inflammatory foods",
            "Monitor symptoms closely"
        ]
    }
    
    private func updateMLModels() async {
        // This would retrain the models with new data
        logger.info("Updating ML models with new data")
        
        // Implementation would involve CreateML to retrain models
        // For now, just log the action
    }
    
    private func startPeriodicAnalysis() {
        Timer.scheduledTimer(withTimeInterval: modelUpdateInterval, repeats: true) { _ in
            Task {
                await self.generatePainPrediction()
            }
        }
    }
    
    // MARK: - Pattern Analysis Methods
    
    private func analyzeTemporalPatterns() -> [TemporalPattern] {
        var patterns: [TemporalPattern] = []
        
        // Analyze hourly patterns
        var hourlyPain: [Int: [Double]] = [:]
        for entry in painHistory {
            let hour = Calendar.current.component(.hour, from: entry.date)
            hourlyPain[hour, default: []].append(entry.painLevel)
        }
        
        for (hour, painLevels) in hourlyPain {
            let avgPain = painLevels.reduce(0, +) / Double(painLevels.count)
            patterns.append(TemporalPattern(
                type: .hourly,
                value: hour,
                averagePain: avgPain,
                sampleSize: painLevels.count
            ))
        }
        
        // Analyze daily patterns
        var dailyPain: [Int: [Double]] = [:]
        for entry in painHistory {
            let dayOfWeek = Calendar.current.component(.weekday, from: entry.date)
            dailyPain[dayOfWeek, default: []].append(entry.painLevel)
        }
        
        for (day, painLevels) in dailyPain {
            let avgPain = painLevels.reduce(0, +) / Double(painLevels.count)
            patterns.append(TemporalPattern(
                type: .daily,
                value: day,
                averagePain: avgPain,
                sampleSize: painLevels.count
            ))
        }
        
        return patterns.filter { $0.sampleSize >= 3 }
    }
    
    private func analyzeWeatherCorrelations() -> [WeatherCorrelation] {
        var correlations: [WeatherCorrelation] = []
        
        // Match pain entries with weather data
        let matchedData = matchPainWithWeather()
        
        if !matchedData.isEmpty {
            // Calculate correlation for each weather parameter
            let pressureCorr = calculateCorrelation(
                matchedData.map { $0.weather.pressure },
                matchedData.map { $0.pain.painLevel }
            )
            
            correlations.append(WeatherCorrelation(
                parameter: .pressure,
                correlation: pressureCorr,
                significance: abs(pressureCorr) > 0.3 ? .significant : .weak
            ))
            
            let humidityCorr = calculateCorrelation(
                matchedData.map { $0.weather.humidity },
                matchedData.map { $0.pain.painLevel }
            )
            
            correlations.append(WeatherCorrelation(
                parameter: .humidity,
                correlation: humidityCorr,
                significance: abs(humidityCorr) > 0.3 ? .significant : .weak
            ))
        }
        
        return correlations
    }
    
    private func analyzeActivityCorrelations() -> [ActivityCorrelation] {
        // Similar implementation for activity correlations
        return []
    }
    
    private func analyzeSleepCorrelations() -> [SleepCorrelation] {
        // Similar implementation for sleep correlations
        return []
    }
    
    private func analyzeStressCorrelations() -> [StressCorrelation] {
        // Similar implementation for stress correlations
        return []
    }
    
    private func matchPainWithWeather() -> [(pain: PainEntry, weather: WeatherData)] {
        var matches: [(pain: PainEntry, weather: WeatherData)] = []
        
        for painEntry in painHistory {
            // Find weather data within 2 hours of pain entry
            if let weatherData = weatherHistory.first(where: {
                abs($0.date.timeIntervalSince(painEntry.date)) < 2 * 60 * 60
            }) {
                matches.append((pain: painEntry, weather: weatherData))
            }
        }
        
        return matches
    }
    
    private func calculateCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count && x.count > 1 else { return 0.0 }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumXX = x.map { $0 * $0 }.reduce(0, +)
        let sumYY = y.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY))
        
        return denominator != 0 ? numerator / denominator : 0.0
    }
}

// MARK: - Supporting Types

struct PainEntry: Codable {
    let date: Date
    let painLevel: Double
    let location: String?
    let triggers: [String]?
    let medications: [String]?
}

struct WeatherData: Codable {
    let date: Date
    let temperature: Double
    let humidity: Double
    let pressure: Double
    let windSpeed: Double
    let precipitationChance: Double
    let uvIndex: Double
}

struct ActivityData: Codable {
    let date: Date
    let stepCount: Int
    let activeEnergyBurned: Int
    let exerciseMinutes: Int
}

struct SleepData: Codable {
    let date: Date
    let duration: TimeInterval
    let quality: Double
    let efficiency: Double
}

struct StressData: Codable {
    let date: Date
    let level: Double
    let heartRateVariability: Double
}

struct PredictionInputData {
    let currentWeather: WeatherData?
    let recentPainLevels: [Double]
    let recentActivity: [ActivityData]
    let recentSleep: [SleepData]
    let recentStress: [StressData]
    let timeOfDay: Int
    let dayOfWeek: Int
    let seasonalFactor: Double
}

struct PainPrediction {
    let date: Date
    let predictedPainLevel: Double
    let confidence: Double
    let flareRisk: Double
    let contributingFactors: [String]
    let recommendations: [String]
}

struct RiskFactor {
    let type: RiskFactorType
    let severity: RiskSeverity
    let description: String
    let impact: Double
}

enum RiskFactorType {
    case weather
    case sleep
    case stress
    case activity
    case medication
    case seasonal
}

enum RiskSeverity {
    case mild
    case moderate
    case high
}

struct FlareRiskAssessment {
    let riskLevel: FlareRiskLevel
    let confidence: Double
    let timeToFlare: TimeInterval
    let riskFactors: [RiskFactor]
    let preventiveActions: [String]
}

enum FlareRiskLevel {
    case low
    case moderate
    case high
}

struct PatternAnalysisResult {
    let temporalPatterns: [TemporalPattern]
    let weatherCorrelations: [WeatherCorrelation]
    let activityCorrelations: [ActivityCorrelation]
    let sleepCorrelations: [SleepCorrelation]
    let stressCorrelations: [StressCorrelation]
    let analysisDate: Date
}

struct TemporalPattern {
    let type: TemporalPatternType
    let value: Int // Hour of day or day of week
    let averagePain: Double
    let sampleSize: Int
}

enum TemporalPatternType {
    case hourly
    case daily
    case monthly
}

struct WeatherCorrelation {
    let parameter: WeatherParameter
    let correlation: Double
    let significance: CorrelationSignificance
}

enum WeatherParameter {
    case temperature
    case humidity
    case pressure
    case windSpeed
    case precipitation
}

enum CorrelationSignificance {
    case weak
    case moderate
    case significant
}

struct ActivityCorrelation {
    let parameter: ActivityParameter
    let correlation: Double
    let significance: CorrelationSignificance
}

enum ActivityParameter {
    case stepCount
    case activeEnergy
    case exerciseMinutes
}

struct SleepCorrelation {
    let parameter: SleepParameter
    let correlation: Double
    let significance: CorrelationSignificance
}

enum SleepParameter {
    case duration
    case quality
    case efficiency
}

struct StressCorrelation {
    let parameter: StressParameter
    let correlation: Double
    let significance: CorrelationSignificance
}

enum StressParameter {
    case level
    case heartRateVariability
}