//
//  MLModelManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreML
import CreateML
import Combine

// MARK: - ML Model Manager

class MLModelManager: ObservableObject {
    // MARK: - Properties
    
    private var painPredictionModel: MLModel?
    private var flareDetectionModel: MLModel?
    private var correlationAnalysisModel: MLModel?
    private let modelUpdateQueue = DispatchQueue(label: "ml.model.update", qos: .background)
    
    // MARK: - Initialization
    
    init() {
        loadModels()
    }
    
    // MARK: - Public Methods
    
    func generatePainPredictions(historicalPain: [PainEntry], context: AnalysisContext) async -> [PainPrediction] {
        var predictions: [PainPrediction] = []
        
        // Generate predictions for the next 7 days
        let calendar = Calendar.current
        let today = Date()
        
        for dayOffset in 1...7 {
            guard let targetDate = calendar.date(byAdding: .day, value: dayOffset, to: today) else { continue }
            
            let features = createPredictionFeatures(
                targetDate: targetDate,
                historicalPain: historicalPain,
                context: context
            )
            
            let prediction = await predictPainLevel(features: features)
            predictions.append(prediction)
        }
        
        return predictions
    }
    
    func predictPain(features: PredictionFeatures) async throws -> PainPrediction {
        let mlFeatures = convertToMLFeatures(features)
        
        // Use simple heuristic model if ML model is not available
        if painPredictionModel == nil {
            return generateHeuristicPainPrediction(features: features)
        }
        
        // TODO: Implement actual ML model prediction
        return generateHeuristicPainPrediction(features: features)
    }
    
    func analyzeCorrelations(painData: [PainEntry], context: AnalysisContext) async -> [CorrelationInsight] {
        var insights: [CorrelationInsight] = []
        
        // Analyze medication correlations
        insights.append(contentsOf: analyzeMedicationCorrelations(painData: painData, medications: context.medications))
        
        // Analyze activity correlations
        if let activities = context.activities {
            insights.append(contentsOf: analyzeActivityCorrelations(painData: painData, activities: activities))
        }
        
        // Analyze sleep correlations
        if let sleepData = context.sleep {
            insights.append(contentsOf: analyzeSleepCorrelations(painData: painData, sleepData: sleepData))
        }
        
        // Analyze weather correlations
        if let weatherData = context.weather {
            insights.append(contentsOf: analyzeWeatherCorrelations(painData: painData, weatherData: weatherData))
        }
        
        return insights.sorted { $0.correlationStrength > $1.correlationStrength }
    }
    
    func assessFlareRisk(from painData: [PainEntry]) async -> FlareRiskLevel {
        guard !painData.isEmpty else { return .low }
        
        let recentPain = painData.suffix(7) // Last 7 entries
        let avgRecentPain = recentPain.map { $0.painLevel }.reduce(0, +) / Double(recentPain.count)
        
        // Calculate pain trend
        let painTrend = calculatePainTrend(painData: Array(recentPain))
        
        // Calculate variability
        let painVariability = calculatePainVariability(painData: Array(recentPain))
        
        // Risk assessment based on multiple factors
        let riskScore = calculateFlareRiskScore(
            averagePain: avgRecentPain,
            trend: painTrend,
            variability: painVariability
        )
        
        return mapRiskScoreToLevel(riskScore)
    }
    
    func detectFlarePatterns(from painData: [PainEntry]) async -> [FlarePattern] {
        var patterns: [FlarePattern] = []
        
        // Detect seasonal patterns
        if let seasonalPattern = detectSeasonalPattern(painData: painData) {
            patterns.append(seasonalPattern)
        }
        
        // Detect weekly patterns
        if let weeklyPattern = detectWeeklyPattern(painData: painData) {
            patterns.append(weeklyPattern)
        }
        
        // Detect monthly patterns
        if let monthlyPattern = detectMonthlyPattern(painData: painData) {
            patterns.append(monthlyPattern)
        }
        
        return patterns
    }
    
    func generateMedicationOptimizations(painData: [PainEntry], medicationData: [MedicationEntry]) async -> [MedicationOptimization] {
        var optimizations: [MedicationOptimization] = []
        
        // Group medications by type
        let medicationsByType = Dictionary(grouping: medicationData) { $0.medicationType }
        
        for (medicationType, medications) in medicationsByType {
            if let optimization = analyzeMedicationEffectiveness(
                medicationType: medicationType,
                medications: medications,
                painData: painData
            ) {
                optimizations.append(optimization)
            }
        }
        
        return optimizations
    }
    
    func optimizeMedicationSchedule(painData: [PainEntry], medicationData: [MedicationEntry]) async -> [MedicationOptimization] {
        return await generateMedicationOptimizations(painData: painData, medicationData: medicationData)
    }
    
    func generatePersonalizedRecommendations(userProfile: UserProfile?, recentData: RecentUserData) async -> [PersonalizedRecommendation] {
        var recommendations: [PersonalizedRecommendation] = []
        
        // Analyze recent pain patterns
        let avgPain = recentData.painEntries.map { $0.painLevel }.reduce(0, +) / Double(max(recentData.painEntries.count, 1))
        
        // Generate exercise recommendations
        if let exerciseRec = generateExerciseRecommendation(avgPain: avgPain, activityData: recentData.activityData) {
            recommendations.append(exerciseRec)
        }
        
        // Generate sleep recommendations
        if let sleepRec = generateSleepRecommendation(sleepData: recentData.sleepData) {
            recommendations.append(sleepRec)
        }
        
        // Generate medication recommendations
        if let medicationRec = generateMedicationRecommendation(
            painData: recentData.painEntries,
            medicationData: recentData.medicationEntries
        ) {
            recommendations.append(medicationRec)
        }
        
        // Generate lifestyle recommendations
        recommendations.append(contentsOf: generateLifestyleRecommendations(avgPain: avgPain))
        
        return recommendations.sorted { $0.priority.rawValue > $1.priority.rawValue }
    }
    
    func trainModel(with data: [PainEntry]) async {
        // This would implement model training in a real scenario
        // For now, we'll simulate model updates
        await updateModelWeights(based: data)
    }
    
    // MARK: - Private Methods
    
    private func loadModels() {
        modelUpdateQueue.async { [weak self] in
            // Load pre-trained models if available
            self?.loadPainPredictionModel()
            self?.loadFlareDetectionModel()
            self?.loadCorrelationAnalysisModel()
        }
    }
    
    private func loadPainPredictionModel() {
        // In a real implementation, load from bundle or download from server
        // For now, we'll use heuristic-based predictions
    }
    
    private func loadFlareDetectionModel() {
        // Load flare detection model
    }
    
    private func loadCorrelationAnalysisModel() {
        // Load correlation analysis model
    }
    
    private func createPredictionFeatures(targetDate: Date, historicalPain: [PainEntry], context: AnalysisContext) -> PredictionFeatures {
        let calendar = Calendar.current
        let recentPain = historicalPain.suffix(7)
        
        return PredictionFeatures(
            date: targetDate,
            recentPainLevels: recentPain.map { $0.painLevel },
            recentMedications: context.medications.suffix(7).map { $0 },
            weatherConditions: context.weather?.last,
            sleepQuality: context.sleep?.last?.quality ?? 0.5,
            dayOfWeek: calendar.component(.weekday, from: targetDate),
            timeOfYear: calendar.dayOfYear(for: targetDate) ?? 1
        )
    }
    
    private func convertToMLFeatures(_ features: PredictionFeatures) -> [String: Any] {
        return [
            "avgRecentPain": features.recentPainLevels.reduce(0, +) / Double(max(features.recentPainLevels.count, 1)),
            "medicationCount": features.recentMedications.count,
            "sleepQuality": features.sleepQuality,
            "dayOfWeek": features.dayOfWeek,
            "timeOfYear": features.timeOfYear,
            "barometricPressure": features.weatherConditions?.barometricPressure ?? 1013.25,
            "humidity": features.weatherConditions?.humidity ?? 50.0,
            "temperature": features.weatherConditions?.temperature ?? 20.0
        ]
    }
    
    private func generateHeuristicPainPrediction(features: PredictionFeatures) -> PainPrediction {
        let avgRecentPain = features.recentPainLevels.reduce(0, +) / Double(max(features.recentPainLevels.count, 1))
        
        // Apply various factors
        var predictedPain = avgRecentPain
        var factors: [PredictionFactor] = []
        
        // Weather impact
        if let weather = features.weatherConditions {
            let pressureImpact = (1013.25 - weather.barometricPressure) / 100.0
            predictedPain += pressureImpact * 0.5
            
            factors.append(PredictionFactor(
                name: "Barometric Pressure",
                impact: pressureImpact,
                description: "Low pressure systems may increase joint pain"
            ))
        }
        
        // Sleep impact
        let sleepImpact = (0.8 - features.sleepQuality) * 2.0
        predictedPain += sleepImpact
        
        factors.append(PredictionFactor(
            name: "Sleep Quality",
            impact: sleepImpact,
            description: "Poor sleep quality can increase pain sensitivity"
        ))
        
        // Medication impact
        let medicationImpact = -Double(features.recentMedications.count) * 0.3
        predictedPain += medicationImpact
        
        factors.append(PredictionFactor(
            name: "Medication Adherence",
            impact: medicationImpact,
            description: "Regular medication can help manage pain levels"
        ))
        
        // Day of week impact (weekends might be different)
        let weekendImpact = (features.dayOfWeek == 1 || features.dayOfWeek == 7) ? -0.2 : 0.1
        predictedPain += weekendImpact
        
        factors.append(PredictionFactor(
            name: "Day of Week",
            impact: weekendImpact,
            description: "Stress levels may vary by day of week"
        ))
        
        // Clamp prediction to valid range
        predictedPain = max(0, min(10, predictedPain))
        
        let recommendation = generatePainRecommendation(predictedPain: predictedPain, factors: factors)
        
        return PainPrediction(
            date: features.date,
            predictedPainLevel: predictedPain,
            confidence: 0.7,
            factors: factors,
            recommendation: recommendation
        )
    }
    
    private func predictPainLevel(features: PredictionFeatures) async -> PainPrediction {
        return generateHeuristicPainPrediction(features: features)
    }
    
    private func analyzeMedicationCorrelations(painData: [PainEntry], medications: [MedicationEntry]) -> [CorrelationInsight] {
        var insights: [CorrelationInsight] = []
        
        let medicationsByType = Dictionary(grouping: medications) { $0.medicationType }
        
        for (medicationType, meds) in medicationsByType {
            let correlation = calculateMedicationPainCorrelation(painData: painData, medications: meds)
            
            if abs(correlation) > 0.3 {
                insights.append(CorrelationInsight(
                    id: UUID(),
                    type: .medicationEffectiveness,
                    title: "\(medicationType) Effectiveness",
                    description: correlation < 0 ? "\(medicationType) appears to reduce pain levels" : "\(medicationType) correlation with pain needs review",
                    correlationStrength: abs(correlation),
                    confidence: 0.8,
                    actionable: true,
                    recommendation: correlation < 0 ? "Continue current \(medicationType) regimen" : "Discuss \(medicationType) effectiveness with your doctor"
                ))
            }
        }
        
        return insights
    }
    
    private func analyzeActivityCorrelations(painData: [PainEntry], activities: [ActivityData]) -> [CorrelationInsight] {
        var insights: [CorrelationInsight] = []
        
        let correlation = calculateActivityPainCorrelation(painData: painData, activities: activities)
        
        if abs(correlation) > 0.3 {
            insights.append(CorrelationInsight(
                id: UUID(),
                type: .activityPattern,
                title: "Physical Activity Impact",
                description: correlation > 0 ? "Higher activity levels correlate with increased pain" : "Higher activity levels correlate with reduced pain",
                correlationStrength: abs(correlation),
                confidence: 0.7,
                actionable: true,
                recommendation: correlation > 0 ? "Consider gentle, low-impact exercises" : "Maintain regular physical activity"
            ))
        }
        
        return insights
    }
    
    private func analyzeSleepCorrelations(painData: [PainEntry], sleepData: [SleepData]) -> [CorrelationInsight] {
        var insights: [CorrelationInsight] = []
        
        let correlation = calculateSleepPainCorrelation(painData: painData, sleepData: sleepData)
        
        if abs(correlation) > 0.3 {
            insights.append(CorrelationInsight(
                id: UUID(),
                type: .sleepPattern,
                title: "Sleep Quality Impact",
                description: correlation > 0 ? "Better sleep quality correlates with higher pain" : "Better sleep quality correlates with lower pain",
                correlationStrength: abs(correlation),
                confidence: 0.8,
                actionable: true,
                recommendation: correlation < 0 ? "Prioritize good sleep hygiene" : "Monitor sleep patterns and pain relationship"
            ))
        }
        
        return insights
    }
    
    private func analyzeWeatherCorrelations(painData: [PainEntry], weatherData: [WeatherData]) -> [CorrelationInsight] {
        var insights: [CorrelationInsight] = []
        
        // Analyze barometric pressure
        let pressureCorrelation = calculateWeatherPainCorrelation(
            painData: painData,
            weatherData: weatherData,
            weatherProperty: \.barometricPressure
        )
        
        if abs(pressureCorrelation) > 0.3 {
            insights.append(CorrelationInsight(
                id: UUID(),
                type: .weatherPattern,
                title: "Barometric Pressure Impact",
                description: pressureCorrelation > 0 ? "Higher pressure correlates with increased pain" : "Lower pressure correlates with increased pain",
                correlationStrength: abs(pressureCorrelation),
                confidence: 0.7,
                actionable: true,
                recommendation: "Monitor weather forecasts and prepare for pressure changes"
            ))
        }
        
        return insights
    }
    
    private func calculateMedicationPainCorrelation(painData: [PainEntry], medications: [MedicationEntry]) -> Double {
        let medicationDates = Set(medications.map { Calendar.current.startOfDay(for: $0.dateTaken) })
        let painByDate = Dictionary(grouping: painData) { Calendar.current.startOfDay(for: $0.date) }
        
        var correlationPairs: [(medication: Double, pain: Double)] = []
        
        for (date, painEntries) in painByDate {
            let avgPain = painEntries.map { $0.painLevel }.reduce(0, +) / Double(painEntries.count)
            let medicationTaken = medicationDates.contains(date) ? 1.0 : 0.0
            correlationPairs.append((medication: medicationTaken, pain: avgPain))
        }
        
        return calculatePearsonCorrelation(correlationPairs.map { ($0.medication, $0.pain) })
    }
    
    private func calculateActivityPainCorrelation(painData: [PainEntry], activities: [ActivityData]) -> Double {
        let activityByDate = Dictionary(uniqueKeysWithValues: activities.map { (Calendar.current.startOfDay(for: $0.date), $0.stepCount) })
        
        var correlationPairs: [(activity: Double, pain: Double)] = []
        
        for painEntry in painData {
            let date = Calendar.current.startOfDay(for: painEntry.date)
            if let stepCount = activityByDate[date] {
                correlationPairs.append((activity: stepCount, pain: painEntry.painLevel))
            }
        }
        
        return calculatePearsonCorrelation(correlationPairs.map { ($0.activity, $0.pain) })
    }
    
    private func calculateSleepPainCorrelation(painData: [PainEntry], sleepData: [SleepData]) -> Double {
        let sleepByDate = Dictionary(uniqueKeysWithValues: sleepData.map { (Calendar.current.startOfDay(for: $0.date), $0.quality) })
        
        var correlationPairs: [(sleep: Double, pain: Double)] = []
        
        for painEntry in painData {
            let date = Calendar.current.startOfDay(for: painEntry.date)
            if let sleepQuality = sleepByDate[date] {
                correlationPairs.append((sleep: sleepQuality, pain: painEntry.painLevel))
            }
        }
        
        return calculatePearsonCorrelation(correlationPairs.map { ($0.sleep, $0.pain) })
    }
    
    private func calculateWeatherPainCorrelation(painData: [PainEntry], weatherData: [WeatherData], weatherProperty: KeyPath<WeatherData, Double>) -> Double {
        let weatherByDate = Dictionary(uniqueKeysWithValues: weatherData.map { (Calendar.current.startOfDay(for: $0.date), $0[keyPath: weatherProperty]) })
        
        var correlationPairs: [(weather: Double, pain: Double)] = []
        
        for painEntry in painData {
            let date = Calendar.current.startOfDay(for: painEntry.date)
            if let weatherValue = weatherByDate[date] {
                correlationPairs.append((weather: weatherValue, pain: painEntry.painLevel))
            }
        }
        
        return calculatePearsonCorrelation(correlationPairs.map { ($0.weather, $0.pain) })
    }
    
    private func calculatePearsonCorrelation(_ pairs: [(Double, Double)]) -> Double {
        let n = Double(pairs.count)
        guard n > 1 else { return 0.0 }
        
        let sumX = pairs.map { $0.0 }.reduce(0, +)
        let sumY = pairs.map { $0.1 }.reduce(0, +)
        let sumXY = pairs.map { $0.0 * $0.1 }.reduce(0, +)
        let sumX2 = pairs.map { $0.0 * $0.0 }.reduce(0, +)
        let sumY2 = pairs.map { $0.1 * $0.1 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        guard denominator != 0 else { return 0.0 }
        
        return numerator / denominator
    }
    
    private func calculatePainTrend(painData: [PainEntry]) -> Double {
        guard painData.count > 1 else { return 0.0 }
        
        let sortedData = painData.sorted { $0.date < $1.date }
        let firstHalf = sortedData.prefix(sortedData.count / 2)
        let secondHalf = sortedData.suffix(sortedData.count / 2)
        
        let firstAvg = firstHalf.map { $0.painLevel }.reduce(0, +) / Double(firstHalf.count)
        let secondAvg = secondHalf.map { $0.painLevel }.reduce(0, +) / Double(secondHalf.count)
        
        return secondAvg - firstAvg
    }
    
    private func calculatePainVariability(painData: [PainEntry]) -> Double {
        guard !painData.isEmpty else { return 0.0 }
        
        let painLevels = painData.map { $0.painLevel }
        let mean = painLevels.reduce(0, +) / Double(painLevels.count)
        let variance = painLevels.map { pow($0 - mean, 2) }.reduce(0, +) / Double(painLevels.count)
        
        return sqrt(variance)
    }
    
    private func calculateFlareRiskScore(averagePain: Double, trend: Double, variability: Double) -> Double {
        // Weighted risk calculation
        let painWeight = 0.4
        let trendWeight = 0.4
        let variabilityWeight = 0.2
        
        let painScore = averagePain / 10.0 // Normalize to 0-1
        let trendScore = max(0, trend / 5.0) // Positive trend increases risk
        let variabilityScore = variability / 5.0 // High variability increases risk
        
        return painWeight * painScore + trendWeight * trendScore + variabilityWeight * variabilityScore
    }
    
    private func mapRiskScoreToLevel(_ score: Double) -> FlareRiskLevel {
        switch score {
        case 0.0..<0.25:
            return .low
        case 0.25..<0.5:
            return .moderate
        case 0.5..<0.75:
            return .high
        default:
            return .critical
        }
    }
    
    private func detectSeasonalPattern(painData: [PainEntry]) -> FlarePattern? {
        // Group pain data by season
        let calendar = Calendar.current
        let painByMonth = Dictionary(grouping: painData) { calendar.component(.month, from: $0.date) }
        
        var seasonalAverages: [String: Double] = [:]
        
        // Calculate seasonal averages
        let seasons = [
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11]
        ]
        
        for (season, months) in seasons {
            let seasonPain = months.compactMap { painByMonth[$0] }.flatMap { $0 }
            if !seasonPain.isEmpty {
                seasonalAverages[season] = seasonPain.map { $0.painLevel }.reduce(0, +) / Double(seasonPain.count)
            }
        }
        
        // Find the season with highest pain
        guard let maxSeason = seasonalAverages.max(by: { $0.value < $1.value }) else { return nil }
        
        // Check if the difference is significant
        let avgPain = seasonalAverages.values.reduce(0, +) / Double(seasonalAverages.count)
        if maxSeason.value - avgPain > 1.0 {
            return FlarePattern(
                patternType: .seasonal,
                frequency: "Seasonal",
                triggers: ["\(maxSeason.key) weather conditions"],
                duration: "3 months",
                severity: "Moderate to High",
                confidence: 0.7
            )
        }
        
        return nil
    }
    
    private func detectWeeklyPattern(painData: [PainEntry]) -> FlarePattern? {
        let calendar = Calendar.current
        let painByWeekday = Dictionary(grouping: painData) { calendar.component(.weekday, from: $0.date) }
        
        var weekdayAverages: [Int: Double] = [:]
        
        for (weekday, entries) in painByWeekday {
            weekdayAverages[weekday] = entries.map { $0.painLevel }.reduce(0, +) / Double(entries.count)
        }
        
        guard let maxWeekday = weekdayAverages.max(by: { $0.value < $1.value }) else { return nil }
        
        let avgPain = weekdayAverages.values.reduce(0, +) / Double(weekdayAverages.count)
        if maxWeekday.value - avgPain > 1.0 {
            let dayName = calendar.weekdaySymbols[maxWeekday.key - 1]
            return FlarePattern(
                patternType: .weekly,
                frequency: "Weekly",
                triggers: ["\(dayName) stress patterns"],
                duration: "1 day",
                severity: "Moderate",
                confidence: 0.6
            )
        }
        
        return nil
    }
    
    private func detectMonthlyPattern(painData: [PainEntry]) -> FlarePattern? {
        // Simplified monthly pattern detection
        // In a real implementation, this would be more sophisticated
        return nil
    }
    
    private func analyzeMedicationEffectiveness(medicationType: String, medications: [MedicationEntry], painData: [PainEntry]) -> MedicationOptimization? {
        let medicationDates = Set(medications.map { Calendar.current.startOfDay(for: $0.dateTaken) })
        let painByDate = Dictionary(grouping: painData) { Calendar.current.startOfDay(for: $0.date) }
        
        var medicatedPain: [Double] = []
        var nonMedicatedPain: [Double] = []
        
        for (date, painEntries) in painByDate {
            let avgPain = painEntries.map { $0.painLevel }.reduce(0, +) / Double(painEntries.count)
            
            if medicationDates.contains(date) {
                medicatedPain.append(avgPain)
            } else {
                nonMedicatedPain.append(avgPain)
            }
        }
        
        guard !medicatedPain.isEmpty && !nonMedicatedPain.isEmpty else { return nil }
        
        let medicatedAvg = medicatedPain.reduce(0, +) / Double(medicatedPain.count)
        let nonMedicatedAvg = nonMedicatedPain.reduce(0, +) / Double(nonMedicatedPain.count)
        
        let improvement = nonMedicatedAvg - medicatedAvg
        
        if improvement > 0.5 {
            return MedicationOptimization(
                medicationType: medicationType,
                currentSchedule: "As needed",
                suggestedSchedule: "Regular schedule",
                expectedImprovement: improvement,
                confidence: 0.7,
                reasoning: "\(medicationType) shows effectiveness in reducing pain levels"
            )
        }
        
        return nil
    }
    
    private func generateExerciseRecommendation(avgPain: Double, activityData: [ActivityData]?) -> PersonalizedRecommendation? {
        let avgSteps = activityData?.map { $0.stepCount }.reduce(0, +) ?? 0
        let dailyAvgSteps = avgSteps / Double(max(activityData?.count ?? 1, 1))
        
        if avgPain > 6.0 && dailyAvgSteps < 5000 {
            return PersonalizedRecommendation(
                category: .exercise,
                title: "Gentle Movement Therapy",
                description: "Low-impact exercises can help manage pain and improve mobility",
                priority: .high,
                actionSteps: [
                    "Start with 10-minute gentle walks",
                    "Try water-based exercises",
                    "Practice gentle stretching",
                    "Consider tai chi or yoga"
                ],
                expectedBenefit: "Reduced pain and improved joint mobility",
                timeframe: "2-4 weeks"
            )
        }
        
        return nil
    }
    
    private func generateSleepRecommendation(sleepData: [SleepData]?) -> PersonalizedRecommendation? {
        guard let sleepData = sleepData, !sleepData.isEmpty else { return nil }
        
        let avgSleepQuality = sleepData.map { $0.quality }.reduce(0, +) / Double(sleepData.count)
        
        if avgSleepQuality < 0.6 {
            return PersonalizedRecommendation(
                category: .sleep,
                title: "Sleep Quality Improvement",
                description: "Better sleep can significantly reduce pain sensitivity",
                priority: .high,
                actionSteps: [
                    "Maintain consistent sleep schedule",
                    "Create a relaxing bedtime routine",
                    "Limit screen time before bed",
                    "Consider meditation or relaxation techniques"
                ],
                expectedBenefit: "Improved pain management and energy levels",
                timeframe: "1-2 weeks"
            )
        }
        
        return nil
    }
    
    private func generateMedicationRecommendation(painData: [PainEntry], medicationData: [MedicationEntry]) -> PersonalizedRecommendation? {
        let avgPain = painData.map { $0.painLevel }.reduce(0, +) / Double(max(painData.count, 1))
        let medicationDays = Set(medicationData.map { Calendar.current.startOfDay(for: $0.dateTaken) }).count
        let totalDays = max(Calendar.current.dateInterval(of: .month, for: Date())?.duration ?? 2592000, 2592000) / 86400
        let adherenceRate = Double(medicationDays) / totalDays
        
        if avgPain > 5.0 && adherenceRate < 0.8 {
            return PersonalizedRecommendation(
                category: .medication,
                title: "Medication Adherence",
                description: "Consistent medication use is crucial for pain management",
                priority: .high,
                actionSteps: [
                    "Set daily medication reminders",
                    "Use a pill organizer",
                    "Track medication timing",
                    "Discuss concerns with your doctor"
                ],
                expectedBenefit: "Better pain control and reduced flare frequency",
                timeframe: "Immediate"
            )
        }
        
        return nil
    }
    
    private func generateLifestyleRecommendations(avgPain: Double) -> [PersonalizedRecommendation] {
        var recommendations: [PersonalizedRecommendation] = []
        
        if avgPain > 5.0 {
            recommendations.append(PersonalizedRecommendation(
                category: .stress,
                title: "Stress Management",
                description: "Stress can worsen rheumatoid arthritis symptoms",
                priority: .medium,
                actionSteps: [
                    "Practice deep breathing exercises",
                    "Try mindfulness meditation",
                    "Consider counseling or support groups",
                    "Engage in relaxing hobbies"
                ],
                expectedBenefit: "Reduced stress-related pain flares",
                timeframe: "2-4 weeks"
            ))
            
            recommendations.append(PersonalizedRecommendation(
                category: .diet,
                title: "Anti-Inflammatory Diet",
                description: "Certain foods can help reduce inflammation",
                priority: .medium,
                actionSteps: [
                    "Increase omega-3 rich foods",
                    "Add more colorful vegetables",
                    "Reduce processed foods",
                    "Consider Mediterranean diet patterns"
                ],
                expectedBenefit: "Reduced inflammation and pain levels",
                timeframe: "4-8 weeks"
            ))
        }
        
        return recommendations
    }
    
    private func generatePainRecommendation(predictedPain: Double, factors: [PredictionFactor]) -> String? {
        if predictedPain > 7.0 {
            return "High pain predicted. Consider taking preventive medication and planning a lighter day."
        } else if predictedPain > 5.0 {
            return "Moderate pain expected. Prepare comfort measures and avoid strenuous activities."
        } else if predictedPain < 3.0 {
            return "Low pain predicted. Good day for gentle exercise and activities."
        }
        
        return nil
    }
    
    private func updateModelWeights(based data: [PainEntry]) async {
        // Simulate model weight updates based on new data
        // In a real implementation, this would retrain or fine-tune the model
    }
}