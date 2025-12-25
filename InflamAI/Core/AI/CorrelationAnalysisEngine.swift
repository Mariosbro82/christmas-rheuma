//
//  CorrelationAnalysisEngine.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Combine
import Accelerate

// MARK: - Correlation Analysis Engine
@MainActor
class CorrelationAnalysisEngine: ObservableObject {
    
    // MARK: - Published Properties
    @Published var isAnalyzing = false
    @Published var lastAnalysis: CorrelationAnalysis?
    @Published var correlationMatrix: [[Double]] = []
    @Published var significantCorrelations: [CorrelationResult] = []
    @Published var analysisProgress: Double = 0.0
    
    // MARK: - Private Properties
    private var dataPoints: [CorrelationDataPoint] = []
    private let minimumDataPoints = 30
    private let significanceThreshold = 0.3
    private let analysisQueue = DispatchQueue(label: "correlation.analysis", qos: .userInitiated)
    
    // MARK: - Initialization
    init() {
        loadHistoricalData()
    }
    
    // MARK: - Public Methods
    
    func addDataPoint(_ dataPoint: CorrelationDataPoint) async {
        dataPoints.append(dataPoint)
        
        // Trigger analysis if we have enough data
        if dataPoints.count >= minimumDataPoints && dataPoints.count % 10 == 0 {
            await performCorrelationAnalysis()
        }
    }
    
    func performCorrelationAnalysis() async {
        guard dataPoints.count >= minimumDataPoints else {
            print("Insufficient data for correlation analysis")
            return
        }
        
        isAnalyzing = true
        analysisProgress = 0.0
        
        await withCheckedContinuation { continuation in
            analysisQueue.async {
                Task {
                    await self.runAnalysis()
                    continuation.resume()
                }
            }
        }
        
        isAnalyzing = false
    }
    
    func getCorrelationBetween(_ factor1: String, _ factor2: String) async -> Double? {
        guard let analysis = lastAnalysis else {
            await performCorrelationAnalysis()
            return lastAnalysis?.getCorrelation(between: factor1, and: factor2)
        }
        
        return analysis.getCorrelation(between: factor1, and: factor2)
    }
    
    func getStrongestCorrelations(for factor: String, limit: Int = 5) async -> [CorrelationResult] {
        guard let analysis = lastAnalysis else {
            await performCorrelationAnalysis()
            return lastAnalysis?.getStrongestCorrelations(for: factor, limit: limit) ?? []
        }
        
        return analysis.getStrongestCorrelations(for: factor, limit: limit)
    }
    
    // MARK: - Private Analysis Methods
    
    private func runAnalysis() async {
        let factors = extractFactors()
        let matrix = await calculateCorrelationMatrix(factors: factors)
        let results = await identifySignificantCorrelations(matrix: matrix, factors: factors)
        
        await MainActor.run {
            self.correlationMatrix = matrix
            self.significantCorrelations = results
            self.lastAnalysis = CorrelationAnalysis(
                timestamp: Date(),
                correlationMatrix: matrix,
                factors: factors,
                significantCorrelations: results,
                dataPointCount: self.dataPoints.count
            )
            self.analysisProgress = 1.0
        }
    }
    
    private func extractFactors() -> [String] {
        return [
            "painLevel",
            "weatherPressure",
            "temperature",
            "humidity",
            "sleepQuality",
            "stressLevel",
            "activityLevel",
            "medicationAdherence",
            "moodScore",
            "fatigueLevel",
            "jointStiffness",
            "inflammation"
        ]
    }
    
    private func calculateCorrelationMatrix(factors: [String]) async -> [[Double]] {
        let factorCount = factors.count
        var matrix = Array(repeating: Array(repeating: 0.0, count: factorCount), count: factorCount)
        
        await MainActor.run {
            self.analysisProgress = 0.2
        }
        
        for i in 0..<factorCount {
            for j in i..<factorCount {
                let correlation = calculatePearsonCorrelation(
                    x: getFactorValues(factors[i]),
                    y: getFactorValues(factors[j])
                )
                
                matrix[i][j] = correlation
                matrix[j][i] = correlation // Symmetric matrix
            }
            
            await MainActor.run {
                self.analysisProgress = 0.2 + (0.6 * Double(i) / Double(factorCount))
            }
        }
        
        return matrix
    }
    
    private func calculatePearsonCorrelation(x: [Double], y: [Double]) -> Double {
        guard x.count == y.count && x.count > 1 else { return 0.0 }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        guard denominator != 0 else { return 0.0 }
        
        return numerator / denominator
    }
    
    private func getFactorValues(_ factor: String) -> [Double] {
        return dataPoints.compactMap { dataPoint in
            switch factor {
            case "painLevel":
                return dataPoint.painLevel
            case "weatherPressure":
                return dataPoint.weatherPressure
            case "temperature":
                return dataPoint.temperature
            case "humidity":
                return dataPoint.humidity
            case "sleepQuality":
                return dataPoint.sleepQuality
            case "stressLevel":
                return dataPoint.stressLevel
            case "activityLevel":
                return dataPoint.activityLevel
            case "medicationAdherence":
                return dataPoint.medicationAdherence
            case "moodScore":
                return dataPoint.moodScore
            case "fatigueLevel":
                return dataPoint.fatigueLevel
            case "jointStiffness":
                return dataPoint.jointStiffness
            case "inflammation":
                return dataPoint.inflammation
            default:
                return nil
            }
        }
    }
    
    private func identifySignificantCorrelations(matrix: [[Double]], factors: [String]) async -> [CorrelationResult] {
        var results: [CorrelationResult] = []
        
        for i in 0..<factors.count {
            for j in (i+1)..<factors.count {
                let correlation = matrix[i][j]
                
                if abs(correlation) >= significanceThreshold {
                    let significance = calculateSignificance(correlation: correlation, sampleSize: dataPoints.count)
                    
                    results.append(CorrelationResult(
                        factor1: factors[i],
                        factor2: factors[j],
                        correlation: correlation,
                        significance: significance,
                        strength: getCorrelationStrength(correlation),
                        direction: correlation > 0 ? .positive : .negative,
                        interpretation: generateInterpretation(factor1: factors[i], factor2: factors[j], correlation: correlation)
                    ))
                }
            }
        }
        
        await MainActor.run {
            self.analysisProgress = 0.9
        }
        
        // Sort by absolute correlation strength
        results.sort { abs($0.correlation) > abs($1.correlation) }
        
        return results
    }
    
    private func calculateSignificance(correlation: Double, sampleSize: Int) -> Double {
        // Calculate t-statistic for correlation significance
        let df = Double(sampleSize - 2)
        let t = correlation * sqrt(df / (1 - correlation * correlation))
        
        // Simplified p-value calculation (in practice, use proper statistical library)
        let pValue = 2 * (1 - normalCDF(abs(t)))
        
        return 1 - pValue // Return significance as 1 - p-value
    }
    
    private func normalCDF(_ x: Double) -> Double {
        // Simplified normal CDF approximation
        return 0.5 * (1 + erf(x / sqrt(2)))
    }
    
    private func getCorrelationStrength(_ correlation: Double) -> CorrelationStrength {
        let absCorr = abs(correlation)
        
        switch absCorr {
        case 0.0..<0.3:
            return .weak
        case 0.3..<0.7:
            return .moderate
        case 0.7...1.0:
            return .strong
        default:
            return .weak
        }
    }
    
    private func generateInterpretation(factor1: String, factor2: String, correlation: Double) -> String {
        let strength = getCorrelationStrength(correlation)
        let direction = correlation > 0 ? "positive" : "negative"
        let strengthText = strength.rawValue.lowercased()
        
        let factor1Name = getFriendlyFactorName(factor1)
        let factor2Name = getFriendlyFactorName(factor2)
        
        if correlation > 0 {
            return "There is a \(strengthText) positive relationship between \(factor1Name) and \(factor2Name). As \(factor1Name) increases, \(factor2Name) tends to increase as well."
        } else {
            return "There is a \(strengthText) negative relationship between \(factor1Name) and \(factor2Name). As \(factor1Name) increases, \(factor2Name) tends to decrease."
        }
    }
    
    private func getFriendlyFactorName(_ factor: String) -> String {
        switch factor {
        case "painLevel":
            return "pain level"
        case "weatherPressure":
            return "atmospheric pressure"
        case "temperature":
            return "temperature"
        case "humidity":
            return "humidity"
        case "sleepQuality":
            return "sleep quality"
        case "stressLevel":
            return "stress level"
        case "activityLevel":
            return "activity level"
        case "medicationAdherence":
            return "medication adherence"
        case "moodScore":
            return "mood score"
        case "fatigueLevel":
            return "fatigue level"
        case "jointStiffness":
            return "joint stiffness"
        case "inflammation":
            return "inflammation level"
        default:
            return factor
        }
    }
    
    // MARK: - Advanced Analysis Methods
    
    func performTimeSeriesAnalysis() async -> TimeSeriesAnalysis? {
        guard dataPoints.count >= 50 else { return nil }
        
        let painLevels = dataPoints.map { $0.painLevel }
        let timestamps = dataPoints.map { $0.timestamp }
        
        let trend = calculateTrend(values: painLevels)
        let seasonality = detectSeasonality(values: painLevels, timestamps: timestamps)
        let volatility = calculateVolatility(values: painLevels)
        
        return TimeSeriesAnalysis(
            trend: trend,
            seasonality: seasonality,
            volatility: volatility,
            forecast: generateForecast(values: painLevels)
        )
    }
    
    private func calculateTrend(values: [Double]) -> TrendDirection {
        guard values.count > 1 else { return .stable }
        
        let firstHalf = Array(values.prefix(values.count / 2))
        let secondHalf = Array(values.suffix(values.count / 2))
        
        let firstAvg = firstHalf.reduce(0, +) / Double(firstHalf.count)
        let secondAvg = secondHalf.reduce(0, +) / Double(secondHalf.count)
        
        let difference = secondAvg - firstAvg
        
        if difference > 0.5 {
            return .increasing
        } else if difference < -0.5 {
            return .decreasing
        } else {
            return .stable
        }
    }
    
    private func detectSeasonality(values: [Double], timestamps: [Date]) -> SeasonalityPattern {
        // Simplified seasonality detection
        let calendar = Calendar.current
        var weeklyPattern: [Double] = Array(repeating: 0, count: 7)
        var weeklyCounts: [Int] = Array(repeating: 0, count: 7)
        
        for (value, timestamp) in zip(values, timestamps) {
            let weekday = calendar.component(.weekday, from: timestamp) - 1
            weeklyPattern[weekday] += value
            weeklyCounts[weekday] += 1
        }
        
        // Calculate averages
        for i in 0..<7 {
            if weeklyCounts[i] > 0 {
                weeklyPattern[i] /= Double(weeklyCounts[i])
            }
        }
        
        let maxValue = weeklyPattern.max() ?? 0
        let minValue = weeklyPattern.min() ?? 0
        
        if maxValue - minValue > 1.0 {
            return .weekly
        } else {
            return .none
        }
    }
    
    private func calculateVolatility(values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count - 1)
        
        return sqrt(variance)
    }
    
    private func generateForecast(values: [Double]) -> [Double] {
        // Simple moving average forecast
        guard values.count >= 7 else { return [] }
        
        let lastWeek = Array(values.suffix(7))
        let average = lastWeek.reduce(0, +) / Double(lastWeek.count)
        
        return Array(repeating: average, count: 7) // 7-day forecast
    }
    
    // MARK: - Data Management
    
    private func loadHistoricalData() {
        // In a real implementation, load from Core Data or other persistence
        generateSampleData()
    }
    
    private func generateSampleData() {
        let calendar = Calendar.current
        
        for i in 0..<100 {
            let date = Date().addingTimeInterval(-Double(i) * 86400) // Daily data for last 100 days
            
            let dataPoint = CorrelationDataPoint(
                timestamp: date,
                painLevel: Double.random(in: 1...10),
                weatherPressure: Double.random(in: 980...1040),
                temperature: Double.random(in: -10...35),
                humidity: Double.random(in: 20...90),
                sleepQuality: Double.random(in: 1...10),
                stressLevel: Double.random(in: 1...10),
                activityLevel: Double.random(in: 0...10),
                medicationAdherence: Double.random(in: 0.5...1.0),
                moodScore: Double.random(in: 1...10),
                fatigueLevel: Double.random(in: 1...10),
                jointStiffness: Double.random(in: 1...10),
                inflammation: Double.random(in: 1...10)
            )
            
            dataPoints.append(dataPoint)
        }
    }
}

// MARK: - Supporting Types

struct CorrelationDataPoint {
    let timestamp: Date
    let painLevel: Double
    let weatherPressure: Double
    let temperature: Double
    let humidity: Double
    let sleepQuality: Double
    let stressLevel: Double
    let activityLevel: Double
    let medicationAdherence: Double
    let moodScore: Double
    let fatigueLevel: Double
    let jointStiffness: Double
    let inflammation: Double
}

struct CorrelationAnalysis {
    let timestamp: Date
    let correlationMatrix: [[Double]]
    let factors: [String]
    let significantCorrelations: [CorrelationResult]
    let dataPointCount: Int
    
    func getCorrelation(between factor1: String, and factor2: String) -> Double? {
        guard let index1 = factors.firstIndex(of: factor1),
              let index2 = factors.firstIndex(of: factor2),
              index1 < correlationMatrix.count,
              index2 < correlationMatrix[index1].count else {
            return nil
        }
        
        return correlationMatrix[index1][index2]
    }
    
    func getStrongestCorrelations(for factor: String, limit: Int = 5) -> [CorrelationResult] {
        return significantCorrelations
            .filter { $0.factor1 == factor || $0.factor2 == factor }
            .prefix(limit)
            .map { $0 }
    }
}

struct CorrelationResult {
    let factor1: String
    let factor2: String
    let correlation: Double
    let significance: Double
    let strength: CorrelationStrength
    let direction: CorrelationDirection
    let interpretation: String
}

enum CorrelationStrength: String, CaseIterable {
    case weak = "Weak"
    case moderate = "Moderate"
    case strong = "Strong"
}

enum CorrelationDirection: String, CaseIterable {
    case positive = "Positive"
    case negative = "Negative"
}

struct TimeSeriesAnalysis {
    let trend: TrendDirection
    let seasonality: SeasonalityPattern
    let volatility: Double
    let forecast: [Double]
}

enum TrendDirection: String, CaseIterable {
    case increasing = "Increasing"
    case decreasing = "Decreasing"
    case stable = "Stable"
}

enum SeasonalityPattern: String, CaseIterable {
    case none = "None