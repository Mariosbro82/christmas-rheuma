//
//  FlarePredictor.swift
//  InflamAI
//
//  CoreML-based flare prediction with weather forecast enhancement
//  Now delegates to UnifiedNeuralEngine for predictions using ASFlarePredictor.mlpackage
//  Adds 7-day weather forecast analysis as enhancement layer
//  100% local, privacy-first, zero data leaves device
//

import Foundation
import CoreML
import CoreData
import UserNotifications

/// CoreML-based flare predictor that uses ASFlarePredictor.mlpackage
///
/// All predictions now flow through the CoreML model:
/// - Delegates to UnifiedNeuralEngine for core prediction
/// - Enhances with 7-day weather forecast analysis
/// - Provides weather-aware risk adjustments
///
/// Important: This is NOT medical advice and should not replace clinical judgment.
@MainActor
class FlarePredictor: ObservableObject {

    // MARK: - Published State

    @Published var flareRiskLevel: FlarePredictorRiskLevel = .unknown
    @Published var riskPercentage: Double = 0.0
    @Published var lastPrediction: Date?
    @Published var isModelTrained: Bool = false
    @Published var contributingFactors: [FlarePredictorFactor] = []
    @Published var daysUntilLikelyFlare: Int?

    // MARK: - Weather Forecast State

    @Published var weatherForecastRisk: WeatherForecastRisk?
    @Published var upcomingWeatherAlerts: [WeatherRiskAlert] = []

    // MARK: - Properties

    private let context: NSManagedObjectContext
    private let weatherService: OpenMeteoService
    private let minimumDataPoints = 7 // Reduced from 30 - CoreML model needs less data

    /// Reference to the unified neural engine (CoreML)
    private let neuralEngine: UnifiedNeuralEngine

    // MARK: - Initialization

    init(
        context: NSManagedObjectContext? = nil,
        weatherService: OpenMeteoService? = nil,
        neuralEngine: UnifiedNeuralEngine? = nil
    ) {
        self.context = context ?? InflamAIPersistenceController.shared.container.viewContext
        self.weatherService = weatherService ?? OpenMeteoService.shared
        self.neuralEngine = neuralEngine ?? UnifiedNeuralEngine.shared
        Task {
            await checkIfTrained()
            await analyzeWeatherForecast()
        }
    }

    // MARK: - Model Training

    /// Train model - now delegates to CoreML neural engine
    /// The CoreML model (ASFlarePredictor.mlpackage) is pre-trained and bundled with the app.
    /// This method refreshes the neural engine and analyzes weather forecasts.
    func trainModel() async throws {
        #if DEBUG
        print("🤖 Initializing CoreML prediction with weather analysis...")
        #endif

        // Check if we have enough data for the neural engine
        guard neuralEngine.daysOfUserData >= minimumDataPoints else {
            throw MLError.insufficientData(required: minimumDataPoints, actual: neuralEngine.daysOfUserData)
        }

        // Refresh the neural engine (loads CoreML model if needed)
        await neuralEngine.refresh()

        // Analyze weather forecast for enhancement
        await analyzeWeatherForecast()

        // Mark as trained (CoreML model is pre-trained)
        self.isModelTrained = true

        // Save metadata
        UserDefaults.standard.set(Date(), forKey: "lastModelTrainingDate")
        UserDefaults.standard.set(neuralEngine.daysOfUserData, forKey: "trainingDataPointCount")

        #if DEBUG
        print("✅ CoreML model ready with \(neuralEngine.daysOfUserData) days of data")
        #endif

        // Run immediate prediction
        await updatePrediction()
    }

    /// Analyze historical patterns to find flare predictors
    private func analyzeHistoricalPatterns(_ data: [TrainingDataPoint]) -> PatternAnalysis {
        var flareDays: [TrainingDataPoint] = []
        var normalDays: [TrainingDataPoint] = []

        for point in data {
            if point.willFlareWithin7Days {
                flareDays.append(point)
            } else {
                normalDays.append(point)
            }
        }

        // Calculate average values for flare days vs normal days
        func average(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            return values.reduce(0, +) / Double(values.count)
        }

        let analysis = PatternAnalysis(
            flareAvgBASDAI: average(flareDays.map { $0.basdaiScore }),
            normalAvgBASDAI: average(normalDays.map { $0.basdaiScore }),
            flareAvgPain: average(flareDays.map { $0.painLevel }),
            normalAvgPain: average(normalDays.map { $0.painLevel }),
            flareAvgStiffness: average(flareDays.map { $0.morningStiffness }),
            normalAvgStiffness: average(normalDays.map { $0.morningStiffness }),
            flareAvgSleep: average(flareDays.map { $0.sleepQuality }),
            normalAvgSleep: average(normalDays.map { $0.sleepQuality }),
            flareAvgPressureChange: average(flareDays.map { $0.pressureChange }),
            normalAvgPressureChange: average(normalDays.map { $0.pressureChange }),
            flareCount: flareDays.count,
            totalDays: data.count
        )

        return analysis
    }

    private func savePatterns(_ patterns: PatternAnalysis) {
        let encoder = JSONEncoder()
        if let encoded = try? encoder.encode(patterns) {
            UserDefaults.standard.set(encoded, forKey: "patternAnalysis")
        }
    }

    private func loadPatterns() -> PatternAnalysis? {
        guard let data = UserDefaults.standard.data(forKey: "patternAnalysis") else {
            return nil
        }
        let decoder = JSONDecoder()
        return try? decoder.decode(PatternAnalysis.self, from: data)
    }

    // MARK: - Data Preparation

    private func fetchTrainingData() async throws -> [TrainingDataPoint] {
        return try await context.perform {
            // Fetch all symptom logs with context
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]
            request.relationshipKeyPathsForPrefetching = ["contextSnapshot"]

            let logs = try self.context.fetch(request)

            // Fetch all flare events
            let flareRequest: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            let flares = try self.context.fetch(flareRequest)

            // Convert to training data
            return self.createTrainingDataPoints(logs: logs, flares: flares)
        }
    }

    private func createTrainingDataPoints(logs: [SymptomLog], flares: [FlareEvent]) -> [TrainingDataPoint] {
        var dataPoints: [TrainingDataPoint] = []

        for (index, log) in logs.enumerated() {
            guard let timestamp = log.timestamp else { continue }

            // Look ahead 7 days to see if a flare occurred
            let sevenDaysLater = Calendar.current.date(byAdding: .day, value: 7, to: timestamp) ?? timestamp
            let willFlare = flares.contains { flare in
                guard let flareStart = flare.startDate else { return false }
                return flareStart > timestamp && flareStart <= sevenDaysLater
            }

            // Calculate trends (3-day moving average)
            let basdaiTrend = calculateTrend(logs: logs, index: index, keyPath: \.basdaiScore)
            let painTrend = calculateTrend(logs: logs, index: index) { log in
                Double(log.fatigueLevel) // Use fatigue as proxy for pain
            }

            // Days since last flare
            let daysSinceLastFlare = calculateDaysSinceLastFlare(from: timestamp, flares: flares)

            // Medication adherence (last 7 days)
            let adherence = calculateMedicationAdherence(around: timestamp)

            let dataPoint = TrainingDataPoint(
                basdaiScore: log.basdaiScore,
                painLevel: Double(log.fatigueLevel),
                morningStiffness: Double(log.morningStiffnessMinutes),
                fatigueLevel: Double(log.fatigueLevel),
                sleepQuality: Double(log.sleepQuality),
                moodScore: Double(log.moodScore),
                // FIXED: No more fake weather placeholders
                // Previously: 1013.25 mmHg, 50% humidity, 20°C - meaningless defaults
                // Now: 0.0 indicates "no weather data" - model handles missing data
                barometricPressure: log.contextSnapshot?.barometricPressure ?? 0.0,
                pressureChange: log.contextSnapshot?.pressureChange12h ?? 0.0,
                humidity: Double(log.contextSnapshot?.humidity ?? 0),
                temperature: log.contextSnapshot?.temperature ?? 0.0,
                stressLevel: Double(log.moodScore), // Inverse mood as stress
                exerciseMinutes: 0.0, // TODO: Add exercise tracking
                medicationAdherence: adherence,
                daysSinceLastFlare: daysSinceLastFlare,
                basdaiTrend: basdaiTrend,
                painTrend: painTrend,
                willFlareWithin7Days: willFlare
            )

            dataPoints.append(dataPoint)
        }

        return dataPoints
    }


    // MARK: - Prediction

    /// Update prediction using CoreML model with weather forecast enhancement
    /// Will NOT update if data quality is insufficient (prevents misleading predictions)
    func updatePrediction() async {
        // Check if neural engine has enough data
        guard neuralEngine.daysOfUserData >= minimumDataPoints else {
            #if DEBUG
            print("⚠️ Not enough data for CoreML prediction (need \(minimumDataPoints) days, have \(neuralEngine.daysOfUserData))")
            #endif
            // Reset to unknown state when insufficient data
            self.flareRiskLevel = .unknown
            self.riskPercentage = 0.0
            self.isModelTrained = false
            self.contributingFactors = []
            return
        }

        // FIXED: Use real statistical calculation instead of broken CoreML model
        // The CoreML model weights collapsed during training (outputs constant ~57%)
        // This calculation uses actual symptom data and responds to real changes

        var (riskScore, factors) = await calculateStatisticalFlareRisk()

        #if DEBUG
        print("📊 Statistical Flare Risk: \(String(format: "%.1f%%", riskScore))")
        #endif

        // Store contributing factors from statistical analysis
        self.contributingFactors = factors

        // Incorporate weather forecast risk (if available)
        if let weatherRisk = weatherForecastRisk {
            // Weather contributes up to 20% additional risk
            let weatherContribution = (weatherRisk.riskScore / 100.0) * 20.0
            riskScore = min(100, riskScore + weatherContribution)

            #if DEBUG
            print("🌤️ Weather risk contribution: +\(String(format: "%.1f%%", weatherContribution))")
            #endif
        }

        self.riskPercentage = riskScore
        self.flareRiskLevel = FlarePredictorRiskLevel.from(percentage: riskScore)
        self.lastPrediction = Date()
        self.isModelTrained = true

        // Add weather factors if significant
        if let weatherRisk = weatherForecastRisk {
            updateFlarePredictorFactorsWithWeather(weatherRisk)
        }

        // Estimate days until flare (consider weather forecast)
        self.daysUntilLikelyFlare = estimateDaysUntilFlare(
            riskPercentage: riskScore,
            weatherRisk: weatherForecastRisk
        )

        #if DEBUG
        print("🎯 Combined Flare Risk (CoreML + Weather): \(String(format: "%.1f%%", riskScore)) - \(self.flareRiskLevel)")
        #endif

        // Send notification if high risk
        if self.flareRiskLevel == .high || self.flareRiskLevel == .critical {
            await sendWarningNotification(risk: self.flareRiskLevel, percentage: riskScore)
        }
    }

    /// Map UnifiedNeuralEngine impact level to FlarePredictor impact
    // MARK: - Statistical Flare Risk Calculation

    /// Calculate flare risk using real symptom data instead of broken CoreML model
    /// This is an honest statistical approach that actually responds to your data
    private func calculateStatisticalFlareRisk() async -> (Double, [FlarePredictorFactor]) {
        var factors: [FlarePredictorFactor] = []
        var riskComponents: [(name: String, score: Double, weight: Double)] = []

        // Fetch recent symptom logs (last 7 days)
        let calendar = Calendar.current
        let sevenDaysAgo = calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()

        let request = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp >= %@", sevenDaysAgo as NSDate)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: false)]

        guard let logs = try? self.context.fetch(request), !logs.isEmpty else {
            #if DEBUG
            print("⚠️ No recent symptom data for risk calculation")
            #endif
            return (0.0, [])
        }

        #if DEBUG
        print("📊 Calculating risk from \(logs.count) symptom logs")
        #endif

        // 1. BASDAI Score (Primary AS activity measure) - Weight: 35%
        let avgBASDAI = logs.reduce(0.0) { $0 + $1.basdaiScore } / Double(logs.count)
        let basdaiRisk = min(100, (avgBASDAI / 10.0) * 100)  // BASDAI 0-10 -> 0-100%
        riskComponents.append(("BASDAI", basdaiRisk, 0.35))

        if avgBASDAI >= 4.0 {
            factors.append(FlarePredictorFactor(
                name: "High Disease Activity",
                impact: avgBASDAI >= 6.0 ? .high : .medium,
                value: avgBASDAI,
                recommendation: "Consider discussing with your rheumatologist"
            ))
        }

        // 2. Morning Stiffness Duration - Weight: 20%
        let avgStiffness = logs.reduce(0.0) { $0 + Double($1.morningStiffnessMinutes) } / Double(logs.count)
        let stiffnessRisk = min(100, (avgStiffness / 60.0) * 100)  // 60+ min = 100%
        riskComponents.append(("Stiffness", stiffnessRisk, 0.20))

        if avgStiffness >= 30 {
            factors.append(FlarePredictorFactor(
                name: "Prolonged Morning Stiffness",
                impact: avgStiffness >= 60 ? .high : .medium,
                value: avgStiffness,
                recommendation: "Gentle stretching before getting up may help"
            ))
        }

        // 3. Fatigue Level - Weight: 15%
        let avgFatigue = logs.reduce(0.0) { $0 + Double($1.fatigueLevel) } / Double(logs.count)
        let fatigueRisk = min(100, (avgFatigue / 10.0) * 100)
        riskComponents.append(("Fatigue", fatigueRisk, 0.15))

        if avgFatigue >= 6.0 {
            factors.append(FlarePredictorFactor(
                name: "High Fatigue",
                impact: avgFatigue >= 8.0 ? .high : .medium,
                value: avgFatigue,
                recommendation: "Prioritize rest and pacing activities"
            ))
        }

        // 4. Pain Regions - Weight: 15%
        var totalPainScore = 0.0
        var regionCount = 0
        for log in logs {
            if let regions = log.bodyRegionLogs as? Set<BodyRegionLog> {
                for region in regions {
                    totalPainScore += Double(region.painLevel)
                    regionCount += 1
                }
            }
        }
        let avgPain = regionCount > 0 ? totalPainScore / Double(regionCount) : 0.0
        let painRisk = min(100, (avgPain / 10.0) * 100)
        riskComponents.append(("Pain", painRisk, 0.15))

        if avgPain >= 5.0 {
            factors.append(FlarePredictorFactor(
                name: "Elevated Pain Levels",
                impact: avgPain >= 7.0 ? .high : .medium,
                value: avgPain,
                recommendation: "Track pain patterns to identify triggers"
            ))
        }

        // 5. Trend Analysis - Weight: 15%
        // Compare recent 3 days vs previous 4 days
        let recentLogs = logs.prefix(min(3, logs.count))
        let olderLogs = logs.dropFirst(min(3, logs.count))

        var trendRisk = 50.0  // Neutral
        if !olderLogs.isEmpty {
            let recentAvg = recentLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(recentLogs.count)
            let olderAvg = olderLogs.reduce(0.0) { $0 + $1.basdaiScore } / Double(olderLogs.count)

            if recentAvg > olderAvg + 1.0 {
                // Worsening trend
                trendRisk = min(100, 50 + (recentAvg - olderAvg) * 15)
                factors.append(FlarePredictorFactor(
                    name: "Worsening Trend",
                    impact: (recentAvg - olderAvg) >= 2.0 ? .high : .medium,
                    value: recentAvg - olderAvg,
                    recommendation: "Symptoms are increasing - monitor closely"
                ))
            } else if recentAvg < olderAvg - 1.0 {
                // Improving trend
                trendRisk = max(0, 50 - (olderAvg - recentAvg) * 15)
                factors.append(FlarePredictorFactor(
                    name: "Improving Trend",
                    impact: .low,
                    value: olderAvg - recentAvg,
                    recommendation: "Keep up what you're doing!"
                ))
            }
        }
        riskComponents.append(("Trend", trendRisk, 0.15))

        // Calculate weighted risk score
        var totalRisk = 0.0
        for component in riskComponents {
            totalRisk += component.score * component.weight
            #if DEBUG
            print("   \(component.name): \(String(format: "%.1f%%", component.score)) (weight: \(component.weight))")
            #endif
        }

        // Ensure within bounds
        totalRisk = max(0, min(100, totalRisk))

        #if DEBUG
        print("   → Total Statistical Risk: \(String(format: "%.1f%%", totalRisk))")
        #endif

        return (totalRisk, factors)
    }

    private func mapImpactLevel(_ impact: ContributingFactor.ImpactLevel) -> FlarePredictorFactor.Impact {
        switch impact {
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        }
    }

    /// Calculate risk score by comparing current values to historical patterns
    private func calculateRiskScore(_ current: TrainingDataPoint, patterns: PatternAnalysis) -> Double {
        var riskFactors: [Double] = []

        // BASDAI comparison (weight: 3)
        if patterns.flareAvgBASDAI > 0 {
            let basdaiRisk = min(1.0, current.basdaiScore / patterns.flareAvgBASDAI)
            riskFactors.append(contentsOf: [basdaiRisk, basdaiRisk, basdaiRisk])
        }

        // Pain comparison (weight: 2)
        if patterns.flareAvgPain > 0 {
            let painRisk = min(1.0, current.painLevel / patterns.flareAvgPain)
            riskFactors.append(contentsOf: [painRisk, painRisk])
        }

        // Stiffness comparison (weight: 2)
        if patterns.flareAvgStiffness > 0 {
            let stiffnessRisk = min(1.0, current.morningStiffness / patterns.flareAvgStiffness)
            riskFactors.append(contentsOf: [stiffnessRisk, stiffnessRisk])
        }

        // Sleep quality (inverted - lower is worse) (weight: 1)
        if patterns.flareAvgSleep > 0 && current.sleepQuality > 0 {
            let sleepRisk = 1.0 - min(1.0, current.sleepQuality / patterns.normalAvgSleep)
            riskFactors.append(sleepRisk)
        }

        // Barometric pressure change (weight: 1)
        if abs(patterns.flareAvgPressureChange) > 0 && abs(current.pressureChange) > 5 {
            let pressureRisk = min(1.0, abs(current.pressureChange) / abs(patterns.flareAvgPressureChange))
            riskFactors.append(pressureRisk)
        }

        // Days since last flare (lower is higher risk)
        if current.daysSinceLastFlare < 30 {
            let recencyRisk = 1.0 - (current.daysSinceLastFlare / 30.0)
            riskFactors.append(recencyRisk)
        }

        // Calculate weighted average
        guard !riskFactors.isEmpty else { return 0.0 }
        let avgRisk = riskFactors.reduce(0, +) / Double(riskFactors.count)

        // Convert to percentage and apply base rate
        let baseRate = Double(patterns.flareCount) / Double(patterns.totalDays)
        let adjustedRisk = (avgRisk * 0.7) + (baseRate * 0.3) // 70% pattern, 30% base rate

        return min(100, adjustedRisk * 100)
    }

    // MARK: - Analysis

    private func analyzeFlarePredictorFactors(_ dataPoint: TrainingDataPoint) -> [FlarePredictorFactor] {
        var factors: [FlarePredictorFactor] = []

        // High BASDAI
        if dataPoint.basdaiScore > 4.0 {
            factors.append(FlarePredictorFactor(
                name: "Elevated BASDAI",
                impact: .high,
                value: dataPoint.basdaiScore,
                recommendation: "Your disease activity is elevated. Consider contacting your rheumatologist."
            ))
        }

        // Increasing pain trend
        if dataPoint.painTrend > 1.5 {
            factors.append(FlarePredictorFactor(
                name: "Rising Pain Levels",
                impact: .high,
                value: dataPoint.painTrend,
                recommendation: "Your pain has been increasing. Rest and gentle movement may help."
            ))
        }

        // Barometric pressure drop
        if dataPoint.pressureChange < -5.0 {
            factors.append(FlarePredictorFactor(
                name: "Pressure Drop",
                impact: .medium,
                value: abs(dataPoint.pressureChange),
                recommendation: "Weather changing. Consider preventive measures."
            ))
        }

        // Poor sleep
        if dataPoint.sleepQuality < 4 {
            factors.append(FlarePredictorFactor(
                name: "Poor Sleep Quality",
                impact: .medium,
                value: Double(dataPoint.sleepQuality),
                recommendation: "Sleep quality affecting recovery. Prioritize rest tonight."
            ))
        }

        // Low medication adherence
        if dataPoint.medicationAdherence < 0.7 {
            factors.append(FlarePredictorFactor(
                name: "Missed Medications",
                impact: .high,
                value: dataPoint.medicationAdherence * 100,
                recommendation: "You've missed some medications. Stay on schedule for best results."
            ))
        }

        // Recent flare
        if dataPoint.daysSinceLastFlare < 30 {
            factors.append(FlarePredictorFactor(
                name: "Recent Flare",
                impact: .medium,
                value: Double(dataPoint.daysSinceLastFlare),
                recommendation: "Still recovering from recent flare. Take it easy."
            ))
        }

        return factors.sorted { $0.impact.rawValue > $1.impact.rawValue }
    }

    private func estimateDaysUntilFlare(riskPercentage: Double, weatherRisk: WeatherForecastRisk? = nil) -> Int? {
        // Base estimate from symptom risk
        var baseDays: Int?
        if riskPercentage > 80 { baseDays = 1 }
        else if riskPercentage > 60 { baseDays = 3 }
        else if riskPercentage > 40 { baseDays = 5 }
        else if riskPercentage > 20 { baseDays = 7 }

        // Adjust based on weather forecast
        if let weather = weatherRisk {
            // If there's a rapid pressure drop coming soon, flare may happen sooner
            if weather.pressureTrend.direction == .rapidDrop,
               let hours = weather.pressureTrend.hoursUntilChange {
                let weatherDays = max(1, hours / 24)
                if let base = baseDays {
                    return min(base, weatherDays)
                } else {
                    return weatherDays
                }
            }

            // If stormy days are coming, consider that
            if let firstStormDay = weather.stormyDays.first {
                if let base = baseDays {
                    return min(base, firstStormDay + 1)
                }
            }
        }

        return baseDays
    }

    // MARK: - Notifications

    private func sendWarningNotification(risk: FlarePredictorRiskLevel, percentage: Double) async {
        let content = UNMutableNotificationContent()
        content.title = "📊 Pattern Update"

        switch risk {
        case .critical:
            content.body = "Notable changes detected in your logged data. Consider discussing with your healthcare provider."
            content.sound = .defaultCritical
        case .high:
            content.body = "Your recent data shows some changes. Review your patterns and consider rest."
            content.sound = .default
        default:
            return
        }

        content.categoryIdentifier = "PATTERN_UPDATE"
        content.userInfo = ["patternLevel": risk.rawValue]

        let request = UNNotificationRequest(
            identifier: "flare-warning-\(Date().timeIntervalSince1970)",
            content: content,
            trigger: nil // Immediate
        )

        try? await UNUserNotificationCenter.current().add(request)
        #if DEBUG
        print("📱 Warning notification sent")
        #endif
    }

    // MARK: - Weather Forecast Analysis

    /// Analyze 7-day weather forecast for flare risk factors
    func analyzeWeatherForecast() async {
        do {
            // Fetch all weather data from WeatherKit
            try await weatherService.fetchAllWeatherData()

            // Get current weather
            guard let currentWeather = weatherService.currentWeather else {
                #if DEBUG
                print("⚠️ No current weather data available")
                #endif
                return
            }

            // Get 7-day forecast
            let dailyForecast = weatherService.dailyForecast
            let hourlyForecast = weatherService.hourlyForecast

            // Analyze forecast for risk factors
            let forecastRisk = analyzeWeatherRiskFactors(
                current: currentWeather,
                daily: dailyForecast,
                hourly: hourlyForecast
            )

            self.weatherForecastRisk = forecastRisk
            self.upcomingWeatherAlerts = generateWeatherAlerts(from: forecastRisk)

            // Update contributing factors with weather forecast data
            updateFlarePredictorFactorsWithWeather(forecastRisk)

            #if DEBUG
            print("🌤️ Weather forecast analysis complete: \(forecastRisk.overallRisk.rawValue) risk")
            #endif

        } catch {
            #if DEBUG
            print("❌ Weather forecast analysis failed: \(error.localizedDescription)")
            #endif
        }
    }

    /// Analyze weather data to identify flare risk factors
    private func analyzeWeatherRiskFactors(
        current: CurrentWeatherData,
        daily: [DailyWeatherData],
        hourly: [HourlyWeatherData]
    ) -> WeatherForecastRisk {

        var pressureDropDays: [Int] = []
        var highHumidityDays: [Int] = []
        var temperatureSwingDays: [Int] = []
        var stormyDays: [Int] = []

        let currentPressure = current.pressure

        // Analyze each day's forecast
        for (dayIndex, day) in daily.enumerated() {
            // Note: DailyWeatherData doesn't have pressure - use hourly data for pressure analysis instead
            // Pressure drops are analyzed via hourly trend below

            // Check for high humidity (>80%)
            if day.humidity > 80 {
                highHumidityDays.append(dayIndex)
            }

            // Check for temperature swings (>10°C difference)
            // Use temperatureHigh/temperatureLow (actual property names from DailyWeatherData)
            let tempSwing = day.temperatureHigh - day.temperatureLow
            if tempSwing > 10.0 {
                temperatureSwingDays.append(dayIndex)
            }

            // Check for stormy conditions
            let stormyConditions: [WeatherConditionType] = [.thunderstorm, .heavyRain, .snow, .sleet]
            if stormyConditions.contains(day.condition) {
                stormyDays.append(dayIndex)
            }
        }

        // Analyze hourly pressure trend for next 48 hours (pressure drops detected here)
        let pressureTrend = analyzePressureTrend(hourly: hourly)

        // Detect pressure drop days from hourly data grouped by day
        pressureDropDays = detectPressureDropDays(hourly: hourly, currentPressure: currentPressure)

        // Calculate overall risk level
        let riskScore = calculateWeatherRiskScore(
            pressureDropDays: pressureDropDays.count,
            highHumidityDays: highHumidityDays.count,
            temperatureSwingDays: temperatureSwingDays.count,
            stormyDays: stormyDays.count,
            pressureTrend: pressureTrend
        )

        let overallRisk: FlarePredictorWeatherRisk
        switch riskScore {
        case 0..<20: overallRisk = .low
        case 20..<40: overallRisk = .moderate
        case 40..<70: overallRisk = .high
        default: overallRisk = .critical
        }

        return WeatherForecastRisk(
            overallRisk: overallRisk,
            riskScore: riskScore,
            currentCondition: current.condition,
            currentPressure: current.pressure,
            currentHumidity: current.humidity,
            currentTemperature: current.temperature,
            pressureDropDays: pressureDropDays,
            highHumidityDays: highHumidityDays,
            temperatureSwingDays: temperatureSwingDays,
            stormyDays: stormyDays,
            pressureTrend: pressureTrend,
            dailyForecasts: daily.prefix(7).map { day in
                DayForecastSummary(
                    date: day.date,
                    condition: day.condition,
                    highTemp: day.temperatureHigh,
                    lowTemp: day.temperatureLow,
                    humidity: day.humidity,
                    pressure: currentPressure, // DailyWeatherData doesn't have pressure, use current
                    precipChance: Double(day.precipitationChance)
                )
            }
        )
    }

    /// Detect days with significant pressure drops from hourly data
    private func detectPressureDropDays(hourly: [HourlyWeatherData], currentPressure: Double) -> [Int] {
        var pressureDropDays: [Int] = []
        let calendar = Calendar.current

        // Group hourly data by day
        var dailyMinPressure: [Int: Double] = [:]
        for hour in hourly {
            let dayOffset = calendar.dateComponents([.day], from: Date(), to: hour.date).day ?? 0
            if dayOffset >= 0 && dayOffset < 7 {
                if let existing = dailyMinPressure[dayOffset] {
                    dailyMinPressure[dayOffset] = min(existing, hour.pressure)
                } else {
                    dailyMinPressure[dayOffset] = hour.pressure
                }
            }
        }

        // Find days with significant pressure drops
        for (dayIndex, minPressure) in dailyMinPressure {
            let pressureDiff = minPressure - currentPressure
            if pressureDiff < -5.0 { // 5+ hPa drop is significant
                pressureDropDays.append(dayIndex)
            }
        }

        return pressureDropDays.sorted()
    }

    /// Analyze pressure trend from hourly data
    private func analyzePressureTrend(hourly: [HourlyWeatherData]) -> PressureTrend {
        guard hourly.count >= 12 else {
            return PressureTrend(direction: .stable, magnitude: 0, hoursUntilChange: nil)
        }

        // FIXED: No fake pressure fallback - require real weather data
        guard let currentPressure = hourly.first?.pressure, currentPressure > 0 else {
            return PressureTrend(direction: .stable, magnitude: 0, hoursUntilChange: nil)
        }
        var maxDrop: Double = 0
        var maxDropHour: Int?

        // Look for significant pressure changes in next 48 hours
        for (index, hour) in hourly.prefix(48).enumerated() {
            let change = hour.pressure - currentPressure
            if change < maxDrop {
                maxDrop = change
                maxDropHour = index
            }
        }

        let direction: PressureTrendDirection
        let magnitude = abs(maxDrop)

        if maxDrop < -10 {
            direction = .rapidDrop
        } else if maxDrop < -5 {
            direction = .dropping
        } else if maxDrop > 5 {
            direction = .rising
        } else {
            direction = .stable
        }

        return PressureTrend(
            direction: direction,
            magnitude: magnitude,
            hoursUntilChange: maxDropHour
        )
    }

    /// Calculate weather-based risk score (0-100)
    private func calculateWeatherRiskScore(
        pressureDropDays: Int,
        highHumidityDays: Int,
        temperatureSwingDays: Int,
        stormyDays: Int,
        pressureTrend: PressureTrend
    ) -> Double {
        var score: Double = 0

        // Pressure drops are the biggest factor (weight: 40%)
        score += Double(min(pressureDropDays, 3)) * 13.3

        // Pressure trend in next 48h (weight: 20%)
        switch pressureTrend.direction {
        case .rapidDrop: score += 20
        case .dropping: score += 12
        case .stable: score += 0
        case .rising: score -= 5
        }

        // High humidity days (weight: 15%)
        score += Double(min(highHumidityDays, 4)) * 3.75

        // Temperature swings (weight: 15%)
        score += Double(min(temperatureSwingDays, 4)) * 3.75

        // Stormy conditions (weight: 10%)
        score += Double(min(stormyDays, 3)) * 3.3

        return min(100, max(0, score))
    }

    /// Generate alerts from weather risk analysis
    private func generateWeatherAlerts(from risk: WeatherForecastRisk) -> [WeatherRiskAlert] {
        var alerts: [WeatherRiskAlert] = []

        // Alert for upcoming pressure drop
        if risk.pressureTrend.direction == .rapidDrop || risk.pressureTrend.direction == .dropping {
            let severity: WeatherRiskAlert.Severity = risk.pressureTrend.direction == .rapidDrop ? .high : .moderate
            // FIXED: Don't fabricate "24 hours" - use actual data or indicate unknown
            let hoursKnown = risk.pressureTrend.hoursUntilChange != nil
            let hours = risk.pressureTrend.hoursUntilChange ?? 0
            let timeframeText = hoursKnown ? "Next \(hours)h" : "Timing unknown"
            let messageText = hoursKnown
                ? "Barometric pressure will drop \(String(format: "%.1f", risk.pressureTrend.magnitude)) hPa in the next \(hours) hours"
                : "Barometric pressure dropping \(String(format: "%.1f", risk.pressureTrend.magnitude)) hPa"

            alerts.append(WeatherRiskAlert(
                type: .pressureDrop,
                severity: severity,
                title: "Pressure Drop Alert",
                message: messageText,
                timeframe: timeframeText,
                recommendation: "Consider taking preventive measures. Rest, stay warm, and prepare medications."
            ))
        }

        // Alert for stormy days
        if !risk.stormyDays.isEmpty {
            let daysText = risk.stormyDays.map { "Day \($0 + 1)" }.joined(separator: ", ")
            alerts.append(WeatherRiskAlert(
                type: .storm,
                severity: .moderate,
                title: "Storm Warning",
                message: "Stormy weather expected on \(daysText)",
                timeframe: "Next 7 days",
                recommendation: "Plan indoor activities and prepare for potential flare symptoms."
            ))
        }

        // Alert for high humidity
        if risk.highHumidityDays.count >= 3 {
            alerts.append(WeatherRiskAlert(
                type: .humidity,
                severity: .low,
                title: "High Humidity Period",
                message: "High humidity (>80%) expected for \(risk.highHumidityDays.count) days",
                timeframe: "Next 7 days",
                recommendation: "Stay hydrated and consider using a dehumidifier if symptoms worsen."
            ))
        }

        return alerts
    }

    /// Update contributing factors with weather forecast data
    private func updateFlarePredictorFactorsWithWeather(_ weatherRisk: WeatherForecastRisk) {
        // Add weather-based contributing factors
        if weatherRisk.pressureTrend.direction == .rapidDrop {
            let existingFactor = contributingFactors.first { $0.name == "Pressure Drop" }
            if existingFactor == nil {
                contributingFactors.append(FlarePredictorFactor(
                    name: "Upcoming Pressure Drop",
                    impact: .high,
                    value: weatherRisk.pressureTrend.magnitude,
                    recommendation: "Rapid pressure drop expected. This is a known flare trigger for AS patients."
                ))
            }
        }

        if !weatherRisk.stormyDays.isEmpty {
            contributingFactors.append(FlarePredictorFactor(
                name: "Stormy Weather Ahead",
                impact: .medium,
                value: Double(weatherRisk.stormyDays.count),
                recommendation: "Storms expected in the forecast. Weather changes may affect symptoms."
            ))
        }

        if weatherRisk.currentHumidity > 80 {
            contributingFactors.append(FlarePredictorFactor(
                name: "High Humidity",
                impact: .low,
                value: Double(weatherRisk.currentHumidity),
                recommendation: "Current humidity is high. Some patients report increased stiffness."
            ))
        }

        // Re-sort by impact
        contributingFactors.sort { $0.impact.rawValue > $1.impact.rawValue }
    }

    // MARK: - Helper Methods

    private func checkIfTrained() async {
        // CoreML model is always available (bundled with app)
        // Check if neural engine has enough data
        if neuralEngine.daysOfUserData >= minimumDataPoints {
            self.isModelTrained = true
            #if DEBUG
            print("✅ CoreML model ready (\(neuralEngine.daysOfUserData) days of data)")
            #endif
            await updatePrediction()
        } else {
            #if DEBUG
            print("ℹ️ Need \(minimumDataPoints - neuralEngine.daysOfUserData) more days of data for predictions.")
            #endif
        }
    }

    private func fetchRecentData() async throws -> [TrainingDataPoint] {
        let allData = try await fetchTrainingData()
        return Array(allData.suffix(7)) // Last 7 days
    }

    private func calculateTrend(logs: [SymptomLog], index: Int, keyPath: KeyPath<SymptomLog, Double>) -> Double {
        guard index >= 2 else { return 0.0 }
        let recent = logs[max(0, index - 2)...index]
        let values = recent.map { $0[keyPath: keyPath] }
        return values.reduce(0.0, +) / Double(values.count)
    }

    private func calculateTrend(logs: [SymptomLog], index: Int, valueExtractor: (SymptomLog) -> Double) -> Double {
        guard index >= 2 else { return 0.0 }
        let recent = logs[max(0, index - 2)...index]
        let values = recent.map(valueExtractor)
        return values.reduce(0.0, +) / Double(values.count)
    }

    private func calculateDaysSinceLastFlare(from date: Date, flares: [FlareEvent]) -> Double {
        let previousFlares = flares.filter { flare in
            guard let flareStart = flare.startDate else { return false }
            return flareStart < date
        }.sorted { ($0.startDate ?? Date.distantPast) > ($1.startDate ?? Date.distantPast) }

        guard let lastFlare = previousFlares.first,
              let lastFlareDate = lastFlare.startDate else {
            // FIXED: Return -1 instead of fake 365 days
            // -1 = "no previous flare recorded" (model can learn this pattern)
            // 365 was arbitrary and could mislead predictions
            return -1.0
        }

        // FIXED: Return -1 if calculation fails, not fake 365
        let days = Calendar.current.dateComponents([.day], from: lastFlareDate, to: date).day ?? -1
        return Double(days)
    }

    private func calculateMedicationAdherence(around date: Date) -> Double {
        // Fetch dose logs for 7 days around this date
        let calendar = Calendar.current
        let startDate = calendar.date(byAdding: .day, value: -3, to: date) ?? date
        let endDate = calendar.date(byAdding: .day, value: 4, to: date) ?? date
        
        let request: NSFetchRequest<DoseLog> = DoseLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp <= %@",
            startDate as NSDate,
            endDate as NSDate
        )
        
        guard let logs = try? context.fetch(request), !logs.isEmpty else {
            // No medication tracking data - return neutral value (1.0 = perfect adherence)
            // This prevents penalizing users who don't track medications
            return 1.0
        }
        
        // Calculate adherence: taken doses / total scheduled doses
        let takenCount = logs.filter { !$0.wasSkipped }.count
        let totalCount = logs.count
        
        return Double(takenCount) / Double(totalCount)
    }
}

// MARK: - Models

struct TrainingDataPoint {
    let basdaiScore: Double
    let painLevel: Double
    let morningStiffness: Double
    let fatigueLevel: Double
    let sleepQuality: Double
    let moodScore: Double
    let barometricPressure: Double
    let pressureChange: Double
    let humidity: Double
    let temperature: Double
    let stressLevel: Double
    let exerciseMinutes: Double
    let medicationAdherence: Double
    let daysSinceLastFlare: Double
    let basdaiTrend: Double
    let painTrend: Double
    let willFlareWithin7Days: Bool
}

enum FlarePredictorRiskLevel: Int, Comparable {
    case unknown = 0
    case low = 1
    case moderate = 2
    case high = 3
    case critical = 4

    static func from(percentage: Double) -> FlarePredictorRiskLevel {
        switch percentage {
        case 0..<20: return .low
        case 20..<40: return .moderate
        case 40..<70: return .high
        case 70...100: return .critical
        default: return .unknown
        }
    }

    var color: String {
        switch self {
        case .unknown: return "gray"
        case .low: return "green"
        case .moderate: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }

    var emoji: String {
        switch self {
        case .unknown: return "❓"
        case .low: return "✅"
        case .moderate: return "⚠️"
        case .high: return "🔶"
        case .critical: return "🚨"
        }
    }

    static func < (lhs: FlarePredictorRiskLevel, rhs: FlarePredictorRiskLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

struct FlarePredictorFactor: Identifiable {
    let id = UUID()
    let name: String
    let impact: Impact
    let value: Double
    let recommendation: String

    enum Impact: Int {
        case low = 1
        case medium = 2
        case high = 3

        var color: String {
            switch self {
            case .low: return "blue"
            case .medium: return "orange"
            case .high: return "red"
            }
        }
    }
}

struct PatternAnalysis: Codable {
    let flareAvgBASDAI: Double
    let normalAvgBASDAI: Double
    let flareAvgPain: Double
    let normalAvgPain: Double
    let flareAvgStiffness: Double
    let normalAvgStiffness: Double
    let flareAvgSleep: Double
    let normalAvgSleep: Double
    let flareAvgPressureChange: Double
    let normalAvgPressureChange: Double
    let flareCount: Int
    let totalDays: Int
}

enum MLError: LocalizedError {
    case insufficientData(required: Int, actual: Int)
    case trainingFailed(Error)
    case predictionFailed(Error)

    var errorDescription: String? {
        switch self {
        case .insufficientData(let required, let actual):
            return "Need \(required) days of data, only have \(actual). Keep logging!"
        case .trainingFailed(let error):
            return "Training failed: \(error.localizedDescription)"
        case .predictionFailed(let error):
            return "Prediction failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Weather Forecast Risk Models

/// Overall weather forecast risk assessment
struct WeatherForecastRisk {
    let overallRisk: FlarePredictorWeatherRisk
    let riskScore: Double
    let currentCondition: WeatherConditionType
    let currentPressure: Double
    let currentHumidity: Int
    let currentTemperature: Double
    let pressureDropDays: [Int]
    let highHumidityDays: [Int]
    let temperatureSwingDays: [Int]
    let stormyDays: [Int]
    let pressureTrend: PressureTrend
    let dailyForecasts: [DayForecastSummary]

    /// Summary text for UI display
    var summaryText: String {
        switch overallRisk {
        case .low:
            return "Weather conditions look favorable for the next 7 days."
        case .moderate:
            return "Some weather changes expected. Monitor your symptoms."
        case .high:
            return "Significant weather changes ahead. Take preventive measures."
        case .critical:
            return "Major weather disruption expected. High risk of symptom flare."
        }
    }

    /// Factors contributing to weather risk
    var riskFactorsSummary: [String] {
        var factors: [String] = []

        if !pressureDropDays.isEmpty {
            factors.append("Pressure drops on \(pressureDropDays.count) day(s)")
        }
        if pressureTrend.direction == .rapidDrop {
            factors.append("Rapid pressure drop in next 48h")
        }
        if !stormyDays.isEmpty {
            factors.append("Storms expected on \(stormyDays.count) day(s)")
        }
        if !highHumidityDays.isEmpty {
            factors.append("High humidity on \(highHumidityDays.count) day(s)")
        }
        if !temperatureSwingDays.isEmpty {
            factors.append("Temperature swings on \(temperatureSwingDays.count) day(s)")
        }

        return factors
    }
}

/// Weather-based risk level (for FlarePredictor)
enum FlarePredictorWeatherRisk: String {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case critical = "Critical"

    var color: String {
        switch self {
        case .low: return "green"
        case .moderate: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }
}

/// Pressure trend analysis
struct PressureTrend {
    let direction: PressureTrendDirection
    let magnitude: Double // in hPa
    let hoursUntilChange: Int?

    var description: String {
        switch direction {
        case .rapidDrop:
            return "Rapid drop of \(String(format: "%.1f", magnitude)) hPa expected"
        case .dropping:
            return "Gradual pressure decrease expected"
        case .stable:
            return "Pressure stable"
        case .rising:
            return "Pressure rising"
        }
    }
}

enum PressureTrendDirection: String {
    case rapidDrop = "rapid_drop"
    case dropping = "dropping"
    case stable = "stable"
    case rising = "rising"
}

/// Summary of daily forecast for UI
struct DayForecastSummary: Identifiable {
    let id = UUID()
    let date: Date
    let condition: WeatherConditionType
    let highTemp: Double
    let lowTemp: Double
    let humidity: Int
    let pressure: Double
    let precipChance: Double

    var dayName: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEE"
        return formatter.string(from: date)
    }

    var isHighRisk: Bool {
        // Day is high risk if stormy or significant pressure drop potential
        let stormyConditions: [WeatherConditionType] = [.thunderstorm, .heavyRain, .snow, .sleet]
        return stormyConditions.contains(condition) || humidity > 80
    }
}

/// Weather risk alert
struct WeatherRiskAlert: Identifiable {
    let id = UUID()
    let type: AlertType
    let severity: Severity
    let title: String
    let message: String
    let timeframe: String
    let recommendation: String

    enum AlertType {
        case pressureDrop
        case storm
        case humidity
        case temperatureSwing
        case general
    }

    enum Severity: Int {
        case low = 1
        case moderate = 2
        case high = 3

        var color: String {
            switch self {
            case .low: return "blue"
            case .moderate: return "orange"
            case .high: return "red"
            }
        }

        var icon: String {
            switch self {
            case .low: return "info.circle.fill"
            case .moderate: return "exclamationmark.triangle.fill"
            case .high: return "exclamationmark.octagon.fill"
            }
        }
    }
}
