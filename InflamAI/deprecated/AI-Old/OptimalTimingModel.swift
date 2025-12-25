//
//  OptimalTimingModel.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreML
import CreateML
import os.log

// MARK: - Optimal Timing Model Implementation
class OptimalTimingModelImpl: OptimalTimingModel {
    
    private let logger = Logger(subsystem: "InflamAI", category: "OptimalTimingModel")
    private var model: MLModel?
    private var trainingData: [AdherenceRecord] = []
    private let modelName = "OptimalTimingModel"
    
    // Timing analysis parameters
    private let timeSlotDuration: TimeInterval = 30 * 60 // 30 minutes
    private let analysisWindow = 30 // Days to analyze for patterns
    private let minimumDataPoints = 20 // Minimum records for reliable prediction
    
    // User timing patterns
    private var userTimingPatterns: [String: TimingPattern] = [:]
    private var medicationTimingPreferences: [String: [TimePreference]] = [:]
    
    required init() async throws {
        logger.info("Initializing OptimalTimingModel")
        
        // Load existing model and patterns
        await loadExistingModel()
        loadTimingPatterns()
        
        logger.info("OptimalTimingModel initialized")
    }
    
    func predictOptimalTime(for medication: Medication, scheduledTime: Date) async -> Date {
        logger.info("Predicting optimal time for \(medication.name)")
        
        // Get user's timing patterns for this medication
        let patterns = getUserTimingPatterns(for: medication)
        
        // If we have ML model, use it
        if let mlModel = model {
            return await predictWithMLModel(mlModel, medication: medication, scheduledTime: scheduledTime, patterns: patterns)
        } else {
            // Use pattern-based prediction
            return predictWithPatterns(medication: medication, scheduledTime: scheduledTime, patterns: patterns)
        }
    }
    
    func updateWithNewData(_ data: [AdherenceRecord]) async {
        logger.info("Updating timing model with \(data.count) new records")
        
        // Add new data
        trainingData.append(contentsOf: data)
        
        // Keep only recent data (last 3 months)
        let threeMonthsAgo = Date().addingTimeInterval(-3 * 30 * 24 * 60 * 60)
        trainingData = trainingData.filter { $0.scheduledTime >= threeMonthsAgo }
        
        // Update timing patterns
        updateTimingPatterns(with: data)
        
        // Retrain model if we have enough data
        if trainingData.count >= minimumDataPoints {
            await retrainModel()
        }
    }
    
    // MARK: - Private Methods
    
    private func loadExistingModel() async {
        do {
            if let modelURL = getModelURL() {
                model = try MLModel(contentsOf: modelURL)
                logger.info("Loaded existing optimal timing model")
            }
        } catch {
            logger.error("Failed to load timing model: \(error.localizedDescription)")
        }
    }
    
    private func loadTimingPatterns() {
        // Load saved timing patterns from UserDefaults
        if let data = UserDefaults.standard.data(forKey: "timing_patterns"),
           let patterns = try? JSONDecoder().decode([String: TimingPattern].self, from: data) {
            userTimingPatterns = patterns
            logger.info("Loaded \(patterns.count) timing patterns")
        }
        
        if let data = UserDefaults.standard.data(forKey: "medication_timing_preferences"),
           let preferences = try? JSONDecoder().decode([String: [TimePreference]].self, from: data) {
            medicationTimingPreferences = preferences
            logger.info("Loaded timing preferences for \(preferences.count) medications")
        }
    }
    
    private func saveTimingPatterns() {
        // Save timing patterns to UserDefaults
        if let data = try? JSONEncoder().encode(userTimingPatterns) {
            UserDefaults.standard.set(data, forKey: "timing_patterns")
        }
        
        if let data = try? JSONEncoder().encode(medicationTimingPreferences) {
            UserDefaults.standard.set(data, forKey: "medication_timing_preferences")
        }
    }
    
    private func predictWithMLModel(_ mlModel: MLModel, medication: Medication, scheduledTime: Date, patterns: [TimingPattern]) async -> Date {
        do {
            // Prepare features for ML model
            let features = prepareTimingFeatures(medication: medication, scheduledTime: scheduledTime, patterns: patterns)
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            
            // Make prediction
            let prediction = try mlModel.prediction(from: input)
            
            // Extract optimal hour offset
            let hourOffset = prediction.featureValue(for: "optimal_hour_offset")?.doubleValue ?? 0.0
            let confidence = prediction.featureValue(for: "confidence")?.doubleValue ?? 0.5
            
            // Apply offset to scheduled time
            let optimalTime = scheduledTime.addingTimeInterval(hourOffset * 3600)
            
            logger.info("ML timing prediction: offset=\(hourOffset)h, confidence=\(confidence)")
            return optimalTime
            
        } catch {
            logger.error("ML timing prediction failed: \(error.localizedDescription)")
            return predictWithPatterns(medication: medication, scheduledTime: scheduledTime, patterns: patterns)
        }
    }
    
    private func predictWithPatterns(medication: Medication, scheduledTime: Date, patterns: [TimingPattern]) -> Date {
        logger.info("Using pattern-based timing prediction")
        
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: scheduledTime)
        let dayOfWeek = calendar.component(.weekday, from: scheduledTime)
        
        // Get medication-specific preferences
        let preferences = medicationTimingPreferences[medication.name] ?? []
        
        // Find best time based on patterns and preferences
        var bestTime = scheduledTime
        var bestScore = 0.0
        
        // Check time slots within ±3 hours of scheduled time
        for hourOffset in -3...3 {
            let candidateTime = scheduledTime.addingTimeInterval(Double(hourOffset) * 3600)
            let candidateHour = calendar.component(.hour, from: candidateTime)
            
            var score = 0.0
            
            // Score based on historical adherence at this hour
            let hourPattern = patterns.first { $0.hour == candidateHour }
            score += (hourPattern?.adherenceRate ?? 0.5) * 0.4
            
            // Score based on user preferences
            let preference = preferences.first { abs($0.hour - candidateHour) <= 1 }
            score += (preference?.preferenceScore ?? 0.5) * 0.3
            
            // Score based on day of week patterns
            let dayPattern = patterns.first { $0.dayOfWeek == dayOfWeek && $0.hour == candidateHour }
            score += (dayPattern?.adherenceRate ?? 0.5) * 0.2
            
            // Penalty for being too far from original time
            let timePenalty = Double(abs(hourOffset)) * 0.1
            score -= timePenalty
            
            // Bonus for optimal timing windows (morning, evening)
            if candidateHour >= 7 && candidateHour <= 9 { // Morning
                score += 0.1
            } else if candidateHour >= 18 && candidateHour <= 20 { // Evening
                score += 0.1
            }
            
            if score > bestScore {
                bestScore = score
                bestTime = candidateTime
            }
        }
        
        let offset = bestTime.timeIntervalSince(scheduledTime) / 3600.0
        logger.info("Pattern-based prediction: offset=\(offset)h, score=\(bestScore)")
        
        return bestTime
    }
    
    private func getUserTimingPatterns(for medication: Medication) -> [TimingPattern] {
        // Get relevant adherence records for this medication
        let medicationRecords = trainingData.filter { $0.medicationName == medication.name }
        
        // Group by hour and day of week
        var hourPatterns: [Int: [AdherenceRecord]] = [:]
        var dayHourPatterns: [String: [AdherenceRecord]] = [:]
        
        for record in medicationRecords {
            let calendar = Calendar.current
            let hour = calendar.component(.hour, from: record.scheduledTime)
            let dayOfWeek = calendar.component(.weekday, from: record.scheduledTime)
            
            // Group by hour
            hourPatterns[hour, default: []].append(record)
            
            // Group by day-hour combination
            let dayHourKey = "\(dayOfWeek)-\(hour)"
            dayHourPatterns[dayHourKey, default: []].append(record)
        }
        
        var patterns: [TimingPattern] = []
        
        // Create hour patterns
        for (hour, records) in hourPatterns {
            let adherentCount = records.filter { $0.wasOnTime }.count
            let adherenceRate = Double(adherentCount) / Double(records.count)
            
            patterns.append(TimingPattern(
                hour: hour,
                dayOfWeek: nil,
                adherenceRate: adherenceRate,
                sampleSize: records.count,
                averageDelay: calculateAverageDelay(records)
            ))
        }
        
        // Create day-hour patterns
        for (dayHourKey, records) in dayHourPatterns {
            let components = dayHourKey.split(separator: "-")
            guard components.count == 2,
                  let dayOfWeek = Int(components[0]),
                  let hour = Int(components[1]) else { continue }
            
            let adherentCount = records.filter { $0.wasOnTime }.count
            let adherenceRate = Double(adherentCount) / Double(records.count)
            
            patterns.append(TimingPattern(
                hour: hour,
                dayOfWeek: dayOfWeek,
                adherenceRate: adherenceRate,
                sampleSize: records.count,
                averageDelay: calculateAverageDelay(records)
            ))
        }
        
        return patterns.filter { $0.sampleSize >= 3 } // Only patterns with sufficient data
    }
    
    private func updateTimingPatterns(with newData: [AdherenceRecord]) {
        // Update user timing patterns based on new adherence data
        for record in newData {
            let calendar = Calendar.current
            let hour = calendar.component(.hour, from: record.actualTime)
            let dayOfWeek = calendar.component(.weekday, from: record.actualTime)
            
            // Update general timing pattern
            let patternKey = "general"
            var pattern = userTimingPatterns[patternKey] ?? TimingPattern(hour: hour, dayOfWeek: nil, adherenceRate: 0.5, sampleSize: 0, averageDelay: 0)
            
            // Update adherence rate using exponential moving average
            let alpha = 0.1 // Learning rate
            let newAdherenceValue = record.wasOnTime ? 1.0 : 0.0
            pattern.adherenceRate = (1 - alpha) * pattern.adherenceRate + alpha * newAdherenceValue
            pattern.sampleSize += 1
            
            userTimingPatterns[patternKey] = pattern
            
            // Update medication-specific preferences
            var preferences = medicationTimingPreferences[record.medicationName] ?? []
            
            // Find or create preference for this hour
            if let index = preferences.firstIndex(where: { abs($0.hour - hour) <= 1 }) {
                // Update existing preference
                let successWeight = record.wasOnTime ? 1.0 : -0.5
                preferences[index].preferenceScore = max(0.0, min(1.0, preferences[index].preferenceScore + alpha * successWeight))
                preferences[index].usageCount += 1
            } else {
                // Create new preference
                let initialScore = record.wasOnTime ? 0.7 : 0.3
                preferences.append(TimePreference(hour: hour, preferenceScore: initialScore, usageCount: 1))
            }
            
            medicationTimingPreferences[record.medicationName] = preferences
        }
        
        // Save updated patterns
        saveTimingPatterns()
    }
    
    private func calculateAverageDelay(_ records: [AdherenceRecord]) -> TimeInterval {
        let delays = records.compactMap { record -> TimeInterval? in
            guard !record.wasOnTime else { return nil }
            return record.actualTime.timeIntervalSince(record.scheduledTime)
        }
        
        guard !delays.isEmpty else { return 0 }
        return delays.reduce(0, +) / Double(delays.count)
    }
    
    private func prepareTimingFeatures(medication: Medication, scheduledTime: Date, patterns: [TimingPattern]) -> [String: Double] {
        let calendar = Calendar.current
        var features: [String: Double] = [:]
        
        // Time-based features
        features["scheduled_hour"] = Double(calendar.component(.hour, from: scheduledTime))
        features["day_of_week"] = Double(calendar.component(.weekday, from: scheduledTime))
        features["is_weekend"] = [1, 7].contains(calendar.component(.weekday, from: scheduledTime)) ? 1.0 : 0.0
        
        // Historical adherence at this hour
        let hour = calendar.component(.hour, from: scheduledTime)
        let hourPattern = patterns.first { $0.hour == hour && $0.dayOfWeek == nil }
        features["hour_adherence_rate"] = hourPattern?.adherenceRate ?? 0.5
        features["hour_sample_size"] = Double(hourPattern?.sampleSize ?? 0)
        
        // Day-hour specific adherence
        let dayOfWeek = calendar.component(.weekday, from: scheduledTime)
        let dayHourPattern = patterns.first { $0.hour == hour && $0.dayOfWeek == dayOfWeek }
        features["day_hour_adherence_rate"] = dayHourPattern?.adherenceRate ?? 0.5
        
        // Medication characteristics
        features["doses_per_day"] = Double(medication.dosesPerDay ?? 1)
        features["medication_complexity"] = calculateMedicationComplexity(medication.name)
        
        // User preferences
        let preferences = medicationTimingPreferences[medication.name] ?? []
        let hourPreference = preferences.first { abs($0.hour - hour) <= 1 }
        features["user_preference_score"] = hourPreference?.preferenceScore ?? 0.5
        
        // Time since last dose (if available)
        if let lastRecord = trainingData.filter({ $0.medicationName == medication.name }).last {
            features["hours_since_last_dose"] = scheduledTime.timeIntervalSince(lastRecord.actualTime) / 3600.0
        } else {
            features["hours_since_last_dose"] = 24.0 // Default
        }
        
        return features
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
    
    private func retrainModel() async {
        logger.info("Retraining optimal timing model with \(trainingData.count) records")
        
        do {
            // Prepare training data
            let trainingTable = prepareTimingTrainingData()
            
            // Create and train the model
            let regressor = try MLRegressor(trainingData: trainingTable, targetColumn: "optimal_hour_offset")
            
            // Save the model
            let modelURL = getModelURL() ?? getDocumentsDirectory().appendingPathComponent("\(modelName).mlmodel")
            try regressor.write(to: modelURL)
            
            // Load the new model
            model = try MLModel(contentsOf: modelURL)
            
            logger.info("Timing model retrained and saved successfully")
            
        } catch {
            logger.error("Failed to retrain timing model: \(error.localizedDescription)")
        }
    }
    
    private func prepareTimingTrainingData() -> MLDataTable {
        var data: [String: [Any]] = [
            "scheduled_hour": [],
            "day_of_week": [],
            "is_weekend": [],
            "hour_adherence_rate": [],
            "doses_per_day": [],
            "medication_complexity": [],
            "optimal_hour_offset": []
        ]
        
        for record in trainingData {
            let calendar = Calendar.current
            let scheduledHour = calendar.component(.hour, from: record.scheduledTime)
            let actualHour = calendar.component(.hour, from: record.actualTime)
            
            // Calculate optimal offset (difference between actual and scheduled)
            let hourOffset = Double(actualHour - scheduledHour)
            
            // Add features
            data["scheduled_hour"]?.append(scheduledHour)
            data["day_of_week"]?.append(calendar.component(.weekday, from: record.scheduledTime))
            data["is_weekend"]?.append([1, 7].contains(calendar.component(.weekday, from: record.scheduledTime)) ? 1 : 0)
            data["hour_adherence_rate"]?.append(0.5) // Placeholder
            data["doses_per_day"]?.append(1) // Placeholder
            data["medication_complexity"]?.append(calculateMedicationComplexity(record.medicationName))
            
            // Target: optimal hour offset (only for successful adherence)
            if record.wasOnTime {
                data["optimal_hour_offset"]?.append(0.0) // No offset needed
            } else {
                data["optimal_hour_offset"]?.append(hourOffset) // Use actual offset
            }
        }
        
        return try! MLDataTable(dictionary: data)
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

// MARK: - Supporting Types

struct TimingPattern: Codable {
    let hour: Int
    let dayOfWeek: Int? // nil for general hour pattern
    var adherenceRate: Double
    var sampleSize: Int
    var averageDelay: TimeInterval
}

struct TimePreference: Codable {
    let hour: Int
    var preferenceScore: Double // 0.0 to 1.0
    var usageCount: Int
}

// MARK: - Advanced Timing Analysis
extension OptimalTimingModelImpl {
    
    func analyzeTimingEffectiveness(for medication: Medication) -> TimingEffectivenessReport {
        let medicationRecords = trainingData.filter { $0.medicationName == medication.name }
        
        var hourlyStats: [Int: HourlyStats] = [:]
        
        // Analyze each hour
        for hour in 0...23 {
            let hourRecords = medicationRecords.filter {
                Calendar.current.component(.hour, from: $0.scheduledTime) == hour
            }
            
            guard !hourRecords.isEmpty else { continue }
            
            let adherentCount = hourRecords.filter { $0.wasOnTime }.count
            let adherenceRate = Double(adherentCount) / Double(hourRecords.count)
            
            let delays = hourRecords.compactMap { record -> TimeInterval? in
                guard !record.wasOnTime else { return nil }
                return record.actualTime.timeIntervalSince(record.scheduledTime)
            }
            
            let averageDelay = delays.isEmpty ? 0 : delays.reduce(0, +) / Double(delays.count)
            
            hourlyStats[hour] = HourlyStats(
                hour: hour,
                adherenceRate: adherenceRate,
                sampleSize: hourRecords.count,
                averageDelay: averageDelay
            )
        }
        
        // Find optimal hours
        let sortedHours = hourlyStats.values.sorted { $0.adherenceRate > $1.adherenceRate }
        let optimalHours = Array(sortedHours.prefix(3))
        
        return TimingEffectivenessReport(
            medication: medication,
            hourlyStats: hourlyStats,
            optimalHours: optimalHours,
            overallAdherenceRate: Double(medicationRecords.filter { $0.wasOnTime }.count) / Double(medicationRecords.count),
            analysisDate: Date()
        )
    }
    
    func generateTimingRecommendations(for medication: Medication) -> [TimingRecommendation] {
        let effectiveness = analyzeTimingEffectiveness(for: medication)
        var recommendations: [TimingRecommendation] = []
        
        // Recommend optimal hours
        for (index, hourStats) in effectiveness.optimalHours.enumerated() {
            let priority: RecommendationPriority = index == 0 ? .high : (index == 1 ? .medium : .low)
            
            recommendations.append(TimingRecommendation(
                type: .optimalTiming,
                priority: priority,
                title: "Optimal Time: \(formatHour(hourStats.hour))",
                description: "You have \(Int(hourStats.adherenceRate * 100))% adherence rate at this time",
                suggestedTime: hourStats.hour,
                confidence: min(1.0, Double(hourStats.sampleSize) / 10.0)
            ))
        }
        
        // Identify problematic times
        let problematicHours = effectiveness.hourlyStats.values.filter {
            $0.adherenceRate < 0.5 && $0.sampleSize >= 3
        }.sorted { $0.adherenceRate < $1.adherenceRate }
        
        for hourStats in problematicHours.prefix(2) {
            recommendations.append(TimingRecommendation(
                type: .avoidTiming,
                priority: .medium,
                title: "Avoid: \(formatHour(hourStats.hour))",
                description: "Low adherence rate (\(Int(hourStats.adherenceRate * 100))%) at this time",
                suggestedTime: hourStats.hour,
                confidence: min(1.0, Double(hourStats.sampleSize) / 10.0)
            ))
        }
        
        return recommendations
    }
    
    private func formatHour(_ hour: Int) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        
        let calendar = Calendar.current
        let date = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: Date()) ?? Date()
        
        return formatter.string(from: date)
    }
}

// MARK: - Additional Supporting Types

struct HourlyStats {
    let hour: Int
    let adherenceRate: Double
    let sampleSize: Int
    let averageDelay: TimeInterval
}

struct TimingEffectivenessReport {
    let medication: Medication
    let hourlyStats: [Int: HourlyStats]
    let optimalHours: [HourlyStats]
    let overallAdherenceRate: Double
    let analysisDate: Date
}

struct TimingRecommendation {
    let type: TimingRecommendationType
    let priority: RecommendationPriority
    let title: String
    let description: String
    let suggestedTime: Int // Hour of day
    let confidence: Double
}

enum TimingRecommendationType {
    case optimalTiming
    case avoidTiming
    case adjustTiming
}

enum RecommendationPriority {
    case low
    case medium
    case high
}