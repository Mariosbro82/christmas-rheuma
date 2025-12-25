//
//  ReminderPersonalizationModel.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreML
import CreateML
import UserNotifications
import os.log

// MARK: - Reminder Personalization Model Implementation
class ReminderPersonalizationModelImpl: ReminderPersonalizationModel {
    
    private let logger = Logger(subsystem: "InflamAI", category: "ReminderPersonalizationModel")
    private var model: MLModel?
    private var interactionData: [ReminderInteraction] = []
    private let modelName = "ReminderPersonalizationModel"
    
    // Personalization parameters
    private let minimumInteractions = 15
    private let learningRate = 0.1
    private let maxPersonalizationHistory = 100
    
    // User preference tracking
    private var userPreferences: UserPreferences
    private var reminderEffectiveness: [String: ReminderEffectiveness] = [:]
    private var contextualPatterns: [String: ContextualPattern] = [:]
    
    required init() async throws {
        logger.info("Initializing ReminderPersonalizationModel")
        
        // Load user preferences
        userPreferences = loadUserPreferences()
        
        // Load existing model and data
        await loadExistingModel()
        loadInteractionData()
        loadEffectivenessData()
        
        logger.info("ReminderPersonalizationModel initialized")
    }
    
    func personalizeReminder(for medication: Medication, baseReminder: MedicationReminder, context: ReminderContext) async -> PersonalizedReminder {
        logger.info("Personalizing reminder for \(medication.name)")
        
        // Get user's current context and preferences
        let currentContext = await getCurrentContext()
        let medicationPrefs = getMedicationPreferences(for: medication.name)
        
        // Use ML model if available
        if let mlModel = model {
            return await personalizeWithMLModel(mlModel, medication: medication, baseReminder: baseReminder, context: currentContext, preferences: medicationPrefs)
        } else {
            // Use rule-based personalization
            return personalizeWithRules(medication: medication, baseReminder: baseReminder, context: currentContext, preferences: medicationPrefs)
        }
    }
    
    func updateWithInteraction(_ interaction: ReminderInteraction) async {
        logger.info("Recording reminder interaction: \(interaction.type)")
        
        // Add interaction to history
        interactionData.append(interaction)
        
        // Keep only recent interactions
        if interactionData.count > maxPersonalizationHistory {
            interactionData = Array(interactionData.suffix(maxPersonalizationHistory))
        }
        
        // Update effectiveness tracking
        updateEffectivenessTracking(with: interaction)
        
        // Update contextual patterns
        updateContextualPatterns(with: interaction)
        
        // Update user preferences based on interaction
        updateUserPreferences(with: interaction)
        
        // Retrain model if we have enough data
        if interactionData.count >= minimumInteractions {
            await retrainModel()
        }
        
        // Save updated data
        saveInteractionData()
        saveEffectivenessData()
        saveUserPreferences()
    }
    
    // MARK: - Private Methods
    
    private func loadExistingModel() async {
        do {
            if let modelURL = getModelURL() {
                model = try MLModel(contentsOf: modelURL)
                logger.info("Loaded existing personalization model")
            }
        } catch {
            logger.error("Failed to load personalization model: \(error.localizedDescription)")
        }
    }
    
    private func loadUserPreferences() -> UserPreferences {
        if let data = UserDefaults.standard.data(forKey: "user_reminder_preferences"),
           let preferences = try? JSONDecoder().decode(UserPreferences.self, from: data) {
            logger.info("Loaded user preferences")
            return preferences
        }
        
        // Default preferences
        return UserPreferences(
            preferredReminderStyle: .gentle,
            preferredFrequency: .standard,
            enableHapticFeedback: true,
            enableSoundAlerts: true,
            quietHoursStart: 22,
            quietHoursEnd: 7,
            adaptiveScheduling: true,
            contextAwareness: true
        )
    }
    
    private func saveUserPreferences() {
        if let data = try? JSONEncoder().encode(userPreferences) {
            UserDefaults.standard.set(data, forKey: "user_reminder_preferences")
        }
    }
    
    private func loadInteractionData() {
        if let data = UserDefaults.standard.data(forKey: "reminder_interactions"),
           let interactions = try? JSONDecoder().decode([ReminderInteraction].self, from: data) {
            interactionData = interactions
            logger.info("Loaded \(interactions.count) reminder interactions")
        }
    }
    
    private func saveInteractionData() {
        if let data = try? JSONEncoder().encode(interactionData) {
            UserDefaults.standard.set(data, forKey: "reminder_interactions")
        }
    }
    
    private func loadEffectivenessData() {
        if let data = UserDefaults.standard.data(forKey: "reminder_effectiveness"),
           let effectiveness = try? JSONDecoder().decode([String: ReminderEffectiveness].self, from: data) {
            reminderEffectiveness = effectiveness
            logger.info("Loaded effectiveness data for \(effectiveness.count) reminder types")
        }
    }
    
    private func saveEffectivenessData() {
        if let data = try? JSONEncoder().encode(reminderEffectiveness) {
            UserDefaults.standard.set(data, forKey: "reminder_effectiveness")
        }
    }
    
    private func getCurrentContext() async -> ReminderContext {
        // Get current context information
        let calendar = Calendar.current
        let now = Date()
        
        return ReminderContext(
            timeOfDay: calendar.component(.hour, from: now),
            dayOfWeek: calendar.component(.weekday, from: now),
            isWeekend: [1, 7].contains(calendar.component(.weekday, from: now)),
            userActivity: await detectUserActivity(),
            location: await getUserLocation(),
            stressLevel: await getStressLevel(),
            sleepQuality: await getRecentSleepQuality()
        )
    }
    
    private func detectUserActivity() async -> UserActivity {
        // Simplified activity detection
        // In a real implementation, this would use CoreMotion
        let hour = Calendar.current.component(.hour, from: Date())
        
        switch hour {
        case 6...8: return .morning_routine
        case 9...17: return .work
        case 18...20: return .evening_routine
        case 21...23: return .relaxing
        default: return .sleeping
        }
    }
    
    private func getUserLocation() async -> UserLocation {
        // Simplified location detection
        let hour = Calendar.current.component(.hour, from: Date())
        
        switch hour {
        case 9...17: return .work
        case 18...23, 0...8: return .home
        default: return .other
        }
    }
    
    private func getStressLevel() async -> StressLevel {
        // This would integrate with HealthKit heart rate variability
        // For now, return a default value
        return .moderate
    }
    
    private func getRecentSleepQuality() async -> SleepQuality {
        // This would integrate with HealthKit sleep data
        // For now, return a default value
        return .good
    }
    
    private func getMedicationPreferences(for medicationName: String) -> MedicationPreferences {
        // Get medication-specific preferences from interaction history
        let medicationInteractions = interactionData.filter { $0.medicationName == medicationName }
        
        // Analyze successful interactions to determine preferences
        let successfulInteractions = medicationInteractions.filter {
            $0.type == .taken_on_time || $0.type == .taken_early
        }
        
        // Determine preferred reminder style
        var styleScores: [ReminderStyle: Double] = [:]
        for interaction in successfulInteractions {
            let style = interaction.reminderStyle
            styleScores[style, default: 0] += 1.0
        }
        
        let preferredStyle = styleScores.max(by: { $0.value < $1.value })?.key ?? .gentle
        
        // Calculate effectiveness score
        let totalInteractions = medicationInteractions.count
        let successfulCount = successfulInteractions.count
        let effectivenessScore = totalInteractions > 0 ? Double(successfulCount) / Double(totalInteractions) : 0.5
        
        return MedicationPreferences(
            preferredStyle: preferredStyle,
            effectivenessScore: effectivenessScore,
            lastUpdated: Date()
        )
    }
    
    private func personalizeWithMLModel(_ mlModel: MLModel, medication: Medication, baseReminder: MedicationReminder, context: ReminderContext, preferences: MedicationPreferences) async -> PersonalizedReminder {
        do {
            // Prepare features for ML model
            let features = preparePersonalizationFeatures(medication: medication, baseReminder: baseReminder, context: context, preferences: preferences)
            let input = try MLDictionaryFeatureProvider(dictionary: features)
            
            // Make prediction
            let prediction = try mlModel.prediction(from: input)
            
            // Extract personalization parameters
            let styleScore = prediction.featureValue(for: "reminder_style_score")?.doubleValue ?? 0.5
            let urgencyScore = prediction.featureValue(for: "urgency_score")?.doubleValue ?? 0.5
            let frequencyMultiplier = prediction.featureValue(for: "frequency_multiplier")?.doubleValue ?? 1.0
            
            // Convert scores to personalization parameters
            let personalizedStyle = scoreToReminderStyle(styleScore)
            let personalizedUrgency = scoreToUrgencyLevel(urgencyScore)
            
            logger.info("ML personalization: style=\(personalizedStyle), urgency=\(personalizedUrgency)")
            
            return PersonalizedReminder(
                baseReminder: baseReminder,
                style: personalizedStyle,
                urgency: personalizedUrgency,
                customMessage: generateCustomMessage(for: medication, style: personalizedStyle, context: context),
                adaptiveFeatures: generateAdaptiveFeatures(context: context, preferences: preferences),
                confidence: 0.8
            )
            
        } catch {
            logger.error("ML personalization failed: \(error.localizedDescription)")
            return personalizeWithRules(medication: medication, baseReminder: baseReminder, context: context, preferences: preferences)
        }
    }
    
    private func personalizeWithRules(medication: Medication, baseReminder: MedicationReminder, context: ReminderContext, preferences: MedicationPreferences) -> PersonalizedReminder {
        logger.info("Using rule-based personalization")
        
        var personalizedStyle = preferences.preferredStyle
        var urgencyLevel = ReminderUrgency.normal
        
        // Adjust based on context
        if context.stressLevel == .high {
            personalizedStyle = .gentle // Use gentler reminders when stressed
        }
        
        if context.userActivity == .sleeping {
            personalizedStyle = .silent // Silent reminders during sleep
            urgencyLevel = .low
        } else if context.userActivity == .work {
            personalizedStyle = .discrete // Discrete reminders at work
        }
        
        // Adjust based on medication importance
        if medication.isImportant {
            urgencyLevel = .high
        }
        
        // Adjust based on time sensitivity
        let timeUntilNext = baseReminder.scheduledTime.timeIntervalSinceNow
        if timeUntilNext < 3600 { // Less than 1 hour
            urgencyLevel = .high
        }
        
        // Generate adaptive features
        let adaptiveFeatures = generateAdaptiveFeatures(context: context, preferences: preferences)
        
        return PersonalizedReminder(
            baseReminder: baseReminder,
            style: personalizedStyle,
            urgency: urgencyLevel,
            customMessage: generateCustomMessage(for: medication, style: personalizedStyle, context: context),
            adaptiveFeatures: adaptiveFeatures,
            confidence: 0.6
        )
    }
    
    private func generateCustomMessage(for medication: Medication, style: ReminderStyle, context: ReminderContext) -> String {
        let medicationName = medication.name
        let timeContext = getTimeContext(context.timeOfDay)
        
        switch style {
        case .gentle:
            return "\(timeContext) It's time for your \(medicationName). Take care of yourself! 💊"
        case .firm:
            return "Important: Take your \(medicationName) now. Your health depends on it."
        case .encouraging:
            return "You're doing great! Time for your \(medicationName) to keep feeling your best! 🌟"
        case .discrete:
            return "\(medicationName) reminder"
        case .silent:
            return "" // No message for silent reminders
        }
    }
    
    private func getTimeContext(_ hour: Int) -> String {
        switch hour {
        case 5...11: return "Good morning!"
        case 12...17: return "Good afternoon!"
        case 18...21: return "Good evening!"
        default: return "Hello!"
        }
    }
    
    private func generateAdaptiveFeatures(context: ReminderContext, preferences: MedicationPreferences) -> [AdaptiveFeature] {
        var features: [AdaptiveFeature] = []
        
        // Context-based features
        if context.stressLevel == .high {
            features.append(AdaptiveFeature(
                type: .stress_adaptation,
                value: 1.0,
                description: "Gentle reminder due to high stress"
            ))
        }
        
        if context.userActivity == .work {
            features.append(AdaptiveFeature(
                type: .location_adaptation,
                value: 1.0,
                description: "Discrete reminder for work environment"
            ))
        }
        
        // Sleep quality adaptation
        if context.sleepQuality == .poor {
            features.append(AdaptiveFeature(
                type: .sleep_adaptation,
                value: 1.0,
                description: "Extra gentle reminder due to poor sleep"
            ))
        }
        
        // Time-based adaptation
        if isQuietHours(context.timeOfDay) {
            features.append(AdaptiveFeature(
                type: .time_adaptation,
                value: 1.0,
                description: "Silent reminder during quiet hours"
            ))
        }
        
        return features
    }
    
    private func isQuietHours(_ hour: Int) -> Bool {
        let start = userPreferences.quietHoursStart
        let end = userPreferences.quietHoursEnd
        
        if start < end {
            return hour >= start || hour < end
        } else {
            return hour >= start && hour < end
        }
    }
    
    private func updateEffectivenessTracking(with interaction: ReminderInteraction) {
        let key = "\(interaction.reminderStyle)_\(interaction.medicationName)"
        var effectiveness = reminderEffectiveness[key] ?? ReminderEffectiveness(
            reminderStyle: interaction.reminderStyle,
            medicationName: interaction.medicationName,
            successRate: 0.5,
            totalInteractions: 0,
            lastUpdated: Date()
        )
        
        // Update success rate using exponential moving average
        let isSuccess = interaction.type == .taken_on_time || interaction.type == .taken_early
        let newSuccessValue = isSuccess ? 1.0 : 0.0
        
        effectiveness.successRate = (1 - learningRate) * effectiveness.successRate + learningRate * newSuccessValue
        effectiveness.totalInteractions += 1
        effectiveness.lastUpdated = Date()
        
        reminderEffectiveness[key] = effectiveness
    }
    
    private func updateContextualPatterns(with interaction: ReminderInteraction) {
        let contextKey = "\(interaction.timeOfDay)_\(interaction.dayOfWeek)"
        var pattern = contextualPatterns[contextKey] ?? ContextualPattern(
            timeOfDay: interaction.timeOfDay,
            dayOfWeek: interaction.dayOfWeek,
            successRate: 0.5,
            preferredStyle: .gentle,
            sampleSize: 0
        )
        
        // Update success rate
        let isSuccess = interaction.type == .taken_on_time || interaction.type == .taken_early
        let newSuccessValue = isSuccess ? 1.0 : 0.0
        
        pattern.successRate = (1 - learningRate) * pattern.successRate + learningRate * newSuccessValue
        pattern.sampleSize += 1
        
        // Update preferred style based on successful interactions
        if isSuccess {
            pattern.preferredStyle = interaction.reminderStyle
        }
        
        contextualPatterns[contextKey] = pattern
    }
    
    private func updateUserPreferences(with interaction: ReminderInteraction) {
        // Update global preferences based on successful interactions
        let isSuccess = interaction.type == .taken_on_time || interaction.type == .taken_early
        
        if isSuccess {
            // Reinforce successful reminder style
            if interaction.reminderStyle != userPreferences.preferredReminderStyle {
                // Gradually shift preference towards successful style
                // This is a simplified approach - in practice, you'd want more sophisticated preference learning
            }
        }
    }
    
    private func preparePersonalizationFeatures(medication: Medication, baseReminder: MedicationReminder, context: ReminderContext, preferences: MedicationPreferences) -> [String: Double] {
        var features: [String: Double] = [:]
        
        // Context features
        features["time_of_day"] = Double(context.timeOfDay)
        features["day_of_week"] = Double(context.dayOfWeek)
        features["is_weekend"] = context.isWeekend ? 1.0 : 0.0
        features["stress_level"] = stressLevelToDouble(context.stressLevel)
        features["sleep_quality"] = sleepQualityToDouble(context.sleepQuality)
        features["user_activity"] = userActivityToDouble(context.userActivity)
        features["location"] = userLocationToDouble(context.location)
        
        // Medication features
        features["medication_importance"] = medication.isImportant ? 1.0 : 0.0
        features["doses_per_day"] = Double(medication.dosesPerDay ?? 1)
        
        // User preference features
        features["preferred_style"] = reminderStyleToDouble(preferences.preferredStyle)
        features["effectiveness_score"] = preferences.effectivenessScore
        
        // Historical effectiveness
        let styleKey = "\(preferences.preferredStyle)_\(medication.name)"
        features["historical_success_rate"] = reminderEffectiveness[styleKey]?.successRate ?? 0.5
        
        // Contextual pattern effectiveness
        let contextKey = "\(context.timeOfDay)_\(context.dayOfWeek)"
        features["context_success_rate"] = contextualPatterns[contextKey]?.successRate ?? 0.5
        
        return features
    }
    
    // Helper conversion functions
    private func stressLevelToDouble(_ level: StressLevel) -> Double {
        switch level {
        case .low: return 0.2
        case .moderate: return 0.5
        case .high: return 0.8
        }
    }
    
    private func sleepQualityToDouble(_ quality: SleepQuality) -> Double {
        switch quality {
        case .poor: return 0.2
        case .fair: return 0.5
        case .good: return 0.8
        }
    }
    
    private func userActivityToDouble(_ activity: UserActivity) -> Double {
        switch activity {
        case .sleeping: return 0.1
        case .morning_routine: return 0.3
        case .work: return 0.5
        case .evening_routine: return 0.7
        case .relaxing: return 0.9
        }
    }
    
    private func userLocationToDouble(_ location: UserLocation) -> Double {
        switch location {
        case .home: return 0.3
        case .work: return 0.6
        case .other: return 0.9
        }
    }
    
    private func reminderStyleToDouble(_ style: ReminderStyle) -> Double {
        switch style {
        case .silent: return 0.1
        case .discrete: return 0.3
        case .gentle: return 0.5
        case .encouraging: return 0.7
        case .firm: return 0.9
        }
    }
    
    private func scoreToReminderStyle(_ score: Double) -> ReminderStyle {
        switch score {
        case 0.0..<0.2: return .silent
        case 0.2..<0.4: return .discrete
        case 0.4..<0.6: return .gentle
        case 0.6..<0.8: return .encouraging
        default: return .firm
        }
    }
    
    private func scoreToUrgencyLevel(_ score: Double) -> ReminderUrgency {
        switch score {
        case 0.0..<0.33: return .low
        case 0.33..<0.67: return .normal
        default: return .high
        }
    }
    
    private func retrainModel() async {
        logger.info("Retraining personalization model with \(interactionData.count) interactions")
        
        do {
            // Prepare training data
            let trainingTable = preparePersonalizationTrainingData()
            
            // Create and train the model
            let classifier = try MLClassifier(trainingData: trainingTable, targetColumn: "reminder_effectiveness")
            
            // Save the model
            let modelURL = getModelURL() ?? getDocumentsDirectory().appendingPathComponent("\(modelName).mlmodel")
            try classifier.write(to: modelURL)
            
            // Load the new model
            model = try MLModel(contentsOf: modelURL)
            
            logger.info("Personalization model retrained and saved successfully")
            
        } catch {
            logger.error("Failed to retrain personalization model: \(error.localizedDescription)")
        }
    }
    
    private func preparePersonalizationTrainingData() -> MLDataTable {
        var data: [String: [Any]] = [
            "time_of_day": [],
            "day_of_week": [],
            "is_weekend": [],
            "stress_level": [],
            "sleep_quality": [],
            "user_activity": [],
            "reminder_style": [],
            "reminder_effectiveness": []
        ]
        
        for interaction in interactionData {
            data["time_of_day"]?.append(interaction.timeOfDay)
            data["day_of_week"]?.append(interaction.dayOfWeek)
            data["is_weekend"]?.append([1, 7].contains(interaction.dayOfWeek) ? 1 : 0)
            data["stress_level"]?.append(0.5) // Placeholder
            data["sleep_quality"]?.append(0.5) // Placeholder
            data["user_activity"]?.append(0.5) // Placeholder
            data["reminder_style"]?.append(reminderStyleToDouble(interaction.reminderStyle))
            
            // Target: effectiveness (success/failure)
            let isEffective = interaction.type == .taken_on_time || interaction.type == .taken_early
            data["reminder_effectiveness"]?.append(isEffective ? 1 : 0)
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

struct ReminderContext {
    let timeOfDay: Int
    let dayOfWeek: Int
    let isWeekend: Bool
    let userActivity: UserActivity
    let location: UserLocation
    let stressLevel: StressLevel
    let sleepQuality: SleepQuality
}

struct MedicationPreferences {
    let preferredStyle: ReminderStyle
    let effectivenessScore: Double
    let lastUpdated: Date
}

struct ContextualPattern: Codable {
    let timeOfDay: Int
    let dayOfWeek: Int
    var successRate: Double
    var preferredStyle: ReminderStyle
    var sampleSize: Int
}

enum UserActivity {
    case sleeping
    case morning_routine
    case work
    case evening_routine
    case relaxing
}

enum UserLocation {
    case home
    case work
    case other
}

enum StressLevel {
    case low
    case moderate
    case high
}

enum SleepQuality {
    case poor
    case fair
    case good
}

enum ReminderUrgency {
    case low
    case normal
    case high
}

// MARK: - Advanced Personalization Features
extension ReminderPersonalizationModelImpl {
    
    func generatePersonalizationInsights() -> [PersonalizationInsight] {
        var insights: [PersonalizationInsight] = []
        
        // Analyze most effective reminder styles
        let styleEffectiveness = analyzeStyleEffectiveness()
        if let bestStyle = styleEffectiveness.max(by: { $0.value < $1.value }) {
            insights.append(PersonalizationInsight(
                type: .optimal_style,
                title: "Most Effective Reminder Style",
                description: "\(bestStyle.key.rawValue.capitalized) reminders work best for you (\(Int(bestStyle.value * 100))% success rate)",
                confidence: min(1.0, Double(getTotalInteractions()) / 50.0),
                actionable: true
            ))
        }
        
        // Analyze time-based patterns
        let timePatterns = analyzeTimePatterns()
        if let bestTime = timePatterns.max(by: { $0.value < $1.value }) {
            insights.append(PersonalizationInsight(
                type: .optimal_timing,
                title: "Best Reminder Times",
                description: "You respond best to reminders around \(formatHour(bestTime.key)) (\(Int(bestTime.value * 100))% success rate)",
                confidence: 0.8,
                actionable: true
            ))
        }
        
        // Analyze context-based patterns
        if let contextInsight = analyzeContextPatterns() {
            insights.append(contextInsight)
        }
        
        return insights
    }
    
    private func analyzeStyleEffectiveness() -> [ReminderStyle: Double] {
        var styleStats: [ReminderStyle: (success: Int, total: Int)] = [:]
        
        for interaction in interactionData {
            let style = interaction.reminderStyle
            let isSuccess = interaction.type == .taken_on_time || interaction.type == .taken_early
            
            let current = styleStats[style] ?? (success: 0, total: 0)
            styleStats[style] = (success: current.success + (isSuccess ? 1 : 0), total: current.total + 1)
        }
        
        return styleStats.mapValues { stats in
            stats.total > 0 ? Double(stats.success) / Double(stats.total) : 0.0
        }
    }
    
    private func analyzeTimePatterns() -> [Int: Double] {
        var timeStats: [Int: (success: Int, total: Int)] = [:]
        
        for interaction in interactionData {
            let hour = interaction.timeOfDay
            let isSuccess = interaction.type == .taken_on_time || interaction.type == .taken_early
            
            let current = timeStats[hour] ?? (success: 0, total: 0)
            timeStats[hour] = (success: current.success + (isSuccess ? 1 : 0), total: current.total + 1)
        }
        
        return timeStats.compactMapValues { stats in
            stats.total >= 3 ? Double(stats.success) / Double(stats.total) : nil
        }
    }
    
    private func analyzeContextPatterns() -> PersonalizationInsight? {
        // Analyze weekend vs weekday patterns
        let weekdaySuccess = interactionData.filter { ![1, 7].contains($0.dayOfWeek) }
        let weekendSuccess = interactionData.filter { [1, 7].contains($0.dayOfWeek) }
        
        let weekdayRate = calculateSuccessRate(weekdaySuccess)
        let weekendRate = calculateSuccessRate(weekendSuccess)
        
        if abs(weekdayRate - weekendRate) > 0.2 {
            let better = weekdayRate > weekendRate ? "weekdays" : "weekends"
            let rate = max(weekdayRate, weekendRate)
            
            return PersonalizationInsight(
                type: .context_pattern,
                title: "Day Type Pattern",
                description: "You respond better to reminders on \(better) (\(Int(rate * 100))% vs \(Int(min(weekdayRate, weekendRate) * 100))%)",
                confidence: 0.7,
                actionable: true
            )
        }
        
        return nil
    }
    
    private func calculateSuccessRate(_ interactions: [ReminderInteraction]) -> Double {
        guard !interactions.isEmpty else { return 0.0 }
        
        let successCount = interactions.filter { $0.type == .taken_on_time || $0.type == .taken_early }.count
        return Double(successCount) / Double(interactions.count)
    }
    
    private func getTotalInteractions() -> Int {
        return interactionData.count
    }
    
    private func formatHour(_ hour: Int) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h a"
        
        let calendar = Calendar.current
        let date = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: Date()) ?? Date()
        
        return formatter.string(from: date)
    }
}

// MARK: - Additional Supporting Types

struct PersonalizationInsight {
    let type: PersonalizationInsightType
    let title: String
    let description: String
    let confidence: Double
    let actionable: Bool
}

enum PersonalizationInsightType {
    case optimal_style
    case optimal_timing
    case context_pattern
    case effectiveness_trend
}