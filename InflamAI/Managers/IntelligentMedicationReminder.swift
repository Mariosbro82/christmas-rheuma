//
//  IntelligentMedicationReminder.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import UserNotifications
import CoreML
import Combine
import HealthKit
import os.log

// MARK: - Intelligent Medication Reminder Manager
@MainActor
class IntelligentMedicationReminder: ObservableObject {
    
    // MARK: - Published Properties
    @Published var upcomingReminders: [MedicationReminder] = []
    @Published var adherenceScore: Double = 0.0
    @Published var missedDoses: [MissedDose] = []
    @Published var adaptiveSchedule: [AdaptiveScheduleEntry] = []
    @Published var isLearningEnabled: Bool = true
    @Published var personalizedInsights: [PersonalizedInsight] = []
    @Published var reminderEffectiveness: [ReminderEffectiveness] = []
    
    // MARK: - Private Properties
    private let logger = Logger(subsystem: "InflamAI", category: "IntelligentMedicationReminder")
    private let notificationCenter = UNUserNotificationCenter.current()
    private let healthStore = HKHealthStore()
    private var cancellables = Set<AnyCancellable>()
    
    // ML Models
    private var adherencePredictionModel: AdherencePredictionModel?
    private var optimalTimingModel: OptimalTimingModel?
    private var reminderPersonalizationModel: ReminderPersonalizationModel?
    
    // Data Storage
    private let userDefaults = UserDefaults.standard
    private let adherenceHistoryKey = "adherence_history"
    private let reminderHistoryKey = "reminder_history"
    private let learningDataKey = "learning_data"
    
    // Learning Parameters
    private var learningData: LearningData
    private let minimumDataPoints = 14 // Minimum days of data for ML predictions
    
    // MARK: - Initialization
    init() {
        self.learningData = loadLearningData()
        setupNotifications()
        loadMLModels()
        startAdaptiveLearning()
        
        logger.info("IntelligentMedicationReminder initialized")
    }
    
    // MARK: - Public Methods
    
    func addMedication(_ medication: Medication, schedule: MedicationSchedule) async {
        logger.info("Adding medication: \(medication.name)")
        
        // Create initial reminders based on schedule
        let reminders = createInitialReminders(for: medication, schedule: schedule)
        
        // Store reminders
        await storeReminders(reminders)
        
        // Schedule notifications
        await scheduleNotifications(for: reminders)
        
        // Update published properties
        await updateUpcomingReminders()
        
        logger.info("Added \(reminders.count) reminders for \(medication.name)")
    }
    
    func recordMedicationTaken(_ reminderId: String, takenAt: Date, actualDose: Double? = nil) async {
        logger.info("Recording medication taken for reminder: \(reminderId)")
        
        // Find the reminder
        guard let reminderIndex = upcomingReminders.firstIndex(where: { $0.id == reminderId }) else {
            logger.error("Reminder not found: \(reminderId)")
            return
        }
        
        let reminder = upcomingReminders[reminderIndex]
        
        // Record adherence
        let adherenceRecord = AdherenceRecord(
            reminderId: reminderId,
            medicationName: reminder.medication.name,
            scheduledTime: reminder.scheduledTime,
            actualTime: takenAt,
            scheduledDose: reminder.dose,
            actualDose: actualDose ?? reminder.dose,
            wasOnTime: abs(takenAt.timeIntervalSince(reminder.scheduledTime)) <= 1800 // 30 minutes tolerance
        )
        
        // Update learning data
        learningData.adherenceHistory.append(adherenceRecord)
        saveLearningData()
        
        // Remove from upcoming reminders
        upcomingReminders.remove(at: reminderIndex)
        
        // Update adherence score
        await updateAdherenceScore()
        
        // Learn from this interaction
        await learnFromAdherence(adherenceRecord)
        
        // Update adaptive schedule if needed
        await updateAdaptiveSchedule(for: reminder.medication)
        
        logger.info("Medication taken recorded successfully")
    }
    
    func recordMissedDose(_ reminderId: String, reason: MissedDoseReason? = nil) async {
        logger.info("Recording missed dose for reminder: \(reminderId)")
        
        guard let reminderIndex = upcomingReminders.firstIndex(where: { $0.id == reminderId }) else {
            logger.error("Reminder not found: \(reminderId)")
            return
        }
        
        let reminder = upcomingReminders[reminderIndex]
        
        // Create missed dose record
        let missedDose = MissedDose(
            reminderId: reminderId,
            medicationName: reminder.medication.name,
            scheduledTime: reminder.scheduledTime,
            missedTime: Date(),
            reason: reason,
            severity: calculateMissedDoseSeverity(for: reminder.medication)
        )
        
        // Add to missed doses
        missedDoses.append(missedDose)
        
        // Update learning data
        learningData.missedDoses.append(missedDose)
        saveLearningData()
        
        // Remove from upcoming reminders
        upcomingReminders.remove(at: reminderIndex)
        
        // Update adherence score
        await updateAdherenceScore()
        
        // Learn from this miss
        await learnFromMissedDose(missedDose)
        
        // Generate adaptive recommendations
        await generateAdaptiveRecommendations(for: missedDose)
        
        logger.info("Missed dose recorded successfully")
    }
    
    func generatePersonalizedReminder(for medication: Medication, at scheduledTime: Date) async -> PersonalizedReminder {
        logger.info("Generating personalized reminder for \(medication.name)")
        
        // Analyze user patterns
        let userPatterns = analyzeUserPatterns(for: medication)
        
        // Predict optimal reminder style
        let reminderStyle = await predictOptimalReminderStyle(for: medication, patterns: userPatterns)
        
        // Generate contextual message
        let message = generateContextualMessage(for: medication, style: reminderStyle, time: scheduledTime)
        
        // Determine optimal notification timing
        let optimalTiming = await predictOptimalTiming(for: medication, scheduledTime: scheduledTime)
        
        return PersonalizedReminder(
            medication: medication,
            scheduledTime: scheduledTime,
            optimalTime: optimalTiming,
            style: reminderStyle,
            message: message,
            priority: calculateReminderPriority(for: medication),
            adaptiveFeatures: generateAdaptiveFeatures(for: medication, patterns: userPatterns)
        )
    }
    
    func predictAdherenceRisk(for medication: Medication, timeframe: TimeInterval = 7 * 24 * 60 * 60) async -> AdherenceRiskAssessment {
        logger.info("Predicting adherence risk for \(medication.name)")
        
        guard let model = adherencePredictionModel else {
            logger.warning("Adherence prediction model not available")
            return AdherenceRiskAssessment(medication: medication, riskLevel: .unknown, confidence: 0.0, factors: [], recommendations: [])
        }
        
        // Prepare input features
        let features = prepareAdherenceFeatures(for: medication, timeframe: timeframe)
        
        // Make prediction
        let prediction = await model.predict(features: features)
        
        // Analyze risk factors
        let riskFactors = analyzeAdherenceRiskFactors(for: medication)
        
        // Generate recommendations
        let recommendations = generateAdherenceRecommendations(riskLevel: prediction.riskLevel, factors: riskFactors)
        
        return AdherenceRiskAssessment(
            medication: medication,
            riskLevel: prediction.riskLevel,
            confidence: prediction.confidence,
            factors: riskFactors,
            recommendations: recommendations
        )
    }
    
    // MARK: - Private Methods
    
    private func setupNotifications() {
        notificationCenter.requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                self.logger.info("Notification permission granted")
            } else {
                self.logger.error("Notification permission denied: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    private func loadMLModels() {
        Task {
            do {
                // Load pre-trained models or initialize new ones
                adherencePredictionModel = try await AdherencePredictionModel()
                optimalTimingModel = try await OptimalTimingModel()
                reminderPersonalizationModel = try await ReminderPersonalizationModel()
                
                logger.info("ML models loaded successfully")
            } catch {
                logger.error("Failed to load ML models: \(error.localizedDescription)")
            }
        }
    }
    
    private func startAdaptiveLearning() {
        // Start periodic learning updates
        Timer.publish(every: 24 * 60 * 60, on: .main, in: .common) // Daily
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.performAdaptiveLearning()
                }
            }
            .store(in: &cancellables)
    }
    
    private func performAdaptiveLearning() async {
        guard isLearningEnabled && learningData.adherenceHistory.count >= minimumDataPoints else {
            logger.info("Insufficient data for adaptive learning")
            return
        }
        
        logger.info("Performing adaptive learning")
        
        // Update ML models with new data
        await updateMLModels()
        
        // Analyze patterns and generate insights
        await generatePersonalizedInsights()
        
        // Update adaptive schedules
        await updateAllAdaptiveSchedules()
        
        logger.info("Adaptive learning completed")
    }
    
    private func updateMLModels() async {
        // Update adherence prediction model
        if let model = adherencePredictionModel {
            await model.updateWithNewData(learningData.adherenceHistory)
        }
        
        // Update optimal timing model
        if let model = optimalTimingModel {
            await model.updateWithNewData(learningData.adherenceHistory)
        }
        
        // Update personalization model
        if let model = reminderPersonalizationModel {
            await model.updateWithNewData(learningData.reminderInteractions)
        }
    }
    
    private func generatePersonalizedInsights() async {
        var insights: [PersonalizedInsight] = []
        
        // Analyze adherence patterns
        let adherencePatterns = analyzeAdherencePatterns()
        if let pattern = adherencePatterns.mostSignificant {
            insights.append(PersonalizedInsight(
                type: .adherencePattern,
                title: "Adherence Pattern Detected",
                description: pattern.description,
                actionable: true,
                recommendations: pattern.recommendations
            ))
        }
        
        // Analyze timing preferences
        let timingPreferences = analyzeTimingPreferences()
        if let preference = timingPreferences.strongestPreference {
            insights.append(PersonalizedInsight(
                type: .timingPreference,
                title: "Optimal Timing Identified",
                description: preference.description,
                actionable: true,
                recommendations: preference.recommendations
            ))
        }
        
        // Analyze missed dose patterns
        let missedDosePatterns = analyzeMissedDosePatterns()
        if let pattern = missedDosePatterns.mostConcerning {
            insights.append(PersonalizedInsight(
                type: .riskFactor,
                title: "Risk Factor Identified",
                description: pattern.description,
                actionable: true,
                recommendations: pattern.recommendations
            ))
        }
        
        personalizedInsights = insights
        logger.info("Generated \(insights.count) personalized insights")
    }
    
    private func createInitialReminders(for medication: Medication, schedule: MedicationSchedule) -> [MedicationReminder] {
        var reminders: [MedicationReminder] = []
        
        let calendar = Calendar.current
        let now = Date()
        
        // Generate reminders for the next 30 days
        for dayOffset in 0..<30 {
            guard let day = calendar.date(byAdding: .day, value: dayOffset, to: now) else { continue }
            
            for timeSlot in schedule.timeSlots {
                guard let reminderTime = calendar.date(bySettingHour: timeSlot.hour, minute: timeSlot.minute, second: 0, of: day) else { continue }
                
                // Skip past times for today
                if dayOffset == 0 && reminderTime < now { continue }
                
                let reminder = MedicationReminder(
                    id: UUID().uuidString,
                    medication: medication,
                    scheduledTime: reminderTime,
                    dose: timeSlot.dose,
                    instructions: timeSlot.instructions,
                    isRecurring: schedule.isRecurring,
                    priority: .normal
                )
                
                reminders.append(reminder)
            }
        }
        
        return reminders
    }
    
    private func scheduleNotifications(for reminders: [MedicationReminder]) async {
        for reminder in reminders {
            let content = UNMutableNotificationContent()
            content.title = "Medication Reminder"
            content.body = "Time to take \(reminder.medication.name)"
            content.sound = .default
            content.userInfo = ["reminderId": reminder.id]
            
            let trigger = UNCalendarNotificationTrigger(
                dateMatching: Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: reminder.scheduledTime),
                repeats: false
            )
            
            let request = UNNotificationRequest(
                identifier: reminder.id,
                content: content,
                trigger: trigger
            )
            
            do {
                try await notificationCenter.add(request)
            } catch {
                logger.error("Failed to schedule notification for reminder \(reminder.id): \(error.localizedDescription)")
            }
        }
    }
    
    private func loadLearningData() -> LearningData {
        if let data = userDefaults.data(forKey: learningDataKey),
           let learningData = try? JSONDecoder().decode(LearningData.self, from: data) {
            return learningData
        }
        return LearningData()
    }
    
    private func saveLearningData() {
        if let data = try? JSONEncoder().encode(learningData) {
            userDefaults.set(data, forKey: learningDataKey)
        }
    }
    
    // Additional helper methods would be implemented here...
    // This is a comprehensive foundation for the intelligent medication reminder system
}

// MARK: - Supporting Types

struct MedicationReminder: Identifiable, Codable {
    let id: String
    let medication: Medication
    let scheduledTime: Date
    let dose: Double
    let instructions: String?
    let isRecurring: Bool
    let priority: ReminderPriority
}

struct MedicationSchedule: Codable {
    let timeSlots: [TimeSlot]
    let isRecurring: Bool
    let startDate: Date
    let endDate: Date?
}

struct TimeSlot: Codable {
    let hour: Int
    let minute: Int
    let dose: Double
    let instructions: String?
}

enum ReminderPriority: String, Codable, CaseIterable {
    case low = "Low"
    case normal = "Normal"
    case high = "High"
    case critical = "Critical"
}

struct AdherenceRecord: Codable {
    let id = UUID()
    let reminderId: String
    let medicationName: String
    let scheduledTime: Date
    let actualTime: Date
    let scheduledDose: Double
    let actualDose: Double
    let wasOnTime: Bool
}

struct MissedDose: Identifiable, Codable {
    let id = UUID()
    let reminderId: String
    let medicationName: String
    let scheduledTime: Date
    let missedTime: Date
    let reason: MissedDoseReason?
    let severity: MissedDoseSeverity
}

enum MissedDoseReason: String, Codable, CaseIterable {
    case forgot = "Forgot"
    case sideEffects = "Side Effects"
    case feelingBetter = "Feeling Better"
    case tooExpensive = "Too Expensive"
    case inconvenient = "Inconvenient"
    case other = "Other"
}

enum MissedDoseSeverity: String, Codable, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case critical = "Critical"
}

struct PersonalizedReminder {
    let medication: Medication
    let scheduledTime: Date
    let optimalTime: Date
    let style: ReminderStyle
    let message: String
    let priority: ReminderPriority
    let adaptiveFeatures: [AdaptiveFeature]
}

enum ReminderStyle: String, CaseIterable {
    case gentle = "Gentle"
    case standard = "Standard"
    case urgent = "Urgent"
    case motivational = "Motivational"
    case educational = "Educational"
}

struct AdaptiveFeature {
    let type: AdaptiveFeatureType
    let value: String
    let confidence: Double
}

enum AdaptiveFeatureType: String, CaseIterable {
    case optimalTiming = "Optimal Timing"
    case preferredStyle = "Preferred Style"
    case contextualCue = "Contextual Cue"
    case motivationalMessage = "Motivational Message"
}

struct AdherenceRiskAssessment {
    let medication: Medication
    let riskLevel: RiskLevel
    let confidence: Double
    let factors: [RiskFactor]
    let recommendations: [String]
}

enum RiskLevel: String, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
    case unknown = "Unknown"
}

struct RiskFactor {
    let type: RiskFactorType
    let description: String
    let impact: Double // 0.0 to 1.0
}

enum RiskFactorType: String, CaseIterable {
    case timePattern = "Time Pattern"
    case missedDoseHistory = "Missed Dose History"
    case sideEffects = "Side Effects"
    case complexity = "Complexity"
    case lifestyle = "Lifestyle"
}

struct LearningData: Codable {
    var adherenceHistory: [AdherenceRecord] = []
    var missedDoses: [MissedDose] = []
    var reminderInteractions: [ReminderInteraction] = []
    var userPreferences: UserPreferences = UserPreferences()
}

struct ReminderInteraction: Codable {
    let id = UUID()
    let reminderId: String
    let interactionType: InteractionType
    let timestamp: Date
    let effectiveness: Double // 0.0 to 1.0
}

enum InteractionType: String, Codable, CaseIterable {
    case viewed = "Viewed"
    case dismissed = "Dismissed"
    case snoozed = "Snoozed"
    case taken = "Taken"
    case missed = "Missed"
}

struct UserPreferences: Codable {
    var preferredReminderStyle: ReminderStyle = .standard
    var preferredTiming: [Int] = [] // Hours of day
    var enableAdaptiveLearning: Bool = true
    var reminderFrequency: ReminderFrequency = .standard
}

enum ReminderFrequency: String, Codable, CaseIterable {
    case minimal = "Minimal"
    case standard = "Standard"
    case frequent = "Frequent"
}

struct PersonalizedInsight: Identifiable {
    let id = UUID()
    let type: InsightType
    let title: String
    let description: String
    let actionable: Bool
    let recommendations: [String]
}

enum InsightType: String, CaseIterable {
    case adherencePattern = "Adherence Pattern"
    case timingPreference = "Timing Preference"
    case riskFactor = "Risk Factor"
    case improvement = "Improvement"
}

struct AdaptiveScheduleEntry: Identifiable {
    let id = UUID()
    let medication: Medication
    let originalTime: Date
    let adaptedTime: Date
    let reason: String
    let confidence: Double
}

struct ReminderEffectiveness: Identifiable {
    let id = UUID()
    let reminderStyle: ReminderStyle
    let successRate: Double
    let averageResponseTime: TimeInterval
    let userSatisfaction: Double
}

// MARK: - ML Model Protocols

protocol AdherencePredictionModel {
    init() async throws
    func predict(features: [String: Double]) async -> (riskLevel: RiskLevel, confidence: Double)
    func updateWithNewData(_ data: [AdherenceRecord]) async
}

protocol OptimalTimingModel {
    init() async throws
    func predictOptimalTime(for medication: Medication, scheduledTime: Date) async -> Date
    func updateWithNewData(_ data: [AdherenceRecord]) async
}

protocol ReminderPersonalizationModel {
    init() async throws
    func predictOptimalStyle(for medication: Medication, userPatterns: [String: Any]) async -> ReminderStyle
    func updateWithNewData(_ data: [ReminderInteraction]) async
}