//
//  IntelligentMedicationReminder.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import UserNotifications
import Combine
import CoreML

// MARK: - Intelligent Medication Reminder
@MainActor
class IntelligentMedicationReminder: NSObject, ObservableObject {
    
    // MARK: - Published Properties
    @Published var activeReminders: [MedicationReminder] = []
    @Published var adherenceScore: Double = 0.0
    @Published var adaptiveSchedule: [AdaptiveScheduleEntry] = []
    @Published var isLearning = false
    @Published var lastOptimization: Date?
    
    // MARK: - Private Properties
    private var medications: [Medication] = []
    private var adherenceHistory: [AdherenceRecord] = []
    private var userBehaviorPatterns: UserBehaviorPattern?
    private var mlModel: MLModel?
    private let notificationCenter = UNUserNotificationCenter.current()
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Configuration
    private let maxMissedDoses = 3
    private let adaptationThreshold = 0.7
    private let learningPeriodDays = 14
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupNotificationDelegate()
        loadMedicationData()
        loadAdherenceHistory()
        startBehaviorLearning()
    }
    
    // MARK: - Public Methods
    
    func addMedication(_ medication: Medication) async {
        medications.append(medication)
        await createRemindersForMedication(medication)
        await optimizeSchedule()
    }
    
    func removeMedication(_ medicationId: String) async {
        medications.removeAll { $0.id == medicationId }
        activeReminders.removeAll { $0.medicationId == medicationId }
        await cancelNotifications(for: medicationId)
    }
    
    func recordMedicationTaken(_ medicationId: String, timestamp: Date = Date()) async {
        let record = AdherenceRecord(
            medicationId: medicationId,
            scheduledTime: timestamp,
            actualTime: timestamp,
            wasTaken: true,
            delayMinutes: 0
        )
        
        adherenceHistory.append(record)
        await updateAdherenceScore()
        await learnFromAdherence(record)
        
        // Remove the completed reminder
        activeReminders.removeAll { $0.medicationId == medicationId && Calendar.current.isDate($0.scheduledTime, inSameDayAs: timestamp) }
    }
    
    func recordMedicationMissed(_ medicationId: String, scheduledTime: Date) async {
        let record = AdherenceRecord(
            medicationId: medicationId,
            scheduledTime: scheduledTime,
            actualTime: nil,
            wasTaken: false,
            delayMinutes: nil
        )
        
        adherenceHistory.append(record)
        await updateAdherenceScore()
        await handleMissedDose(medicationId: medicationId, scheduledTime: scheduledTime)
    }
    
    func snoozeReminder(_ reminderId: String, minutes: Int) async {
        guard let reminderIndex = activeReminders.firstIndex(where: { $0.id == reminderId }) else { return }
        
        let reminder = activeReminders[reminderIndex]
        let newTime = reminder.scheduledTime.addingTimeInterval(TimeInterval(minutes * 60))
        
        activeReminders[reminderIndex].scheduledTime = newTime
        activeReminders[reminderIndex].snoozeCount += 1
        
        await scheduleNotification(for: activeReminders[reminderIndex])
    }
    
    func optimizeSchedule() async {
        isLearning = true
        
        // Analyze user behavior patterns
        await analyzeBehaviorPatterns()
        
        // Generate adaptive schedule
        await generateAdaptiveSchedule()
        
        // Apply machine learning optimizations
        await applyMLOptimizations()
        
        lastOptimization = Date()
        isLearning = false
    }
    
    // MARK: - Private Methods
    
    private func setupNotificationDelegate() {
        notificationCenter.delegate = self
        requestNotificationPermission()
    }
    
    private func requestNotificationPermission() {
        notificationCenter.requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Notification permission error: \(error)")
            }
        }
    }
    
    private func createRemindersForMedication(_ medication: Medication) async {
        for schedule in medication.schedules {
            let reminder = MedicationReminder(
                id: UUID().uuidString,
                medicationId: medication.id,
                medicationName: medication.name,
                dosage: medication.dosage,
                scheduledTime: schedule.time,
                frequency: schedule.frequency,
                isAdaptive: false
            )
            
            activeReminders.append(reminder)
            await scheduleNotification(for: reminder)
        }
    }
    
    private func scheduleNotification(for reminder: MedicationReminder) async {
        let content = UNMutableNotificationContent()
        content.title = "Medication Reminder"
        content.body = "Time to take \(reminder.medicationName) (\(reminder.dosage))"
        content.sound = .default
        content.categoryIdentifier = "MEDICATION_REMINDER"
        
        // Add user info for handling
        content.userInfo = [
            "reminderId": reminder.id,
            "medicationId": reminder.medicationId,
            "medicationName": reminder.medicationName
        ]
        
        // Create trigger
        let calendar = Calendar.current
        let components = calendar.dateComponents([.hour, .minute], from: reminder.scheduledTime)
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
        
        // Create request
        let request = UNNotificationRequest(
            identifier: reminder.id,
            content: content,
            trigger: trigger
        )
        
        do {
            try await notificationCenter.add(request)
        } catch {
            print("Failed to schedule notification: \(error)")
        }
    }
    
    private func cancelNotifications(for medicationId: String) async {
        let identifiers = activeReminders
            .filter { $0.medicationId == medicationId }
            .map { $0.id }
        
        notificationCenter.removePendingNotificationRequests(withIdentifiers: identifiers)
    }
    
    private func updateAdherenceScore() async {
        guard !adherenceHistory.isEmpty else {
            adherenceScore = 0.0
            return
        }
        
        let recentHistory = adherenceHistory.suffix(30) // Last 30 records
        let takenCount = recentHistory.filter { $0.wasTaken }.count
        adherenceScore = Double(takenCount) / Double(recentHistory.count)
    }
    
    private func handleMissedDose(medicationId: String, scheduledTime: Date) async {
        guard let medication = medications.first(where: { $0.id == medicationId }) else { return }
        
        // Count recent missed doses
        let recentMissed = adherenceHistory
            .filter { $0.medicationId == medicationId && !$0.wasTaken }
            .suffix(maxMissedDoses)
        
        if recentMissed.count >= maxMissedDoses {
            await sendCriticalAdherenceAlert(medication: medication)
        } else {
            await suggestMakeupDose(medication: medication, missedTime: scheduledTime)
        }
    }
    
    private func sendCriticalAdherenceAlert(medication: Medication) async {
        let content = UNMutableNotificationContent()
        content.title = "Critical: Multiple Missed Doses"
        content.body = "You've missed several doses of \(medication.name). Please consult your healthcare provider."
        content.sound = .critical
        content.categoryIdentifier = "CRITICAL_ADHERENCE"
        
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: "critical_\(medication.id)_\(Date().timeIntervalSince1970)",
            content: content,
            trigger: trigger
        )
        
        try? await notificationCenter.add(request)
    }
    
    private func suggestMakeupDose(medication: Medication, missedTime: Date) async {
        let content = UNMutableNotificationContent()
        content.title = "Missed Dose Suggestion"
        content.body = "You missed your \(medication.name) dose. Take it now if it's within the safe window."
        content.sound = .default
        content.categoryIdentifier = "MAKEUP_DOSE"
        
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 30 * 60, repeats: false) // 30 minutes later
        let request = UNNotificationRequest(
            identifier: "makeup_\(medication.id)_\(Date().timeIntervalSince1970)",
            content: content,
            trigger: trigger
        )
        
        try? await notificationCenter.add(request)
    }
    
    // MARK: - Behavior Learning
    
    private func startBehaviorLearning() {
        Timer.publish(every: 86400, on: .main, in: .common) // Daily
            .autoconnect()
            .sink { [weak self] _ in
                Task {
                    await self?.analyzeBehaviorPatterns()
                }
            }
            .store(in: &cancellables)
    }
    
    private func analyzeBehaviorPatterns() async {
        guard adherenceHistory.count >= learningPeriodDays else { return }
        
        let recentHistory = adherenceHistory.suffix(learningPeriodDays)
        
        // Analyze time patterns
        let timePatterns = analyzeTimePatterns(from: recentHistory)
        
        // Analyze day-of-week patterns
        let dayPatterns = analyzeDayPatterns(from: recentHistory)
        
        // Analyze delay patterns
        let delayPatterns = analyzeDelayPatterns(from: recentHistory)
        
        userBehaviorPatterns = UserBehaviorPattern(
            preferredTimes: timePatterns,
            problematicDays: dayPatterns.problematicDays,
            averageDelay: delayPatterns.averageDelay,
            consistencyScore: calculateConsistencyScore(from: recentHistory)
        )
    }
    
    private func analyzeTimePatterns(from history: ArraySlice<AdherenceRecord>) -> [TimeOfDay] {
        var timeFrequency: [Int: Int] = [:] // Hour -> Count
        
        for record in history where record.wasTaken {
            let hour = Calendar.current.component(.hour, from: record.actualTime ?? record.scheduledTime)
            timeFrequency[hour, default: 0] += 1
        }
        
        return timeFrequency
            .sorted { $0.value > $1.value }
            .prefix(3)
            .map { TimeOfDay(hour: $0.key, frequency: $0.value) }
    }
    
    private func analyzeDayPatterns(from history: ArraySlice<AdherenceRecord>) -> DayPatternAnalysis {
        var dayAdherence: [Int: (taken: Int, total: Int)] = [:]
        
        for record in history {
            let weekday = Calendar.current.component(.weekday, from: record.scheduledTime)
            let current = dayAdherence[weekday, default: (0, 0)]
            dayAdherence[weekday] = (current.taken + (record.wasTaken ? 1 : 0), current.total + 1)
        }
        
        let problematicDays = dayAdherence
            .filter { $0.value.total > 0 && Double($0.value.taken) / Double($0.value.total) < adaptationThreshold }
            .map { $0.key }
        
        return DayPatternAnalysis(problematicDays: problematicDays)
    }
    
    private func analyzeDelayPatterns(from history: ArraySlice<AdherenceRecord>) -> DelayPatternAnalysis {
        let delays = history
            .compactMap { $0.delayMinutes }
            .filter { $0 > 0 }
        
        let averageDelay = delays.isEmpty ? 0 : delays.reduce(0, +) / delays.count
        
        return DelayPatternAnalysis(averageDelay: averageDelay)
    }
    
    private func calculateConsistencyScore(from history: ArraySlice<AdherenceRecord>) -> Double {
        let adherenceRates = history.map { $0.wasTaken ? 1.0 : 0.0 }
        let mean = adherenceRates.reduce(0, +) / Double(adherenceRates.count)
        let variance = adherenceRates.map { pow($0 - mean, 2) }.reduce(0, +) / Double(adherenceRates.count)
        
        return 1.0 - sqrt(variance) // Higher score = more consistent
    }
    
    // MARK: - Adaptive Scheduling
    
    private func generateAdaptiveSchedule() async {
        guard let patterns = userBehaviorPatterns else { return }
        
        var newSchedule: [AdaptiveScheduleEntry] = []
        
        for medication in medications {
            for schedule in medication.schedules {
                let adaptedEntry = adaptScheduleEntry(
                    original: schedule,
                    patterns: patterns,
                    medication: medication
                )
                newSchedule.append(adaptedEntry)
            }
        }
        
        adaptiveSchedule = newSchedule
    }
    
    private func adaptScheduleEntry(
        original: MedicationSchedule,
        patterns: UserBehaviorPattern,
        medication: Medication
    ) -> AdaptiveScheduleEntry {
        
        let originalHour = Calendar.current.component(.hour, from: original.time)
        
        // Find the closest preferred time
        let preferredTime = patterns.preferredTimes
            .min { abs($0.hour - originalHour) < abs($1.hour - originalHour) }
        
        let adaptedHour = preferredTime?.hour ?? originalHour
        let adaptedTime = Calendar.current.date(bySettingHour: adaptedHour, minute: 0, second: 0, of: original.time) ?? original.time
        
        return AdaptiveScheduleEntry(
            medicationId: medication.id,
            originalTime: original.time,
            adaptedTime: adaptedTime,
            confidence: calculateAdaptationConfidence(patterns: patterns),
            reason: generateAdaptationReason(originalHour: originalHour, adaptedHour: adaptedHour)
        )
    }
    
    private func calculateAdaptationConfidence(patterns: UserBehaviorPattern) -> Double {
        return patterns.consistencyScore * 0.7 + (patterns.preferredTimes.isEmpty ? 0.0 : 0.3)
    }
    
    private func generateAdaptationReason(originalHour: Int, adaptedHour: Int) -> String {
        if originalHour == adaptedHour {
            return "No adaptation needed - current time aligns with your patterns"
        } else {
            return "Adapted to \(adaptedHour):00 based on your medication-taking patterns"
        }
    }
    
    // MARK: - Machine Learning
    
    private func applyMLOptimizations() async {
        // In a real implementation, this would use a trained ML model
        // For now, we'll use rule-based optimization
        
        await optimizeBasedOnAdherence()
        await optimizeBasedOnLifestyle()
    }
    
    private func optimizeBasedOnAdherence() async {
        for i in 0..<activeReminders.count {
            let reminder = activeReminders[i]
            let medicationHistory = adherenceHistory.filter { $0.medicationId == reminder.medicationId }
            
            if !medicationHistory.isEmpty {
                let adherenceRate = Double(medicationHistory.filter { $0.wasTaken }.count) / Double(medicationHistory.count)
                
                if adherenceRate < adaptationThreshold {
                    // Suggest time adjustment
                    activeReminders[i].isAdaptive = true
                    activeReminders[i].adaptationReason = "Low adherence detected - time optimized"
                }
            }
        }
    }
    
    private func optimizeBasedOnLifestyle() async {
        // Analyze user's active hours and suggest optimal timing
        let calendar = Calendar.current
        
        for i in 0..<activeReminders.count {
            let reminder = activeReminders[i]
            let hour = calendar.component(.hour, from: reminder.scheduledTime)
            
            // Avoid very early or very late hours unless specifically set
            if hour < 7 || hour > 22 {
                let suggestedHour = hour < 7 ? 8 : 20
                let newTime = calendar.date(bySettingHour: suggestedHour, minute: 0, second: 0, of: reminder.scheduledTime) ?? reminder.scheduledTime
                
                activeReminders[i].scheduledTime = newTime
                activeReminders[i].isAdaptive = true
                activeReminders[i].adaptationReason = "Optimized for better lifestyle fit"
            }
        }
    }
    
    private func learnFromAdherence(_ record: AdherenceRecord) async {
        // Update behavior patterns in real-time
        if adherenceHistory.count % 7 == 0 { // Weekly learning
            await analyzeBehaviorPatterns()
        }
    }
    
    // MARK: - Data Management
    
    private func loadMedicationData() {
        // In a real implementation, load from Core Data
        // For now, create sample data
        medications = [
            Medication(
                id: "med1",
                name: "Methotrexate",
                dosage: "15mg",
                schedules: [
                    MedicationSchedule(
                        time: Calendar.current.date(bySettingHour: 9, minute: 0, second: 0, of: Date()) ?? Date(),
                        frequency: .weekly
                    )
                ]
            ),
            Medication(
                id: "med2",
                name: "Folic Acid",
                dosage: "5mg",
                schedules: [
                    MedicationSchedule(
                        time: Calendar.current.date(bySettingHour: 10, minute: 0, second: 0, of: Date()) ?? Date(),
                        frequency: .weekly
                    )
                ]
            )
        ]
    }
    
    private func loadAdherenceHistory() {
        // In a real implementation, load from Core Data
        adherenceHistory = []
    }
}

// MARK: - UNUserNotificationCenterDelegate

extension IntelligentMedicationReminder: UNUserNotificationCenterDelegate {
    
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        
        let userInfo = response.notification.request.content.userInfo
        
        guard let medicationId = userInfo["medicationId"] as? String else {
            completionHandler()
            return
        }
        
        Task {
            switch response.actionIdentifier {
            case "TAKE_ACTION":
                await recordMedicationTaken(medicationId)
            case "SNOOZE_ACTION":
                if let reminderId = userInfo["reminderId"] as? String {
                    await snoozeReminder(reminderId, minutes: 15)
                }
            case "SKIP_ACTION":
                await recordMedicationMissed(medicationId, scheduledTime: Date())
            default:
                break
            }
            
            completionHandler()
        }
    }
    
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.alert, .sound, .badge])
    }
}

// MARK: - Supporting Types

struct Medication {
    let id: String
    let name: String
    let dosage: String
    let schedules: [MedicationSchedule]
}

struct MedicationSchedule {
    let time: Date
    let frequency: MedicationFrequency
}

enum MedicationFrequency: String, CaseIterable {
    case daily = "Daily"
    case weekly = "Weekly"
    case monthly = "Monthly"
    case asNeeded = "As Needed"
}

struct MedicationReminder {
    let id: String
    let medicationId: String
    let medicationName: String
    let dosage: String
    var scheduledTime: Date
    let frequency: MedicationFrequency
    var isAdaptive: Bool
    var adaptationReason: String?
    var snoozeCount: Int = 0
}

struct AdherenceRecord {
    let medicationId: String
    let scheduledTime: Date
    let actualTime: Date?
    let wasTaken: Bool
    let delayMinutes: Int?
}

struct UserBehaviorPattern {
    let preferredTimes: [TimeOfDay]
    let problematicDays: [Int]
    let averageDelay: Int
    let consistencyScore: Double
}

struct TimeOfDay {
    let hour: Int
    let frequency: Int
}

struct DayPatternAnalysis {
    let problematicDays: [Int]
}

struct DelayPatternAnalysis {
    let averageDelay: Int
}

struct AdaptiveScheduleEntry {
    let medicationId: String
    let originalTime: Date
    let adaptedTime: Date
    let confidence: Double
    let reason: String
}