//
//  EnhancedMedicationTracker.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Combine
import Foundation
import UserNotifications
import HealthKit
import CoreML

// MARK: - Enhanced Medication Tracker
class EnhancedMedicationTracker: ObservableObject {
    @Published var medications: [Medication] = []
    @Published var medicationSchedule: [MedicationSchedule] = []
    @Published var adherenceHistory: [AdherenceRecord] = []
    @Published var interactions: [DrugInteraction] = []
    @Published var sideEffects: [SideEffectReport] = []
    @Published var refillReminders: [RefillReminder] = []
    @Published var adherenceScore: Double = 0.0
    @Published var weeklyAdherence: [WeeklyAdherence] = []
    @Published var medicationInsights: [MedicationInsight] = []
    
    private let notificationManager = MedicationNotificationManager()
    private let interactionChecker = DrugInteractionChecker()
    private let adherenceAnalyzer = AdherenceAnalyzer()
    private let sideEffectMonitor = SideEffectMonitor()
    private let pharmacyIntegration = PharmacyIntegrationManager()
    private let mlPredictor = MedicationMLPredictor()
    private let healthKitManager = HealthKitMedicationManager()
    
    private var cancellables = Set<AnyCancellable>()
    private var adherenceTimer: Timer?
    private var reminderTimer: Timer?
    
    init() {
        setupMedicationTracking()
        loadStoredData()
        setupPeriodicChecks()
        setupNotifications()
        requestNotificationPermissions()
    }
    
    // MARK: - Setup
    private func setupMedicationTracking() {
        notificationManager.delegate = self
        interactionChecker.delegate = self
        sideEffectMonitor.delegate = self
        
        // Setup HealthKit integration
        healthKitManager.requestAuthorization { [weak self] success in
            if success {
                self?.syncWithHealthKit()
            }
        }
    }
    
    private func loadStoredData() {
        loadMedications()
        loadMedicationSchedule()
        loadAdherenceHistory()
        loadSideEffects()
        calculateAdherenceScore()
    }
    
    private func setupPeriodicChecks() {
        // Check adherence every hour
        adherenceTimer = Timer.scheduledTimer(withTimeInterval: 3600, repeats: true) { [weak self] _ in
            self?.checkMissedDoses()
            self?.updateAdherenceScore()
        }
        
        // Check for refill reminders daily
        reminderTimer = Timer.scheduledTimer(withTimeInterval: 86400, repeats: true) { [weak self] _ in
            self?.checkRefillReminders()
            self?.generateInsights()
        }
    }
    
    private func setupNotifications() {
        NotificationCenter.default.publisher(for: .medicationTaken)
            .sink { [weak self] notification in
                self?.handleMedicationTaken(notification)
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: .sideEffectReported)
            .sink { [weak self] notification in
                self?.handleSideEffectReport(notification)
            }
            .store(in: &cancellables)
    }
    
    private func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                print("Notification permissions granted")
            } else if let error = error {
                print("Notification permission error: \(error.localizedDescription)")
            }
        }
    }
    
    // MARK: - Medication Management
    func addMedication(_ medication: Medication) {
        medications.append(medication)
        saveMedications()
        
        // Check for interactions with existing medications
        checkDrugInteractions(for: medication)
        
        // Create medication schedule
        createMedicationSchedule(for: medication)
        
        // Setup reminders
        setupReminders(for: medication)
        
        // Sync with HealthKit
        healthKitManager.addMedication(medication)
    }
    
    func updateMedication(_ medication: Medication) {
        if let index = medications.firstIndex(where: { $0.id == medication.id }) {
            medications[index] = medication
            saveMedications()
            
            // Update schedule and reminders
            updateMedicationSchedule(for: medication)
            updateReminders(for: medication)
            
            // Re-check interactions
            checkDrugInteractions(for: medication)
            
            // Sync with HealthKit
            healthKitManager.updateMedication(medication)
        }
    }
    
    func removeMedication(_ medicationId: UUID) {
        medications.removeAll { $0.id == medicationId }
        medicationSchedule.removeAll { $0.medicationId == medicationId }
        saveMedications()
        saveMedicationSchedule()
        
        // Remove notifications
        notificationManager.removeNotifications(for: medicationId)
        
        // Remove from HealthKit
        healthKitManager.removeMedication(medicationId)
    }
    
    // MARK: - Medication Schedule
    private func createMedicationSchedule(for medication: Medication) {
        let schedule = MedicationSchedule(
            id: UUID(),
            medicationId: medication.id,
            doseTimes: medication.doseTimes,
            frequency: medication.frequency,
            duration: medication.duration,
            startDate: medication.startDate,
            endDate: medication.endDate,
            isActive: true
        )
        
        medicationSchedule.append(schedule)
        saveMedicationSchedule()
    }
    
    private func updateMedicationSchedule(for medication: Medication) {
        if let index = medicationSchedule.firstIndex(where: { $0.medicationId == medication.id }) {
            medicationSchedule[index].doseTimes = medication.doseTimes
            medicationSchedule[index].frequency = medication.frequency
            medicationSchedule[index].duration = medication.duration
            medicationSchedule[index].endDate = medication.endDate
            saveMedicationSchedule()
        }
    }
    
    // MARK: - Reminders and Notifications
    private func setupReminders(for medication: Medication) {
        for doseTime in medication.doseTimes {
            let reminder = MedicationReminder(
                medicationId: medication.id,
                medicationName: medication.name,
                doseTime: doseTime,
                dosage: medication.dosage,
                instructions: medication.instructions
            )
            
            notificationManager.scheduleReminder(reminder)
        }
    }
    
    private func updateReminders(for medication: Medication) {
        // Remove existing reminders
        notificationManager.removeNotifications(for: medication.id)
        
        // Setup new reminders
        setupReminders(for: medication)
    }
    
    // MARK: - Adherence Tracking
    func recordMedicationTaken(_ medicationId: UUID, timestamp: Date = Date(), notes: String? = nil) {
        let record = AdherenceRecord(
            id: UUID(),
            medicationId: medicationId,
            scheduledTime: getScheduledTime(for: medicationId, near: timestamp),
            actualTime: timestamp,
            status: .taken,
            notes: notes
        )
        
        adherenceHistory.append(record)
        saveAdherenceHistory()
        
        // Update adherence score
        updateAdherenceScore()
        
        // Check for side effects
        sideEffectMonitor.checkForSideEffects(medicationId: medicationId, timestamp: timestamp)
        
        // Sync with HealthKit
        healthKitManager.recordMedicationTaken(medicationId, timestamp: timestamp)
        
        // Generate insights
        generateAdherenceInsights()
    }
    
    func recordMedicationMissed(_ medicationId: UUID, scheduledTime: Date, reason: MissedReason? = nil) {
        let record = AdherenceRecord(
            id: UUID(),
            medicationId: medicationId,
            scheduledTime: scheduledTime,
            actualTime: nil,
            status: .missed,
            missedReason: reason
        )
        
        adherenceHistory.append(record)
        saveAdherenceHistory()
        
        // Update adherence score
        updateAdherenceScore()
        
        // Send missed dose notification
        notificationManager.sendMissedDoseNotification(medicationId, scheduledTime: scheduledTime)
    }
    
    func recordMedicationDelayed(_ medicationId: UUID, scheduledTime: Date, actualTime: Date, reason: String? = nil) {
        let record = AdherenceRecord(
            id: UUID(),
            medicationId: medicationId,
            scheduledTime: scheduledTime,
            actualTime: actualTime,
            status: .delayed,
            notes: reason
        )
        
        adherenceHistory.append(record)
        saveAdherenceHistory()
        
        // Update adherence score
        updateAdherenceScore()
    }
    
    private func checkMissedDoses() {
        let now = Date()
        let calendar = Calendar.current
        
        for schedule in medicationSchedule.filter({ $0.isActive }) {
            for doseTime in schedule.doseTimes {
                let scheduledDateTime = calendar.date(bySettingHour: calendar.component(.hour, from: doseTime),
                                                    minute: calendar.component(.minute, from: doseTime),
                                                    second: 0,
                                                    of: now) ?? now
                
                // Check if dose was missed (more than 30 minutes late)
                if now.timeIntervalSince(scheduledDateTime) > 1800 {
                    let wasRecorded = adherenceHistory.contains { record in
                        record.medicationId == schedule.medicationId &&
                        calendar.isDate(record.scheduledTime, inSameDayAs: scheduledDateTime) &&
                        record.status != .missed
                    }
                    
                    if !wasRecorded {
                        recordMedicationMissed(schedule.medicationId, scheduledTime: scheduledDateTime)
                    }
                }
            }
        }
    }
    
    private func updateAdherenceScore() {
        adherenceScore = adherenceAnalyzer.calculateAdherenceScore(adherenceHistory)
        calculateWeeklyAdherence()
    }
    
    private func calculateAdherenceScore() {
        adherenceScore = adherenceAnalyzer.calculateAdherenceScore(adherenceHistory)
    }
    
    private func calculateWeeklyAdherence() {
        weeklyAdherence = adherenceAnalyzer.calculateWeeklyAdherence(adherenceHistory)
    }
    
    // MARK: - Drug Interaction Checking
    private func checkDrugInteractions(for medication: Medication) {
        let otherMedications = medications.filter { $0.id != medication.id }
        
        for otherMedication in otherMedications {
            interactionChecker.checkInteraction(between: medication, and: otherMedication) { [weak self] interaction in
                if let interaction = interaction {
                    self?.handleDrugInteraction(interaction)
                }
            }
        }
    }
    
    private func handleDrugInteraction(_ interaction: DrugInteraction) {
        interactions.append(interaction)
        
        // Send notification for severe interactions
        if interaction.severity == .severe || interaction.severity == .contraindicated {
            notificationManager.sendInteractionAlert(interaction)
        }
        
        // Generate insight
        let insight = MedicationInsight(
            id: UUID(),
            type: .drugInteraction,
            title: "Drug Interaction Detected",
            description: interaction.description,
            severity: .high,
            actionRequired: true,
            timestamp: Date()
        )
        
        medicationInsights.append(insight)
    }
    
    // MARK: - Side Effect Monitoring
    func reportSideEffect(_ sideEffect: SideEffectReport) {
        sideEffects.append(sideEffect)
        saveSideEffects()
        
        // Analyze side effect patterns
        sideEffectMonitor.analyzeSideEffectPattern(sideEffect, with: adherenceHistory)
        
        // Generate insight
        let insight = MedicationInsight(
            id: UUID(),
            type: .sideEffect,
            title: "Side Effect Reported",
            description: "\(sideEffect.symptom) reported for \(getMedicationName(sideEffect.medicationId))",
            severity: sideEffect.severity == .severe ? .high : .medium,
            actionRequired: sideEffect.severity == .severe,
            timestamp: Date()
        )
        
        medicationInsights.append(insight)
        
        // Notify healthcare provider if severe
        if sideEffect.severity == .severe {
            notifyHealthcareProvider(of: sideEffect)
        }
    }
    
    private func notifyHealthcareProvider(of sideEffect: SideEffectReport) {
        // Integration with healthcare provider portal
        // Implementation depends on your healthcare provider integration
    }
    
    // MARK: - Refill Management
    private func checkRefillReminders() {
        let calendar = Calendar.current
        let today = Date()
        
        for medication in medications {
            if let refillDate = medication.refillDate {
                let daysUntilRefill = calendar.dateComponents([.day], from: today, to: refillDate).day ?? 0
                
                // Remind 7 days before refill needed
                if daysUntilRefill <= 7 && daysUntilRefill > 0 {
                    let reminder = RefillReminder(
                        id: UUID(),
                        medicationId: medication.id,
                        medicationName: medication.name,
                        refillDate: refillDate,
                        daysRemaining: daysUntilRefill,
                        pharmacy: medication.pharmacy
                    )
                    
                    if !refillReminders.contains(where: { $0.medicationId == medication.id }) {
                        refillReminders.append(reminder)
                        notificationManager.sendRefillReminder(reminder)
                    }
                }
            }
        }
    }
    
    func markRefillCompleted(_ medicationId: UUID, newRefillDate: Date) {
        if let index = medications.firstIndex(where: { $0.id == medicationId }) {
            medications[index].refillDate = newRefillDate
            saveMedications()
        }
        
        refillReminders.removeAll { $0.medicationId == medicationId }
    }
    
    // MARK: - Insights and Analytics
    private func generateInsights() {
        generateAdherenceInsights()
        generateEffectivenessInsights()
        generateSideEffectInsights()
        generateOptimizationInsights()
    }
    
    private func generateAdherenceInsights() {
        let insights = adherenceAnalyzer.generateAdherenceInsights(adherenceHistory, medications: medications)
        medicationInsights.append(contentsOf: insights)
    }
    
    private func generateEffectivenessInsights() {
        // Use ML to predict medication effectiveness
        mlPredictor.predictEffectiveness(medications: medications, adherenceHistory: adherenceHistory) { [weak self] predictions in
            for prediction in predictions {
                let insight = MedicationInsight(
                    id: UUID(),
                    type: .effectiveness,
                    title: "Medication Effectiveness Analysis",
                    description: prediction.description,
                    severity: .medium,
                    actionRequired: false,
                    timestamp: Date()
                )
                
                self?.medicationInsights.append(insight)
            }
        }
    }
    
    private func generateSideEffectInsights() {
        let patterns = sideEffectMonitor.analyzeSideEffectPatterns(sideEffects, adherenceHistory: adherenceHistory)
        
        for pattern in patterns {
            let insight = MedicationInsight(
                id: UUID(),
                type: .sideEffect,
                title: "Side Effect Pattern Detected",
                description: pattern.description,
                severity: pattern.severity == .high ? .high : .medium,
                actionRequired: pattern.severity == .high,
                timestamp: Date()
            )
            
            medicationInsights.append(insight)
        }
    }
    
    private func generateOptimizationInsights() {
        // Analyze medication timing and suggest optimizations
        let optimizations = adherenceAnalyzer.suggestOptimizations(medications: medications, adherenceHistory: adherenceHistory)
        
        for optimization in optimizations {
            let insight = MedicationInsight(
                id: UUID(),
                type: .optimization,
                title: "Medication Schedule Optimization",
                description: optimization.description,
                severity: .low,
                actionRequired: false,
                timestamp: Date()
            )
            
            medicationInsights.append(insight)
        }
    }
    
    // MARK: - Pharmacy Integration
    func connectToPharmacy(_ pharmacy: Pharmacy) {
        pharmacyIntegration.connect(to: pharmacy) { [weak self] success in
            if success {
                self?.syncWithPharmacy(pharmacy)
            }
        }
    }
    
    private func syncWithPharmacy(_ pharmacy: Pharmacy) {
        pharmacyIntegration.syncMedications(pharmacy) { [weak self] pharmacyMedications in
            // Update medication information with pharmacy data
            for pharmacyMed in pharmacyMedications {
                if let index = self?.medications.firstIndex(where: { $0.name == pharmacyMed.name }) {
                    self?.medications[index].refillDate = pharmacyMed.refillDate
                    self?.medications[index].pharmacy = pharmacy
                }
            }
            
            self?.saveMedications()
        }
    }
    
    // MARK: - HealthKit Integration
    private func syncWithHealthKit() {
        healthKitManager.syncMedications { [weak self] healthKitMedications in
            // Merge HealthKit medications with local medications
            for hkMed in healthKitMedications {
                if !self?.medications.contains(where: { $0.name == hkMed.name }) ?? true {
                    self?.addMedication(hkMed)
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    private func getScheduledTime(for medicationId: UUID, near timestamp: Date) -> Date {
        guard let schedule = medicationSchedule.first(where: { $0.medicationId == medicationId }) else {
            return timestamp
        }
        
        let calendar = Calendar.current
        let targetTime = schedule.doseTimes.min { time1, time2 in
            abs(timestamp.timeIntervalSince(time1)) < abs(timestamp.timeIntervalSince(time2))
        } ?? timestamp
        
        return calendar.date(bySettingHour: calendar.component(.hour, from: targetTime),
                           minute: calendar.component(.minute, from: targetTime),
                           second: 0,
                           of: timestamp) ?? timestamp
    }
    
    private func getMedicationName(_ medicationId: UUID) -> String {
        return medications.first(where: { $0.id == medicationId })?.name ?? "Unknown Medication"
    }
    
    // MARK: - Notification Handlers
    private func handleMedicationTaken(_ notification: Notification) {
        if let userInfo = notification.userInfo,
           let medicationId = userInfo["medicationId"] as? UUID {
            recordMedicationTaken(medicationId)
        }
    }
    
    private func handleSideEffectReport(_ notification: Notification) {
        if let sideEffect = notification.object as? SideEffectReport {
            reportSideEffect(sideEffect)
        }
    }
    
    // MARK: - Data Persistence
    private func loadMedications() {
        if let data = UserDefaults.standard.data(forKey: "Medications"),
           let loadedMedications = try? JSONDecoder().decode([Medication].self, from: data) {
            medications = loadedMedications
        }
    }
    
    private func saveMedications() {
        if let data = try? JSONEncoder().encode(medications) {
            UserDefaults.standard.set(data, forKey: "Medications")
        }
    }
    
    private func loadMedicationSchedule() {
        if let data = UserDefaults.standard.data(forKey: "MedicationSchedule"),
           let schedule = try? JSONDecoder().decode([MedicationSchedule].self, from: data) {
            medicationSchedule = schedule
        }
    }
    
    private func saveMedicationSchedule() {
        if let data = try? JSONEncoder().encode(medicationSchedule) {
            UserDefaults.standard.set(data, forKey: "MedicationSchedule")
        }
    }
    
    private func loadAdherenceHistory() {
        if let data = UserDefaults.standard.data(forKey: "AdherenceHistory"),
           let history = try? JSONDecoder().decode([AdherenceRecord].self, from: data) {
            adherenceHistory = history
        }
    }
    
    private func saveAdherenceHistory() {
        if let data = try? JSONEncoder().encode(adherenceHistory) {
            UserDefaults.standard.set(data, forKey: "AdherenceHistory")
        }
    }
    
    private func loadSideEffects() {
        if let data = UserDefaults.standard.data(forKey: "SideEffects"),
           let effects = try? JSONDecoder().decode([SideEffectReport].self, from: data) {
            sideEffects = effects
        }
    }
    
    private func saveSideEffects() {
        if let data = try? JSONEncoder().encode(sideEffects) {
            UserDefaults.standard.set(data, forKey: "SideEffects")
        }
    }
    
    // MARK: - Cleanup
    deinit {
        adherenceTimer?.invalidate()
        reminderTimer?.invalidate()
        cancellables.removeAll()
    }
}

// MARK: - Notification Manager Delegate
extension EnhancedMedicationTracker: MedicationNotificationManagerDelegate {
    func notificationManager(_ manager: MedicationNotificationManager, didReceiveResponse response: UNNotificationResponse) {
        // Handle notification responses
        let userInfo = response.notification.request.content.userInfo
        
        if let medicationId = userInfo["medicationId"] as? String,
           let uuid = UUID(uuidString: medicationId) {
            
            switch response.actionIdentifier {
            case "TAKE_ACTION":
                recordMedicationTaken(uuid)
            case "DELAY_ACTION":
                // Schedule delayed reminder
                notificationManager.scheduleDelayedReminder(uuid, delay: 900) // 15 minutes
            case "SKIP_ACTION":
                recordMedicationMissed(uuid, scheduledTime: Date(), reason: .intentionallySkipped)
            default:
                break
            }
        }
    }
}

// MARK: - Drug Interaction Checker Delegate
extension EnhancedMedicationTracker: DrugInteractionCheckerDelegate {
    func interactionChecker(_ checker: DrugInteractionChecker, didFindInteraction interaction: DrugInteraction) {
        handleDrugInteraction(interaction)
    }
}

// MARK: - Side Effect Monitor Delegate
extension EnhancedMedicationTracker: SideEffectMonitorDelegate {
    func sideEffectMonitor(_ monitor: SideEffectMonitor, didDetectPattern pattern: SideEffectPattern) {
        let insight = MedicationInsight(
            id: UUID(),
            type: .sideEffect,
            title: "Side Effect Pattern Detected",
            description: pattern.description,
            severity: pattern.severity == .high ? .high : .medium,
            actionRequired: pattern.severity == .high,
            timestamp: Date()
        )
        
        medicationInsights.append(insight)
    }
}

// MARK: - Supporting Classes
class MedicationNotificationManager {
    weak var delegate: MedicationNotificationManagerDelegate?
    
    func scheduleReminder(_ reminder: MedicationReminder) {
        let content = UNMutableNotificationContent()
        content.title = "Medication Reminder"
        content.body = "Time to take \(reminder.medicationName) - \(reminder.dosage)"
        content.sound = .default
        content.userInfo = ["medicationId": reminder.medicationId.uuidString]
        
        // Add action buttons
        let takeAction = UNNotificationAction(identifier: "TAKE_ACTION", title: "Take Now", options: [])
        let delayAction = UNNotificationAction(identifier: "DELAY_ACTION", title: "Remind in 15 min", options: [])
        let skipAction = UNNotificationAction(identifier: "SKIP_ACTION", title: "Skip", options: [])
        
        let category = UNNotificationCategory(identifier: "MEDICATION_REMINDER",
                                            actions: [takeAction, delayAction, skipAction],
                                            intentIdentifiers: [],
                                            options: [])
        
        UNUserNotificationCenter.current().setNotificationCategories([category])
        content.categoryIdentifier = "MEDICATION_REMINDER"
        
        // Schedule daily repeating notification
        let calendar = Calendar.current
        let components = calendar.dateComponents([.hour, .minute], from: reminder.doseTime)
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
        
        let request = UNNotificationRequest(identifier: "medication_\(reminder.medicationId.uuidString)_\(components.hour ?? 0)_\(components.minute ?? 0)",
                                          content: content,
                                          trigger: trigger)
        
        UNUserNotificationCenter.current().add(request)
    }
    
    func removeNotifications(for medicationId: UUID) {
        UNUserNotificationCenter.current().getPendingNotificationRequests { requests in
            let identifiersToRemove = requests.compactMap { request in
                request.identifier.contains(medicationId.uuidString) ? request.identifier : nil
            }
            
            UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: identifiersToRemove)
        }
    }
    
    func sendMissedDoseNotification(_ medicationId: UUID, scheduledTime: Date) {
        // Implementation for missed dose notification
    }
    
    func sendInteractionAlert(_ interaction: DrugInteraction) {
        // Implementation for drug interaction alert
    }
    
    func sendRefillReminder(_ reminder: RefillReminder) {
        // Implementation for refill reminder
    }
    
    func scheduleDelayedReminder(_ medicationId: UUID, delay: TimeInterval) {
        // Implementation for delayed reminder
    }
}

protocol MedicationNotificationManagerDelegate: AnyObject {
    func notificationManager(_ manager: MedicationNotificationManager, didReceiveResponse response: UNNotificationResponse)
}

class DrugInteractionChecker {
    weak var delegate: DrugInteractionCheckerDelegate?
    
    func checkInteraction(between medication1: Medication, and medication2: Medication, completion: @escaping (DrugInteraction?) -> Void) {
        // Check drug interactions using medical database
        // This would typically involve API calls to drug interaction databases
        completion(nil)
    }
}

protocol DrugInteractionCheckerDelegate: AnyObject {
    func interactionChecker(_ checker: DrugInteractionChecker, didFindInteraction interaction: DrugInteraction)
}

class AdherenceAnalyzer {
    func calculateAdherenceScore(_ history: [AdherenceRecord]) -> Double {
        guard !history.isEmpty else { return 0.0 }
        
        let takenCount = history.filter { $0.status == .taken }.count
        let totalCount = history.count
        
        return Double(takenCount) / Double(totalCount) * 100.0
    }
    
    func calculateWeeklyAdherence(_ history: [AdherenceRecord]) -> [WeeklyAdherence] {
        let calendar = Calendar.current
        let now = Date()
        var weeklyData: [WeeklyAdherence] = []
        
        for weekOffset in 0..<12 { // Last 12 weeks
            guard let weekStart = calendar.date(byAdding: .weekOfYear, value: -weekOffset, to: now),
                  let weekEnd = calendar.date(byAdding: .day, value: 6, to: weekStart) else { continue }
            
            let weekHistory = history.filter { record in
                record.scheduledTime >= weekStart && record.scheduledTime <= weekEnd
            }
            
            let adherenceScore = calculateAdherenceScore(weekHistory)
            
            let weeklyAdherence = WeeklyAdherence(
                weekStart: weekStart,
                weekEnd: weekEnd,
                adherenceScore: adherenceScore,
                totalDoses: weekHistory.count,
                takenDoses: weekHistory.filter { $0.status == .taken }.count
            )
            
            weeklyData.append(weeklyAdherence)
        }
        
        return weeklyData.reversed()
    }
    
    func generateAdherenceInsights(_ history: [AdherenceRecord], medications: [Medication]) -> [MedicationInsight] {
        var insights: [MedicationInsight] = []
        
        // Analyze adherence patterns
        let recentHistory = history.filter { $0.scheduledTime >= Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date() }
        let adherenceScore = calculateAdherenceScore(recentHistory)
        
        if adherenceScore < 80 {
            let insight = MedicationInsight(
                id: UUID(),
                type: .adherence,
                title: "Low Medication Adherence",
                description: "Your medication adherence is \(String(format: "%.1f", adherenceScore))%. Consider setting more reminders or adjusting your schedule.",
                severity: .high,
                actionRequired: true,
                timestamp: Date()
            )
            insights.append(insight)
        }
        
        return insights
    }
    
    func suggestOptimizations(medications: [Medication], adherenceHistory: [AdherenceRecord]) -> [OptimizationSuggestion] {
        var suggestions: [OptimizationSuggestion] = []
        
        // Analyze timing patterns and suggest optimizations
        for medication in medications {
            let medicationHistory = adherenceHistory.filter { $0.medicationId == medication.id }
            let missedDoses = medicationHistory.filter { $0.status == .missed }
            
            if missedDoses.count > medicationHistory.count / 4 { // More than 25% missed
                let suggestion = OptimizationSuggestion(
                    medicationId: medication.id,
                    type: .scheduleAdjustment,
                    description: "Consider adjusting the schedule for \(medication.name) to improve adherence",
                    priority: .high
                )
                suggestions.append(suggestion)
            }
        }
        
        return suggestions
    }
}

class SideEffectMonitor {
    weak var delegate: SideEffectMonitorDelegate?
    
    func checkForSideEffects(medicationId: UUID, timestamp: Date) {
        // Monitor for potential side effects after medication intake
        // This could involve checking health data or prompting user
    }
    
    func analyzeSideEffectPattern(_ sideEffect: SideEffectReport, with adherenceHistory: [AdherenceRecord]) {
        // Analyze correlation between medication timing and side effects
    }
    
    func analyzeSideEffectPatterns(_ sideEffects: [SideEffectReport], adherenceHistory: [AdherenceRecord]) -> [SideEffectPattern] {
        var patterns: [SideEffectPattern] = []
        
        // Group side effects by medication
        let groupedEffects = Dictionary(grouping: sideEffects) { $0.medicationId }
        
        for (medicationId, effects) in groupedEffects {
            if effects.count >= 3 { // Pattern threshold
                let pattern = SideEffectPattern(
                    medicationId: medicationId,
                    symptom: effects.first?.symptom ?? "Unknown",
                    frequency: effects.count,
                    severity: effects.max(by: { $0.severity.rawValue < $1.severity.rawValue })?.severity ?? .mild,
                    description: "Recurring side effect pattern detected"
                )
                patterns.append(pattern)
            }
        }
        
        return patterns
    }
}

protocol SideEffectMonitorDelegate: AnyObject {
    func sideEffectMonitor(_ monitor: SideEffectMonitor, didDetectPattern pattern: SideEffectPattern)
}

class PharmacyIntegrationManager {
    func connect(to pharmacy: Pharmacy, completion: @escaping (Bool) -> Void) {
        // Connect to pharmacy API
        completion(true)
    }
    
    func syncMedications(_ pharmacy: Pharmacy, completion: @escaping ([Medication]) -> Void) {
        // Sync medications with pharmacy
        completion([])
    }
}

class MedicationMLPredictor {
    func predictEffectiveness(medications: [Medication], adherenceHistory: [AdherenceRecord], completion: @escaping ([EffectivenessPrediction]) -> Void) {
        // Use ML to predict medication effectiveness
        completion([])
    }
}

class HealthKitMedicationManager {
    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        // Request HealthKit authorization for medications
        completion(true)
    }
    
    func addMedication(_ medication: Medication) {
        // Add medication to HealthKit
    }
    
    func updateMedication(_ medication: Medication) {
        // Update medication in HealthKit
    }
    
    func removeMedication(_ medicationId: UUID) {
        // Remove medication from HealthKit
    }
    
    func recordMedicationTaken(_ medicationId: UUID, timestamp: Date) {
        // Record medication intake in HealthKit
    }
    
    func syncMedications(completion: @escaping ([Medication]) -> Void) {
        // Sync medications from HealthKit
        completion([])
    }
}

// MARK: - Data Types
struct Medication: Codable, Identifiable {
    let id: UUID
    let name: String
    let dosage: String
    let frequency: MedicationFrequency
    let doseTimes: [Date]
    let instructions: String
    let startDate: Date
    let endDate: Date?
    let duration: TimeInterval?
    var refillDate: Date?
    var pharmacy: Pharmacy?
    let prescribedBy: String
    let indication: String
    let sideEffects: [String]
    let contraindications: [String]
}

struct MedicationSchedule: Codable {
    let id: UUID
    let medicationId: UUID
    var doseTimes: [Date]
    var frequency: MedicationFrequency
    var duration: TimeInterval?
    let startDate: Date
    var endDate: Date?
    var isActive: Bool
}

struct AdherenceRecord: Codable {
    let id: UUID
    let medicationId: UUID
    let scheduledTime: Date
    let actualTime: Date?
    let status: AdherenceStatus
    let notes: String?
    let missedReason: MissedReason?
    
    init(id: UUID, medicationId: UUID, scheduledTime: Date, actualTime: Date?, status: AdherenceStatus, notes: String? = nil, missedReason: MissedReason? = nil) {
        self.id = id
        self.medicationId = medicationId
        self.scheduledTime = scheduledTime
        self.actualTime = actualTime
        self.status = status
        self.notes = notes
        self.missedReason = missedReason
    }
}

struct DrugInteraction: Codable {
    let id: UUID
    let medication1Id: UUID
    let medication2Id: UUID
    let severity: InteractionSeverity
    let description: String
    let recommendation: String
    let source: String
}

struct SideEffectReport: Codable {
    let id: UUID
    let medicationId: UUID
    let symptom: String
    let severity: SideEffectSeverity
    let onset: Date
    let duration: TimeInterval?
    let description: String
    let reportedDate: Date
}

struct RefillReminder: Codable {
    let id: UUID
    let medicationId: UUID
    let medicationName: String
    let refillDate: Date
    let daysRemaining: Int
    let pharmacy: Pharmacy?
}

struct MedicationInsight: Codable {
    let id: UUID
    let type: InsightType
    let title: String
    let description: String
    let severity: InsightSeverity
    let actionRequired: Bool
    let timestamp: Date
}

struct WeeklyAdherence: Codable {
    let weekStart: Date
    let weekEnd: Date
    let adherenceScore: Double
    let totalDoses: Int
    let takenDoses: Int
}

struct MedicationReminder: Codable {
    let medicationId: UUID
    let medicationName: String
    let doseTime: Date
    let dosage: String
    let instructions: String
}

struct Pharmacy: Codable {
    let id: UUID
    let name: String
    let address: String
    let phone: String
    let email: String?
    let apiEndpoint: String?
}

struct SideEffectPattern: Codable {
    let medicationId: UUID
    let symptom: String
    let frequency: Int
    let severity: SideEffectSeverity
    let description: String
}

struct OptimizationSuggestion: Codable {
    let medicationId: UUID
    let type: OptimizationType
    let description: String
    let priority: OptimizationPriority
}

struct EffectivenessPrediction: Codable {
    let medicationId: UUID
    let effectivenessScore: Double
    let confidence: Double
    let description: String
}

// MARK: - Enums
enum MedicationFrequency: String, Codable, CaseIterable {
    case onceDaily = "once_daily"
    case twiceDaily = "twice_daily"
    case threeTimesDaily = "three_times_daily"
    case fourTimesDaily = "four_times_daily"
    case asNeeded = "as_needed"
    case weekly = "weekly"
    case monthly = "monthly"
}

enum AdherenceStatus: String, Codable {
    case taken = "taken"
    case missed = "missed"
    case delayed = "delayed"
    case skipped = "skipped"
}

enum MissedReason: String, Codable {
    case forgot = "forgot"
    case sideEffects = "side_effects"
    case intentionallySkipped = "intentionally_skipped"
    case unavailable = "unavailable"
    case other = "other"
}

enum InteractionSeverity: String, Codable {
    case minor = "minor"
    case moderate = "moderate"
    case major = "major"
    case severe = "severe"
    case contraindicated = "contraindicated"
}

enum SideEffectSeverity: String, Codable {
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    
    var rawValue: Int {
        switch self {
        case .mild: return 1
        case .moderate: return 2
        case .severe: return 3
        }
    }
}

enum InsightType: String, Codable {
    case adherence = "adherence"
    case drugInteraction = "drug_interaction"
    case sideEffect = "side_effect"
    case effectiveness = "effectiveness"
    case optimization = "optimization"
}

enum InsightSeverity: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
}

enum OptimizationType: String, Codable {
    case scheduleAdjustment = "schedule_adjustment"
    case dosageOptimization = "dosage_optimization"
    case timingImprovement = "timing_improvement"
}

enum OptimizationPriority: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let medicationTaken = Notification.Name("medicationTaken")
    static let sideEffectReported = Notification.Name("sideEffectReported")
}