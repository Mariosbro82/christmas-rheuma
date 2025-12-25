//
//  AppleWatchModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import WatchConnectivity
import HealthKit
import CoreLocation
import UserNotifications

// MARK: - Watch Data Models

struct WatchHealthData: Codable {
    let heartRate: Double?
    let steps: Int?
    let activeCalories: Double?
    let workoutMinutes: Int?
    let standHours: Int?
    let timestamp: Date
    let batteryLevel: Double?
    let isCharging: Bool
}

struct WatchSymptomEntry: Codable {
    let id: UUID
    let symptomType: String
    let severity: Int // 1-10
    let location: String?
    let timestamp: Date
    let notes: String?
    let triggers: [String]
}

struct WatchMedicationReminder: Codable {
    let id: UUID
    let medicationName: String
    let dosage: String
    let scheduledTime: Date
    let isTaken: Bool
    let reminderType: ReminderType
    
    enum ReminderType: String, Codable, CaseIterable {
        case daily = "daily"
        case weekly = "weekly"
        case asNeeded = "as_needed"
        case beforeMeals = "before_meals"
        case afterMeals = "after_meals"
    }
}

struct WatchWorkoutData: Codable {
    let id: UUID
    let workoutType: HKWorkoutActivityType
    let startDate: Date
    let endDate: Date?
    let duration: TimeInterval
    let activeCalories: Double
    let totalCalories: Double
    let distance: Double?
    let averageHeartRate: Double?
    let maxHeartRate: Double?
    let isActive: Bool
}

struct WatchEnvironmentalData: Codable {
    let temperature: Double?
    let humidity: Double?
    let airQualityIndex: Int?
    let uvIndex: Int?
    let barometricPressure: Double?
    let timestamp: Date
    let location: CLLocation?
}

struct WatchNotification: Codable {
    let id: UUID
    let title: String
    let body: String
    let category: NotificationCategory
    let scheduledDate: Date
    let isDelivered: Bool
    let actionButtons: [NotificationAction]
    
    enum NotificationCategory: String, Codable, CaseIterable {
        case medicationReminder = "medication_reminder"
        case symptomCheck = "symptom_check"
        case exerciseReminder = "exercise_reminder"
        case hydrationReminder = "hydration_reminder"
        case appointmentReminder = "appointment_reminder"
        case flareUpAlert = "flare_up_alert"
        case emergencyAlert = "emergency_alert"
    }
    
    struct NotificationAction: Codable {
        let id: String
        let title: String
        let isDestructive: Bool
        let requiresAuthentication: Bool
    }
}

struct WatchComplication: Codable {
    let id: UUID
    let type: ComplicationType
    let displayText: String
    let shortText: String
    let value: Double?
    let unit: String?
    let color: String
    let lastUpdated: Date
    
    enum ComplicationType: String, Codable, CaseIterable {
        case painLevel = "pain_level"
        case medicationDue = "medication_due"
        case stepCount = "step_count"
        case heartRate = "heart_rate"
        case nextAppointment = "next_appointment"
        case flareRisk = "flare_risk"
        case hydrationLevel = "hydration_level"
        case sleepQuality = "sleep_quality"
    }
}

// MARK: - Apple Watch Manager

class AppleWatchManager: NSObject, ObservableObject {
    static let shared = AppleWatchManager()
    
    @Published var isWatchConnected = false
    @Published var isWatchAppInstalled = false
    @Published var watchBatteryLevel: Double = 0.0
    @Published var lastSyncDate: Date?
    @Published var pendingDataCount = 0
    
    private let session: WCSession
    private let healthKitManager: HealthKitManager
    private let notificationManager: WatchNotificationManager
    private let complicationManager: WatchComplicationManager
    private let workoutManager: WatchWorkoutManager
    
    private var syncTimer: Timer?
    private var healthDataBuffer: [WatchHealthData] = []
    private var symptomBuffer: [WatchSymptomEntry] = []
    
    override init() {
        self.session = WCSession.default
        self.healthKitManager = HealthKitManager.shared
        self.notificationManager = WatchNotificationManager()
        self.complicationManager = WatchComplicationManager()
        self.workoutManager = WatchWorkoutManager()
        
        super.init()
        
        if WCSession.isSupported() {
            session.delegate = self
            session.activate()
        }
        
        setupPeriodicSync()
        setupHealthKitObservers()
    }
    
    // MARK: - Public Methods
    
    func sendHealthDataToWatch(_ data: [String: Any]) {
        guard session.isReachable else {
            print("Watch is not reachable")
            return
        }
        
        session.sendMessage(data) { response in
            print("Watch responded: \(response)")
        } errorHandler: { error in
            print("Failed to send data to watch: \(error.localizedDescription)")
        }
    }
    
    func sendMedicationReminder(_ reminder: WatchMedicationReminder) {
        let data: [String: Any] = [
            "type": "medication_reminder",
            "reminder": try! JSONEncoder().encode(reminder)
        ]
        
        if session.isReachable {
            sendHealthDataToWatch(data)
        } else {
            // Store for later transmission
            try? session.updateApplicationContext(data)
        }
    }
    
    func requestSymptomUpdate() {
        let data: [String: Any] = [
            "type": "request_symptom_update",
            "timestamp": Date()
        ]
        
        sendHealthDataToWatch(data)
    }
    
    func syncAllData() {
        Task {
            await syncHealthData()
            await syncMedications()
            await syncAppointments()
            await syncComplications()
            
            DispatchQueue.main.async {
                self.lastSyncDate = Date()
            }
        }
    }
    
    func updateComplications() {
        complicationManager.updateAllComplications()
    }
    
    // MARK: - Private Methods
    
    private func setupPeriodicSync() {
        syncTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { _ in
            self.syncAllData()
        }
    }
    
    private func setupHealthKitObservers() {
        // Observe heart rate changes
        healthKitManager.observeHeartRate { [weak self] heartRate in
            self?.processHeartRateData(heartRate)
        }
        
        // Observe step count changes
        healthKitManager.observeStepCount { [weak self] steps in
            self?.processStepData(steps)
        }
        
        // Observe workout data
        healthKitManager.observeWorkouts { [weak self] workout in
            self?.processWorkoutData(workout)
        }
    }
    
    private func processHeartRateData(_ heartRate: Double) {
        let healthData = WatchHealthData(
            heartRate: heartRate,
            steps: nil,
            activeCalories: nil,
            workoutMinutes: nil,
            standHours: nil,
            timestamp: Date(),
            batteryLevel: watchBatteryLevel,
            isCharging: false
        )
        
        healthDataBuffer.append(healthData)
        
        // Send immediately if critical
        if heartRate > 120 || heartRate < 50 {
            sendCriticalHealthAlert(heartRate: heartRate)
        }
    }
    
    private func processStepData(_ steps: Int) {
        let healthData = WatchHealthData(
            heartRate: nil,
            steps: steps,
            activeCalories: nil,
            workoutMinutes: nil,
            standHours: nil,
            timestamp: Date(),
            batteryLevel: watchBatteryLevel,
            isCharging: false
        )
        
        healthDataBuffer.append(healthData)
    }
    
    private func processWorkoutData(_ workout: HKWorkout) {
        let watchWorkout = WatchWorkoutData(
            id: UUID(),
            workoutType: workout.workoutActivityType,
            startDate: workout.startDate,
            endDate: workout.endDate,
            duration: workout.duration,
            activeCalories: workout.totalEnergyBurned?.doubleValue(for: .kilocalorie()) ?? 0,
            totalCalories: workout.totalEnergyBurned?.doubleValue(for: .kilocalorie()) ?? 0,
            distance: workout.totalDistance?.doubleValue(for: .meter()),
            averageHeartRate: nil,
            maxHeartRate: nil,
            isActive: false
        )
        
        workoutManager.processWorkout(watchWorkout)
    }
    
    private func sendCriticalHealthAlert(heartRate: Double) {
        let alert: [String: Any] = [
            "type": "critical_health_alert",
            "heartRate": heartRate,
            "timestamp": Date(),
            "severity": heartRate > 150 || heartRate < 40 ? "high" : "medium"
        ]
        
        sendHealthDataToWatch(alert)
        
        // Also send local notification
        notificationManager.sendCriticalAlert(
            title: "Heart Rate Alert",
            body: "Heart rate detected: \(Int(heartRate)) BPM"
        )
    }
    
    @MainActor
    private func syncHealthData() async {
        guard !healthDataBuffer.isEmpty else { return }
        
        let dataToSync = healthDataBuffer
        healthDataBuffer.removeAll()
        
        let syncData: [String: Any] = [
            "type": "health_data_sync",
            "data": dataToSync.compactMap { try? JSONEncoder().encode($0) },
            "timestamp": Date()
        ]
        
        try? session.updateApplicationContext(syncData)
    }
    
    @MainActor
    private func syncMedications() async {
        // Get upcoming medication reminders
        let upcomingMeds = await getMedicationReminders()
        
        let syncData: [String: Any] = [
            "type": "medication_sync",
            "medications": upcomingMeds.compactMap { try? JSONEncoder().encode($0) },
            "timestamp": Date()
        ]
        
        try? session.updateApplicationContext(syncData)
    }
    
    @MainActor
    private func syncAppointments() async {
        // Get upcoming appointments
        let upcomingAppointments = await getUpcomingAppointments()
        
        let syncData: [String: Any] = [
            "type": "appointment_sync",
            "appointments": upcomingAppointments,
            "timestamp": Date()
        ]
        
        try? session.updateApplicationContext(syncData)
    }
    
    @MainActor
    private func syncComplications() async {
        let complications = await complicationManager.generateComplications()
        
        let syncData: [String: Any] = [
            "type": "complication_sync",
            "complications": complications.compactMap { try? JSONEncoder().encode($0) },
            "timestamp": Date()
        ]
        
        try? session.updateApplicationContext(syncData)
    }
    
    private func getMedicationReminders() async -> [WatchMedicationReminder] {
        // Mock implementation - replace with actual data
        return [
            WatchMedicationReminder(
                id: UUID(),
                medicationName: "Methotrexate",
                dosage: "15mg",
                scheduledTime: Date().addingTimeInterval(3600),
                isTaken: false,
                reminderType: .weekly
            )
        ]
    }
    
    private func getUpcomingAppointments() async -> [[String: Any]] {
        // Mock implementation - replace with actual data
        return [
            [
                "id": UUID().uuidString,
                "title": "Dr. Smith - Rheumatology",
                "date": Date().addingTimeInterval(86400),
                "type": "in_person"
            ]
        ]
    }
}

// MARK: - WCSession Delegate

extension AppleWatchManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.isWatchConnected = activationState == .activated
            self.isWatchAppInstalled = session.isWatchAppInstalled
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isWatchConnected = false
        }
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isWatchConnected = false
        }
        session.activate()
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any], replyHandler: @escaping ([String : Any]) -> Void) {
        handleWatchMessage(message, replyHandler: replyHandler)
    }
    
    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String : Any]) {
        handleWatchMessage(applicationContext)
    }
    
    private func handleWatchMessage(_ message: [String: Any], replyHandler: (([String: Any]) -> Void)? = nil) {
        guard let type = message["type"] as? String else { return }
        
        switch type {
        case "symptom_entry":
            handleSymptomEntry(message)
        case "medication_taken":
            handleMedicationTaken(message)
        case "emergency_alert":
            handleEmergencyAlert(message)
        case "workout_started":
            handleWorkoutStarted(message)
        case "workout_ended":
            handleWorkoutEnded(message)
        case "battery_status":
            handleBatteryStatus(message)
        default:
            print("Unknown message type: \(type)")
        }
        
        replyHandler?(["status": "received"])
    }
    
    private func handleSymptomEntry(_ message: [String: Any]) {
        guard let symptomData = message["symptom"] as? Data,
              let symptom = try? JSONDecoder().decode(WatchSymptomEntry.self, from: symptomData) else {
            return
        }
        
        symptomBuffer.append(symptom)
        
        // Process high severity symptoms immediately
        if symptom.severity >= 8 {
            notificationManager.sendHighSeveritySymptomAlert(symptom)
        }
    }
    
    private func handleMedicationTaken(_ message: [String: Any]) {
        guard let medicationId = message["medicationId"] as? String,
              let timestamp = message["timestamp"] as? Date else {
            return
        }
        
        // Update medication tracking
        Task {
            await updateMedicationStatus(id: medicationId, taken: true, timestamp: timestamp)
        }
    }
    
    private func handleEmergencyAlert(_ message: [String: Any]) {
        guard let alertType = message["alertType"] as? String,
              let severity = message["severity"] as? String else {
            return
        }
        
        notificationManager.sendEmergencyAlert(
            type: alertType,
            severity: severity,
            data: message
        )
    }
    
    private func handleWorkoutStarted(_ message: [String: Any]) {
        guard let workoutData = message["workout"] as? Data,
              let workout = try? JSONDecoder().decode(WatchWorkoutData.self, from: workoutData) else {
            return
        }
        
        workoutManager.startWorkout(workout)
    }
    
    private func handleWorkoutEnded(_ message: [String: Any]) {
        guard let workoutData = message["workout"] as? Data,
              let workout = try? JSONDecoder().decode(WatchWorkoutData.self, from: workoutData) else {
            return
        }
        
        workoutManager.endWorkout(workout)
    }
    
    private func handleBatteryStatus(_ message: [String: Any]) {
        if let batteryLevel = message["batteryLevel"] as? Double {
            DispatchQueue.main.async {
                self.watchBatteryLevel = batteryLevel
            }
        }
    }
    
    private func updateMedicationStatus(id: String, taken: Bool, timestamp: Date) async {
        // Implementation to update medication tracking
        print("Medication \(id) marked as taken at \(timestamp)")
    }
}

// MARK: - Watch Notification Manager

class WatchNotificationManager: ObservableObject {
    private let notificationCenter = UNUserNotificationCenter.current()
    
    func sendCriticalAlert(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .critical
        content.categoryIdentifier = "CRITICAL_HEALTH_ALERT"
        
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        
        notificationCenter.add(request)
    }
    
    func sendHighSeveritySymptomAlert(_ symptom: WatchSymptomEntry) {
        let content = UNMutableNotificationContent()
        content.title = "High Severity Symptom Detected"
        content.body = "\(symptom.symptomType) reported with severity \(symptom.severity)/10"
        content.sound = .default
        content.categoryIdentifier = "HIGH_SEVERITY_SYMPTOM"
        
        let request = UNNotificationRequest(
            identifier: symptom.id.uuidString,
            content: content,
            trigger: nil
        )
        
        notificationCenter.add(request)
    }
    
    func sendEmergencyAlert(type: String, severity: String, data: [String: Any]) {
        let content = UNMutableNotificationContent()
        content.title = "Emergency Alert"
        content.body = "\(type.capitalized) alert detected with \(severity) severity"
        content.sound = .critical
        content.categoryIdentifier = "EMERGENCY_ALERT"
        
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        
        notificationCenter.add(request)
    }
}

// MARK: - Watch Complication Manager

class WatchComplicationManager: ObservableObject {
    @Published var activeComplications: [WatchComplication] = []
    
    func generateComplications() async -> [WatchComplication] {
        var complications: [WatchComplication] = []
        
        // Pain level complication
        if let currentPainLevel = await getCurrentPainLevel() {
            complications.append(WatchComplication(
                id: UUID(),
                type: .painLevel,
                displayText: "Pain: \(currentPainLevel)/10",
                shortText: "\(currentPainLevel)",
                value: Double(currentPainLevel),
                unit: "/10",
                color: painLevelColor(currentPainLevel),
                lastUpdated: Date()
            ))
        }
        
        // Medication due complication
        if let nextMedication = await getNextMedication() {
            complications.append(WatchComplication(
                id: UUID(),
                type: .medicationDue,
                displayText: nextMedication.medicationName,
                shortText: "Med",
                value: nil,
                unit: nil,
                color: "blue",
                lastUpdated: Date()
            ))
        }
        
        // Step count complication
        if let stepCount = await getCurrentStepCount() {
            complications.append(WatchComplication(
                id: UUID(),
                type: .stepCount,
                displayText: "\(stepCount) steps",
                shortText: "\(stepCount)",
                value: Double(stepCount),
                unit: "steps",
                color: "green",
                lastUpdated: Date()
            ))
        }
        
        return complications
    }
    
    func updateAllComplications() {
        Task {
            let newComplications = await generateComplications()
            DispatchQueue.main.async {
                self.activeComplications = newComplications
            }
        }
    }
    
    private func getCurrentPainLevel() async -> Int? {
        // Mock implementation - replace with actual data
        return Int.random(in: 1...10)
    }
    
    private func getNextMedication() async -> WatchMedicationReminder? {
        // Mock implementation - replace with actual data
        return WatchMedicationReminder(
            id: UUID(),
            medicationName: "Methotrexate",
            dosage: "15mg",
            scheduledTime: Date().addingTimeInterval(3600),
            isTaken: false,
            reminderType: .weekly
        )
    }
    
    private func getCurrentStepCount() async -> Int? {
        // Mock implementation - replace with actual HealthKit data
        return Int.random(in: 1000...15000)
    }
    
    private func painLevelColor(_ level: Int) -> String {
        switch level {
        case 1...3:
            return "green"
        case 4...6:
            return "yellow"
        case 7...8:
            return "orange"
        case 9...10:
            return "red"
        default:
            return "gray"
        }
    }
}

// MARK: - Watch Workout Manager

class WatchWorkoutManager: ObservableObject {
    @Published var activeWorkout: WatchWorkoutData?
    @Published var workoutHistory: [WatchWorkoutData] = []
    
    func startWorkout(_ workout: WatchWorkoutData) {
        DispatchQueue.main.async {
            self.activeWorkout = workout
        }
        
        // Start monitoring workout metrics
        startWorkoutMonitoring()
    }
    
    func endWorkout(_ workout: WatchWorkoutData) {
        DispatchQueue.main.async {
            self.activeWorkout = nil
            self.workoutHistory.append(workout)
        }
        
        // Stop monitoring and save workout
        stopWorkoutMonitoring()
        saveWorkout(workout)
    }
    
    func processWorkout(_ workout: WatchWorkoutData) {
        // Analyze workout data for insights
        analyzeWorkoutPerformance(workout)
        
        // Update health trends
        updateHealthTrends(with: workout)
    }
    
    private func startWorkoutMonitoring() {
        // Implementation for real-time workout monitoring
        print("Started workout monitoring")
    }
    
    private func stopWorkoutMonitoring() {
        // Implementation to stop workout monitoring
        print("Stopped workout monitoring")
    }
    
    private func saveWorkout(_ workout: WatchWorkoutData) {
        // Save workout to HealthKit and local storage
        print("Saved workout: \(workout.workoutType)")
    }
    
    private func analyzeWorkoutPerformance(_ workout: WatchWorkoutData) {
        // Analyze workout performance and provide insights
        print("Analyzing workout performance")
    }
    
    private func updateHealthTrends(with workout: WatchWorkoutData) {
        // Update health trends based on workout data
        print("Updated health trends with workout data")
    }
}

// MARK: - Watch Data Sync Manager

class WatchDataSyncManager: ObservableObject {
    @Published var syncStatus: SyncStatus = .idle
    @Published var lastSyncDate: Date?
    @Published var pendingUploads: Int = 0
    
    enum SyncStatus {
        case idle
        case syncing
        case completed
        case failed(Error)
    }
    
    func performFullSync() async {
        DispatchQueue.main.async {
            self.syncStatus = .syncing
        }
        
        do {
            // Sync health data
            try await syncHealthData()
            
            // Sync symptoms
            try await syncSymptoms()
            
            // Sync medications
            try await syncMedications()
            
            // Sync workouts
            try await syncWorkouts()
            
            DispatchQueue.main.async {
                self.syncStatus = .completed
                self.lastSyncDate = Date()
                self.pendingUploads = 0
            }
        } catch {
            DispatchQueue.main.async {
                self.syncStatus = .failed(error)
            }
        }
    }
    
    private func syncHealthData() async throws {
        // Implementation for syncing health data
        try await Task.sleep(nanoseconds: 1_000_000_000) // Simulate network delay
    }
    
    private func syncSymptoms() async throws {
        // Implementation for syncing symptoms
        try await Task.sleep(nanoseconds: 500_000_000)
    }
    
    private func syncMedications() async throws {
        // Implementation for syncing medications
        try await Task.sleep(nanoseconds: 500_000_000)
    }
    
    private func syncWorkouts() async throws {
        // Implementation for syncing workouts
        try await Task.sleep(nanoseconds: 500_000_000)
    }
}

// MARK: - Watch Settings Manager

class WatchSettingsManager: ObservableObject {
    @Published var settings: WatchSettings
    
    init() {
        self.settings = WatchSettings()
        loadSettings()
    }
    
    func updateSettings(_ newSettings: WatchSettings) {
        self.settings = newSettings
        saveSettings()
        
        // Send updated settings to watch
        AppleWatchManager.shared.sendHealthDataToWatch([
            "type": "settings_update",
            "settings": try! JSONEncoder().encode(newSettings)
        ])
    }
    
    private func loadSettings() {
        // Load settings from UserDefaults
        if let data = UserDefaults.standard.data(forKey: "WatchSettings"),
           let settings = try? JSONDecoder().decode(WatchSettings.self, from: data) {
            self.settings = settings
        }
    }
    
    private func saveSettings() {
        // Save settings to UserDefaults
        if let data = try? JSONEncoder().encode(settings) {
            UserDefaults.standard.set(data, forKey: "WatchSettings")
        }
    }
}

struct WatchSettings: Codable {
    var enableHeartRateMonitoring: Bool = true
    var enableStepTracking: Bool = true
    var enableWorkoutTracking: Bool = true
    var medicationReminderEnabled: Bool = true
    var symptomReminderEnabled: Bool = true
    var emergencyContactsEnabled: Bool = true
    var hapticFeedbackEnabled: Bool = true
    var complicationUpdateInterval: TimeInterval = 300 // 5 minutes
    var dataRetentionDays: Int = 30
    var autoSyncEnabled: Bool = true
    var batteryOptimizationEnabled: Bool = true
}

// MARK: - Watch Error Types

enum WatchError: Error, LocalizedError {
    case watchNotConnected
    case watchAppNotInstalled
    case syncFailed(String)
    case dataCorrupted
    case insufficientPermissions
    case batteryTooLow
    
    var errorDescription: String? {
        switch self {
        case .watchNotConnected:
            return "Apple Watch is not connected"
        case .watchAppNotInstalled:
            return "InflamAI is not installed on Apple Watch"
        case .syncFailed(let reason):
            return "Sync failed: \(reason)"
        case .dataCorrupted:
            return "Watch data is corrupted"
        case .insufficientPermissions:
            return "Insufficient permissions for watch communication"
        case .batteryTooLow:
            return "Watch battery is too low for operation"
        }
    }
}