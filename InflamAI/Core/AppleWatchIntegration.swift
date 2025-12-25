//
//  AppleWatchIntegration.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import WatchConnectivity
import HealthKit
import Combine
import Foundation

// MARK: - Apple Watch Integration Manager
class AppleWatchIntegrationManager: NSObject, ObservableObject {
    @Published var isWatchConnected = false
    @Published var isWatchAppInstalled = false
    @Published var watchHealthData: WatchHealthData?
    @Published var watchBatteryLevel: Double = 0.0
    @Published var lastSyncDate: Date?
    @Published var syncStatus: SyncStatus = .idle
    @Published var watchAlerts: [WatchAlert] = []
    @Published var watchWorkouts: [WatchWorkout] = []
    @Published var watchVitalSigns: WatchVitalSigns?
    
    private let session = WCSession.default
    private let healthStore = HKHealthStore()
    private let watchDataProcessor = WatchDataProcessor()
    private let watchNotificationManager = WatchNotificationManager()
    private let watchWorkoutManager = WatchWorkoutManager()
    private let watchComplicationManager = WatchComplicationManager()
    
    private var cancellables = Set<AnyCancellable>()
    private var healthDataTimer: Timer?
    private var syncTimer: Timer?
    
    override init() {
        super.init()
        setupWatchConnectivity()
        setupHealthDataSync()
        setupPeriodicSync()
    }
    
    // MARK: - Watch Connectivity Setup
    private func setupWatchConnectivity() {
        guard WCSession.isSupported() else {
            print("WatchConnectivity not supported")
            return
        }
        
        session.delegate = self
        session.activate()
    }
    
    private func setupHealthDataSync() {
        // Start continuous health data monitoring
        startHealthDataMonitoring()
        
        // Setup data sync timer
        healthDataTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.syncHealthDataFromWatch()
        }
    }
    
    private func setupPeriodicSync() {
        // Full sync every 5 minutes
        syncTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak self] _ in
            self?.performFullSync()
        }
    }
    
    // MARK: - Health Data Monitoring
    private func startHealthDataMonitoring() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        let healthTypes: Set<HKSampleType> = [
            HKQuantityType.quantityType(forIdentifier: .heartRate)!,
            HKQuantityType.quantityType(forIdentifier: .stepCount)!,
            HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!,
            HKQuantityType.quantityType(forIdentifier: .distanceWalkingRunning)!,
            HKQuantityType.quantityType(forIdentifier: .oxygenSaturation)!,
            HKQuantityType.quantityType(forIdentifier: .bodyTemperature)!,
            HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: healthTypes) { [weak self] success, error in
            if success {
                self?.setupHealthKitObservers()
            }
        }
    }
    
    private func setupHealthKitObservers() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let stepCountType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
        
        // Heart rate observer
        let heartRateQuery = HKObserverQuery(sampleType: heartRateType, predicate: nil) { [weak self] _, _, error in
            if error == nil {
                self?.fetchLatestHeartRate()
            }
        }
        
        // Step count observer
        let stepCountQuery = HKObserverQuery(sampleType: stepCountType, predicate: nil) { [weak self] _, _, error in
            if error == nil {
                self?.fetchLatestStepCount()
            }
        }
        
        healthStore.execute(heartRateQuery)
        healthStore.execute(stepCountQuery)
    }
    
    // MARK: - Data Fetching
    private func fetchLatestHeartRate() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: heartRateType, predicate: nil, limit: 1, sortDescriptors: [sortDescriptor]) { [weak self] _, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else { return }
            
            let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            
            DispatchQueue.main.async {
                self?.updateWatchVitalSigns(heartRate: heartRate)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func fetchLatestStepCount() {
        let stepCountType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())
        let predicate = HKQuery.predicateForSamples(withStart: startOfDay, end: Date(), options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: stepCountType, quantitySamplePredicate: predicate, options: .cumulativeSum) { [weak self] _, result, error in
            
            guard let result = result, let sum = result.sumQuantity() else { return }
            
            let steps = sum.doubleValue(for: HKUnit.count())
            
            DispatchQueue.main.async {
                self?.updateWatchHealthData(steps: Int(steps))
            }
        }
        
        healthStore.execute(query)
    }
    
    private func updateWatchVitalSigns(heartRate: Double) {
        if watchVitalSigns == nil {
            watchVitalSigns = WatchVitalSigns(
                heartRate: heartRate,
                oxygenSaturation: nil,
                bodyTemperature: nil,
                respiratoryRate: nil,
                timestamp: Date()
            )
        } else {
            watchVitalSigns?.heartRate = heartRate
            watchVitalSigns?.timestamp = Date()
        }
    }
    
    private func updateWatchHealthData(steps: Int) {
        if watchHealthData == nil {
            watchHealthData = WatchHealthData(
                steps: steps,
                activeCalories: 0,
                distance: 0.0,
                exerciseMinutes: 0,
                standHours: 0,
                timestamp: Date()
            )
        } else {
            watchHealthData?.steps = steps
            watchHealthData?.timestamp = Date()
        }
    }
    
    // MARK: - Watch Communication
    func sendDataToWatch(_ data: [String: Any]) {
        guard session.isReachable else {
            // Store data for later transmission
            queueDataForLaterTransmission(data)
            return
        }
        
        session.sendMessage(data, replyHandler: { [weak self] reply in
            self?.handleWatchReply(reply)
        }) { error in
            print("Failed to send data to watch: \(error.localizedDescription)")
        }
    }
    
    func sendAlertToWatch(_ alert: WatchAlert) {
        let alertData: [String: Any] = [
            "type": "alert",
            "id": alert.id.uuidString,
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.rawValue,
            "timestamp": alert.timestamp.timeIntervalSince1970
        ]
        
        sendDataToWatch(alertData)
    }
    
    func sendMedicationReminderToWatch(_ reminder: MedicationReminder) {
        let reminderData: [String: Any] = [
            "type": "medication_reminder",
            "id": reminder.id.uuidString,
            "medicationName": reminder.medicationName,
            "dosage": reminder.dosage,
            "scheduledTime": reminder.scheduledTime.timeIntervalSince1970,
            "instructions": reminder.instructions
        ]
        
        sendDataToWatch(reminderData)
    }
    
    func sendWorkoutDataToWatch(_ workout: WorkoutSession) {
        let workoutData: [String: Any] = [
            "type": "workout_session",
            "id": workout.id.uuidString,
            "workoutType": workout.workoutType.rawValue,
            "duration": workout.duration,
            "targetHeartRate": workout.targetHeartRate ?? 0,
            "instructions": workout.instructions
        ]
        
        sendDataToWatch(workoutData)
    }
    
    private func queueDataForLaterTransmission(_ data: [String: Any]) {
        // Store data in UserDefaults or Core Data for later transmission
        var queuedData = UserDefaults.standard.array(forKey: "QueuedWatchData") as? [[String: Any]] ?? []
        queuedData.append(data)
        UserDefaults.standard.set(queuedData, forKey: "QueuedWatchData")
    }
    
    private func sendQueuedData() {
        guard let queuedData = UserDefaults.standard.array(forKey: "QueuedWatchData") as? [[String: Any]] else { return }
        
        for data in queuedData {
            sendDataToWatch(data)
        }
        
        UserDefaults.standard.removeObject(forKey: "QueuedWatchData")
    }
    
    private func handleWatchReply(_ reply: [String: Any]) {
        // Process reply from watch
        if let type = reply["type"] as? String {
            switch type {
            case "health_data":
                processHealthDataFromWatch(reply)
            case "workout_completed":
                processWorkoutCompletion(reply)
            case "alert_acknowledged":
                processAlertAcknowledgment(reply)
            case "medication_taken":
                processMedicationTaken(reply)
            default:
                break
            }
        }
    }
    
    // MARK: - Data Processing
    private func processHealthDataFromWatch(_ data: [String: Any]) {
        watchDataProcessor.processHealthData(data) { [weak self] processedData in
            DispatchQueue.main.async {
                self?.watchHealthData = processedData
                self?.lastSyncDate = Date()
            }
        }
    }
    
    private func processWorkoutCompletion(_ data: [String: Any]) {
        watchWorkoutManager.processWorkoutCompletion(data) { [weak self] workout in
            DispatchQueue.main.async {
                self?.watchWorkouts.append(workout)
            }
        }
    }
    
    private func processAlertAcknowledgment(_ data: [String: Any]) {
        guard let alertId = data["alert_id"] as? String,
              let uuid = UUID(uuidString: alertId) else { return }
        
        if let index = watchAlerts.firstIndex(where: { $0.id == uuid }) {
            watchAlerts[index].isAcknowledged = true
            watchAlerts[index].acknowledgedDate = Date()
        }
    }
    
    private func processMedicationTaken(_ data: [String: Any]) {
        // Process medication taken confirmation from watch
        NotificationCenter.default.post(name: .medicationTakenFromWatch, object: data)
    }
    
    // MARK: - Sync Operations
    private func syncHealthDataFromWatch() {
        guard session.isReachable else { return }
        
        syncStatus = .syncing
        
        let syncRequest: [String: Any] = [
            "type": "sync_request",
            "timestamp": Date().timeIntervalSince1970,
            "data_types": ["health", "workouts", "vitals"]
        ]
        
        sendDataToWatch(syncRequest)
    }
    
    private func performFullSync() {
        guard session.isReachable else { return }
        
        syncStatus = .syncing
        
        // Send queued data first
        sendQueuedData()
        
        // Request full data sync from watch
        let fullSyncRequest: [String: Any] = [
            "type": "full_sync_request",
            "timestamp": Date().timeIntervalSince1970
        ]
        
        sendDataToWatch(fullSyncRequest)
    }
    
    // MARK: - Watch App Management
    func checkWatchAppInstallation() {
        guard session.isReachable else {
            isWatchAppInstalled = false
            return
        }
        
        let checkRequest: [String: Any] = [
            "type": "app_check",
            "timestamp": Date().timeIntervalSince1970
        ]
        
        session.sendMessage(checkRequest, replyHandler: { [weak self] reply in
            if let installed = reply["app_installed"] as? Bool {
                DispatchQueue.main.async {
                    self?.isWatchAppInstalled = installed
                }
            }
        }) { error in
            print("Failed to check watch app installation: \(error.localizedDescription)")
        }
    }
    
    func installWatchApp() {
        // Trigger watch app installation
        guard let watchAppURL = URL(string: "https://apps.apple.com/app/inflamai-watch") else { return }
        
        if UIApplication.shared.canOpenURL(watchAppURL) {
            UIApplication.shared.open(watchAppURL)
        }
    }
    
    // MARK: - Complications
    func updateWatchComplications() {
        watchComplicationManager.updateComplications(with: watchHealthData)
    }
    
    // MARK: - Emergency Features
    func sendEmergencyAlert() {
        let emergencyAlert = WatchAlert(
            id: UUID(),
            title: "Emergency Alert",
            message: "Health emergency detected. Seeking help.",
            severity: .critical,
            timestamp: Date()
        )
        
        sendAlertToWatch(emergencyAlert)
        
        // Also trigger emergency contacts
        NotificationCenter.default.post(name: .emergencyAlertTriggered, object: emergencyAlert)
    }
    
    // MARK: - Cleanup
    deinit {
        healthDataTimer?.invalidate()
        syncTimer?.invalidate()
        cancellables.removeAll()
    }
}

// MARK: - WCSessionDelegate
extension AppleWatchIntegrationManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.isWatchConnected = activationState == .activated
            
            if activationState == .activated {
                self.checkWatchAppInstallation()
                self.sendQueuedData()
            }
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
        
        // Reactivate session
        session.activate()
    }
    
    func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isWatchConnected = session.isReachable
            
            if session.isReachable {
                self.sendQueuedData()
            }
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any], replyHandler: @escaping ([String : Any]) -> Void) {
        handleWatchMessage(message, replyHandler: replyHandler)
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        handleWatchMessage(message, replyHandler: nil)
    }
    
    private func handleWatchMessage(_ message: [String: Any], replyHandler: (([String: Any]) -> Void)?) {
        guard let type = message["type"] as? String else { return }
        
        switch type {
        case "health_data_update":
            processHealthDataFromWatch(message)
            replyHandler?(["status": "received"])
            
        case "workout_started":
            processWorkoutStart(message)
            replyHandler?(["status": "acknowledged"])
            
        case "emergency_alert":
            processEmergencyAlert(message)
            replyHandler?(["status": "emergency_acknowledged"])
            
        case "battery_status":
            if let batteryLevel = message["battery_level"] as? Double {
                DispatchQueue.main.async {
                    self.watchBatteryLevel = batteryLevel
                }
            }
            replyHandler?(["status": "received"])
            
        default:
            replyHandler?(["status": "unknown_type"])
        }
    }
    
    private func processWorkoutStart(_ message: [String: Any]) {
        // Process workout start from watch
        NotificationCenter.default.post(name: .workoutStartedFromWatch, object: message)
    }
    
    private func processEmergencyAlert(_ message: [String: Any]) {
        // Process emergency alert from watch
        let emergencyAlert = WatchAlert(
            id: UUID(),
            title: "Emergency Alert from Watch",
            message: message["message"] as? String ?? "Emergency detected",
            severity: .critical,
            timestamp: Date()
        )
        
        DispatchQueue.main.async {
            self.watchAlerts.append(emergencyAlert)
        }
        
        NotificationCenter.default.post(name: .emergencyAlertFromWatch, object: emergencyAlert)
    }
}

// MARK: - Watch Data Processor
class WatchDataProcessor {
    func processHealthData(_ data: [String: Any], completion: @escaping (WatchHealthData) -> Void) {
        // Process raw health data from watch
        let steps = data["steps"] as? Int ?? 0
        let activeCalories = data["active_calories"] as? Int ?? 0
        let distance = data["distance"] as? Double ?? 0.0
        let exerciseMinutes = data["exercise_minutes"] as? Int ?? 0
        let standHours = data["stand_hours"] as? Int ?? 0
        
        let healthData = WatchHealthData(
            steps: steps,
            activeCalories: activeCalories,
            distance: distance,
            exerciseMinutes: exerciseMinutes,
            standHours: standHours,
            timestamp: Date()
        )
        
        completion(healthData)
    }
}

// MARK: - Watch Notification Manager
class WatchNotificationManager {
    func sendNotificationToWatch(_ notification: WatchNotification) {
        // Send notification to watch
        let notificationData: [String: Any] = [
            "type": "notification",
            "id": notification.id.uuidString,
            "title": notification.title,
            "body": notification.body,
            "category": notification.category.rawValue,
            "timestamp": notification.timestamp.timeIntervalSince1970
        ]
        
        NotificationCenter.default.post(name: .sendNotificationToWatch, object: notificationData)
    }
}

// MARK: - Watch Workout Manager
class WatchWorkoutManager {
    func processWorkoutCompletion(_ data: [String: Any], completion: @escaping (WatchWorkout) -> Void) {
        guard let workoutTypeString = data["workout_type"] as? String,
              let workoutType = WatchWorkoutType(rawValue: workoutTypeString),
              let duration = data["duration"] as? TimeInterval,
              let startTime = data["start_time"] as? TimeInterval else { return }
        
        let workout = WatchWorkout(
            id: UUID(),
            workoutType: workoutType,
            startTime: Date(timeIntervalSince1970: startTime),
            duration: duration,
            calories: data["calories"] as? Int ?? 0,
            distance: data["distance"] as? Double ?? 0.0,
            averageHeartRate: data["average_heart_rate"] as? Double ?? 0.0,
            maxHeartRate: data["max_heart_rate"] as? Double ?? 0.0
        )
        
        completion(workout)
    }
}

// MARK: - Watch Complication Manager
class WatchComplicationManager {
    func updateComplications(with healthData: WatchHealthData?) {
        guard let healthData = healthData else { return }
        
        let complicationData: [String: Any] = [
            "type": "complication_update",
            "steps": healthData.steps,
            "active_calories": healthData.activeCalories,
            "timestamp": healthData.timestamp.timeIntervalSince1970
        ]
        
        NotificationCenter.default.post(name: .updateWatchComplications, object: complicationData)
    }
}

// MARK: - Supporting Data Types
struct WatchHealthData {
    var steps: Int
    var activeCalories: Int
    var distance: Double
    var exerciseMinutes: Int
    var standHours: Int
    var timestamp: Date
}

struct WatchVitalSigns {
    var heartRate: Double
    var oxygenSaturation: Double?
    var bodyTemperature: Double?
    var respiratoryRate: Double?
    var timestamp: Date
}

struct WatchAlert {
    let id: UUID
    let title: String
    let message: String
    let severity: AlertSeverity
    let timestamp: Date
    var isAcknowledged: Bool = false
    var acknowledgedDate: Date?
}

struct WatchWorkout {
    let id: UUID
    let workoutType: WatchWorkoutType
    let startTime: Date
    let duration: TimeInterval
    let calories: Int
    let distance: Double
    let averageHeartRate: Double
    let maxHeartRate: Double
}

struct WatchNotification {
    let id: UUID
    let title: String
    let body: String
    let category: NotificationCategory
    let timestamp: Date
}

struct MedicationReminder {
    let id: UUID
    let medicationName: String
    let dosage: String
    let scheduledTime: Date
    let instructions: String
}

struct WorkoutSession {
    let id: UUID
    let workoutType: WatchWorkoutType
    let duration: TimeInterval
    let targetHeartRate: Double?
    let instructions: String
}

// MARK: - Enums
enum SyncStatus {
    case idle
    case syncing
    case completed
    case failed
}

enum AlertSeverity: String {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

enum WatchWorkoutType: String, CaseIterable {
    case walking = "walking"
    case running = "running"
    case cycling = "cycling"
    case swimming = "swimming"
    case yoga = "yoga"
    case strength = "strength"
    case other = "other"
}

enum NotificationCategory: String {
    case medication = "medication"
    case workout = "workout"
    case health = "health"
    case emergency = "emergency"
    case reminder = "reminder"
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let medicationTakenFromWatch = Notification.Name("medicationTakenFromWatch")
    static let workoutStartedFromWatch = Notification.Name("workoutStartedFromWatch")
    static let emergencyAlertTriggered = Notification.Name("emergencyAlertTriggered")
    static let emergencyAlertFromWatch = Notification.Name("emergencyAlertFromWatch")
    static let sendNotificationToWatch = Notification.Name("sendNotificationToWatch")
    static let updateWatchComplications = Notification.Name("updateWatchComplications")
}