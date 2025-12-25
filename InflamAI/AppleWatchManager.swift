//
//  AppleWatchManager.swift
//  InflamAI-Swift
//
//  Apple Watch companion app integration and management
//

import Foundation
import WatchConnectivity
import HealthKit
import UserNotifications
import Combine
import CoreData

// MARK: - Watch Communication Types

enum WatchMessageType: String, CaseIterable {
    case painLevel = "painLevel"
    case medicationTaken = "medicationTaken"
    case moodUpdate = "moodUpdate"
    case activityUpdate = "activityUpdate"
    case emergencyAlert = "emergencyAlert"
    case syncRequest = "syncRequest"
    case settingsUpdate = "settingsUpdate"
    case reminderResponse = "reminderResponse"
    case heartRateData = "heartRateData"
    case sleepData = "sleepData"
    case stepsData = "stepsData"
    case workoutData = "workoutData"
}

struct WatchMessage: Codable {
    let type: String
    let timestamp: Date
    let data: [String: Any]
    let messageId: String
    
    init(type: WatchMessageType, data: [String: Any] = [:]) {
        self.type = type.rawValue
        self.timestamp = Date()
        self.data = data
        self.messageId = UUID().uuidString
    }
    
    // Custom encoding/decoding for [String: Any]
    enum CodingKeys: String, CodingKey {
        case type, timestamp, messageId, data
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        messageId = try container.decode(String.self, forKey: .messageId)
        
        // Decode data as JSON
        if let jsonData = try? container.decode(Data.self, forKey: .data),
           let jsonObject = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
            data = jsonObject
        } else {
            data = [:]
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(messageId, forKey: .messageId)
        
        // Encode data as JSON
        let jsonData = try JSONSerialization.data(withJSONObject: data)
        try container.encode(jsonData, forKey: .data)
    }
}

struct WatchAppContext: Codable {
    let medications: [WatchMedication]
    let todaysPainLevels: [WatchPainEntry]
    let upcomingReminders: [WatchReminder]
    let userSettings: WatchUserSettings
    let lastSync: Date
}

struct WatchMedication: Codable, Identifiable {
    let id: String
    let name: String
    let dosage: String
    let frequency: String
    let nextDueTime: Date?
    let isActive: Bool
    let color: String
}

struct WatchPainEntry: Codable, Identifiable {
    let id: String
    let level: Int
    let location: String
    let timestamp: Date
    let notes: String?
}

struct WatchReminder: Codable, Identifiable {
    let id: String
    let title: String
    let message: String
    let scheduledTime: Date
    let type: String // medication, exercise, checkup
    let isCompleted: Bool
}

struct WatchUserSettings: Codable {
    let enableHapticFeedback: Bool
    let enableVoiceReminders: Bool
    let reminderSound: String
    let complicationStyle: String
    let autoTrackWorkouts: Bool
    let emergencyContactEnabled: Bool
}

// MARK: - Apple Watch Manager

class AppleWatchManager: NSObject, ObservableObject {
    // Core Data
    private let context: NSManagedObjectContext
    
    // Watch Connectivity
    private let session = WCSession.default
    
    // Published Properties
    @Published var isWatchAppInstalled = false
    @Published var isWatchReachable = false
    @Published var lastSyncTime: Date?
    @Published var pendingMessages: [WatchMessage] = []
    @Published var watchBatteryLevel: Float = 0.0
    @Published var connectionStatus: WatchConnectionStatus = .disconnected
    
    // Health Data
    @Published var watchHealthData: WatchHealthData = WatchHealthData()
    
    // Settings
    @Published var autoSyncEnabled = true
    @Published var backgroundSyncEnabled = true
    @Published var emergencyFeaturesEnabled = true
    
    // Internal State
    private var messageQueue: [WatchMessage] = []
    private var syncTimer: Timer?
    private var healthKitManager: HealthKitManager?
    
    // Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    init(context: NSManagedObjectContext) {
        self.context = context
        super.init()
        
        setupWatchConnectivity()
        loadSettings()
        setupPeriodicSync()
        setupHealthKitIntegration()
    }
    
    // MARK: - Setup
    
    private func setupWatchConnectivity() {
        guard WCSession.isSupported() else {
            print("Watch Connectivity not supported")
            return
        }
        
        session.delegate = self
        session.activate()
    }
    
    private func loadSettings() {
        autoSyncEnabled = UserDefaults.standard.bool(forKey: "watch_autoSync")
        if !autoSyncEnabled && UserDefaults.standard.object(forKey: "watch_autoSync") == nil {
            autoSyncEnabled = true
        }
        
        backgroundSyncEnabled = UserDefaults.standard.bool(forKey: "watch_backgroundSync")
        if !backgroundSyncEnabled && UserDefaults.standard.object(forKey: "watch_backgroundSync") == nil {
            backgroundSyncEnabled = true
        }
        
        emergencyFeaturesEnabled = UserDefaults.standard.bool(forKey: "watch_emergencyFeatures")
        if !emergencyFeaturesEnabled && UserDefaults.standard.object(forKey: "watch_emergencyFeatures") == nil {
            emergencyFeaturesEnabled = true
        }
        
        if let lastSync = UserDefaults.standard.object(forKey: "watch_lastSync") as? Date {
            lastSyncTime = lastSync
        }
    }
    
    private func setupPeriodicSync() {
        guard autoSyncEnabled else { return }
        
        // Sync every 5 minutes when app is active
        syncTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak self] _ in
            self?.syncWithWatch()
        }
    }
    
    private func setupHealthKitIntegration() {
        healthKitManager = HealthKitManager()
        
        // Listen for health data updates
        healthKitManager?.healthDataPublisher
            .sink { [weak self] healthData in
                self?.sendHealthDataToWatch(healthData)
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Public Methods
    
    func syncWithWatch() {
        guard session.isReachable else {
            queueSyncForLater()
            return
        }
        
        let context = createWatchAppContext()
        sendApplicationContext(context)
        
        // Process pending messages
        processPendingMessages()
        
        lastSyncTime = Date()
        UserDefaults.standard.set(lastSyncTime, forKey: "watch_lastSync")
    }
    
    func sendPainLevel(_ level: Int, location: String, notes: String? = nil) {
        let message = WatchMessage(type: .painLevel, data: [
            "level": level,
            "location": location,
            "notes": notes ?? "",
            "timestamp": Date().timeIntervalSince1970
        ])
        
        sendMessage(message)
    }
    
    func sendMedicationTaken(_ medicationId: String, timestamp: Date = Date()) {
        let message = WatchMessage(type: .medicationTaken, data: [
            "medicationId": medicationId,
            "timestamp": timestamp.timeIntervalSince1970
        ])
        
        sendMessage(message)
    }
    
    func sendMoodUpdate(_ mood: String, energy: Int, notes: String? = nil) {
        let message = WatchMessage(type: .moodUpdate, data: [
            "mood": mood,
            "energy": energy,
            "notes": notes ?? "",
            "timestamp": Date().timeIntervalSince1970
        ])
        
        sendMessage(message)
    }
    
    func sendEmergencyAlert(_ type: String, location: String? = nil) {
        guard emergencyFeaturesEnabled else { return }
        
        let message = WatchMessage(type: .emergencyAlert, data: [
            "alertType": type,
            "location": location ?? "",
            "timestamp": Date().timeIntervalSince1970,
            "urgent": true
        ])
        
        sendMessage(message, priority: .high)
    }
    
    func requestWatchSync() {
        let message = WatchMessage(type: .syncRequest)
        sendMessage(message)
    }
    
    func updateWatchSettings(_ settings: WatchUserSettings) {
        let message = WatchMessage(type: .settingsUpdate, data: [
            "settings": try? JSONEncoder().encode(settings)
        ])
        
        sendMessage(message)
    }
    
    // MARK: - Private Methods
    
    private func createWatchAppContext() -> WatchAppContext {
        let medications = fetchWatchMedications()
        let painLevels = fetchTodaysPainLevels()
        let reminders = fetchUpcomingReminders()
        let settings = createWatchUserSettings()
        
        return WatchAppContext(
            medications: medications,
            todaysPainLevels: painLevels,
            upcomingReminders: reminders,
            userSettings: settings,
            lastSync: Date()
        )
    }
    
    private func fetchWatchMedications() -> [WatchMedication] {
        let request: NSFetchRequest<Medication> = Medication.fetchRequest()
        request.predicate = NSPredicate(format: "isActive == YES")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \Medication.name, ascending: true)]
        
        do {
            let medications = try context.fetch(request)
            return medications.map { medication in
                WatchMedication(
                    id: medication.id?.uuidString ?? UUID().uuidString,
                    name: medication.name ?? "Unknown",
                    dosage: medication.dosage ?? "",
                    frequency: medication.frequency ?? "",
                    nextDueTime: calculateNextDueTime(for: medication),
                    isActive: medication.isActive,
                    color: medication.color ?? "blue"
                )
            }
        } catch {
            print("Error fetching medications for watch: \(error)")
            return []
        }
    }
    
    private func fetchTodaysPainLevels() -> [WatchPainEntry] {
        let request: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        let today = Calendar.current.startOfDay(for: Date())
        let tomorrow = Calendar.current.date(byAdding: .day, value: 1, to: today)!
        
        request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@", today as NSDate, tomorrow as NSDate)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: false)]
        request.fetchLimit = 10
        
        do {
            let painEntries = try context.fetch(request)
            return painEntries.map { entry in
                WatchPainEntry(
                    id: entry.id?.uuidString ?? UUID().uuidString,
                    level: Int(entry.painLevel),
                    location: entry.location ?? "General",
                    timestamp: entry.timestamp ?? Date(),
                    notes: entry.notes
                )
            }
        } catch {
            print("Error fetching pain levels for watch: \(error)")
            return []
        }
    }
    
    private func fetchUpcomingReminders() -> [WatchReminder] {
        // This would fetch from a reminders/notifications system
        // For now, return mock data
        let now = Date()
        let calendar = Calendar.current
        
        return [
            WatchReminder(
                id: UUID().uuidString,
                title: "Take Medication",
                message: "Time for your morning medication",
                scheduledTime: calendar.date(byAdding: .hour, value: 1, to: now) ?? now,
                type: "medication",
                isCompleted: false
            ),
            WatchReminder(
                id: UUID().uuidString,
                title: "Exercise",
                message: "Light stretching routine",
                scheduledTime: calendar.date(byAdding: .hour, value: 3, to: now) ?? now,
                type: "exercise",
                isCompleted: false
            )
        ]
    }
    
    private func createWatchUserSettings() -> WatchUserSettings {
        return WatchUserSettings(
            enableHapticFeedback: UserDefaults.standard.bool(forKey: "watch_hapticFeedback"),
            enableVoiceReminders: UserDefaults.standard.bool(forKey: "watch_voiceReminders"),
            reminderSound: UserDefaults.standard.string(forKey: "watch_reminderSound") ?? "default",
            complicationStyle: UserDefaults.standard.string(forKey: "watch_complicationStyle") ?? "circular",
            autoTrackWorkouts: UserDefaults.standard.bool(forKey: "watch_autoTrackWorkouts"),
            emergencyContactEnabled: emergencyFeaturesEnabled
        )
    }
    
    private func calculateNextDueTime(for medication: Medication) -> Date? {
        // This would calculate based on the medication schedule
        // For now, return a simple calculation
        guard let frequency = medication.frequency else { return nil }
        
        let calendar = Calendar.current
        let now = Date()
        
        if frequency.contains("daily") {
            return calendar.date(byAdding: .day, value: 1, to: now)
        } else if frequency.contains("twice") {
            return calendar.date(byAdding: .hour, value: 12, to: now)
        } else if frequency.contains("weekly") {
            return calendar.date(byAdding: .weekOfYear, value: 1, to: now)
        }
        
        return nil
    }
    
    private func sendMessage(_ message: WatchMessage, priority: MessagePriority = .normal) {
        guard session.isReachable else {
            queueMessage(message)
            return
        }
        
        do {
            let data = try JSONEncoder().encode(message)
            let messageDict = ["messageData": data]
            
            if priority == .high {
                // Use immediate message for high priority
                session.sendMessage(messageDict, replyHandler: { reply in
                    self.handleMessageReply(reply, for: message)
                }, errorHandler: { error in
                    print("Error sending high priority message: \(error)")
                    self.queueMessage(message)
                })
            } else {
                // Use user info transfer for normal priority
                session.transferUserInfo(messageDict)
            }
        } catch {
            print("Error encoding message: \(error)")
        }
    }
    
    private func sendApplicationContext(_ context: WatchAppContext) {
        do {
            let data = try JSONEncoder().encode(context)
            let contextDict = ["appContext": data]
            
            try session.updateApplicationContext(contextDict)
        } catch {
            print("Error sending application context: \(error)")
        }
    }
    
    private func queueMessage(_ message: WatchMessage) {
        messageQueue.append(message)
        
        // Limit queue size
        if messageQueue.count > 50 {
            messageQueue.removeFirst()
        }
    }
    
    private func queueSyncForLater() {
        // Schedule sync for when watch becomes reachable
        DispatchQueue.main.asyncAfter(deadline: .now() + 30) {
            if self.session.isReachable {
                self.syncWithWatch()
            } else {
                self.queueSyncForLater()
            }
        }
    }
    
    private func processPendingMessages() {
        guard !messageQueue.isEmpty && session.isReachable else { return }
        
        let messagesToSend = Array(messageQueue.prefix(10)) // Send up to 10 messages at once
        messageQueue.removeFirst(min(10, messageQueue.count))
        
        for message in messagesToSend {
            sendMessage(message)
        }
    }
    
    private func handleMessageReply(_ reply: [String: Any], for message: WatchMessage) {
        // Handle reply from watch
        if let status = reply["status"] as? String {
            print("Message \(message.messageId) status: \(status)")
        }
    }
    
    private func sendHealthDataToWatch(_ healthData: HealthData) {
        let message = WatchMessage(type: .heartRateData, data: [
            "heartRate": healthData.heartRate,
            "steps": healthData.steps,
            "activeEnergy": healthData.activeEnergy,
            "timestamp": Date().timeIntervalSince1970
        ])
        
        sendMessage(message)
    }
    
    // MARK: - Settings
    
    func updateAutoSync(_ enabled: Bool) {
        autoSyncEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "watch_autoSync")
        
        if enabled {
            setupPeriodicSync()
        } else {
            syncTimer?.invalidate()
            syncTimer = nil
        }
    }
    
    func updateBackgroundSync(_ enabled: Bool) {
        backgroundSyncEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "watch_backgroundSync")
    }
    
    func updateEmergencyFeatures(_ enabled: Bool) {
        emergencyFeaturesEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "watch_emergencyFeatures")
    }
    
    // MARK: - Utility
    
    func forceSync() {
        syncWithWatch()
    }
    
    func clearMessageQueue() {
        messageQueue.removeAll()
    }
    
    func getConnectionInfo() -> [String: Any] {
        return [
            "isWatchAppInstalled": isWatchAppInstalled,
            "isReachable": isWatchReachable,
            "lastSync": lastSyncTime?.timeIntervalSince1970 ?? 0,
            "pendingMessages": messageQueue.count,
            "batteryLevel": watchBatteryLevel
        ]
    }
}

// MARK: - WCSessionDelegate

extension AppleWatchManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            switch activationState {
            case .activated:
                self.connectionStatus = .connected
                self.isWatchAppInstalled = session.isWatchAppInstalled
                self.isWatchReachable = session.isReachable
                
                // Sync immediately after activation
                if self.autoSyncEnabled {
                    self.syncWithWatch()
                }
                
            case .inactive:
                self.connectionStatus = .inactive
                
            case .notActivated:
                self.connectionStatus = .disconnected
                
            @unknown default:
                self.connectionStatus = .disconnected
            }
        }
        
        if let error = error {
            print("Watch session activation error: \(error)")
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.connectionStatus = .inactive
        }
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        DispatchQueue.main.async {
            self.connectionStatus = .disconnected
        }
        
        // Reactivate session
        session.activate()
    }
    
    func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isWatchReachable = session.isReachable
            
            if session.isReachable && self.autoSyncEnabled {
                self.syncWithWatch()
            }
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any], replyHandler: @escaping ([String : Any]) -> Void) {
        handleReceivedMessage(message, replyHandler: replyHandler)
    }
    
    func session(_ session: WCSession, didReceiveUserInfo userInfo: [String : Any]) {
        handleReceivedMessage(userInfo)
    }
    
    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String : Any]) {
        // Handle application context updates from watch
        print("Received application context from watch")
    }
    
    private func handleReceivedMessage(_ message: [String: Any], replyHandler: (([String: Any]) -> Void)? = nil) {
        guard let messageData = message["messageData"] as? Data,
              let watchMessage = try? JSONDecoder().decode(WatchMessage.self, from: messageData) else {
            replyHandler?(["status": "error", "message": "Invalid message format"])
            return
        }
        
        DispatchQueue.main.async {
            self.processWatchMessage(watchMessage)
        }
        
        replyHandler?(["status": "received", "messageId": watchMessage.messageId])
    }
    
    private func processWatchMessage(_ message: WatchMessage) {
        guard let messageType = WatchMessageType(rawValue: message.type) else {
            print("Unknown message type: \(message.type)")
            return
        }
        
        switch messageType {
        case .painLevel:
            handlePainLevelFromWatch(message)
        case .medicationTaken:
            handleMedicationTakenFromWatch(message)
        case .moodUpdate:
            handleMoodUpdateFromWatch(message)
        case .emergencyAlert:
            handleEmergencyAlertFromWatch(message)
        case .reminderResponse:
            handleReminderResponseFromWatch(message)
        case .heartRateData, .sleepData, .stepsData, .workoutData:
            handleHealthDataFromWatch(message)
        default:
            print("Unhandled message type: \(messageType)")
        }
    }
    
    private func handlePainLevelFromWatch(_ message: WatchMessage) {
        guard let level = message.data["level"] as? Int,
              let location = message.data["location"] as? String else { return }
        
        let notes = message.data["notes"] as? String
        
        // Save pain entry to Core Data
        let painEntry = PainEntry(context: context)
        painEntry.id = UUID()
        painEntry.painLevel = Int16(level)
        painEntry.location = location
        painEntry.notes = notes
        painEntry.timestamp = message.timestamp
        
        do {
            try context.save()
            print("Pain level saved from watch: \(level) at \(location)")
        } catch {
            print("Error saving pain level from watch: \(error)")
        }
    }
    
    private func handleMedicationTakenFromWatch(_ message: WatchMessage) {
        guard let medicationId = message.data["medicationId"] as? String else { return }
        
        // Save medication log to Core Data
        let medicationLog = MedicationLog(context: context)
        medicationLog.id = UUID()
        medicationLog.medicationId = UUID(uuidString: medicationId)
        medicationLog.timestamp = message.timestamp
        medicationLog.taken = true
        
        do {
            try context.save()
            print("Medication taken logged from watch: \(medicationId)")
        } catch {
            print("Error saving medication log from watch: \(error)")
        }
    }
    
    private func handleMoodUpdateFromWatch(_ message: WatchMessage) {
        guard let mood = message.data["mood"] as? String,
              let energy = message.data["energy"] as? Int else { return }
        
        let notes = message.data["notes"] as? String
        
        // Save journal entry to Core Data
        let journalEntry = JournalEntry(context: context)
        journalEntry.id = UUID()
        journalEntry.mood = mood
        journalEntry.energyLevel = Int16(energy)
        journalEntry.notes = notes
        journalEntry.timestamp = message.timestamp
        
        do {
            try context.save()
            print("Mood update saved from watch: \(mood), energy: \(energy)")
        } catch {
            print("Error saving mood update from watch: \(error)")
        }
    }
    
    private func handleEmergencyAlertFromWatch(_ message: WatchMessage) {
        guard let alertType = message.data["alertType"] as? String else { return }
        
        // Handle emergency alert
        print("Emergency alert from watch: \(alertType)")
        
        // Trigger emergency protocols
        NotificationCenter.default.post(name: .emergencyAlertReceived, object: message.data)
    }
    
    private func handleReminderResponseFromWatch(_ message: WatchMessage) {
        guard let reminderId = message.data["reminderId"] as? String,
              let completed = message.data["completed"] as? Bool else { return }
        
        print("Reminder response from watch: \(reminderId), completed: \(completed)")
        
        // Update reminder status
        NotificationCenter.default.post(name: .reminderResponseReceived, object: [
            "reminderId": reminderId,
            "completed": completed
        ])
    }
    
    private func handleHealthDataFromWatch(_ message: WatchMessage) {
        // Process health data from watch
        DispatchQueue.main.async {
            self.watchHealthData.updateFromMessage(message)
        }
    }
}

// MARK: - Supporting Types

enum WatchConnectionStatus {
    case connected
    case disconnected
    case inactive
    case connecting
}

enum MessagePriority {
    case low
    case normal
    case high
    case critical
}

struct WatchHealthData {
    var heartRate: Double = 0
    var steps: Int = 0
    var activeEnergy: Double = 0
    var sleepHours: Double = 0
    var workoutMinutes: Int = 0
    var lastUpdate: Date = Date()
    
    mutating func updateFromMessage(_ message: WatchMessage) {
        if let heartRate = message.data["heartRate"] as? Double {
            self.heartRate = heartRate
        }
        if let steps = message.data["steps"] as? Int {
            self.steps = steps
        }
        if let activeEnergy = message.data["activeEnergy"] as? Double {
            self.activeEnergy = activeEnergy
        }
        if let sleepHours = message.data["sleepHours"] as? Double {
            self.sleepHours = sleepHours
        }
        if let workoutMinutes = message.data["workoutMinutes"] as? Int {
            self.workoutMinutes = workoutMinutes
        }
        
        lastUpdate = Date()
    }
}

// MARK: - Notification Extensions

extension Notification.Name {
    static let emergencyAlertReceived = Notification.Name("emergencyAlertReceived")
    static let reminderResponseReceived = Notification.Name("reminderResponseReceived")
    static let watchDataSynced = Notification.Name("watchDataSynced")
}