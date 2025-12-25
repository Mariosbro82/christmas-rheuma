//
//  EmergencyResponseSystem.swift
//  InflamAI-Swift
//
//  Emergency response system for crisis situations and urgent health alerts
//

import Foundation
import Combine
import CoreLocation
import UserNotifications
import HealthKit
import CallKit
import MessageUI
import ContactsUI
import AVFoundation
import CoreHaptics
import WatchConnectivity
import CloudKit

// MARK: - Emergency Models

struct EmergencyContact: Codable, Identifiable {
    let id: String
    let name: String
    let relationship: ContactRelationship
    let phoneNumber: String
    let email: String?
    let address: String?
    let isPrimary: Bool
    let isHealthcareProvider: Bool
    let specialization: String?
    let availabilityHours: AvailabilityHours?
    let preferredContactMethod: ContactMethod
    let emergencyOnly: Bool
    let notes: String?
    let lastContacted: Date?
    let responseTime: TimeInterval?
    let isVerified: Bool
    let photoURL: String?
    let languages: [String]
    let timeZone: String?
}

enum ContactRelationship: String, Codable, CaseIterable {
    case spouse = "spouse"
    case partner = "partner"
    case parent = "parent"
    case child = "child"
    case sibling = "sibling"
    case friend = "friend"
    case caregiver = "caregiver"
    case doctor = "doctor"
    case nurse = "nurse"
    case therapist = "therapist"
    case pharmacist = "pharmacist"
    case emergencyServices = "emergency_services"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .spouse: return "Spouse"
        case .partner: return "Partner"
        case .parent: return "Parent"
        case .child: return "Child"
        case .sibling: return "Sibling"
        case .friend: return "Friend"
        case .caregiver: return "Caregiver"
        case .doctor: return "Doctor"
        case .nurse: return "Nurse"
        case .therapist: return "Therapist"
        case .pharmacist: return "Pharmacist"
        case .emergencyServices: return "Emergency Services"
        case .other: return "Other"
        }
    }
    
    var icon: String {
        switch self {
        case .spouse, .partner: return "heart.circle"
        case .parent: return "person.2.circle"
        case .child: return "figure.child.circle"
        case .sibling: return "person.3.circle"
        case .friend: return "person.circle"
        case .caregiver: return "hands.sparkles"
        case .doctor: return "stethoscope.circle"
        case .nurse: return "cross.circle"
        case .therapist: return "brain.head.profile"
        case .pharmacist: return "pills.circle"
        case .emergencyServices: return "phone.circle"
        case .other: return "person.crop.circle"
        }
    }
}

enum ContactMethod: String, Codable {
    case phone = "phone"
    case sms = "sms"
    case email = "email"
    case app = "app"
    case any = "any"
    
    var displayName: String {
        switch self {
        case .phone: return "Phone Call"
        case .sms: return "Text Message"
        case .email: return "Email"
        case .app: return "App Notification"
        case .any: return "Any Method"
        }
    }
}

struct AvailabilityHours: Codable {
    let monday: DayAvailability?
    let tuesday: DayAvailability?
    let wednesday: DayAvailability?
    let thursday: DayAvailability?
    let friday: DayAvailability?
    let saturday: DayAvailability?
    let sunday: DayAvailability?
    let timeZone: String
    let emergencyOverride: Bool // Available 24/7 for emergencies
}

struct DayAvailability: Codable {
    let startTime: String // HH:mm format
    let endTime: String
    let isAvailable: Bool
    let breaks: [TimeBreak]?
}

struct TimeBreak: Codable {
    let startTime: String
    let endTime: String
    let reason: String?
}

struct EmergencyAlert: Codable, Identifiable {
    let id: String
    let type: EmergencyType
    let severity: EmergencySeverity
    let title: String
    let message: String
    let location: EmergencyLocation?
    let healthData: EmergencyHealthData?
    let triggeredAt: Date
    let resolvedAt: Date?
    let status: EmergencyStatus
    let contactsNotified: [String] // Contact IDs
    let responseReceived: [EmergencyResponse]
    let escalationLevel: Int
    let autoTriggered: Bool
    let userConfirmed: Bool
    let falseAlarm: Bool
    let notes: String?
    let attachments: [EmergencyAttachment]
    let followUpRequired: Bool
    let followUpDate: Date?
}

enum EmergencyType: String, Codable, CaseIterable {
    case medicalEmergency = "medical_emergency"
    case severeFlare = "severe_flare"
    case medicationReaction = "medication_reaction"
    case fall = "fall"
    case mentalHealthCrisis = "mental_health_crisis"
    case painCrisis = "pain_crisis"
    case breathingDifficulty = "breathing_difficulty"
    case heartIssue = "heart_issue"
    case seizure = "seizure"
    case unconsciousness = "unconsciousness"
    case severeInjury = "severe_injury"
    case allergicReaction = "allergic_reaction"
    case overdose = "overdose"
    case stroke = "stroke"
    case other = "other"
    
    var displayName: String {
        switch self {
        case .medicalEmergency: return "Medical Emergency"
        case .severeFlare: return "Severe Flare-Up"
        case .medicationReaction: return "Medication Reaction"
        case .fall: return "Fall or Injury"
        case .mentalHealthCrisis: return "Mental Health Crisis"
        case .painCrisis: return "Severe Pain Crisis"
        case .breathingDifficulty: return "Breathing Difficulty"
        case .heartIssue: return "Heart Problem"
        case .seizure: return "Seizure"
        case .unconsciousness: return "Loss of Consciousness"
        case .severeInjury: return "Severe Injury"
        case .allergicReaction: return "Allergic Reaction"
        case .overdose: return "Overdose"
        case .stroke: return "Stroke"
        case .other: return "Other Emergency"
        }
    }
    
    var icon: String {
        switch self {
        case .medicalEmergency: return "cross.circle.fill"
        case .severeFlare: return "flame.circle.fill"
        case .medicationReaction: return "pills.circle.fill"
        case .fall: return "figure.fall.circle.fill"
        case .mentalHealthCrisis: return "brain.head.profile"
        case .painCrisis: return "bolt.circle.fill"
        case .breathingDifficulty: return "lungs.fill"
        case .heartIssue: return "heart.circle.fill"
        case .seizure: return "waveform.circle.fill"
        case .unconsciousness: return "person.circle.fill"
        case .severeInjury: return "bandage.fill"
        case .allergicReaction: return "allergens.fill"
        case .overdose: return "exclamationmark.triangle.fill"
        case .stroke: return "brain.fill"
        case .other: return "exclamationmark.circle.fill"
        }
    }
    
    var color: String {
        switch self {
        case .medicalEmergency, .unconsciousness, .stroke: return "red"
        case .severeFlare, .painCrisis: return "orange"
        case .medicationReaction, .allergicReaction, .overdose: return "purple"
        case .fall, .severeInjury: return "brown"
        case .mentalHealthCrisis: return "blue"
        case .breathingDifficulty, .heartIssue: return "red"
        case .seizure: return "yellow"
        case .other: return "gray"
        }
    }
    
    var requiresImmediateResponse: Bool {
        switch self {
        case .medicalEmergency, .unconsciousness, .stroke, .heartIssue, .seizure, .breathingDifficulty, .allergicReaction, .overdose:
            return true
        default:
            return false
        }
    }
}

enum EmergencySeverity: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
    
    var displayName: String {
        switch self {
        case .low: return "Low Priority"
        case .medium: return "Medium Priority"
        case .high: return "High Priority"
        case .critical: return "Critical - Call 911"
        }
    }
    
    var color: String {
        switch self {
        case .low: return "green"
        case .medium: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }
    
    var shouldCallEmergencyServices: Bool {
        return self == .critical
    }
}

enum EmergencyStatus: String, Codable {
    case active = "active"
    case acknowledged = "acknowledged"
    case responding = "responding"
    case resolved = "resolved"
    case falseAlarm = "false_alarm"
    case escalated = "escalated"
}

struct EmergencyLocation: Codable {
    let latitude: Double
    let longitude: Double
    let altitude: Double?
    let accuracy: Double
    let timestamp: Date
    let address: String?
    let landmark: String?
    let floor: String?
    let room: String?
    let additionalInfo: String?
}

struct EmergencyHealthData: Codable {
    let heartRate: Double?
    let bloodPressure: BloodPressureReading?
    let oxygenSaturation: Double?
    let temperature: Double?
    let painLevel: Int?
    let symptoms: [String]
    let medications: [String]
    let allergies: [String]
    let medicalConditions: [String]
    let recentChanges: String?
    let consciousness: ConsciousnessLevel
    let mobility: MobilityStatus
    let breathing: BreathingStatus
}

struct BloodPressureReading: Codable {
    let systolic: Double
    let diastolic: Double
    let timestamp: Date
}

enum ConsciousnessLevel: String, Codable {
    case alert = "alert"
    case drowsy = "drowsy"
    case confused = "confused"
    case unconscious = "unconscious"
    case unknown = "unknown"
}

enum MobilityStatus: String, Codable {
    case normal = "normal"
    case limited = "limited"
    case immobile = "immobile"
    case fallen = "fallen"
    case unknown = "unknown"
}

enum BreathingStatus: String, Codable {
    case normal = "normal"
    case labored = "labored"
    case shallow = "shallow"
    case stopped = "stopped"
    case unknown = "unknown"
}

struct EmergencyResponse: Codable, Identifiable {
    let id: String
    let alertId: String
    let contactId: String
    let contactName: String
    let responseType: ResponseType
    let message: String?
    let estimatedArrival: Date?
    let actualArrival: Date?
    let responseTime: TimeInterval
    let timestamp: Date
    let location: EmergencyLocation?
    let status: ResponseStatus
}

enum ResponseType: String, Codable {
    case acknowledged = "acknowledged"
    case onTheWay = "on_the_way"
    case arrived = "arrived"
    case cannotRespond = "cannot_respond"
    case falseAlarm = "false_alarm"
    case needMoreInfo = "need_more_info"
    case callingEmergencyServices = "calling_emergency_services"
}

enum ResponseStatus: String, Codable {
    case pending = "pending"
    case active = "active"
    case completed = "completed"
    case cancelled = "cancelled"
}

struct EmergencyAttachment: Codable, Identifiable {
    let id: String
    let type: AttachmentType
    let url: String
    let filename: String?
    let fileSize: Int64?
    let mimeType: String?
    let timestamp: Date
    let description: String?
}

enum AttachmentType: String, Codable {
    case photo = "photo"
    case video = "video"
    case audio = "audio"
    case document = "document"
    case healthData = "health_data"
}

// MARK: - Emergency Settings

struct EmergencySettings: Codable {
    let isEnabled: Bool
    let autoDetectionEnabled: Bool
    let fallDetectionEnabled: Bool
    let heartRateMonitoringEnabled: Bool
    let locationSharingEnabled: Bool
    let emergencyContactsEnabled: Bool
    let emergencyServicesEnabled: Bool
    let watchIntegrationEnabled: Bool
    let voiceActivationEnabled: Bool
    let countdownDuration: TimeInterval // Seconds before auto-calling
    let escalationDelay: TimeInterval // Time before escalating
    let maxEscalationLevel: Int
    let quietHours: QuietHours?
    let medicalInfo: MedicalInformation
    let preferences: EmergencyPreferences
}

struct QuietHours: Codable {
    let startTime: String // HH:mm format
    let endTime: String
    let isEnabled: Bool
    let allowCriticalAlerts: Bool
    let reducedNotifications: Bool
}

struct MedicalInformation: Codable {
    let bloodType: String?
    let allergies: [String]
    let medications: [String]
    let medicalConditions: [String]
    let emergencyNotes: String?
    let organDonor: Bool?
    let emergencyContact: String?
    let preferredHospital: String?
    let insurance: InsuranceInfo?
    let lastUpdated: Date
}

struct InsuranceInfo: Codable {
    let provider: String
    let policyNumber: String
    let groupNumber: String?
    let memberID: String?
    let phoneNumber: String?
}

struct EmergencyPreferences: Codable {
    let preferredLanguage: String
    let accessibilityNeeds: [String]
    let communicationPreferences: [ContactMethod]
    let privacyLevel: PrivacyLevel
    let shareHealthData: Bool
    let shareLocation: Bool
    let allowThirdPartyAccess: Bool
    let autoShareWithEmergencyServices: Bool
}

enum PrivacyLevel: String, Codable {
    case minimal = "minimal"
    case standard = "standard"
    case detailed = "detailed"
    case full = "full"
}

// MARK: - Emergency Response System Manager

class EmergencyResponseSystem: NSObject, ObservableObject {
    // Core Services
    private let healthStore = HKHealthStore()
    private let locationManager = CLLocationManager()
    private let notificationCenter = UNUserNotificationCenter.current()
    
    // Audio and Haptics
    private var audioPlayer: AVAudioPlayer?
    private var hapticEngine: CHHapticEngine?
    
    // Watch Connectivity
    private let watchSession = WCSession.default
    
    // CloudKit
    private let cloudKitContainer = CKContainer.default()
    
    // Published Properties
    @Published var emergencyContacts: [EmergencyContact] = []
    @Published var activeAlerts: [EmergencyAlert] = []
    @Published var emergencyHistory: [EmergencyAlert] = []
    @Published var settings: EmergencySettings
    @Published var currentLocation: EmergencyLocation?
    @Published var isEmergencyMode = false
    @Published var lastHeartRate: Double?
    @Published var connectionStatus: ConnectionStatus = .unknown
    
    // Internal State
    private var cancellables = Set<AnyCancellable>()
    private var heartRateQuery: HKAnchoredObjectQuery?
    private var fallDetectionTimer: Timer?
    private var emergencyCountdownTimer: Timer?
    private var currentCountdown: Int = 0
    
    // Emergency Detection
    private var lastKnownLocation: CLLocation?
    private var accelerometerData: [Double] = []
    private var isMonitoringFalls = false
    private var consecutiveHighHeartRate = 0
    private var lastMovementTime = Date()
    
    override init() {
        self.settings = EmergencySettings(
            isEnabled: true,
            autoDetectionEnabled: true,
            fallDetectionEnabled: true,
            heartRateMonitoringEnabled: true,
            locationSharingEnabled: true,
            emergencyContactsEnabled: true,
            emergencyServicesEnabled: true,
            watchIntegrationEnabled: true,
            voiceActivationEnabled: true,
            countdownDuration: 30,
            escalationDelay: 300, // 5 minutes
            maxEscalationLevel: 3,
            quietHours: nil,
            medicalInfo: MedicalInformation(
                bloodType: nil,
                allergies: [],
                medications: [],
                medicalConditions: [],
                emergencyNotes: nil,
                organDonor: nil,
                emergencyContact: nil,
                preferredHospital: nil,
                insurance: nil,
                lastUpdated: Date()
            ),
            preferences: EmergencyPreferences(
                preferredLanguage: "en",
                accessibilityNeeds: [],
                communicationPreferences: [.phone, .sms],
                privacyLevel: .standard,
                shareHealthData: true,
                shareLocation: true,
                allowThirdPartyAccess: false,
                autoShareWithEmergencyServices: true
            )
        )
        
        super.init()
        
        setupLocationManager()
        setupHealthKit()
        setupHaptics()
        setupWatchConnectivity()
        setupNotifications()
        loadEmergencyContacts()
        loadEmergencyHistory()
        
        if settings.autoDetectionEnabled {
            startEmergencyMonitoring()
        }
    }
    
    // MARK: - Setup
    
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.requestWhenInUseAuthorization()
        locationManager.requestAlwaysAuthorization()
    }
    
    private func setupHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!,
            HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.workoutType()
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            if success {
                self?.startHealthMonitoring()
            }
        }
    }
    
    private func setupHaptics() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        
        do {
            hapticEngine = try CHHapticEngine()
            try hapticEngine?.start()
        } catch {
            print("Haptic engine error: \(error)")
        }
    }
    
    private func setupWatchConnectivity() {
        if WCSession.isSupported() {
            watchSession.delegate = self
            watchSession.activate()
        }
    }
    
    private func setupNotifications() {
        notificationCenter.requestAuthorization(options: [.alert, .sound, .badge, .criticalAlert]) { granted, error in
            if let error = error {
                print("Notification authorization error: \(error)")
            }
        }
    }
    
    // MARK: - Emergency Triggering
    
    func triggerEmergency(type: EmergencyType, severity: EmergencySeverity, message: String? = nil, autoTriggered: Bool = false) async {
        let alert = EmergencyAlert(
            id: UUID().uuidString,
            type: type,
            severity: severity,
            title: type.displayName,
            message: message ?? "Emergency assistance needed",
            location: currentLocation,
            healthData: await collectCurrentHealthData(),
            triggeredAt: Date(),
            resolvedAt: nil,
            status: .active,
            contactsNotified: [],
            responseReceived: [],
            escalationLevel: 0,
            autoTriggered: autoTriggered,
            userConfirmed: !autoTriggered,
            falseAlarm: false,
            notes: nil,
            attachments: [],
            followUpRequired: true,
            followUpDate: Calendar.current.date(byAdding: .hour, value: 24, to: Date())
        )
        
        DispatchQueue.main.async {
            self.activeAlerts.append(alert)
            self.isEmergencyMode = true
        }
        
        // Save to CloudKit
        await saveEmergencyAlert(alert)
        
        // Start emergency response protocol
        await executeEmergencyProtocol(for: alert)
    }
    
    func triggerEmergencyWithCountdown(type: EmergencyType, severity: EmergencySeverity, message: String? = nil) {
        guard !isEmergencyMode else { return }
        
        currentCountdown = Int(settings.countdownDuration)
        
        // Show countdown UI
        DispatchQueue.main.async {
            self.isEmergencyMode = true
        }
        
        // Start countdown timer
        emergencyCountdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            
            self.currentCountdown -= 1
            
            if self.currentCountdown <= 0 {
                timer.invalidate()
                Task {
                    await self.triggerEmergency(type: type, severity: severity, message: message, autoTriggered: true)
                }
            } else {
                // Play countdown sound and haptic
                self.playCountdownFeedback()
            }
        }
    }
    
    func cancelEmergencyCountdown() {
        emergencyCountdownTimer?.invalidate()
        emergencyCountdownTimer = nil
        currentCountdown = 0
        
        DispatchQueue.main.async {
            self.isEmergencyMode = false
        }
    }
    
    // MARK: - Emergency Protocol Execution
    
    private func executeEmergencyProtocol(for alert: EmergencyAlert) async {
        // 1. Update location
        await updateCurrentLocation()
        
        // 2. Play emergency sounds and haptics
        playEmergencyAlert(for: alert.severity)
        
        // 3. Send to Apple Watch
        sendEmergencyToWatch(alert)
        
        // 4. Notify emergency contacts
        await notifyEmergencyContacts(for: alert)
        
        // 5. If critical, prepare to call emergency services
        if alert.severity.shouldCallEmergencyServices {
            await prepareEmergencyServicesCall(for: alert)
        }
        
        // 6. Start escalation timer
        startEscalationTimer(for: alert)
        
        // 7. Send push notifications
        await sendEmergencyNotifications(for: alert)
        
        // 8. Log emergency event
        logEmergencyEvent(alert)
    }
    
    private func notifyEmergencyContacts(for alert: EmergencyAlert) async {
        let priorityContacts = emergencyContacts
            .filter { $0.isPrimary || !$0.emergencyOnly }
            .sorted { $0.isPrimary && !$1.isPrimary }
        
        for contact in priorityContacts {
            await sendEmergencyMessage(to: contact, for: alert)
            
            // Add small delay between contacts to avoid overwhelming
            try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        }
    }
    
    private func sendEmergencyMessage(to contact: EmergencyContact, for alert: EmergencyAlert) async {
        let message = generateEmergencyMessage(for: alert, contact: contact)
        
        switch contact.preferredContactMethod {
        case .phone:
            await makeEmergencyCall(to: contact, for: alert)
        case .sms:
            await sendEmergencySMS(to: contact, message: message, alert: alert)
        case .email:
            await sendEmergencyEmail(to: contact, message: message, alert: alert)
        case .app:
            await sendEmergencyAppNotification(to: contact, for: alert)
        case .any:
            // Try multiple methods
            await sendEmergencySMS(to: contact, message: message, alert: alert)
            await sendEmergencyAppNotification(to: contact, for: alert)
        }
    }
    
    // MARK: - Health Monitoring
    
    private func startHealthMonitoring() {
        if settings.heartRateMonitoringEnabled {
            startHeartRateMonitoring()
        }
    }
    
    private func startHeartRateMonitoring() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: nil,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            for sample in samples {
                let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                
                DispatchQueue.main.async {
                    self?.lastHeartRate = heartRate
                }
                
                self?.analyzeHeartRate(heartRate)
            }
        }
        
        query.updateHandler = { [weak self] query, samples, deletedObjects, anchor, error in
            guard let samples = samples as? [HKQuantitySample] else { return }
            
            for sample in samples {
                let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                
                DispatchQueue.main.async {
                    self?.lastHeartRate = heartRate
                }
                
                self?.analyzeHeartRate(heartRate)
            }
        }
        
        healthStore.execute(query)
        heartRateQuery = query
    }
    
    private func analyzeHeartRate(_ heartRate: Double) {
        // Detect abnormal heart rate patterns
        if heartRate > 150 || heartRate < 40 {
            consecutiveHighHeartRate += 1
            
            if consecutiveHighHeartRate >= 3 {
                Task {
                    await self.triggerEmergency(
                        type: .heartIssue,
                        severity: .high,
                        message: "Abnormal heart rate detected: \(Int(heartRate)) BPM",
                        autoTriggered: true
                    )
                }
            }
        } else {
            consecutiveHighHeartRate = 0
        }
    }
    
    private func startEmergencyMonitoring() {
        if settings.fallDetectionEnabled {
            startFallDetection()
        }
    }
    
    private func startFallDetection() {
        // Implement fall detection using accelerometer data
        isMonitoringFalls = true
    }
    
    // MARK: - Communication Methods
    
    private func makeEmergencyCall(to contact: EmergencyContact, for alert: EmergencyAlert) async {
        guard let url = URL(string: "tel://\(contact.phoneNumber)") else { return }
        
        DispatchQueue.main.async {
            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url)
            }
        }
    }
    
    private func sendEmergencySMS(to contact: EmergencyContact, message: String, alert: EmergencyAlert) async {
        // Implement SMS sending
    }
    
    private func sendEmergencyEmail(to contact: EmergencyContact, message: String, alert: EmergencyAlert) async {
        // Implement email sending
    }
    
    private func sendEmergencyAppNotification(to contact: EmergencyContact, for alert: EmergencyAlert) async {
        // Send push notification through app
    }
    
    private func prepareEmergencyServicesCall(for alert: EmergencyAlert) async {
        // Prepare emergency services call with location and health data
        let emergencyNumber = getEmergencyNumber()
        
        DispatchQueue.main.async {
            if let url = URL(string: "tel://\(emergencyNumber)") {
                UIApplication.shared.open(url)
            }
        }
    }
    
    private func getEmergencyNumber() -> String {
        // Return appropriate emergency number based on location
        return "911" // Default for US
    }
    
    // MARK: - Audio and Haptic Feedback
    
    private func playEmergencyAlert(for severity: EmergencySeverity) {
        playEmergencySound(for: severity)
        playEmergencyHaptic(for: severity)
    }
    
    private func playEmergencySound(for severity: EmergencySeverity) {
        let soundName: String
        
        switch severity {
        case .low:
            soundName = "emergency_low"
        case .medium:
            soundName = "emergency_medium"
        case .high:
            soundName = "emergency_high"
        case .critical:
            soundName = "emergency_critical"
        }
        
        guard let url = Bundle.main.url(forResource: soundName, withExtension: "wav") else { return }
        
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.numberOfLoops = severity == .critical ? -1 : 3
            audioPlayer?.volume = 1.0
            audioPlayer?.play()
        } catch {
            print("Error playing emergency sound: \(error)")
        }
    }
    
    private func playEmergencyHaptic(for severity: EmergencySeverity) {
        guard let hapticEngine = hapticEngine else { return }
        
        let intensity: Float
        let sharpness: Float
        let duration: TimeInterval
        
        switch severity {
        case .low:
            intensity = 0.5
            sharpness = 0.5
            duration = 0.5
        case .medium:
            intensity = 0.7
            sharpness = 0.7
            duration = 1.0
        case .high:
            intensity = 0.9
            sharpness = 0.9
            duration = 1.5
        case .critical:
            intensity = 1.0
            sharpness = 1.0
            duration = 2.0
        }
        
        do {
            let hapticEvent = CHHapticEvent(
                eventType: .hapticTransient,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness)
                ],
                relativeTime: 0,
                duration: duration
            )
            
            let pattern = try CHHapticPattern(events: [hapticEvent], parameters: [])
            let player = try hapticEngine.makePlayer(with: pattern)
            try player.start(atTime: 0)
        } catch {
            print("Error playing emergency haptic: \(error)")
        }
    }
    
    private func playCountdownFeedback() {
        // Play countdown sound and haptic
        AudioServicesPlaySystemSound(1057) // Tock sound
        
        let impactFeedback = UIImpactFeedbackGenerator(style: .heavy)
        impactFeedback.impactOccurred()
    }
    
    // MARK: - Watch Integration
    
    private func sendEmergencyToWatch(_ alert: EmergencyAlert) {
        guard watchSession.isReachable else { return }
        
        let message: [String: Any] = [
            "type": "emergency_alert",
            "alertId": alert.id,
            "emergencyType": alert.type.rawValue,
            "severity": alert.severity.rawValue,
            "message": alert.message,
            "timestamp": alert.triggeredAt.timeIntervalSince1970
        ]
        
        watchSession.sendMessage(message, replyHandler: nil) { error in
            print("Error sending emergency to watch: \(error)")
        }
    }
    
    // MARK: - Utility Methods
    
    private func updateCurrentLocation() async {
        locationManager.requestLocation()
    }
    
    private func collectCurrentHealthData() async -> EmergencyHealthData {
        return EmergencyHealthData(
            heartRate: lastHeartRate,
            bloodPressure: nil,
            oxygenSaturation: nil,
            temperature: nil,
            painLevel: nil,
            symptoms: [],
            medications: settings.medicalInfo.medications,
            allergies: settings.medicalInfo.allergies,
            medicalConditions: settings.medicalInfo.medicalConditions,
            recentChanges: nil,
            consciousness: .alert,
            mobility: .normal,
            breathing: .normal
        )
    }
    
    private func generateEmergencyMessage(for alert: EmergencyAlert, contact: EmergencyContact) -> String {
        var message = "🚨 EMERGENCY ALERT 🚨\n\n"
        message += "Type: \(alert.type.displayName)\n"
        message += "Severity: \(alert.severity.displayName)\n"
        message += "Time: \(DateFormatter.localizedString(from: alert.triggeredAt, dateStyle: .short, timeStyle: .medium))\n\n"
        message += "Message: \(alert.message)\n\n"
        
        if let location = alert.location {
            message += "Location: \(location.latitude), \(location.longitude)\n"
            if let address = location.address {
                message += "Address: \(address)\n"
            }
        }
        
        if let healthData = alert.healthData {
            message += "\nHealth Info:\n"
            if let heartRate = healthData.heartRate {
                message += "Heart Rate: \(Int(heartRate)) BPM\n"
            }
            if let painLevel = healthData.painLevel {
                message += "Pain Level: \(painLevel)/10\n"
            }
            if !healthData.symptoms.isEmpty {
                message += "Symptoms: \(healthData.symptoms.joined(separator: ", "))\n"
            }
        }
        
        message += "\nThis is an automated emergency alert from InflamAI."
        
        return message
    }
    
    private func startEscalationTimer(for alert: EmergencyAlert) {
        Timer.scheduledTimer(withTimeInterval: settings.escalationDelay, repeats: false) { [weak self] _ in
            Task {
                await self?.escalateEmergency(alert)
            }
        }
    }
    
    private func escalateEmergency(_ alert: EmergencyAlert) async {
        // Escalate emergency if no response received
        if alert.responseReceived.isEmpty && alert.escalationLevel < settings.maxEscalationLevel {
            // Notify additional contacts or emergency services
        }
    }
    
    private func sendEmergencyNotifications(for alert: EmergencyAlert) async {
        let content = UNMutableNotificationContent()
        content.title = "🚨 Emergency Alert"
        content.body = alert.message
        content.sound = .defaultCritical
        content.categoryIdentifier = "EMERGENCY_ALERT"
        
        let request = UNNotificationRequest(
            identifier: alert.id,
            content: content,
            trigger: nil
        )
        
        try? await notificationCenter.add(request)
    }
    
    private func logEmergencyEvent(_ alert: EmergencyAlert) {
        // Log emergency event for analytics and improvement
    }
    
    private func saveEmergencyAlert(_ alert: EmergencyAlert) async {
        // Save to CloudKit and local storage
    }
    
    private func loadEmergencyContacts() {
        // Load emergency contacts from storage
    }
    
    private func loadEmergencyHistory() {
        // Load emergency history from storage
    }
    
    // MARK: - Public Methods
    
    func addEmergencyContact(_ contact: EmergencyContact) {
        emergencyContacts.append(contact)
        // Save to storage
    }
    
    func removeEmergencyContact(_ contactId: String) {
        emergencyContacts.removeAll { $0.id == contactId }
        // Update storage
    }
    
    func updateEmergencySettings(_ newSettings: EmergencySettings) {
        settings = newSettings
        // Save to storage
        
        if newSettings.autoDetectionEnabled {
            startEmergencyMonitoring()
        }
    }
    
    func resolveEmergency(_ alertId: String, falseAlarm: Bool = false) {
        if let index = activeAlerts.firstIndex(where: { $0.id == alertId }) {
            var alert = activeAlerts[index]
            alert = EmergencyAlert(
                id: alert.id,
                type: alert.type,
                severity: alert.severity,
                title: alert.title,
                message: alert.message,
                location: alert.location,
                healthData: alert.healthData,
                triggeredAt: alert.triggeredAt,
                resolvedAt: Date(),
                status: falseAlarm ? .falseAlarm : .resolved,
                contactsNotified: alert.contactsNotified,
                responseReceived: alert.responseReceived,
                escalationLevel: alert.escalationLevel,
                autoTriggered: alert.autoTriggered,
                userConfirmed: alert.userConfirmed,
                falseAlarm: falseAlarm,
                notes: alert.notes,
                attachments: alert.attachments,
                followUpRequired: alert.followUpRequired,
                followUpDate: alert.followUpDate
            )
            
            activeAlerts.remove(at: index)
            emergencyHistory.insert(alert, at: 0)
            
            if activeAlerts.isEmpty {
                isEmergencyMode = false
            }
            
            // Stop emergency sounds
            audioPlayer?.stop()
        }
    }
}

// MARK: - Location Manager Delegate

extension EmergencyResponseSystem: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        
        lastKnownLocation = location
        
        let emergencyLocation = EmergencyLocation(
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude,
            altitude: location.altitude,
            accuracy: location.horizontalAccuracy,
            timestamp: location.timestamp,
            address: nil,
            landmark: nil,
            floor: nil,
            room: nil,
            additionalInfo: nil
        )
        
        DispatchQueue.main.async {
            self.currentLocation = emergencyLocation
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
    }
}

// MARK: - Watch Connectivity Delegate

extension EmergencyResponseSystem: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.connectionStatus = activationState == .activated ? .connected : .disconnected
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.connectionStatus = .disconnected
        }
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        DispatchQueue.main.async {
            self.connectionStatus = .disconnected
        }
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Handle messages from Apple Watch
        if let type = message["type"] as? String {
            switch type {
            case "emergency_trigger":
                if let emergencyType = message["emergencyType"] as? String,
                   let severity = message["severity"] as? String {
                    Task {
                        await triggerEmergency(
                            type: EmergencyType(rawValue: emergencyType) ?? .other,
                            severity: EmergencySeverity(rawValue: severity) ?? .medium,
                            message: "Emergency triggered from Apple Watch",
                            autoTriggered: false
                        )
                    }
                }
            case "emergency_response":
                // Handle emergency response from watch
                break
            default:
                break
            }
        }
    }
}

// MARK: - Supporting Types

enum ConnectionStatus {
    case unknown
    case connected
    case disconnected
    case connecting
}