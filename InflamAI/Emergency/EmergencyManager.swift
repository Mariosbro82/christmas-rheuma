//
//  EmergencyManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CoreLocation
import Contacts
import MessageUI
import CallKit
import UserNotifications
import HealthKit
import SwiftUI
import os.log
import Combine

// MARK: - Emergency Manager
class EmergencyManager: NSObject, ObservableObject {
    
    static let shared = EmergencyManager()
    
    private let logger = Logger(subsystem: "InflamAI", category: "Emergency")
    private let locationManager = CLLocationManager()
    private let healthStore = HKHealthStore()
    private let notificationCenter = UNUserNotificationCenter.current()
    
    // Published properties
    @Published var emergencyContacts: [EmergencyContact] = []
    @Published var emergencyProfile = EmergencyProfile()
    @Published var emergencySettings = EmergencySettings()
    @Published var activeEmergencies: [EmergencyEvent] = []
    @Published var nearbyEmergencyServices: [EmergencyService] = []
    @Published var currentLocation: CLLocation?
    @Published var isLocationAuthorized = false
    @Published var isEmergencyModeActive = false
    @Published var lastEmergencyCheck = Date()
    
    // Emergency detection
    @Published var currentPainLevel: Int = 0
    @Published var currentVitalSigns = VitalSigns()
    @Published var currentSymptoms: [Symptom] = []
    @Published var emergencyRiskLevel: EmergencyRiskLevel = .low
    
    // Internal state
    private var cancellables = Set<AnyCancellable>()
    private var emergencyTimer: Timer?
    private var locationUpdateTimer: Timer?
    private var healthObservers: [HKObserverQuery] = []
    
    override init() {
        super.init()
        setupLocationManager()
        setupHealthKit()
        setupNotifications()
        loadEmergencyData()
        startEmergencyMonitoring()
    }
    
    deinit {
        stopEmergencyMonitoring()
        stopHealthObservers()
    }
    
    // MARK: - Public Methods
    
    func addEmergencyContact(_ contact: EmergencyContact) {
        emergencyContacts.append(contact)
        saveEmergencyData()
        
        logger.info("Emergency contact added: \(contact.name)")
    }
    
    func removeEmergencyContact(_ contact: EmergencyContact) {
        emergencyContacts.removeAll { $0.id == contact.id }
        saveEmergencyData()
        
        logger.info("Emergency contact removed: \(contact.name)")
    }
    
    func updateEmergencyContact(_ contact: EmergencyContact) {
        if let index = emergencyContacts.firstIndex(where: { $0.id == contact.id }) {
            emergencyContacts[index] = contact
            saveEmergencyData()
            
            logger.info("Emergency contact updated: \(contact.name)")
        }
    }
    
    func updateEmergencyProfile(_ profile: EmergencyProfile) {
        emergencyProfile = profile
        saveEmergencyData()
        
        logger.info("Emergency profile updated")
    }
    
    func updateEmergencySettings(_ settings: EmergencySettings) {
        emergencySettings = settings
        saveEmergencyData()
        
        if settings.enableAutomaticDetection {
            startEmergencyMonitoring()
        } else {
            stopEmergencyMonitoring()
        }
        
        logger.info("Emergency settings updated")
    }
    
    func triggerEmergency(type: EmergencyType, severity: EmergencySeverity, description: String? = nil) {
        let emergency = EmergencyEvent(
            id: UUID(),
            type: type,
            severity: severity,
            timestamp: Date(),
            location: currentLocation,
            description: description,
            isResolved: false,
            responseActions: []
        )
        
        activeEmergencies.append(emergency)
        isEmergencyModeActive = true
        
        handleEmergencyEvent(emergency)
        
        logger.critical("Emergency triggered: \(type) - \(severity)")
    }
    
    func resolveEmergency(_ emergencyId: UUID) {
        if let index = activeEmergencies.firstIndex(where: { $0.id == emergencyId }) {
            activeEmergencies[index].isResolved = true
            activeEmergencies[index].resolvedAt = Date()
            
            // Check if all emergencies are resolved
            if activeEmergencies.allSatisfy({ $0.isResolved }) {
                isEmergencyModeActive = false
            }
            
            saveEmergencyData()
            
            logger.info("Emergency resolved: \(emergencyId)")
        }
    }
    
    func findNearbyEmergencyServices() {
        guard let location = currentLocation else {
            logger.warning("Cannot find nearby services without location")
            return
        }
        
        let searchRadius: CLLocationDistance = 10000 // 10km
        
        Task {
            do {
                let services = try await searchEmergencyServices(near: location, radius: searchRadius)
                
                await MainActor.run {
                    self.nearbyEmergencyServices = services
                }
                
                logger.info("Found \(services.count) nearby emergency services")
            } catch {
                logger.error("Failed to find nearby emergency services: \(error.localizedDescription)")
            }
        }
    }
    
    func sendEmergencyMessage(to contact: EmergencyContact, message: String) {
        guard MFMessageComposeViewController.canSendText() else {
            logger.error("Cannot send text messages")
            return
        }
        
        let messageComposer = MFMessageComposeViewController()
        messageComposer.recipients = [contact.phoneNumber]
        messageComposer.body = message
        messageComposer.messageComposeDelegate = self
        
        // Present the message composer
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootViewController = window.rootViewController {
            rootViewController.present(messageComposer, animated: true)
        }
        
        logger.info("Emergency message sent to \(contact.name)")
    }
    
    func callEmergencyContact(_ contact: EmergencyContact) {
        guard let url = URL(string: "tel://\(contact.phoneNumber)") else {
            logger.error("Invalid phone number: \(contact.phoneNumber)")
            return
        }
        
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
            logger.info("Calling emergency contact: \(contact.name)")
        } else {
            logger.error("Cannot make phone calls")
        }
    }
    
    func callEmergencyServices() {
        let emergencyNumber = emergencySettings.emergencyNumber
        
        guard let url = URL(string: "tel://\(emergencyNumber)") else {
            logger.error("Invalid emergency number: \(emergencyNumber)")
            return
        }
        
        if UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
            logger.critical("Calling emergency services: \(emergencyNumber)")
        } else {
            logger.error("Cannot make phone calls")
        }
    }
    
    func updatePainLevel(_ level: Int) {
        currentPainLevel = level
        checkForEmergencyConditions()
        
        logger.debug("Pain level updated: \(level)")
    }
    
    func updateVitalSigns(_ vitalSigns: VitalSigns) {
        currentVitalSigns = vitalSigns
        checkForEmergencyConditions()
        
        logger.debug("Vital signs updated")
    }
    
    func updateSymptoms(_ symptoms: [Symptom]) {
        currentSymptoms = symptoms
        checkForEmergencyConditions()
        
        logger.debug("Symptoms updated: \(symptoms.count) symptoms")
    }
    
    func getEmergencyMedicalCard() -> EmergencyMedicalCard {
        return EmergencyMedicalCard(
            profile: emergencyProfile,
            contacts: emergencyContacts,
            currentMedications: emergencyProfile.currentMedications,
            allergies: emergencyProfile.allergies,
            medicalConditions: emergencyProfile.medicalConditions,
            emergencyInstructions: emergencyProfile.emergencyInstructions,
            lastUpdated: Date()
        )
    }
    
    func exportEmergencyData() -> Data? {
        do {
            let export = EmergencyDataExport(
                timestamp: Date(),
                contacts: emergencyContacts,
                profile: emergencyProfile,
                settings: emergencySettings,
                emergencyHistory: activeEmergencies
            )
            
            return try JSONEncoder().encode(export)
        } catch {
            logger.error("Failed to export emergency data: \(error.localizedDescription)")
            return nil
        }
    }
    
    func importEmergencyData(from data: Data) -> Bool {
        do {
            let export = try JSONDecoder().decode(EmergencyDataExport.self, from: data)
            
            emergencyContacts = export.contacts
            emergencyProfile = export.profile
            emergencySettings = export.settings
            
            saveEmergencyData()
            
            logger.info("Emergency data imported successfully")
            return true
        } catch {
            logger.error("Failed to import emergency data: \(error.localizedDescription)")
            return false
        }
    }
    
    // MARK: - Private Methods
    
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.requestWhenInUseAuthorization()
    }
    
    private func setupHealthKit() {
        guard HKHealthStore.isHealthDataAvailable() else {
            logger.warning("HealthKit not available")
            return
        }
        
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!,
            HKObjectType.quantityType(forIdentifier: .bodyTemperature)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!
        ]
        
        healthStore.requestAuthorization(toShare: [], read: typesToRead) { [weak self] success, error in
            if success {
                self?.setupHealthObservers()
            } else {
                self?.logger.error("HealthKit authorization failed: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    private func setupHealthObservers() {
        let heartRateType = HKObjectType.quantityType(forIdentifier: .heartRate)!
        let heartRateObserver = HKObserverQuery(sampleType: heartRateType, predicate: nil) { [weak self] _, _, error in
            if let error = error {
                self?.logger.error("Heart rate observer error: \(error.localizedDescription)")
                return
            }
            
            self?.fetchLatestHeartRate()
        }
        
        healthStore.execute(heartRateObserver)
        healthObservers.append(heartRateObserver)
        
        // Add observers for other vital signs...
    }
    
    private func setupNotifications() {
        notificationCenter.requestAuthorization(options: [.alert, .sound, .badge, .criticalAlert]) { granted, error in
            if granted {
                self.logger.info("Notification authorization granted")
            } else {
                self.logger.warning("Notification authorization denied")
            }
        }
    }
    
    private func startEmergencyMonitoring() {
        guard emergencySettings.enableAutomaticDetection else { return }
        
        emergencyTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.checkForEmergencyConditions()
        }
        
        locationUpdateTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak self] _ in
            self?.updateLocation()
        }
        
        logger.info("Emergency monitoring started")
    }
    
    private func stopEmergencyMonitoring() {
        emergencyTimer?.invalidate()
        emergencyTimer = nil
        
        locationUpdateTimer?.invalidate()
        locationUpdateTimer = nil
        
        logger.info("Emergency monitoring stopped")
    }
    
    private func stopHealthObservers() {
        healthObservers.forEach { observer in
            healthStore.stop(observer)
        }
        healthObservers.removeAll()
    }
    
    private func checkForEmergencyConditions() {
        lastEmergencyCheck = Date()
        
        let riskFactors = assessEmergencyRisk()
        emergencyRiskLevel = calculateOverallRisk(from: riskFactors)
        
        if emergencyRiskLevel == .critical {
            handleCriticalCondition(riskFactors: riskFactors)
        } else if emergencyRiskLevel == .high {
            handleHighRiskCondition(riskFactors: riskFactors)
        }
        
        logger.debug("Emergency risk level: \(emergencyRiskLevel)")
    }
    
    private func assessEmergencyRisk() -> [EmergencyRiskFactor] {
        var riskFactors: [EmergencyRiskFactor] = []
        
        // Check severe pain
        if currentPainLevel >= emergencySettings.severePainThreshold {
            riskFactors.append(.severePain(level: currentPainLevel))
        }
        
        // Check vital signs
        if let heartRate = currentVitalSigns.heartRate {
            if heartRate > emergencySettings.maxHeartRate || heartRate < emergencySettings.minHeartRate {
                riskFactors.append(.abnormalHeartRate(rate: heartRate))
            }
        }
        
        if let systolic = currentVitalSigns.bloodPressureSystolic,
           let diastolic = currentVitalSigns.bloodPressureDiastolic {
            if systolic > emergencySettings.maxSystolicBP || diastolic > emergencySettings.maxDiastolicBP {
                riskFactors.append(.highBloodPressure(systolic: systolic, diastolic: diastolic))
            }
        }
        
        if let oxygenSaturation = currentVitalSigns.oxygenSaturation {
            if oxygenSaturation < emergencySettings.minOxygenSaturation {
                riskFactors.append(.lowOxygenSaturation(level: oxygenSaturation))
            }
        }
        
        if let temperature = currentVitalSigns.bodyTemperature {
            if temperature > emergencySettings.maxBodyTemperature {
                riskFactors.append(.highFever(temperature: temperature))
            }
        }
        
        // Check dangerous symptom combinations
        let dangerousSymptoms = currentSymptoms.filter { $0.severity == .severe }
        if dangerousSymptoms.count >= 3 {
            riskFactors.append(.multipleSymptoms(symptoms: dangerousSymptoms))
        }
        
        // Check for specific dangerous symptoms
        if currentSymptoms.contains(where: { $0.type == .chestPain && $0.severity == .severe }) {
            riskFactors.append(.chestPain)
        }
        
        if currentSymptoms.contains(where: { $0.type == .difficultyBreathing && $0.severity == .severe }) {
            riskFactors.append(.breathingDifficulty)
        }
        
        if currentSymptoms.contains(where: { $0.type == .severeHeadache && $0.severity == .severe }) {
            riskFactors.append(.severeHeadache)
        }
        
        return riskFactors
    }
    
    private func calculateOverallRisk(from riskFactors: [EmergencyRiskFactor]) -> EmergencyRiskLevel {
        if riskFactors.isEmpty {
            return .low
        }
        
        let criticalFactors = riskFactors.filter { $0.isCritical }
        if !criticalFactors.isEmpty {
            return .critical
        }
        
        let highRiskFactors = riskFactors.filter { $0.isHighRisk }
        if highRiskFactors.count >= 2 {
            return .critical
        } else if !highRiskFactors.isEmpty {
            return .high
        }
        
        return .medium
    }
    
    private func handleCriticalCondition(riskFactors: [EmergencyRiskFactor]) {
        let emergency = EmergencyEvent(
            id: UUID(),
            type: .medicalEmergency,
            severity: .critical,
            timestamp: Date(),
            location: currentLocation,
            description: "Critical condition detected: \(riskFactors.map { $0.description }.joined(separator: ", "))",
            isResolved: false,
            responseActions: []
        )
        
        activeEmergencies.append(emergency)
        isEmergencyModeActive = true
        
        handleEmergencyEvent(emergency)
        
        logger.critical("Critical emergency condition detected")
    }
    
    private func handleHighRiskCondition(riskFactors: [EmergencyRiskFactor]) {
        // Send warning notification
        sendEmergencyNotification(
            title: "High Risk Condition Detected",
            body: "Please check your symptoms and consider contacting your healthcare provider.",
            isUrgent: true
        )
        
        logger.warning("High risk condition detected: \(riskFactors.map { $0.description }.joined(separator: ", "))")
    }
    
    private func handleEmergencyEvent(_ emergency: EmergencyEvent) {
        // Send critical notification
        sendEmergencyNotification(
            title: "Emergency Detected",
            body: emergency.description ?? "Critical condition detected. Immediate attention required.",
            isUrgent: true
        )
        
        // Auto-contact emergency services if enabled
        if emergencySettings.autoContactEmergencyServices {
            callEmergencyServices()
        }
        
        // Contact emergency contacts
        if emergencySettings.autoContactEmergencyContacts {
            contactEmergencyContacts(for: emergency)
        }
        
        // Update location
        updateLocation()
        
        saveEmergencyData()
    }
    
    private func contactEmergencyContacts(for emergency: EmergencyEvent) {
        let message = generateEmergencyMessage(for: emergency)
        
        for contact in emergencyContacts.prefix(3) { // Contact first 3 contacts
            if contact.contactMethods.contains(.sms) {
                sendEmergencyMessage(to: contact, message: message)
            }
            
            if contact.contactMethods.contains(.call) && contact.isPrimary {
                callEmergencyContact(contact)
            }
        }
    }
    
    private func generateEmergencyMessage(for emergency: EmergencyEvent) -> String {
        var message = "EMERGENCY ALERT: \(emergencyProfile.name) needs immediate assistance.\n\n"
        
        if let description = emergency.description {
            message += "Condition: \(description)\n\n"
        }
        
        if let location = emergency.location {
            message += "Location: \(location.coordinate.latitude), \(location.coordinate.longitude)\n\n"
        }
        
        message += "Medical Info:\n"
        message += "Conditions: \(emergencyProfile.medicalConditions.joined(separator: ", "))\n"
        message += "Medications: \(emergencyProfile.currentMedications.joined(separator: ", "))\n"
        message += "Allergies: \(emergencyProfile.allergies.joined(separator: ", "))\n\n"
        
        message += "Time: \(DateFormatter.localizedString(from: emergency.timestamp, dateStyle: .short, timeStyle: .medium))"
        
        return message
    }
    
    private func sendEmergencyNotification(title: String, body: String, isUrgent: Bool) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = isUrgent ? .defaultCritical : .default
        
        if isUrgent {
            content.interruptionLevel = .critical
        }
        
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil // Immediate
        )
        
        notificationCenter.add(request) { error in
            if let error = error {
                self.logger.error("Failed to send emergency notification: \(error.localizedDescription)")
            }
        }
    }
    
    private func updateLocation() {
        guard isLocationAuthorized else { return }
        locationManager.requestLocation()
    }
    
    private func fetchLatestHeartRate() {
        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: heartRateType, predicate: nil, limit: 1, sortDescriptors: [sortDescriptor]) { [weak self] _, samples, error in
            
            if let error = error {
                self?.logger.error("Failed to fetch heart rate: \(error.localizedDescription)")
                return
            }
            
            guard let sample = samples?.first as? HKQuantitySample else { return }
            
            let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            
            DispatchQueue.main.async {
                self?.currentVitalSigns.heartRate = heartRate
                self?.checkForEmergencyConditions()
            }
        }
        
        healthStore.execute(query)
    }
    
    private func searchEmergencyServices(near location: CLLocation, radius: CLLocationDistance) async throws -> [EmergencyService] {
        // Implementation would use MapKit or other location services
        // to find nearby hospitals, urgent care centers, etc.
        return [] // Placeholder
    }
    
    private func saveEmergencyData() {
        let data = EmergencyData(
            contacts: emergencyContacts,
            profile: emergencyProfile,
            settings: emergencySettings,
            activeEmergencies: activeEmergencies
        )
        
        do {
            let encodedData = try JSONEncoder().encode(data)
            UserDefaults.standard.set(encodedData, forKey: "EmergencyData")
        } catch {
            logger.error("Failed to save emergency data: \(error.localizedDescription)")
        }
    }
    
    private func loadEmergencyData() {
        guard let data = UserDefaults.standard.data(forKey: "EmergencyData"),
              let emergencyData = try? JSONDecoder().decode(EmergencyData.self, from: data) else {
            return
        }
        
        emergencyContacts = emergencyData.contacts
        emergencyProfile = emergencyData.profile
        emergencySettings = emergencyData.settings
        activeEmergencies = emergencyData.activeEmergencies
        
        // Check if any emergencies are still active
        isEmergencyModeActive = activeEmergencies.contains { !$0.isResolved }
    }
}

// MARK: - CLLocationManagerDelegate

extension EmergencyManager: CLLocationManagerDelegate {
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        currentLocation = location
        
        logger.debug("Location updated: \(location.coordinate.latitude), \(location.coordinate.longitude)")
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        logger.error("Location update failed: \(error.localizedDescription)")
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        switch status {
        case .authorizedWhenInUse, .authorizedAlways:
            isLocationAuthorized = true
            updateLocation()
        case .denied, .restricted:
            isLocationAuthorized = false
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        @unknown default:
            break
        }
    }
}

// MARK: - MFMessageComposeViewControllerDelegate

extension EmergencyManager: MFMessageComposeViewControllerDelegate {
    
    func messageComposeViewController(_ controller: MFMessageComposeViewController, didFinishWith result: MessageComposeResult) {
        controller.dismiss(animated: true)
        
        switch result {
        case .sent:
            logger.info("Emergency message sent successfully")
        case .failed:
            logger.error("Failed to send emergency message")
        case .cancelled:
            logger.info("Emergency message cancelled")
        @unknown default:
            break
        }
    }
}

// MARK: - Supporting Types

struct EmergencyContact: Identifiable, Codable {
    let id = UUID()
    var name: String
    var phoneNumber: String
    var email: String?
    var relationship: String
    var isPrimary: Bool
    var contactMethods: Set<ContactMethod>
    var notes: String?
    
    enum ContactMethod: String, Codable, CaseIterable {
        case call = "call"
        case sms = "sms"
        case email = "email"
    }
}

struct EmergencyProfile: Codable {
    var name: String = ""
    var dateOfBirth: Date = Date()
    var bloodType: String = ""
    var allergies: [String] = []
    var currentMedications: [String] = []
    var medicalConditions: [String] = []
    var emergencyInstructions: String = ""
    var insuranceInfo: String = ""
    var doctorName: String = ""
    var doctorPhone: String = ""
    var preferredHospital: String = ""
}

struct EmergencySettings: Codable {
    var enableAutomaticDetection = true
    var autoContactEmergencyServices = false
    var autoContactEmergencyContacts = true
    var emergencyNumber = "911"
    var severePainThreshold = 8
    var maxHeartRate: Double = 120
    var minHeartRate: Double = 50
    var maxSystolicBP: Double = 180
    var maxDiastolicBP: Double = 110
    var minOxygenSaturation: Double = 90
    var maxBodyTemperature: Double = 102.0
    var enableLocationSharing = true
    var emergencyCheckInterval: TimeInterval = 30
}

struct EmergencyEvent: Identifiable, Codable {
    let id: UUID
    let type: EmergencyType
    let severity: EmergencySeverity
    let timestamp: Date
    let location: CLLocation?
    let description: String?
    var isResolved: Bool
    var resolvedAt: Date?
    var responseActions: [EmergencyResponseAction]
    
    private enum CodingKeys: String, CodingKey {
        case id, type, severity, timestamp, description, isResolved, resolvedAt, responseActions
    }
    
    init(id: UUID, type: EmergencyType, severity: EmergencySeverity, timestamp: Date, location: CLLocation?, description: String?, isResolved: Bool, responseActions: [EmergencyResponseAction]) {
        self.id = id
        self.type = type
        self.severity = severity
        self.timestamp = timestamp
        self.location = location
        self.description = description
        self.isResolved = isResolved
        self.responseActions = responseActions
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        type = try container.decode(EmergencyType.self, forKey: .type)
        severity = try container.decode(EmergencySeverity.self, forKey: .severity)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        location = nil // CLLocation is not Codable
        description = try container.decodeIfPresent(String.self, forKey: .description)
        isResolved = try container.decode(Bool.self, forKey: .isResolved)
        resolvedAt = try container.decodeIfPresent(Date.self, forKey: .resolvedAt)
        responseActions = try container.decode([EmergencyResponseAction].self, forKey: .responseActions)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(type, forKey: .type)
        try container.encode(severity, forKey: .severity)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encode(isResolved, forKey: .isResolved)
        try container.encodeIfPresent(resolvedAt, forKey: .resolvedAt)
        try container.encode(responseActions, forKey: .responseActions)
    }
}

struct EmergencyService: Identifiable, Codable {
    let id = UUID()
    let name: String
    let type: ServiceType
    let address: String
    let phoneNumber: String
    let distance: Double // in meters
    let isOpen24Hours: Bool
    let rating: Double?
    
    enum ServiceType: String, Codable {
        case hospital = "hospital"
        case urgentCare = "urgent_care"
        case pharmacy = "pharmacy"
        case fireStation = "fire_station"
        case policeStation = "police_station"
    }
}

struct VitalSigns: Codable {
    var heartRate: Double?
    var bloodPressureSystolic: Double?
    var bloodPressureDiastolic: Double?
    var oxygenSaturation: Double?
    var bodyTemperature: Double?
    var respiratoryRate: Double?
    var timestamp: Date = Date()
}

struct Symptom: Identifiable, Codable {
    let id = UUID()
    let type: SymptomType
    let severity: SymptomSeverity
    let description: String?
    let timestamp: Date
}

enum EmergencyType: String, Codable {
    case medicalEmergency = "medical_emergency"
    case severeFlare = "severe_flare"
    case medicationReaction = "medication_reaction"
    case fall = "fall"
    case other = "other"
}

enum EmergencySeverity: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

enum EmergencyRiskLevel: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

enum EmergencyRiskFactor {
    case severePain(level: Int)
    case abnormalHeartRate(rate: Double)
    case highBloodPressure(systolic: Double, diastolic: Double)
    case lowOxygenSaturation(level: Double)
    case highFever(temperature: Double)
    case multipleSymptoms(symptoms: [Symptom])
    case chestPain
    case breathingDifficulty
    case severeHeadache
    
    var isCritical: Bool {
        switch self {
        case .chestPain, .breathingDifficulty:
            return true
        case .severePain(let level):
            return level >= 9
        case .abnormalHeartRate(let rate):
            return rate > 150 || rate < 40
        case .lowOxygenSaturation(let level):
            return level < 85
        case .highFever(let temperature):
            return temperature > 104.0
        default:
            return false
        }
    }
    
    var isHighRisk: Bool {
        switch self {
        case .severePain(let level):
            return level >= 7
        case .abnormalHeartRate(let rate):
            return rate > 120 || rate < 50
        case .highBloodPressure(let systolic, let diastolic):
            return systolic > 160 || diastolic > 100
        case .lowOxygenSaturation(let level):
            return level < 90
        case .highFever(let temperature):
            return temperature > 102.0
        case .multipleSymptoms:
            return true
        case .severeHeadache:
            return true
        default:
            return isCritical
        }
    }
    
    var description: String {
        switch self {
        case .severePain(let level):
            return "Severe pain (level \(level))"
        case .abnormalHeartRate(let rate):
            return "Abnormal heart rate (\(Int(rate)) bpm)"
        case .highBloodPressure(let systolic, let diastolic):
            return "High blood pressure (\(Int(systolic))/\(Int(diastolic)))"
        case .lowOxygenSaturation(let level):
            return "Low oxygen saturation (\(Int(level))%)"
        case .highFever(let temperature):
            return "High fever (\(temperature)°F)"
        case .multipleSymptoms(let symptoms):
            return "Multiple severe symptoms (\(symptoms.count))"
        case .chestPain:
            return "Chest pain"
        case .breathingDifficulty:
            return "Difficulty breathing"
        case .severeHeadache:
            return "Severe headache"
        }
    }
}

enum SymptomType: String, Codable {
    case chestPain = "chest_pain"
    case difficultyBreathing = "difficulty_breathing"
    case severeHeadache = "severe_headache"
    case nausea = "nausea"
    case dizziness = "dizziness"
    case fatigue = "fatigue"
    case jointPain = "joint_pain"
    case swelling = "swelling"
    case rash = "rash"
    case fever = "fever"
    case other = "other"
}

enum SymptomSeverity: String, Codable {
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
}

struct EmergencyResponseAction: Codable {
    let type: ActionType
    let timestamp: Date
    let description: String
    let wasSuccessful: Bool
    
    enum ActionType: String, Codable {
        case notificationSent = "notification_sent"
        case contactCalled = "contact_called"
        case messageSent = "message_sent"
        case emergencyServicesCalled = "emergency_services_called"
        case locationShared = "location_shared"
    }
}

struct EmergencyMedicalCard: Codable {
    let profile: EmergencyProfile
    let contacts: [EmergencyContact]
    let currentMedications: [String]
    let allergies: [String]
    let medicalConditions: [String]
    let emergencyInstructions: String
    let lastUpdated: Date
}

struct EmergencyData: Codable {
    let contacts: [EmergencyContact]
    let profile: EmergencyProfile
    let settings: EmergencySettings
    let activeEmergencies: [EmergencyEvent]
}

struct EmergencyDataExport: Codable {
    let timestamp: Date
    let contacts: [EmergencyContact]
    let profile: EmergencyProfile
    let settings: EmergencySettings
    let emergencyHistory: [EmergencyEvent]
}