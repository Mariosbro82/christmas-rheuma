//
//  EmergencyManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import CoreLocation
import Contacts
import ContactsUI
import UserNotifications
import HealthKit
import Combine
import os.log

// MARK: - Emergency Manager
class EmergencyManager: NSObject, ObservableObject {
    static let shared = EmergencyManager()
    
    // MARK: - Properties
    @Published var emergencyContacts: [EmergencyContact] = []
    @Published var emergencyProfile: EmergencyProfile?
    @Published var currentEmergencyStatus: EmergencyStatus = .normal
    @Published var nearbyEmergencyServices: [EmergencyService] = []
    @Published var emergencyAlerts: [EmergencyAlert] = []
    @Published var isLocationEnabled = false
    @Published var emergencySettings = EmergencySettings()
    @Published var medicalInformation = MedicalInformation()
    @Published var emergencyHistory: [EmergencyEvent] = []
    
    // Location and Services
    private let locationManager = CLLocationManager()
    private var currentLocation: CLLocation?
    private let geocoder = CLGeocoder()
    
    // Health Monitoring
    private let healthStore = HKHealthStore()
    private var healthObservers: [HKObserverQuery] = []
    
    // Emergency Detection
    private let emergencyDetector = EmergencyDetector()
    private let flareDetector = FlareDetector()
    private let vitalSignsMonitor = VitalSignsMonitor()
    
    // Notification and Communication
    private let notificationManager = EmergencyNotificationManager()
    private let communicationManager = EmergencyCommunicationManager()
    
    // Data Management
    private let dataManager = EmergencyDataManager()
    
    private let logger = Logger(subsystem: "com.inflamai.emergency", category: "EmergencyManager")
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    override init() {
        super.init()
        
        setupLocationManager()
        setupHealthMonitoring()
        setupEmergencyDetection()
        setupNotifications()
        loadEmergencyData()
        requestPermissions()
    }
    
    deinit {
        stopHealthMonitoring()
        stopEmergencyDetection()
    }
    
    // MARK: - Setup
    private func setupLocationManager() {
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = 100 // 100 meters
    }
    
    private func setupHealthMonitoring() {
        guard HKHealthStore.isHealthDataAvailable() else {
            logger.warning("HealthKit not available")
            return
        }
        
        // Setup health data observers for emergency detection
        setupHeartRateMonitoring()
        setupBloodPressureMonitoring()
        setupActivityMonitoring()
    }
    
    private func setupEmergencyDetection() {
        emergencyDetector.delegate = self
        flareDetector.delegate = self
        vitalSignsMonitor.delegate = self
        
        // Start monitoring
        emergencyDetector.startMonitoring()
        flareDetector.startMonitoring()
        vitalSignsMonitor.startMonitoring()
    }
    
    private func setupNotifications() {
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.handleAppDidBecomeActive()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .sink { [weak self] _ in
                self?.handleAppWillResignActive()
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Permissions
    private func requestPermissions() {
        requestLocationPermission()
        requestHealthPermissions()
        requestNotificationPermissions()
        requestContactsPermission()
    }
    
    private func requestLocationPermission() {
        switch locationManager.authorizationStatus {
        case .notDetermined:
            locationManager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            logger.warning("Location permission denied")
        case .authorizedWhenInUse, .authorizedAlways:
            isLocationEnabled = true
            startLocationUpdates()
        @unknown default:
            logger.error("Unknown location authorization status")
        }
    }
    
    private func requestHealthPermissions() {
        let typesToRead: Set<HKObjectType> = [
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureSystolic)!,
            HKObjectType.quantityType(forIdentifier: .bloodPressureDiastolic)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning)!,
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!
        ]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            if let error = error {
                self?.logger.error("Health permission error: \(error.localizedDescription)")
            } else if success {
                self?.logger.info("Health permissions granted")
                self?.startHealthMonitoring()
            }
        }
    }
    
    private func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge, .criticalAlert]) { granted, error in
            if let error = error {
                self.logger.error("Notification permission error: \(error.localizedDescription)")
            } else if granted {
                self.logger.info("Notification permissions granted")
            }
        }
    }
    
    private func requestContactsPermission() {
        CNContactStore().requestAccess(for: .contacts) { granted, error in
            if let error = error {
                self.logger.error("Contacts permission error: \(error.localizedDescription)")
            } else if granted {
                self.logger.info("Contacts permissions granted")
            }
        }
    }
    
    // MARK: - Emergency Contacts
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
    
    func importContactFromAddressBook() -> CNContactPickerViewController {
        let picker = CNContactPickerViewController()
        picker.predicateForEnablingContact = NSPredicate(format: "phoneNumbers.@count > 0")
        return picker
    }
    
    func createEmergencyContactFromCNContact(_ cnContact: CNContact, relationship: ContactRelationship) -> EmergencyContact {
        let phoneNumber = cnContact.phoneNumbers.first?.value.stringValue ?? ""
        let email = cnContact.emailAddresses.first?.value as String? ?? ""
        
        return EmergencyContact(
            id: UUID().uuidString,
            name: "\(cnContact.givenName) \(cnContact.familyName)",
            phoneNumber: phoneNumber,
            email: email,
            relationship: relationship,
            isPrimary: emergencyContacts.isEmpty,
            notificationPreferences: ContactNotificationPreferences()
        )
    }
    
    // MARK: - Emergency Profile
    func updateEmergencyProfile(_ profile: EmergencyProfile) {
        emergencyProfile = profile
        saveEmergencyData()
        logger.info("Emergency profile updated")
    }
    
    func updateMedicalInformation(_ info: MedicalInformation) {
        medicalInformation = info
        saveEmergencyData()
        logger.info("Medical information updated")
    }
    
    // MARK: - Emergency Detection
    func triggerManualEmergency(type: EmergencyType, severity: EmergencySeverity) {
        let emergency = EmergencyEvent(
            id: UUID().uuidString,
            type: type,
            severity: severity,
            timestamp: Date(),
            location: currentLocation,
            isManual: true,
            description: "Manual emergency triggered",
            vitalSigns: vitalSignsMonitor.getCurrentVitalSigns(),
            symptoms: [],
            actions: []
        )
        
        handleEmergencyEvent(emergency)
    }
    
    private func handleEmergencyEvent(_ event: EmergencyEvent) {
        logger.critical("Emergency detected: \(event.type.rawValue) - \(event.severity.rawValue)")
        
        // Update status
        currentEmergencyStatus = .emergency(event.severity)
        
        // Add to history
        emergencyHistory.append(event)
        
        // Create alert
        let alert = EmergencyAlert(
            id: UUID().uuidString,
            type: .emergency,
            severity: event.severity,
            title: "Emergency Detected",
            message: getEmergencyMessage(for: event),
            timestamp: Date(),
            isActive: true,
            actions: getEmergencyActions(for: event)
        )
        
        emergencyAlerts.append(alert)
        
        // Execute emergency response
        Task {
            await executeEmergencyResponse(for: event)
        }
        
        // Save data
        saveEmergencyData()
        
        // Post notification
        NotificationCenter.default.post(name: .emergencyDetected, object: event)
    }
    
    private func executeEmergencyResponse(for event: EmergencyEvent) async {
        logger.info("Executing emergency response for: \(event.type.rawValue)")
        
        // 1. Send notifications to emergency contacts
        await notifyEmergencyContacts(for: event)
        
        // 2. Send local notifications
        await sendEmergencyNotification(for: event)
        
        // 3. Find nearby emergency services
        await findNearbyEmergencyServices(for: event.type)
        
        // 4. Prepare emergency information
        await prepareEmergencyInformation()
        
        // 5. Start continuous monitoring
        startEmergencyMonitoring()
        
        // 6. Log emergency actions
        logEmergencyActions(for: event)
    }
    
    private func notifyEmergencyContacts(for event: EmergencyEvent) async {
        let relevantContacts = emergencyContacts.filter { contact in
            shouldNotifyContact(contact, for: event)
        }
        
        for contact in relevantContacts {
            await communicationManager.notifyContact(contact, about: event, location: currentLocation)
        }
    }
    
    private func shouldNotifyContact(_ contact: EmergencyContact, for event: EmergencyEvent) -> Bool {
        // Check notification preferences and emergency severity
        switch event.severity {
        case .low:
            return contact.notificationPreferences.notifyForLowSeverity
        case .medium:
            return contact.notificationPreferences.notifyForMediumSeverity
        case .high:
            return contact.notificationPreferences.notifyForHighSeverity
        case .critical:
            return true // Always notify for critical emergencies
        }
    }
    
    private func sendEmergencyNotification(for event: EmergencyEvent) async {
        await notificationManager.sendEmergencyNotification(
            title: "Emergency Alert",
            body: getEmergencyMessage(for: event),
            severity: event.severity
        )
    }
    
    // MARK: - Emergency Services
    func findNearbyEmergencyServices(for type: EmergencyType) async {
        guard let location = currentLocation else {
