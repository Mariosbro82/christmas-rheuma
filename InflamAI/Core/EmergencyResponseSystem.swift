//
//  EmergencyResponseSystem.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Combine
import Foundation
import HealthKit
import CoreLocation
import UserNotifications
import CallKit
import MessageUI
import CoreML

// MARK: - Emergency Response System
class EmergencyResponseSystem: NSObject, ObservableObject {
    @Published var emergencyContacts: [EmergencyContact] = []
    @Published var emergencyAlerts: [EmergencyAlert] = []
    @Published var emergencyHistory: [EmergencyEvent] = []
    @Published var isEmergencyActive: Bool = false
    @Published var currentEmergencyLevel: EmergencyLevel = .none
    @Published var locationPermissionStatus: CLAuthorizationStatus = .notDetermined
    @Published var emergencySettings: EmergencySettings = EmergencySettings()
    @Published var vitalSignsStatus: VitalSignsStatus = VitalSignsStatus()
    @Published var emergencyInsights: [EmergencyInsight] = []
    
    private let healthKitManager = HealthKitEmergencyManager()
    private let locationManager = CLLocationManager()
    private let notificationManager = EmergencyNotificationManager()
    private let communicationManager = EmergencyCommunicationManager()
    private let mlPredictor = EmergencyMLPredictor()
    private let fallDetector = FallDetectionEngine()
    private let vitalSignsMonitor = EmergencyVitalSignsMonitor()
    private let geofenceManager = EmergencyGeofenceManager()
    private let wearableIntegration = EmergencyWearableIntegration()
    
    private var cancellables = Set<AnyCancellable>()
    private var emergencyTimer: Timer?
    private var vitalSignsTimer: Timer?
    private var locationTimer: Timer?
    
    private var currentLocation: CLLocation?
    private var emergencyCallManager: CXCallController?
    
    override init() {
        super.init()
        setupEmergencySystem()
        loadStoredData()
        setupHealthKitMonitoring()
        setupLocationServices()
        setupNotifications()
        setupFallDetection()
        setupVitalSignsMonitoring()
        requestPermissions()
    }
    
    // MARK: - Setup
    private func setupEmergencySystem() {
        healthKitManager.delegate = self
        fallDetector.delegate = self
        vitalSignsMonitor.delegate = self
        locationManager.delegate = self
        notificationManager.delegate = self
        
        emergencyCallManager = CXCallController()
    }
    
    private func loadStoredData() {
        loadEmergencyContacts()
        loadEmergencyHistory()
        loadEmergencySettings()
    }
    
    private func setupHealthKitMonitoring() {
        healthKitManager.requestAuthorization { [weak self] success in
            if success {
                self?.startHealthKitMonitoring()
            }
        }
    }
    
    private func setupLocationServices() {
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = 10.0
        
        if emergencySettings.enableLocationTracking {
            requestLocationPermission()
        }
    }
    
    private func setupNotifications() {
        NotificationCenter.default.publisher(for: .emergencyTriggered)
            .sink { [weak self] notification in
                self?.handleEmergencyTrigger(notification)
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: .vitalSignsAlert)
            .sink { [weak self] notification in
                self?.handleVitalSignsAlert(notification)
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: .fallDetected)
            .sink { [weak self] notification in
                self?.handleFallDetection(notification)
            }
            .store(in: &cancellables)
    }
    
    private func setupFallDetection() {
        if emergencySettings.enableFallDetection {
            fallDetector.startMonitoring()
        }
    }
    
    private func setupVitalSignsMonitoring() {
        if emergencySettings.enableVitalSignsMonitoring {
            startVitalSignsMonitoring()
        }
    }
    
    private func requestPermissions() {
        requestNotificationPermissions()
        requestLocationPermission()
        requestHealthKitPermissions()
    }
    
    private func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound, .criticalAlert]) { granted, error in
            if let error = error {
                print("Notification permission error: \(error.localizedDescription)")
            }
        }
    }
    
    private func requestLocationPermission() {
        switch locationManager.authorizationStatus {
        case .notDetermined:
            locationManager.requestAlwaysAuthorization()
        case .denied, .restricted:
            // Show settings alert
            break
        case .authorizedWhenInUse:
            locationManager.requestAlwaysAuthorization()
        case .authorizedAlways:
            startLocationTracking()
        @unknown default:
            break
        }
    }
    
    private func requestHealthKitPermissions() {
        healthKitManager.requestAuthorization { success in
            if success {
                print("HealthKit permissions granted")
            }
        }
    }
    
    // MARK: - Emergency Contact Management
    func addEmergencyContact(_ contact: EmergencyContact) {
        emergencyContacts.append(contact)
        saveEmergencyContacts()
        
        // Validate contact information
        validateEmergencyContact(contact)
    }
    
    func updateEmergencyContact(_ contact: EmergencyContact) {
        if let index = emergencyContacts.firstIndex(where: { $0.id == contact.id }) {
            emergencyContacts[index] = contact
            saveEmergencyContacts()
            validateEmergencyContact(contact)
        }
    }
    
    func removeEmergencyContact(_ contactId: UUID) {
        emergencyContacts.removeAll { $0.id == contactId }
        saveEmergencyContacts()
    }
    
    private func validateEmergencyContact(_ contact: EmergencyContact) {
        // Validate phone number format
        let phoneRegex = "^[+]?[0-9]{10,15}$"
        let phonePredicate = NSPredicate(format: "SELF MATCHES %@", phoneRegex)
        
        if !phonePredicate.evaluate(with: contact.phoneNumber) {
            let insight = EmergencyInsight(
                id: UUID(),
                type: .contactValidation,
                title: "Invalid Emergency Contact",
                description: "Phone number for \(contact.name) appears to be invalid",
                severity: .medium,
                timestamp: Date()
            )
            emergencyInsights.append(insight)
        }
    }
    
    // MARK: - Emergency Detection and Response
    func triggerEmergency(_ type: EmergencyType, severity: EmergencyLevel = .high, location: CLLocation? = nil, context: String? = nil) {
        let emergency = EmergencyEvent(
            id: UUID(),
            type: type,
            severity: severity,
            timestamp: Date(),
            location: location ?? currentLocation,
            context: context,
            status: .active,
            responseTime: nil,
            resolvedTime: nil
        )
        
        emergencyHistory.append(emergency)
        saveEmergencyHistory()
        
        isEmergencyActive = true
        currentEmergencyLevel = severity
        
        // Create emergency alert
        let alert = EmergencyAlert(
            id: UUID(),
            emergencyId: emergency.id,
            type: type,
            severity: severity,
            message: generateEmergencyMessage(type, severity: severity, context: context),
            timestamp: Date(),
            location: emergency.location,
            isActive: true,
            responseReceived: false
        )
        
        emergencyAlerts.append(alert)
        
        // Execute emergency response protocol
        executeEmergencyResponse(emergency, alert: alert)
        
        // Start emergency timer
        startEmergencyTimer(emergency)
        
        // Log emergency event
        logEmergencyEvent(emergency)
    }
    
    private func executeEmergencyResponse(_ emergency: EmergencyEvent, alert: EmergencyAlert) {
        // 1. Send immediate notifications
        sendEmergencyNotifications(alert)
        
        // 2. Contact emergency contacts
        contactEmergencyContacts(alert)
        
        // 3. Share location if available
        if let location = emergency.location {
            shareEmergencyLocation(location, alert: alert)
        }
        
        // 4. Call emergency services if critical
        if emergency.severity == .critical && emergencySettings.autoCallEmergencyServices {
            callEmergencyServices(alert)
        }
        
        // 5. Send health data to emergency contacts
        if emergencySettings.shareHealthDataInEmergency {
            shareEmergencyHealthData(alert)
        }
        
        // 6. Activate emergency mode on connected devices
        activateEmergencyModeOnDevices(alert)
        
        // 7. Start continuous monitoring
        startEmergencyMonitoring(emergency)
    }
    
    private func sendEmergencyNotifications(_ alert: EmergencyAlert) {
        // Send critical alert notification
        notificationManager.sendCriticalAlert(alert)
        
        // Send to Apple Watch if connected
        wearableIntegration.sendEmergencyAlert(alert)
        
        // Send push notifications to emergency contacts
        for contact in emergencyContacts.filter({ $0.notificationEnabled }) {
            notificationManager.sendEmergencyNotificationToContact(contact, alert: alert)
        }
    }
    
    private func contactEmergencyContacts(_ alert: EmergencyAlert) {
        let priorityContacts = emergencyContacts.filter { $0.priority == .primary }
        let secondaryContacts = emergencyContacts.filter { $0.priority == .secondary }
        
        // Contact primary contacts immediately
        for contact in priorityContacts {
            communicationManager.contactEmergencyContact(contact, alert: alert)
        }
        
        // Contact secondary contacts after delay if no response
        DispatchQueue.main.asyncAfter(deadline: .now() + 300) { // 5 minutes
            if self.isEmergencyActive && !alert.responseReceived {
                for contact in secondaryContacts {
                    self.communicationManager.contactEmergencyContact(contact, alert: alert)
                }
            }
        }
    }
    
    private func shareEmergencyLocation(_ location: CLLocation, alert: EmergencyAlert) {
        let locationMessage = "Emergency location: https://maps.apple.com/?ll=\(location.coordinate.latitude),\(location.coordinate.longitude)"
        
        for contact in emergencyContacts {
            communicationManager.sendLocationMessage(to: contact, message: locationMessage, alert: alert)
        }
    }
    
    private func callEmergencyServices(_ alert: EmergencyAlert) {
        guard let emergencyNumber = emergencySettings.emergencyServiceNumber else { return }
        
        let handle = CXHandle(type: .phoneNumber, value: emergencyNumber)
        let startCallAction = CXStartCallAction(call: UUID(), handle: handle)
        
        let transaction = CXTransaction(action: startCallAction)
        
        emergencyCallManager?.request(transaction) { error in
            if let error = error {
                print("Emergency call failed: \(error.localizedDescription)")
                // Fallback to direct URL call
                if let url = URL(string: "tel://\(emergencyNumber)") {
                    DispatchQueue.main.async {
                        UIApplication.shared.open(url)
                    }
                }
            }
        }
    }
    
    private func shareEmergencyHealthData(_ alert: EmergencyAlert) {
        healthKitManager.generateEmergencyHealthReport { [weak self] report in
            guard let report = report else { return }
            
            for contact in self?.emergencyContacts.filter({ $0.canReceiveHealthData }) ?? [] {
                self?.communicationManager.sendHealthReport(to: contact, report: report, alert: alert)
            }
        }
    }
    
    private func activateEmergencyModeOnDevices(_ alert: EmergencyAlert) {
        // Activate emergency mode on Apple Watch
        wearableIntegration.activateEmergencyMode(alert)
        
        // Increase screen brightness
        UIScreen.main.brightness = 1.0
        
        // Disable auto-lock
        UIApplication.shared.isIdleTimerDisabled = true
        
        // Start location tracking
        if locationPermissionStatus == .authorizedAlways {
            startContinuousLocationTracking()
        }
    }
    
    private func startEmergencyMonitoring(_ emergency: EmergencyEvent) {
        // Start continuous vital signs monitoring
        vitalSignsMonitor.startEmergencyMonitoring()
        
        // Start fall detection
        fallDetector.increasesensitivity()
        
        // Start location tracking
        startContinuousLocationTracking()
        
        // Monitor for emergency resolution
        emergencyTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.checkEmergencyStatus(emergency)
        }
    }
    
    private func startEmergencyTimer(_ emergency: EmergencyEvent) {
        // Auto-escalate if no response within time limit
        let escalationTime: TimeInterval = emergency.severity == .critical ? 300 : 900 // 5 or 15 minutes
        
        DispatchQueue.main.asyncAfter(deadline: .now() + escalationTime) { [weak self] in
            if self?.isEmergencyActive == true {
                self?.escalateEmergency(emergency)
            }
        }
    }
    
    private func escalateEmergency(_ emergency: EmergencyEvent) {
        // Escalate emergency level
        let escalatedSeverity: EmergencyLevel = emergency.severity == .high ? .critical : emergency.severity
        
        if escalatedSeverity != emergency.severity {
            triggerEmergency(emergency.type, severity: escalatedSeverity, location: emergency.location, context: "Escalated: No response received")
        }
        
        // Contact additional emergency contacts
        let tertiaryContacts = emergencyContacts.filter { $0.priority == .tertiary }
        
        for contact in tertiaryContacts {
            if let alert = emergencyAlerts.first(where: { $0.emergencyId == emergency.id }) {
                communicationManager.contactEmergencyContact(contact, alert: alert)
            }
        }
        
        // Call emergency services if not already done
        if !emergencySettings.autoCallEmergencyServices {
            if let alert = emergencyAlerts.first(where: { $0.emergencyId == emergency.id }) {
                callEmergencyServices(alert)
            }
        }
    }
    
    // MARK: - Emergency Resolution
    func resolveEmergency(_ emergencyId: UUID, resolution: EmergencyResolution) {
        if let index = emergencyHistory.firstIndex(where: { $0.id == emergencyId }) {
            emergencyHistory[index].status = .resolved
            emergencyHistory[index].resolvedTime = Date()
            emergencyHistory[index].resolution = resolution
            saveEmergencyHistory()
        }
        
        // Deactivate emergency alerts
        for i in emergencyAlerts.indices {
            if emergencyAlerts[i].emergencyId == emergencyId {
                emergencyAlerts[i].isActive = false
            }
        }
        
        // Check if all emergencies are resolved
        let activeEmergencies = emergencyHistory.filter { $0.status == .active }
        if activeEmergencies.isEmpty {
            deactivateEmergencyMode()
        }
        
        // Send resolution notifications
        sendResolutionNotifications(emergencyId, resolution: resolution)
        
        // Generate post-emergency insights
        generatePostEmergencyInsights(emergencyId)
    }
    
    private func deactivateEmergencyMode() {
        isEmergencyActive = false
        currentEmergencyLevel = .none
        
        // Stop emergency monitoring
        emergencyTimer?.invalidate()
        emergencyTimer = nil
        
        // Restore normal device settings
        UIApplication.shared.isIdleTimerDisabled = false
        
        // Stop continuous location tracking
        stopContinuousLocationTracking()
        
        // Deactivate emergency mode on devices
        wearableIntegration.deactivateEmergencyMode()
        
        // Return vital signs monitoring to normal
        vitalSignsMonitor.stopEmergencyMonitoring()
        
        // Return fall detection to normal sensitivity
        fallDetector.resetSensitivity()
    }
    
    private func sendResolutionNotifications(_ emergencyId: UUID, resolution: EmergencyResolution) {
        let message = "Emergency resolved: \(resolution.description)"
        
        for contact in emergencyContacts {
            communicationManager.sendResolutionMessage(to: contact, message: message)
        }
        
        // Send notification to user
        notificationManager.sendResolutionNotification(emergencyId, resolution: resolution)
    }
    
    // MARK: - Vital Signs Monitoring
    private func startVitalSignsMonitoring() {
        vitalSignsTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.checkVitalSigns()
        }
    }
    
    private func checkVitalSigns() {
        vitalSignsMonitor.getCurrentVitalSigns { [weak self] vitalSigns in
            self?.analyzeVitalSigns(vitalSigns)
        }
    }
    
    private func analyzeVitalSigns(_ vitalSigns: VitalSigns) {
        // Update vital signs status
        vitalSignsStatus = VitalSignsStatus(
            heartRate: vitalSigns.heartRate,
            bloodPressure: vitalSigns.bloodPressure,
            oxygenSaturation: vitalSigns.oxygenSaturation,
            respiratoryRate: vitalSigns.respiratoryRate,
            bodyTemperature: vitalSigns.bodyTemperature,
            lastUpdated: Date(),
            status: .normal
        )
        
        // Check for emergency conditions
        let emergencyConditions = checkForEmergencyConditions(vitalSigns)
        
        for condition in emergencyConditions {
            triggerEmergency(.medicalEmergency, severity: condition.severity, context: condition.description)
        }
        
        // Use ML to predict potential emergencies
        mlPredictor.predictEmergencyRisk(vitalSigns: vitalSigns) { [weak self] prediction in
            if prediction.riskLevel > 0.8 {
                self?.triggerEmergency(.predictedEmergency, severity: .medium, context: "High emergency risk predicted: \(prediction.description)")
            }
        }
    }
    
    private func checkForEmergencyConditions(_ vitalSigns: VitalSigns) -> [EmergencyCondition] {
        var conditions: [EmergencyCondition] = []
        
        // Heart rate checks
        if let heartRate = vitalSigns.heartRate {
            if heartRate < 50 || heartRate > 120 {
                conditions.append(EmergencyCondition(
                    type: .abnormalHeartRate,
                    severity: heartRate < 40 || heartRate > 150 ? .critical : .high,
                    description: "Heart rate: \(heartRate) BPM"
                ))
            }
        }
        
        // Blood pressure checks
        if let systolic = vitalSigns.bloodPressure?.systolic,
           let diastolic = vitalSigns.bloodPressure?.diastolic {
            if systolic > 180 || diastolic > 110 {
                conditions.append(EmergencyCondition(
                    type: .hypertensiveCrisis,
                    severity: .critical,
                    description: "Blood pressure: \(systolic)/\(diastolic) mmHg"
                ))
            }
        }
        
        // Oxygen saturation checks
        if let oxygenSat = vitalSigns.oxygenSaturation {
            if oxygenSat < 90 {
                conditions.append(EmergencyCondition(
                    type: .lowOxygenSaturation,
                    severity: oxygenSat < 85 ? .critical : .high,
                    description: "Oxygen saturation: \(oxygenSat)%"
                ))
            }
        }
        
        return conditions
    }
    
    // MARK: - Location Services
    private func startLocationTracking() {
        guard locationPermissionStatus == .authorizedAlways else { return }
        
        locationManager.startUpdatingLocation()
        
        if emergencySettings.enableGeofencing {
            setupGeofences()
        }
    }
    
    private func startContinuousLocationTracking() {
        guard locationPermissionStatus == .authorizedAlways else { return }
        
        locationManager.startUpdatingLocation()
        locationManager.startMonitoringSignificantLocationChanges()
        
        // Update location every minute during emergency
        locationTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.updateEmergencyLocation()
        }
    }
    
    private func stopContinuousLocationTracking() {
        locationManager.stopUpdatingLocation()
        locationManager.stopMonitoringSignificantLocationChanges()
        
        locationTimer?.invalidate()
        locationTimer = nil
    }
    
    private func updateEmergencyLocation() {
        guard let location = currentLocation else { return }
        
        // Update active emergency alerts with current location
        for i in emergencyAlerts.indices {
            if emergencyAlerts[i].isActive {
                emergencyAlerts[i].location = location
            }
        }
        
        // Share updated location with emergency contacts
        if isEmergencyActive {
            shareEmergencyLocation(location, alert: emergencyAlerts.first { $0.isActive }!)
        }
    }
    
    private func setupGeofences() {
        geofenceManager.setupSafeZones(emergencySettings.safeZones) { [weak self] success in
            if success {
                self?.geofenceManager.startMonitoring()
            }
        }
    }
    
    // MARK: - Fall Detection
    private func handleFallDetection(_ notification: Notification) {
        guard let fallData = notification.object as? FallDetectionData else { return }
        
        // Start fall detection countdown
        startFallDetectionCountdown(fallData)
    }
    
    private func startFallDetectionCountdown(_ fallData: FallDetectionData) {
        let alert = UIAlertController(
            title: "Fall Detected",
            message: "A fall has been detected. Emergency services will be contacted in 60 seconds unless you respond.",
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "I'm OK", style: .default) { _ in
            // Cancel emergency response
        })
        
        alert.addAction(UIAlertAction(title: "Call Emergency Services", style: .destructive) { [weak self] _ in
            self?.triggerEmergency(.fall, severity: .critical, context: "Fall detected with user confirmation")
        })
        
        // Auto-trigger emergency after 60 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 60) { [weak self] in
            if alert.presentingViewController != nil {
                alert.dismiss(animated: true) {
                    self?.triggerEmergency(.fall, severity: .high, context: "Fall detected - no user response")
                }
            }
        }
        
        // Present alert
        DispatchQueue.main.async {
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let window = windowScene.windows.first {
                window.rootViewController?.present(alert, animated: true)
            }
        }
    }
    
    // MARK: - Settings Management
    func updateEmergencySettings(_ settings: EmergencySettings) {
        emergencySettings = settings
        saveEmergencySettings()
        
        // Apply settings changes
        applySettingsChanges(settings)
    }
    
    private func applySettingsChanges(_ settings: EmergencySettings) {
        // Update fall detection
        if settings.enableFallDetection {
            fallDetector.startMonitoring()
        } else {
            fallDetector.stopMonitoring()
        }
        
        // Update vital signs monitoring
        if settings.enableVitalSignsMonitoring {
            startVitalSignsMonitoring()
        } else {
            vitalSignsTimer?.invalidate()
            vitalSignsTimer = nil
        }
        
        // Update location tracking
        if settings.enableLocationTracking {
            requestLocationPermission()
        } else {
            locationManager.stopUpdatingLocation()
        }
        
        // Update geofencing
        if settings.enableGeofencing {
            setupGeofences()
        } else {
            geofenceManager.stopMonitoring()
        }
    }
    
    // MARK: - Insights and Analytics
    private func generatePostEmergencyInsights(_ emergencyId: UUID) {
        guard let emergency = emergencyHistory.first(where: { $0.id == emergencyId }) else { return }
        
        let responseTime = emergency.responseTime ?? 0
        let resolutionTime = emergency.resolvedTime?.timeIntervalSince(emergency.timestamp) ?? 0
        
        // Generate response time insight
        if responseTime > 300 { // More than 5 minutes
            let insight = EmergencyInsight(
                id: UUID(),
                type: .responseTime,
                title: "Slow Emergency Response",
                description: "Emergency response took \(Int(responseTime/60)) minutes. Consider reviewing emergency contacts.",
                severity: .medium,
                timestamp: Date()
            )
            emergencyInsights.append(insight)
        }
        
        // Generate pattern analysis
        analyzeEmergencyPatterns()
    }
    
    private func analyzeEmergencyPatterns() {
        let recentEmergencies = emergencyHistory.filter {
            $0.timestamp >= Calendar.current.date(byAdding: .month, value: -3, to: Date()) ?? Date()
        }
        
        if recentEmergencies.count >= 3 {
            let insight = EmergencyInsight(
                id: UUID(),
                type: .pattern,
                title: "Frequent Emergencies",
                description: "\(recentEmergencies.count) emergencies in the last 3 months. Consider consulting with healthcare provider.",
                severity: .high,
                timestamp: Date()
            )
            emergencyInsights.append(insight)
        }
    }
    
    // MARK: - Helper Methods
    private func generateEmergencyMessage(_ type: EmergencyType, severity: EmergencyLevel, context: String?) -> String {
        var message = "EMERGENCY ALERT: \(type.description)"
        
        if let context = context {
            message += " - \(context)"
        }
        
        message += " Severity: \(severity.description)"
        
        if let location = currentLocation {
            message += " Location: https://maps.apple.com/?ll=\(location.coordinate.latitude),\(location.coordinate.longitude)"
        }
        
        return message
    }
    
    private func checkEmergencyStatus(_ emergency: EmergencyEvent) {
        // Check if emergency is still active and needs attention
        let timeSinceEmergency = Date().timeIntervalSince(emergency.timestamp)
        
        if timeSinceEmergency > 3600 && emergency.status == .active { // 1 hour
            // Auto-resolve if no activity
            resolveEmergency(emergency.id, resolution: EmergencyResolution(
                type: .autoResolved,
                description: "Auto-resolved after 1 hour of inactivity",
                timestamp: Date()
            ))
        }
    }
    
    private func logEmergencyEvent(_ emergency: EmergencyEvent) {
        // Log emergency event for analytics and reporting
        print("Emergency logged: \(emergency.type.description) at \(emergency.timestamp)")
    }
    
    // MARK: - Data Persistence
    private func loadEmergencyContacts() {
        if let data = UserDefaults.standard.data(forKey: "EmergencyContacts"),
           let contacts = try? JSONDecoder().decode([EmergencyContact].self, from: data) {
            emergencyContacts = contacts
        }
    }
    
    private func saveEmergencyContacts() {
        if let data = try? JSONEncoder().encode(emergencyContacts) {
            UserDefaults.standard.set(data, forKey: "EmergencyContacts")
        }
    }
    
    private func loadEmergencyHistory() {
        if let data = UserDefaults.standard.data(forKey: "EmergencyHistory"),
           let history = try? JSONDecoder().decode([EmergencyEvent].self, from: data) {
            emergencyHistory = history
        }
    }
    
    private func saveEmergencyHistory() {
        if let data = try? JSONEncoder().encode(emergencyHistory) {
            UserDefaults.standard.set(data, forKey: "EmergencyHistory")
        }
    }
    
    private func loadEmergencySettings() {
        if let data = UserDefaults.standard.data(forKey: "EmergencySettings"),
           let settings = try? JSONDecoder().decode(EmergencySettings.self, from: data) {
            emergencySettings = settings
        }
    }
    
    private func saveEmergencySettings() {
        if let data = try? JSONEncoder().encode(emergencySettings) {
            UserDefaults.standard.set(data, forKey: "EmergencySettings")
        }
    }
    
    // MARK: - Cleanup
    deinit {
        emergencyTimer?.invalidate()
        vitalSignsTimer?.invalidate()
        locationTimer?.invalidate()
        cancellables.removeAll()
    }
}

// MARK: - Delegate Extensions
extension EmergencyResponseSystem: HealthKitEmergencyManagerDelegate {
    func healthKitManager(_ manager: HealthKitEmergencyManager, didDetectAbnormalReading reading: HealthKitReading) {
        triggerEmergency(.medicalEmergency, severity: reading.severity, context: reading.description)
    }
}

extension EmergencyResponseSystem: FallDetectionEngineDelegate {
    func fallDetectionEngine(_ engine: FallDetectionEngine, didDetectFall fallData: FallDetectionData) {
        NotificationCenter.default.post(name: .fallDetected, object: fallData)
    }
}

extension EmergencyResponseSystem: EmergencyVitalSignsMonitorDelegate {
    func vitalSignsMonitor(_ monitor: EmergencyVitalSignsMonitor, didDetectAbnormalVitalSigns vitalSigns: VitalSigns) {
        NotificationCenter.default.post(name: .vitalSignsAlert, object: vitalSigns)
    }
}

extension EmergencyResponseSystem: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        currentLocation = locations.last
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        locationPermissionStatus = status
        
        switch status {
        case .authorizedAlways:
            startLocationTracking()
        case .denied, .restricted:
            // Handle permission denied
            break
        default:
            break
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didEnterRegion region: CLRegion) {
        // Handle entering safe zone
        geofenceManager.handleRegionEntry(region)
    }
    
    func locationManager(_ manager: CLLocationManager, didExitRegion region: CLRegion) {
        // Handle exiting safe zone
        geofenceManager.handleRegionExit(region)
    }
}

extension EmergencyResponseSystem: EmergencyNotificationManagerDelegate {
    func notificationManager(_ manager: EmergencyNotificationManager, didReceiveResponse response: UNNotificationResponse) {
        // Handle emergency notification responses
        let userInfo = response.notification.request.content.userInfo
        
        if let emergencyId = userInfo["emergencyId"] as? String,
           let uuid = UUID(uuidString: emergencyId) {
            
            switch response.actionIdentifier {
            case "RESOLVE_ACTION":
                resolveEmergency(uuid, resolution: EmergencyResolution(
                    type: .userResolved,
                    description: "Resolved by user",
                    timestamp: Date()
                ))
            case "ESCALATE_ACTION":
                if let emergency = emergencyHistory.first(where: { $0.id == uuid }) {
                    escalateEmergency(emergency)
                }
            default:
                break
            }
        }
    }
}

// MARK: - Supporting Classes
class HealthKitEmergencyManager {
    weak var delegate: HealthKitEmergencyManagerDelegate?
    
    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        // Request HealthKit authorization for emergency monitoring
        completion(true)
    }
    
    func generateEmergencyHealthReport(completion: @escaping (EmergencyHealthReport?) -> Void) {
        // Generate comprehensive health report for emergency
        completion(nil)
    }
}

protocol HealthKitEmergencyManagerDelegate: AnyObject {
    func healthKitManager(_ manager: HealthKitEmergencyManager, didDetectAbnormalReading reading: HealthKitReading)
}

class EmergencyNotificationManager {
    weak var delegate: EmergencyNotificationManagerDelegate?
    
    func sendCriticalAlert(_ alert: EmergencyAlert) {
        let content = UNMutableNotificationContent()
        content.title = "EMERGENCY ALERT"
        content.body = alert.message
        content.sound = .defaultCritical
        content.interruptionLevel = .critical
        content.userInfo = ["emergencyId": alert.emergencyId.uuidString]
        
        // Add action buttons
        let resolveAction = UNNotificationAction(identifier: "RESOLVE_ACTION", title: "I'm Safe", options: [])
        let escalateAction = UNNotificationAction(identifier: "ESCALATE_ACTION", title: "Need Help", options: [])
        
        let category = UNNotificationCategory(identifier: "EMERGENCY_ALERT",
                                            actions: [resolveAction, escalateAction],
                                            intentIdentifiers: [],
                                            options: [])
        
        UNUserNotificationCenter.current().setNotificationCategories([category])
        content.categoryIdentifier = "EMERGENCY_ALERT"
        
        let request = UNNotificationRequest(identifier: "emergency_\(alert.id.uuidString)",
                                          content: content,
                                          trigger: nil)
        
        UNUserNotificationCenter.current().add(request)
    }
    
    func sendEmergencyNotificationToContact(_ contact: EmergencyContact, alert: EmergencyAlert) {
        // Send push notification to emergency contact's device
    }
    
    func sendResolutionNotification(_ emergencyId: UUID, resolution: EmergencyResolution) {
        // Send notification about emergency resolution
    }
}

protocol EmergencyNotificationManagerDelegate: AnyObject {
    func notificationManager(_ manager: EmergencyNotificationManager, didReceiveResponse response: UNNotificationResponse)
}

class EmergencyCommunicationManager {
    func contactEmergencyContact(_ contact: EmergencyContact, alert: EmergencyAlert) {
        // Contact emergency contact via phone, SMS, or push notification
        switch contact.preferredContactMethod {
        case .phone:
            makePhoneCall(to: contact.phoneNumber)
        case .sms:
            sendSMS(to: contact.phoneNumber, message: alert.message)
        case .push:
            sendPushNotification(to: contact, alert: alert)
        case .email:
            sendEmail(to: contact.email ?? "", message: alert.message)
        }
    }
    
    func sendLocationMessage(to contact: EmergencyContact, message: String, alert: EmergencyAlert) {
        sendSMS(to: contact.phoneNumber, message: message)
    }
    
    func sendHealthReport(to contact: EmergencyContact, report: EmergencyHealthReport, alert: EmergencyAlert) {
        // Send health report to emergency contact
    }
    
    func sendResolutionMessage(to contact: EmergencyContact, message: String) {
        sendSMS(to: contact.phoneNumber, message: message)
    }
    
    private func makePhoneCall(to phoneNumber: String) {
        if let url = URL(string: "tel://\(phoneNumber)") {
            DispatchQueue.main.async {
                UIApplication.shared.open(url)
            }
        }
    }
    
    private func sendSMS(to phoneNumber: String, message: String) {
        if let url = URL(string: "sms:\(phoneNumber)&body=\(message.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")") {
            DispatchQueue.main.async {
                UIApplication.shared.open(url)
            }
        }
    }
    
    private func sendPushNotification(to contact: EmergencyContact, alert: EmergencyAlert) {
        // Send push notification to contact's device
    }
    
    private func sendEmail(to email: String, message: String) {
        if let url = URL(string: "mailto:\(email)?subject=Emergency Alert&body=\(message.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")") {
            DispatchQueue.main.async {
                UIApplication.shared.open(url)
            }
        }
    }
}

class EmergencyMLPredictor {
    func predictEmergencyRisk(vitalSigns: VitalSigns, completion: @escaping (EmergencyRiskPrediction) -> Void) {
        // Use ML to predict emergency risk based on vital signs
        let prediction = EmergencyRiskPrediction(
            riskLevel: 0.1,
            confidence: 0.8,
            description: "Low emergency risk",
            factors: []
        )
        completion(prediction)
    }
}

class FallDetectionEngine {
    weak var delegate: FallDetectionEngineDelegate?
    
    func startMonitoring() {
        // Start fall detection monitoring
    }
    
    func stopMonitoring() {
        // Stop fall detection monitoring
    }
    
    func increasesensitivity() {
        // Increase fall detection sensitivity during emergency
    }
    
    func resetSensitivity() {
        // Reset fall detection sensitivity to normal
    }
}

protocol FallDetectionEngineDelegate: AnyObject {
    func fallDetectionEngine(_ engine: FallDetectionEngine, didDetectFall fallData: FallDetectionData)
}

class EmergencyVitalSignsMonitor {
    weak var delegate: EmergencyVitalSignsMonitorDelegate?
    
    func getCurrentVitalSigns(completion: @escaping (VitalSigns) -> Void) {
        // Get current vital signs from HealthKit or connected devices
        let vitalSigns = VitalSigns(
            heartRate: 75,
            bloodPressure: BloodPressure(systolic: 120, diastolic: 80),
            oxygenSaturation: 98,
            respiratoryRate: 16,
            bodyTemperature: 98.6
        )
        completion(vitalSigns)
    }
    
    func startEmergencyMonitoring() {
        // Start intensive vital signs monitoring during emergency
    }
    
    func stopEmergencyMonitoring() {
        // Stop emergency vital signs monitoring
    }
}

protocol EmergencyVitalSignsMonitorDelegate: AnyObject {
    func vitalSignsMonitor(_ monitor: EmergencyVitalSignsMonitor, didDetectAbnormalVitalSigns vitalSigns: VitalSigns)
}

class EmergencyGeofenceManager {
    func setupSafeZones(_ safeZones: [SafeZone], completion: @escaping (Bool) -> Void) {
        // Setup geofences for safe zones
        completion(true)
    }
    
    func startMonitoring() {
        // Start monitoring geofences
    }
    
    func stopMonitoring() {
        // Stop monitoring geofences
    }
    
    func handleRegionEntry(_ region: CLRegion) {
        // Handle entering a safe zone
    }
    
    func handleRegionExit(_ region: CLRegion) {
        // Handle exiting a safe zone
    }
}

class EmergencyWearableIntegration {
    func sendEmergencyAlert(_ alert: EmergencyAlert) {
        // Send emergency alert to Apple Watch
    }
    
    func activateEmergencyMode(_ alert: EmergencyAlert) {
        // Activate emergency mode on Apple Watch
    }
    
    func deactivateEmergencyMode() {
        // Deactivate emergency mode on Apple Watch
    }
}

// MARK: - Data Types
struct EmergencyContact: Codable, Identifiable {
    let id: UUID
    let name: String
    let phoneNumber: String
    let email: String?
    let relationship: String
    let priority: ContactPriority
    let preferredContactMethod: ContactMethod
    let notificationEnabled: Bool
    let canReceiveHealthData: Bool
    let isHealthcareProvider: Bool
}

struct EmergencyAlert: Codable, Identifiable {
    let id: UUID
    let emergencyId: UUID
    let type: EmergencyType
    let severity: EmergencyLevel
    let message: String
    let timestamp: Date
    var location: CLLocation?
    var isActive: Bool
    var responseReceived: Bool
}

struct EmergencyEvent: Codable, Identifiable {
    let id: UUID
    let type: EmergencyType
    let severity: EmergencyLevel
    let timestamp: Date
    let location: CLLocation?
    let context: String?
    var status: EmergencyStatus
    var responseTime: TimeInterval?
    var resolvedTime: Date?
    var resolution: EmergencyResolution?
}

struct EmergencySettings: Codable {
    var enableFallDetection: Bool = true
    var enableVitalSignsMonitoring: Bool = true
    var enableLocationTracking: Bool = true
    var enableGeofencing: Bool = false
    var autoCallEmergencyServices: Bool = false
    var shareHealthDataInEmergency: Bool = true
    var emergencyServiceNumber: String? = "911"
    var safeZones: [SafeZone] = []
    var emergencyEscalationTime: TimeInterval = 900 // 15 minutes
}

struct VitalSignsStatus: Codable {
    var heartRate: Double?
    var bloodPressure: BloodPressure?
    var oxygenSaturation: Double?
    var respiratoryRate: Double?
    var bodyTemperature: Double?
    var lastUpdated: Date?
    var status: VitalSignsHealthStatus
    
    init() {
        self.status = .normal
    }
    
    init(heartRate: Double?, bloodPressure: BloodPressure?, oxygenSaturation: Double?, respiratoryRate: Double?, bodyTemperature: Double?, lastUpdated: Date?, status: VitalSignsHealthStatus) {
        self.heartRate = heartRate
        self.bloodPressure = bloodPressure
        self.oxygenSaturation = oxygenSaturation
        self.respiratoryRate = respiratoryRate
        self.bodyTemperature = bodyTemperature
        self.lastUpdated = lastUpdated
        self.status = status
    }
}

struct EmergencyInsight: Codable, Identifiable {
    let id: UUID
    let type: EmergencyInsightType
    let title: String
    let description: String
    let severity: InsightSeverity
    let timestamp: Date
}

struct VitalSigns: Codable {
    let heartRate: Double?
    let bloodPressure: BloodPressure?
    let oxygenSaturation: Double?
    let respiratoryRate: Double?
    let bodyTemperature: Double?
}

struct BloodPressure: Codable {
    let systolic: Double
    let diastolic: Double
}

struct EmergencyCondition: Codable {
    let type: EmergencyConditionType
    let severity: EmergencyLevel
    let description: String
}

struct EmergencyResolution: Codable {
    let type: ResolutionType
    let description: String
    let timestamp: Date
}

struct FallDetectionData: Codable {
    let timestamp: Date
    let confidence: Double
    let impactForce: Double
    let location: CLLocation?
}

struct HealthKitReading: Codable {
    let type: String
    let value: Double
    let timestamp: Date
    let severity: EmergencyLevel
    let description: String
}

struct EmergencyHealthReport: Codable {
    let patientId: UUID
    let timestamp: Date
    let vitalSigns: VitalSigns
    let medications: [String]
    let allergies: [String]
    let medicalConditions: [String]
    let emergencyContacts: [EmergencyContact]
}

struct EmergencyRiskPrediction: Codable {
    let riskLevel: Double
    let confidence: Double
    let description: String
    let factors: [String]
}

struct SafeZone: Codable, Identifiable {
    let id: UUID
    let name: String
    let center: CLLocationCoordinate2D
    let radius: Double
    let isActive: Bool
}

// MARK: - Enums
enum EmergencyType: String, Codable, CaseIterable {
    case medicalEmergency = "medical_emergency"
    case fall = "fall"
    case heartAttack = "heart_attack"
    case stroke = "stroke"
    case seizure = "seizure"
    case severeAllergicReaction = "severe_allergic_reaction"
    case respiratoryDistress = "respiratory_distress"
    case unconsciousness = "unconsciousness"
    case severeInjury = "severe_injury"
    case predictedEmergency = "predicted_emergency"
    case panicAttack = "panic_attack"
    case other = "other"
    
    var description: String {
        switch self {
        case .medicalEmergency: return "Medical Emergency"
        case .fall: return "Fall Detected"
        case .heartAttack: return "Heart Attack"
        case .stroke: return "Stroke"
        case .seizure: return "Seizure"
        case .severeAllergicReaction: return "Severe Allergic Reaction"
        case .respiratoryDistress: return "Respiratory Distress"
        case .unconsciousness: return "Unconsciousness"
        case .severeInjury: return "Severe Injury"
        case .predictedEmergency: return "Predicted Emergency"
        case .panicAttack: return "Panic Attack"
        case .other: return "Other Emergency"
        }
    }
}

enum EmergencyLevel: String, Codable, CaseIterable {
    case none = "none"
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
    
    var description: String {
        switch self {
        case .none: return "None"
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .critical: return "Critical"
        }
    }
}

enum EmergencyStatus: String, Codable {
    case active = "active"
    case responding = "responding"
    case resolved = "resolved"
    case cancelled = "cancelled"
}

enum ContactPriority: String, Codable {
    case primary = "primary"
    case secondary = "secondary"
    case tertiary = "tertiary"
}

enum ContactMethod: String, Codable {
    case phone = "phone"
    case sms = "sms"
    case email = "email"
    case push = "push"
}

enum VitalSignsHealthStatus: String, Codable {
    case normal = "normal"
    case warning = "warning"
    case critical = "critical"
    case unknown = "unknown"
}

enum EmergencyInsightType: String, Codable {
    case contactValidation = "contact_validation"
    case responseTime = "response_time"
    case pattern = "pattern"
    case prediction = "prediction"
    case optimization = "optimization"
}

enum InsightSeverity: String, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
}

enum EmergencyConditionType: String, Codable {
    case abnormalHeartRate = "abnormal_heart_rate"
    case hypertensiveCrisis = "hypertensive_crisis"
    case lowOxygenSaturation = "low_oxygen_saturation"
    case abnormalRespiratoryRate = "abnormal_respiratory_rate"
    case abnormalBodyTemperature = "abnormal_body_temperature"
}

enum ResolutionType: String, Codable {
    case userResolved = "user_resolved"
    case emergencyServicesResolved = "emergency_services_resolved"
    case contactResolved = "contact_resolved"
    case autoResolved = "auto_resolved"
    case falseAlarm = "false_alarm"
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let emergencyTriggered = Notification.Name("emergencyTriggered")
    static let vitalSignsAlert = Notification.Name("vitalSignsAlert")
    static let fallDetected = Notification.Name("fallDetected")
}

// MARK: - CLLocation Codable Extension
extension CLLocation: @unchecked Sendable {}

extension CLLocation {
    private enum CodingKeys: String, CodingKey {
        case latitude, longitude, altitude, horizontalAccuracy, verticalAccuracy, timestamp
    }
}

extension CLLocation: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(coordinate.latitude, forKey: .latitude)
        try container.encode(coordinate.longitude, forKey: .longitude)
        try container.encode(altitude, forKey: .altitude)
        try container.encode(horizontalAccuracy, forKey: .horizontalAccuracy)
        try container.encode(verticalAccuracy, forKey: .verticalAccuracy)
        try container.encode(timestamp, forKey: .timestamp)
    }
    
    public convenience init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let latitude = try container.decode(Double.self, forKey: .latitude)
        let longitude = try container.decode(Double.self, forKey: .longitude)
        let altitude = try container.decode(Double.self, forKey: .altitude)
        let horizontalAccuracy = try container.decode(Double.self, forKey: .horizontalAccuracy)
        let verticalAccuracy = try container.decode(Double.self, forKey: .verticalAccuracy)
        let timestamp = try container.decode(Date.self, forKey: .timestamp)
        
        self.init(
            coordinate: CLLocationCoordinate2D(latitude: latitude, longitude: longitude),
            altitude: altitude,
            horizontalAccuracy: horizontalAccuracy,
            verticalAccuracy: verticalAccuracy,
            timestamp: timestamp
        )
    }
}

extension CLLocationCoordinate2D: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(latitude)
        try container.encode(longitude)
    }
    
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        let latitude = try container.decode(Double.self)
        let longitude = try container.decode(Double.self)
        self.init(latitude: latitude, longitude: longitude)
    }
}