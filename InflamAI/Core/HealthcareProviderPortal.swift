//
//  HealthcareProviderPortal.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Combine
import Foundation
import HealthKit
import CryptoKit

// MARK: - Healthcare Provider Portal Manager
class HealthcareProviderPortalManager: ObservableObject {
    @Published var connectedProviders: [HealthcareProvider] = []
    @Published var pendingInvitations: [ProviderInvitation] = []
    @Published var sharedReports: [SharedHealthReport] = []
    @Published var providerMessages: [ProviderMessage] = []
    @Published var appointmentRequests: [AppointmentRequest] = []
    @Published var treatmentPlans: [TreatmentPlan] = []
    @Published var connectionStatus: PortalConnectionStatus = .disconnected
    @Published var dataSharePermissions: DataSharePermissions?
    @Published var emergencyContacts: [EmergencyContact] = []
    
    private let networkManager = HealthcareNetworkManager()
    private let encryptionManager = HealthcareEncryptionManager()
    private let reportGenerator = HealthReportGenerator()
    private let communicationManager = ProviderCommunicationManager()
    private let appointmentManager = AppointmentManager()
    private let treatmentPlanManager = TreatmentPlanManager()
    private let complianceManager = HIPAAComplianceManager()
    
    private var cancellables = Set<AnyCancellable>()
    private var syncTimer: Timer?
    
    init() {
        setupPortalConnection()
        loadStoredData()
        setupPeriodicSync()
        setupNotifications()
    }
    
    // MARK: - Setup
    private func setupPortalConnection() {
        networkManager.delegate = self
        communicationManager.delegate = self
        
        // Establish secure connection to healthcare portal
        establishSecureConnection()
    }
    
    private func loadStoredData() {
        loadConnectedProviders()
        loadSharedReports()
        loadDataSharePermissions()
        loadEmergencyContacts()
    }
    
    private func setupPeriodicSync() {
        // Sync with healthcare portal every 30 minutes
        syncTimer = Timer.scheduledTimer(withTimeInterval: 1800, repeats: true) { [weak self] _ in
            self?.syncWithPortal()
        }
    }
    
    private func setupNotifications() {
        // Setup notifications for provider messages and appointment updates
        NotificationCenter.default.publisher(for: .providerMessageReceived)
            .sink { [weak self] notification in
                self?.handleProviderMessage(notification)
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: .appointmentStatusChanged)
            .sink { [weak self] notification in
                self?.handleAppointmentUpdate(notification)
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Provider Connection
    func connectToProvider(with invitationCode: String) {
        connectionStatus = .connecting
        
        networkManager.validateInvitation(invitationCode) { [weak self] result in
            switch result {
            case .success(let invitation):
                self?.processProviderInvitation(invitation)
            case .failure(let error):
                self?.handleConnectionError(error)
            }
        }
    }
    
    private func processProviderInvitation(_ invitation: ProviderInvitation) {
        // Verify provider credentials and establish secure connection
        verifyProviderCredentials(invitation.provider) { [weak self] isValid in
            if isValid {
                self?.establishProviderConnection(invitation.provider)
            } else {
                self?.handleConnectionError(PortalError.invalidCredentials)
            }
        }
    }
    
    private func verifyProviderCredentials(_ provider: HealthcareProvider, completion: @escaping (Bool) -> Void) {
        // Verify provider license and credentials through healthcare registry
        networkManager.verifyProvider(provider) { result in
            switch result {
            case .success(let isValid):
                completion(isValid)
            case .failure:
                completion(false)
            }
        }
    }
    
    private func establishProviderConnection(_ provider: HealthcareProvider) {
        // Create secure encrypted connection
        let connectionKey = generateConnectionKey()
        
        networkManager.establishConnection(with: provider, key: connectionKey) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self?.connectedProviders.append(provider)
                    self?.connectionStatus = .connected
                    self?.setupDataSharing(with: provider)
                case .failure(let error):
                    self?.handleConnectionError(error)
                }
            }
        }
    }
    
    private func generateConnectionKey() -> String {
        let key = SymmetricKey(size: .bits256)
        return key.withUnsafeBytes { Data($0).base64EncodedString() }
    }
    
    // MARK: - Data Sharing
    private func setupDataSharing(with provider: HealthcareProvider) {
        // Configure data sharing permissions
        let permissions = DataSharePermissions(
            providerId: provider.id,
            allowedDataTypes: [.healthMetrics, .painTracking, .medications, .journalEntries],
            shareFrequency: .weekly,
            autoShare: true,
            emergencyAccess: true
        )
        
        dataSharePermissions = permissions
        saveDataSharePermissions()
    }
    
    func updateDataSharePermissions(_ permissions: DataSharePermissions) {
        dataSharePermissions = permissions
        saveDataSharePermissions()
        
        // Notify provider of permission changes
        notifyProviderOfPermissionChanges(permissions)
    }
    
    private func notifyProviderOfPermissionChanges(_ permissions: DataSharePermissions) {
        guard let provider = connectedProviders.first(where: { $0.id == permissions.providerId }) else { return }
        
        let notification = ProviderNotification(
            type: .permissionUpdate,
            providerId: provider.id,
            data: permissions,
            timestamp: Date()
        )
        
        communicationManager.sendNotification(notification, to: provider)
    }
    
    // MARK: - Health Report Generation
    func generateHealthReport(for provider: HealthcareProvider, timeRange: TimeRange) {
        reportGenerator.generateReport(
            providerId: provider.id,
            timeRange: timeRange,
            includeData: dataSharePermissions?.allowedDataTypes ?? []
        ) { [weak self] result in
            switch result {
            case .success(let report):
                self?.shareHealthReport(report, with: provider)
            case .failure(let error):
                self?.handleReportError(error)
            }
        }
    }
    
    private func shareHealthReport(_ report: SharedHealthReport, with provider: HealthcareProvider) {
        // Encrypt report before sharing
        encryptionManager.encryptReport(report) { [weak self] encryptedReport in
            guard let encryptedReport = encryptedReport else {
                self?.handleReportError(PortalError.encryptionFailed)
                return
            }
            
            self?.networkManager.shareReport(encryptedReport, with: provider) { result in
                DispatchQueue.main.async {
                    switch result {
                    case .success:
                        self?.sharedReports.append(report)
                        self?.notifyReportShared(report, provider: provider)
                    case .failure(let error):
                        self?.handleReportError(error)
                    }
                }
            }
        }
    }
    
    private func notifyReportShared(_ report: SharedHealthReport, provider: HealthcareProvider) {
        let message = ProviderMessage(
            id: UUID(),
            providerId: provider.id,
            type: .reportShared,
            subject: "Health Report Shared",
            content: "A new health report has been shared with you covering \(report.timeRange.description)",
            timestamp: Date(),
            isRead: false,
            attachments: [report.id.uuidString]
        )
        
        providerMessages.append(message)
    }
    
    // MARK: - Communication
    func sendMessageToProvider(_ message: String, to provider: HealthcareProvider, priority: MessagePriority = .normal) {
        let providerMessage = ProviderMessage(
            id: UUID(),
            providerId: provider.id,
            type: .patientMessage,
            subject: "Message from Patient",
            content: message,
            timestamp: Date(),
            isRead: false,
            priority: priority
        )
        
        communicationManager.sendMessage(providerMessage, to: provider) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self?.providerMessages.append(providerMessage)
                case .failure(let error):
                    self?.handleCommunicationError(error)
                }
            }
        }
    }
    
    func requestAppointment(with provider: HealthcareProvider, preferredDates: [Date], reason: String, urgency: AppointmentUrgency) {
        let request = AppointmentRequest(
            id: UUID(),
            providerId: provider.id,
            preferredDates: preferredDates,
            reason: reason,
            urgency: urgency,
            status: .pending,
            requestDate: Date()
        )
        
        appointmentManager.submitRequest(request, to: provider) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self?.appointmentRequests.append(request)
                case .failure(let error):
                    self?.handleAppointmentError(error)
                }
            }
        }
    }
    
    // MARK: - Treatment Plans
    func requestTreatmentPlan(from provider: HealthcareProvider, symptoms: [String], currentMedications: [String]) {
        let request = TreatmentPlanRequest(
            providerId: provider.id,
            symptoms: symptoms,
            currentMedications: currentMedications,
            requestDate: Date()
        )
        
        treatmentPlanManager.requestPlan(request, from: provider) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let plan):
                    self?.treatmentPlans.append(plan)
                case .failure(let error):
                    self?.handleTreatmentPlanError(error)
                }
            }
        }
    }
    
    func updateTreatmentPlanProgress(_ planId: UUID, progress: TreatmentProgress) {
        guard let planIndex = treatmentPlans.firstIndex(where: { $0.id == planId }) else { return }
        
        treatmentPlans[planIndex].progress = progress
        
        // Notify provider of progress update
        if let provider = connectedProviders.first(where: { $0.id == treatmentPlans[planIndex].providerId }) {
            let update = TreatmentProgressUpdate(
                planId: planId,
                progress: progress,
                timestamp: Date()
            )
            
            treatmentPlanManager.updateProgress(update, to: provider)
        }
    }
    
    // MARK: - Emergency Features
    func triggerEmergencyAlert() {
        let alert = EmergencyAlert(
            id: UUID(),
            patientId: getCurrentPatientId(),
            alertType: .medicalEmergency,
            location: getCurrentLocation(),
            timestamp: Date(),
            vitalSigns: getCurrentVitalSigns(),
            medications: getCurrentMedications()
        )
        
        // Send to all connected providers
        for provider in connectedProviders {
            communicationManager.sendEmergencyAlert(alert, to: provider)
        }
        
        // Send to emergency contacts
        for contact in emergencyContacts {
            communicationManager.sendEmergencyAlert(alert, to: contact)
        }
    }
    
    func addEmergencyContact(_ contact: EmergencyContact) {
        emergencyContacts.append(contact)
        saveEmergencyContacts()
    }
    
    func removeEmergencyContact(_ contactId: UUID) {
        emergencyContacts.removeAll { $0.id == contactId }
        saveEmergencyContacts()
    }
    
    // MARK: - Sync Operations
    private func syncWithPortal() {
        // Sync messages
        syncProviderMessages()
        
        // Sync appointment updates
        syncAppointmentUpdates()
        
        // Sync treatment plan updates
        syncTreatmentPlanUpdates()
        
        // Check for new invitations
        checkForNewInvitations()
    }
    
    private func syncProviderMessages() {
        for provider in connectedProviders {
            communicationManager.fetchNewMessages(from: provider) { [weak self] messages in
                DispatchQueue.main.async {
                    self?.providerMessages.append(contentsOf: messages)
                }
            }
        }
    }
    
    private func syncAppointmentUpdates() {
        appointmentManager.fetchUpdates { [weak self] updates in
            DispatchQueue.main.async {
                for update in updates {
                    self?.processAppointmentUpdate(update)
                }
            }
        }
    }
    
    private func syncTreatmentPlanUpdates() {
        treatmentPlanManager.fetchUpdates { [weak self] updates in
            DispatchQueue.main.async {
                for update in updates {
                    self?.processTreatmentPlanUpdate(update)
                }
            }
        }
    }
    
    private func checkForNewInvitations() {
        networkManager.fetchPendingInvitations { [weak self] invitations in
            DispatchQueue.main.async {
                self?.pendingInvitations = invitations
            }
        }
    }
    
    // MARK: - Data Processing
    private func processAppointmentUpdate(_ update: AppointmentUpdate) {
        if let index = appointmentRequests.firstIndex(where: { $0.id == update.requestId }) {
            appointmentRequests[index].status = update.status
            appointmentRequests[index].scheduledDate = update.scheduledDate
            appointmentRequests[index].notes = update.notes
        }
    }
    
    private func processTreatmentPlanUpdate(_ update: TreatmentPlanUpdate) {
        if let index = treatmentPlans.firstIndex(where: { $0.id == update.planId }) {
            treatmentPlans[index] = update.updatedPlan
        }
    }
    
    // MARK: - Error Handling
    private func handleConnectionError(_ error: Error) {
        connectionStatus = .failed
        print("Portal connection error: \(error.localizedDescription)")
    }
    
    private func handleReportError(_ error: Error) {
        print("Report generation error: \(error.localizedDescription)")
    }
    
    private func handleCommunicationError(_ error: Error) {
        print("Communication error: \(error.localizedDescription)")
    }
    
    private func handleAppointmentError(_ error: Error) {
        print("Appointment error: \(error.localizedDescription)")
    }
    
    private func handleTreatmentPlanError(_ error: Error) {
        print("Treatment plan error: \(error.localizedDescription)")
    }
    
    // MARK: - Notification Handlers
    private func handleProviderMessage(_ notification: Notification) {
        // Process incoming provider message
        if let message = notification.object as? ProviderMessage {
            providerMessages.append(message)
        }
    }
    
    private func handleAppointmentUpdate(_ notification: Notification) {
        // Process appointment status change
        if let update = notification.object as? AppointmentUpdate {
            processAppointmentUpdate(update)
        }
    }
    
    // MARK: - Helper Methods
    private func getCurrentPatientId() -> String {
        return UserDefaults.standard.string(forKey: "PatientID") ?? UUID().uuidString
    }
    
    private func getCurrentLocation() -> CLLocation? {
        // Get current location for emergency alerts
        return nil // Implementation depends on location services
    }
    
    private func getCurrentVitalSigns() -> VitalSigns? {
        // Get current vital signs for emergency alerts
        return nil // Implementation depends on health monitoring
    }
    
    private func getCurrentMedications() -> [String] {
        // Get current medications for emergency alerts
        return [] // Implementation depends on medication tracking
    }
    
    // MARK: - Data Persistence
    private func loadConnectedProviders() {
        if let data = UserDefaults.standard.data(forKey: "ConnectedProviders"),
           let providers = try? JSONDecoder().decode([HealthcareProvider].self, from: data) {
            connectedProviders = providers
        }
    }
    
    private func saveConnectedProviders() {
        if let data = try? JSONEncoder().encode(connectedProviders) {
            UserDefaults.standard.set(data, forKey: "ConnectedProviders")
        }
    }
    
    private func loadSharedReports() {
        if let data = UserDefaults.standard.data(forKey: "SharedReports"),
           let reports = try? JSONDecoder().decode([SharedHealthReport].self, from: data) {
            sharedReports = reports
        }
    }
    
    private func saveSharedReports() {
        if let data = try? JSONEncoder().encode(sharedReports) {
            UserDefaults.standard.set(data, forKey: "SharedReports")
        }
    }
    
    private func loadDataSharePermissions() {
        if let data = UserDefaults.standard.data(forKey: "DataSharePermissions"),
           let permissions = try? JSONDecoder().decode(DataSharePermissions.self, from: data) {
            dataSharePermissions = permissions
        }
    }
    
    private func saveDataSharePermissions() {
        if let data = try? JSONEncoder().encode(dataSharePermissions) {
            UserDefaults.standard.set(data, forKey: "DataSharePermissions")
        }
    }
    
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
    
    // MARK: - Security
    private func establishSecureConnection() {
        // Implement secure connection establishment
        complianceManager.ensureHIPAACompliance { [weak self] isCompliant in
            if isCompliant {
                self?.connectionStatus = .ready
            } else {
                self?.connectionStatus = .complianceError
            }
        }
    }
    
    // MARK: - Cleanup
    deinit {
        syncTimer?.invalidate()
        cancellables.removeAll()
    }
}

// MARK: - Network Manager Delegate
extension HealthcareProviderPortalManager: HealthcareNetworkManagerDelegate {
    func networkManager(_ manager: HealthcareNetworkManager, didReceiveMessage message: ProviderMessage) {
        DispatchQueue.main.async {
            self.providerMessages.append(message)
        }
    }
    
    func networkManager(_ manager: HealthcareNetworkManager, didUpdateConnectionStatus status: PortalConnectionStatus) {
        DispatchQueue.main.async {
            self.connectionStatus = status
        }
    }
}

// MARK: - Communication Manager Delegate
extension HealthcareProviderPortalManager: ProviderCommunicationManagerDelegate {
    func communicationManager(_ manager: ProviderCommunicationManager, didReceiveEmergencyResponse response: EmergencyResponse) {
        // Handle emergency response from provider
        print("Emergency response received: \(response.message)")
    }
}

// MARK: - Supporting Classes
class HealthcareNetworkManager {
    weak var delegate: HealthcareNetworkManagerDelegate?
    
    func validateInvitation(_ code: String, completion: @escaping (Result<ProviderInvitation, Error>) -> Void) {
        // Validate invitation code with healthcare portal
        // Implementation depends on your backend API
    }
    
    func verifyProvider(_ provider: HealthcareProvider, completion: @escaping (Result<Bool, Error>) -> Void) {
        // Verify provider credentials
        // Implementation depends on healthcare registry API
    }
    
    func establishConnection(with provider: HealthcareProvider, key: String, completion: @escaping (Result<Void, Error>) -> Void) {
        // Establish secure connection
        // Implementation depends on your security protocol
    }
    
    func shareReport(_ report: SharedHealthReport, with provider: HealthcareProvider, completion: @escaping (Result<Void, Error>) -> Void) {
        // Share encrypted health report
        // Implementation depends on your API
    }
    
    func fetchPendingInvitations(completion: @escaping ([ProviderInvitation]) -> Void) {
        // Fetch pending provider invitations
        // Implementation depends on your API
    }
}

protocol HealthcareNetworkManagerDelegate: AnyObject {
    func networkManager(_ manager: HealthcareNetworkManager, didReceiveMessage message: ProviderMessage)
    func networkManager(_ manager: HealthcareNetworkManager, didUpdateConnectionStatus status: PortalConnectionStatus)
}

class HealthcareEncryptionManager {
    func encryptReport(_ report: SharedHealthReport, completion: @escaping (SharedHealthReport?) -> Void) {
        // Encrypt health report using AES-256
        // Implementation depends on your encryption requirements
        completion(report)
    }
    
    func decryptReport(_ report: SharedHealthReport, completion: @escaping (SharedHealthReport?) -> Void) {
        // Decrypt health report
        // Implementation depends on your encryption requirements
        completion(report)
    }
}

class HealthReportGenerator {
    func generateReport(providerId: UUID, timeRange: TimeRange, includeData: [SharedDataType], completion: @escaping (Result<SharedHealthReport, Error>) -> Void) {
        // Generate comprehensive health report
        let report = SharedHealthReport(
            id: UUID(),
            providerId: providerId,
            timeRange: timeRange,
            generatedDate: Date(),
            dataTypes: includeData,
            summary: "Health report summary",
            recommendations: []
        )
        
        completion(.success(report))
    }
}

class ProviderCommunicationManager {
    weak var delegate: ProviderCommunicationManagerDelegate?
    
    func sendMessage(_ message: ProviderMessage, to provider: HealthcareProvider, completion: @escaping (Result<Void, Error>) -> Void) {
        // Send message to provider
        // Implementation depends on your messaging API
    }
    
    func sendNotification(_ notification: ProviderNotification, to provider: HealthcareProvider) {
        // Send notification to provider
        // Implementation depends on your notification system
    }
    
    func sendEmergencyAlert(_ alert: EmergencyAlert, to provider: HealthcareProvider) {
        // Send emergency alert to provider
        // Implementation depends on your emergency system
    }
    
    func sendEmergencyAlert(_ alert: EmergencyAlert, to contact: EmergencyContact) {
        // Send emergency alert to emergency contact
        // Implementation depends on your emergency system
    }
    
    func fetchNewMessages(from provider: HealthcareProvider, completion: @escaping ([ProviderMessage]) -> Void) {
        // Fetch new messages from provider
        // Implementation depends on your messaging API
        completion([])
    }
}

protocol ProviderCommunicationManagerDelegate: AnyObject {
    func communicationManager(_ manager: ProviderCommunicationManager, didReceiveEmergencyResponse response: EmergencyResponse)
}

class AppointmentManager {
    func submitRequest(_ request: AppointmentRequest, to provider: HealthcareProvider, completion: @escaping (Result<Void, Error>) -> Void) {
        // Submit appointment request
        // Implementation depends on your appointment API
    }
    
    func fetchUpdates(completion: @escaping ([AppointmentUpdate]) -> Void) {
        // Fetch appointment updates
        // Implementation depends on your appointment API
        completion([])
    }
}

class TreatmentPlanManager {
    func requestPlan(_ request: TreatmentPlanRequest, from provider: HealthcareProvider, completion: @escaping (Result<TreatmentPlan, Error>) -> Void) {
        // Request treatment plan from provider
        // Implementation depends on your treatment plan API
    }
    
    func updateProgress(_ update: TreatmentProgressUpdate, to provider: HealthcareProvider) {
        // Update treatment plan progress
        // Implementation depends on your treatment plan API
    }
    
    func fetchUpdates(completion: @escaping ([TreatmentPlanUpdate]) -> Void) {
        // Fetch treatment plan updates
        // Implementation depends on your treatment plan API
        completion([])
    }
}

class HIPAAComplianceManager {
    func ensureHIPAACompliance(completion: @escaping (Bool) -> Void) {
        // Ensure HIPAA compliance for healthcare data
        // Implementation depends on your compliance requirements
        completion(true)
    }
}

// MARK: - Data Types
struct HealthcareProvider: Codable {
    let id: UUID
    let name: String
    let specialty: String
    let licenseNumber: String
    let hospitalAffiliation: String?
    let contactInfo: ContactInfo
    let credentials: [String]
    let isVerified: Bool
}

struct ContactInfo: Codable {
    let email: String
    let phone: String
    let address: String
}

struct ProviderInvitation: Codable {
    let id: UUID
    let provider: HealthcareProvider
    let invitationCode: String
    let expirationDate: Date
    let permissions: [SharedDataType]
}

struct SharedHealthReport: Codable {
    let id: UUID
    let providerId: UUID
    let timeRange: TimeRange
    let generatedDate: Date
    let dataTypes: [SharedDataType]
    let summary: String
    let recommendations: [String]
}

struct ProviderMessage: Codable {
    let id: UUID
    let providerId: UUID
    let type: MessageType
    let subject: String
    let content: String
    let timestamp: Date
    var isRead: Bool
    let priority: MessagePriority
    let attachments: [String]
    
    init(id: UUID, providerId: UUID, type: MessageType, subject: String, content: String, timestamp: Date, isRead: Bool, priority: MessagePriority = .normal, attachments: [String] = []) {
        self.id = id
        self.providerId = providerId
        self.type = type
        self.subject = subject
        self.content = content
        self.timestamp = timestamp
        self.isRead = isRead
        self.priority = priority
        self.attachments = attachments
    }
}

struct AppointmentRequest: Codable {
    let id: UUID
    let providerId: UUID
    let preferredDates: [Date]
    let reason: String
    let urgency: AppointmentUrgency
    var status: AppointmentStatus
    let requestDate: Date
    var scheduledDate: Date?
    var notes: String?
}

struct TreatmentPlan: Codable {
    let id: UUID
    let providerId: UUID
    let title: String
    let description: String
    let medications: [String]
    let exercises: [String]
    let goals: [String]
    let duration: TimeInterval
    let createdDate: Date
    var progress: TreatmentProgress?
}

struct DataSharePermissions: Codable {
    let providerId: UUID
    let allowedDataTypes: [SharedDataType]
    let shareFrequency: ShareFrequency
    let autoShare: Bool
    let emergencyAccess: Bool
}

struct EmergencyContact: Codable {
    let id: UUID
    let name: String
    let relationship: String
    let phone: String
    let email: String?
    let isHealthcareProvider: Bool
}

struct EmergencyAlert: Codable {
    let id: UUID
    let patientId: String
    let alertType: EmergencyAlertType
    let location: CLLocation?
    let timestamp: Date
    let vitalSigns: VitalSigns?
    let medications: [String]
}

struct EmergencyResponse: Codable {
    let id: UUID
    let alertId: UUID
    let responderId: UUID
    let message: String
    let instructions: [String]
    let timestamp: Date
}

struct ProviderNotification: Codable {
    let type: NotificationType
    let providerId: UUID
    let data: DataSharePermissions
    let timestamp: Date
}

struct TreatmentPlanRequest: Codable {
    let providerId: UUID
    let symptoms: [String]
    let currentMedications: [String]
    let requestDate: Date
}

struct TreatmentProgress: Codable {
    let completedGoals: [String]
    let adherenceRate: Double
    let sideEffects: [String]
    let notes: String
    let lastUpdated: Date
}

struct TreatmentProgressUpdate: Codable {
    let planId: UUID
    let progress: TreatmentProgress
    let timestamp: Date
}

struct AppointmentUpdate: Codable {
    let requestId: UUID
    let status: AppointmentStatus
    let scheduledDate: Date?
    let notes: String?
}

struct TreatmentPlanUpdate: Codable {
    let planId: UUID
    let updatedPlan: TreatmentPlan
}

struct TimeRange: Codable {
    let startDate: Date
    let endDate: Date
    
    var description: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return "\(formatter.string(from: startDate)) - \(formatter.string(from: endDate))"
    }
}

// MARK: - Enums
enum PortalConnectionStatus {
    case disconnected
    case connecting
    case connected
    case ready
    case failed
    case complianceError
}

enum SharedDataType: String, Codable, CaseIterable {
    case healthMetrics = "health_metrics"
    case painTracking = "pain_tracking"
    case medications = "medications"
    case journalEntries = "journal_entries"
    case vitalSigns = "vital_signs"
    case workoutData = "workout_data"
    case sleepData = "sleep_data"
    case nutritionData = "nutrition_data"
}

enum ShareFrequency: String, Codable {
    case realTime = "real_time"
    case daily = "daily"
    case weekly = "weekly"
    case monthly = "monthly"
    case onDemand = "on_demand"
}

enum MessageType: String, Codable {
    case patientMessage = "patient_message"
    case providerMessage = "provider_message"
    case reportShared = "report_shared"
    case appointmentUpdate = "appointment_update"
    case treatmentPlanUpdate = "treatment_plan_update"
    case emergencyAlert = "emergency_alert"
}

enum MessagePriority: String, Codable {
    case low = "low"
    case normal = "normal"
    case high = "high"
    case urgent = "urgent"
}

enum AppointmentUrgency: String, Codable {
    case routine = "routine"
    case urgent = "urgent"
    case emergency = "emergency"
}

enum AppointmentStatus: String, Codable {
    case pending = "pending"
    case scheduled = "scheduled"
    case confirmed = "confirmed"
    case cancelled = "cancelled"
    case completed = "completed"
}

enum EmergencyAlertType: String, Codable {
    case medicalEmergency = "medical_emergency"
    case severeSymptoms = "severe_symptoms"
    case medicationReaction = "medication_reaction"
    case fallDetected = "fall_detected"
    case vitalSignsAbnormal = "vital_signs_abnormal"
}

enum NotificationType: String, Codable {
    case permissionUpdate = "permission_update"
    case dataShared = "data_shared"
    case emergencyAlert = "emergency_alert"
}

enum PortalError: Error {
    case invalidCredentials
    case encryptionFailed
    case networkError
    case complianceViolation
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let providerMessageReceived = Notification.Name("providerMessageReceived")
    static let appointmentStatusChanged = Notification.Name("appointmentStatusChanged")
}