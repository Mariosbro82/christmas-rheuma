//
//  HealthcareProviderPortal.swift
//  InflamAI-Swift
//
//  Healthcare provider portal integration for medical professional communication
//

import Foundation
import Combine
import CoreData
import HealthKit
import CryptoKit
import Network

// MARK: - Provider Models

struct HealthcareProvider: Codable, Identifiable {
    let id: String
    let name: String
    let specialty: String
    let organization: String
    let email: String
    let phone: String?
    let licenseNumber: String
    let isVerified: Bool
    let profileImageURL: String?
    let availableHours: [AvailabilitySlot]
    let communicationPreferences: CommunicationPreferences
    let accessLevel: AccessLevel
    let lastActive: Date?
}

struct AvailabilitySlot: Codable {
    let dayOfWeek: Int // 1-7, Monday-Sunday
    let startTime: String // "09:00"
    let endTime: String // "17:00"
    let timeZone: String
}

struct CommunicationPreferences: Codable {
    let allowDirectMessages: Bool
    let allowVideoConsultations: Bool
    let allowDataSharing: Bool
    let responseTimeExpectation: String // "24 hours", "same day", etc.
    let preferredContactMethod: ContactMethod
}

enum ContactMethod: String, Codable, CaseIterable {
    case inApp = "in_app"
    case email = "email"
    case phone = "phone"
    case videoCall = "video_call"
}

enum AccessLevel: String, Codable {
    case view = "view" // Can view shared data
    case limited = "limited" // Can view and comment
    case full = "full" // Can view, comment, and request changes
    case emergency = "emergency" // Full access plus emergency alerts
}

// MARK: - Communication Models

struct ProviderMessage: Codable, Identifiable {
    let id: String
    let providerId: String
    let patientId: String
    let content: String
    let messageType: MessageType
    let timestamp: Date
    let isRead: Bool
    let attachments: [MessageAttachment]
    let priority: MessagePriority
    let replyToId: String?
    let encryptedContent: Data?
}

enum MessageType: String, Codable {
    case text = "text"
    case dataRequest = "data_request"
    case appointment = "appointment"
    case prescription = "prescription"
    case recommendation = "recommendation"
    case emergency = "emergency"
    case followUp = "follow_up"
}

enum MessagePriority: String, Codable {
    case low = "low"
    case normal = "normal"
    case high = "high"
    case urgent = "urgent"
}

struct MessageAttachment: Codable, Identifiable {
    let id: String
    let fileName: String
    let fileType: String
    let fileSize: Int
    let downloadURL: String?
    let isEncrypted: Bool
}

// MARK: - Data Sharing Models

struct DataSharingRequest: Codable, Identifiable {
    let id: String
    let providerId: String
    let patientId: String
    let requestedDataTypes: [SharedDataType]
    let dateRange: DateRange
    let purpose: String
    let urgency: RequestUrgency
    let status: RequestStatus
    let requestDate: Date
    let expirationDate: Date?
    let approvedDate: Date?
    let deniedReason: String?
}

enum SharedDataType: String, Codable, CaseIterable {
    case painLevels = "pain_levels"
    case medications = "medications"
    case journalEntries = "journal_entries"
    case vitalSigns = "vital_signs"
    case labResults = "lab_results"
    case symptoms = "symptoms"
    case activities = "activities"
    case sleep = "sleep"
    case mood = "mood"
    case bassDAI = "bassdai"
}

struct DateRange: Codable {
    let startDate: Date
    let endDate: Date
}

enum RequestUrgency: String, Codable {
    case routine = "routine"
    case priority = "priority"
    case urgent = "urgent"
    case emergency = "emergency"
}

enum RequestStatus: String, Codable {
    case pending = "pending"
    case approved = "approved"
    case denied = "denied"
    case expired = "expired"
    case revoked = "revoked"
}

// MARK: - Shared Health Report

struct SharedHealthReport: Codable {
    let id: String
    let patientId: String
    let providerId: String
    let reportType: ReportType
    let dateRange: DateRange
    let generatedDate: Date
    let data: HealthReportData
    let insights: [HealthInsight]
    let recommendations: [String]
    let isEncrypted: Bool
    let accessExpirationDate: Date?
}

enum ReportType: String, Codable {
    case comprehensive = "comprehensive"
    case painSummary = "pain_summary"
    case medicationAdherence = "medication_adherence"
    case symptomTracking = "symptom_tracking"
    case activitySummary = "activity_summary"
    case custom = "custom"
}

struct HealthReportData: Codable {
    let painData: [PainDataPoint]?
    let medicationData: [MedicationDataPoint]?
    let vitalSigns: [VitalSignsDataPoint]?
    let symptoms: [SymptomDataPoint]?
    let activities: [ActivityDataPoint]?
    let bassDAIScores: [BASSDAIDataPoint]?
}

struct PainDataPoint: Codable {
    let date: Date
    let level: Int
    let location: String
    let duration: TimeInterval?
    let triggers: [String]?
}

struct MedicationDataPoint: Codable {
    let date: Date
    let medicationName: String
    let dosage: String
    let taken: Bool
    let sideEffects: [String]?
}

struct VitalSignsDataPoint: Codable {
    let date: Date
    let heartRate: Double?
    let bloodPressure: String?
    let temperature: Double?
    let weight: Double?
}

struct SymptomDataPoint: Codable {
    let date: Date
    let symptom: String
    let severity: Int
    let notes: String?
}

struct ActivityDataPoint: Codable {
    let date: Date
    let steps: Int?
    let activeMinutes: Int?
    let exerciseType: String?
    let duration: TimeInterval?
}

struct BASSDAIDataPoint: Codable {
    let date: Date
    let score: Double
    let components: [String: Int]
}

struct HealthInsight: Codable, Identifiable {
    let id: String
    let type: InsightType
    let title: String
    let description: String
    let severity: InsightSeverity
    let actionRequired: Bool
    let relatedData: [String]
}

enum InsightType: String, Codable {
    case trend = "trend"
    case anomaly = "anomaly"
    case correlation = "correlation"
    case improvement = "improvement"
    case concern = "concern"
}

enum InsightSeverity: String, Codable {
    case info = "info"
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

// MARK: - Healthcare Provider Portal Manager

class HealthcareProviderPortal: ObservableObject {
    // Core Data
    private let context: NSManagedObjectContext
    
    // Network
    private let networkManager: NetworkManager
    private let encryptionManager: EncryptionManager
    
    // Published Properties
    @Published var connectedProviders: [HealthcareProvider] = []
    @Published var pendingInvitations: [ProviderInvitation] = []
    @Published var messages: [ProviderMessage] = []
    @Published var dataSharingRequests: [DataSharingRequest] = []
    @Published var sharedReports: [SharedHealthReport] = []
    @Published var isOnline = false
    @Published var lastSyncTime: Date?
    
    // Settings
    @Published var autoShareEnabled = false
    @Published var emergencyAccessEnabled = true
    @Published var dataRetentionDays = 90
    @Published var notificationsEnabled = true
    
    // Internal State
    private var syncTimer: Timer?
    private let apiBaseURL = "https://api.inflamai.com/provider-portal"
    
    // Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    init(context: NSManagedObjectContext) {
        self.context = context
        self.networkManager = NetworkManager()
        self.encryptionManager = EncryptionManager()
        
        setupNetworkMonitoring()
        loadSettings()
        loadCachedData()
        setupPeriodicSync()
    }
    
    // MARK: - Setup
    
    private func setupNetworkMonitoring() {
        networkManager.isOnlinePublisher
            .sink { [weak self] isOnline in
                self?.isOnline = isOnline
                if isOnline {
                    self?.syncWithPortal()
                }
            }
            .store(in: &cancellables)
    }
    
    private func loadSettings() {
        autoShareEnabled = UserDefaults.standard.bool(forKey: "provider_autoShare")
        emergencyAccessEnabled = UserDefaults.standard.bool(forKey: "provider_emergencyAccess")
        if !emergencyAccessEnabled && UserDefaults.standard.object(forKey: "provider_emergencyAccess") == nil {
            emergencyAccessEnabled = true
        }
        
        dataRetentionDays = UserDefaults.standard.integer(forKey: "provider_dataRetention")
        if dataRetentionDays == 0 {
            dataRetentionDays = 90
        }
        
        notificationsEnabled = UserDefaults.standard.bool(forKey: "provider_notifications")
        if !notificationsEnabled && UserDefaults.standard.object(forKey: "provider_notifications") == nil {
            notificationsEnabled = true
        }
        
        if let lastSync = UserDefaults.standard.object(forKey: "provider_lastSync") as? Date {
            lastSyncTime = lastSync
        }
    }
    
    private func loadCachedData() {
        // Load cached provider data from Core Data or UserDefaults
        if let providersData = UserDefaults.standard.data(forKey: "cached_providers"),
           let providers = try? JSONDecoder().decode([HealthcareProvider].self, from: providersData) {
            connectedProviders = providers
        }
        
        if let messagesData = UserDefaults.standard.data(forKey: "cached_messages"),
           let cachedMessages = try? JSONDecoder().decode([ProviderMessage].self, from: messagesData) {
            messages = cachedMessages
        }
    }
    
    private func setupPeriodicSync() {
        // Sync every 5 minutes
        syncTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak self] _ in
            self?.syncWithPortal()
        }
    }
    
    // MARK: - Provider Management
    
    func inviteProvider(email: String, accessLevel: AccessLevel, message: String?) async throws {
        let invitation = ProviderInvitation(
            id: UUID().uuidString,
            providerEmail: email,
            patientId: getCurrentPatientId(),
            accessLevel: accessLevel,
            message: message,
            status: .pending,
            sentDate: Date(),
            expirationDate: Calendar.current.date(byAdding: .day, value: 7, to: Date())
        )
        
        try await networkManager.post("\(apiBaseURL)/invitations", body: invitation)
        
        DispatchQueue.main.async {
            self.pendingInvitations.append(invitation)
        }
    }
    
    func removeProvider(_ providerId: String) async throws {
        try await networkManager.delete("\(apiBaseURL)/providers/\(providerId)")
        
        DispatchQueue.main.async {
            self.connectedProviders.removeAll { $0.id == providerId }
            self.messages.removeAll { $0.providerId == providerId }
        }
    }
    
    func updateProviderAccess(_ providerId: String, accessLevel: AccessLevel) async throws {
        let updateRequest = ["accessLevel": accessLevel.rawValue]
        try await networkManager.patch("\(apiBaseURL)/providers/\(providerId)/access", body: updateRequest)
        
        DispatchQueue.main.async {
            if let index = self.connectedProviders.firstIndex(where: { $0.id == providerId }) {
                var provider = self.connectedProviders[index]
                provider = HealthcareProvider(
                    id: provider.id,
                    name: provider.name,
                    specialty: provider.specialty,
                    organization: provider.organization,
                    email: provider.email,
                    phone: provider.phone,
                    licenseNumber: provider.licenseNumber,
                    isVerified: provider.isVerified,
                    profileImageURL: provider.profileImageURL,
                    availableHours: provider.availableHours,
                    communicationPreferences: provider.communicationPreferences,
                    accessLevel: accessLevel,
                    lastActive: provider.lastActive
                )
                self.connectedProviders[index] = provider
            }
        }
    }
    
    // MARK: - Messaging
    
    func sendMessage(to providerId: String, content: String, type: MessageType = .text, priority: MessagePriority = .normal) async throws {
        let message = ProviderMessage(
            id: UUID().uuidString,
            providerId: providerId,
            patientId: getCurrentPatientId(),
            content: content,
            messageType: type,
            timestamp: Date(),
            isRead: false,
            attachments: [],
            priority: priority,
            replyToId: nil,
            encryptedContent: nil
        )
        
        try await networkManager.post("\(apiBaseURL)/messages", body: message)
        
        DispatchQueue.main.async {
            self.messages.append(message)
        }
    }
    
    func sendMessageWithAttachment(to providerId: String, content: String, attachment: Data, fileName: String, fileType: String) async throws {
        // Upload attachment first
        let attachmentResponse: MessageAttachment = try await networkManager.uploadFile(
            "\(apiBaseURL)/attachments",
            fileData: attachment,
            fileName: fileName,
            fileType: fileType
        )
        
        let message = ProviderMessage(
            id: UUID().uuidString,
            providerId: providerId,
            patientId: getCurrentPatientId(),
            content: content,
            messageType: .text,
            timestamp: Date(),
            isRead: false,
            attachments: [attachmentResponse],
            priority: .normal,
            replyToId: nil,
            encryptedContent: nil
        )
        
        try await networkManager.post("\(apiBaseURL)/messages", body: message)
        
        DispatchQueue.main.async {
            self.messages.append(message)
        }
    }
    
    func markMessageAsRead(_ messageId: String) async throws {
        try await networkManager.patch("\(apiBaseURL)/messages/\(messageId)/read", body: ["isRead": true])
        
        DispatchQueue.main.async {
            if let index = self.messages.firstIndex(where: { $0.id == messageId }) {
                var message = self.messages[index]
                message = ProviderMessage(
                    id: message.id,
                    providerId: message.providerId,
                    patientId: message.patientId,
                    content: message.content,
                    messageType: message.messageType,
                    timestamp: message.timestamp,
                    isRead: true,
                    attachments: message.attachments,
                    priority: message.priority,
                    replyToId: message.replyToId,
                    encryptedContent: message.encryptedContent
                )
                self.messages[index] = message
            }
        }
    }
    
    // MARK: - Data Sharing
    
    func approveDataSharingRequest(_ requestId: String) async throws {
        try await networkManager.patch("\(apiBaseURL)/data-requests/\(requestId)/approve", body: ["status": "approved"])
        
        DispatchQueue.main.async {
            if let index = self.dataSharingRequests.firstIndex(where: { $0.id == requestId }) {
                var request = self.dataSharingRequests[index]
                request = DataSharingRequest(
                    id: request.id,
                    providerId: request.providerId,
                    patientId: request.patientId,
                    requestedDataTypes: request.requestedDataTypes,
                    dateRange: request.dateRange,
                    purpose: request.purpose,
                    urgency: request.urgency,
                    status: .approved,
                    requestDate: request.requestDate,
                    expirationDate: request.expirationDate,
                    approvedDate: Date(),
                    deniedReason: nil
                )
                self.dataSharingRequests[index] = request
            }
        }
        
        // Generate and share the requested data
        try await generateAndShareHealthReport(for: requestId)
    }
    
    func denyDataSharingRequest(_ requestId: String, reason: String) async throws {
        let body = ["status": "denied", "deniedReason": reason]
        try await networkManager.patch("\(apiBaseURL)/data-requests/\(requestId)/deny", body: body)
        
        DispatchQueue.main.async {
            if let index = self.dataSharingRequests.firstIndex(where: { $0.id == requestId }) {
                var request = self.dataSharingRequests[index]
                request = DataSharingRequest(
                    id: request.id,
                    providerId: request.providerId,
                    patientId: request.patientId,
                    requestedDataTypes: request.requestedDataTypes,
                    dateRange: request.dateRange,
                    purpose: request.purpose,
                    urgency: request.urgency,
                    status: .denied,
                    requestDate: request.requestDate,
                    expirationDate: request.expirationDate,
                    approvedDate: nil,
                    deniedReason: reason
                )
                self.dataSharingRequests[index] = request
            }
        }
    }
    
    func generateHealthReport(for providerId: String, dataTypes: [SharedDataType], dateRange: DateRange, reportType: ReportType = .comprehensive) async throws -> SharedHealthReport {
        let reportData = try await generateHealthReportData(dataTypes: dataTypes, dateRange: dateRange)
        let insights = generateHealthInsights(from: reportData)
        
        let report = SharedHealthReport(
            id: UUID().uuidString,
            patientId: getCurrentPatientId(),
            providerId: providerId,
            reportType: reportType,
            dateRange: dateRange,
            generatedDate: Date(),
            data: reportData,
            insights: insights,
            recommendations: generateRecommendations(from: insights),
            isEncrypted: true,
            accessExpirationDate: Calendar.current.date(byAdding: .day, value: dataRetentionDays, to: Date())
        )
        
        try await networkManager.post("\(apiBaseURL)/reports", body: report)
        
        DispatchQueue.main.async {
            self.sharedReports.append(report)
        }
        
        return report
    }
    
    // MARK: - Emergency Features
    
    func sendEmergencyAlert(to providerId: String, alertType: String, severity: InsightSeverity, additionalInfo: String?) async throws {
        guard emergencyAccessEnabled else {
            throw ProviderPortalError.emergencyAccessDisabled
        }
        
        let emergencyMessage = ProviderMessage(
            id: UUID().uuidString,
            providerId: providerId,
            patientId: getCurrentPatientId(),
            content: "EMERGENCY ALERT: \(alertType)\n\(additionalInfo ?? "")",
            messageType: .emergency,
            timestamp: Date(),
            isRead: false,
            attachments: [],
            priority: .urgent,
            replyToId: nil,
            encryptedContent: nil
        )
        
        try await networkManager.post("\(apiBaseURL)/emergency", body: emergencyMessage)
        
        DispatchQueue.main.async {
            self.messages.append(emergencyMessage)
        }
    }
    
    func enableEmergencyAccess(for providerId: String, duration: TimeInterval = 86400) async throws { // 24 hours default
        let emergencyAccess = [
            "providerId": providerId,
            "duration": duration,
            "accessLevel": AccessLevel.emergency.rawValue
        ] as [String : Any]
        
        try await networkManager.post("\(apiBaseURL)/emergency-access", body: emergencyAccess)
    }
    
    // MARK: - Private Methods
    
    private func syncWithPortal() {
        guard isOnline else { return }
        
        Task {
            do {
                // Sync providers
                let providers: [HealthcareProvider] = try await networkManager.get("\(apiBaseURL)/providers")
                
                // Sync messages
                let newMessages: [ProviderMessage] = try await networkManager.get("\(apiBaseURL)/messages")
                
                // Sync data requests
                let requests: [DataSharingRequest] = try await networkManager.get("\(apiBaseURL)/data-requests")
                
                DispatchQueue.main.async {
                    self.connectedProviders = providers
                    self.messages = newMessages
                    self.dataSharingRequests = requests
                    self.lastSyncTime = Date()
                    
                    // Cache data
                    self.cacheData()
                }
                
            } catch {
                print("Error syncing with provider portal: \(error)")
            }
        }
    }
    
    private func cacheData() {
        if let providersData = try? JSONEncoder().encode(connectedProviders) {
            UserDefaults.standard.set(providersData, forKey: "cached_providers")
        }
        
        if let messagesData = try? JSONEncoder().encode(messages) {
            UserDefaults.standard.set(messagesData, forKey: "cached_messages")
        }
        
        UserDefaults.standard.set(lastSyncTime, forKey: "provider_lastSync")
    }
    
    private func generateAndShareHealthReport(for requestId: String) async throws {
        guard let request = dataSharingRequests.first(where: { $0.id == requestId }) else {
            throw ProviderPortalError.requestNotFound
        }
        
        let _ = try await generateHealthReport(
            for: request.providerId,
            dataTypes: request.requestedDataTypes,
            dateRange: request.dateRange
        )
    }
    
    private func generateHealthReportData(dataTypes: [SharedDataType], dateRange: DateRange) async throws -> HealthReportData {
        var reportData = HealthReportData(
            painData: nil,
            medicationData: nil,
            vitalSigns: nil,
            symptoms: nil,
            activities: nil,
            bassDAIScores: nil
        )
        
        for dataType in dataTypes {
            switch dataType {
            case .painLevels:
                reportData = HealthReportData(
                    painData: try await fetchPainData(dateRange: dateRange),
                    medicationData: reportData.medicationData,
                    vitalSigns: reportData.vitalSigns,
                    symptoms: reportData.symptoms,
                    activities: reportData.activities,
                    bassDAIScores: reportData.bassDAIScores
                )
            case .medications:
                reportData = HealthReportData(
                    painData: reportData.painData,
                    medicationData: try await fetchMedicationData(dateRange: dateRange),
                    vitalSigns: reportData.vitalSigns,
                    symptoms: reportData.symptoms,
                    activities: reportData.activities,
                    bassDAIScores: reportData.bassDAIScores
                )
            case .bassDAI:
                reportData = HealthReportData(
                    painData: reportData.painData,
                    medicationData: reportData.medicationData,
                    vitalSigns: reportData.vitalSigns,
                    symptoms: reportData.symptoms,
                    activities: reportData.activities,
                    bassDAIScores: try await fetchBASSDAIData(dateRange: dateRange)
                )
            default:
                // Handle other data types
                break
            }
        }
        
        return reportData
    }
    
    private func fetchPainData(dateRange: DateRange) async throws -> [PainDataPoint] {
        let request: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp <= %@", dateRange.startDate as NSDate, dateRange.endDate as NSDate)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)]
        
        let painEntries = try context.fetch(request)
        
        return painEntries.map { entry in
            PainDataPoint(
                date: entry.timestamp ?? Date(),
                level: Int(entry.painLevel),
                location: entry.location ?? "Unknown",
                duration: nil,
                triggers: nil
            )
        }
    }
    
    private func fetchMedicationData(dateRange: DateRange) async throws -> [MedicationDataPoint] {
        // Fetch medication logs from Core Data
        let request: NSFetchRequest<MedicationLog> = MedicationLog.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp <= %@", dateRange.startDate as NSDate, dateRange.endDate as NSDate)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \MedicationLog.timestamp, ascending: true)]
        
        let medicationLogs = try context.fetch(request)
        
        return medicationLogs.compactMap { log in
            guard let timestamp = log.timestamp else { return nil }
            
            return MedicationDataPoint(
                date: timestamp,
                medicationName: "Medication", // Would need to fetch from medication entity
                dosage: "Unknown",
                taken: log.taken,
                sideEffects: nil
            )
        }
    }
    
    private func fetchBASSDAIData(dateRange: DateRange) async throws -> [BASSDAIDataPoint] {
        // Fetch BASSDAI scores from Core Data
        let request: NSFetchRequest<BASSDAIEntry> = BASSDAIEntry.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp <= %@", dateRange.startDate as NSDate, dateRange.endDate as NSDate)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \BASSDAIEntry.timestamp, ascending: true)]
        
        let bassDAIEntries = try context.fetch(request)
        
        return bassDAIEntries.compactMap { entry in
            guard let timestamp = entry.timestamp else { return nil }
            
            return BASSDAIDataPoint(
                date: timestamp,
                score: entry.totalScore,
                components: [
                    "fatigue": Int(entry.fatigue),
                    "spinalPain": Int(entry.spinalPain),
                    "jointSwelling": Int(entry.jointSwelling),
                    "tenderness": Int(entry.tenderness),
                    "morningStiffness": Int(entry.morningStiffness),
                    "morningStiffnessDuration": Int(entry.morningStiffnessDuration)
                ]
            )
        }
    }
    
    private func generateHealthInsights(from data: HealthReportData) -> [HealthInsight] {
        var insights: [HealthInsight] = []
        
        // Analyze pain trends
        if let painData = data.painData, !painData.isEmpty {
            let averagePain = painData.map { $0.level }.reduce(0, +) / painData.count
            
            if averagePain > 7 {
                insights.append(HealthInsight(
                    id: UUID().uuidString,
                    type: .concern,
                    title: "High Pain Levels",
                    description: "Average pain level is \(averagePain)/10, which is concerning.",
                    severity: .high,
                    actionRequired: true,
                    relatedData: ["pain_levels"]
                ))
            }
        }
        
        // Analyze medication adherence
        if let medicationData = data.medicationData, !medicationData.isEmpty {
            let adherenceRate = Double(medicationData.filter { $0.taken }.count) / Double(medicationData.count)
            
            if adherenceRate < 0.8 {
                insights.append(HealthInsight(
                    id: UUID().uuidString,
                    type: .concern,
                    title: "Low Medication Adherence",
                    description: "Medication adherence is \(Int(adherenceRate * 100))%, which is below recommended levels.",
                    severity: .medium,
                    actionRequired: true,
                    relatedData: ["medications"]
                ))
            }
        }
        
        return insights
    }
    
    private func generateRecommendations(from insights: [HealthInsight]) -> [String] {
        var recommendations: [String] = []
        
        for insight in insights {
            switch insight.type {
            case .concern:
                if insight.relatedData.contains("pain_levels") {
                    recommendations.append("Consider adjusting pain management strategy")
                    recommendations.append("Schedule follow-up appointment to discuss pain levels")
                }
                if insight.relatedData.contains("medications") {
                    recommendations.append("Review medication schedule and barriers to adherence")
                    recommendations.append("Consider medication reminder system")
                }
            case .trend:
                recommendations.append("Continue monitoring current trends")
            default:
                break
            }
        }
        
        return recommendations
    }
    
    private func getCurrentPatientId() -> String {
        // This would return the current patient's ID
        return UserDefaults.standard.string(forKey: "currentPatientId") ?? UUID().uuidString
    }
    
    // MARK: - Settings
    
    func updateAutoShare(_ enabled: Bool) {
        autoShareEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "provider_autoShare")
    }
    
    func updateEmergencyAccess(_ enabled: Bool) {
        emergencyAccessEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "provider_emergencyAccess")
    }
    
    func updateDataRetention(_ days: Int) {
        dataRetentionDays = days
        UserDefaults.standard.set(days, forKey: "provider_dataRetention")
    }
    
    func updateNotifications(_ enabled: Bool) {
        notificationsEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "provider_notifications")
    }
}

// MARK: - Supporting Types

struct ProviderInvitation: Codable, Identifiable {
    let id: String
    let providerEmail: String
    let patientId: String
    let accessLevel: AccessLevel
    let message: String?
    let status: InvitationStatus
    let sentDate: Date
    let expirationDate: Date?
}

enum InvitationStatus: String, Codable {
    case pending = "pending"
    case accepted = "accepted"
    case declined = "declined"
    case expired = "expired"
}

enum ProviderPortalError: Error {
    case emergencyAccessDisabled
    case requestNotFound
    case providerNotFound
    case invalidAccessLevel
    case networkError
    case authenticationFailed
}

// MARK: - Network Manager (Simplified)

class NetworkManager {
    @Published var isOnline = true
    
    var isOnlinePublisher: Published<Bool>.Publisher {
        $isOnline
    }
    
    func get<T: Codable>(_ url: String) async throws -> T {
        // Simplified network implementation
        throw NetworkError.notImplemented
    }
    
    func post<T: Codable>(_ url: String, body: T) async throws {
        // Simplified network implementation
        throw NetworkError.notImplemented
    }
    
    func patch<T: Codable>(_ url: String, body: T) async throws {
        // Simplified network implementation
        throw NetworkError.notImplemented
    }
    
    func delete(_ url: String) async throws {
        // Simplified network implementation
        throw NetworkError.notImplemented
    }
    
    func uploadFile<T: Codable>(_ url: String, fileData: Data, fileName: String, fileType: String) async throws -> T {
        // Simplified file upload implementation
        throw NetworkError.notImplemented
    }
}

enum NetworkError: Error {
    case notImplemented
}