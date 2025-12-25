//
//  CloudSyncManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import CloudKit
import CryptoKit
import Combine
import Network

// MARK: - Cloud Sync Models

struct SyncableRecord: Codable {
    let id: UUID
    let recordType: RecordType
    let data: Data
    let encryptedData: Data?
    let lastModified: Date
    let deviceId: String
    let version: Int
    let checksum: String
    let syncStatus: SyncStatus
    let conflictResolution: ConflictResolution?
}

enum RecordType: String, CaseIterable, Codable {
    case symptomEntry = "SymptomEntry"
    case medicationLog = "MedicationLog"
    case painEntry = "PainEntry"
    case moodEntry = "MoodEntry"
    case exerciseSession = "ExerciseSession"
    case sleepData = "SleepData"
    case userProfile = "UserProfile"
    case userPreferences = "UserPreferences"
    case healthMetrics = "HealthMetrics"
    case journalEntry = "JournalEntry"
    case appointment = "Appointment"
    case medication = "Medication"
    case reminder = "Reminder"
    case report = "Report"
    case backup = "Backup"
    case analytics = "Analytics"
    case socialPost = "SocialPost"
    case emergencyContact = "EmergencyContact"
    case telemedicineSession = "TelemedicineSession"
    case meditationSession = "MeditationSession"
    
    var displayName: String {
        switch self {
        case .symptomEntry: return "Symptom Entry"
        case .medicationLog: return "Medication Log"
        case .painEntry: return "Pain Entry"
        case .moodEntry: return "Mood Entry"
        case .exerciseSession: return "Exercise Session"
        case .sleepData: return "Sleep Data"
        case .userProfile: return "User Profile"
        case .userPreferences: return "User Preferences"
        case .healthMetrics: return "Health Metrics"
        case .journalEntry: return "Journal Entry"
        case .appointment: return "Appointment"
        case .medication: return "Medication"
        case .reminder: return "Reminder"
        case .report: return "Report"
        case .backup: return "Backup"
        case .analytics: return "Analytics"
        case .socialPost: return "Social Post"
        case .emergencyContact: return "Emergency Contact"
        case .telemedicineSession: return "Telemedicine Session"
        case .meditationSession: return "Meditation Session"
        }
    }
    
    var priority: SyncPriority {
        switch self {
        case .emergencyContact, .userProfile: return .critical
        case .medicationLog, .painEntry, .symptomEntry: return .high
        case .moodEntry, .exerciseSession, .sleepData: return .medium
        case .journalEntry, .appointment, .reminder: return .normal
        case .analytics, .backup, .report: return .low
        default: return .normal
        }
    }
    
    var encryptionRequired: Bool {
        switch self {
        case .userProfile, .healthMetrics, .medicationLog, .painEntry, .symptomEntry, .moodEntry, .journalEntry, .emergencyContact, .telemedicineSession:
            return true
        default:
            return false
        }
    }
}

enum SyncStatus: String, CaseIterable, Codable {
    case pending = "pending"
    case syncing = "syncing"
    case synced = "synced"
    case failed = "failed"
    case conflict = "conflict"
    case deleted = "deleted"
    case encrypted = "encrypted"
    case decrypted = "decrypted"
    
    var displayName: String {
        switch self {
        case .pending: return "Pending"
        case .syncing: return "Syncing"
        case .synced: return "Synced"
        case .failed: return "Failed"
        case .conflict: return "Conflict"
        case .deleted: return "Deleted"
        case .encrypted: return "Encrypted"
        case .decrypted: return "Decrypted"
        }
    }
    
    var color: String {
        switch self {
        case .pending: return "orange"
        case .syncing: return "blue"
        case .synced: return "green"
        case .failed: return "red"
        case .conflict: return "purple"
        case .deleted: return "gray"
        case .encrypted: return "cyan"
        case .decrypted: return "mint"
        }
    }
}

enum SyncPriority: Int, CaseIterable, Codable {
    case critical = 0
    case high = 1
    case medium = 2
    case normal = 3
    case low = 4
    
    var displayName: String {
        switch self {
        case .critical: return "Critical"
        case .high: return "High"
        case .medium: return "Medium"
        case .normal: return "Normal"
        case .low: return "Low"
        }
    }
}

struct ConflictResolution: Codable {
    let strategy: ConflictStrategy
    let resolvedAt: Date
    let resolvedBy: String
    let originalRecord: SyncableRecord?
    let mergedData: Data?
    let reason: String
}

enum ConflictStrategy: String, CaseIterable, Codable {
    case useLocal = "useLocal"
    case useRemote = "useRemote"
    case merge = "merge"
    case userChoice = "userChoice"
    case timestamp = "timestamp"
    case version = "version"
    
    var displayName: String {
        switch self {
        case .useLocal: return "Use Local"
        case .useRemote: return "Use Remote"
        case .merge: return "Merge"
        case .userChoice: return "User Choice"
        case .timestamp: return "Latest Timestamp"
        case .version: return "Highest Version"
        }
    }
}

struct SyncConfiguration: Codable {
    var isEnabled: Bool = true
    var autoSyncEnabled: Bool = true
    var syncInterval: TimeInterval = 300 // 5 minutes
    var batchSize: Int = 50
    var maxRetries: Int = 3
    var retryDelay: TimeInterval = 5.0
    var encryptionEnabled: Bool = true
    var compressionEnabled: Bool = true
    var wifiOnlySync: Bool = false
    var backgroundSyncEnabled: Bool = true
    var conflictResolutionStrategy: ConflictStrategy = .timestamp
    var syncPriorityThreshold: SyncPriority = .normal
    var maxSyncAge: TimeInterval = 2592000 // 30 days
    var enableDeltaSync: Bool = true
    var enableRealTimeSync: Bool = false
}

struct SyncStatistics: Codable {
    var totalRecords: Int = 0
    var syncedRecords: Int = 0
    var pendingRecords: Int = 0
    var failedRecords: Int = 0
    var conflictedRecords: Int = 0
    var lastSyncDate: Date?
    var lastFullSyncDate: Date?
    var totalSyncTime: TimeInterval = 0
    var averageSyncTime: TimeInterval = 0
    var syncSuccessRate: Double = 0.0
    var dataTransferred: Int64 = 0
    var encryptedRecords: Int = 0
    var compressedRecords: Int = 0
    var syncErrors: [SyncError] = []
}

struct SyncError: Codable, Identifiable {
    let id = UUID()
    let timestamp: Date
    let recordId: UUID?
    let recordType: RecordType?
    let errorCode: String
    let errorMessage: String
    let isRetryable: Bool
    let retryCount: Int
    let context: [String: String]
}

struct EncryptionKey: Codable {
    let keyId: String
    let algorithm: String
    let createdAt: Date
    let expiresAt: Date?
    let isActive: Bool
    let keyData: Data // Encrypted key data
}

struct SyncProgress {
    let recordType: RecordType
    let totalRecords: Int
    let processedRecords: Int
    let currentOperation: SyncOperation
    let estimatedTimeRemaining: TimeInterval
    let bytesTransferred: Int64
    let totalBytes: Int64
}

enum SyncOperation: String, CaseIterable {
    case preparing = "preparing"
    case uploading = "uploading"
    case downloading = "downloading"
    case encrypting = "encrypting"
    case decrypting = "decrypting"
    case compressing = "compressing"
    case decompressing = "decompressing"
    case resolving = "resolving"
    case finalizing = "finalizing"
    
    var displayName: String {
        switch self {
        case .preparing: return "Preparing"
        case .uploading: return "Uploading"
        case .downloading: return "Downloading"
        case .encrypting: return "Encrypting"
        case .decrypting: return "Decrypting"
        case .compressing: return "Compressing"
        case .decompressing: return "Decompressing"
        case .resolving: return "Resolving Conflicts"
        case .finalizing: return "Finalizing"
        }
    }
}

// MARK: - Cloud Sync Manager

@MainActor
class CloudSyncManager: ObservableObject {
    // MARK: - Published Properties
    @Published var configuration: SyncConfiguration = SyncConfiguration()
    @Published var statistics: SyncStatistics = SyncStatistics()
    @Published var isSyncing: Bool = false
    @Published var syncProgress: SyncProgress?
    @Published var pendingRecords: [SyncableRecord] = []
    @Published var conflictedRecords: [SyncableRecord] = []
    @Published var isCloudKitAvailable: Bool = false
    @Published var accountStatus: CKAccountStatus = .couldNotDetermine
    @Published var lastSyncError: SyncError?
    
    // MARK: - Private Properties
    private let container: CKContainer
    private let privateDatabase: CKDatabase
    private let publicDatabase: CKDatabase
    private let sharedDatabase: CKDatabase
    
    // Encryption
    private var encryptionManager: EncryptionManager
    private var compressionManager: CompressionManager
    
    // Network monitoring
    private let networkMonitor = NWPathMonitor()
    private let networkQueue = DispatchQueue(label: "NetworkMonitor")
    private var isNetworkAvailable: Bool = false
    
    // Sync management
    private var syncQueue = DispatchQueue(label: "CloudSync", qos: .utility)
    private var syncTimer: Timer?
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    
    // Observers
    private var cancellables = Set<AnyCancellable>()
    
    // Constants
    private let deviceId = UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString
    private let maxBatchSize = 100
    private let syncTimeoutInterval: TimeInterval = 300 // 5 minutes
    
    init() {
        self.container = CKContainer.default()
        self.privateDatabase = container.privateCloudDatabase
        self.publicDatabase = container.publicCloudDatabase
        self.sharedDatabase = container.sharedCloudDatabase
        
        self.encryptionManager = EncryptionManager()
        self.compressionManager = CompressionManager()
        
        setupCloudKit()
        setupNetworkMonitoring()
        loadConfiguration()
        loadStatistics()
        loadPendingRecords()
        
        setupAutoSync()
        observeAppLifecycle()
    }
    
    deinit {
        syncTimer?.invalidate()
        networkMonitor.cancel()
        endBackgroundTask()
    }
    
    // MARK: - Setup
    
    private func setupCloudKit() {
        Task {
            do {
                let status = try await container.accountStatus()
                await MainActor.run {
                    self.accountStatus = status
                    self.isCloudKitAvailable = status == .available
                }
                
                if status == .available {
                    try await setupCloudKitSchema()
                }
            } catch {
                await MainActor.run {
                    self.handleSyncError(SyncError(
                        timestamp: Date(),
                        recordId: nil,
                        recordType: nil,
                        errorCode: "CLOUDKIT_SETUP_FAILED",
                        errorMessage: error.localizedDescription,
                        isRetryable: true,
                        retryCount: 0,
                        context: [:]
                    ))
                }
            }
        }
    }
    
    private func setupCloudKitSchema() async throws {
        // Create record types for each syncable record type
        for recordType in RecordType.allCases {
            let ckRecordType = CKRecord.RecordType(recordType.rawValue)
            // Schema would be created automatically when first record is saved
        }
    }
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isNetworkAvailable = path.status == .satisfied
                
                if path.status == .satisfied && self?.configuration.autoSyncEnabled == true {
                    self?.triggerSync()
                }
            }
        }
        networkMonitor.start(queue: networkQueue)
    }
    
    private func loadConfiguration() {
        if let data = UserDefaults.standard.data(forKey: "cloudSyncConfiguration"),
           let config = try? JSONDecoder().decode(SyncConfiguration.self, from: data) {
            configuration = config
        }
    }
    
    private func saveConfiguration() {
        do {
            let data = try JSONEncoder().encode(configuration)
            UserDefaults.standard.set(data, forKey: "cloudSyncConfiguration")
        } catch {
            print("Failed to save sync configuration: \(error)")
        }
    }
    
    private func loadStatistics() {
        if let data = UserDefaults.standard.data(forKey: "cloudSyncStatistics"),
           let stats = try? JSONDecoder().decode(SyncStatistics.self, from: data) {
            statistics = stats
        }
    }
    
    private func saveStatistics() {
        do {
            let data = try JSONEncoder().encode(statistics)
            UserDefaults.standard.set(data, forKey: "cloudSyncStatistics")
        } catch {
            print("Failed to save sync statistics: \(error)")
        }
    }
    
    private func loadPendingRecords() {
        if let data = UserDefaults.standard.data(forKey: "pendingSyncRecords"),
           let records = try? JSONDecoder().decode([SyncableRecord].self, from: data) {
            pendingRecords = records
        }
    }
    
    private func savePendingRecords() {
        do {
            let data = try JSONEncoder().encode(pendingRecords)
            UserDefaults.standard.set(data, forKey: "pendingSyncRecords")
        } catch {
            print("Failed to save pending records: \(error)")
        }
    }
    
    private func setupAutoSync() {
        if configuration.autoSyncEnabled {
            syncTimer = Timer.scheduledTimer(withTimeInterval: configuration.syncInterval, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.triggerSync()
                }
            }
        }
    }
    
    private func observeAppLifecycle() {
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { [weak self] _ in
                self?.handleAppDidEnterBackground()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.handleAppWillEnterForeground()
                }
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Public API
    
    func addRecord<T: Codable>(_ record: T, type: RecordType, priority: SyncPriority? = nil) {
        do {
            let data = try JSONEncoder().encode(record)
            let checksum = calculateChecksum(data)
            
            var encryptedData: Data?
            if type.encryptionRequired && configuration.encryptionEnabled {
                encryptedData = try encryptionManager.encrypt(data)
            }
            
            let syncableRecord = SyncableRecord(
                id: UUID(),
                recordType: type,
                data: data,
                encryptedData: encryptedData,
                lastModified: Date(),
                deviceId: deviceId,
                version: 1,
                checksum: checksum,
                syncStatus: .pending,
                conflictResolution: nil
            )
            
            pendingRecords.append(syncableRecord)
            statistics.totalRecords += 1
            statistics.pendingRecords += 1
            
            savePendingRecords()
            saveStatistics()
            
            // Trigger immediate sync for high priority records
            if (priority ?? type.priority).rawValue <= SyncPriority.high.rawValue {
                triggerSync()
            }
        } catch {
            handleSyncError(SyncError(
                timestamp: Date(),
                recordId: nil,
                recordType: type,
                errorCode: "RECORD_ENCODING_FAILED",
                errorMessage: error.localizedDescription,
                isRetryable: false,
                retryCount: 0,
                context: ["recordType": type.rawValue]
            ))
        }
    }
    
    func updateRecord<T: Codable>(_ record: T, id: UUID, type: RecordType) {
        if let index = pendingRecords.firstIndex(where: { $0.id == id }) {
            do {
                let data = try JSONEncoder().encode(record)
                let checksum = calculateChecksum(data)
                
                var encryptedData: Data?
                if type.encryptionRequired && configuration.encryptionEnabled {
                    encryptedData = try encryptionManager.encrypt(data)
                }
                
                var updatedRecord = pendingRecords[index]
                updatedRecord = SyncableRecord(
                    id: updatedRecord.id,
                    recordType: updatedRecord.recordType,
                    data: data,
                    encryptedData: encryptedData,
                    lastModified: Date(),
                    deviceId: deviceId,
                    version: updatedRecord.version + 1,
                    checksum: checksum,
                    syncStatus: .pending,
                    conflictResolution: nil
                )
                
                pendingRecords[index] = updatedRecord
                savePendingRecords()
                
                triggerSync()
            } catch {
                handleSyncError(SyncError(
                    timestamp: Date(),
                    recordId: id,
                    recordType: type,
                    errorCode: "RECORD_UPDATE_FAILED",
                    errorMessage: error.localizedDescription,
                    isRetryable: false,
                    retryCount: 0,
                    context: ["recordId": id.uuidString]
                ))
            }
        }
    }
    
    func deleteRecord(id: UUID) {
        if let index = pendingRecords.firstIndex(where: { $0.id == id }) {
            var deletedRecord = pendingRecords[index]
            deletedRecord = SyncableRecord(
                id: deletedRecord.id,
                recordType: deletedRecord.recordType,
                data: deletedRecord.data,
                encryptedData: deletedRecord.encryptedData,
                lastModified: Date(),
                deviceId: deviceId,
                version: deletedRecord.version + 1,
                checksum: deletedRecord.checksum,
                syncStatus: .deleted,
                conflictResolution: nil
            )
            
            pendingRecords[index] = deletedRecord
            savePendingRecords()
            
            triggerSync()
        }
    }
    
    func triggerSync() {
        guard configuration.isEnabled,
              isCloudKitAvailable,
              isNetworkAvailable,
              !isSyncing else { return }
        
        if configuration.wifiOnlySync && !isWiFiConnection() {
            return
        }
        
        Task {
            await performSync()
        }
    }
    
    func forceSyncAll() {
        guard isCloudKitAvailable else { return }
        
        Task {
            await performFullSync()
        }
    }
    
    func resolveConflict(_ record: SyncableRecord, strategy: ConflictStrategy) {
        let resolution = ConflictResolution(
            strategy: strategy,
            resolvedAt: Date(),
            resolvedBy: deviceId,
            originalRecord: record,
            mergedData: nil,
            reason: "User resolved conflict using \(strategy.displayName)"
        )
        
        var resolvedRecord = record
        resolvedRecord = SyncableRecord(
            id: record.id,
            recordType: record.recordType,
            data: record.data,
            encryptedData: record.encryptedData,
            lastModified: Date(),
            deviceId: deviceId,
            version: record.version + 1,
            checksum: record.checksum,
            syncStatus: .pending,
            conflictResolution: resolution
        )
        
        // Remove from conflicts and add to pending
        conflictedRecords.removeAll { $0.id == record.id }
        pendingRecords.append(resolvedRecord)
        
        statistics.conflictedRecords -= 1
        statistics.pendingRecords += 1
        
        savePendingRecords()
        saveStatistics()
        
        triggerSync()
    }
    
    func clearSyncData() {
        pendingRecords.removeAll()
        conflictedRecords.removeAll()
        statistics = SyncStatistics()
        
        savePendingRecords()
        saveStatistics()
    }
    
    func exportSyncData() -> Data? {
        let exportData = SyncDataExport(
            configuration: configuration,
            statistics: statistics,
            pendingRecords: pendingRecords,
            conflictedRecords: conflictedRecords,
            exportDate: Date()
        )
        
        return try? JSONEncoder().encode(exportData)
    }
    
    func importSyncData(_ data: Data) -> Bool {
        do {
            let importData = try JSONDecoder().decode(SyncDataExport.self, from: data)
            
            configuration = importData.configuration
            statistics = importData.statistics
            pendingRecords = importData.pendingRecords
            conflictedRecords = importData.conflictedRecords
            
            saveConfiguration()
            saveStatistics()
            savePendingRecords()
            
            return true
        } catch {
            print("Failed to import sync data: \(error)")
            return false
        }
    }
    
    // MARK: - Private Sync Methods
    
    private func performSync() async {
        await MainActor.run {
            isSyncing = true
        }
        
        let startTime = Date()
        
        do {
            // Start background task
            beginBackgroundTask()
            
            // Upload pending records
            await uploadPendingRecords()
            
            // Download remote changes
            await downloadRemoteChanges()
            
            // Update statistics
            let syncTime = Date().timeIntervalSince(startTime)
            await MainActor.run {
                self.statistics.lastSyncDate = Date()
                self.statistics.totalSyncTime += syncTime
                self.statistics.averageSyncTime = self.statistics.totalSyncTime / Double(max(1, self.statistics.syncedRecords))
                self.saveStatistics()
            }
        } catch {
            await MainActor.run {
                self.handleSyncError(SyncError(
                    timestamp: Date(),
                    recordId: nil,
                    recordType: nil,
                    errorCode: "SYNC_FAILED",
                    errorMessage: error.localizedDescription,
                    isRetryable: true,
                    retryCount: 0,
                    context: [:]
                ))
            }
        }
        
        await MainActor.run {
            isSyncing = false
            syncProgress = nil
        }
        
        endBackgroundTask()
    }
    
    private func performFullSync() async {
        await MainActor.run {
            isSyncing = true
        }
        
        do {
            beginBackgroundTask()
            
            // Clear all local data and re-download everything
            await downloadAllRemoteRecords()
            
            await MainActor.run {
                self.statistics.lastFullSyncDate = Date()
                self.saveStatistics()
            }
        } catch {
            await MainActor.run {
                self.handleSyncError(SyncError(
                    timestamp: Date(),
                    recordId: nil,
                    recordType: nil,
                    errorCode: "FULL_SYNC_FAILED",
                    errorMessage: error.localizedDescription,
                    isRetryable: true,
                    retryCount: 0,
                    context: [:]
                ))
            }
        }
        
        await MainActor.run {
            isSyncing = false
            syncProgress = nil
        }
        
        endBackgroundTask()
    }
    
    private func uploadPendingRecords() async {
        let recordsToUpload = await MainActor.run {
            pendingRecords.filter { $0.syncStatus == .pending }
        }
        
        guard !recordsToUpload.isEmpty else { return }
        
        await MainActor.run {
            syncProgress = SyncProgress(
                recordType: .backup,
                totalRecords: recordsToUpload.count,
                processedRecords: 0,
                currentOperation: .uploading,
                estimatedTimeRemaining: 0,
                bytesTransferred: 0,
                totalBytes: Int64(recordsToUpload.reduce(0) { $0 + $1.data.count })
            )
        }
        
        // Process records in batches
        let batches = recordsToUpload.chunked(into: configuration.batchSize)
        
        for (batchIndex, batch) in batches.enumerated() {
            do {
                let ckRecords = try await prepareCKRecords(from: batch)
                let savedRecords = try await privateDatabase.modifyRecords(saving: ckRecords, deleting: [])
                
                await MainActor.run {
                    // Update sync status for successfully uploaded records
                    for (index, record) in batch.enumerated() {
                        if let pendingIndex = self.pendingRecords.firstIndex(where: { $0.id == record.id }) {
                            var updatedRecord = self.pendingRecords[pendingIndex]
                            updatedRecord = SyncableRecord(
                                id: updatedRecord.id,
                                recordType: updatedRecord.recordType,
                                data: updatedRecord.data,
                                encryptedData: updatedRecord.encryptedData,
                                lastModified: updatedRecord.lastModified,
                                deviceId: updatedRecord.deviceId,
                                version: updatedRecord.version,
                                checksum: updatedRecord.checksum,
                                syncStatus: .synced,
                                conflictResolution: updatedRecord.conflictResolution
                            )
                            self.pendingRecords[pendingIndex] = updatedRecord
                        }
                    }
                    
                    // Update statistics
                    self.statistics.syncedRecords += batch.count
                    self.statistics.pendingRecords -= batch.count
                    self.statistics.dataTransferred += Int64(batch.reduce(0) { $0 + $1.data.count })
                    
                    // Update progress
                    if var progress = self.syncProgress {
                        progress.processedRecords = (batchIndex + 1) * self.configuration.batchSize
                        self.syncProgress = progress
                    }
                }
            } catch {
                await MainActor.run {
                    self.handleBatchUploadError(batch, error: error)
                }
            }
        }
        
        await MainActor.run {
            savePendingRecords()
            saveStatistics()
        }
    }
    
    private func downloadRemoteChanges() async {
        await MainActor.run {
            syncProgress = SyncProgress(
                recordType: .backup,
                totalRecords: 0,
                processedRecords: 0,
                currentOperation: .downloading,
                estimatedTimeRemaining: 0,
                bytesTransferred: 0,
                totalBytes: 0
            )
        }
        
        do {
            // Query for changes since last sync
            let lastSyncDate = await MainActor.run { statistics.lastSyncDate }
            let query = CKQuery(recordType: "SyncableRecord", predicate: NSPredicate(value: true))
            
            if let lastSync = lastSyncDate {
                query.predicate = NSPredicate(format: "modificationDate > %@", lastSync as NSDate)
            }
            
            let (matchResults, _) = try await privateDatabase.records(matching: query)
            
            var downloadedRecords: [SyncableRecord] = []
            
            for (_, result) in matchResults {
                switch result {
                case .success(let ckRecord):
                    if let syncableRecord = try? convertCKRecordToSyncableRecord(ckRecord) {
                        downloadedRecords.append(syncableRecord)
                    }
                case .failure(let error):
                    await MainActor.run {
                        self.handleSyncError(SyncError(
                            timestamp: Date(),
                            recordId: nil,
                            recordType: nil,
                            errorCode: "DOWNLOAD_RECORD_FAILED",
                            errorMessage: error.localizedDescription,
                            isRetryable: true,
                            retryCount: 0,
                            context: [:]
                        ))
                    }
                }
            }
            
            await MainActor.run {
                self.processDownloadedRecords(downloadedRecords)
            }
        } catch {
            await MainActor.run {
                self.handleSyncError(SyncError(
                    timestamp: Date(),
                    recordId: nil,
                    recordType: nil,
                    errorCode: "DOWNLOAD_FAILED",
                    errorMessage: error.localizedDescription,
                    isRetryable: true,
                    retryCount: 0,
                    context: [:]
                ))
            }
        }
    }
    
    private func downloadAllRemoteRecords() async {
        // Implementation for full sync
        await downloadRemoteChanges()
    }
    
    private func prepareCKRecords(from records: [SyncableRecord]) async throws -> [CKRecord] {
        var ckRecords: [CKRecord] = []
        
        for record in records {
            await MainActor.run {
                if var progress = self.syncProgress {
                    progress.currentOperation = record.recordType.encryptionRequired ? .encrypting : .uploading
                    self.syncProgress = progress
                }
            }
            
            let ckRecord = CKRecord(recordType: record.recordType.rawValue, recordID: CKRecord.ID(recordName: record.id.uuidString))
            
            // Add metadata
            ckRecord["deviceId"] = record.deviceId
            ckRecord["version"] = record.version
            ckRecord["checksum"] = record.checksum
            ckRecord["lastModified"] = record.lastModified
            
            // Add data (encrypted if required)
            if let encryptedData = record.encryptedData {
                ckRecord["encryptedData"] = encryptedData
                ckRecord["isEncrypted"] = true
            } else {
                ckRecord["data"] = record.data
                ckRecord["isEncrypted"] = false
            }
            
            // Compress if enabled
            if configuration.compressionEnabled {
                await MainActor.run {
                    if var progress = self.syncProgress {
                        progress.currentOperation = .compressing
                        self.syncProgress = progress
                    }
                }
                
                if let compressedData = try? compressionManager.compress(record.data) {
                    ckRecord["compressedData"] = compressedData
                    ckRecord["isCompressed"] = true
                }
            }
            
            ckRecords.append(ckRecord)
        }
        
        return ckRecords
    }
    
    private func convertCKRecordToSyncableRecord(_ ckRecord: CKRecord) throws -> SyncableRecord {
        guard let recordTypeString = ckRecord.recordType as String?,
              let recordType = RecordType(rawValue: recordTypeString),
              let deviceId = ckRecord["deviceId"] as? String,
              let version = ckRecord["version"] as? Int,
              let checksum = ckRecord["checksum"] as? String,
              let lastModified = ckRecord["lastModified"] as? Date else {
            throw SyncManagerError.invalidRecord
        }
        
        let isEncrypted = ckRecord["isEncrypted"] as? Bool ?? false
        let data: Data
        let encryptedData: Data?
        
        if isEncrypted {
            guard let encrypted = ckRecord["encryptedData"] as? Data else {
                throw SyncManagerError.missingEncryptedData
            }
            encryptedData = encrypted
            data = try encryptionManager.decrypt(encrypted)
        } else {
            guard let recordData = ckRecord["data"] as? Data else {
                throw SyncManagerError.missingData
            }
            data = recordData
            encryptedData = nil
        }
        
        return SyncableRecord(
            id: UUID(uuidString: ckRecord.recordID.recordName) ?? UUID(),
            recordType: recordType,
            data: data,
            encryptedData: encryptedData,
            lastModified: lastModified,
            deviceId: deviceId,
            version: version,
            checksum: checksum,
            syncStatus: .synced,
            conflictResolution: nil
        )
    }
    
    private func processDownloadedRecords(_ records: [SyncableRecord]) {
        for record in records {
            // Check for conflicts
            if let existingIndex = pendingRecords.firstIndex(where: { $0.id == record.id }) {
                let existingRecord = pendingRecords[existingIndex]
                
                if existingRecord.version != record.version || existingRecord.checksum != record.checksum {
                    // Conflict detected
                    conflictedRecords.append(record)
                    statistics.conflictedRecords += 1
                } else {
                    // Update existing record
                    pendingRecords[existingIndex] = record
                }
            } else {
                // New record
                pendingRecords.append(record)
                statistics.totalRecords += 1
            }
        }
        
        savePendingRecords()
        saveStatistics()
    }
    
    // MARK: - Helper Methods
    
    private func calculateChecksum(_ data: Data) -> String {
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    private func isWiFiConnection() -> Bool {
        // Implementation would check current network type
        return true // Simplified for now
    }
    
    private func handleSyncError(_ error: SyncError) {
        statistics.syncErrors.append(error)
        lastSyncError = error
        
        // Keep only recent errors
        if statistics.syncErrors.count > 100 {
            statistics.syncErrors = Array(statistics.syncErrors.suffix(100))
        }
        
        saveStatistics()
    }
    
    private func handleBatchUploadError(_ batch: [SyncableRecord], error: Error) {
        for record in batch {
            if let index = pendingRecords.firstIndex(where: { $0.id == record.id }) {
                var failedRecord = pendingRecords[index]
                failedRecord = SyncableRecord(
                    id: failedRecord.id,
                    recordType: failedRecord.recordType,
                    data: failedRecord.data,
                    encryptedData: failedRecord.encryptedData,
                    lastModified: failedRecord.lastModified,
                    deviceId: failedRecord.deviceId,
                    version: failedRecord.version,
                    checksum: failedRecord.checksum,
                    syncStatus: .failed,
                    conflictResolution: failedRecord.conflictResolution
                )
                pendingRecords[index] = failedRecord
            }
        }
        
        statistics.failedRecords += batch.count
        
        handleSyncError(SyncError(
            timestamp: Date(),
            recordId: nil,
            recordType: nil,
            errorCode: "BATCH_UPLOAD_FAILED",
            errorMessage: error.localizedDescription,
            isRetryable: true,
            retryCount: 0,
            context: ["batchSize": "\(batch.count)"]
        ))
    }
    
    private func beginBackgroundTask() {
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "CloudSync") { [weak self] in
            self?.endBackgroundTask()
        }
    }
    
    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }
    
    private func handleAppDidEnterBackground() {
        if configuration.backgroundSyncEnabled && !pendingRecords.isEmpty {
            triggerSync()
        }
    }
    
    private func handleAppWillEnterForeground() {
        if configuration.autoSyncEnabled {
            triggerSync()
        }
    }
}

// MARK: - Supporting Classes

class EncryptionManager {
    private let keySize = 256 / 8 // 32 bytes for AES-256
    private var currentKey: SymmetricKey?
    
    init() {
        loadOrGenerateKey()
    }
    
    private func loadOrGenerateKey() {
        if let keyData = UserDefaults.standard.data(forKey: "encryptionKey") {
            currentKey = SymmetricKey(data: keyData)
        } else {
            currentKey = SymmetricKey(size: .bits256)
            if let keyData = currentKey?.withUnsafeBytes({ Data($0) }) {
                UserDefaults.standard.set(keyData, forKey: "encryptionKey")
            }
        }
    }
    
    func encrypt(_ data: Data) throws -> Data {
        guard let key = currentKey else {
            throw EncryptionError.noKey
        }
        
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined ?? Data()
    }
    
    func decrypt(_ encryptedData: Data) throws -> Data {
        guard let key = currentKey else {
            throw EncryptionError.noKey
        }
        
        let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
        return try AES.GCM.open(sealedBox, using: key)
    }
}

class CompressionManager {
    func compress(_ data: Data) throws -> Data {
        return try (data as NSData).compressed(using: .lzfse) as Data
    }
    
    func decompress(_ compressedData: Data) throws -> Data {
        return try (compressedData as NSData).decompressed(using: .lzfse) as Data
    }
}

struct SyncDataExport: Codable {
    let configuration: SyncConfiguration
    let statistics: SyncStatistics
    let pendingRecords: [SyncableRecord]
    let conflictedRecords: [SyncableRecord]
    let exportDate: Date
}

enum SyncManagerError: Error {
    case invalidRecord
    case missingData
    case missingEncryptedData
    case encryptionFailed
    case decryptionFailed
    case compressionFailed
    case decompressionFailed
    case networkUnavailable
    case cloudKitUnavailable
    case quotaExceeded
    case authenticationFailed
}

enum EncryptionError: Error {
    case noKey
    case encryptionFailed
    case decryptionFailed
}

// MARK: - Extensions

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}