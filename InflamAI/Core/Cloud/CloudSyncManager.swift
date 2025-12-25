//
//  CloudSyncManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import CloudKit
import CryptoKit
import Combine
import Network

// MARK: - Cloud Sync Manager
@MainActor
class CloudSyncManager: ObservableObject {
    
    // MARK: - Published Properties
    @Published var syncStatus: SyncStatus = .idle
    @Published var lastSyncDate: Date?
    @Published var isCloudAvailable = false
    @Published var syncProgress: Double = 0.0
    @Published var pendingUploads: Int = 0
    @Published var pendingDownloads: Int = 0
    @Published var storageUsed: Int64 = 0
    @Published var storageLimit: Int64 = 0
    @Published var conflictResolutions: [SyncConflict] = []
    @Published var backupStatus: BackupStatus = .none
    @Published var encryptionEnabled = true
    
    // MARK: - Private Properties
    private let container: CKContainer
    private let privateDatabase: CKDatabase
    private let publicDatabase: CKDatabase
    private let encryptionManager = EncryptionManager()
    private let networkMonitor = NWPathMonitor()
    private let syncQueue = DispatchQueue(label: "com.inflamai.sync", qos: .utility)
    private var cancellables = Set<AnyCancellable>()
    private var syncTimer: Timer?
    private var backupTimer: Timer?
    
    // MARK: - Sync Configuration
    private let syncInterval: TimeInterval = 300 // 5 minutes
    private let backupInterval: TimeInterval = 86400 // 24 hours
    private let maxRetryAttempts = 3
    private let batchSize = 50
    
    // MARK: - Record Types
    private let recordTypes = [
        "PainEntry",
        "MedicationEntry",
        "JournalEntry",
        "SymptomEntry",
        "WeatherData",
        "ActivityData",
        "SleepData",
        "UserSettings",
        "HealthMetrics"
    ]
    
    // MARK: - Initialization
    init() {
        self.container = CKContainer(identifier: "iCloud.com.inflamai.data")
        self.privateDatabase = container.privateCloudDatabase
        self.publicDatabase = container.publicCloudDatabase
        
        setupNetworkMonitoring()
        setupPeriodicSync()
        setupAutomaticBackup()
        checkCloudKitAvailability()
    }
    
    deinit {
        syncTimer?.invalidate()
        backupTimer?.invalidate()
        networkMonitor.cancel()
    }
    
    // MARK: - Public Methods
    
    func startSync() async {
        guard isCloudAvailable else {
            print("CloudKit not available")
            return
        }
        
        syncStatus = .syncing
        syncProgress = 0.0
        
        do {
            // Upload local changes
            await uploadLocalChanges()
            
            // Download remote changes
            await downloadRemoteChanges()
            
            // Resolve conflicts
            await resolveConflicts()
            
            syncStatus = .completed
            lastSyncDate = Date()
            
        } catch {
            syncStatus = .failed(error)
            print("Sync failed: \(error)")
        }
    }
    
    func forceSyncAll() async {
        syncStatus = .syncing
        
        do {
            // Clear sync tokens to force full sync
            clearSyncTokens()
            
            // Perform full sync
            await startSync()
            
        } catch {
            syncStatus = .failed(error)
        }
    }
    
    func createBackup() async -> BackupResult {
        backupStatus = .inProgress
        
        do {
            let backupData = await gatherBackupData()
            let encryptedData = try encryptionManager.encrypt(data: backupData)
            
            let backupRecord = CKRecord(recordType: "Backup")
            backupRecord["data"] = encryptedData
            backupRecord["timestamp"] = Date()
            backupRecord["version"] = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
            
            let savedRecord = try await privateDatabase.save(backupRecord)
            
            backupStatus = .completed(Date())
            return .success(savedRecord.recordID)
            
        } catch {
            backupStatus = .failed(error)
            return .failure(error)
        }
    }
    
    func restoreFromBackup(backupID: CKRecord.ID) async -> RestoreResult {
        do {
            let record = try await privateDatabase.record(for: backupID)
            
            guard let encryptedData = record["data"] as? Data else {
                throw CloudSyncError.invalidBackupData
            }
            
            let decryptedData = try encryptionManager.decrypt(data: encryptedData)
            let backupData = try JSONDecoder().decode(BackupData.self, from: decryptedData)
            
            await restoreData(from: backupData)
            
            return .success
            
        } catch {
            return .failure(error)
        }
    }
    
    func getAvailableBackups() async -> [BackupInfo] {
        do {
            let query = CKQuery(recordType: "Backup", predicate: NSPredicate(value: true))
            query.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
            
            let (matchResults, _) = try await privateDatabase.records(matching: query)
            
            return matchResults.compactMap { (recordID, result) in
                switch result {
                case .success(let record):
                    return BackupInfo(
                        id: recordID,
                        timestamp: record["timestamp"] as? Date ?? Date(),
                        version: record["version"] as? String ?? "Unknown",
                        size: (record["data"] as? Data)?.count ?? 0
                    )
                case .failure:
                    return nil
                }
            }
            
        } catch {
            print("Failed to fetch backups: \(error)")
            return []
        }
    }
    
    func deleteBackup(backupID: CKRecord.ID) async -> Bool {
        do {
            try await privateDatabase.deleteRecord(withID: backupID)
            return true
        } catch {
            print("Failed to delete backup: \(error)")
            return false
        }
    }
    
    func resolveConflict(_ conflict: SyncConflict, resolution: ConflictResolution) async {
        switch resolution {
        case .useLocal:
            await uploadRecord(conflict.localRecord)
        case .useRemote:
            await downloadRecord(conflict.remoteRecord)
        case .merge:
            let mergedRecord = await mergeRecords(local: conflict.localRecord, remote: conflict.remoteRecord)
            await uploadRecord(mergedRecord)
        }
        
        // Remove resolved conflict
        conflictResolutions.removeAll { $0.id == conflict.id }
    }
    
    func getStorageUsage() async {
        do {
            // Calculate storage usage across all record types
            var totalSize: Int64 = 0
            
            for recordType in recordTypes {
                let query = CKQuery(recordType: recordType, predicate: NSPredicate(value: true))
                let (matchResults, _) = try await privateDatabase.records(matching: query)
                
                for (_, result) in matchResults {
                    if case .success(let record) = result {
                        totalSize += calculateRecordSize(record)
                    }
                }
            }
            
            storageUsed = totalSize
            storageLimit = 1_000_000_000 // 1GB limit (example)
            
        } catch {
            print("Failed to calculate storage usage: \(error)")
        }
    }
    
    func enableEncryption(_ enabled: Bool) {
        encryptionEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "CloudSyncEncryptionEnabled")
    }
    
    func exportData(format: ExportFormat) async -> Data? {
        do {
            let allData = await gatherAllData()
            
            switch format {
            case .json:
                return try JSONEncoder().encode(allData)
            case .csv:
                return convertToCSV(allData)
            case .pdf:
                return await generatePDF(from: allData)
            case .hl7fhir:
                return try convertToHL7FHIR(allData)
            }
            
        } catch {
            print("Failed to export data: \(error)")
            return nil
        }
    }
    
    // MARK: - Private Sync Methods
    
    private func uploadLocalChanges() async {
        let localChanges = await getLocalChanges()
        pendingUploads = localChanges.count
        
        for (index, change) in localChanges.enumerated() {
            do {
                let record = try await createCloudKitRecord(from: change)
                
                if encryptionEnabled {
                    try await encryptRecord(record)
                }
                
                _ = try await privateDatabase.save(record)
                await markAsUploaded(change)
                
                syncProgress = Double(index + 1) / Double(localChanges.count) * 0.5
                
            } catch {
                print("Failed to upload record: \(error)")
                await handleUploadError(change, error: error)
            }
        }
        
        pendingUploads = 0
    }
    
    private func downloadRemoteChanges() async {
        for recordType in recordTypes {
            do {
                let changes = try await fetchRemoteChanges(for: recordType)
                pendingDownloads = changes.count
                
                for (index, record) in changes.enumerated() {
                    do {
                        if encryptionEnabled {
                            try await decryptRecord(record)
                        }
                        
                        await processRemoteRecord(record)
                        
                        syncProgress = 0.5 + (Double(index + 1) / Double(changes.count) * 0.5)
                        
                    } catch {
                        print("Failed to process remote record: \(error)")
                    }
                }
                
            } catch {
                print("Failed to fetch remote changes for \(recordType): \(error)")
            }
        }
        
        pendingDownloads = 0
    }
    
    private func resolveConflicts() async {
        let conflicts = await detectConflicts()
        
        for conflict in conflicts {
            // Auto-resolve simple conflicts
            if canAutoResolve(conflict) {
                await autoResolveConflict(conflict)
            } else {
                // Add to manual resolution queue
                conflictResolutions.append(conflict)
            }
        }
    }
    
    private func getLocalChanges() async -> [LocalChange] {
        // Implementation to get local changes from Core Data
        return []
    }
    
    private func createCloudKitRecord(from change: LocalChange) async throws -> CKRecord {
        let record = CKRecord(recordType: change.recordType)
        
        // Map local data to CloudKit record
        for (key, value) in change.data {
            record[key] = value as? CKRecordValue
        }
        
        record["lastModified"] = change.timestamp
        record["deviceID"] = UIDevice.current.identifierForVendor?.uuidString
        
        return record
    }
    
    private func encryptRecord(_ record: CKRecord) async throws {
        guard encryptionEnabled else { return }
        
        for key in record.allKeys() {
            if let value = record[key] as? Data {
                let encryptedValue = try encryptionManager.encrypt(data: value)
                record[key] = encryptedValue
            } else if let value = record[key] as? String {
                let data = value.data(using: .utf8) ?? Data()
                let encryptedValue = try encryptionManager.encrypt(data: data)
                record[key] = encryptedValue
            }
        }
    }
    
    private func decryptRecord(_ record: CKRecord) async throws {
        guard encryptionEnabled else { return }
        
        for key in record.allKeys() {
            if let encryptedValue = record[key] as? Data {
                let decryptedValue = try encryptionManager.decrypt(data: encryptedValue)
                
                // Try to convert back to string if possible
                if let string = String(data: decryptedValue, encoding: .utf8) {
                    record[key] = string
                } else {
                    record[key] = decryptedValue
                }
            }
        }
    }
    
    private func fetchRemoteChanges(for recordType: String) async throws -> [CKRecord] {
        let query = CKQuery(recordType: recordType, predicate: NSPredicate(value: true))
        
        // Use change tokens for incremental sync
        let changeToken = getSyncToken(for: recordType)
        
        let (matchResults, _) = try await privateDatabase.records(matching: query)
        
        return matchResults.compactMap { (_, result) in
            switch result {
            case .success(let record):
                return record
            case .failure:
                return nil
            }
        }
    }
    
    private func processRemoteRecord(_ record: CKRecord) async {
        // Convert CloudKit record to local data model and save
        // Implementation depends on your data model
    }
    
    private func markAsUploaded(_ change: LocalChange) async {
        // Mark the local change as uploaded in Core Data
    }
    
    private func handleUploadError(_ change: LocalChange, error: Error) async {
        // Handle upload errors, implement retry logic
    }
    
    // MARK: - Conflict Resolution
    
    private func detectConflicts() async -> [SyncConflict] {
        // Implementation to detect conflicts between local and remote data
        return []
    }
    
    private func canAutoResolve(_ conflict: SyncConflict) -> Bool {
        // Simple auto-resolution rules
        switch conflict.type {
        case .timestampConflict:
            return true // Use most recent
        case .dataConflict:
            return false // Requires manual resolution
        case .deletionConflict:
            return false // Requires manual resolution
        }
    }
    
    private func autoResolveConflict(_ conflict: SyncConflict) async {
        switch conflict.type {
        case .timestampConflict:
            // Use the record with the most recent timestamp
            if conflict.localRecord.modificationDate ?? Date.distantPast > conflict.remoteRecord.modificationDate ?? Date.distantPast {
                await uploadRecord(conflict.localRecord)
            } else {
                await downloadRecord(conflict.remoteRecord)
            }
        default:
            break
        }
    }
    
    private func mergeRecords(local: CKRecord, remote: CKRecord) async -> CKRecord {
        // Implement intelligent record merging
        let merged = local.copy() as! CKRecord
        
        // Merge logic depends on record type and fields
        for key in remote.allKeys() {
            if merged[key] == nil {
                merged[key] = remote[key]
            }
        }
        
        merged["lastModified"] = Date()
        return merged
    }
    
    private func uploadRecord(_ record: CKRecord) async {
        do {
            _ = try await privateDatabase.save(record)
        } catch {
            print("Failed to upload record: \(error)")
        }
    }
    
    private func downloadRecord(_ record: CKRecord) async {
        await processRemoteRecord(record)
    }
    
    // MARK: - Backup Methods
    
    private func gatherBackupData() async -> Data {
        let backupData = BackupData(
            painEntries: await getAllPainEntries(),
            medications: await getAllMedications(),
            journalEntries: await getAllJournalEntries(),
            symptoms: await getAllSymptoms(),
            settings: await getAllSettings(),
            timestamp: Date()
        )
        
        do {
            return try JSONEncoder().encode(backupData)
        } catch {
            print("Failed to encode backup data: \(error)")
            return Data()
        }
    }
    
    private func restoreData(from backupData: BackupData) async {
        // Restore all data types from backup
        await restorePainEntries(backupData.painEntries)
        await restoreMedications(backupData.medications)
        await restoreJournalEntries(backupData.journalEntries)
        await restoreSymptoms(backupData.symptoms)
        await restoreSettings(backupData.settings)
    }
    
    // MARK: - Data Export Methods
    
    private func gatherAllData() async -> ExportData {
        return ExportData(
            painEntries: await getAllPainEntries(),
            medications: await getAllMedications(),
            journalEntries: await getAllJournalEntries(),
            symptoms: await getAllSymptoms(),
            healthMetrics: await getAllHealthMetrics(),
            exportDate: Date()
        )
    }
    
    private func convertToCSV(_ data: ExportData) -> Data {
        var csv = "Type,Date,Value,Notes\n"
        
        // Convert pain entries
        for entry in data.painEntries {
            csv += "Pain,\(entry.date),\(entry.level),\"\(entry.notes)\"\n"
        }
        
        // Convert other data types...
        
        return csv.data(using: .utf8) ?? Data()
    }
    
    private func generatePDF(from data: ExportData) async -> Data {
        // Implementation for PDF generation
        return Data()
    }
    
    private func convertToHL7FHIR(_ data: ExportData) throws -> Data {
        // Implementation for HL7 FHIR format conversion
        return Data()
    }
    
    // MARK: - Utility Methods
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isCloudAvailable = path.status == .satisfied
            }
        }
        
        let queue = DispatchQueue(label: "NetworkMonitor")
        networkMonitor.start(queue: queue)
    }
    
    private func setupPeriodicSync() {
        syncTimer = Timer.scheduledTimer(withTimeInterval: syncInterval, repeats: true) { [weak self] _ in
            Task {
                await self?.startSync()
            }
        }
    }
    
    private func setupAutomaticBackup() {
        backupTimer = Timer.scheduledTimer(withTimeInterval: backupInterval, repeats: true) { [weak self] _ in
            Task {
                _ = await self?.createBackup()
            }
        }
    }
    
    private func checkCloudKitAvailability() {
        container.accountStatus { [weak self] status, error in
            DispatchQueue.main.async {
                self?.isCloudAvailable = status == .available
            }
        }
    }
    
    private func getSyncToken(for recordType: String) -> CKServerChangeToken? {
        let key = "SyncToken_\(recordType)"
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        return try? NSKeyedUnarchiver.unarchivedObject(ofClass: CKServerChangeToken.self, from: data)
    }
    
    private func saveSyncToken(_ token: CKServerChangeToken, for recordType: String) {
        let key = "SyncToken_\(recordType)"
        let data = try? NSKeyedArchiver.archivedData(withRootObject: token, requiringSecureCoding: true)
        UserDefaults.standard.set(data, forKey: key)
    }
    
    private func clearSyncTokens() {
        for recordType in recordTypes {
            let key = "SyncToken_\(recordType)"
            UserDefaults.standard.removeObject(forKey: key)
        }
    }
    
    private func calculateRecordSize(_ record: CKRecord) -> Int64 {
        var size: Int64 = 0
        
        for key in record.allKeys() {
            if let data = record[key] as? Data {
                size += Int64(data.count)
            } else if let string = record[key] as? String {
                size += Int64(string.utf8.count)
            }
        }
        
        return size
    }
    
    // MARK: - Data Fetching Placeholder Methods
    
    private func getAllPainEntries() async -> [PainEntryData] {
        // Implementation to fetch all pain entries from Core Data
        return []
    }
    
    private func getAllMedications() async -> [MedicationData] {
        // Implementation to fetch all medications from Core Data
        return []
    }
    
    private func getAllJournalEntries() async -> [JournalEntryData] {
        // Implementation to fetch all journal entries from Core Data
        return []
    }
    
    private func getAllSymptoms() async -> [SymptomData] {
        // Implementation to fetch all symptoms from Core Data
        return []
    }
    
    private func getAllSettings() async -> [SettingData] {
        // Implementation to fetch all settings
        return []
    }
    
    private func getAllHealthMetrics() async -> [HealthMetricData] {
        // Implementation to fetch all health metrics
        return []
    }
    
    private func restorePainEntries(_ entries: [PainEntryData]) async {
        // Implementation to restore pain entries to Core Data
    }
    
    private func restoreMedications(_ medications: [MedicationData]) async {
        // Implementation to restore medications to Core Data
    }
    
    private func restoreJournalEntries(_ entries: [JournalEntryData]) async {
        // Implementation to restore journal entries to Core Data
    }
    
    private func restoreSymptoms(_ symptoms: [SymptomData]) async {
        // Implementation to restore symptoms to Core Data
    }
    
    private func restoreSettings(_ settings: [SettingData]) async {
        // Implementation to restore settings
    }
}

// MARK: - Encryption Manager

class EncryptionManager {
    private let key: SymmetricKey
    
    init() {
        // Generate or retrieve encryption key
        if let keyData = Keychain.load(key: "CloudSyncEncryptionKey") {
            self.key = SymmetricKey(data: keyData)
        } else {
            self.key = SymmetricKey(size: .bits256)
            let keyData = key.withUnsafeBytes { Data($0) }
            Keychain.save(key: "CloudSyncEncryptionKey", data: keyData)
        }
    }
    
    func encrypt(data: Data) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined!
    }
    
    func decrypt(data: Data) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: data)
        return try AES.GCM.open(sealedBox, using: key)
    }
}

// MARK: - Supporting Types

enum SyncStatus: Equatable {
    case idle
    case syncing
    case completed
    case failed(Error)
    
    static func == (lhs: SyncStatus, rhs: SyncStatus) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.syncing, .syncing), (.completed, .completed):
            return true
        case (.failed, .failed):
            return true
        default:
            return false
        }
    }
}

enum BackupStatus {
    case none
    case inProgress
    case completed(Date)
    case failed(Error)
}

enum BackupResult {
    case success(CKRecord.ID)
    case failure(Error)
}

enum RestoreResult {
    case success
    case failure(Error)
}

enum ConflictResolution {
    case useLocal
    case useRemote
    case merge
}

enum ConflictType {
    case timestampConflict
    case dataConflict
    case deletionConflict
}

enum ExportFormat {
    case json
    case csv
    case pdf
    case hl7fhir
}

enum CloudSyncError: Error {
    case invalidBackupData
    case encryptionFailed
    case networkUnavailable
    case quotaExceeded
    case authenticationFailed
}

struct SyncConflict: Identifiable {
    let id = UUID()
    let type: ConflictType
    let localRecord: CKRecord
    let remoteRecord: CKRecord
    let timestamp: Date
}

struct BackupInfo: Identifiable {
    let id: CKRecord.ID
    let timestamp: Date
    let version: String
    let size: Int
}

struct LocalChange {
    let id: UUID
    let recordType: String
    let data: [String: Any]
    let timestamp: Date
    let operation: ChangeOperation
}

enum ChangeOperation {
    case create
    case update
    case delete
}

// MARK: - Data Models

struct BackupData: Codable {
    let painEntries: [PainEntryData]
    let medications: [MedicationData]
    let journalEntries: [JournalEntryData]
    let symptoms: [SymptomData]
    let settings: [SettingData]
    let timestamp: Date
}

struct ExportData: Codable {
    let painEntries: [PainEntryData]
    let medications: [MedicationData]
    let journalEntries: [JournalEntryData]
    let symptoms: [SymptomData]
    let healthMetrics: [HealthMetricData]
    let exportDate: Date
}

struct PainEntryData: Codable {
    let id: UUID
    let date: Date
    let level: Int
    let location: String
    let notes: String
}

struct MedicationData: Codable {
    let id: UUID
    let name: String
    let dosage: String
    let frequency: String
    let startDate: Date
    let endDate: Date?
}

struct JournalEntryData: Codable {
    let id: UUID
    let date: Date
    let title: String
    let content: String
    let mood: String
}

struct SymptomData: Codable {
    let id: UUID
    let date: Date
    let type: String
    let severity: Int
    let notes: String
}

struct SettingData: Codable {
    let key: String
    let value: String
    let timestamp: Date
}

struct HealthMetricData: Codable {
    let id: UUID
    let date: Date
    let type: String
    let value: Double
    let unit: String
}

// MARK: - Keychain Helper

class Keychain {
    static func save(key: String, data: Data) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data
        ]
        
        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }
    
    static func load(key: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var dataTypeRef: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &dataTypeRef)
        
        if status == errSecSuccess {
            return dataTypeRef as? Data
        }
        
        return nil
    }
    
    static func delete(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        
        SecItemDelete(query as CFDictionary)
    }
}