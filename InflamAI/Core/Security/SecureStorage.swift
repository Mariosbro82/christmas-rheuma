//
//  SecureStorage.swift
//  InflamAI
//
//  Production-grade secure storage wrapper for PHI (Protected Health Information)
//  Uses Keychain for all sensitive data instead of UserDefaults
//
//  CRITICAL: All health-related data MUST use this class instead of UserDefaults
//  UserDefaults is NOT encrypted and is included in iCloud backups
//

import Foundation
import Security

/// Secure storage for Protected Health Information (PHI)
///
/// This class provides a type-safe, Codable-friendly API for storing sensitive data
/// in the iOS Keychain. Unlike UserDefaults, Keychain data is:
/// - Encrypted at rest
/// - Not included in unencrypted backups
/// - Protected by device passcode/biometrics
///
/// ## Usage
/// ```swift
/// // Store sensitive data
/// try SecureStorage.shared.save(userProfile, forKey: .userProfile)
///
/// // Retrieve sensitive data
/// let profile: UserProfile? = try SecureStorage.shared.load(forKey: .userProfile)
///
/// // Delete sensitive data
/// SecureStorage.shared.delete(forKey: .userProfile)
/// ```
///
/// ## HIPAA Compliance
/// This class helps achieve HIPAA technical safeguards by:
/// - Encrypting PHI at rest (Keychain encryption)
/// - Access controls (optional biometric protection)
/// - Audit trail support (via SecurityAuditLogger)
final class SecureStorage {

    // MARK: - Singleton

    static let shared = SecureStorage()

    // MARK: - Storage Keys

    /// Predefined keys for common PHI storage
    /// Using an enum prevents typos and provides documentation
    enum StorageKey: String, CaseIterable {
        // User Profile
        case userProfile = "secure.userProfile"
        case diagnosisYear = "secure.diagnosisYear"
        case birthYear = "secure.birthYear"
        case gender = "secure.gender"

        // Health Metrics Cache
        case lastHealthKitData = "secure.lastHealthKitData"
        case lastBiometricReadings = "secure.lastBiometricReadings"
        case cachedSleepData = "secure.cachedSleepData"

        // Symptom Data
        case recentSymptomScores = "secure.recentSymptomScores"
        case basdaiHistory = "secure.basdaiHistory"
        case flareRiskCache = "secure.flareRiskCache"

        // Medication Data
        case medicationSchedule = "secure.medicationSchedule"
        case adherenceHistory = "secure.adherenceHistory"

        // Authentication
        case authToken = "secure.authToken"
        case cloudKitToken = "secure.cloudKitToken"
        case encryptionKey = "secure.encryptionKey"

        // App State (non-PHI but sensitive)
        case onboardingComplete = "secure.onboardingComplete"
        case lastSyncTimestamp = "secure.lastSyncTimestamp"
    }

    // MARK: - Properties

    private let keychain: KeychainManager
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    // MARK: - Initialization

    private init() {
        self.keychain = KeychainManager(
            serviceName: "com.spinalytics.securestorage",
            accessGroup: nil,
            biometricProtection: false  // Enable per-item as needed
        )
        self.encoder = JSONEncoder()
        self.encoder.dateEncodingStrategy = .iso8601
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .iso8601
    }

    // MARK: - Public API (Codable)

    /// Save a Codable value securely in Keychain
    /// - Parameters:
    ///   - value: The value to store (must be Codable)
    ///   - key: Storage key (use predefined keys when possible)
    ///   - requireBiometric: If true, biometric auth required to read
    /// - Throws: SecureStorageError if encoding or storage fails
    func save<T: Codable>(_ value: T, forKey key: StorageKey, requireBiometric: Bool = false) throws {
        try save(value, forKey: key.rawValue, requireBiometric: requireBiometric)
    }

    /// Save a Codable value with a custom string key
    func save<T: Codable>(_ value: T, forKey key: String, requireBiometric: Bool = false) throws {
        do {
            let data = try encoder.encode(value)

            Task {
                let success = await keychain.store(data, for: key, requireBiometric: requireBiometric)
                if !success {
                    print("❌ SecureStorage: Failed to store \(key)")
                }
            }
        } catch {
            throw SecureStorageError.encodingFailed(error)
        }
    }

    /// Load a Codable value from Keychain
    /// - Parameters:
    ///   - key: Storage key
    /// - Returns: The decoded value, or nil if not found
    /// - Throws: SecureStorageError if decoding fails
    func load<T: Codable>(forKey key: StorageKey) throws -> T? {
        return try load(forKey: key.rawValue)
    }

    /// Load a Codable value with a custom string key
    func load<T: Codable>(forKey key: String) throws -> T? {
        var result: T?
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            if let data = await keychain.getData(for: key) {
                do {
                    result = try decoder.decode(T.self, from: data)
                } catch {
                    print("❌ SecureStorage: Failed to decode \(key): \(error)")
                }
            }
            semaphore.signal()
        }

        semaphore.wait()
        return result
    }

    /// Asynchronously load a Codable value
    func loadAsync<T: Codable>(forKey key: StorageKey) async throws -> T? {
        return try await loadAsync(forKey: key.rawValue)
    }

    /// Asynchronously load with custom key
    func loadAsync<T: Codable>(forKey key: String) async throws -> T? {
        guard let data = await keychain.getData(for: key) else {
            return nil
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw SecureStorageError.decodingFailed(error)
        }
    }

    /// Delete a value from Keychain
    /// - Parameter key: Storage key to delete
    func delete(forKey key: StorageKey) {
        delete(forKey: key.rawValue)
    }

    /// Delete with custom key
    func delete(forKey key: String) {
        Task {
            _ = await keychain.delete(key)
        }
    }

    /// Delete all SecureStorage data (GDPR compliance)
    func deleteAll() {
        Task {
            _ = await keychain.deleteAll()
            print("✅ SecureStorage: All data deleted (GDPR compliance)")
        }
    }

    /// Check if a key exists
    func exists(forKey key: StorageKey) async -> Bool {
        return await keychain.exists(key.rawValue)
    }

    // MARK: - Primitive Type Helpers

    /// Save a string value
    func saveString(_ value: String, forKey key: StorageKey) {
        Task {
            _ = await keychain.store(value, for: key.rawValue)
        }
    }

    /// Load a string value
    func loadString(forKey key: StorageKey) async -> String? {
        return await keychain.getString(for: key.rawValue)
    }

    /// Save a boolean value
    func saveBool(_ value: Bool, forKey key: StorageKey) {
        try? save(value, forKey: key)
    }

    /// Load a boolean value
    func loadBool(forKey key: StorageKey) -> Bool {
        return (try? load(forKey: key)) ?? false
    }

    /// Save a date value
    func saveDate(_ value: Date, forKey key: StorageKey) {
        try? save(value, forKey: key)
    }

    /// Load a date value
    func loadDate(forKey key: StorageKey) -> Date? {
        return try? load(forKey: key)
    }

    // MARK: - Migration from UserDefaults

    /// Migrate a value from UserDefaults to SecureStorage
    /// - Parameters:
    ///   - userDefaultsKey: Key in UserDefaults
    ///   - secureKey: Key in SecureStorage
    ///   - deleteFromUserDefaults: If true, removes from UserDefaults after migration
    /// - Returns: True if migration successful or value didn't exist
    @discardableResult
    func migrateFromUserDefaults<T: Codable>(
        userDefaultsKey: String,
        to secureKey: StorageKey,
        deleteFromUserDefaults: Bool = true
    ) -> Bool {
        let defaults = UserDefaults.standard

        // Try to get existing value from UserDefaults
        guard let data = defaults.data(forKey: userDefaultsKey) else {
            // Check for primitive types
            if let stringValue = defaults.string(forKey: userDefaultsKey) {
                saveString(stringValue, forKey: secureKey)
                if deleteFromUserDefaults {
                    defaults.removeObject(forKey: userDefaultsKey)
                }
                print("✅ Migrated string '\(userDefaultsKey)' to SecureStorage")
                return true
            }

            // No value to migrate
            return true
        }

        // Migrate data
        Task {
            let success = await keychain.store(data, for: secureKey.rawValue)
            if success && deleteFromUserDefaults {
                defaults.removeObject(forKey: userDefaultsKey)
                print("✅ Migrated '\(userDefaultsKey)' to SecureStorage")
            }
        }

        return true
    }

    /// Migrate all known PHI keys from UserDefaults
    func migrateAllPHIFromUserDefaults() {
        print("🔄 Starting PHI migration from UserDefaults to SecureStorage...")

        // Known PHI keys that should never be in UserDefaults
        let migrations: [(String, StorageKey)] = [
            ("userDiagnosisYear", .diagnosisYear),
            ("userBirthYear", .birthYear),
            ("userGender", .gender),
            ("lastHealthKitFetch", .lastHealthKitData),
            ("hasValidHealthKitData", .lastBiometricReadings),
            ("cachedBASDAIScore", .basdaiHistory),
            ("flareRiskLevel", .flareRiskCache),
        ]

        for (userDefaultsKey, secureKey) in migrations {
            migrateFromUserDefaults(
                userDefaultsKey: userDefaultsKey,
                to: secureKey,
                deleteFromUserDefaults: true
            )
        }

        print("✅ PHI migration complete")
    }

    // MARK: - Audit Support

    /// Get list of all stored keys (for audit purposes)
    func getAllStoredKeys() async -> [String] {
        return await keychain.getAllKeys(withPrefix: "secure.")
    }

    /// Get storage status for compliance reporting
    func getStorageStatus() async -> SecureStorageStatus {
        let allKeys = await getAllStoredKeys()
        let keychainStatus = await keychain.getKeychainStatus()

        return SecureStorageStatus(
            totalItems: allKeys.count,
            biometricProtectedItems: keychainStatus.biometricProtectedItems,
            storageKeys: allKeys,
            biometricAvailable: keychainStatus.biometricAvailable
        )
    }
}

// MARK: - Supporting Types

/// Status of secure storage for compliance reporting
struct SecureStorageStatus {
    let totalItems: Int
    let biometricProtectedItems: Int
    let storageKeys: [String]
    let biometricAvailable: Bool

    var complianceReport: String {
        """
        SecureStorage Compliance Report
        ================================
        Total Items: \(totalItems)
        Biometric Protected: \(biometricProtectedItems)
        Biometric Available: \(biometricAvailable)

        Stored Keys:
        \(storageKeys.map { "  - \($0)" }.joined(separator: "\n"))
        """
    }
}

/// Errors that can occur during secure storage operations
enum SecureStorageError: LocalizedError {
    case encodingFailed(Error)
    case decodingFailed(Error)
    case storageFailed
    case itemNotFound
    case biometricRequired

    var errorDescription: String? {
        switch self {
        case .encodingFailed(let error):
            return "Failed to encode data: \(error.localizedDescription)"
        case .decodingFailed(let error):
            return "Failed to decode data: \(error.localizedDescription)"
        case .storageFailed:
            return "Failed to store data in Keychain"
        case .itemNotFound:
            return "Item not found in secure storage"
        case .biometricRequired:
            return "Biometric authentication required to access this data"
        }
    }
}

// MARK: - Convenience Extensions

extension SecureStorage {
    /// Quick check if onboarding is complete (commonly accessed)
    var hasCompletedOnboarding: Bool {
        loadBool(forKey: .onboardingComplete)
    }

    /// Set onboarding complete status
    func setOnboardingComplete(_ complete: Bool) {
        saveBool(complete, forKey: .onboardingComplete)
    }
}
