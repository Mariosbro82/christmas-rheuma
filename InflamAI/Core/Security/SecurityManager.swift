//
//  SecurityManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import LocalAuthentication
import CryptoKit
import Security
import Combine
import OSLog

// MARK: - Security Manager
@MainActor
class SecurityManager: ObservableObject {
    
    // MARK: - Singleton
    static let shared = SecurityManager()
    
    // MARK: - Published Properties
    @Published var isAuthenticated = false
    @Published var biometricType: BiometricType = .none
    @Published var isEncryptionEnabled = true
    @Published var securityLevel: SecurityLevel = .standard
    @Published var lastAuthenticationDate: Date?
    @Published var failedAttempts = 0
    @Published var isLocked = false
    @Published var privacySettings = PrivacySettings()
    
    // MARK: - Private Properties
    private let logger = Logger(subsystem: "com.inflamai", category: "Security")
    private let context = LAContext()
    private let keychain = KeychainManager()
    private let encryption = EncryptionManager()
    
    private var cancellables = Set<AnyCancellable>()
    private var lockTimer: Timer?
    private var authenticationTimer: Timer?
    
    // MARK: - Constants
    private let maxFailedAttempts = 5
    private let lockoutDuration: TimeInterval = 300 // 5 minutes
    private let authenticationTimeout: TimeInterval = 900 // 15 minutes
    private let masterKeyTag = "com.inflamai.masterkey"
    private let encryptionKeyTag = "com.inflamai.encryptionkey"
    
    // MARK: - Initialization
    private init() {
        setupSecurity()
        checkBiometricAvailability()
        loadSecuritySettings()
        setupAutoLock()
    }
    
    deinit {
        lockTimer?.invalidate()
        authenticationTimer?.invalidate()
    }
    
    // MARK: - Authentication Methods
    
    func authenticateUser(reason: String = "Authenticate to access your health data") async -> AuthenticationResult {
        guard !isLocked else {
            return .failure(.accountLocked)
        }
        
        // Check if biometric authentication is available
        let biometricResult = await checkBiometricAuthentication()
        if biometricResult.isAvailable {
            return await performBiometricAuthentication(reason: reason)
        }
        
        // Fallback to passcode
        return await performPasscodeAuthentication()
    }
    
    func authenticateWithBiometrics(reason: String = "Authenticate to access your health data") async -> AuthenticationResult {
        guard !isLocked else {
            return .failure(.accountLocked)
        }
        
        return await performBiometricAuthentication(reason: reason)
    }
    
    func authenticateWithPasscode(_ passcode: String) async -> AuthenticationResult {
        guard !isLocked else {
            return .failure(.accountLocked)
        }
        
        return await performPasscodeAuthentication(passcode: passcode)
    }
    
    func logout() {
        isAuthenticated = false
        lastAuthenticationDate = nil
        authenticationTimer?.invalidate()
        
        // Clear sensitive data from memory
        clearSensitiveData()
        
        logger.info("User logged out")
    }
    
    func lockApp() {
        isAuthenticated = false
        authenticationTimer?.invalidate()
        
        logger.info("App locked")
    }
    
    func unlockApp() async -> Bool {
        let result = await authenticateUser(reason: "Unlock the app")
        return result.isSuccess
    }
    
    // MARK: - Biometric Methods
    
    func checkBiometricAvailability() {
        var error: NSError?
        let available = context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
        
        if available {
            switch context.biometryType {
            case .faceID:
                biometricType = .faceID
            case .touchID:
                biometricType = .touchID
            case .opticID:
                biometricType = .opticID
            default:
                biometricType = .none
            }
        } else {
            biometricType = .none
            logger.warning("Biometric authentication not available: \(error?.localizedDescription ?? "Unknown error")")
        }
    }
    
    func enableBiometricAuthentication() async -> Bool {
        let result = await authenticateWithBiometrics(reason: "Enable biometric authentication")
        if result.isSuccess {
            UserDefaults.standard.set(true, forKey: "BiometricAuthEnabled")
            logger.info("Biometric authentication enabled")
            return true
        }
        return false
    }
    
    func disableBiometricAuthentication() {
        UserDefaults.standard.set(false, forKey: "BiometricAuthEnabled")
        logger.info("Biometric authentication disabled")
    }
    
    // MARK: - Encryption Methods
    
    func encryptData(_ data: Data) throws -> EncryptedData {
        guard isEncryptionEnabled else {
            throw SecurityError.encryptionDisabled
        }
        
        return try encryption.encrypt(data)
    }
    
    func decryptData(_ encryptedData: EncryptedData) throws -> Data {
        guard isEncryptionEnabled else {
            throw SecurityError.encryptionDisabled
        }
        
        return try encryption.decrypt(encryptedData)
    }
    
    func encryptString(_ string: String) throws -> EncryptedData {
        guard let data = string.data(using: .utf8) else {
            throw SecurityError.invalidData
        }
        return try encryptData(data)
    }
    
    func decryptString(_ encryptedData: EncryptedData) throws -> String {
        let data = try decryptData(encryptedData)
        guard let string = String(data: data, encoding: .utf8) else {
            throw SecurityError.invalidData
        }
        return string
    }
    
    func generateSecureToken() -> String {
        return encryption.generateSecureToken()
    }
    
    func hashPassword(_ password: String, salt: Data? = nil) -> HashedPassword {
        return encryption.hashPassword(password, salt: salt)
    }
    
    // MARK: - Keychain Methods
    
    func storeSecureData(_ data: Data, for key: String) throws {
        try keychain.store(data, for: key)
    }
    
    func retrieveSecureData(for key: String) throws -> Data? {
        return try keychain.retrieve(for: key)
    }
    
    func deleteSecureData(for key: String) throws {
        try keychain.delete(for: key)
    }
    
    func storeCredentials(_ credentials: UserCredentials) throws {
        let data = try JSONEncoder().encode(credentials)
        let encryptedData = try encryptData(data)
        try keychain.store(encryptedData.data, for: "user_credentials")
    }
    
    func retrieveCredentials() throws -> UserCredentials? {
        guard let data = try keychain.retrieve(for: "user_credentials") else {
            return nil
        }
        
        let encryptedData = EncryptedData(data: data, iv: Data(), tag: Data())
        let decryptedData = try decryptData(encryptedData)
        return try JSONDecoder().decode(UserCredentials.self, from: decryptedData)
    }
    
    // MARK: - Privacy Methods
    
    func updatePrivacySettings(_ settings: PrivacySettings) {
        privacySettings = settings
        savePrivacySettings()
        logger.info("Privacy settings updated")
    }
    
    func anonymizeData<T: Anonymizable>(_ data: T) -> T {
        return data.anonymized()
    }
    
    func shouldShareData(for purpose: DataSharingPurpose) -> Bool {
        switch purpose {
        case .analytics:
            return privacySettings.allowAnalytics
        case .research:
            return privacySettings.allowResearch
        case .healthcare:
            return privacySettings.allowHealthcareSharing
        case .emergency:
            return privacySettings.allowEmergencySharing
        }
    }
    
    func requestDataSharingPermission(for purpose: DataSharingPurpose) async -> Bool {
        // Implementation would show permission dialog
        // For now, return current setting
        return shouldShareData(for: purpose)
    }
    
    // MARK: - Security Level Methods
    
    func setSecurityLevel(_ level: SecurityLevel) {
        securityLevel = level
        UserDefaults.standard.set(level.rawValue, forKey: "SecurityLevel")
        
        // Adjust security settings based on level
        switch level {
        case .basic:
            isEncryptionEnabled = false
        case .standard:
            isEncryptionEnabled = true
        case .high:
            isEncryptionEnabled = true
            // Enable additional security measures
        case .maximum:
            isEncryptionEnabled = true
            // Enable all security measures
        }
        
        logger.info("Security level set to \(level.rawValue)")
    }
    
    func validateSecurityCompliance() -> SecurityComplianceReport {
        var issues: [SecurityIssue] = []
        var recommendations: [SecurityRecommendation] = []
        
        // Check biometric authentication
        if biometricType == .none {
            issues.append(.biometricNotAvailable)
            recommendations.append(.enableBiometrics)
        }
        
        // Check encryption
        if !isEncryptionEnabled {
            issues.append(.encryptionDisabled)
            recommendations.append(.enableEncryption)
        }
        
        // Check passcode strength
        if !hasStrongPasscode() {
            issues.append(.weakPasscode)
            recommendations.append(.strengthenPasscode)
        }
        
        // Check auto-lock settings
        if !hasAutoLockEnabled() {
            issues.append(.autoLockDisabled)
            recommendations.append(.enableAutoLock)
        }
        
        let complianceLevel: ComplianceLevel
        if issues.isEmpty {
            complianceLevel = .compliant
        } else if issues.contains(where: { $0.severity == .high }) {
            complianceLevel = .nonCompliant
        } else {
            complianceLevel = .partiallyCompliant
        }
        
        return SecurityComplianceReport(
            level: complianceLevel,
            issues: issues,
            recommendations: recommendations,
            lastChecked: Date()
        )
    }
    
    // MARK: - HIPAA Compliance Methods
    
    func enableHIPAACompliance() {
        setSecurityLevel(.maximum)
        
        // Enable audit logging
        privacySettings.enableAuditLogging = true
        
        // Enable data minimization
        privacySettings.enableDataMinimization = true
        
        // Disable analytics by default
        privacySettings.allowAnalytics = false
        
        updatePrivacySettings(privacySettings)
        
        logger.info("HIPAA compliance enabled")
    }
    
    func logDataAccess(_ access: DataAccessLog) {
        guard privacySettings.enableAuditLogging else { return }
        
        // Store audit log
        let logData = try? JSONEncoder().encode(access)
        if let data = logData {
            try? storeSecureData(data, for: "audit_log_\(access.id)")
        }
        
        logger.info("Data access logged: \(access.description)")
    }
    
    func getAuditLogs(from startDate: Date, to endDate: Date) -> [DataAccessLog] {
        // Implementation would retrieve audit logs from secure storage
        // For now, return empty array
        return []
    }
    
    // MARK: - Emergency Access Methods
    
    func enableEmergencyAccess(for contacts: [EmergencyContact]) {
        privacySettings.emergencyContacts = contacts
        privacySettings.allowEmergencySharing = true
        updatePrivacySettings(privacySettings)
        
        logger.info("Emergency access enabled for \(contacts.count) contacts")
    }
    
    func grantEmergencyAccess(to contactId: String) -> EmergencyAccessToken? {
        guard let contact = privacySettings.emergencyContacts.first(where: { $0.id == contactId }) else {
            return nil
        }
        
        let token = EmergencyAccessToken(
            contactId: contactId,
            token: generateSecureToken(),
            expiresAt: Date().addingTimeInterval(3600), // 1 hour
            permissions: [.readHealthData, .readMedications, .readEmergencyInfo]
        )
        
        // Store token securely
        let tokenData = try? JSONEncoder().encode(token)
        if let data = tokenData {
            try? storeSecureData(data, for: "emergency_token_\(token.id)")
        }
        
        logger.info("Emergency access granted to \(contact.name)")
        return token
    }
    
    func revokeEmergencyAccess(tokenId: String) {
        try? deleteSecureData(for: "emergency_token_\(tokenId)")
        logger.info("Emergency access revoked for token \(tokenId)")
    }
    
    // MARK: - Private Methods
    
    private func setupSecurity() {
        // Initialize encryption keys
        encryption.initializeKeys()
        
        // Set up authentication timeout
        setupAuthenticationTimeout()
        
        // Load failed attempts
        failedAttempts = UserDefaults.standard.integer(forKey: "FailedAttempts")
        
        // Check if account is locked
        if let lockDate = UserDefaults.standard.object(forKey: "LockDate") as? Date {
            if Date().timeIntervalSince(lockDate) < lockoutDuration {
                isLocked = true
                startLockTimer(until: lockDate.addingTimeInterval(lockoutDuration))
            } else {
                unlockAccount()
            }
        }
    }
    
    private func loadSecuritySettings() {
        let levelRawValue = UserDefaults.standard.string(forKey: "SecurityLevel") ?? SecurityLevel.standard.rawValue
        securityLevel = SecurityLevel(rawValue: levelRawValue) ?? .standard
        
        isEncryptionEnabled = UserDefaults.standard.bool(forKey: "EncryptionEnabled")
        if UserDefaults.standard.object(forKey: "EncryptionEnabled") == nil {
            isEncryptionEnabled = true // Default to enabled
        }
        
        loadPrivacySettings()
    }
    
    private func loadPrivacySettings() {
        if let data = UserDefaults.standard.data(forKey: "PrivacySettings"),
           let settings = try? JSONDecoder().decode(PrivacySettings.self, from: data) {
            privacySettings = settings
        }
    }
    
    private func savePrivacySettings() {
        if let data = try? JSONEncoder().encode(privacySettings) {
            UserDefaults.standard.set(data, forKey: "PrivacySettings")
        }
    }
    
    private func setupAutoLock() {
        // Set up auto-lock timer based on security level
        let timeout: TimeInterval
        switch securityLevel {
        case .basic:
            timeout = 1800 // 30 minutes
        case .standard:
            timeout = 900 // 15 minutes
        case .high:
            timeout = 300 // 5 minutes
        case .maximum:
            timeout = 60 // 1 minute
        }
        
        lockTimer?.invalidate()
        lockTimer = Timer.scheduledTimer(withTimeInterval: timeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.lockApp()
            }
        }
    }
    
    private func setupAuthenticationTimeout() {
        authenticationTimer?.invalidate()
        authenticationTimer = Timer.scheduledTimer(withTimeInterval: authenticationTimeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.logout()
            }
        }
    }
    
    private func checkBiometricAuthentication() async -> BiometricAvailability {
        return await withCheckedContinuation { continuation in
            var error: NSError?
            let available = context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
            
            let result = BiometricAvailability(
                isAvailable: available,
                biometryType: context.biometryType,
                error: error
            )
            
            continuation.resume(returning: result)
        }
    }
    
    private func performBiometricAuthentication(reason: String) async -> AuthenticationResult {
        guard UserDefaults.standard.bool(forKey: "BiometricAuthEnabled") else {
            return .failure(.biometricNotEnabled)
        }
        
        return await withCheckedContinuation { continuation in
            context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) { [weak self] success, error in
                Task { @MainActor in
                    if success {
                        self?.handleSuccessfulAuthentication()
                        continuation.resume(returning: .success)
                    } else {
                        let authError = self?.handleAuthenticationError(error)
                        continuation.resume(returning: .failure(authError ?? .unknown))
                    }
                }
            }
        }
    }
    
    private func performPasscodeAuthentication(passcode: String? = nil) async -> AuthenticationResult {
        // If no passcode provided, use system passcode authentication
        if passcode == nil {
            return await withCheckedContinuation { continuation in
                context.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: "Authenticate with passcode") { [weak self] success, error in
                    Task { @MainActor in
                        if success {
                            self?.handleSuccessfulAuthentication()
                            continuation.resume(returning: .success)
                        } else {
                            let authError = self?.handleAuthenticationError(error)
                            continuation.resume(returning: .failure(authError ?? .unknown))
                        }
                    }
                }
            }
        }
        
        // Validate provided passcode
        guard let passcode = passcode,
              let storedHash = try? retrieveSecureData(for: "passcode_hash"),
              let hashData = try? JSONDecoder().decode(HashedPassword.self, from: storedHash) else {
            handleFailedAuthentication()
            return .failure(.invalidCredentials)
        }
        
        let inputHash = hashPassword(passcode, salt: hashData.salt)
        if inputHash.hash == hashData.hash {
            handleSuccessfulAuthentication()
            return .success
        } else {
            handleFailedAuthentication()
            return .failure(.invalidCredentials)
        }
    }
    
    private func handleSuccessfulAuthentication() {
        isAuthenticated = true
        lastAuthenticationDate = Date()
        failedAttempts = 0
        isLocked = false
        
        UserDefaults.standard.set(0, forKey: "FailedAttempts")
        UserDefaults.standard.removeObject(forKey: "LockDate")
        
        setupAuthenticationTimeout()
        setupAutoLock()
        
        logger.info("Authentication successful")
    }
    
    private func handleFailedAuthentication() {
        failedAttempts += 1
        UserDefaults.standard.set(failedAttempts, forKey: "FailedAttempts")
        
        if failedAttempts >= maxFailedAttempts {
            lockAccount()
        }
        
        logger.warning("Authentication failed. Attempts: \(failedAttempts)")
    }
    
    private func handleAuthenticationError(_ error: Error?) -> AuthenticationError {
        guard let error = error as? LAError else {
            return .unknown
        }
        
        switch error.code {
        case .userCancel:
            return .userCancelled
        case .userFallback:
            return .userFallback
        case .biometryNotAvailable:
            return .biometricNotAvailable
        case .biometryNotEnrolled:
            return .biometricNotEnrolled
        case .biometryLockout:
            return .biometricLockout
        case .authenticationFailed:
            handleFailedAuthentication()
            return .authenticationFailed
        default:
            return .unknown
        }
    }
    
    private func lockAccount() {
        isLocked = true
        isAuthenticated = false
        
        let lockDate = Date()
        UserDefaults.standard.set(lockDate, forKey: "LockDate")
        
        startLockTimer(until: lockDate.addingTimeInterval(lockoutDuration))
        
        logger.warning("Account locked due to too many failed attempts")
    }
    
    private func unlockAccount() {
        isLocked = false
        failedAttempts = 0
        
        UserDefaults.standard.set(0, forKey: "FailedAttempts")
        UserDefaults.standard.removeObject(forKey: "LockDate")
        
        lockTimer?.invalidate()
        
        logger.info("Account unlocked")
    }
    
    private func startLockTimer(until unlockDate: Date) {
        lockTimer?.invalidate()
        
        let timeInterval = unlockDate.timeIntervalSinceNow
        if timeInterval > 0 {
            lockTimer = Timer.scheduledTimer(withTimeInterval: timeInterval, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    self?.unlockAccount()
                }
            }
        } else {
            unlockAccount()
        }
    }
    
    private func clearSensitiveData() {
        // Clear any sensitive data from memory
        // This would include clearing caches, temporary data, etc.
    }
    
    private func hasStrongPasscode() -> Bool {
        // Check if a strong passcode is set
        // This would involve checking passcode complexity
        return true // Placeholder
    }
    
    private func hasAutoLockEnabled() -> Bool {
        // Check if auto-lock is enabled
        return lockTimer != nil
    }
}

// MARK: - Supporting Types

enum BiometricType {
    case none
    case touchID
    case faceID
    case opticID
    
    var displayName: String {
        switch self {
        case .none:
            return "None"
        case .touchID:
            return "Touch ID"
        case .faceID:
            return "Face ID"
        case .opticID:
            return "Optic ID"
        }
    }
}

enum AuthenticationResult {
    case success
    case failure(AuthenticationError)
    
    var isSuccess: Bool {
        if case .success = self {
            return true
        }
        return false
    }
}

enum AuthenticationError: Error, LocalizedError {
    case biometricNotAvailable
    case biometricNotEnrolled
    case biometricNotEnabled
    case biometricLockout
    case userCancelled
    case userFallback
    case authenticationFailed
    case invalidCredentials
    case accountLocked
    case unknown
    
    var errorDescription: String? {
        switch self {
        case .biometricNotAvailable:
            return "Biometric authentication is not available on this device"
        case .biometricNotEnrolled:
            return "No biometric data is enrolled on this device"
        case .biometricNotEnabled:
            return "Biometric authentication is not enabled for this app"
        case .biometricLockout:
            return "Biometric authentication is locked out"
        case .userCancelled:
            return "Authentication was cancelled by the user"
        case .userFallback:
            return "User chose to use fallback authentication"
        case .authenticationFailed:
            return "Authentication failed"
        case .invalidCredentials:
            return "Invalid credentials provided"
        case .accountLocked:
            return "Account is locked due to too many failed attempts"
        case .unknown:
            return "An unknown authentication error occurred"
        }
    }
}

enum SecurityError: Error, LocalizedError {
    case encryptionDisabled
    case invalidData
    case keyGenerationFailed
    case encryptionFailed
    case decryptionFailed
    
    var errorDescription: String? {
        switch self {
        case .encryptionDisabled:
            return "Encryption is disabled"
        case .invalidData:
            return "Invalid data provided"
        case .keyGenerationFailed:
            return "Failed to generate encryption key"
        case .encryptionFailed:
            return "Failed to encrypt data"
        case .decryptionFailed:
            return "Failed to decrypt data"
        }
    }
}

enum SecurityLevel: String, CaseIterable {
    case basic = "basic"
    case standard = "standard"
    case high = "high"
    case maximum = "maximum"
    
    var displayName: String {
        switch self {
        case .basic:
            return "Basic"
        case .standard:
            return "Standard"
        case .high:
            return "High"
        case .maximum:
            return "Maximum"
        }
    }
    
    var description: String {
        switch self {
        case .basic:
            return "Basic security with minimal encryption"
        case .standard:
            return "Standard security with encryption enabled"
        case .high:
            return "High security with enhanced protection"
        case .maximum:
            return "Maximum security with all features enabled"
        }
    }
}

struct BiometricAvailability {
    let isAvailable: Bool
    let biometryType: LABiometryType
    let error: NSError?
}

struct EncryptedData {
    let data: Data
    let iv: Data
    let tag: Data
}

struct HashedPassword: Codable {
    let hash: Data
    let salt: Data
    let iterations: Int
}

struct UserCredentials: Codable {
    let username: String
    let passwordHash: HashedPassword
    let createdAt: Date
    let lastUpdated: Date
}

struct PrivacySettings: Codable {
    var allowAnalytics = false
    var allowResearch = false
    var allowHealthcareSharing = true
    var allowEmergencySharing = true
    var enableAuditLogging = true
    var enableDataMinimization = true
    var dataRetentionDays = 365
    var emergencyContacts: [EmergencyContact] = []
}

struct EmergencyContact: Codable, Identifiable {
    let id = UUID().uuidString
    let name: String
    let relationship: String
    let phoneNumber: String
    let email: String?
    let isHealthcareProvider: Bool
}

enum DataSharingPurpose {
    case analytics
    case research
    case healthcare
    case emergency
}

struct DataAccessLog: Codable, Identifiable {
    let id = UUID().uuidString
    let userId: String
    let dataType: String
    let action: DataAccessAction
    let timestamp: Date
    let ipAddress: String?
    let userAgent: String?
    let purpose: String?
    
    var description: String {
        return "\(action.rawValue) \(dataType) at \(timestamp)"
    }
}

enum DataAccessAction: String, Codable {
    case read = "read"
    case write = "write"
    case delete = "delete"
    case export = "export"
    case share = "share"
}

struct EmergencyAccessToken: Codable, Identifiable {
    let id = UUID().uuidString
    let contactId: String
    let token: String
    let expiresAt: Date
    let permissions: [EmergencyPermission]
    let createdAt = Date()
}

enum EmergencyPermission: String, Codable {
    case readHealthData = "readHealthData"
    case readMedications = "readMedications"
    case readEmergencyInfo = "readEmergencyInfo"
    case contactEmergencyServices = "contactEmergencyServices"
}

struct SecurityComplianceReport {
    let level: ComplianceLevel
    let issues: [SecurityIssue]
    let recommendations: [SecurityRecommendation]
    let lastChecked: Date
}

enum ComplianceLevel {
    case compliant
    case partiallyCompliant
    case nonCompliant
}

enum SecurityIssue {
    case biometricNotAvailable
    case encryptionDisabled
    case weakPasscode
    case autoLockDisabled
    case auditLoggingDisabled
    
    var severity: SecurityIssueSeverity {
        switch self {
        case .encryptionDisabled, .auditLoggingDisabled:
            return .high
        case .biometricNotAvailable, .autoLockDisabled:
            return .medium
        case .weakPasscode:
            return .low
        }
    }
}

enum SecurityIssueSeverity {
    case low
    case medium
    case high
}

enum SecurityRecommendation {
    case enableBiometrics
    case enableEncryption
    case strengthenPasscode
    case enableAutoLock
    case enableAuditLogging
}

// MARK: - Anonymizable Protocol

protocol Anonymizable {
    func anonymized() -> Self
}

// MARK: - Keychain Manager

class KeychainManager {
    private let service = "com.inflamai.keychain"
    
    func store(_ data: Data, for key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Delete existing item
        SecItemDelete(query as CFDictionary)
        
        // Add new item
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw SecurityError.encryptionFailed
        }
    }
    
    func retrieve(for key: String) throws -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess else {
            if status == errSecItemNotFound {
                return nil
            }
            throw SecurityError.decryptionFailed
        }
        
        return result as? Data
    }
    
    func delete(for key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw SecurityError.decryptionFailed
        }
    }
}

// MARK: - Encryption Manager

class EncryptionManager {
    private var masterKey: SymmetricKey?
    
    func initializeKeys() {
        // Generate or retrieve master key
        if let keyData = try? KeychainManager().retrieve(for: "master_key") {
            masterKey = SymmetricKey(data: keyData)
        } else {
            let newKey = SymmetricKey(size: .bits256)
            masterKey = newKey
            try? KeychainManager().store(newKey.withUnsafeBytes { Data($0) }, for: "master_key")
        }
    }
    
    func encrypt(_ data: Data) throws -> EncryptedData {
        guard let key = masterKey else {
            throw SecurityError.keyGenerationFailed
        }
        
        do {
            let sealedBox = try AES.GCM.seal(data, using: key)
            return EncryptedData(
                data: sealedBox.ciphertext,
                iv: sealedBox.nonce.withUnsafeBytes { Data($0) },
                tag: sealedBox.tag
            )
        } catch {
            throw SecurityError.encryptionFailed
        }
    }
    
    func decrypt(_ encryptedData: EncryptedData) throws -> Data {
        guard let key = masterKey else {
            throw SecurityError.keyGenerationFailed
        }
        
        do {
            let nonce = try AES.GCM.Nonce(data: encryptedData.iv)
            let sealedBox = try AES.GCM.SealedBox(
                nonce: nonce,
                ciphertext: encryptedData.data,
                tag: encryptedData.tag
            )
            return try AES.GCM.open(sealedBox, using: key)
        } catch {
            throw SecurityError.decryptionFailed
        }
    }
    
    func generateSecureToken() -> String {
        let tokenData = Data((0..<32).map { _ in UInt8.random(in: 0...255) })
        return tokenData.base64EncodedString()
    }
    
    func hashPassword(_ password: String, salt: Data? = nil) -> HashedPassword {
        let saltData = salt ?? Data((0..<16).map { _ in UInt8.random(in: 0...255) })
        let iterations = 100000
        
        let hash = PBKDF2.deriveKey(
            from: password.data(using: .utf8)!,
            salt: saltData,
            using: .sha256,
            outputByteCount: 32,
            rounds: iterations
        )
        
        return HashedPassword(
            hash: hash,
            salt: saltData,
            iterations: iterations
        )
    }
}

// MARK: - PBKDF2 Extension

extension PBKDF2 {
    static func deriveKey(from password: Data, salt: Data, using algorithm: CryptoKit.HashFunction, outputByteCount: Int, rounds: Int) -> Data {
        // Simplified PBKDF2 implementation
        // In a real implementation, you would use a proper PBKDF2 library
        var result = Data()
        var currentHash = password + salt
        
        for _ in 0..<rounds {
            switch algorithm {
            case .sha256:
                currentHash = Data(SHA256.hash(data: currentHash))
            default:
                currentHash = Data(SHA256.hash(data: currentHash))
            }
        }
        
        result.append(currentHash.prefix(outputByteCount))
        return result
    }
}

enum HashFunction {
    case sha256
}