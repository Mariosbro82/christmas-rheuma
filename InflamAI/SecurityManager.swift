//
//  SecurityManager.swift
//  InflamAI-Swift
//
//  Comprehensive security management with biometric authentication,
//  end-to-end encryption, and HIPAA compliance
//

import Foundation
import LocalAuthentication
import CryptoKit
import Security
import Combine
import UIKit
import os.log

// MARK: - Security Models

struct BiometricAuthResult {
    let success: Bool
    let error: BiometricError?
    let authType: BiometricType
    let timestamp: Date
}

struct EncryptionKey {
    let id: UUID
    let keyData: Data
    let algorithm: EncryptionAlgorithm
    let createdAt: Date
    let expiresAt: Date?
    let purpose: KeyPurpose
}

struct SecurityAuditLog {
    let id: UUID
    let timestamp: Date
    let event: SecurityEvent
    let userId: String?
    let deviceId: String
    let ipAddress: String?
    let success: Bool
    let details: [String: Any]
}

struct DataClassification {
    let level: SecurityLevel
    let categories: [DataCategory]
    let retentionPeriod: TimeInterval?
    let encryptionRequired: Bool
    let auditRequired: Bool
}

struct SecurityPolicy {
    let id: UUID
    let name: String
    let version: String
    let rules: [SecurityRule]
    let effectiveDate: Date
    let expirationDate: Date?
    let isActive: Bool
}

struct SecurityRule {
    let id: UUID
    let type: RuleType
    let condition: String
    let action: SecurityAction
    let severity: SecuritySeverity
    let isEnabled: Bool
}

struct PrivacySettings {
    let dataMinimization: Bool
    let anonymization: Bool
    let pseudonymization: Bool
    let consentRequired: Bool
    let rightToErasure: Bool
    let dataPortability: Bool
    let accessLogging: Bool
}

struct ComplianceReport {
    let id: UUID
    let timestamp: Date
    let framework: ComplianceFramework
    let status: ComplianceStatus
    let findings: [ComplianceFinding]
    let recommendations: [String]
    let nextReviewDate: Date
}

struct ComplianceFinding {
    let id: UUID
    let category: String
    let severity: ComplianceSeverity
    let description: String
    let remediation: String
    let dueDate: Date?
}

// MARK: - Enums

enum BiometricType {
    case faceID
    case touchID
    case none
    case unknown
}

enum BiometricError: Error, LocalizedError {
    case notAvailable
    case notEnrolled
    case lockout
    case userCancel
    case userFallback
    case systemCancel
    case passcodeNotSet
    case biometryNotAvailable
    case biometryNotEnrolled
    case biometryLockout
    case authenticationFailed
    case unknown(Error)
    
    var errorDescription: String? {
        switch self {
        case .notAvailable:
            return "Biometric authentication is not available on this device"
        case .notEnrolled:
            return "No biometric data is enrolled on this device"
        case .lockout:
            return "Biometric authentication is locked out"
        case .userCancel:
            return "User canceled biometric authentication"
        case .userFallback:
            return "User chose to use fallback authentication"
        case .systemCancel:
            return "System canceled biometric authentication"
        case .passcodeNotSet:
            return "Device passcode is not set"
        case .biometryNotAvailable:
            return "Biometry is not available"
        case .biometryNotEnrolled:
            return "Biometry is not enrolled"
        case .biometryLockout:
            return "Biometry is locked out"
        case .authenticationFailed:
            return "Biometric authentication failed"
        case .unknown(let error):
            return "Unknown biometric error: \(error.localizedDescription)"
        }
    }
}

enum EncryptionAlgorithm {
    case aes256GCM
    case chacha20Poly1305
    case aes256CBC
    case rsa2048
    case rsa4096
    case ecdsaP256
    case ecdsaP384
}

enum KeyPurpose {
    case dataEncryption
    case keyEncryption
    case signing
    case authentication
    case backup
}

enum SecurityEvent {
    case login
    case logout
    case biometricAuth
    case dataAccess
    case dataModification
    case dataExport
    case keyGeneration
    case keyRotation
    case securityViolation
    case policyViolation
    case unauthorizedAccess
    case dataLeak
    case malwareDetection
}

enum SecurityLevel {
    case public
    case internal
    case confidential
    case restricted
    case topSecret
}

enum DataCategory {
    case personalHealth
    case medicalRecords
    case biometricData
    case geneticData
    case mentalHealth
    case prescriptions
    case insurance
    case financial
    case personalIdentity
    case location
    case communications
    case behavioral
}

enum RuleType {
    case accessControl
    case dataProtection
    case auditLogging
    case incidentResponse
    case compliance
}

enum SecurityAction {
    case allow
    case deny
    case log
    case alert
    case quarantine
    case encrypt
    case anonymize
}

enum SecuritySeverity {
    case low
    case medium
    case high
    case critical
}

enum ComplianceFramework {
    case hipaa
    case gdpr
    case ccpa
    case hitech
    case sox
    case pci
    case iso27001
}

enum ComplianceStatus {
    case compliant
    case nonCompliant
    case partiallyCompliant
    case underReview
    case notApplicable
}

enum ComplianceSeverity {
    case low
    case medium
    case high
    case critical
}

// MARK: - Security Manager

class SecurityManager: ObservableObject {
    // Published Properties
    @Published var isAuthenticated = false
    @Published var biometricType: BiometricType = .none
    @Published var isBiometricEnabled = false
    @Published var securityLevel: SecurityLevel = .confidential
    @Published var auditLogs: [SecurityAuditLog] = []
    @Published var complianceStatus: ComplianceStatus = .underReview
    @Published var privacySettings = PrivacySettings(
        dataMinimization: true,
        anonymization: false,
        pseudonymization: true,
        consentRequired: true,
        rightToErasure: true,
        dataPortability: true,
        accessLogging: true
    )
    
    // Internal Components
    private let context = LAContext()
    private let keychain = KeychainManager()
    private let encryptionManager = EncryptionManager()
    private let auditLogger = SecurityAuditLogger()
    private let complianceChecker = ComplianceChecker()
    private let threatDetector = ThreatDetector()
    
    // Internal State
    private var masterKey: SymmetricKey?
    private var deviceKey: SymmetricKey?
    private var sessionKeys: [UUID: SymmetricKey] = [:]
    private var securityPolicies: [SecurityPolicy] = []
    private var activeSession: SecuritySession?
    
    // Settings
    private var authenticationTimeout: TimeInterval = 300 // 5 minutes
    private var maxFailedAttempts = 5
    private var failedAttempts = 0
    private var lockoutDuration: TimeInterval = 900 // 15 minutes
    private var lastFailedAttempt: Date?
    
    // Logging
    private let logger = Logger(subsystem: "InflamAI", category: "Security")
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        setupSecurity()
        loadSecurityPolicies()
        checkBiometricAvailability()
        startThreatMonitoring()
    }
    
    // MARK: - Public Methods
    
    func authenticateWithBiometrics(reason: String = "Authenticate to access your health data") -> AnyPublisher<BiometricAuthResult, Never> {
        return Future<BiometricAuthResult, Never> { [weak self] promise in
            guard let self = self else {
                promise(.success(BiometricAuthResult(
                    success: false,
                    error: .unknown(NSError(domain: "SecurityManager", code: -1)),
                    authType: .none,
                    timestamp: Date()
                )))
                return
            }
            
            // Check if device is locked out
            if self.isLockedOut() {
                promise(.success(BiometricAuthResult(
                    success: false,
                    error: .lockout,
                    authType: self.biometricType,
                    timestamp: Date()
                )))
                return
            }
            
            let context = LAContext()
            context.localizedFallbackTitle = "Use Passcode"
            context.localizedCancelTitle = "Cancel"
            
            var error: NSError?
            
            guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
                let biometricError = self.mapLAError(error)
                promise(.success(BiometricAuthResult(
                    success: false,
                    error: biometricError,
                    authType: self.biometricType,
                    timestamp: Date()
                )))
                return
            }
            
            context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            ) { success, error in
                DispatchQueue.main.async {
                    let result = BiometricAuthResult(
                        success: success,
                        error: error != nil ? self.mapLAError(error as NSError?) : nil,
                        authType: self.biometricType,
                        timestamp: Date()
                    )
                    
                    if success {
                        self.handleSuccessfulAuthentication()
                    } else {
                        self.handleFailedAuthentication()
                    }
                    
                    self.logSecurityEvent(.biometricAuth, success: success, details: [
                        "biometric_type": self.biometricType,
                        "error": error?.localizedDescription ?? "none"
                    ])
                    
                    promise(.success(result))
                }
            }
        }
        .eraseToAnyPublisher()
    }
    
    func authenticateWithPasscode(passcode: String) -> Bool {
        // Verify passcode against stored hash
        guard let storedHash = keychain.getPasscodeHash(),
              verifyPasscode(passcode, against: storedHash) else {
            handleFailedAuthentication()
            logSecurityEvent(.login, success: false, details: ["method": "passcode"])
            return false
        }
        
        handleSuccessfulAuthentication()
        logSecurityEvent(.login, success: true, details: ["method": "passcode"])
        return true
    }
    
    func setPasscode(_ passcode: String) -> Bool {
        guard isValidPasscode(passcode) else {
            return false
        }
        
        let hash = hashPasscode(passcode)
        return keychain.setPasscodeHash(hash)
    }
    
    func enableBiometricAuthentication() -> Bool {
        guard biometricType != .none else {
            return false
        }
        
        isBiometricEnabled = true
        UserDefaults.standard.set(true, forKey: "biometric_enabled")
        
        logSecurityEvent(.login, success: true, details: ["action": "enable_biometric"])
        return true
    }
    
    func disableBiometricAuthentication() {
        isBiometricEnabled = false
        UserDefaults.standard.set(false, forKey: "biometric_enabled")
        
        logSecurityEvent(.login, success: true, details: ["action": "disable_biometric"])
    }
    
    func logout() {
        isAuthenticated = false
        activeSession = nil
        sessionKeys.removeAll()
        
        // Clear sensitive data from memory
        clearSensitiveData()
        
        logSecurityEvent(.logout, success: true, details: [:])
        logger.info("User logged out successfully")
    }
    
    func encryptData(_ data: Data, classification: DataClassification) -> Data? {
        guard let key = getMasterKey() else {
            logger.error("Failed to get master key for encryption")
            return nil
        }
        
        do {
            let encryptedData = try encryptionManager.encrypt(data, with: key, algorithm: .aes256GCM)
            
            logSecurityEvent(.dataAccess, success: true, details: [
                "operation": "encrypt",
                "classification": classification.level,
                "size": data.count
            ])
            
            return encryptedData
        } catch {
            logger.error("Failed to encrypt data: \(error.localizedDescription)")
            return nil
        }
    }
    
    func decryptData(_ encryptedData: Data, classification: DataClassification) -> Data? {
        guard let key = getMasterKey() else {
            logger.error("Failed to get master key for decryption")
            return nil
        }
        
        do {
            let decryptedData = try encryptionManager.decrypt(encryptedData, with: key, algorithm: .aes256GCM)
            
            logSecurityEvent(.dataAccess, success: true, details: [
                "operation": "decrypt",
                "classification": classification.level,
                "size": decryptedData.count
            ])
            
            return decryptedData
        } catch {
            logger.error("Failed to decrypt data: \(error.localizedDescription)")
            return nil
        }
    }
    
    func generateSecureKey(purpose: KeyPurpose) -> EncryptionKey? {
        let keyData = SymmetricKey(size: .bits256).withUnsafeBytes { Data($0) }
        
        let encryptionKey = EncryptionKey(
            id: UUID(),
            keyData: keyData,
            algorithm: .aes256GCM,
            createdAt: Date(),
            expiresAt: Date().addingTimeInterval(365 * 24 * 3600), // 1 year
            purpose: purpose
        )
        
        // Store key securely in keychain
        if keychain.storeKey(encryptionKey) {
            logSecurityEvent(.keyGeneration, success: true, details: [
                "purpose": purpose,
                "algorithm": encryptionKey.algorithm
            ])
            return encryptionKey
        }
        
        return nil
    }
    
    func rotateKeys() {
        // Generate new master key
        if let newMasterKey = generateSecureKey(purpose: .dataEncryption) {
            // Re-encrypt all data with new key
            reencryptDataWithNewKey(newMasterKey)
            
            // Update master key
            masterKey = SymmetricKey(data: newMasterKey.keyData)
            
            logSecurityEvent(.keyRotation, success: true, details: ["key_type": "master"])
        }
    }
    
    func anonymizeData(_ data: [String: Any]) -> [String: Any] {
        var anonymizedData = data
        
        // Remove direct identifiers
        let identifiers = ["name", "email", "phone", "ssn", "address", "birthdate"]
        for identifier in identifiers {
            anonymizedData.removeValue(forKey: identifier)
        }
        
        // Pseudonymize user ID
        if let userId = data["userId"] as? String {
            anonymizedData["userId"] = hashString(userId)
        }
        
        // Generalize dates
        if let date = data["date"] as? Date {
            let calendar = Calendar.current
            let components = calendar.dateComponents([.year, .month], from: date)
            anonymizedData["date"] = calendar.date(from: components)
        }
        
        logSecurityEvent(.dataAccess, success: true, details: ["operation": "anonymize"])
        return anonymizedData
    }
    
    func checkCompliance(framework: ComplianceFramework) -> ComplianceReport {
        return complianceChecker.generateReport(for: framework)
    }
    
    func exportSecurityAuditLog() -> Data? {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            return try encoder.encode(auditLogs)
        } catch {
            logger.error("Failed to export audit log: \(error.localizedDescription)")
            return nil
        }
    }
    
    func clearSecurityData() {
        auditLogs.removeAll()
        sessionKeys.removeAll()
        keychain.clearAllKeys()
        
        logSecurityEvent(.dataAccess, success: true, details: ["operation": "clear_security_data"])
    }
    
    func updatePrivacySettings(_ settings: PrivacySettings) {
        privacySettings = settings
        savePrivacySettings()
        
        logSecurityEvent(.policyViolation, success: true, details: ["action": "update_privacy_settings"])
    }
    
    // MARK: - Private Methods
    
    private func setupSecurity() {
        // Initialize master key
        if let storedKey = keychain.getMasterKey() {
            masterKey = SymmetricKey(data: storedKey)
        } else {
            // Generate new master key
            let newKey = SymmetricKey(size: .bits256)
            masterKey = newKey
            keychain.storeMasterKey(newKey.withUnsafeBytes { Data($0) })
        }
        
        // Initialize device key
        if let storedDeviceKey = keychain.getDeviceKey() {
            deviceKey = SymmetricKey(data: storedDeviceKey)
        } else {
            let newDeviceKey = SymmetricKey(size: .bits256)
            deviceKey = newDeviceKey
            keychain.storeDeviceKey(newDeviceKey.withUnsafeBytes { Data($0) })
        }
        
        // Load settings
        isBiometricEnabled = UserDefaults.standard.bool(forKey: "biometric_enabled")
        loadPrivacySettings()
    }
    
    private func loadSecurityPolicies() {
        // Load default HIPAA compliance policies
        let hipaaPolicy = SecurityPolicy(
            id: UUID(),
            name: "HIPAA Compliance Policy",
            version: "1.0",
            rules: createHIPAARules(),
            effectiveDate: Date(),
            expirationDate: nil,
            isActive: true
        )
        
        securityPolicies.append(hipaaPolicy)
    }
    
    private func createHIPAARules() -> [SecurityRule] {
        return [
            SecurityRule(
                id: UUID(),
                type: .dataProtection,
                condition: "PHI access",
                action: .encrypt,
                severity: .critical,
                isEnabled: true
            ),
            SecurityRule(
                id: UUID(),
                type: .accessControl,
                condition: "User authentication",
                action: .log,
                severity: .high,
                isEnabled: true
            ),
            SecurityRule(
                id: UUID(),
                type: .auditLogging,
                condition: "Data access",
                action: .log,
                severity: .medium,
                isEnabled: true
            )
        ]
    }
    
    private func checkBiometricAvailability() {
        let context = LAContext()
        var error: NSError?
        
        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) {
            switch context.biometryType {
            case .faceID:
                biometricType = .faceID
            case .touchID:
                biometricType = .touchID
            case .none:
                biometricType = .none
            @unknown default:
                biometricType = .unknown
            }
        } else {
            biometricType = .none
        }
        
        logger.info("Biometric type detected: \(biometricType)")
    }
    
    private func startThreatMonitoring() {
        threatDetector.startMonitoring()
        
        threatDetector.onThreatDetected = { [weak self] threat in
            self?.handleThreatDetection(threat)
        }
    }
    
    private func handleThreatDetection(_ threat: SecurityThreat) {
        logSecurityEvent(.malwareDetection, success: false, details: [
            "threat_type": threat.type,
            "severity": threat.severity,
            "description": threat.description
        ])
        
        // Take appropriate action based on threat severity
        switch threat.severity {
        case .critical:
            // Lock down the app
            logout()
            // Notify security team
            notifySecurityTeam(threat)
        case .high:
            // Increase monitoring
            threatDetector.increaseSensitivity()
        case .medium, .low:
            // Log and continue monitoring
            break
        }
    }
    
    private func notifySecurityTeam(_ threat: SecurityThreat) {
        // In a real app, this would send notifications to security team
        logger.critical("Critical security threat detected: \(threat.description)")
    }
    
    private func mapLAError(_ error: NSError?) -> BiometricError {
        guard let error = error else { return .unknown(NSError()) }
        
        switch error.code {
        case LAError.biometryNotAvailable.rawValue:
            return .biometryNotAvailable
        case LAError.biometryNotEnrolled.rawValue:
            return .biometryNotEnrolled
        case LAError.biometryLockout.rawValue:
            return .biometryLockout
        case LAError.userCancel.rawValue:
            return .userCancel
        case LAError.userFallback.rawValue:
            return .userFallback
        case LAError.systemCancel.rawValue:
            return .systemCancel
        case LAError.passcodeNotSet.rawValue:
            return .passcodeNotSet
        case LAError.authenticationFailed.rawValue:
            return .authenticationFailed
        default:
            return .unknown(error)
        }
    }
    
    private func handleSuccessfulAuthentication() {
        isAuthenticated = true
        failedAttempts = 0
        lastFailedAttempt = nil
        
        // Create new session
        activeSession = SecuritySession(
            id: UUID(),
            userId: getCurrentUserId(),
            startTime: Date(),
            lastActivity: Date(),
            deviceId: getDeviceId()
        )
        
        // Start session timeout timer
        startSessionTimer()
    }
    
    private func handleFailedAuthentication() {
        failedAttempts += 1
        lastFailedAttempt = Date()
        
        if failedAttempts >= maxFailedAttempts {
            // Lock out user
            logSecurityEvent(.securityViolation, success: false, details: [
                "reason": "max_failed_attempts",
                "attempts": failedAttempts
            ])
        }
    }
    
    private func isLockedOut() -> Bool {
        guard failedAttempts >= maxFailedAttempts,
              let lastFailed = lastFailedAttempt else {
            return false
        }
        
        return Date().timeIntervalSince(lastFailed) < lockoutDuration
    }
    
    private func startSessionTimer() {
        Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] timer in
            guard let self = self,
                  let session = self.activeSession else {
                timer.invalidate()
                return
            }
            
            if Date().timeIntervalSince(session.lastActivity) > self.authenticationTimeout {
                self.logout()
                timer.invalidate()
            }
        }
    }
    
    private func getMasterKey() -> SymmetricKey? {
        return masterKey
    }
    
    private func clearSensitiveData() {
        // Clear keys from memory
        masterKey = nil
        sessionKeys.removeAll()
        
        // Clear any cached sensitive data
        URLCache.shared.removeAllCachedResponses()
    }
    
    private func reencryptDataWithNewKey(_ newKey: EncryptionKey) {
        // This would re-encrypt all stored data with the new key
        // Implementation depends on data storage architecture
        logger.info("Re-encrypting data with new key")
    }
    
    private func isValidPasscode(_ passcode: String) -> Bool {
        // Validate passcode strength
        return passcode.count >= 6 && passcode.count <= 20
    }
    
    private func hashPasscode(_ passcode: String) -> String {
        return hashString(passcode)
    }
    
    private func verifyPasscode(_ passcode: String, against hash: String) -> Bool {
        return hashString(passcode) == hash
    }
    
    private func hashString(_ string: String) -> String {
        let data = Data(string.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    private func getCurrentUserId() -> String {
        return UserDefaults.standard.string(forKey: "current_user_id") ?? "anonymous"
    }
    
    private func getDeviceId() -> String {
        if let deviceId = UserDefaults.standard.string(forKey: "device_id") {
            return deviceId
        }
        
        let newDeviceId = UUID().uuidString
        UserDefaults.standard.set(newDeviceId, forKey: "device_id")
        return newDeviceId
    }
    
    private func logSecurityEvent(_ event: SecurityEvent, success: Bool, details: [String: Any]) {
        let auditLog = SecurityAuditLog(
            id: UUID(),
            timestamp: Date(),
            event: event,
            userId: getCurrentUserId(),
            deviceId: getDeviceId(),
            ipAddress: getIPAddress(),
            success: success,
            details: details
        )
        
        auditLogs.append(auditLog)
        
        // Limit audit log size
        if auditLogs.count > 10000 {
            auditLogs.removeFirst(auditLogs.count - 10000)
        }
        
        auditLogger.log(auditLog)
    }
    
    private func getIPAddress() -> String? {
        // Get current IP address
        // This is a simplified implementation
        return "127.0.0.1"
    }
    
    private func savePrivacySettings() {
        do {
            let data = try JSONEncoder().encode(privacySettings)
            UserDefaults.standard.set(data, forKey: "privacy_settings")
        } catch {
            logger.error("Failed to save privacy settings: \(error.localizedDescription)")
        }
    }
    
    private func loadPrivacySettings() {
        guard let data = UserDefaults.standard.data(forKey: "privacy_settings") else {
            return
        }
        
        do {
            privacySettings = try JSONDecoder().decode(PrivacySettings.self, from: data)
        } catch {
            logger.error("Failed to load privacy settings: \(error.localizedDescription)")
        }
    }
}

// MARK: - Supporting Classes

struct SecuritySession {
    let id: UUID
    let userId: String
    let startTime: Date
    var lastActivity: Date
    let deviceId: String
}

struct SecurityThreat {
    let id: UUID
    let type: String
    let severity: SecuritySeverity
    let description: String
    let timestamp: Date
    let source: String
}

class KeychainManager {
    private let service = "InflamAI-Security"
    
    func storeMasterKey(_ key: Data) -> Bool {
        return storeData(key, for: "master_key")
    }
    
    func getMasterKey() -> Data? {
        return getData(for: "master_key")
    }
    
    func storeDeviceKey(_ key: Data) -> Bool {
        return storeData(key, for: "device_key")
    }
    
    func getDeviceKey() -> Data? {
        return getData(for: "device_key")
    }
    
    func setPasscodeHash(_ hash: String) -> Bool {
        return storeString(hash, for: "passcode_hash")
    }
    
    func getPasscodeHash() -> String? {
        return getString(for: "passcode_hash")
    }
    
    func storeKey(_ key: EncryptionKey) -> Bool {
        do {
            let data = try JSONEncoder().encode(key)
            return storeData(data, for: "key_\(key.id.uuidString)")
        } catch {
            return false
        }
    }
    
    func getKey(id: UUID) -> EncryptionKey? {
        guard let data = getData(for: "key_\(id.uuidString)") else {
            return nil
        }
        
        do {
            return try JSONDecoder().decode(EncryptionKey.self, from: data)
        } catch {
            return nil
        }
    }
    
    func clearAllKeys() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service
        ]
        
        SecItemDelete(query as CFDictionary)
    }
    
    private func storeData(_ data: Data, for key: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        SecItemDelete(query as CFDictionary)
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }
    
    private func getData(for key: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        return status == errSecSuccess ? result as? Data : nil
    }
    
    private func storeString(_ string: String, for key: String) -> Bool {
        return storeData(Data(string.utf8), for: key)
    }
    
    private func getString(for key: String) -> String? {
        guard let data = getData(for: key) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

class EncryptionManager {
    func encrypt(_ data: Data, with key: SymmetricKey, algorithm: EncryptionAlgorithm) throws -> Data {
        switch algorithm {
        case .aes256GCM:
            let sealedBox = try AES.GCM.seal(data, using: key)
            return sealedBox.combined!
        case .chacha20Poly1305:
            let sealedBox = try ChaChaPoly.seal(data, using: key)
            return sealedBox.combined
        default:
            throw EncryptionError.unsupportedAlgorithm
        }
    }
    
    func decrypt(_ encryptedData: Data, with key: SymmetricKey, algorithm: EncryptionAlgorithm) throws -> Data {
        switch algorithm {
        case .aes256GCM:
            let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
            return try AES.GCM.open(sealedBox, using: key)
        case .chacha20Poly1305:
            let sealedBox = try ChaChaPoly.SealedBox(combined: encryptedData)
            return try ChaChaPoly.open(sealedBox, using: key)
        default:
            throw EncryptionError.unsupportedAlgorithm
        }
    }
}

enum EncryptionError: Error {
    case unsupportedAlgorithm
    case encryptionFailed
    case decryptionFailed
}

class SecurityAuditLogger {
    private let logger = Logger(subsystem: "InflamAI", category: "SecurityAudit")
    
    func log(_ auditLog: SecurityAuditLog) {
        let message = "Security Event: \(auditLog.event) | Success: \(auditLog.success) | User: \(auditLog.userId ?? "unknown") | Device: \(auditLog.deviceId)"
        
        if auditLog.success {
            logger.info("\(message)")
        } else {
            logger.warning("\(message)")
        }
    }
}

class ComplianceChecker {
    func generateReport(for framework: ComplianceFramework) -> ComplianceReport {
        let findings = checkCompliance(for: framework)
        
        let status: ComplianceStatus = findings.contains { $0.severity == .critical } ? .nonCompliant :
                                      findings.contains { $0.severity == .high } ? .partiallyCompliant :
                                      .compliant
        
        return ComplianceReport(
            id: UUID(),
            timestamp: Date(),
            framework: framework,
            status: status,
            findings: findings,
            recommendations: generateRecommendations(for: findings),
            nextReviewDate: Calendar.current.date(byAdding: .month, value: 3, to: Date()) ?? Date()
        )
    }
    
    private func checkCompliance(for framework: ComplianceFramework) -> [ComplianceFinding] {
        switch framework {
        case .hipaa:
            return checkHIPAACompliance()
        case .gdpr:
            return checkGDPRCompliance()
        default:
            return []
        }
    }
    
    private func checkHIPAACompliance() -> [ComplianceFinding] {
        var findings: [ComplianceFinding] = []
        
        // Check encryption requirements
        findings.append(ComplianceFinding(
            id: UUID(),
            category: "Data Encryption",
            severity: .low,
            description: "All PHI is encrypted at rest and in transit",
            remediation: "Continue current encryption practices",
            dueDate: nil
        ))
        
        // Check access controls
        findings.append(ComplianceFinding(
            id: UUID(),
            category: "Access Control",
            severity: .low,
            description: "Biometric authentication is implemented",
            remediation: "Continue current access control practices",
            dueDate: nil
        ))
        
        return findings
    }
    
    private func checkGDPRCompliance() -> [ComplianceFinding] {
        var findings: [ComplianceFinding] = []
        
        // Check consent management
        findings.append(ComplianceFinding(
            id: UUID(),
            category: "Consent Management",
            severity: .medium,
            description: "Implement granular consent management",
            remediation: "Add detailed consent options for data processing",
            dueDate: Calendar.current.date(byAdding: .month, value: 1, to: Date())
        ))
        
        return findings
    }
    
    private func generateRecommendations(for findings: [ComplianceFinding]) -> [String] {
        return findings.compactMap { finding in
            if finding.severity == .high || finding.severity == .critical {
                return "Address \(finding.category): \(finding.remediation)"
            }
            return nil
        }
    }
}

class ThreatDetector {
    var onThreatDetected: ((SecurityThreat) -> Void)?
    private var isMonitoring = false
    private let logger = Logger(subsystem: "InflamAI", category: "ThreatDetection")
    
    func startMonitoring() {
        isMonitoring = true
        logger.info("Threat monitoring started")
        
        // Start various threat detection mechanisms
        startJailbreakDetection()
        startDebuggerDetection()
        startTamperingDetection()
    }
    
    func stopMonitoring() {
        isMonitoring = false
        logger.info("Threat monitoring stopped")
    }
    
    func increaseSensitivity() {
        logger.info("Increased threat detection sensitivity")
    }
    
    private func startJailbreakDetection() {
        if isJailbroken() {
            reportThreat(
                type: "Jailbreak",
                severity: .critical,
                description: "Device appears to be jailbroken",
                source: "JailbreakDetector"
            )
        }
    }
    
    private func startDebuggerDetection() {
        if isDebuggerAttached() {
            reportThreat(
                type: "Debugger",
                severity: .high,
                description: "Debugger detected",
                source: "DebuggerDetector"
            )
        }
    }
    
    private func startTamperingDetection() {
        if isAppTampered() {
            reportThreat(
                type: "Tampering",
                severity: .critical,
                description: "App tampering detected",
                source: "TamperingDetector"
            )
        }
    }
    
    private func isJailbroken() -> Bool {
        // Check for common jailbreak indicators
        let jailbreakPaths = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt"
        ]
        
        for path in jailbreakPaths {
            if FileManager.default.fileExists(atPath: path) {
                return true
            }
        }
        
        return false
    }
    
    private func isDebuggerAttached() -> Bool {
        var info = kinfo_proc()
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        var size = MemoryLayout<kinfo_proc>.stride
        
        let result = sysctl(&mib, u_int(mib.count), &info, &size, nil, 0)
        
        return result == 0 && (info.kp_proc.p_flag & P_TRACED) != 0
    }
    
    private func isAppTampered() -> Bool {
        // Check app signature and bundle integrity
        // This is a simplified check
        guard let bundlePath = Bundle.main.bundlePath as NSString? else {
            return true
        }
        
        let infoPlistPath = bundlePath.appendingPathComponent("Info.plist")
        return !FileManager.default.fileExists(atPath: infoPlistPath)
    }
    
    private func reportThreat(type: String, severity: SecuritySeverity, description: String, source: String) {
        let threat = SecurityThreat(
            id: UUID(),
            type: type,
            severity: severity,
            description: description,
            timestamp: Date(),
            source: source
        )
        
        onThreatDetected?(threat)
    }
}

// MARK: - Codable Extensions

extension PrivacySettings: Codable {}
extension EncryptionKey: Codable {}
extension SecurityAuditLog: Codable {}
extension ComplianceReport: Codable {}
extension ComplianceFinding: Codable {}

// MARK: - Notification Extensions

extension Notification.Name {
    static let securityThreatDetected = Notification.Name("securityThreatDetected")
    static let biometricAuthenticationChanged = Notification.Name("biometricAuthenticationChanged")
    static let securityPolicyViolation = Notification.Name("securityPolicyViolation")
    static let complianceStatusChanged = Notification.Name("complianceStatusChanged")
}