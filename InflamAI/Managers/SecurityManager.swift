//
//  SecurityManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import LocalAuthentication
import CryptoKit
import Security
import UIKit
import Combine
import os.log

// MARK: - Security Manager
class SecurityManager: ObservableObject {
    static let shared = SecurityManager()
    
    // MARK: - Properties
    @Published var isAuthenticated = false
    @Published var biometricType: BiometricType = .none
    @Published var isEncryptionEnabled = true
    @Published var privacySettings = PrivacySettings()
    @Published var securitySettings = SecuritySettings()
    @Published var auditLogs: [SecurityAuditLog] = []
    @Published var securityAlerts: [SecurityAlert] = []
    @Published var complianceStatus = ComplianceStatus()
    
    // Security Components
    private let biometricAuthenticator = BiometricAuthenticator()
    private let encryptionManager = EncryptionManager()
    private let privacyManager = PrivacyManager()
    private let auditLogger = SecurityAuditLogger()
    private let complianceChecker = ComplianceChecker()
    private let dataAnonymizer = DataAnonymizer()
    private let accessController = AccessController()
    
    // Session Management
    private var sessionToken: String?
    private var sessionExpiry: Date?
    private var sessionTimer: Timer?
    private var failedAttempts = 0
    private var lockoutUntil: Date?
    
    // Encryption Keys
    private var masterKey: SymmetricKey?
    private var deviceKey: SymmetricKey?
    private var userKey: SymmetricKey?
    
    private let logger = Logger(subsystem: "com.inflamai.security", category: "SecurityManager")
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    init() {
        setupSecurity()
        loadSecuritySettings()
        checkBiometricAvailability()
        initializeEncryption()
        startSecurityMonitoring()
    }
    
    deinit {
        sessionTimer?.invalidate()
    }
    
    // MARK: - Setup
    private func setupSecurity() {
        setupAuditLogging()
        setupComplianceMonitoring()
        setupSessionManagement()
    }
    
    private func setupAuditLogging() {
        auditLogger.delegate = self
        auditLogger.startLogging()
    }
    
    private func setupComplianceMonitoring() {
        complianceChecker.delegate = self
        complianceChecker.startMonitoring()
    }
    
    private func setupSessionManagement() {
        // Set default session timeout to 15 minutes
        securitySettings.sessionTimeout = 15 * 60
    }
    
    private func startSecurityMonitoring() {
        // Monitor app state changes
        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .sink { [weak self] _ in
                self?.handleAppWillResignActive()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.handleAppDidBecomeActive()
            }
            .store(in: &cancellables)
    }
    
    // MARK: - Biometric Authentication
    func checkBiometricAvailability() {
        biometricAuthenticator.checkAvailability { [weak self] type in
            DispatchQueue.main.async {
                self?.biometricType = type
            }
        }
    }
    
    func authenticateWithBiometrics() async -> AuthenticationResult {
        guard biometricType != .none else {
            return .failure(.biometricNotAvailable)
        }
        
        guard !isLockedOut() else {
            return .failure(.accountLocked)
        }
        
        let result = await biometricAuthenticator.authenticate(reason: "Authenticate to access your health data")
        
        switch result {
        case .success:
            await handleSuccessfulAuthentication()
            return .success
        case .failure(let error):
            await handleFailedAuthentication()
            return .failure(error)
        }
    }
    
    func authenticateWithPasscode(_ passcode: String) async -> AuthenticationResult {
        guard !isLockedOut() else {
            return .failure(.accountLocked)
        }
        
        let storedPasscodeHash = getStoredPasscodeHash()
        let inputPasscodeHash = hashPasscode(passcode)
        
        if storedPasscodeHash == inputPasscodeHash {
            await handleSuccessfulAuthentication()
            return .success
        } else {
            await handleFailedAuthentication()
            return .failure(.invalidCredentials)
        }
    }
    
    private func handleSuccessfulAuthentication() async {
        DispatchQueue.main.async {
            self.isAuthenticated = true
            self.failedAttempts = 0
            self.lockoutUntil = nil
        }
        
        generateSessionToken()
        startSessionTimer()
        
        await auditLogger.logEvent(.authenticationSuccess, details: [
            "method": biometricType.rawValue,
            "timestamp": Date().iso8601String
        ])
    }
    
    private func handleFailedAuthentication() async {
        failedAttempts += 1
        
        if failedAttempts >= securitySettings.maxFailedAttempts {
            lockoutUntil = Date().addingTimeInterval(TimeInterval(securitySettings.lockoutDuration))
            
            let alert = SecurityAlert(
                type: .accountLocked,
                severity: .high,
                message: "Account locked due to multiple failed authentication attempts",
                timestamp: Date()
            )
            
            DispatchQueue.main.async {
                self.securityAlerts.append(alert)
            }
        }
        
        await auditLogger.logEvent(.authenticationFailure, details: [
            "attempts": String(failedAttempts),
            "timestamp": Date().iso8601String
        ])
    }
    
    private func isLockedOut() -> Bool {
        guard let lockoutUntil = lockoutUntil else { return false }
        return Date() < lockoutUntil
    }
    
    // MARK: - Session Management
    private func generateSessionToken() {
        sessionToken = UUID().uuidString
        sessionExpiry = Date().addingTimeInterval(TimeInterval(securitySettings.sessionTimeout))
    }
    
    private func startSessionTimer() {
        sessionTimer?.invalidate()
        
        sessionTimer = Timer.scheduledTimer(withTimeInterval: TimeInterval(securitySettings.sessionTimeout), repeats: false) { [weak self] _ in
            self?.expireSession()
        }
    }
    
    private func expireSession() {
        DispatchQueue.main.async {
            self.isAuthenticated = false
        }
        
        sessionToken = nil
        sessionExpiry = nil
        sessionTimer?.invalidate()
        
        Task {
            await auditLogger.logEvent(.sessionExpired, details: [
                "timestamp": Date().iso8601String
            ])
        }
    }
    
    func extendSession() {
        guard isAuthenticated else { return }
        
        sessionExpiry = Date().addingTimeInterval(TimeInterval(securitySettings.sessionTimeout))
        startSessionTimer()
    }
    
    func logout() {
        DispatchQueue.main.async {
            self.isAuthenticated = false
        }
        
        sessionToken = nil
        sessionExpiry = nil
        sessionTimer?.invalidate()
        
        Task {
            await auditLogger.logEvent(.userLogout, details: [
                "timestamp": Date().iso8601String
            ])
        }
    }
    
    // MARK: - Encryption
    private func initializeEncryption() {
        generateEncryptionKeys()
    }
    
    private func generateEncryptionKeys() {
        // Generate master key
        masterKey = SymmetricKey(size: .bits256)
        
        // Generate device-specific key
        deviceKey = SymmetricKey(size: .bits256)
        
        // Generate user-specific key (derived from user credentials)
        userKey = SymmetricKey(size: .bits256)
        
        // Store keys securely in Keychain
        storeKeyInKeychain(masterKey!, identifier: "master_key")
        storeKeyInKeychain(deviceKey!, identifier: "device_key")
        storeKeyInKeychain(userKey!, identifier: "user_key")
    }
    
    func encryptData(_ data: Data) throws -> EncryptedData {
        guard let masterKey = masterKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return try encryptionManager.encrypt(data, with: masterKey)
    }
    
    func decryptData(_ encryptedData: EncryptedData) throws -> Data {
        guard let masterKey = masterKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return try encryptionManager.decrypt(encryptedData, with: masterKey)
    }
    
    func encryptSensitiveData(_ data: Data) throws -> EncryptedData {
        guard let userKey = userKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return try encryptionManager.encrypt(data, with: userKey)
    }
    
    func decryptSensitiveData(_ encryptedData: EncryptedData) throws -> Data {
        guard let userKey = userKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return try encryptionManager.decrypt(encryptedData, with: userKey)
    }
    
    // MARK: - Privacy Controls
    func updatePrivacySettings(_ settings: PrivacySettings) {
        privacySettings = settings
        savePrivacySettings()
        
        Task {
            await auditLogger.logEvent(.privacySettingsChanged, details: [
                "dataSharing": String(settings.allowDataSharing),
                "analytics": String(settings.allowAnalytics),
                "anonymization": String(settings.enableDataAnonymization),
                "timestamp": Date().iso8601String
            ])
        }
    }
    
    func anonymizeUserData() async -> Bool {
        do {
            let success = await dataAnonymizer.anonymizeAllUserData()
            
            if success {
                await auditLogger.logEvent(.dataAnonymized, details: [
                    "timestamp": Date().iso8601String
                ])
            }
            
            return success
        } catch {
            logger.error("Failed to anonymize user data: \(error.localizedDescription)")
            return false
        }
    }
    
    func deleteUserData() async -> Bool {
        do {
            let success = await dataAnonymizer.deleteAllUserData()
            
            if success {
                await auditLogger.logEvent(.dataDeleted, details: [
                    "timestamp": Date().iso8601String
                ])
            }
            
            return success
        } catch {
            logger.error("Failed to delete user data: \(error.localizedDescription)")
            return false
        }
    }
    
    // MARK: - Access Control
    func checkDataAccess(for dataType: DataType, operation: DataOperation) -> Bool {
        return accessController.checkAccess(for: dataType, operation: operation, settings: privacySettings)
    }
    
    func requestDataAccess(for dataType: DataType, operation: DataOperation, reason: String) async -> Bool {
        let granted = await accessController.requestAccess(for: dataType, operation: operation, reason: reason)
        
        await auditLogger.logEvent(.dataAccessRequested, details: [
            "dataType": dataType.rawValue,
            "operation": operation.rawValue,
            "granted": String(granted),
            "reason": reason,
            "timestamp": Date().iso8601String
        ])
        
        return granted
    }
    
    func revokeDataAccess(for dataType: DataType) {
        accessController.revokeAccess(for: dataType)
        
        Task {
            await auditLogger.logEvent(.dataAccessRevoked, details: [
                "dataType": dataType.rawValue,
                "timestamp": Date().iso8601String
            ])
        }
    }
    
    // MARK: - HIPAA Compliance
    func checkHIPAACompliance() -> ComplianceStatus {
        let status = complianceChecker.checkCompliance()
        
        DispatchQueue.main.async {
            self.complianceStatus = status
        }
        
        return status
    }
    
    func generateComplianceReport() -> ComplianceReport {
        return complianceChecker.generateReport(
            auditLogs: auditLogs,
            securitySettings: securitySettings,
            privacySettings: privacySettings
        )
    }
    
    func enableHIPAAMode() {
        securitySettings.hipaaMode = true
        securitySettings.encryptionRequired = true
        securitySettings.auditLoggingRequired = true
        securitySettings.sessionTimeout = 10 * 60 // 10 minutes
        securitySettings.maxFailedAttempts = 3
        
        privacySettings.allowDataSharing = false
        privacySettings.allowAnalytics = false
        privacySettings.enableDataAnonymization = true
        privacySettings.requireExplicitConsent = true
        
        saveSecuritySettings()
        savePrivacySettings()
        
        Task {
            await auditLogger.logEvent(.hipaaEnabled, details: [
                "timestamp": Date().iso8601String
            ])
        }
    }
    
    // MARK: - Security Monitoring
    private func handleAppWillResignActive() {
        if securitySettings.lockOnBackground {
            DispatchQueue.main.async {
                self.isAuthenticated = false
            }
        }
    }
    
    private func handleAppDidBecomeActive() {
        if securitySettings.requireAuthOnForeground && !isAuthenticated {
            // Trigger authentication
            Task {
                _ = await authenticateWithBiometrics()
            }
        }
    }
    
    func detectSecurityThreats() {
        // Check for jailbreak
        if isDeviceJailbroken() {
            let alert = SecurityAlert(
                type: .jailbrokenDevice,
                severity: .critical,
                message: "Device appears to be jailbroken. This may compromise security.",
                timestamp: Date()
            )
            
            DispatchQueue.main.async {
                self.securityAlerts.append(alert)
            }
        }
        
        // Check for debugger
        if isDebuggerAttached() {
            let alert = SecurityAlert(
                type: .debuggerDetected,
                severity: .high,
                message: "Debugger detected. This may indicate a security threat.",
                timestamp: Date()
            )
            
            DispatchQueue.main.async {
                self.securityAlerts.append(alert)
            }
        }
        
        // Check for suspicious network activity
        checkNetworkSecurity()
    }
    
    private func isDeviceJailbroken() -> Bool {
        // Check for common jailbreak indicators
        let jailbreakPaths = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/"
        ]
        
        for path in jailbreakPaths {
            if FileManager.default.fileExists(atPath: path) {
                return true
            }
        }
        
        // Check if we can write to system directories
        let testPath = "/private/test_jailbreak.txt"
        do {
            try "test".write(toFile: testPath, atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(atPath: testPath)
            return true
        } catch {
            // Normal behavior - we shouldn't be able to write here
        }
        
        return false
    }
    
    private func isDebuggerAttached() -> Bool {
        var info = kinfo_proc()
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        var size = MemoryLayout<kinfo_proc>.stride
        
        let result = sysctl(&mib, u_int(mib.count), &info, &size, nil, 0)
        
        if result != 0 {
            return false
        }
        
        return (info.kp_proc.p_flag & P_TRACED) != 0
    }
    
    private func checkNetworkSecurity() {
        // Implementation would check for suspicious network connections
        // This is a placeholder for network security monitoring
    }
    
    // MARK: - Keychain Operations
    private func storeKeyInKeychain(_ key: SymmetricKey, identifier: String) {
        let keyData = key.withUnsafeBytes { Data($0) }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: identifier,
            kSecValueData as String: keyData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Delete existing key
        SecItemDelete(query as CFDictionary)
        
        // Add new key
        let status = SecItemAdd(query as CFDictionary, nil)
        
        if status != errSecSuccess {
            logger.error("Failed to store key in keychain: \(status)")
        }
    }
    
    private func loadKeyFromKeychain(identifier: String) -> SymmetricKey? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: identifier,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        if status == errSecSuccess, let keyData = result as? Data {
            return SymmetricKey(data: keyData)
        }
        
        return nil
    }
    
    // MARK: - Passcode Management
    private func hashPasscode(_ passcode: String) -> String {
        let data = Data(passcode.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    private func getStoredPasscodeHash() -> String? {
        return UserDefaults.standard.string(forKey: "stored_passcode_hash")
    }
    
    func setPasscode(_ passcode: String) {
        let hash = hashPasscode(passcode)
        UserDefaults.standard.set(hash, forKey: "stored_passcode_hash")
        
        Task {
            await auditLogger.logEvent(.passcodeChanged, details: [
                "timestamp": Date().iso8601String
            ])
        }
    }
    
    // MARK: - Data Persistence
    private func loadSecuritySettings() {
        if let data = UserDefaults.standard.data(forKey: "security_settings"),
           let settings = try? JSONDecoder().decode(SecuritySettings.self, from: data) {
            self.securitySettings = settings
        }
        
        if let data = UserDefaults.standard.data(forKey: "privacy_settings"),
           let settings = try? JSONDecoder().decode(PrivacySettings.self, from: data) {
            self.privacySettings = settings
        }
        
        // Load encryption keys from keychain
        masterKey = loadKeyFromKeychain(identifier: "master_key")
        deviceKey = loadKeyFromKeychain(identifier: "device_key")
        userKey = loadKeyFromKeychain(identifier: "user_key")
        
        // Generate keys if they don't exist
        if masterKey == nil || deviceKey == nil || userKey == nil {
            generateEncryptionKeys()
        }
    }
    
    private func saveSecuritySettings() {
        if let data = try? JSONEncoder().encode(securitySettings) {
            UserDefaults.standard.set(data, forKey: "security_settings")
        }
    }
    
    private func savePrivacySettings() {
        if let data = try? JSONEncoder().encode(privacySettings) {
            UserDefaults.standard.set(data, forKey: "privacy_settings")
        }
    }
}

// MARK: - SecurityAuditLoggerDelegate
extension SecurityManager: SecurityAuditLoggerDelegate {
    func auditLogger(_ logger: SecurityAuditLogger, didLogEvent event: SecurityAuditLog) {
        DispatchQueue.main.async {
            self.auditLogs.append(event)
            
            // Keep only last 1000 logs
            if self.auditLogs.count > 1000 {
                self.auditLogs.removeFirst()
            }
        }
    }
}

// MARK: - ComplianceCheckerDelegate
extension SecurityManager: ComplianceCheckerDelegate {
    func complianceChecker(_ checker: ComplianceChecker, didUpdateStatus status: ComplianceStatus) {
        DispatchQueue.main.async {
            self.complianceStatus = status
        }
    }
    
    func complianceChecker(_ checker: ComplianceChecker, didDetectViolation violation: ComplianceViolation) {
        let alert = SecurityAlert(
            type: .complianceViolation,
            severity: .high,
            message: "Compliance violation detected: \(violation.description)",
            timestamp: Date()
        )
        
        DispatchQueue.main.async {
            self.securityAlerts.append(alert)
        }
    }
}

// MARK: - Supporting Classes
class BiometricAuthenticator {
    private let context = LAContext()
    
    func checkAvailability(completion: @escaping (BiometricType) -> Void) {
        var error: NSError?
        
        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) {
            switch context.biometryType {
            case .faceID:
                completion(.faceID)
            case .touchID:
                completion(.touchID)
            default:
                completion(.none)
            }
        } else {
            completion(.none)
        }
    }
    
    func authenticate(reason: String) async -> AuthenticationResult {
        let context = LAContext()
        
        do {
            let success = try await context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason)
            return success ? .success : .failure(.authenticationFailed)
        } catch {
            if let laError = error as? LAError {
                switch laError.code {
                case .userCancel:
                    return .failure(.userCancelled)
                case .userFallback:
                    return .failure(.userFallback)
                case .biometryNotAvailable:
                    return .failure(.biometricNotAvailable)
                case .biometryNotEnrolled:
                    return .failure(.biometricNotEnrolled)
                case .biometryLockout:
                    return .failure(.biometricLockout)
                default:
                    return .failure(.authenticationFailed)
                }
            }
            return .failure(.authenticationFailed)
        }
    }
}

class EncryptionManager {
    func encrypt(_ data: Data, with key: SymmetricKey) throws -> EncryptedData {
        let sealedBox = try AES.GCM.seal(data, using: key)
        
        guard let encryptedData = sealedBox.combined else {
            throw SecurityError.encryptionFailed
        }
        
        return EncryptedData(
            data: encryptedData,
            algorithm: .aesGCM,
            keySize: .bits256
        )
    }
    
    func decrypt(_ encryptedData: EncryptedData, with key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: encryptedData.data)
        return try AES.GCM.open(sealedBox, using: key)
    }
}

class PrivacyManager {
    func applyPrivacySettings(_ settings: PrivacySettings, to data: Data) -> Data {
        if settings.enableDataAnonymization {
            return anonymizeData(data)
        }
        return data
    }
    
    private func anonymizeData(_ data: Data) -> Data {
        // Implementation would anonymize sensitive data
        return data
    }
}

class SecurityAuditLogger {
    weak var delegate: SecurityAuditLoggerDelegate?
    private let logger = Logger(subsystem: "com.inflamai.security", category: "AuditLogger")
    
    func startLogging() {
        logger.info("Security audit logging started")
    }
    
    func logEvent(_ type: SecurityEventType, details: [String: String]) async {
        let event = SecurityAuditLog(
            id: UUID().uuidString,
            timestamp: Date(),
            eventType: type,
            details: details,
            userId: getCurrentUserId(),
            sessionId: getCurrentSessionId()
        )
        
        delegate?.auditLogger(self, didLogEvent: event)
        
        // Log to system logger
        logger.info("Security event: \(type.rawValue) - \(details)")
        
        // Store in persistent storage
        await storeAuditLog(event)
    }
    
    private func getCurrentUserId() -> String? {
        // Implementation would get current user ID
        return nil
    }
    
    private func getCurrentSessionId() -> String? {
        // Implementation would get current session ID
        return nil
    }
    
    private func storeAuditLog(_ log: SecurityAuditLog) async {
        // Implementation would store audit log persistently
    }
}

class ComplianceChecker {
    weak var delegate: ComplianceCheckerDelegate?
    
    func startMonitoring() {
        // Start compliance monitoring
    }
    
    func checkCompliance() -> ComplianceStatus {
        var violations: [ComplianceViolation] = []
        
        // Check HIPAA requirements
        if !checkEncryptionCompliance() {
            violations.append(ComplianceViolation(
                type: .encryptionRequired,
                description: "Data encryption is required for HIPAA compliance",
                severity: .high
            ))
        }
        
        if !checkAuditLoggingCompliance() {
            violations.append(ComplianceViolation(
                type: .auditLoggingRequired,
                description: "Audit logging is required for HIPAA compliance",
                severity: .high
            ))
        }
        
        if !checkAccessControlCompliance() {
            violations.append(ComplianceViolation(
                type: .accessControlRequired,
                description: "Proper access controls are required for HIPAA compliance",
                severity: .medium
            ))
        }
        
        return ComplianceStatus(
            isCompliant: violations.isEmpty,
            violations: violations,
            lastChecked: Date()
        )
    }
    
    func generateReport(auditLogs: [SecurityAuditLog], securitySettings: SecuritySettings, privacySettings: PrivacySettings) -> ComplianceReport {
        return ComplianceReport(
            timestamp: Date(),
            complianceStatus: checkCompliance(),
            auditLogCount: auditLogs.count,
            securitySettings: securitySettings,
            privacySettings: privacySettings,
            recommendations: generateRecommendations()
        )
    }
    
    private func checkEncryptionCompliance() -> Bool {
        // Check if encryption is properly implemented
        return true // Placeholder
    }
    
    private func checkAuditLoggingCompliance() -> Bool {
        // Check if audit logging is properly implemented
        return true // Placeholder
    }
    
    private func checkAccessControlCompliance() -> Bool {
        // Check if access controls are properly implemented
        return true // Placeholder
    }
    
    private func generateRecommendations() -> [String] {
        return [
            "Enable two-factor authentication",
            "Implement regular security audits",
            "Update encryption algorithms regularly",
            "Train users on security best practices"
        ]
    }
}

class DataAnonymizer {
    func anonymizeAllUserData() async -> Bool {
        // Implementation would anonymize all user data
        return true
    }
    
    func deleteAllUserData() async -> Bool {
        // Implementation would delete all user data
        return true
    }
}

class AccessController {
    private var accessPermissions: [DataType: [DataOperation: Bool]] = [:]
    
    func checkAccess(for dataType: DataType, operation: DataOperation, settings: PrivacySettings) -> Bool {
        // Check if access is allowed based on privacy settings
        if settings.requireExplicitConsent {
            return accessPermissions[dataType]?[operation] ?? false
        }
        
        return true
    }
    
    func requestAccess(for dataType: DataType, operation: DataOperation, reason: String) async -> Bool {
        // In a real implementation, this would show a user consent dialog
        // For now, return true as placeholder
        
        if accessPermissions[dataType] == nil {
            accessPermissions[dataType] = [:]
        }
        
        accessPermissions[dataType]?[operation] = true
        return true
    }
    
    func revokeAccess(for dataType: DataType) {
        accessPermissions[dataType] = nil
    }
}

// MARK: - Protocols
protocol SecurityAuditLoggerDelegate: AnyObject {
    func auditLogger(_ logger: SecurityAuditLogger, didLogEvent event: SecurityAuditLog)
}

protocol ComplianceCheckerDelegate: AnyObject {
    func complianceChecker(_ checker: ComplianceChecker, didUpdateStatus status: ComplianceStatus)
    func complianceChecker(_ checker: ComplianceChecker, didDetectViolation violation: ComplianceViolation)
}

// MARK: - Supporting Types
enum BiometricType: String, CaseIterable {
    case none = "None"
    case touchID = "Touch ID"
    case faceID = "Face ID"
}

enum AuthenticationResult {
    case success
    case failure(AuthenticationError)
}

enum AuthenticationError: Error, LocalizedError {
    case biometricNotAvailable
    case biometricNotEnrolled
    case biometricLockout
    case userCancelled
    case userFallback
    case authenticationFailed
    case invalidCredentials
    case accountLocked
    
    var errorDescription: String? {
        switch self {
        case .biometricNotAvailable:
            return "Biometric authentication is not available on this device"
        case .biometricNotEnrolled:
            return "No biometric data is enrolled on this device"
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
        }
    }
}

enum SecurityError: Error, LocalizedError {
    case encryptionFailed
    case decryptionFailed
    case encryptionKeyNotFound
    case invalidData
    
    var errorDescription: String? {
        switch self {
        case .encryptionFailed:
            return "Failed to encrypt data"
        case .decryptionFailed:
            return "Failed to decrypt data"
        case .encryptionKeyNotFound:
            return "Encryption key not found"
        case .invalidData:
            return "Invalid data provided"
        }
    }
}

struct SecuritySettings: Codable {
    var sessionTimeout: Int = 15 * 60 // 15 minutes
    var maxFailedAttempts: Int = 5
    var lockoutDuration: Int = 5 * 60 // 5 minutes
    var requireBiometric: Bool = true
    var requirePasscode: Bool = true
    var lockOnBackground: Bool = true
    var requireAuthOnForeground: Bool = true
    var encryptionRequired: Bool = true
    var auditLoggingRequired: Bool = true
    var hipaaMode: Bool = false
}

struct PrivacySettings: Codable {
    var allowDataSharing: Bool = false
    var allowAnalytics: Bool = false
    var enableDataAnonymization: Bool = true
    var requireExplicitConsent: Bool = true
    var dataRetentionDays: Int = 365
    var allowThirdPartyAccess: Bool = false
    var enableLocationTracking: Bool = false
    var shareWithHealthcareProviders: Bool = false
}

struct EncryptedData {
    var data: Data
    var algorithm: EncryptionAlgorithm
    var keySize: KeySize
}

enum EncryptionAlgorithm: String, Codable {
    case aesGCM = "AES-GCM"
    case aesECB = "AES-ECB"
    case aesCBC = "AES-CBC"
}

enum KeySize: String, Codable {
    case bits128 = "128"
    case bits192 = "192"
    case bits256 = "256"
}

struct SecurityAuditLog: Codable {
    var id: String
    var timestamp: Date
    var eventType: SecurityEventType
    var details: [String: String]
    var userId: String?
    var sessionId: String?
}

enum SecurityEventType: String, Codable, CaseIterable {
    case authenticationSuccess = "Authentication Success"
    case authenticationFailure = "Authentication Failure"
    case sessionExpired = "Session Expired"
    case userLogout = "User Logout"
    case dataAccessed = "Data Accessed"
    case dataModified = "Data Modified"
    case dataDeleted = "Data Deleted"
    case dataAnonymized = "Data Anonymized"
    case privacySettingsChanged = "Privacy Settings Changed"
    case securitySettingsChanged = "Security Settings Changed"
    case passcodeChanged = "Passcode Changed"
    case dataAccessRequested = "Data Access Requested"
    case dataAccessRevoked = "Data Access Revoked"
    case hipaaEnabled = "HIPAA Mode Enabled"
    case complianceViolation = "Compliance Violation"
}

struct SecurityAlert {
    var type: SecurityAlertType
    var severity: SecurityAlertSeverity
    var message: String
    var timestamp: Date
}

enum SecurityAlertType: String, CaseIterable {
    case accountLocked = "Account Locked"
    case jailbrokenDevice = "Jailbroken Device"
    case debuggerDetected = "Debugger Detected"
    case suspiciousActivity = "Suspicious Activity"
    case complianceViolation = "Compliance Violation"
    case dataBreachAttempt = "Data Breach Attempt"
}

enum SecurityAlertSeverity: String, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
    case critical = "Critical"
}

struct ComplianceStatus {
    var isCompliant: Bool = false
    var violations: [ComplianceViolation] = []
    var lastChecked: Date = Date()
}

struct ComplianceViolation {
    var type: ComplianceViolationType
    var description: String
    var severity: ComplianceViolationSeverity
}

enum ComplianceViolationType: String, CaseIterable {
    case encryptionRequired = "Encryption Required"
    case auditLoggingRequired = "Audit Logging Required"
    case accessControlRequired = "Access Control Required"
    case dataRetentionViolation = "Data Retention Violation"
    case consentRequired = "Consent Required"
}

enum ComplianceViolationSeverity: String, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
    case critical = "Critical"
}

struct ComplianceReport {
    var timestamp: Date
    var complianceStatus: ComplianceStatus
    var auditLogCount: Int
    var securitySettings: SecuritySettings
    var privacySettings: PrivacySettings
    var recommendations: [String]
}

enum DataType: String, CaseIterable {
    case healthData = "Health Data"
    case personalInfo = "Personal Information"
    case locationData = "Location Data"
    case usageAnalytics = "Usage Analytics"
    case medicalRecords = "Medical Records"
    case communicationData = "Communication Data"
}

enum DataOperation: String, CaseIterable {
    case read = "Read"
    case write = "Write"
    case delete = "Delete"
    case share = "Share"
    case export = "Export"
    case anonymize = "Anonymize"
}

// MARK: - Extensions
extension Date {
    var iso8601String: String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: self)
    }
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let securityAlertGenerated = Notification.Name("securityAlertGenerated")
    static let authenticationRequired = Notification.Name("authenticationRequired")
    static let sessionExpired = Notification.Name("sessionExpired")
    static let complianceViolationDetected = Notification.Name("complianceViolationDetected")
    static let securityThreatDetected = Notification.Name("securityThreatDetected")
}