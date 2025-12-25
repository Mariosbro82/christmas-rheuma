//
//  SecurityManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import LocalAuthentication
import CryptoKit
import Security
import UIKit
import os.log
import Combine

// MARK: - Security Manager
class SecurityManager: NSObject, ObservableObject {
    
    static let shared = SecurityManager()
    
    private let logger = Logger(subsystem: "InflamAI", category: "Security")
    private let keychain = KeychainManager()
    private let encryption = EncryptionManager()
    
    // Published properties
    @Published var isAuthenticated = false
    @Published var biometricType: BiometricType = .none
    @Published var isSecurityEnabled = false
    @Published var privacySettings = PrivacySettings()
    @Published var securityEvents: [SecurityEvent] = []
    @Published var encryptionStatus = EncryptionStatus()
    
    // Configuration
    private let maxFailedAttempts = 5
    private let lockoutDuration: TimeInterval = 300 // 5 minutes
    private let sessionTimeout: TimeInterval = 900 // 15 minutes
    
    // Internal state
    private var failedAttempts = 0
    private var lastFailedAttempt: Date?
    private var sessionStartTime: Date?
    private var sessionTimer: Timer?
    private var cancellables = Set<AnyCancellable>()
    
    // Encryption keys
    private var masterKey: SymmetricKey?
    private var dataEncryptionKey: SymmetricKey?
    
    override init() {
        super.init()
        setupSecurity()
        loadSecuritySettings()
        checkBiometricAvailability()
    }
    
    deinit {
        sessionTimer?.invalidate()
    }
    
    // MARK: - Public Methods
    
    func enableSecurity(with passcode: String? = nil) async throws {
        guard !isSecurityEnabled else { return }
        
        // Generate master key
        masterKey = SymmetricKey(size: .bits256)
        dataEncryptionKey = SymmetricKey(size: .bits256)
        
        // Store keys securely
        try await storeEncryptionKeys()
        
        // Set up passcode if provided
        if let passcode = passcode {
            try await setPasscode(passcode)
        }
        
        // Enable security
        isSecurityEnabled = true
        saveSecuritySettings()
        
        // Log security event
        logSecurityEvent(.securityEnabled, details: "Security features enabled")
        
        logger.info("Security enabled successfully")
    }
    
    func disableSecurity() async throws {
        guard isSecurityEnabled else { return }
        
        // Require authentication to disable
        let authenticated = try await authenticateUser(reason: "Disable security features")
        guard authenticated else {
            throw SecurityError.authenticationFailed
        }
        
        // Clear encryption keys
        try await clearEncryptionKeys()
        
        // Clear passcode
        try await clearPasscode()
        
        // Disable security
        isSecurityEnabled = false
        isAuthenticated = false
        saveSecuritySettings()
        
        // Log security event
        logSecurityEvent(.securityDisabled, details: "Security features disabled")
        
        logger.info("Security disabled successfully")
    }
    
    func authenticateUser(reason: String = "Authenticate to access app") async throws -> Bool {
        guard isSecurityEnabled else { return true }
        
        // Check if locked out
        if isLockedOut() {
            throw SecurityError.lockedOut(until: getLockoutEndTime())
        }
        
        let context = LAContext()
        var error: NSError?
        
        // Check if biometric authentication is available
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            // Fall back to passcode authentication
            return try await authenticateWithPasscode()
        }
        
        do {
            let success = try await context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            )
            
            if success {
                await handleSuccessfulAuthentication()
                return true
            } else {
                await handleFailedAuthentication()
                return false
            }
        } catch {
            logger.error("Biometric authentication failed: \(error.localizedDescription)")
            await handleFailedAuthentication()
            
            // Fall back to passcode if biometric fails
            return try await authenticateWithPasscode()
        }
    }
    
    func authenticateWithPasscode() async throws -> Bool {
        // This would typically show a passcode entry UI
        // For now, we'll simulate it
        return false // Placeholder
    }
    
    func setPasscode(_ passcode: String) async throws {
        let hashedPasscode = try hashPasscode(passcode)
        try keychain.store(hashedPasscode, for: .passcode)
        
        logSecurityEvent(.passcodeSet, details: "Passcode updated")
        logger.info("Passcode set successfully")
    }
    
    func clearPasscode() async throws {
        try keychain.delete(.passcode)
        
        logSecurityEvent(.passcodeCleared, details: "Passcode removed")
        logger.info("Passcode cleared successfully")
    }
    
    func encryptData(_ data: Data) throws -> Data {
        guard let key = dataEncryptionKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return try encryption.encrypt(data, with: key)
    }
    
    func decryptData(_ encryptedData: Data) throws -> Data {
        guard let key = dataEncryptionKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return try encryption.decrypt(encryptedData, with: key)
    }
    
    func encryptString(_ string: String) throws -> Data {
        guard let data = string.data(using: .utf8) else {
            throw SecurityError.invalidData
        }
        
        return try encryptData(data)
    }
    
    func decryptString(_ encryptedData: Data) throws -> String {
        let data = try decryptData(encryptedData)
        
        guard let string = String(data: data, encoding: .utf8) else {
            throw SecurityError.invalidData
        }
        
        return string
    }
    
    func updatePrivacySettings(_ settings: PrivacySettings) {
        privacySettings = settings
        saveSecuritySettings()
        
        logSecurityEvent(.privacySettingsChanged, details: "Privacy settings updated")
        logger.info("Privacy settings updated")
    }
    
    func anonymizeData<T: Anonymizable>(_ data: T) -> T {
        return data.anonymized()
    }
    
    func generateSecureToken() -> String {
        let tokenData = Data((0..<32).map { _ in UInt8.random(in: 0...255) })
        return tokenData.base64EncodedString()
    }
    
    func validateDataIntegrity(_ data: Data, signature: Data) -> Bool {
        guard let key = masterKey else { return false }
        
        let computedSignature = HMAC<SHA256>.authenticationCode(for: data, using: key)
        return Data(computedSignature) == signature
    }
    
    func signData(_ data: Data) throws -> Data {
        guard let key = masterKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        let signature = HMAC<SHA256>.authenticationCode(for: data, using: key)
        return Data(signature)
    }
    
    func secureDelete(_ data: inout Data) {
        data.withUnsafeMutableBytes { bytes in
            memset_s(bytes.baseAddress, bytes.count, 0, bytes.count)
        }
        data.removeAll()
    }
    
    func getSecurityReport() -> SecurityReport {
        return SecurityReport(
            timestamp: Date(),
            isSecurityEnabled: isSecurityEnabled,
            biometricType: biometricType,
            encryptionStatus: encryptionStatus,
            privacySettings: privacySettings,
            recentEvents: Array(securityEvents.suffix(20)),
            failedAttempts: failedAttempts,
            lastFailedAttempt: lastFailedAttempt,
            isLockedOut: isLockedOut()
        )
    }
    
    func exportSecurityLogs() -> Data? {
        do {
            let export = SecurityLogExport(
                timestamp: Date(),
                events: securityEvents,
                settings: privacySettings
            )
            
            return try JSONEncoder().encode(export)
        } catch {
            logger.error("Failed to export security logs: \(error.localizedDescription)")
            return nil
        }
    }
    
    func clearSecurityLogs() {
        securityEvents.removeAll()
        saveSecuritySettings()
        
        logSecurityEvent(.logsCleared, details: "Security logs cleared")
        logger.info("Security logs cleared")
    }
    
    // MARK: - Private Methods
    
    private func setupSecurity() {
        // Set up session monitoring
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { [weak self] _ in
                self?.handleAppBackground()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { [weak self] _ in
                self?.handleAppForeground()
            }
            .store(in: &cancellables)
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
            case .opticID:
                biometricType = .opticID
            default:
                biometricType = .none
            }
        } else {
            biometricType = .none
        }
        
        logger.info("Biometric type: \(biometricType)")
    }
    
    private func storeEncryptionKeys() async throws {
        guard let masterKey = masterKey,
              let dataKey = dataEncryptionKey else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        try keychain.store(masterKey.withUnsafeBytes { Data($0) }, for: .masterKey)
        try keychain.store(dataKey.withUnsafeBytes { Data($0) }, for: .dataEncryptionKey)
        
        updateEncryptionStatus()
    }
    
    private func loadEncryptionKeys() async throws {
        do {
            let masterKeyData = try keychain.retrieve(.masterKey)
            let dataKeyData = try keychain.retrieve(.dataEncryptionKey)
            
            masterKey = SymmetricKey(data: masterKeyData)
            dataEncryptionKey = SymmetricKey(data: dataKeyData)
            
            updateEncryptionStatus()
        } catch {
            logger.error("Failed to load encryption keys: \(error.localizedDescription)")
            throw SecurityError.encryptionKeyNotFound
        }
    }
    
    private func clearEncryptionKeys() async throws {
        try keychain.delete(.masterKey)
        try keychain.delete(.dataEncryptionKey)
        
        masterKey = nil
        dataEncryptionKey = nil
        
        updateEncryptionStatus()
    }
    
    private func updateEncryptionStatus() {
        encryptionStatus = EncryptionStatus(
            isEnabled: masterKey != nil && dataEncryptionKey != nil,
            algorithm: "AES-256-GCM",
            keySize: 256,
            lastKeyRotation: Date() // Placeholder
        )
    }
    
    private func hashPasscode(_ passcode: String) throws -> Data {
        guard let passcodeData = passcode.data(using: .utf8) else {
            throw SecurityError.invalidData
        }
        
        let salt = Data((0..<16).map { _ in UInt8.random(in: 0...255) })
        let hash = SHA256.hash(data: passcodeData + salt)
        
        return salt + Data(hash)
    }
    
    private func verifyPasscode(_ passcode: String, against storedHash: Data) -> Bool {
        guard let passcodeData = passcode.data(using: .utf8),
              storedHash.count >= 16 else {
            return false
        }
        
        let salt = storedHash.prefix(16)
        let storedHashValue = storedHash.dropFirst(16)
        
        let computedHash = SHA256.hash(data: passcodeData + salt)
        
        return Data(computedHash) == storedHashValue
    }
    
    @MainActor
    private func handleSuccessfulAuthentication() {
        isAuthenticated = true
        failedAttempts = 0
        lastFailedAttempt = nil
        sessionStartTime = Date()
        
        startSessionTimer()
        
        logSecurityEvent(.authenticationSuccess, details: "User authenticated successfully")
        logger.info("Authentication successful")
    }
    
    @MainActor
    private func handleFailedAuthentication() {
        failedAttempts += 1
        lastFailedAttempt = Date()
        
        logSecurityEvent(.authenticationFailure, details: "Authentication failed (attempt \(failedAttempts))")
        
        if failedAttempts >= maxFailedAttempts {
            logSecurityEvent(.accountLocked, details: "Account locked due to too many failed attempts")
            logger.warning("Account locked due to failed attempts")
        }
        
        logger.warning("Authentication failed (attempt \(failedAttempts))")
    }
    
    private func isLockedOut() -> Bool {
        guard failedAttempts >= maxFailedAttempts,
              let lastFailure = lastFailedAttempt else {
            return false
        }
        
        return Date().timeIntervalSince(lastFailure) < lockoutDuration
    }
    
    private func getLockoutEndTime() -> Date? {
        guard let lastFailure = lastFailedAttempt else { return nil }
        return lastFailure.addingTimeInterval(lockoutDuration)
    }
    
    private func startSessionTimer() {
        sessionTimer?.invalidate()
        
        sessionTimer = Timer.scheduledTimer(withTimeInterval: sessionTimeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.handleSessionTimeout()
            }
        }
    }
    
    @MainActor
    private func handleSessionTimeout() {
        isAuthenticated = false
        sessionStartTime = nil
        
        logSecurityEvent(.sessionTimeout, details: "User session timed out")
        logger.info("Session timed out")
    }
    
    private func handleAppBackground() {
        if privacySettings.lockOnBackground {
            Task { @MainActor in
                isAuthenticated = false
                logSecurityEvent(.appBackgrounded, details: "App backgrounded, authentication required")
            }
        }
    }
    
    private func handleAppForeground() {
        if !isAuthenticated && isSecurityEnabled {
            logSecurityEvent(.appForegrounded, details: "App foregrounded, authentication required")
        }
    }
    
    private func logSecurityEvent(_ type: SecurityEventType, details: String) {
        let event = SecurityEvent(
            type: type,
            timestamp: Date(),
            details: details,
            deviceInfo: getDeviceInfo()
        )
        
        DispatchQueue.main.async {
            self.securityEvents.append(event)
            
            // Keep only recent events
            if self.securityEvents.count > 1000 {
                self.securityEvents = Array(self.securityEvents.suffix(1000))
            }
        }
        
        saveSecuritySettings()
    }
    
    private func getDeviceInfo() -> DeviceSecurityInfo {
        let device = UIDevice.current
        
        return DeviceSecurityInfo(
            deviceModel: device.model,
            systemVersion: device.systemVersion,
            isJailbroken: isDeviceJailbroken(),
            hasPasscode: LAContext().canEvaluatePolicy(.deviceOwnerAuthentication, error: nil),
            biometricType: biometricType
        )
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
        let testPath = "/private/test_jailbreak"
        do {
            try "test".write(toFile: testPath, atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(atPath: testPath)
            return true
        } catch {
            // Normal behavior for non-jailbroken devices
        }
        
        return false
    }
    
    private func saveSecuritySettings() {
        let settings = SecuritySettings(
            isSecurityEnabled: isSecurityEnabled,
            privacySettings: privacySettings,
            failedAttempts: failedAttempts,
            lastFailedAttempt: lastFailedAttempt,
            events: securityEvents
        )
        
        do {
            let data = try JSONEncoder().encode(settings)
            UserDefaults.standard.set(data, forKey: "SecuritySettings")
        } catch {
            logger.error("Failed to save security settings: \(error.localizedDescription)")
        }
    }
    
    private func loadSecuritySettings() {
        guard let data = UserDefaults.standard.data(forKey: "SecuritySettings"),
              let settings = try? JSONDecoder().decode(SecuritySettings.self, from: data) else {
            return
        }
        
        isSecurityEnabled = settings.isSecurityEnabled
        privacySettings = settings.privacySettings
        failedAttempts = settings.failedAttempts
        lastFailedAttempt = settings.lastFailedAttempt
        securityEvents = settings.events
        
        // Load encryption keys if security is enabled
        if isSecurityEnabled {
            Task {
                try? await loadEncryptionKeys()
            }
        }
    }
}

// MARK: - Supporting Types

enum BiometricType: String, CaseIterable {
    case none = "none"
    case touchID = "touchID"
    case faceID = "faceID"
    case opticID = "opticID"
    
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

struct PrivacySettings: Codable {
    var dataAnonymization = false
    var shareAnalytics = false
    var shareWithResearchers = false
    var lockOnBackground = true
    var requireBiometric = true
    var autoLockTimeout: TimeInterval = 300
    var encryptBackups = true
    var allowScreenshots = false
    var hideInAppSwitcher = true
}

struct EncryptionStatus: Codable {
    var isEnabled = false
    var algorithm = ""
    var keySize = 0
    var lastKeyRotation: Date?
}

struct SecurityEvent: Codable {
    let type: SecurityEventType
    let timestamp: Date
    let details: String
    let deviceInfo: DeviceSecurityInfo
}

enum SecurityEventType: String, Codable {
    case securityEnabled = "security_enabled"
    case securityDisabled = "security_disabled"
    case authenticationSuccess = "auth_success"
    case authenticationFailure = "auth_failure"
    case accountLocked = "account_locked"
    case sessionTimeout = "session_timeout"
    case passcodeSet = "passcode_set"
    case passcodeCleared = "passcode_cleared"
    case privacySettingsChanged = "privacy_settings_changed"
    case appBackgrounded = "app_backgrounded"
    case appForegrounded = "app_foregrounded"
    case logsCleared = "logs_cleared"
    case dataEncrypted = "data_encrypted"
    case dataDecrypted = "data_decrypted"
    case keyRotation = "key_rotation"
    case suspiciousActivity = "suspicious_activity"
}

struct DeviceSecurityInfo: Codable {
    let deviceModel: String
    let systemVersion: String
    let isJailbroken: Bool
    let hasPasscode: Bool
    let biometricType: BiometricType
}

struct SecurityReport {
    let timestamp: Date
    let isSecurityEnabled: Bool
    let biometricType: BiometricType
    let encryptionStatus: EncryptionStatus
    let privacySettings: PrivacySettings
    let recentEvents: [SecurityEvent]
    let failedAttempts: Int
    let lastFailedAttempt: Date?
    let isLockedOut: Bool
}

struct SecuritySettings: Codable {
    let isSecurityEnabled: Bool
    let privacySettings: PrivacySettings
    let failedAttempts: Int
    let lastFailedAttempt: Date?
    let events: [SecurityEvent]
}

struct SecurityLogExport: Codable {
    let timestamp: Date
    let events: [SecurityEvent]
    let settings: PrivacySettings
}

enum SecurityError: LocalizedError {
    case authenticationFailed
    case encryptionKeyNotFound
    case invalidData
    case lockedOut(until: Date?)
    case biometricNotAvailable
    case passcodeNotSet
    
    var errorDescription: String? {
        switch self {
        case .authenticationFailed:
            return "Authentication failed"
        case .encryptionKeyNotFound:
            return "Encryption key not found"
        case .invalidData:
            return "Invalid data format"
        case .lockedOut(let until):
            if let until = until {
                return "Account locked until \(until)"
            } else {
                return "Account locked"
            }
        case .biometricNotAvailable:
            return "Biometric authentication not available"
        case .passcodeNotSet:
            return "Passcode not set"
        }
    }
}

// MARK: - Anonymizable Protocol

protocol Anonymizable {
    func anonymized() -> Self
}

// MARK: - Keychain Manager

class KeychainManager {
    
    enum KeychainKey: String {
        case masterKey = "master_key"
        case dataEncryptionKey = "data_encryption_key"
        case passcode = "user_passcode"
    }
    
    func store(_ data: Data, for key: KeychainKey) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key.rawValue,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Delete existing item first
        SecItemDelete(query as CFDictionary)
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        guard status == errSecSuccess else {
            throw SecurityError.invalidData
        }
    }
    
    func retrieve(_ key: KeychainKey) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key.rawValue,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let data = result as? Data else {
            throw SecurityError.encryptionKeyNotFound
        }
        
        return data
    }
    
    func delete(_ key: KeychainKey) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key.rawValue
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw SecurityError.invalidData
        }
    }
}

// MARK: - Encryption Manager

class EncryptionManager {
    
    func encrypt(_ data: Data, with key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined!
    }
    
    func decrypt(_ encryptedData: Data, with key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
        return try AES.GCM.open(sealedBox, using: key)
    }
}