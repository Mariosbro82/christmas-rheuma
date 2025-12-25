//
//  EncryptionManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import CryptoKit
import Security

// MARK: - Encryption Manager
class EncryptionManager {
    
    // MARK: - Private Properties
    private let keychain = KeychainManager()
    private var encryptionKey: SymmetricKey?
    private var cache: [String: Any] = [:]
    private let cacheQueue = DispatchQueue(label: "encryption.cache", attributes: .concurrent)
    
    // MARK: - Constants
    private let keyIdentifier = "InflamAI.EncryptionKey"
    private let keySize = SymmetricKeySize.bits256
    
    init() {
        loadOrCreateEncryptionKey()
    }
    
    // MARK: - Key Management
    private func loadOrCreateEncryptionKey() {
        if let existingKey = loadEncryptionKey() {
            encryptionKey = existingKey
        } else {
            let newKey = SymmetricKey(size: keySize)
            saveEncryptionKey(newKey)
            encryptionKey = newKey
        }
    }
    
    private func loadEncryptionKey() -> SymmetricKey? {
        guard let keyData = keychain.getData(for: keyIdentifier) else {
            return nil
        }
        
        return SymmetricKey(data: keyData)
    }
    
    private func saveEncryptionKey(_ key: SymmetricKey) {
        let keyData = key.withUnsafeBytes { Data($0) }
        keychain.save(keyData, for: keyIdentifier)
    }
    
    func rotateEncryptionKey() throws {
        let newKey = SymmetricKey(size: keySize)
        
        // TODO: Re-encrypt existing data with new key
        // This would require a migration process for existing encrypted data
        
        saveEncryptionKey(newKey)
        encryptionKey = newKey
        clearCache()
    }
    
    // MARK: - Data Encryption
    func encrypt<T: Codable>(_ data: T) throws -> Data {
        guard let key = encryptionKey else {
            throw EncryptionError.keyNotAvailable
        }
        
        let jsonData = try JSONEncoder().encode(data)
        return try encrypt(jsonData, with: key)
    }
    
    func decrypt<T: Codable>(_ encryptedData: Data, as type: T.Type) throws -> T {
        guard let key = encryptionKey else {
            throw EncryptionError.keyNotAvailable
        }
        
        let decryptedData = try decrypt(encryptedData, with: key)
        return try JSONDecoder().decode(type, from: decryptedData)
    }
    
    func encryptString(_ string: String) throws -> Data {
        guard let key = encryptionKey else {
            throw EncryptionError.keyNotAvailable
        }
        
        guard let stringData = string.data(using: .utf8) else {
            throw EncryptionError.invalidData
        }
        
        return try encrypt(stringData, with: key)
    }
    
    func decryptString(_ encryptedData: Data) throws -> String {
        guard let key = encryptionKey else {
            throw EncryptionError.keyNotAvailable
        }
        
        let decryptedData = try decrypt(encryptedData, with: key)
        
        guard let string = String(data: decryptedData, encoding: .utf8) else {
            throw EncryptionError.invalidData
        }
        
        return string
    }
    
    // MARK: - Core Encryption Methods
    private func encrypt(_ data: Data, with key: SymmetricKey) throws -> Data {
        do {
            let sealedBox = try AES.GCM.seal(data, using: key)
            return sealedBox.combined!
        } catch {
            throw EncryptionError.encryptionFailed(error.localizedDescription)
        }
    }
    
    private func decrypt(_ encryptedData: Data, with key: SymmetricKey) throws -> Data {
        do {
            let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
            return try AES.GCM.open(sealedBox, using: key)
        } catch {
            throw EncryptionError.decryptionFailed(error.localizedDescription)
        }
    }
    
    // MARK: - File Encryption
    func encryptFile(at url: URL) throws -> URL {
        guard let key = encryptionKey else {
            throw EncryptionError.keyNotAvailable
        }
        
        let data = try Data(contentsOf: url)
        let encryptedData = try encrypt(data, with: key)
        
        let encryptedURL = url.appendingPathExtension("encrypted")
        try encryptedData.write(to: encryptedURL)
        
        return encryptedURL
    }
    
    func decryptFile(at url: URL) throws -> URL {
        guard let key = encryptionKey else {
            throw EncryptionError.keyNotAvailable
        }
        
        let encryptedData = try Data(contentsOf: url)
        let decryptedData = try decrypt(encryptedData, with: key)
        
        let decryptedURL = url.deletingPathExtension()
        try decryptedData.write(to: decryptedURL)
        
        return decryptedURL
    }
    
    // MARK: - Hashing
    func hash(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    func hashString(_ string: String) -> String {
        guard let data = string.data(using: .utf8) else {
            return ""
        }
        return hash(data)
    }
    
    func verifyHash(_ data: Data, expectedHash: String) -> Bool {
        let actualHash = hash(data)
        return actualHash == expectedHash
    }
    
    // MARK: - Digital Signatures
    func generateKeyPair() throws -> (privateKey: P256.Signing.PrivateKey, publicKey: P256.Signing.PublicKey) {
        let privateKey = P256.Signing.PrivateKey()
        let publicKey = privateKey.publicKey
        return (privateKey, publicKey)
    }
    
    func sign(_ data: Data, with privateKey: P256.Signing.PrivateKey) throws -> Data {
        let signature = try privateKey.signature(for: data)
        return signature.rawRepresentation
    }
    
    func verify(_ signature: Data, for data: Data, with publicKey: P256.Signing.PublicKey) throws -> Bool {
        let ecdsaSignature = try P256.Signing.ECDSASignature(rawRepresentation: signature)
        return publicKey.isValidSignature(ecdsaSignature, for: data)
    }
    
    // MARK: - Key Derivation
    func deriveKey(from password: String, salt: Data) throws -> SymmetricKey {
        guard let passwordData = password.data(using: .utf8) else {
            throw EncryptionError.invalidData
        }
        
        let derivedKey = HKDF<SHA256>.deriveKey(
            inputKeyMaterial: SymmetricKey(data: passwordData),
            salt: salt,
            outputByteCount: 32
        )
        
        return derivedKey
    }
    
    func generateSalt() -> Data {
        var salt = Data(count: 16)
        _ = salt.withUnsafeMutableBytes { bytes in
            SecRandomCopyBytes(kSecRandomDefault, 16, bytes.bindMemory(to: UInt8.self).baseAddress!)
        }
        return salt
    }
    
    // MARK: - Secure Random Generation
    func generateSecureRandomData(length: Int) -> Data {
        var data = Data(count: length)
        _ = data.withUnsafeMutableBytes { bytes in
            SecRandomCopyBytes(kSecRandomDefault, length, bytes.bindMemory(to: UInt8.self).baseAddress!)
        }
        return data
    }
    
    func generateSecureRandomString(length: Int) -> String {
        let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        let randomData = generateSecureRandomData(length: length)
        
        return randomData.map { byte in
            characters[characters.index(characters.startIndex, offsetBy: Int(byte) % characters.count)]
        }.joined()
    }
    
    // MARK: - Cache Management
    func clearCache() {
        cacheQueue.async(flags: .barrier) {
            self.cache.removeAll()
        }
    }
    
    private func getCachedValue<T>(for key: String, as type: T.Type) -> T? {
        return cacheQueue.sync {
            return cache[key] as? T
        }
    }
    
    private func setCachedValue<T>(_ value: T, for key: String) {
        cacheQueue.async(flags: .barrier) {
            self.cache[key] = value
        }
    }
    
    // MARK: - Encryption Validation
    func validateEncryption() throws -> EncryptionValidationResult {
        let testData = "InflamAI Encryption Test".data(using: .utf8)!
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test encryption/decryption
        let encryptedData = try encrypt(testData, with: encryptionKey!)
        let decryptedData = try decrypt(encryptedData, with: encryptionKey!)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let duration = endTime - startTime
        
        let isValid = testData == decryptedData
        
        return EncryptionValidationResult(
            isValid: isValid,
            encryptionTime: duration,
            keyStrength: keySize.bitCount,
            algorithm: "AES-GCM",
            timestamp: Date()
        )
    }
    
    // MARK: - Backup and Recovery
    func createEncryptedBackup<T: Codable>(_ data: T, with password: String) throws -> EncryptedBackup {
        let salt = generateSalt()
        let backupKey = try deriveKey(from: password, salt: salt)
        
        let jsonData = try JSONEncoder().encode(data)
        let encryptedData = try encrypt(jsonData, with: backupKey)
        
        let backup = EncryptedBackup(
            encryptedData: encryptedData,
            salt: salt,
            timestamp: Date(),
            version: "1.0"
        )
        
        return backup
    }
    
    func restoreFromEncryptedBackup<T: Codable>(_ backup: EncryptedBackup, with password: String, as type: T.Type) throws -> T {
        let backupKey = try deriveKey(from: password, salt: backup.salt)
        let decryptedData = try decrypt(backup.encryptedData, with: backupKey)
        
        return try JSONDecoder().decode(type, from: decryptedData)
    }
}

// MARK: - Supporting Types
enum EncryptionError: LocalizedError {
    case keyNotAvailable
    case invalidData
    case encryptionFailed(String)
    case decryptionFailed(String)
    case keyDerivationFailed
    case signatureFailed
    case verificationFailed
    
    var errorDescription: String? {
        switch self {
        case .keyNotAvailable:
            return "Encryption key is not available"
        case .invalidData:
            return "Invalid data provided for encryption"
        case .encryptionFailed(let message):
            return "Encryption failed: \(message)"
        case .decryptionFailed(let message):
            return "Decryption failed: \(message)"
        case .keyDerivationFailed:
            return "Key derivation failed"
        case .signatureFailed:
            return "Digital signature creation failed"
        case .verificationFailed:
            return "Digital signature verification failed"
        }
    }
}

struct EncryptionValidationResult {
    let isValid: Bool
    let encryptionTime: TimeInterval
    let keyStrength: Int
    let algorithm: String
    let timestamp: Date
}

struct EncryptedBackup: Codable {
    let encryptedData: Data
    let salt: Data
    let timestamp: Date
    let version: String
}

// MARK: - Keychain Manager
class KeychainManager {
    
    private let service = "InflamAI"
    
    func save(_ data: Data, for key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Delete any existing item
        SecItemDelete(query as CFDictionary)
        
        // Add the new item
        let status = SecItemAdd(query as CFDictionary, nil)
        
        if status != errSecSuccess {
            print("Failed to save to keychain: \(status)")
        }
    }
    
    func getData(for key: String) -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        if status == errSecSuccess {
            return result as? Data
        }
        
        return nil
    }
    
    func delete(for key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key
        ]
        
        SecItemDelete(query as CFDictionary)
    }
    
    func deleteAll() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service
        ]
        
        SecItemDelete(query as CFDictionary)
    }
}

// MARK: - Audit Manager
class AuditManager {
    
    private let auditQueue = DispatchQueue(label: "audit.logging", qos: .utility)
    private let maxLogEntries = 10000
    
    func logEvent(_ event: SecurityAuditLog) {
        auditQueue.async {
            self.persistAuditLog(event)
        }
    }
    
    private func persistAuditLog(_ log: SecurityAuditLog) {
        // In a real implementation, this would write to a secure log file
        // or send to a logging service
        
        let logEntry = "[\(log.timestamp)] \(log.eventType.rawValue): \(log.details)"
        print("AUDIT: \(logEntry)")
        
        // Save to UserDefaults for demo purposes
        // In production, use a more secure storage mechanism
        var existingLogs = UserDefaults.standard.array(forKey: "AuditLogs") as? [String] ?? []
        existingLogs.append(logEntry)
        
        // Keep only recent logs
        if existingLogs.count > maxLogEntries {
            existingLogs.removeFirst(existingLogs.count - maxLogEntries)
        }
        
        UserDefaults.standard.set(existingLogs, forKey: "AuditLogs")
    }
    
    func getAuditLogs() -> [String] {
        return UserDefaults.standard.array(forKey: "AuditLogs") as? [String] ?? []
    }
    
    func clearAuditLogs() {
        UserDefaults.standard.removeObject(forKey: "AuditLogs")
    }
}