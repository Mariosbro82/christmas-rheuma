//
//  EncryptionKeyManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import CryptoKit
import Security
import Combine

class EncryptionKeyManager {
    // MARK: - Properties
    private var masterKey: SymmetricKey?
    private var dataEncryptionKeys: [String: SymmetricKey] = [:]
    private var keyEncryptionKeys: [String: SymmetricKey] = [:]
    private var keyMetadata: [String: KeyMetadata] = [:]
    private var currentKeyId: String?
    private var forwardSecrecyEnabled: Bool = false
    
    private let keychain: KeychainManager
    private let secureRandom: SecureRandomGenerator
    private let keyDerivation: KeyDerivationEngine
    
    // MARK: - Constants
    private let masterKeyTag = "com.inflamai.masterkey"
    private let keyRotationInterval: TimeInterval = 86400 * 30 // 30 days
    private let maxKeyAge: TimeInterval = 86400 * 90 // 90 days
    
    init() {
        self.keychain = KeychainManager()
        self.secureRandom = SecureRandomGenerator()
        self.keyDerivation = KeyDerivationEngine()
    }
    
    // MARK: - Initialization
    func initialize() async {
        await loadMasterKey()
        await loadDataEncryptionKeys()
        await loadKeyEncryptionKeys()
        await validateKeys()
    }
    
    private func loadMasterKey() async {
        if let keyData = await keychain.getData(for: masterKeyTag) {
            self.masterKey = SymmetricKey(data: keyData)
        }
    }
    
    private func loadDataEncryptionKeys() async {
        let keyIds = await keychain.getAllKeys(withPrefix: "dek_")
        for keyId in keyIds {
            if let keyData = await keychain.getData(for: keyId) {
                dataEncryptionKeys[keyId] = SymmetricKey(data: keyData)
            }
        }
    }
    
    private func loadKeyEncryptionKeys() async {
        let keyIds = await keychain.getAllKeys(withPrefix: "kek_")
        for keyId in keyIds {
            if let keyData = await keychain.getData(for: keyId) {
                keyEncryptionKeys[keyId] = SymmetricKey(data: keyData)
            }
        }
    }
    
    private func validateKeys() async {
        // Remove expired keys
        let expiredKeys = keyMetadata.filter { _, metadata in
            Date().timeIntervalSince(metadata.createdAt) > maxKeyAge
        }
        
        for (keyId, _) in expiredKeys {
            await removeKey(keyId)
        }
        
        // Set current key if none exists
        if currentKeyId == nil || dataEncryptionKeys[currentKeyId!] == nil {
            currentKeyId = dataEncryptionKeys.keys.first
        }
    }
    
    // MARK: - Key Generation
    func generateMasterKey() async {
        let key = SymmetricKey(size: .bits256)
        self.masterKey = key
        
        let keyData = key.withUnsafeBytes { Data($0) }
        await keychain.store(keyData, for: masterKeyTag)
    }
    
    func generateDataEncryptionKeys() async {
        let keyId = "dek_" + UUID().uuidString
        let key = SymmetricKey(size: .bits256)
        
        dataEncryptionKeys[keyId] = key
        currentKeyId = keyId
        
        let metadata = KeyMetadata(
            keyId: keyId,
            keyType: .dataEncryption,
            createdAt: Date(),
            algorithm: "AES-256-GCM",
            purpose: "Data encryption"
        )
        keyMetadata[keyId] = metadata
        
        let keyData = key.withUnsafeBytes { Data($0) }
        await keychain.store(keyData, for: keyId)
        await keychain.store(try! JSONEncoder().encode(metadata), for: "meta_" + keyId)
    }
    
    func generateKeyEncryptionKeys() async {
        let keyId = "kek_" + UUID().uuidString
        let key = SymmetricKey(size: .bits256)
        
        keyEncryptionKeys[keyId] = key
        
        let metadata = KeyMetadata(
            keyId: keyId,
            keyType: .keyEncryption,
            createdAt: Date(),
            algorithm: "AES-256-GCM",
            purpose: "Key encryption"
        )
        keyMetadata[keyId] = metadata
        
        let keyData = key.withUnsafeBytes { Data($0) }
        await keychain.store(keyData, for: keyId)
        await keychain.store(try! JSONEncoder().encode(metadata), for: "meta_" + keyId)
    }
    
    // MARK: - Key Access
    func getCurrentDataKey() async -> SymmetricKey {
        guard let keyId = currentKeyId,
              let key = dataEncryptionKeys[keyId] else {
            // Generate new key if none exists
            await generateDataEncryptionKeys()
            return dataEncryptionKeys[currentKeyId!]!
        }
        return key
    }
    
    func getDataKey(for keyId: String) async -> SymmetricKey? {
        return dataEncryptionKeys[keyId]
    }
    
    func getCurrentKeyId() async -> String {
        if let keyId = currentKeyId {
            return keyId
        }
        
        await generateDataEncryptionKeys()
        return currentKeyId!
    }
    
    func hasValidKeys() async -> Bool {
        return masterKey != nil && !dataEncryptionKeys.isEmpty
    }
    
    // MARK: - Key Rotation
    func rotateKeys() async {
        // Generate new data encryption key
        await generateDataEncryptionKeys()
        
        // If forward secrecy is enabled, remove old keys
        if forwardSecrecyEnabled {
            await removeOldKeys()
        }
    }
    
    private func removeOldKeys() async {
        let cutoffDate = Date().addingTimeInterval(-keyRotationInterval)
        
        let oldKeys = keyMetadata.filter { _, metadata in
            metadata.createdAt < cutoffDate
        }
        
        for (keyId, _) in oldKeys {
            await removeKey(keyId)
        }
    }
    
    private func removeKey(_ keyId: String) async {
        dataEncryptionKeys.removeValue(forKey: keyId)
        keyEncryptionKeys.removeValue(forKey: keyId)
        keyMetadata.removeValue(forKey: keyId)
        
        await keychain.delete(keyId)
        await keychain.delete("meta_" + keyId)
    }
    
    // MARK: - Forward Secrecy
    func enableForwardSecrecy() async {
        forwardSecrecyEnabled = true
    }
    
    func disableForwardSecrecy() async {
        forwardSecrecyEnabled = false
    }
    
    // MARK: - Key Derivation
    func deriveKey(from password: String, salt: Data) async -> SymmetricKey {
        return await keyDerivation.deriveKey(from: password, salt: salt)
    }
    
    func deriveKeyFromBiometric(_ biometricData: Data) async -> SymmetricKey {
        return await keyDerivation.deriveKeyFromBiometric(biometricData)
    }
    
    // MARK: - Key Export/Import
    func exportKeys(password: String) async throws -> Data {
        guard let masterKey = masterKey else {
            throw KeyManagerError.noMasterKey
        }
        
        let exportData = KeyExportData(
            masterKey: masterKey.withUnsafeBytes { Data($0) },
            dataKeys: dataEncryptionKeys.mapValues { $0.withUnsafeBytes { Data($0) } },
            keyMetadata: keyMetadata
        )
        
        let jsonData = try JSONEncoder().encode(exportData)
        
        // Encrypt with password
        let derivedKey = await deriveKey(from: password, salt: secureRandom.generateBytes(32))
        return try await encryptData(jsonData, with: derivedKey)
    }
    
    func importKeys(from data: Data, password: String) async throws {
        let derivedKey = await deriveKey(from: password, salt: secureRandom.generateBytes(32))
        let decryptedData = try await decryptData(data, with: derivedKey)
        
        let exportData = try JSONDecoder().decode(KeyExportData.self, from: decryptedData)
        
        // Import keys
        self.masterKey = SymmetricKey(data: exportData.masterKey)
        self.dataEncryptionKeys = exportData.dataKeys.mapValues { SymmetricKey(data: $0) }
        self.keyMetadata = exportData.keyMetadata
        
        // Store in keychain
        await keychain.store(exportData.masterKey, for: masterKeyTag)
        
        for (keyId, keyData) in exportData.dataKeys {
            await keychain.store(keyData, for: keyId)
        }
        
        for (keyId, metadata) in exportData.keyMetadata {
            let metadataData = try JSONEncoder().encode(metadata)
            await keychain.store(metadataData, for: "meta_" + keyId)
        }
    }
    
    // MARK: - Utility Methods
    private func encryptData(_ data: Data, with key: SymmetricKey) async throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined!
    }
    
    private func decryptData(_ data: Data, with key: SymmetricKey) async throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: data)
        return try AES.GCM.open(sealedBox, using: key)
    }
    
    // MARK: - Key Information
    func getKeyInfo() async -> KeyManagerInfo {
        return KeyManagerInfo(
            totalKeys: dataEncryptionKeys.count,
            currentKeyId: currentKeyId,
            oldestKeyDate: keyMetadata.values.map { $0.createdAt }.min(),
            newestKeyDate: keyMetadata.values.map { $0.createdAt }.max(),
            forwardSecrecyEnabled: forwardSecrecyEnabled
        )
    }
}

// MARK: - Supporting Types
enum KeyManagerError: Error {
    case noMasterKey
    case keyNotFound
    case invalidKeyData
    case encryptionFailed
    case decryptionFailed
}

enum KeyType {
    case master
    case dataEncryption
    case keyEncryption
    case derived
}

struct KeyMetadata: Codable {
    let keyId: String
    let keyType: KeyType
    let createdAt: Date
    let algorithm: String
    let purpose: String
}

struct KeyExportData: Codable {
    let masterKey: Data
    let dataKeys: [String: Data]
    let keyMetadata: [String: KeyMetadata]
}

struct KeyManagerInfo {
    let totalKeys: Int
    let currentKeyId: String?
    let oldestKeyDate: Date?
    let newestKeyDate: Date?
    let forwardSecrecyEnabled: Bool
}

// MARK: - KeyType Codable Extension
extension KeyType: Codable {
    enum CodingKeys: String, CodingKey {
        case type
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        
        switch type {
        case "master": self = .master
        case "dataEncryption": self = .dataEncryption
        case "keyEncryption": self = .keyEncryption
        case "derived": self = .derived
        default: throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Invalid key type"))
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        switch self {
        case .master: try container.encode("master", forKey: .type)
        case .dataEncryption: try container.encode("dataEncryption", forKey: .type)
        case .keyEncryption: try container.encode("keyEncryption", forKey: .type)
        case .derived: try container.encode("derived", forKey: .type)
        }
    }
}