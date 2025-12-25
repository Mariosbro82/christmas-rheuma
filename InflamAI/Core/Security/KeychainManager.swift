//
//  KeychainManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Security
import LocalAuthentication

class KeychainManager {
    // MARK: - Properties
    private let serviceName: String
    private let accessGroup: String?
    private let biometricProtection: Bool
    
    // MARK: - Initialization
    init(serviceName: String = "com.inflamai.keychain", accessGroup: String? = nil, biometricProtection: Bool = true) {
        self.serviceName = serviceName
        self.accessGroup = accessGroup
        self.biometricProtection = biometricProtection
    }
    
    // MARK: - Store Methods
    func store(_ data: Data, for key: String, requireBiometric: Bool = false) async -> Bool {
        let query = buildQuery(for: key)
        
        // Delete existing item first
        SecItemDelete(query as CFDictionary)
        
        var attributes = query
        attributes[kSecValueData] = data
        
        // Add access control if biometric protection is enabled
        if biometricProtection || requireBiometric {
            let accessControl = createAccessControl(requireBiometric: requireBiometric)
            if let accessControl = accessControl {
                attributes[kSecAttrAccessControl] = accessControl
            }
        }
        
        let status = SecItemAdd(attributes as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    func store(_ string: String, for key: String, requireBiometric: Bool = false) async -> Bool {
        guard let data = string.data(using: .utf8) else { return false }
        return await store(data, for: key, requireBiometric: requireBiometric)
    }
    
    // MARK: - Retrieve Methods
    func getData(for key: String) async -> Data? {
        var query = buildQuery(for: key)
        query[kSecReturnData] = true
        query[kSecMatchLimit] = kSecMatchLimitOne
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess else {
            if status == errSecUserCancel {
                // User cancelled biometric authentication
                return nil
            }
            return nil
        }
        
        return result as? Data
    }
    
    func getString(for key: String) async -> String? {
        guard let data = await getData(for: key) else { return nil }
        return String(data: data, encoding: .utf8)
    }
    
    // MARK: - Update Methods
    func update(_ data: Data, for key: String) async -> Bool {
        let query = buildQuery(for: key)
        let attributes: [String: Any] = [kSecValueData as String: data]
        
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        return status == errSecSuccess
    }
    
    func update(_ string: String, for key: String) async -> Bool {
        guard let data = string.data(using: .utf8) else { return false }
        return await update(data, for: key)
    }
    
    // MARK: - Delete Methods
    func delete(_ key: String) async -> Bool {
        let query = buildQuery(for: key)
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
    
    func deleteAll() async -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
    
    // MARK: - Existence Check
    func exists(_ key: String) async -> Bool {
        var query = buildQuery(for: key)
        query[kSecReturnData] = false
        query[kSecMatchLimit] = kSecMatchLimitOne
        
        let status = SecItemCopyMatching(query as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    // MARK: - List Keys
    func getAllKeys(withPrefix prefix: String = "") async -> [String] {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let items = result as? [[String: Any]] else {
            return []
        }
        
        let keys = items.compactMap { item in
            item[kSecAttrAccount as String] as? String
        }
        
        if prefix.isEmpty {
            return keys
        } else {
            return keys.filter { $0.hasPrefix(prefix) }
        }
    }
    
    // MARK: - Secure Storage for Sensitive Data
    func storeSecurely(_ data: Data, for key: String, context: LAContext? = nil) async -> Bool {
        let query = buildSecureQuery(for: key, context: context)
        
        // Delete existing item first
        SecItemDelete(query as CFDictionary)
        
        var attributes = query
        attributes[kSecValueData] = data
        
        let status = SecItemAdd(attributes as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    func getSecureData(for key: String, context: LAContext? = nil) async -> Data? {
        var query = buildSecureQuery(for: key, context: context)
        query[kSecReturnData] = true
        query[kSecMatchLimit] = kSecMatchLimitOne
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess else { return nil }
        return result as? Data
    }
    
    // MARK: - Biometric Authentication
    func authenticateWithBiometrics() async -> Bool {
        let context = LAContext()
        var error: NSError?
        
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            return false
        }
        
        do {
            let result = try await context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: "Authenticate to access your health data"
            )
            return result
        } catch {
            return false
        }
    }
    
    // MARK: - Backup and Restore
    func exportKeychain(password: String) async -> Data? {
        let allKeys = await getAllKeys()
        var exportData: [String: Data] = [:]
        
        for key in allKeys {
            if let data = await getData(for: key) {
                exportData[key] = data
            }
        }
        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: exportData.mapValues { $0.base64EncodedString() })
            return try await encryptData(jsonData, password: password)
        } catch {
            return nil
        }
    }
    
    func importKeychain(from data: Data, password: String) async -> Bool {
        do {
            let decryptedData = try await decryptData(data, password: password)
            let jsonObject = try JSONSerialization.jsonObject(with: decryptedData) as? [String: String]
            
            guard let exportData = jsonObject else { return false }
            
            for (key, base64String) in exportData {
                guard let data = Data(base64Encoded: base64String) else { continue }
                _ = await store(data, for: key)
            }
            
            return true
        } catch {
            return false
        }
    }
    
    // MARK: - Private Helper Methods
    private func buildQuery(for key: String) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key
        ]
        
        if let accessGroup = accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        
        return query
    }
    
    private func buildSecureQuery(for key: String, context: LAContext?) -> [String: Any] {
        var query = buildQuery(for: key)
        
        let accessControl = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .biometryAny,
            nil
        )
        
        if let accessControl = accessControl {
            query[kSecAttrAccessControl] = accessControl
        }
        
        if let context = context {
            query[kSecUseAuthenticationContext] = context
        }
        
        return query
    }
    
    private func createAccessControl(requireBiometric: Bool) -> SecAccessControl? {
        let flags: SecAccessControlCreateFlags = requireBiometric ? .biometryAny : .devicePasscode
        
        return SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            flags,
            nil
        )
    }
    
    private func encryptData(_ data: Data, password: String) async throws -> Data {
        // Simple encryption using password - in production, use proper key derivation
        let key = password.data(using: .utf8)?.sha256() ?? Data()
        return try await performAESEncryption(data, key: key)
    }
    
    private func decryptData(_ data: Data, password: String) async throws -> Data {
        // Simple decryption using password - in production, use proper key derivation
        let key = password.data(using: .utf8)?.sha256() ?? Data()
        return try await performAESDecryption(data, key: key)
    }
    
    private func performAESEncryption(_ data: Data, key: Data) async throws -> Data {
        // Placeholder for AES encryption
        // In a real implementation, use CryptoKit or CommonCrypto
        return data
    }
    
    private func performAESDecryption(_ data: Data, key: Data) async throws -> Data {
        // Placeholder for AES decryption
        // In a real implementation, use CryptoKit or CommonCrypto
        return data
    }
    
    // MARK: - Keychain Status
    func getKeychainStatus() async -> KeychainStatus {
        let allKeys = await getAllKeys()
        let totalItems = allKeys.count
        
        var biometricProtectedItems = 0
        for key in allKeys {
            if await isItemBiometricProtected(key) {
                biometricProtectedItems += 1
            }
        }
        
        return KeychainStatus(
            totalItems: totalItems,
            biometricProtectedItems: biometricProtectedItems,
            serviceName: serviceName,
            accessGroup: accessGroup,
            biometricAvailable: await isBiometricAvailable()
        )
    }
    
    private func isItemBiometricProtected(_ key: String) async -> Bool {
        var query = buildQuery(for: key)
        query[kSecReturnAttributes] = true
        query[kSecMatchLimit] = kSecMatchLimitOne
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        guard status == errSecSuccess,
              let attributes = result as? [String: Any] else {
            return false
        }
        
        return attributes[kSecAttrAccessControl as String] != nil
    }
    
    private func isBiometricAvailable() async -> Bool {
        let context = LAContext()
        var error: NSError?
        return context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
    }
    
    // MARK: - Error Handling
    func getLastError() -> KeychainError? {
        // Implementation would track the last error that occurred
        return nil
    }
}

// MARK: - Supporting Types
struct KeychainStatus {
    let totalItems: Int
    let biometricProtectedItems: Int
    let serviceName: String
    let accessGroup: String?
    let biometricAvailable: Bool
}

enum KeychainError: Error {
    case itemNotFound
    case duplicateItem
    case invalidData
    case authenticationFailed
    case biometricNotAvailable
    case userCancel
    case unknown(OSStatus)
    
    init(status: OSStatus) {
        switch status {
        case errSecItemNotFound:
            self = .itemNotFound
        case errSecDuplicateItem:
            self = .duplicateItem
        case errSecAuthFailed:
            self = .authenticationFailed
        case errSecUserCancel:
            self = .userCancel
        default:
            self = .unknown(status)
        }
    }
}

// MARK: - Data Extension for SHA256
extension Data {
    func sha256() -> Data {
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        self.withUnsafeBytes {
            _ = CC_SHA256($0.baseAddress, CC_LONG(self.count), &hash)
        }
        return Data(hash)
    }
}

// MARK: - CommonCrypto Import
import CommonCrypto