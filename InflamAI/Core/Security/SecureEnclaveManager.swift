//
//  SecureEnclaveManager.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Security
import CryptoKit
import LocalAuthentication

class SecureEnclaveManager {
    // MARK: - Properties
    private let keyTag = "com.inflamai.secureenclave.key"
    private let deviceIdTag = "com.inflamai.device.id"
    private var isInitialized = false
    
    // MARK: - Initialization
    init() {}
    
    func initialize() async {
        guard !isInitialized else { return }
        
        await createSecureEnclaveKey()
        await generateDeviceId()
        isInitialized = true
    }
    
    // MARK: - Key Management
    private func createSecureEnclaveKey() async {
        let access = SecAccessControlCreateWithFlags(
            kCFAllocatorDefault,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            [.privateKeyUsage, .biometryAny],
            nil
        )
        
        let attributes: [String: Any] = [
            kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
            kSecAttrKeySizeInBits as String: 256,
            kSecAttrTokenID as String: kSecAttrTokenIDSecureEnclave,
            kSecPrivateKeyAttrs as String: [
                kSecAttrIsPermanent as String: true,
                kSecAttrApplicationTag as String: keyTag.data(using: .utf8)!,
                kSecAttrAccessControl as String: access as Any
            ]
        ]
        
        var error: Unmanaged<CFError>?
        guard let privateKey = SecKeyCreateRandomKey(attributes as CFDictionary, &error) else {
            if let error = error?.takeRetainedValue() {
                print("Failed to create Secure Enclave key: \(error)")
            }
            return
        }
        
        print("Secure Enclave key created successfully")
    }
    
    func getSecureEnclaveKey() async -> SecKey? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationTag as String: keyTag.data(using: .utf8)!,
            kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
            kSecReturnRef as String: true
        ]
        
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        
        guard status == errSecSuccess else {
            print("Failed to retrieve Secure Enclave key: \(status)")
            return nil
        }
        
        return (item as! SecKey)
    }
    
    // MARK: - Device ID Management
    private func generateDeviceId() async {
        guard await getDeviceId().isEmpty else { return }
        
        let deviceId = UUID().uuidString
        await storeDeviceId(deviceId)
    }
    
    func getDeviceId() async -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: deviceIdTag,
            kSecReturnData as String: true
        ]
        
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        
        guard status == errSecSuccess,
              let data = item as? Data,
              let deviceId = String(data: data, encoding: .utf8) else {
            return ""
        }
        
        return deviceId
    }
    
    private func storeDeviceId(_ deviceId: String) async {
        let data = deviceId.data(using: .utf8)!
        
        let attributes: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: deviceIdTag,
            kSecValueData as String: data
        ]
        
        let status = SecItemAdd(attributes as CFDictionary, nil)
        if status != errSecSuccess {
            print("Failed to store device ID: \(status)")
        }
    }
    
    // MARK: - Encryption/Decryption
    func encrypt(data: Data) async -> Data? {
        guard let key = await getSecureEnclaveKey() else { return nil }
        
        var error: Unmanaged<CFError>?
        guard let encryptedData = SecKeyCreateEncryptedData(
            key,
            .eciesEncryptionCofactorVariableIVX963SHA256AESGCM,
            data as CFData,
            &error
        ) else {
            if let error = error?.takeRetainedValue() {
                print("Encryption failed: \(error)")
            }
            return nil
        }
        
        return encryptedData as Data
    }
    
    func decrypt(data: Data) async -> Data? {
        guard let key = await getSecureEnclaveKey() else { return nil }
        
        var error: Unmanaged<CFError>?
        guard let decryptedData = SecKeyCreateDecryptedData(
            key,
            .eciesEncryptionCofactorVariableIVX963SHA256AESGCM,
            data as CFData,
            &error
        ) else {
            if let error = error?.takeRetainedValue() {
                print("Decryption failed: \(error)")
            }
            return nil
        }
        
        return decryptedData as Data
    }
    
    // MARK: - Device Integrity
    func checkDeviceIntegrity() async -> DeviceIntegrityStatus {
        var status: DeviceIntegrityStatus = .secure
        
        // Check for jailbreak indicators
        if await isJailbroken() {
            status = .compromised
        }
        
        // Check for debugging
        if await isBeingDebugged() {
            status = .compromised
        }
        
        // Check for tampering
        if await isTampered() {
            status = .compromised
        }
        
        return status
    }
    
    private func isJailbroken() async -> Bool {
        // Check for common jailbreak indicators
        let jailbreakPaths = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/",
            "/private/var/lib/cydia",
            "/private/var/mobile/Library/SBSettings/Themes",
            "/Library/MobileSubstrate/DynamicLibraries/LiveClock.plist",
            "/System/Library/LaunchDaemons/com.ikey.bbot.plist",
            "/Library/MobileSubstrate/DynamicLibraries/Veency.plist",
            "/private/var/lib/dpkg/info/com.saurik.Cydia.list",
            "/Applications/FakeCarrier.app",
            "/Applications/Icy.app",
            "/Applications/IntelliScreen.app",
            "/Applications/MxTube.app",
            "/Applications/RockApp.app",
            "/Applications/SBSettings.app",
            "/Applications/WinterBoard.app"
        ]
        
        for path in jailbreakPaths {
            if FileManager.default.fileExists(atPath: path) {
                return true
            }
        }
        
        // Check if we can write to system directories
        let testString = "test"
        do {
            try testString.write(toFile: "/private/test.txt", atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(atPath: "/private/test.txt")
            return true // Should not be able to write here
        } catch {
            // Good, we can't write to system directories
        }
        
        return false
    }
    
    private func isBeingDebugged() async -> Bool {
        var info = kinfo_proc()
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        var size = MemoryLayout<kinfo_proc>.stride
        
        let result = sysctl(&mib, u_int(mib.count), &info, &size, nil, 0)
        
        if result != 0 {
            return false
        }
        
        return (info.kp_proc.p_flag & P_TRACED) != 0
    }
    
    private func isTampered() async -> Bool {
        // Check app bundle integrity
        guard let bundlePath = Bundle.main.bundlePath as NSString? else {
            return true
        }
        
        // Check if the app is running from expected location
        if !bundlePath.contains("/var/containers/Bundle/Application/") &&
           !bundlePath.contains("/Applications/") {
            return true
        }
        
        // Additional integrity checks can be added here
        return false
    }
    
    // MARK: - Secure Storage
    func secureStore(key: String, data: Data) async -> Bool {
        guard let encryptedData = await encrypt(data: data) else {
            return false
        }
        
        let attributes: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: encryptedData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Delete existing item first
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(deleteQuery as CFDictionary)
        
        let status = SecItemAdd(attributes as CFDictionary, nil)
        return status == errSecSuccess
    }
    
    func secureRetrieve(key: String) async -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]
        
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        
        guard status == errSecSuccess,
              let encryptedData = item as? Data else {
            return nil
        }
        
        return await decrypt(data: encryptedData)
    }
    
    func secureDelete(key: String) async -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess
    }
}