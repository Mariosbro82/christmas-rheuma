//
//  SecureRandomGenerator.swift
//  InflamAI-Swift
//
//  Created by SOLO Coding on 2024-01-21.
//

import Foundation
import Security
import CryptoKit

// MARK: - Secure Random Generator
class SecureRandomGenerator {
    
    // MARK: - Properties
    private let queue = DispatchQueue(label: "secure.random.generator", qos: .userInitiated)
    
    // MARK: - Initialization
    init() {}
    
    // MARK: - Random Data Generation
    
    /// Generates cryptographically secure random bytes
    /// - Parameter count: Number of bytes to generate
    /// - Returns: Random data
    func generateBytes(_ count: Int) async -> Data {
        return await withCheckedContinuation { continuation in
            queue.async {
                var randomBytes = Data(count: count)
                let result = randomBytes.withUnsafeMutableBytes { bytes in
                    SecRandomCopyBytes(kSecRandomDefault, count, bytes.bindMemory(to: UInt8.self).baseAddress!)
                }
                
                if result == errSecSuccess {
                    continuation.resume(returning: randomBytes)
                } else {
                    // Fallback to CryptoKit if SecRandomCopyBytes fails
                    let fallbackData = Data((0..<count).map { _ in UInt8.random(in: 0...255) })
                    continuation.resume(returning: fallbackData)
                }
            }
        }
    }
    
    /// Generates a cryptographically secure random string
    /// - Parameters:
    ///   - length: Length of the string
    ///   - charset: Character set to use (default: alphanumeric)
    /// - Returns: Random string
    func generateString(length: Int, charset: CharacterSet = .alphanumerics) async -> String {
        let characters = charset.characters
        guard !characters.isEmpty else { return "" }
        
        let randomData = await generateBytes(length)
        return String(randomData.map { byte in
            characters[Int(byte) % characters.count]
        })
    }
    
    /// Generates a cryptographically secure random integer
    /// - Parameters:
    ///   - min: Minimum value (inclusive)
    ///   - max: Maximum value (inclusive)
    /// - Returns: Random integer
    func generateInt(min: Int, max: Int) async -> Int {
        guard min <= max else { return min }
        
        let range = max - min + 1
        let randomData = await generateBytes(MemoryLayout<UInt32>.size)
        let randomValue = randomData.withUnsafeBytes { $0.load(as: UInt32.self) }
        
        return min + Int(randomValue % UInt32(range))
    }
    
    /// Generates a cryptographically secure random UUID
    /// - Returns: Random UUID
    func generateUUID() async -> UUID {
        return await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: UUID())
            }
        }
    }
    
    /// Generates a cryptographically secure random salt
    /// - Parameter length: Length of the salt (default: 32 bytes)
    /// - Returns: Random salt data
    func generateSalt(length: Int = 32) async -> Data {
        return await generateBytes(length)
    }
    
    /// Generates a cryptographically secure random initialization vector (IV)
    /// - Parameter length: Length of the IV (default: 16 bytes for AES)
    /// - Returns: Random IV data
    func generateIV(length: Int = 16) async -> Data {
        return await generateBytes(length)
    }
    
    /// Generates a cryptographically secure random nonce
    /// - Parameter length: Length of the nonce (default: 12 bytes for AES-GCM)
    /// - Returns: Random nonce data
    func generateNonce(length: Int = 12) async -> Data {
        return await generateBytes(length)
    }
    
    /// Generates a cryptographically secure random symmetric key
    /// - Parameter size: Key size (default: 256 bits)
    /// - Returns: Random symmetric key
    func generateSymmetricKey(size: SymmetricKeySize = .bits256) async -> SymmetricKey {
        return SymmetricKey(size: size)
    }
    
    /// Generates a cryptographically secure random password
    /// - Parameters:
    ///   - length: Password length
    ///   - includeUppercase: Include uppercase letters
    ///   - includeLowercase: Include lowercase letters
    ///   - includeNumbers: Include numbers
    ///   - includeSymbols: Include symbols
    /// - Returns: Random password
    func generatePassword(
        length: Int,
        includeUppercase: Bool = true,
        includeLowercase: Bool = true,
        includeNumbers: Bool = true,
        includeSymbols: Bool = false
    ) async -> String {
        var charset = ""
        
        if includeUppercase {
            charset += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        }
        if includeLowercase {
            charset += "abcdefghijklmnopqrstuvwxyz"
        }
        if includeNumbers {
            charset += "0123456789"
        }
        if includeSymbols {
            charset += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        }
        
        guard !charset.isEmpty else { return "" }
        
        let characters = Array(charset)
        let randomData = await generateBytes(length)
        
        return String(randomData.map { byte in
            characters[Int(byte) % characters.count]
        })
    }
    
    /// Generates a cryptographically secure random token
    /// - Parameter length: Token length in bytes (default: 32)
    /// - Returns: Random token as hex string
    func generateToken(length: Int = 32) async -> String {
        let randomData = await generateBytes(length)
        return randomData.map { String(format: "%02x", $0) }.joined()
    }
    
    /// Generates a cryptographically secure random API key
    /// - Parameter length: API key length (default: 32 characters)
    /// - Returns: Random API key
    func generateAPIKey(length: Int = 32) async -> String {
        let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        let characters = Array(charset)
        let randomData = await generateBytes(length)
        
        return String(randomData.map { byte in
            characters[Int(byte) % characters.count]
        })
    }
    
    // MARK: - Entropy Testing
    
    /// Tests the entropy of generated random data
    /// - Parameter sampleSize: Size of the sample to test
    /// - Returns: Entropy score (0.0 to 1.0, higher is better)
    func testEntropy(sampleSize: Int = 1024) async -> Double {
        let randomData = await generateBytes(sampleSize)
        return calculateShannonEntropy(data: randomData)
    }
    
    private func calculateShannonEntropy(data: Data) -> Double {
        var frequency = [UInt8: Int]()
        
        // Count frequency of each byte value
        for byte in data {
            frequency[byte, default: 0] += 1
        }
        
        let dataLength = Double(data.count)
        var entropy: Double = 0.0
        
        // Calculate Shannon entropy
        for count in frequency.values {
            let probability = Double(count) / dataLength
            if probability > 0 {
                entropy -= probability * log2(probability)
            }
        }
        
        // Normalize to 0-1 range (max entropy for byte data is 8 bits)
        return entropy / 8.0
    }
    
    // MARK: - Secure Comparison
    
    /// Performs constant-time comparison of two data objects
    /// - Parameters:
    ///   - lhs: First data object
    ///   - rhs: Second data object
    /// - Returns: True if data objects are equal
    func constantTimeCompare(_ lhs: Data, _ rhs: Data) -> Bool {
        guard lhs.count == rhs.count else { return false }
        
        var result: UInt8 = 0
        for i in 0..<lhs.count {
            result |= lhs[i] ^ rhs[i]
        }
        
        return result == 0
    }
    
    /// Securely wipes data from memory
    /// - Parameter data: Data to wipe
    func secureWipe(_ data: inout Data) {
        data.withUnsafeMutableBytes { bytes in
            memset_s(bytes.baseAddress, bytes.count, 0, bytes.count)
        }
    }
}

// MARK: - CharacterSet Extension
extension CharacterSet {
    var characters: [Character] {
        var chars: [Character] = []
        for plane in 0...16 where self.hasMember(inPlane: UInt8(plane)) {
            for unicode in UInt32(plane) << 16..<UInt32(plane + 1) << 16 {
                if let uniChar = UnicodeScalar(unicode), self.contains(uniChar) {
                    chars.append(Character(uniChar))
                }
            }
        }
        return chars
    }
    
    static var alphanumerics: CharacterSet {
        return CharacterSet.alphanumerics
    }
    
    static var letters: CharacterSet {
        return CharacterSet.letters
    }
    
    static var decimalDigits: CharacterSet {
        return CharacterSet.decimalDigits
    }
}

// MARK: - Random Number Quality Metrics
struct RandomQualityMetrics {
    let entropy: Double
    let uniformity: Double
    let independence: Double
    let timestamp: Date
    
    var overallQuality: Double {
        return (entropy + uniformity + independence) / 3.0
    }
    
    var isHighQuality: Bool {
        return overallQuality > 0.8
    }
}

// MARK: - Random Generator Configuration
struct RandomGeneratorConfig {
    let enableEntropyTesting: Bool
    let minimumEntropyThreshold: Double
    let enableSecureWipe: Bool
    let enableConstantTimeOperations: Bool
    
    static let `default` = RandomGeneratorConfig(
        enableEntropyTesting: true,
        minimumEntropyThreshold: 0.7,
        enableSecureWipe: true,
        enableConstantTimeOperations: true
    )
    
    static let highSecurity = RandomGeneratorConfig(
        enableEntropyTesting: true,
        minimumEntropyThreshold: 0.9,
        enableSecureWipe: true,
        enableConstantTimeOperations: true
    )
}