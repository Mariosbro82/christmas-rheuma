//
//  BodyMappingModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import SceneKit
import ARKit
import CoreML
import Vision

// MARK: - Body Mapping Models

struct BodyRegion: Identifiable, Codable {
    let id = UUID()
    let name: String
    let anatomicalName: String
    let category: BodyCategory
    let position: SIMD3<Float>
    let boundingBox: BoundingBox
    var painLevel: PainLevel
    var symptoms: [String]
    var lastUpdated: Date
    var notes: String
    
    enum BodyCategory: String, CaseIterable, Codable {
        case head, neck, torso, arms, hands, legs, feet, joints
        
        var displayName: String {
            switch self {
            case .head: return "Head"
            case .neck: return "Neck"
            case .torso: return "Torso"
            case .arms: return "Arms"
            case .hands: return "Hands"
            case .legs: return "Legs"
            case .feet: return "Feet"
            case .joints: return "Joints"
            }
        }
        
        var color: Color {
            switch self {
            case .head: return .purple
            case .neck: return .blue
            case .torso: return .green
            case .arms: return .orange
            case .hands: return .red
            case .legs: return .yellow
            case .feet: return .pink
            case .joints: return .brown
            }
        }
    }
}

struct BoundingBox: Codable {
    let min: SIMD3<Float>
    let max: SIMD3<Float>
    
    func contains(point: SIMD3<Float>) -> Bool {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z
    }
}

enum PainLevel: Int, CaseIterable, Codable {
    case none = 0
    case mild = 1
    case moderate = 2
    case severe = 3
    case extreme = 4
    
    var displayName: String {
        switch self {
        case .none: return "No Pain"
        case .mild: return "Mild"
        case .moderate: return "Moderate"
        case .severe: return "Severe"
        case .extreme: return "Extreme"
        }
    }
    
    var color: Color {
        switch self {
        case .none: return .green
        case .mild: return .yellow
        case .moderate: return .orange
        case .severe: return .red
        case .extreme: return .purple
        }
    }
    
    var intensity: Float {
        return Float(rawValue) / 4.0
    }
}

struct PainEntry: Identifiable, Codable {
    let id = UUID()
    let regionId: UUID
    let painLevel: PainLevel
    let symptoms: [String]
    let triggers: [String]
    let timestamp: Date
    let duration: TimeInterval?
    let notes: String
    let weatherConditions: WeatherConditions?
    let medicationTaken: [String]
    let activityLevel: ActivityLevel
    
    enum ActivityLevel: String, CaseIterable, Codable {
        case resting, light, moderate, intense
        
        var displayName: String {
            switch self {
            case .resting: return "Resting"
            case .light: return "Light Activity"
            case .moderate: return "Moderate Activity"
            case .intense: return "Intense Activity"
            }
        }
    }
}

struct WeatherConditions: Codable {
    let temperature: Double
    let humidity: Double
    let pressure: Double
    let conditions: String
}

struct BodyScan: Identifiable, Codable {
    let id = UUID()
    let timestamp: Date
    let scanType: ScanType
    let regions: [BodyRegion]
    let overallPainScore: Float
    let notes: String
    let arData: ARScanData?
    
    enum ScanType: String, CaseIterable, Codable {
        case manual, ar, ai
        
        var displayName: String {
            switch self {
            case .manual: return "Manual Entry"
            case .ar: return "AR Scan"
            case .ai: return "AI Analysis"
            }
        }
    }
}

struct ARScanData: Codable {
    let bodyAnchorData: Data
    let meshData: Data
    let postureAnalysis: PostureAnalysis
    let movementPatterns: [MovementPattern]
}

struct PostureAnalysis: Codable {
    let spinalAlignment: Float
    let shoulderLevel: Float
    let hipAlignment: Float
    let headPosition: Float
    let overallScore: Float
    let recommendations: [String]
}

struct MovementPattern: Codable {
    let jointName: String
    let rangeOfMotion: Float
    let smoothness: Float
    let compensation: Bool
    let recommendations: [String]
}

struct BodyMappingInsight: Identifiable, Codable {
    let id = UUID()
    let title: String
    let description: String
    let category: InsightCategory
    let severity: InsightSeverity
    let affectedRegions: [UUID]
    let recommendations: [String]
    let confidence: Float
    let timestamp: Date
    
    enum InsightCategory: String, CaseIterable, Codable {
        case pattern, correlation, prediction, anomaly
        
        var displayName: String {
            switch self {
            case .pattern: return "Pattern Detection"
            case .correlation: return "Correlation Analysis"
            case .prediction: return "Predictive Insight"
            case .anomaly: return "Anomaly Detection"
            }
        }
    }
    
    enum InsightSeverity: String, CaseIterable, Codable {
        case low, medium, high, critical
        
        var color: Color {
            switch self {
            case .low: return .green
            case .medium: return .yellow
            case .high: return .orange
            case .critical: return .red
            }
        }
    }
}

// MARK: - Body Mapping Manager

@MainActor
class BodyMappingManager: ObservableObject {
    @Published var bodyRegions: [BodyRegion] = []
    @Published var painEntries: [PainEntry] = []
    @Published var bodyScans: [BodyScan] = []
    @Published var insights: [BodyMappingInsight] = []
    @Published var selectedRegion: BodyRegion?
    @Published var isARSessionActive = false
    @Published var currentScan: BodyScan?
    @Published var scanProgress: Float = 0.0
    @Published var isAnalyzing = false
    
    // AR and 3D components
    private var arSession: ARSession?
    private var bodyTracker: ARBodyTrackingConfiguration?
    private var sceneRenderer: SCNRenderer?
    private var mlModel: VNCoreMLModel?
    
    // Data managers
    private let dataManager = BodyMappingDataManager()
    private let arManager = ARBodyMappingManager()
    private let aiAnalyzer = BodyMappingAIAnalyzer()
    private let postureAnalyzer = PostureAnalyzer()
    
    init() {
        setupBodyRegions()
        loadData()
        setupMLModel()
    }
    
    // MARK: - Setup Methods
    
    private func setupBodyRegions() {
        bodyRegions = BodyMappingConstants.defaultBodyRegions
    }
    
    private func setupMLModel() {
        Task {
            do {
                if let modelURL = Bundle.main.url(forResource: "BodyMappingModel", withExtension: "mlmodelc") {
                    let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
                    self.mlModel = model
                }
            } catch {
                print("Failed to load ML model: \(error)")
            }
        }
    }
    
    // MARK: - Data Management
    
    func loadData() {
        Task {
            do {
                bodyRegions = try await dataManager.loadBodyRegions()
                painEntries = try await dataManager.loadPainEntries()
                bodyScans = try await dataManager.loadBodyScans()
                insights = try await dataManager.loadInsights()
            } catch {
                print("Failed to load data: \(error)")
            }
        }
    }
    
    func saveData() {
        Task {
            do {
                try await dataManager.saveBodyRegions(bodyRegions)
                try await dataManager.savePainEntries(painEntries)
                try await dataManager.saveBodyScans(bodyScans)
                try await dataManager.saveInsights(insights)
            } catch {
                print("Failed to save data: \(error)")
            }
        }
    }
    
    // MARK: - Pain Entry Methods
    
    func addPainEntry(for regionId: UUID, painLevel: PainLevel, symptoms: [String], notes: String) {
        let entry = PainEntry(
            regionId: regionId,
            painLevel: painLevel,
            symptoms: symptoms,
            triggers: [],
            timestamp: Date(),
            duration: nil,
            notes: notes,
            weatherConditions: nil,
            medicationTaken: [],
            activityLevel: .resting
        )
        
        painEntries.append(entry)
        updateRegionPainLevel(regionId: regionId, painLevel: painLevel)
        saveData()
        
        // Trigger AI analysis
        Task {
            await analyzePatterns()
        }
    }
    
    func updateRegionPainLevel(regionId: UUID, painLevel: PainLevel) {
        if let index = bodyRegions.firstIndex(where: { $0.id == regionId }) {
            bodyRegions[index].painLevel = painLevel
            bodyRegions[index].lastUpdated = Date()
        }
    }
    
    // MARK: - AR Scanning Methods
    
    func startARSession() {
        guard ARBodyTrackingConfiguration.isSupported else {
            print("AR Body Tracking not supported")
            return
        }
        
        isARSessionActive = true
        arManager.startSession { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let scanData):
                    self?.processARScanData(scanData)
                case .failure(let error):
                    print("AR scan failed: \(error)")
                    self?.isARSessionActive = false
                }
            }
        }
    }
    
    func stopARSession() {
        isARSessionActive = false
        arManager.stopSession()
    }
    
    private func processARScanData(_ scanData: ARScanData) {
        Task {
            isAnalyzing = true
            
            do {
                // Analyze posture
                let postureAnalysis = try await postureAnalyzer.analyzePosture(from: scanData)
                
                // Create body scan
                let scan = BodyScan(
                    timestamp: Date(),
                    scanType: .ar,
                    regions: bodyRegions,
                    overallPainScore: calculateOverallPainScore(),
                    notes: "AR scan with posture analysis",
                    arData: scanData
                )
                
                bodyScans.append(scan)
                currentScan = scan
                
                // Generate insights
                let newInsights = try await aiAnalyzer.generateInsights(from: scan)
                insights.append(contentsOf: newInsights)
                
                saveData()
            } catch {
                print("Failed to process AR scan: \(error)")
            }
            
            isAnalyzing = false
        }
    }
    
    // MARK: - AI Analysis Methods
    
    func analyzePatterns() async {
        do {
            let newInsights = try await aiAnalyzer.analyzePatterns(
                regions: bodyRegions,
                entries: painEntries,
                scans: bodyScans
            )
            
            await MainActor.run {
                insights.append(contentsOf: newInsights)
                saveData()
            }
        } catch {
            print("Pattern analysis failed: \(error)")
        }
    }
    
    func predictFlareUp() async -> [BodyMappingInsight] {
        do {
            return try await aiAnalyzer.predictFlareUp(
                regions: bodyRegions,
                entries: painEntries,
                historicalData: bodyScans
            )
        } catch {
            print("Flare-up prediction failed: \(error)")
            return []
        }
    }
    
    // MARK: - Utility Methods
    
    func calculateOverallPainScore() -> Float {
        let totalPain = bodyRegions.reduce(0) { $0 + $1.painLevel.intensity }
        return totalPain / Float(bodyRegions.count)
    }
    
    func getRegionHistory(for regionId: UUID) -> [PainEntry] {
        return painEntries.filter { $0.regionId == regionId }
            .sorted { $0.timestamp > $1.timestamp }
    }
    
    func getRecentInsights(limit: Int = 5) -> [BodyMappingInsight] {
        return insights.sorted { $0.timestamp > $1.timestamp }
            .prefix(limit)
            .map { $0 }
    }
    
    // MARK: - Export Methods
    
    func exportBodyMappingData() async throws -> Data {
        let exportData = BodyMappingExport(
            regions: bodyRegions,
            entries: painEntries,
            scans: bodyScans,
            insights: insights,
            exportDate: Date()
        )
        
        return try JSONEncoder().encode(exportData)
    }
    
    func importBodyMappingData(_ data: Data) async throws {
        let importData = try JSONDecoder().decode(BodyMappingExport.self, from: data)
        
        await MainActor.run {
            bodyRegions = importData.regions
            painEntries = importData.entries
            bodyScans = importData.scans
            insights = importData.insights
            saveData()
        }
    }
}

struct BodyMappingExport: Codable {
    let regions: [BodyRegion]
    let entries: [PainEntry]
    let scans: [BodyScan]
    let insights: [BodyMappingInsight]
    let exportDate: Date
}

// MARK: - Supporting Classes

class BodyMappingDataManager {
    private let userDefaults = UserDefaults.standard
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    
    func loadBodyRegions() async throws -> [BodyRegion] {
        guard let data = userDefaults.data(forKey: "bodyRegions") else {
            return BodyMappingConstants.defaultBodyRegions
        }
        return try decoder.decode([BodyRegion].self, from: data)
    }
    
    func saveBodyRegions(_ regions: [BodyRegion]) async throws {
        let data = try encoder.encode(regions)
        userDefaults.set(data, forKey: "bodyRegions")
    }
    
    func loadPainEntries() async throws -> [PainEntry] {
        guard let data = userDefaults.data(forKey: "painEntries") else {
            return []
        }
        return try decoder.decode([PainEntry].self, from: data)
    }
    
    func savePainEntries(_ entries: [PainEntry]) async throws {
        let data = try encoder.encode(entries)
        userDefaults.set(data, forKey: "painEntries")
    }
    
    func loadBodyScans() async throws -> [BodyScan] {
        guard let data = userDefaults.data(forKey: "bodyScans") else {
            return []
        }
        return try decoder.decode([BodyScan].self, from: data)
    }
    
    func saveBodyScans(_ scans: [BodyScan]) async throws {
        let data = try encoder.encode(scans)
        userDefaults.set(data, forKey: "bodyScans")
    }
    
    func loadInsights() async throws -> [BodyMappingInsight] {
        guard let data = userDefaults.data(forKey: "bodyMappingInsights") else {
            return []
        }
        return try decoder.decode([BodyMappingInsight].self, from: data)
    }
    
    func saveInsights(_ insights: [BodyMappingInsight]) async throws {
        let data = try encoder.encode(insights)
        userDefaults.set(data, forKey: "bodyMappingInsights")
    }
}

class ARBodyMappingManager {
    private var arSession: ARSession?
    private var completion: ((Result<ARScanData, Error>) -> Void)?
    
    func startSession(completion: @escaping (Result<ARScanData, Error>) -> Void) {
        self.completion = completion
        
        let configuration = ARBodyTrackingConfiguration()
        arSession = ARSession()
        arSession?.run(configuration)
        
        // Simulate AR scanning process
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            let mockScanData = ARScanData(
                bodyAnchorData: Data(),
                meshData: Data(),
                postureAnalysis: PostureAnalysis(
                    spinalAlignment: 0.8,
                    shoulderLevel: 0.9,
                    hipAlignment: 0.85,
                    headPosition: 0.7,
                    overallScore: 0.8,
                    recommendations: ["Improve head posture", "Strengthen core muscles"]
                ),
                movementPatterns: []
            )
            
            completion(.success(mockScanData))
        }
    }
    
    func stopSession() {
        arSession?.pause()
        arSession = nil
        completion = nil
    }
}

class BodyMappingAIAnalyzer {
    func analyzePatterns(regions: [BodyRegion], entries: [PainEntry], scans: [BodyScan]) async throws -> [BodyMappingInsight] {
        // Simulate AI pattern analysis
        await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        var insights: [BodyMappingInsight] = []
        
        // Pattern detection
        if let pattern = detectPainPattern(entries: entries) {
            insights.append(pattern)
        }
        
        // Correlation analysis
        if let correlation = analyzeCorrelations(regions: regions, entries: entries) {
            insights.append(correlation)
        }
        
        return insights
    }
    
    func generateInsights(from scan: BodyScan) async throws -> [BodyMappingInsight] {
        // Simulate insight generation from scan
        await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        
        return [
            BodyMappingInsight(
                title: "Posture Analysis Complete",
                description: "AR scan reveals potential posture issues affecting pain levels.",
                category: .pattern,
                severity: .medium,
                affectedRegions: scan.regions.filter { $0.painLevel.rawValue > 1 }.map { $0.id },
                recommendations: ["Consider ergonomic adjustments", "Practice posture exercises"],
                confidence: 0.85,
                timestamp: Date()
            )
        ]
    }
    
    func predictFlareUp(regions: [BodyRegion], entries: [PainEntry], historicalData: [BodyScan]) async throws -> [BodyMappingInsight] {
        // Simulate flare-up prediction
        await Task.sleep(nanoseconds: 1_500_000_000) // 1.5 seconds
        
        return [
            BodyMappingInsight(
                title: "Potential Flare-up Detected",
                description: "Based on current pain patterns, a flare-up may occur in the next 2-3 days.",
                category: .prediction,
                severity: .high,
                affectedRegions: regions.filter { $0.painLevel.rawValue > 2 }.map { $0.id },
                recommendations: ["Increase rest periods", "Consider preventive medication", "Monitor symptoms closely"],
                confidence: 0.78,
                timestamp: Date()
            )
        ]
    }
    
    private func detectPainPattern(entries: [PainEntry]) -> BodyMappingInsight? {
        // Simple pattern detection logic
        let recentEntries = entries.filter { $0.timestamp > Date().addingTimeInterval(-7 * 24 * 60 * 60) }
        
        if recentEntries.count > 5 {
            return BodyMappingInsight(
                title: "Increased Pain Activity",
                description: "Pain entries have increased significantly in the past week.",
                category: .pattern,
                severity: .medium,
                affectedRegions: Array(Set(recentEntries.map { $0.regionId })),
                recommendations: ["Review recent activities", "Consider stress management"],
                confidence: 0.7,
                timestamp: Date()
            )
        }
        
        return nil
    }
    
    private func analyzeCorrelations(regions: [BodyRegion], entries: [PainEntry]) -> BodyMappingInsight? {
        // Simple correlation analysis
        let painfulRegions = regions.filter { $0.painLevel.rawValue > 1 }
        
        if painfulRegions.count > 3 {
            return BodyMappingInsight(
                title: "Multiple Region Involvement",
                description: "Pain is affecting multiple body regions simultaneously.",
                category: .correlation,
                severity: .high,
                affectedRegions: painfulRegions.map { $0.id },
                recommendations: ["Consider systemic treatment approach", "Consult healthcare provider"],
                confidence: 0.8,
                timestamp: Date()
            )
        }
        
        return nil
    }
}

class PostureAnalyzer {
    func analyzePosture(from scanData: ARScanData) async throws -> PostureAnalysis {
        // Simulate posture analysis
        await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        
        return PostureAnalysis(
            spinalAlignment: Float.random(in: 0.6...0.9),
            shoulderLevel: Float.random(in: 0.7...0.95),
            hipAlignment: Float.random(in: 0.75...0.9),
            headPosition: Float.random(in: 0.6...0.85),
            overallScore: Float.random(in: 0.65...0.88),
            recommendations: [
                "Strengthen core muscles",
                "Improve neck posture",
                "Consider ergonomic workspace setup"
            ]
        )
    }
}

// MARK: - Constants

struct BodyMappingConstants {
    static let defaultBodyRegions: [BodyRegion] = [
        // Head and Neck
        BodyRegion(
            name: "Head",
            anatomicalName: "Cranium",
            category: .head,
            position: SIMD3<Float>(0, 1.7, 0),
            boundingBox: BoundingBox(min: SIMD3<Float>(-0.1, 1.6, -0.1), max: SIMD3<Float>(0.1, 1.8, 0.1)),
            painLevel: .none,
            symptoms: [],
            lastUpdated: Date(),
            notes: ""
        ),
        BodyRegion(
            name: "Neck",
            anatomicalName: "Cervical Spine",
            category: .neck,
            position: SIMD3<Float>(0, 1.5, 0),
            boundingBox: BoundingBox(min: SIMD3<Float>(-0.08, 1.45, -0.08), max: SIMD3<Float>(0.08, 1.55, 0.08)),
            painLevel: .none,
            symptoms: [],
            lastUpdated: Date(),
            notes: ""
        ),
        // Torso
        BodyRegion(
            name: "Upper Back",
            anatomicalName: "Thoracic Spine",
            category: .torso,
            position: SIMD3<Float>(0, 1.3, -0.1),
            boundingBox: BoundingBox(min: SIMD3<Float>(-0.15, 1.1, -0.15), max: SIMD3<Float>(0.15, 1.5, 0.05)),
            painLevel: .none,
            symptoms: [],
            lastUpdated: Date(),
            notes: ""
        ),
        BodyRegion(
            name: "Lower Back",
            anatomicalName: "Lumbar Spine",
            category: .torso,
            position: SIMD3<Float>(0, 1.0, -0.1),
            boundingBox: BoundingBox(min: SIMD3<Float>(-0.12, 0.9, -0.12), max: SIMD3<Float>(0.12, 1.1, 0.05)),
            painLevel: .none,
            symptoms: [],
            lastUpdated: Date(),
            notes: ""
        ),
        // Arms
        BodyRegion(
            name: "Left Shoulder",
            anatomicalName: "Left Glenohumeral Joint",
            category: .joints,
            position: SIMD3<Float>(-0.2, 1.4, 0),
            boundingBox: BoundingBox(min: SIMD3<Float>(-0.25, 1.35, -0.05), max: SIMD3<Float>(-0.15, 1.45, 0.05)),
            painLevel: .none,
            symptoms: [],
            lastUpdated: Date(),
            notes: ""
        ),
        BodyRegion(
            name: "Right Shoulder",
            anatomicalName: "Right Glenohumeral Joint",
            category: .joints,
            position: SIMD3<Float>(0.2, 1.4, 0),
            boundingBox: BoundingBox(min: SIMD3<Float>(0.15, 1.35, -0.05), max: SIMD3<Float>(0.25, 1.45, 0.05)),
            painLevel: .none,
            symptoms: [],
            lastUpdated: Date(),
            notes: ""
        ),
        // Add more regions as needed...
    ]
}

// MARK: - Error Types

enum BodyMappingError: Error, LocalizedError {
    case arNotSupported
    case scanFailed(String)
    case analysisError(String)
    case dataCorrupted
    case mlModelNotFound
    
    var errorDescription: String? {
        switch self {
        case .arNotSupported:
            return "AR Body Tracking is not supported on this device"
        case .scanFailed(let message):
            return "Scan failed: \(message)"
        case .analysisError(let message):
            return "Analysis error: \(message)"
        case .dataCorrupted:
            return "Body mapping data is corrupted"
        case .mlModelNotFound:
            return "Machine learning model not found"
        }
    }
}