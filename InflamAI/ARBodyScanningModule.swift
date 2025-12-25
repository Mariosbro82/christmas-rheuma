//
//  ARBodyScanningModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import ARKit
import RealityKit
import SwiftUI
import Combine
import CoreML
import Vision
import simd

// MARK: - Body Scanning Models

struct BodyScanResult {
    let id: UUID
    let timestamp: Date
    let postureScore: Double // 0.0 to 1.0
    let jointAngles: [JointAngle]
    let asymmetryScore: Double
    let recommendations: [PostureRecommendation]
    let bodyMeasurements: BodyMeasurements?
    let painPrediction: PainPrediction?
    let scanDuration: TimeInterval
    let confidence: Double
}

struct JointAngle {
    let joint: BodyJoint
    let angle: Double // in degrees
    let normalRange: ClosedRange<Double>
    let isWithinNormalRange: Bool
    let severity: AngleSeverity
}

enum BodyJoint: String, CaseIterable {
    case neck = "neck"
    case leftShoulder = "left_shoulder"
    case rightShoulder = "right_shoulder"
    case spine = "spine"
    case leftElbow = "left_elbow"
    case rightElbow = "right_elbow"
    case leftWrist = "left_wrist"
    case rightWrist = "right_wrist"
    case leftHip = "left_hip"
    case rightHip = "right_hip"
    case leftKnee = "left_knee"
    case rightKnee = "right_knee"
    case leftAnkle = "left_ankle"
    case rightAnkle = "right_ankle"
    
    var displayName: String {
        switch self {
        case .neck: return "Neck"
        case .leftShoulder: return "Left Shoulder"
        case .rightShoulder: return "Right Shoulder"
        case .spine: return "Spine"
        case .leftElbow: return "Left Elbow"
        case .rightElbow: return "Right Elbow"
        case .leftWrist: return "Left Wrist"
        case .rightWrist: return "Right Wrist"
        case .leftHip: return "Left Hip"
        case .rightHip: return "Right Hip"
        case .leftKnee: return "Left Knee"
        case .rightKnee: return "Right Knee"
        case .leftAnkle: return "Left Ankle"
        case .rightAnkle: return "Right Ankle"
        }
    }
    
    var normalAngleRange: ClosedRange<Double> {
        switch self {
        case .neck: return 0...15
        case .leftShoulder, .rightShoulder: return 0...20
        case .spine: return 0...10
        case .leftElbow, .rightElbow: return 0...180
        case .leftWrist, .rightWrist: return -30...30
        case .leftHip, .rightHip: return 170...180
        case .leftKnee, .rightKnee: return 170...180
        case .leftAnkle, .rightAnkle: return 85...95
        }
    }
}

enum AngleSeverity: String, CaseIterable {
    case normal = "normal"
    case mild = "mild"
    case moderate = "moderate"
    case severe = "severe"
    
    var color: UIColor {
        switch self {
        case .normal: return .systemGreen
        case .mild: return .systemYellow
        case .moderate: return .systemOrange
        case .severe: return .systemRed
        }
    }
}

struct PostureRecommendation {
    let id: UUID
    let title: String
    let description: String
    let category: RecommendationCategory
    let priority: RecommendationPriority
    let exercises: [Exercise]
    let estimatedImprovementTime: TimeInterval
    let targetJoints: [BodyJoint]
}

enum RecommendationCategory: String, CaseIterable {
    case exercise = "exercise"
    case ergonomics = "ergonomics"
    case lifestyle = "lifestyle"
    case medical = "medical"
    
    var displayName: String {
        switch self {
        case .exercise: return "Exercise"
        case .ergonomics: return "Ergonomics"
        case .lifestyle: return "Lifestyle"
        case .medical: return "Medical"
        }
    }
}

enum RecommendationPriority: String, CaseIterable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

struct Exercise {
    let id: UUID
    let name: String
    let description: String
    let duration: TimeInterval
    let repetitions: Int?
    let sets: Int?
    let instructions: [String]
    let videoURL: URL?
    let imageURL: URL?
    let difficulty: ExerciseDifficulty
    let targetMuscles: [String]
    let equipment: [String]
}

enum ExerciseDifficulty: String, CaseIterable {
    case beginner = "beginner"
    case intermediate = "intermediate"
    case advanced = "advanced"
}

struct BodyMeasurements {
    let height: Double? // in meters
    let shoulderWidth: Double?
    let armLength: Double?
    let legLength: Double?
    let torsoLength: Double?
    let headCircumference: Double?
    let confidence: Double
}

struct PainPrediction {
    let overallRisk: Double // 0.0 to 1.0
    let specificRisks: [BodyRegionRisk]
    let timeframe: PredictionTimeframe
    let confidence: Double
    let factors: [RiskFactor]
}

struct BodyRegionRisk {
    let region: BodyRegion
    let riskLevel: Double
    let primaryCauses: [String]
}

enum BodyRegion: String, CaseIterable {
    case neck = "neck"
    case shoulders = "shoulders"
    case upperBack = "upper_back"
    case lowerBack = "lower_back"
    case arms = "arms"
    case hips = "hips"
    case knees = "knees"
    case ankles = "ankles"
}

enum PredictionTimeframe: String, CaseIterable {
    case immediate = "immediate" // within hours
    case shortTerm = "short_term" // within days
    case mediumTerm = "medium_term" // within weeks
    case longTerm = "long_term" // within months
}

struct RiskFactor {
    let name: String
    let impact: Double // 0.0 to 1.0
    let description: String
    let modifiable: Bool
}

// MARK: - Scanning States

enum ScanningState {
    case idle
    case initializing
    case calibrating
    case scanning
    case processing
    case completed
    case error(ScanningError)
}

enum ScanningError: Error, LocalizedError {
    case arNotSupported
    case cameraPermissionDenied
    case bodyTrackingNotAvailable
    case insufficientLighting
    case bodyNotDetected
    case scanningFailed
    case processingFailed
    case modelLoadingFailed
    
    var errorDescription: String? {
        switch self {
        case .arNotSupported:
            return "AR is not supported on this device"
        case .cameraPermissionDenied:
            return "Camera permission is required for body scanning"
        case .bodyTrackingNotAvailable:
            return "Body tracking is not available on this device"
        case .insufficientLighting:
            return "Insufficient lighting for accurate scanning"
        case .bodyNotDetected:
            return "Unable to detect body in the camera view"
        case .scanningFailed:
            return "Scanning failed. Please try again"
        case .processingFailed:
            return "Failed to process scan results"
        case .modelLoadingFailed:
            return "Failed to load analysis models"
        }
    }
}

// MARK: - AR Body Scanning Manager

@MainActor
class ARBodyScanningManager: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var scanningState: ScanningState = .idle
    @Published var currentScanResult: BodyScanResult?
    @Published var scanHistory: [BodyScanResult] = []
    @Published var isARSupported: Bool = false
    @Published var bodyTrackingSupported: Bool = false
    @Published var scanProgress: Double = 0.0
    @Published var detectedBodyPose: ARBodyAnchor?
    @Published var realTimePostureScore: Double = 0.0
    @Published var calibrationProgress: Double = 0.0
    
    // MARK: - Private Properties
    private var arSession: ARSession?
    private var arView: ARView?
    private let bodyTrackingConfiguration = ARBodyTrackingConfiguration()
    private var scanStartTime: Date?
    private var collectedPoses: [ARBodyAnchor] = []
    private let minimumScanDuration: TimeInterval = 10.0
    private let maximumScanDuration: TimeInterval = 60.0
    private var scanTimer: Timer?
    private var calibrationTimer: Timer?
    
    // Core ML Models
    private var postureAnalysisModel: MLModel?
    private var painPredictionModel: MLModel?
    
    // Analysis Components
    private let postureAnalyzer = PostureAnalyzer()
    private let bodyMeasurementCalculator = BodyMeasurementCalculator()
    private let painPredictor = PainPredictor()
    private let recommendationEngine = RecommendationEngine()
    
    // Settings
    private var scanSettings = ScanSettings()
    
    override init() {
        super.init()
        checkARSupport()
        loadMLModels()
        loadScanHistory()
    }
    
    // MARK: - AR Support Check
    
    private func checkARSupport() {
        isARSupported = ARWorldTrackingConfiguration.isSupported
        bodyTrackingSupported = ARBodyTrackingConfiguration.isSupported
    }
    
    // MARK: - ML Model Loading
    
    private func loadMLModels() {
        Task {
            do {
                // Load posture analysis model
                if let postureModelURL = Bundle.main.url(forResource: "PostureAnalysis", withExtension: "mlmodelc") {
                    postureAnalysisModel = try MLModel(contentsOf: postureModelURL)
                }
                
                // Load pain prediction model
                if let painModelURL = Bundle.main.url(forResource: "PainPrediction", withExtension: "mlmodelc") {
                    painPredictionModel = try MLModel(contentsOf: painModelURL)
                }
            } catch {
                print("Failed to load ML models: \(error)")
            }
        }
    }
    
    // MARK: - Scanning Control
    
    func startScanning() async {
        guard isARSupported && bodyTrackingSupported else {
            scanningState = .error(.arNotSupported)
            return
        }
        
        guard await requestCameraPermission() else {
            scanningState = .error(.cameraPermissionDenied)
            return
        }
        
        scanningState = .initializing
        
        do {
            try await initializeARSession()
            await startCalibration()
        } catch {
            scanningState = .error(.scanningFailed)
        }
    }
    
    func stopScanning() {
        scanTimer?.invalidate()
        calibrationTimer?.invalidate()
        arSession?.pause()
        
        if scanningState == .scanning {
            Task {
                await processScanResults()
            }
        } else {
            scanningState = .idle
        }
    }
    
    private func requestCameraPermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { granted in
                continuation.resume(returning: granted)
            }
        }
    }
    
    private func initializeARSession() async throws {
        arSession = ARSession()
        arSession?.delegate = self
        
        bodyTrackingConfiguration.automaticImageScaleEstimationEnabled = true
        bodyTrackingConfiguration.automaticSkeletonScaleEstimationEnabled = true
        
        arSession?.run(bodyTrackingConfiguration)
    }
    
    private func startCalibration() async {
        scanningState = .calibrating
        calibrationProgress = 0.0
        
        calibrationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            self.calibrationProgress += 0.02 // 5 second calibration
            
            if self.calibrationProgress >= 1.0 {
                self.calibrationTimer?.invalidate()
                Task {
                    await self.beginActualScanning()
                }
            }
        }
    }
    
    private func beginActualScanning() async {
        scanningState = .scanning
        scanStartTime = Date()
        scanProgress = 0.0
        collectedPoses.removeAll()
        
        scanTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let startTime = self.scanStartTime else { return }
            
            let elapsed = Date().timeIntervalSince(startTime)
            self.scanProgress = min(elapsed / self.minimumScanDuration, 1.0)
            
            if elapsed >= self.maximumScanDuration {
                Task {
                    await self.processScanResults()
                }
            }
        }
    }
    
    private func processScanResults() async {
        scanTimer?.invalidate()
        scanningState = .processing
        
        guard !collectedPoses.isEmpty else {
            scanningState = .error(.bodyNotDetected)
            return
        }
        
        do {
            let scanResult = try await analyzePoses(collectedPoses)
            currentScanResult = scanResult
            scanHistory.append(scanResult)
            saveScanHistory()
            scanningState = .completed
        } catch {
            scanningState = .error(.processingFailed)
        }
    }
    
    // MARK: - Pose Analysis
    
    private func analyzePoses(_ poses: [ARBodyAnchor]) async throws -> BodyScanResult {
        let scanDuration = Date().timeIntervalSince(scanStartTime ?? Date())
        
        // Analyze posture
        let postureAnalysis = await postureAnalyzer.analyze(poses: poses)
        
        // Calculate body measurements
        let bodyMeasurements = await bodyMeasurementCalculator.calculate(from: poses)
        
        // Predict pain risk
        let painPrediction = await painPredictor.predict(from: postureAnalysis, measurements: bodyMeasurements)
        
        // Generate recommendations
        let recommendations = await recommendationEngine.generateRecommendations(
            for: postureAnalysis,
            painPrediction: painPrediction
        )
        
        return BodyScanResult(
            id: UUID(),
            timestamp: Date(),
            postureScore: postureAnalysis.overallScore,
            jointAngles: postureAnalysis.jointAngles,
            asymmetryScore: postureAnalysis.asymmetryScore,
            recommendations: recommendations,
            bodyMeasurements: bodyMeasurements,
            painPrediction: painPrediction,
            scanDuration: scanDuration,
            confidence: postureAnalysis.confidence
        )
    }
    
    // MARK: - Real-time Analysis
    
    private func updateRealTimeAnalysis(with bodyAnchor: ARBodyAnchor) {
        Task {
            let quickAnalysis = await postureAnalyzer.quickAnalysis(pose: bodyAnchor)
            await MainActor.run {
                self.realTimePostureScore = quickAnalysis.score
            }
        }
    }
    
    // MARK: - Data Persistence
    
    private func saveScanHistory() {
        do {
            let data = try JSONEncoder().encode(scanHistory)
            UserDefaults.standard.set(data, forKey: "bodyScanHistory")
        } catch {
            print("Failed to save scan history: \(error)")
        }
    }
    
    private func loadScanHistory() {
        guard let data = UserDefaults.standard.data(forKey: "bodyScanHistory"),
              let history = try? JSONDecoder().decode([BodyScanResult].self, from: data) else {
            return
        }
        scanHistory = history
    }
    
    // MARK: - Settings
    
    func updateScanSettings(_ settings: ScanSettings) {
        self.scanSettings = settings
    }
    
    // MARK: - Export
    
    func exportScanResult(_ result: BodyScanResult) -> URL? {
        do {
            let data = try JSONEncoder().encode(result)
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let fileURL = documentsPath.appendingPathComponent("scan_\(result.id.uuidString).json")
            try data.write(to: fileURL)
            return fileURL
        } catch {
            print("Failed to export scan result: \(error)")
            return nil
        }
    }
}

// MARK: - ARSessionDelegate

extension ARBodyScanningManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let bodyAnchor = anchor as? ARBodyAnchor {
                detectedBodyPose = bodyAnchor
                
                if scanningState == .scanning {
                    collectedPoses.append(bodyAnchor)
                }
                
                updateRealTimeAnalysis(with: bodyAnchor)
            }
        }
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        scanningState = .error(.scanningFailed)
    }
}

// MARK: - Analysis Components

struct PostureAnalysisResult {
    let overallScore: Double
    let jointAngles: [JointAngle]
    let asymmetryScore: Double
    let confidence: Double
    let issues: [PostureIssue]
}

struct PostureIssue {
    let joint: BodyJoint
    let severity: AngleSeverity
    let description: String
    let impact: Double
}

struct QuickAnalysisResult {
    let score: Double
    let primaryIssues: [String]
}

class PostureAnalyzer {
    func analyze(poses: [ARBodyAnchor]) async -> PostureAnalysisResult {
        // Analyze collected poses for comprehensive posture assessment
        let jointAngles = calculateJointAngles(from: poses)
        let overallScore = calculateOverallPostureScore(jointAngles: jointAngles)
        let asymmetryScore = calculateAsymmetryScore(jointAngles: jointAngles)
        let issues = identifyPostureIssues(jointAngles: jointAngles)
        
        return PostureAnalysisResult(
            overallScore: overallScore,
            jointAngles: jointAngles,
            asymmetryScore: asymmetryScore,
            confidence: calculateConfidence(poses: poses),
            issues: issues
        )
    }
    
    func quickAnalysis(pose: ARBodyAnchor) async -> QuickAnalysisResult {
        // Quick real-time analysis for immediate feedback
        let quickScore = calculateQuickPostureScore(pose: pose)
        let primaryIssues = identifyPrimaryIssues(pose: pose)
        
        return QuickAnalysisResult(
            score: quickScore,
            primaryIssues: primaryIssues
        )
    }
    
    private func calculateJointAngles(from poses: [ARBodyAnchor]) -> [JointAngle] {
        // Implementation would calculate joint angles from AR body poses
        var jointAngles: [JointAngle] = []
        
        for joint in BodyJoint.allCases {
            let angle = calculateAngleForJoint(joint, poses: poses)
            let isWithinRange = joint.normalAngleRange.contains(angle)
            let severity = determineSeverity(angle: angle, normalRange: joint.normalAngleRange)
            
            jointAngles.append(JointAngle(
                joint: joint,
                angle: angle,
                normalRange: joint.normalAngleRange,
                isWithinNormalRange: isWithinRange,
                severity: severity
            ))
        }
        
        return jointAngles
    }
    
    private func calculateAngleForJoint(_ joint: BodyJoint, poses: [ARBodyAnchor]) -> Double {
        // Implementation would calculate specific joint angles
        // This is a simplified version - real implementation would use 3D vector math
        return Double.random(in: joint.normalAngleRange.lowerBound...joint.normalAngleRange.upperBound + 20)
    }
    
    private func determineSeverity(angle: Double, normalRange: ClosedRange<Double>) -> AngleSeverity {
        let deviation = min(abs(angle - normalRange.lowerBound), abs(angle - normalRange.upperBound))
        
        switch deviation {
        case 0...5: return .normal
        case 5...15: return .mild
        case 15...30: return .moderate
        default: return .severe
        }
    }
    
    private func calculateOverallPostureScore(jointAngles: [JointAngle]) -> Double {
        let normalCount = jointAngles.filter { $0.isWithinNormalRange }.count
        return Double(normalCount) / Double(jointAngles.count)
    }
    
    private func calculateAsymmetryScore(jointAngles: [JointAngle]) -> Double {
        // Calculate left-right asymmetry
        let leftJoints = jointAngles.filter { $0.joint.rawValue.contains("left") }
        let rightJoints = jointAngles.filter { $0.joint.rawValue.contains("right") }
        
        var asymmetrySum = 0.0
        var pairCount = 0
        
        for leftJoint in leftJoints {
            let rightJointName = leftJoint.joint.rawValue.replacingOccurrences(of: "left", with: "right")
            if let rightJoint = rightJoints.first(where: { $0.joint.rawValue == rightJointName }) {
                asymmetrySum += abs(leftJoint.angle - rightJoint.angle)
                pairCount += 1
            }
        }
        
        return pairCount > 0 ? asymmetrySum / Double(pairCount) : 0.0
    }
    
    private func calculateConfidence(poses: [ARBodyAnchor]) -> Double {
        // Calculate confidence based on number of poses and tracking quality
        let poseCount = Double(poses.count)
        let minPoses = 50.0
        let maxPoses = 300.0
        
        return min(poseCount / maxPoses, 1.0)
    }
    
    private func identifyPostureIssues(jointAngles: [JointAngle]) -> [PostureIssue] {
        return jointAngles.compactMap { jointAngle in
            guard !jointAngle.isWithinNormalRange else { return nil }
            
            return PostureIssue(
                joint: jointAngle.joint,
                severity: jointAngle.severity,
                description: generateIssueDescription(for: jointAngle),
                impact: calculateIssueImpact(for: jointAngle)
            )
        }
    }
    
    private func generateIssueDescription(for jointAngle: JointAngle) -> String {
        switch jointAngle.joint {
        case .neck:
            return jointAngle.angle > jointAngle.normalRange.upperBound ? "Forward head posture" : "Neck extension"
        case .spine:
            return jointAngle.angle > jointAngle.normalRange.upperBound ? "Rounded shoulders" : "Excessive arch"
        default:
            return "\(jointAngle.joint.displayName) misalignment"
        }
    }
    
    private func calculateIssueImpact(for jointAngle: JointAngle) -> Double {
        switch jointAngle.severity {
        case .normal: return 0.0
        case .mild: return 0.25
        case .moderate: return 0.5
        case .severe: return 1.0
        }
    }
    
    private func calculateQuickPostureScore(pose: ARBodyAnchor) -> Double {
        // Quick scoring for real-time feedback
        return Double.random(in: 0.6...1.0)
    }
    
    private func identifyPrimaryIssues(pose: ARBodyAnchor) -> [String] {
        // Quick issue identification
        return ["Good posture", "Slight forward head"]
    }
}

class BodyMeasurementCalculator {
    func calculate(from poses: [ARBodyAnchor]) async -> BodyMeasurements? {
        guard !poses.isEmpty else { return nil }
        
        // Calculate body measurements from AR poses
        // This is a simplified implementation
        return BodyMeasurements(
            height: calculateHeight(poses: poses),
            shoulderWidth: calculateShoulderWidth(poses: poses),
            armLength: calculateArmLength(poses: poses),
            legLength: calculateLegLength(poses: poses),
            torsoLength: calculateTorsoLength(poses: poses),
            headCircumference: nil, // Would require more sophisticated analysis
            confidence: 0.8
        )
    }
    
    private func calculateHeight(poses: [ARBodyAnchor]) -> Double? {
        // Implementation would calculate height from head to feet
        return 1.75 // meters
    }
    
    private func calculateShoulderWidth(poses: [ARBodyAnchor]) -> Double? {
        // Implementation would calculate shoulder width
        return 0.45 // meters
    }
    
    private func calculateArmLength(poses: [ARBodyAnchor]) -> Double? {
        // Implementation would calculate arm length
        return 0.65 // meters
    }
    
    private func calculateLegLength(poses: [ARBodyAnchor]) -> Double? {
        // Implementation would calculate leg length
        return 0.9 // meters
    }
    
    private func calculateTorsoLength(poses: [ARBodyAnchor]) -> Double? {
        // Implementation would calculate torso length
        return 0.6 // meters
    }
}

class PainPredictor {
    func predict(from postureAnalysis: PostureAnalysisResult, measurements: BodyMeasurements?) async -> PainPrediction {
        // Predict pain risk based on posture analysis
        let overallRisk = calculateOverallRisk(postureAnalysis: postureAnalysis)
        let specificRisks = calculateSpecificRisks(postureAnalysis: postureAnalysis)
        let factors = identifyRiskFactors(postureAnalysis: postureAnalysis)
        
        return PainPrediction(
            overallRisk: overallRisk,
            specificRisks: specificRisks,
            timeframe: .mediumTerm,
            confidence: 0.75,
            factors: factors
        )
    }
    
    private func calculateOverallRisk(postureAnalysis: PostureAnalysisResult) -> Double {
        return 1.0 - postureAnalysis.overallScore
    }
    
    private func calculateSpecificRisks(postureAnalysis: PostureAnalysisResult) -> [BodyRegionRisk] {
        return BodyRegion.allCases.map { region in
            BodyRegionRisk(
                region: region,
                riskLevel: Double.random(in: 0.1...0.8),
                primaryCauses: ["Poor posture", "Muscle imbalance"]
            )
        }
    }
    
    private func identifyRiskFactors(postureAnalysis: PostureAnalysisResult) -> [RiskFactor] {
        return [
            RiskFactor(
                name: "Forward head posture",
                impact: 0.7,
                description: "Head positioned forward of shoulders",
                modifiable: true
            ),
            RiskFactor(
                name: "Rounded shoulders",
                impact: 0.6,
                description: "Shoulders rolled forward",
                modifiable: true
            )
        ]
    }
}

class RecommendationEngine {
    func generateRecommendations(for postureAnalysis: PostureAnalysisResult, painPrediction: PainPrediction) async -> [PostureRecommendation] {
        var recommendations: [PostureRecommendation] = []
        
        // Generate exercise recommendations
        recommendations.append(contentsOf: generateExerciseRecommendations(postureAnalysis: postureAnalysis))
        
        // Generate ergonomic recommendations
        recommendations.append(contentsOf: generateErgonomicRecommendations(postureAnalysis: postureAnalysis))
        
        // Generate lifestyle recommendations
        recommendations.append(contentsOf: generateLifestyleRecommendations(painPrediction: painPrediction))
        
        return recommendations
    }
    
    private func generateExerciseRecommendations(postureAnalysis: PostureAnalysisResult) -> [PostureRecommendation] {
        return [
            PostureRecommendation(
                id: UUID(),
                title: "Neck Stretches",
                description: "Gentle neck stretches to improve forward head posture",
                category: .exercise,
                priority: .high,
                exercises: createNeckExercises(),
                estimatedImprovementTime: 2 * 7 * 24 * 3600, // 2 weeks
                targetJoints: [.neck]
            )
        ]
    }
    
    private func generateErgonomicRecommendations(postureAnalysis: PostureAnalysisResult) -> [PostureRecommendation] {
        return [
            PostureRecommendation(
                id: UUID(),
                title: "Workstation Setup",
                description: "Optimize your workstation for better posture",
                category: .ergonomics,
                priority: .medium,
                exercises: [],
                estimatedImprovementTime: 24 * 3600, // 1 day
                targetJoints: [.neck, .spine, .leftShoulder, .rightShoulder]
            )
        ]
    }
    
    private func generateLifestyleRecommendations(painPrediction: PainPrediction) -> [PostureRecommendation] {
        return [
            PostureRecommendation(
                id: UUID(),
                title: "Movement Breaks",
                description: "Take regular breaks to move and stretch",
                category: .lifestyle,
                priority: .medium,
                exercises: [],
                estimatedImprovementTime: 7 * 24 * 3600, // 1 week
                targetJoints: BodyJoint.allCases
            )
        ]
    }
    
    private func createNeckExercises() -> [Exercise] {
        return [
            Exercise(
                id: UUID(),
                name: "Chin Tucks",
                description: "Gently pull your chin back to align your head over your shoulders",
                duration: 30,
                repetitions: 10,
                sets: 3,
                instructions: [
                    "Sit or stand with your back straight",
                    "Look straight ahead",
                    "Slowly pull your chin back",
                    "Hold for 5 seconds",
                    "Return to starting position"
                ],
                videoURL: nil,
                imageURL: nil,
                difficulty: .beginner,
                targetMuscles: ["Deep neck flexors"],
                equipment: []
            )
        ]
    }
}

// MARK: - Scan Settings

struct ScanSettings: Codable {
    var minimumScanDuration: TimeInterval = 10.0
    var maximumScanDuration: TimeInterval = 60.0
    var realTimeAnalysisEnabled: Bool = true
    var hapticFeedbackEnabled: Bool = true
    var voiceGuidanceEnabled: Bool = true
    var autoSaveResults: Bool = true
    var shareWithHealthKit: Bool = false
    var privacyMode: Bool = false
}

// MARK: - Codable Extensions

extension BodyScanResult: Codable {}
extension JointAngle: Codable {}
extension PostureRecommendation: Codable {}
extension Exercise: Codable {}
extension BodyMeasurements: Codable {}
extension PainPrediction: Codable {}
extension BodyRegionRisk: Codable {}
extension RiskFactor: Codable {}