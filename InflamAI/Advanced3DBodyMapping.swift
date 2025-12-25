//
//  Advanced3DBodyMapping.swift
//  InflamAI-Swift
//
//  Advanced 3D interactive body mapping for pain visualization and analysis
//

import Foundation
import SwiftUI
import SceneKit
import ARKit
import CoreData
import Combine

// MARK: - 3D Body Mapping Models
struct BodyRegion {
    let id: UUID
    let name: String
    let anatomicalName: String
    let position: SCNVector3
    let meshNodes: [String]
    let parentRegion: UUID?
    let childRegions: [UUID]
    let associatedJoints: [String]
    let nervePaths: [String]
}

struct PainMapping {
    let id: UUID
    let bodyRegion: BodyRegion
    let intensity: Double // 0-10
    let painType: PainType
    let timestamp: Date
    let duration: TimeInterval?
    let triggers: [String]
    let symptoms: [String]
    let position: SCNVector3
    let radiationPattern: RadiationPattern?
}

enum PainType: String, CaseIterable {
    case sharp = "Sharp"
    case dull = "Dull"
    case burning = "Burning"
    case throbbing = "Throbbing"
    case stabbing = "Stabbing"
    case cramping = "Cramping"
    case tingling = "Tingling"
    case numbness = "Numbness"
    case stiffness = "Stiffness"
    case swelling = "Swelling"
}

struct RadiationPattern {
    let originPoint: SCNVector3
    let radiationPoints: [SCNVector3]
    let intensity: Double
    let pattern: RadiationType
}

enum RadiationType: String, CaseIterable {
    case linear = "Linear"
    case radial = "Radial"
    case diffuse = "Diffuse"
    case nerve = "Nerve Path"
}

// MARK: - 3D Visualization Components
struct Advanced3DBodyView: UIViewRepresentable {
    @ObservedObject var bodyMappingManager: BodyMappingManager
    @Binding var selectedRegion: BodyRegion?
    @Binding var isARMode: Bool
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = bodyMappingManager.scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor.systemBackground
        
        // Add gesture recognizers
        let tapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        scnView.addGestureRecognizer(tapGesture)
        
        let longPressGesture = UILongPressGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleLongPress(_:)))
        scnView.addGestureRecognizer(longPressGesture)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        bodyMappingManager.updateVisualization()
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject {
        var parent: Advanced3DBodyView
        
        init(_ parent: Advanced3DBodyView) {
            self.parent = parent
        }
        
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let scnView = gesture.view as? SCNView else { return }
            
            let location = gesture.location(in: scnView)
            let hitResults = scnView.hitTest(location, options: [:])
            
            if let hitResult = hitResults.first {
                parent.bodyMappingManager.handleBodyRegionSelection(hitResult.node)
            }
        }
        
        @objc func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
            guard gesture.state == .began,
                  let scnView = gesture.view as? SCNView else { return }
            
            let location = gesture.location(in: scnView)
            let hitResults = scnView.hitTest(location, options: [:])
            
            if let hitResult = hitResults.first {
                parent.bodyMappingManager.handlePainMapping(at: hitResult.worldCoordinates, node: hitResult.node)
            }
        }
    }
}

// MARK: - AR Body Scanning View
struct ARBodyScanView: UIViewRepresentable {
    @ObservedObject var bodyMappingManager: BodyMappingManager
    @Binding var isScanning: Bool
    
    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView()
        arView.delegate = context.coordinator
        arView.session.delegate = context.coordinator
        
        // Configure AR session for body tracking
        let configuration = ARBodyTrackingConfiguration()
        arView.session.run(configuration)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {
        if isScanning {
            bodyMappingManager.startARBodyScanning(arView: uiView)
        } else {
            bodyMappingManager.stopARBodyScanning()
        }
    }
    
    func makeCoordinator() -> ARCoordinator {
        ARCoordinator(self)
    }
    
    class ARCoordinator: NSObject, ARSCNViewDelegate, ARSessionDelegate {
        var parent: ARBodyScanView
        
        init(_ parent: ARBodyScanView) {
            self.parent = parent
        }
        
        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard let bodyAnchor = anchor as? ARBodyAnchor else { return }
            parent.bodyMappingManager.processARBodyAnchor(bodyAnchor, node: node)
        }
        
        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let bodyAnchor = anchor as? ARBodyAnchor else { return }
            parent.bodyMappingManager.updateARBodyAnchor(bodyAnchor, node: node)
        }
    }
}

// MARK: - Body Mapping Manager
class BodyMappingManager: ObservableObject {
    let scene = SCNScene()
    private let context: NSManagedObjectContext
    
    @Published var bodyRegions: [BodyRegion] = []
    @Published var painMappings: [PainMapping] = []
    @Published var selectedRegion: BodyRegion?
    @Published var isARMode = false
    @Published var postureAnalysis: PostureAnalysis?
    
    // 3D Model components
    private var bodyModel: SCNNode?
    private var painNodes: [UUID: SCNNode] = [:]
    private var heatmapOverlay: SCNNode?
    private var animationController: BodyAnimationController?
    
    // AR components
    private var arBodyAnchor: ARBodyAnchor?
    private var arBodyNode: SCNNode?
    
    init(context: NSManagedObjectContext) {
        self.context = context
        setupScene()
        loadBodyModel()
        setupBodyRegions()
        animationController = BodyAnimationController(scene: scene)
    }
    
    // MARK: - Scene Setup
    
    private func setupScene() {
        // Configure lighting
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.intensity = 300
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        scene.rootNode.addChildNode(ambientNode)
        
        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.intensity = 1000
        directionalLight.castsShadow = true
        let lightNode = SCNNode()
        lightNode.light = directionalLight
        lightNode.position = SCNVector3(0, 10, 10)
        lightNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(lightNode)
        
        // Configure camera
        let camera = SCNCamera()
        camera.fieldOfView = 60
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 5)
        scene.rootNode.addChildNode(cameraNode)
    }
    
    private func loadBodyModel() {
        // Load 3D body model (in production, this would load a detailed anatomical model)
        guard let modelScene = SCNScene(named: "BodyModel.scn") else {
            // Create a simplified body model if file doesn't exist
            createSimplifiedBodyModel()
            return
        }
        
        bodyModel = modelScene.rootNode.childNodes.first
        if let bodyModel = bodyModel {
            scene.rootNode.addChildNode(bodyModel)
            setupBodyMaterials()
        }
    }
    
    private func createSimplifiedBodyModel() {
        // Create a simplified 3D body representation
        let bodyGeometry = SCNCapsule(capRadius: 0.3, height: 1.8)
        bodyModel = SCNNode(geometry: bodyGeometry)
        
        // Add basic body parts
        addBodyPart("head", geometry: SCNSphere(radius: 0.15), position: SCNVector3(0, 1.05, 0))
        addBodyPart("torso", geometry: SCNBox(width: 0.6, height: 0.8, length: 0.3, chamferRadius: 0.05), position: SCNVector3(0, 0.3, 0))
        addBodyPart("leftArm", geometry: SCNCapsule(capRadius: 0.08, height: 0.6), position: SCNVector3(-0.4, 0.5, 0))
        addBodyPart("rightArm", geometry: SCNCapsule(capRadius: 0.08, height: 0.6), position: SCNVector3(0.4, 0.5, 0))
        addBodyPart("leftLeg", geometry: SCNCapsule(capRadius: 0.1, height: 0.8), position: SCNVector3(-0.15, -0.5, 0))
        addBodyPart("rightLeg", geometry: SCNCapsule(capRadius: 0.1, height: 0.8), position: SCNVector3(0.15, -0.5, 0))
        
        if let bodyModel = bodyModel {
            scene.rootNode.addChildNode(bodyModel)
        }
    }
    
    private func addBodyPart(_ name: String, geometry: SCNGeometry, position: SCNVector3) {
        let node = SCNNode(geometry: geometry)
        node.position = position
        node.name = name
        
        // Add interactive material
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.systemBlue.withAlphaComponent(0.7)
        material.specular.contents = UIColor.white
        material.shininess = 0.8
        geometry.materials = [material]
        
        bodyModel?.addChildNode(node)
    }
    
    private func setupBodyMaterials() {
        bodyModel?.enumerateChildNodes { node, _ in
            if let geometry = node.geometry {
                let material = SCNMaterial()
                material.diffuse.contents = UIColor.systemBlue.withAlphaComponent(0.7)
                material.specular.contents = UIColor.white
                material.shininess = 0.8
                geometry.materials = [material]
            }
        }
    }
    
    // MARK: - Body Regions Setup
    
    private func setupBodyRegions() {
        bodyRegions = [
            BodyRegion(id: UUID(), name: "Head", anatomicalName: "Cranium", position: SCNVector3(0, 1.05, 0), meshNodes: ["head"], parentRegion: nil, childRegions: [], associatedJoints: ["neck"], nervePaths: ["trigeminal", "facial"]),
            BodyRegion(id: UUID(), name: "Neck", anatomicalName: "Cervical Spine", position: SCNVector3(0, 0.85, 0), meshNodes: ["neck"], parentRegion: nil, childRegions: [], associatedJoints: ["c1-c7"], nervePaths: ["cervical"]),
            BodyRegion(id: UUID(), name: "Upper Back", anatomicalName: "Thoracic Spine", position: SCNVector3(0, 0.5, -0.15), meshNodes: ["upperBack"], parentRegion: nil, childRegions: [], associatedJoints: ["t1-t12"], nervePaths: ["thoracic"]),
            BodyRegion(id: UUID(), name: "Lower Back", anatomicalName: "Lumbar Spine", position: SCNVector3(0, 0.1, -0.15), meshNodes: ["lowerBack"], parentRegion: nil, childRegions: [], associatedJoints: ["l1-l5"], nervePaths: ["lumbar"]),
            BodyRegion(id: UUID(), name: "Left Shoulder", anatomicalName: "Left Glenohumeral Joint", position: SCNVector3(-0.3, 0.7, 0), meshNodes: ["leftShoulder"], parentRegion: nil, childRegions: [], associatedJoints: ["glenohumeral"], nervePaths: ["brachial"]),
            BodyRegion(id: UUID(), name: "Right Shoulder", anatomicalName: "Right Glenohumeral Joint", position: SCNVector3(0.3, 0.7, 0), meshNodes: ["rightShoulder"], parentRegion: nil, childRegions: [], associatedJoints: ["glenohumeral"], nervePaths: ["brachial"]),
            BodyRegion(id: UUID(), name: "Left Knee", anatomicalName: "Left Tibiofemoral Joint", position: SCNVector3(-0.15, -0.3, 0), meshNodes: ["leftKnee"], parentRegion: nil, childRegions: [], associatedJoints: ["tibiofemoral"], nervePaths: ["femoral", "sciatic"]),
            BodyRegion(id: UUID(), name: "Right Knee", anatomicalName: "Right Tibiofemoral Joint", position: SCNVector3(0.15, -0.3, 0), meshNodes: ["rightKnee"], parentRegion: nil, childRegions: [], associatedJoints: ["tibiofemoral"], nervePaths: ["femoral", "sciatic"])
        ]
    }
    
    // MARK: - Pain Mapping
    
    func addPainMapping(_ painMapping: PainMapping) {
        painMappings.append(painMapping)
        createPainVisualization(painMapping)
        updateHeatmap()
        savePainMapping(painMapping)
    }
    
    private func createPainVisualization(_ painMapping: PainMapping) {
        let painNode = SCNNode()
        
        // Create pain visualization based on type and intensity
        let geometry = createPainGeometry(for: painMapping)
        painNode.geometry = geometry
        painNode.position = painMapping.position
        painNode.name = "pain_\(painMapping.id.uuidString)"
        
        // Add pulsing animation for active pain
        if let duration = painMapping.duration, duration > 0 {
            addPulsingAnimation(to: painNode, intensity: painMapping.intensity)
        }
        
        // Add radiation pattern if present
        if let radiation = painMapping.radiationPattern {
            addRadiationVisualization(to: painNode, pattern: radiation)
        }
        
        scene.rootNode.addChildNode(painNode)
        painNodes[painMapping.id] = painNode
    }
    
    private func createPainGeometry(for painMapping: PainMapping) -> SCNGeometry {
        let radius = 0.05 + (painMapping.intensity / 10.0) * 0.1
        let sphere = SCNSphere(radius: radius)
        
        let material = SCNMaterial()
        material.diffuse.contents = getPainColor(for: painMapping.intensity, type: painMapping.painType)
        material.emission.contents = material.diffuse.contents
        material.transparency = 0.8
        
        sphere.materials = [material]
        return sphere
    }
    
    private func getPainColor(for intensity: Double, type: PainType) -> UIColor {
        let alpha = 0.3 + (intensity / 10.0) * 0.7
        
        switch type {
        case .sharp, .stabbing:
            return UIColor.red.withAlphaComponent(alpha)
        case .burning:
            return UIColor.orange.withAlphaComponent(alpha)
        case .throbbing:
            return UIColor.purple.withAlphaComponent(alpha)
        case .dull:
            return UIColor.brown.withAlphaComponent(alpha)
        case .tingling, .numbness:
            return UIColor.blue.withAlphaComponent(alpha)
        case .stiffness:
            return UIColor.gray.withAlphaComponent(alpha)
        case .swelling:
            return UIColor.cyan.withAlphaComponent(alpha)
        case .cramping:
            return UIColor.yellow.withAlphaComponent(alpha)
        }
    }
    
    private func addPulsingAnimation(to node: SCNNode, intensity: Double) {
        let scaleUp = SCNAction.scale(to: 1.0 + intensity / 20.0, duration: 0.5)
        let scaleDown = SCNAction.scale(to: 1.0, duration: 0.5)
        let pulse = SCNAction.sequence([scaleUp, scaleDown])
        let repeatPulse = SCNAction.repeatForever(pulse)
        
        node.runAction(repeatPulse, forKey: "pulse")
    }
    
    private func addRadiationVisualization(to node: SCNNode, pattern: RadiationPattern) {
        for radiationPoint in pattern.radiationPoints {
            let line = createRadiationLine(from: pattern.originPoint, to: radiationPoint, intensity: pattern.intensity)
            node.addChildNode(line)
        }
    }
    
    private func createRadiationLine(from start: SCNVector3, to end: SCNVector3, intensity: Double) -> SCNNode {
        let vector = SCNVector3(end.x - start.x, end.y - start.y, end.z - start.z)
        let distance = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
        
        let cylinder = SCNCylinder(radius: 0.005 * intensity, height: CGFloat(distance))
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.red.withAlphaComponent(0.5)
        cylinder.materials = [material]
        
        let lineNode = SCNNode(geometry: cylinder)
        lineNode.position = SCNVector3((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
        
        // Orient the cylinder to point from start to end
        lineNode.look(at: end, up: SCNVector3(0, 1, 0), localFront: SCNVector3(0, 1, 0))
        
        return lineNode
    }
    
    // MARK: - Heatmap Generation
    
    private func updateHeatmap() {
        // Remove existing heatmap
        heatmapOverlay?.removeFromParentNode()
        
        // Generate new heatmap based on pain data
        let heatmapData = generateHeatmapData()
        heatmapOverlay = createHeatmapOverlay(data: heatmapData)
        
        if let heatmapOverlay = heatmapOverlay {
            scene.rootNode.addChildNode(heatmapOverlay)
        }
    }
    
    private func generateHeatmapData() -> [[Double]] {
        // Create a grid-based heatmap of pain intensity
        let gridSize = 50
        var heatmapData = Array(repeating: Array(repeating: 0.0, count: gridSize), count: gridSize)
        
        for painMapping in painMappings {
            let x = Int((painMapping.position.x + 1.0) * Double(gridSize) / 2.0)
            let y = Int((painMapping.position.y + 1.0) * Double(gridSize) / 2.0)
            
            if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
                heatmapData[x][y] = max(heatmapData[x][y], painMapping.intensity)
                
                // Apply Gaussian blur for smooth heatmap
                applyGaussianBlur(to: &heatmapData, centerX: x, centerY: y, intensity: painMapping.intensity)
            }
        }
        
        return heatmapData
    }
    
    private func applyGaussianBlur(to data: inout [[Double]], centerX: Int, centerY: Int, intensity: Double) {
        let radius = 3
        let sigma = 1.0
        
        for dx in -radius...radius {
            for dy in -radius...radius {
                let x = centerX + dx
                let y = centerY + dy
                
                if x >= 0 && x < data.count && y >= 0 && y < data[0].count {
                    let distance = sqrt(Double(dx * dx + dy * dy))
                    let weight = exp(-(distance * distance) / (2 * sigma * sigma))
                    data[x][y] = max(data[x][y], intensity * weight)
                }
            }
        }
    }
    
    private func createHeatmapOverlay(data: [[Double]]) -> SCNNode {
        // Create a texture from heatmap data
        let heatmapTexture = createHeatmapTexture(from: data)
        
        let plane = SCNPlane(width: 2.0, height: 2.0)
        let material = SCNMaterial()
        material.diffuse.contents = heatmapTexture
        material.transparency = 0.5
        plane.materials = [material]
        
        let heatmapNode = SCNNode(geometry: plane)
        heatmapNode.position = SCNVector3(0, 0, -0.01) // Slightly behind the body
        
        return heatmapNode
    }
    
    private func createHeatmapTexture(from data: [[Double]]) -> UIImage {
        let size = CGSize(width: data.count, height: data[0].count)
        let renderer = UIGraphicsImageRenderer(size: size)
        
        return renderer.image { context in
            for x in 0..<data.count {
                for y in 0..<data[0].count {
                    let intensity = data[x][y]
                    let color = getHeatmapColor(for: intensity)
                    
                    context.cgContext.setFillColor(color.cgColor)
                    context.cgContext.fill(CGRect(x: x, y: y, width: 1, height: 1))
                }
            }
        }
    }
    
    private func getHeatmapColor(for intensity: Double) -> UIColor {
        if intensity == 0 {
            return UIColor.clear
        }
        
        let normalizedIntensity = min(intensity / 10.0, 1.0)
        
        if normalizedIntensity < 0.5 {
            // Blue to green
            let ratio = normalizedIntensity * 2
            return UIColor(red: 0, green: ratio, blue: 1 - ratio, alpha: normalizedIntensity)
        } else {
            // Green to red
            let ratio = (normalizedIntensity - 0.5) * 2
            return UIColor(red: ratio, green: 1 - ratio, blue: 0, alpha: normalizedIntensity)
        }
    }
    
    // MARK: - AR Body Scanning
    
    func startARBodyScanning(arView: ARSCNView) {
        // Configure AR session for body tracking
        let configuration = ARBodyTrackingConfiguration()
        arView.session.run(configuration)
    }
    
    func stopARBodyScanning() {
        // Stop AR session
    }
    
    func processARBodyAnchor(_ bodyAnchor: ARBodyAnchor, node: SCNNode) {
        arBodyAnchor = bodyAnchor
        arBodyNode = node
        
        // Analyze posture from AR body tracking
        analyzePosture(from: bodyAnchor)
    }
    
    func updateARBodyAnchor(_ bodyAnchor: ARBodyAnchor, node: SCNNode) {
        arBodyAnchor = bodyAnchor
        
        // Update posture analysis
        analyzePosture(from: bodyAnchor)
    }
    
    private func analyzePosture(from bodyAnchor: ARBodyAnchor) {
        let skeleton = bodyAnchor.skeleton
        
        // Analyze key joint positions for posture assessment
        let headPosition = skeleton.modelTransform(for: .head)
        let neckPosition = skeleton.modelTransform(for: .neck_1)
        let spinePosition = skeleton.modelTransform(for: .spine_7)
        let hipPosition = skeleton.modelTransform(for: .hips)
        
        // Calculate posture metrics
        let headForwardAngle = calculateHeadForwardAngle(head: headPosition, neck: neckPosition)
        let spinalCurvature = calculateSpinalCurvature(spine: spinePosition, hip: hipPosition)
        let shoulderAlignment = calculateShoulderAlignment(skeleton: skeleton)
        
        let analysis = PostureAnalysis(
            headForwardAngle: headForwardAngle,
            spinalCurvature: spinalCurvature,
            shoulderAlignment: shoulderAlignment,
            overallScore: calculateOverallPostureScore(headForwardAngle, spinalCurvature, shoulderAlignment),
            recommendations: generatePostureRecommendations(headForwardAngle, spinalCurvature, shoulderAlignment),
            timestamp: Date()
        )
        
        DispatchQueue.main.async {
            self.postureAnalysis = analysis
        }
    }
    
    // MARK: - Interaction Handlers
    
    func handleBodyRegionSelection(_ node: SCNNode) {
        guard let nodeName = node.name else { return }
        
        // Find corresponding body region
        selectedRegion = bodyRegions.first { region in
            region.meshNodes.contains(nodeName)
        }
        
        // Highlight selected region
        highlightBodyRegion(node)
    }
    
    func handlePainMapping(at position: SCNVector3, node: SCNNode) {
        // Create new pain mapping at the touched location
        let bodyRegion = findBodyRegion(for: node)
        
        // This would typically open a pain input dialog
        // For now, create a sample pain mapping
        let painMapping = PainMapping(
            id: UUID(),
            bodyRegion: bodyRegion ?? bodyRegions[0],
            intensity: 5.0,
            painType: .dull,
            timestamp: Date(),
            duration: nil,
            triggers: [],
            symptoms: [],
            position: position,
            radiationPattern: nil
        )
        
        addPainMapping(painMapping)
    }
    
    private func findBodyRegion(for node: SCNNode) -> BodyRegion? {
        guard let nodeName = node.name else { return nil }
        
        return bodyRegions.first { region in
            region.meshNodes.contains(nodeName)
        }
    }
    
    private func highlightBodyRegion(_ node: SCNNode) {
        // Remove previous highlights
        scene.rootNode.enumerateChildNodes { childNode, _ in
            if childNode.name?.hasPrefix("highlight_") == true {
                childNode.removeFromParentNode()
            }
        }
        
        // Add highlight to selected region
        if let geometry = node.geometry?.copy() as? SCNGeometry {
            let highlightMaterial = SCNMaterial()
            highlightMaterial.diffuse.contents = UIColor.yellow.withAlphaComponent(0.5)
            highlightMaterial.emission.contents = UIColor.yellow
            geometry.materials = [highlightMaterial]
            
            let highlightNode = SCNNode(geometry: geometry)
            highlightNode.position = node.position
            highlightNode.name = "highlight_\(node.name ?? "unknown")"
            
            scene.rootNode.addChildNode(highlightNode)
        }
    }
    
    func updateVisualization() {
        // Update all visualizations
        updateHeatmap()
        
        // Update pain node animations
        for (id, node) in painNodes {
            if let painMapping = painMappings.first(where: { $0.id == id }) {
                updatePainNodeVisualization(node, for: painMapping)
            }
        }
    }
    
    private func updatePainNodeVisualization(_ node: SCNNode, for painMapping: PainMapping) {
        // Update color and size based on current pain data
        if let geometry = node.geometry as? SCNSphere {
            let radius = 0.05 + (painMapping.intensity / 10.0) * 0.1
            geometry.radius = radius
            
            if let material = geometry.materials.first {
                material.diffuse.contents = getPainColor(for: painMapping.intensity, type: painMapping.painType)
                material.emission.contents = material.diffuse.contents
            }
        }
    }
    
    // MARK: - Data Persistence
    
    private func savePainMapping(_ painMapping: PainMapping) {
        // Save to Core Data
        let painEntry = PainEntry(context: context)
        painEntry.id = painMapping.id
        painEntry.painLevel = Int16(painMapping.intensity)
        painEntry.timestamp = painMapping.timestamp
        painEntry.bodyRegion = painMapping.bodyRegion.name
        painEntry.painType = painMapping.painType.rawValue
        
        do {
            try context.save()
        } catch {
            print("Failed to save pain mapping: \(error)")
        }
    }
}

// MARK: - Supporting Structures

struct PostureAnalysis {
    let headForwardAngle: Double
    let spinalCurvature: Double
    let shoulderAlignment: Double
    let overallScore: Double
    let recommendations: [String]
    let timestamp: Date
}

class BodyAnimationController {
    private let scene: SCNScene
    
    init(scene: SCNScene) {
        self.scene = scene
    }
    
    func animateBreathing() {
        // Add breathing animation to torso
    }
    
    func animateHeartbeat() {
        // Add heartbeat visualization
    }
    
    func animateBloodFlow() {
        // Add blood flow visualization
    }
}

// MARK: - Helper Functions
extension BodyMappingManager {
    private func calculateHeadForwardAngle(head: simd_float4x4, neck: simd_float4x4) -> Double {
        // Calculate the angle of head forward posture
        let headPosition = simd_float3(head.columns.3.x, head.columns.3.y, head.columns.3.z)
        let neckPosition = simd_float3(neck.columns.3.x, neck.columns.3.y, neck.columns.3.z)
        
        let vector = headPosition - neckPosition
        let angle = atan2(vector.z, vector.y) * 180 / .pi
        
        return Double(angle)
    }
    
    private func calculateSpinalCurvature(spine: simd_float4x4, hip: simd_float4x4) -> Double {
        // Calculate spinal curvature
        let spinePosition = simd_float3(spine.columns.3.x, spine.columns.3.y, spine.columns.3.z)
        let hipPosition = simd_float3(hip.columns.3.x, hip.columns.3.y, hip.columns.3.z)
        
        let vector = spinePosition - hipPosition
        let curvature = atan2(vector.x, vector.y) * 180 / .pi
        
        return Double(curvature)
    }
    
    private func calculateShoulderAlignment(skeleton: ARSkeleton3D) -> Double {
        // Calculate shoulder alignment
        let leftShoulder = skeleton.modelTransform(for: .leftShoulder)
        let rightShoulder = skeleton.modelTransform(for: .rightShoulder)
        
        let leftPos = simd_float3(leftShoulder.columns.3.x, leftShoulder.columns.3.y, leftShoulder.columns.3.z)
        let rightPos = simd_float3(rightShoulder.columns.3.x, rightShoulder.columns.3.y, rightShoulder.columns.3.z)
        
        let heightDifference = abs(leftPos.y - rightPos.y)
        
        return Double(heightDifference)
    }
    
    private func calculateOverallPostureScore(_ headAngle: Double, _ spinalCurvature: Double, _ shoulderAlignment: Double) -> Double {
        // Calculate overall posture score (0-100)
        let headScore = max(0, 100 - abs(headAngle) * 2)
        let spineScore = max(0, 100 - abs(spinalCurvature) * 3)
        let shoulderScore = max(0, 100 - shoulderAlignment * 100)
        
        return (headScore + spineScore + shoulderScore) / 3
    }
    
    private func generatePostureRecommendations(_ headAngle: Double, _ spinalCurvature: Double, _ shoulderAlignment: Double) -> [String] {
        var recommendations: [String] = []
        
        if abs(headAngle) > 15 {
            recommendations.append("Reduce forward head posture by adjusting your workspace ergonomics")
        }
        
        if abs(spinalCurvature) > 10 {
            recommendations.append("Improve spinal alignment with core strengthening exercises")
        }
        
        if shoulderAlignment > 0.05 {
            recommendations.append("Work on shoulder alignment with targeted stretching")
        }
        
        if recommendations.isEmpty {
            recommendations.append("Great posture! Keep up the good work")
        }
        
        return recommendations
    }
}