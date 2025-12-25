//
//  Anky3DView.swift
//  InflamAI
//
//  3D Character View using SceneKit
//  Renders the USDZ 3D model with animations
//

import SwiftUI
import SceneKit

// MARK: - 3D Anky View

struct Anky3DView: View {
    var size: CGFloat = 200
    var state: AnkyState = .idle
    var showShadow: Bool = true
    var onTap: (() -> Void)?

    @State private var scene: SCNScene?
    @State private var cameraNode: SCNNode?
    @State private var characterNode: SCNNode?
    @State private var animationTimer: Timer?
    @State private var bouncePhase: CGFloat = 0
    @State private var rotationY: CGFloat = 0

    var body: some View {
        ZStack {
            // 3D Scene
            SceneView(
                scene: scene,
                pointOfView: cameraNode,
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .frame(width: size, height: size)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .onTapGesture {
                triggerTapAnimation()
                onTap?()
            }

            // Optional shadow underneath
            if showShadow {
                Ellipse()
                    .fill(
                        RadialGradient(
                            colors: [Color.black.opacity(0.2), Color.clear],
                            center: .center,
                            startRadius: 0,
                            endRadius: size * 0.3
                        )
                    )
                    .frame(width: size * 0.5, height: size * 0.12)
                    .offset(y: size * 0.42)
            }
        }
        .frame(width: size, height: size)
        .onAppear {
            setupScene()
            startIdleAnimation()
        }
        .onDisappear {
            animationTimer?.invalidate()
        }
        .onChange(of: state) { newState in
            handleStateChange(newState)
        }
    }

    // MARK: - Scene Setup

    private func setupScene() {
        // Try to load the USDZ model
        if let modelURL = Bundle.main.url(forResource: "AnkyCharacter", withExtension: "usdz") {
            do {
                scene = try SCNScene(url: modelURL, options: [.checkConsistency: true])
                setupCamera()
                setupLighting()
                findCharacterNode()
            } catch {
                print("Failed to load 3D model: \(error)")
                createFallbackScene()
            }
        } else {
            print("USDZ file not found in bundle, creating fallback")
            createFallbackScene()
        }
    }

    private func setupCamera() {
        let camera = SCNCamera()
        camera.usesOrthographicProjection = false
        camera.fieldOfView = 45

        cameraNode = SCNNode()
        cameraNode?.camera = camera
        cameraNode?.position = SCNVector3(0, 0.5, 3)
        cameraNode?.look(at: SCNVector3(0, 0, 0))

        scene?.rootNode.addChildNode(cameraNode!)
    }

    private func setupLighting() {
        // Main light
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.light?.intensity = 1000
        lightNode.light?.color = UIColor.white
        lightNode.position = SCNVector3(2, 3, 2)
        scene?.rootNode.addChildNode(lightNode)

        // Fill light
        let fillLight = SCNNode()
        fillLight.light = SCNLight()
        fillLight.light?.type = .omni
        fillLight.light?.intensity = 500
        fillLight.light?.color = UIColor(red: 0.8, green: 0.9, blue: 1.0, alpha: 1.0)
        fillLight.position = SCNVector3(-2, 1, 2)
        scene?.rootNode.addChildNode(fillLight)

        // Ambient
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 300
        ambientLight.light?.color = UIColor(red: 0.6, green: 0.8, blue: 0.9, alpha: 1.0)
        scene?.rootNode.addChildNode(ambientLight)
    }

    private func findCharacterNode() {
        // Find the main character node in the scene
        characterNode = scene?.rootNode.childNodes.first { node in
            node.geometry != nil || !node.childNodes.isEmpty
        }
    }

    // MARK: - Fallback Scene (if USDZ not available)

    private func createFallbackScene() {
        scene = SCNScene()

        // Create a cute 3D dinosaur shape using primitives
        let bodyNode = createDinoBody()
        scene?.rootNode.addChildNode(bodyNode)
        characterNode = bodyNode

        setupCamera()
        setupLighting()
    }

    private func createDinoBody() -> SCNNode {
        let parentNode = SCNNode()

        // Main body (sphere, slightly squashed)
        let bodySphere = SCNSphere(radius: 0.5)
        bodySphere.firstMaterial?.diffuse.contents = UIColor(red: 0.15, green: 0.78, blue: 0.70, alpha: 1.0)
        bodySphere.firstMaterial?.specular.contents = UIColor.white
        bodySphere.firstMaterial?.shininess = 0.3

        let bodyNode = SCNNode(geometry: bodySphere)
        bodyNode.scale = SCNVector3(1.0, 0.9, 0.8)
        bodyNode.position = SCNVector3(0, 0, 0)
        parentNode.addChildNode(bodyNode)

        // Belly (lighter sphere)
        let bellySphere = SCNSphere(radius: 0.35)
        bellySphere.firstMaterial?.diffuse.contents = UIColor(red: 0.85, green: 0.98, blue: 0.95, alpha: 1.0)

        let bellyNode = SCNNode(geometry: bellySphere)
        bellyNode.position = SCNVector3(0, -0.05, 0.2)
        bellyNode.scale = SCNVector3(0.9, 0.85, 0.6)
        parentNode.addChildNode(bellyNode)

        // Eyes (white spheres with dark pupils)
        for xOffset in [-0.18, 0.18] {
            // Eye white
            let eyeSphere = SCNSphere(radius: 0.12)
            eyeSphere.firstMaterial?.diffuse.contents = UIColor.white

            let eyeNode = SCNNode(geometry: eyeSphere)
            eyeNode.position = SCNVector3(Float(xOffset), 0.15, 0.4)
            parentNode.addChildNode(eyeNode)

            // Pupil
            let pupilSphere = SCNSphere(radius: 0.06)
            pupilSphere.firstMaterial?.diffuse.contents = UIColor(red: 0.1, green: 0.15, blue: 0.2, alpha: 1.0)

            let pupilNode = SCNNode(geometry: pupilSphere)
            pupilNode.position = SCNVector3(Float(xOffset), 0.15, 0.48)
            parentNode.addChildNode(pupilNode)

            // Eye highlight
            let highlightSphere = SCNSphere(radius: 0.025)
            highlightSphere.firstMaterial?.diffuse.contents = UIColor.white
            highlightSphere.firstMaterial?.emission.contents = UIColor.white

            let highlightNode = SCNNode(geometry: highlightSphere)
            highlightNode.position = SCNVector3(Float(xOffset) + 0.03, 0.18, 0.52)
            parentNode.addChildNode(highlightNode)
        }

        // Spikes on top (3 cones)
        let spikePositions: [(x: Float, y: Float, scale: Float)] = [
            (-0.15, 0.45, 0.8),
            (0.0, 0.55, 1.0),
            (0.15, 0.45, 0.8)
        ]

        for spike in spikePositions {
            let spikeCone = SCNCone(topRadius: 0, bottomRadius: 0.08, height: 0.2)
            spikeCone.firstMaterial?.diffuse.contents = UIColor(red: 0.12, green: 0.65, blue: 0.58, alpha: 1.0)

            let spikeNode = SCNNode(geometry: spikeCone)
            spikeNode.position = SCNVector3(spike.x, spike.y, 0)
            spikeNode.scale = SCNVector3(spike.scale, spike.scale, spike.scale)
            parentNode.addChildNode(spikeNode)
        }

        // Legs (2 cylinders)
        for xOffset in [-0.2, 0.2] {
            let legCapsule = SCNCapsule(capRadius: 0.1, height: 0.3)
            legCapsule.firstMaterial?.diffuse.contents = UIColor(red: 0.12, green: 0.65, blue: 0.58, alpha: 1.0)

            let legNode = SCNNode(geometry: legCapsule)
            legNode.position = SCNVector3(Float(xOffset), -0.55, 0.1)
            parentNode.addChildNode(legNode)

            // Foot
            let footSphere = SCNSphere(radius: 0.12)
            footSphere.firstMaterial?.diffuse.contents = UIColor(red: 0.08, green: 0.52, blue: 0.48, alpha: 1.0)

            let footNode = SCNNode(geometry: footSphere)
            footNode.position = SCNVector3(Float(xOffset), -0.7, 0.12)
            footNode.scale = SCNVector3(1.0, 0.5, 1.2)
            parentNode.addChildNode(footNode)
        }

        // Small arms
        for xOffset in [-0.45, 0.45] {
            let armCapsule = SCNCapsule(capRadius: 0.06, height: 0.15)
            armCapsule.firstMaterial?.diffuse.contents = UIColor(red: 0.15, green: 0.78, blue: 0.70, alpha: 1.0)

            let armNode = SCNNode(geometry: armCapsule)
            armNode.position = SCNVector3(Float(xOffset), 0.0, 0.15)
            armNode.eulerAngles = SCNVector3(0, 0, Float(xOffset > 0 ? -0.5 : 0.5))
            parentNode.addChildNode(armNode)
        }

        // Tail
        let tailCapsule = SCNCapsule(capRadius: 0.08, height: 0.3)
        tailCapsule.firstMaterial?.diffuse.contents = UIColor(red: 0.15, green: 0.78, blue: 0.70, alpha: 1.0)

        let tailNode = SCNNode(geometry: tailCapsule)
        tailNode.position = SCNVector3(0.35, -0.1, -0.2)
        tailNode.eulerAngles = SCNVector3(0, 0, Float.pi / 4)
        parentNode.addChildNode(tailNode)

        return parentNode
    }

    // MARK: - Animations

    private func startIdleAnimation() {
        animationTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { _ in
            bouncePhase += 0.05

            let bounce = sin(bouncePhase) * 0.02
            let sway = sin(bouncePhase * 0.5) * 0.03

            characterNode?.position.y = Float(bounce)
            characterNode?.eulerAngles.z = Float(sway)
        }
    }

    private func handleStateChange(_ newState: AnkyState) {
        switch newState {
        case .waving:
            triggerWaveAnimation()
        case .celebrating:
            triggerCelebrationAnimation()
        case .happy:
            triggerHappyAnimation()
        default:
            break
        }
    }

    private func triggerTapAnimation() {
        HapticFeedback.light()

        // Quick scale bounce
        let scaleUp = SCNAction.scale(to: 1.1, duration: 0.15)
        let scaleDown = SCNAction.scale(to: 1.0, duration: 0.15)
        scaleUp.timingMode = .easeOut
        scaleDown.timingMode = .easeIn

        characterNode?.runAction(SCNAction.sequence([scaleUp, scaleDown]))
    }

    private func triggerWaveAnimation() {
        // Rotate slightly and bounce
        let rotateRight = SCNAction.rotateBy(x: 0, y: 0, z: 0.2, duration: 0.2)
        let rotateLeft = SCNAction.rotateBy(x: 0, y: 0, z: -0.4, duration: 0.4)
        let rotateBack = SCNAction.rotateBy(x: 0, y: 0, z: 0.2, duration: 0.2)

        let waveSequence = SCNAction.sequence([rotateRight, rotateLeft, rotateBack])
        let wave3Times = SCNAction.repeat(waveSequence, count: 2)

        characterNode?.runAction(wave3Times)
    }

    private func triggerCelebrationAnimation() {
        // Jump and spin
        let jumpUp = SCNAction.moveBy(x: 0, y: 0.3, z: 0, duration: 0.2)
        let spin = SCNAction.rotateBy(x: 0, y: CGFloat.pi * 2, z: 0, duration: 0.4)
        let jumpDown = SCNAction.moveBy(x: 0, y: -0.3, z: 0, duration: 0.2)

        jumpUp.timingMode = .easeOut
        jumpDown.timingMode = .easeIn

        let jumpAndSpin = SCNAction.group([
            SCNAction.sequence([jumpUp, jumpDown]),
            spin
        ])

        characterNode?.runAction(jumpAndSpin)
    }

    private func triggerHappyAnimation() {
        // Bounce more energetically
        let bounceUp = SCNAction.moveBy(x: 0, y: 0.15, z: 0, duration: 0.15)
        let bounceDown = SCNAction.moveBy(x: 0, y: -0.15, z: 0, duration: 0.15)

        let bounce = SCNAction.sequence([bounceUp, bounceDown])
        let bounceMultiple = SCNAction.repeat(bounce, count: 3)

        characterNode?.runAction(bounceMultiple)
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 30) {
        Anky3DView(size: 200, state: .idle)

        Anky3DView(size: 150, state: .happy)
    }
    .padding()
}
