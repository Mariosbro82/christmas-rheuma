//
//  ARBodyScanningView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import ARKit
import RealityKit
import Vision

struct ARBodyScanningView: View {
    @StateObject private var arManager = ARBodyScanningManager()
    @StateObject private var aiEngine = AIMLEngine.shared
    
    @Binding var selectedRegions: Set<BodyRegion>
    @Binding var painIntensity: [BodyRegion: Double]
    
    @State private var isScanning = false
    @State private var scanProgress: Double = 0.0
    @State private var detectedBodyParts: [DetectedBodyPart] = []
    @State private var showingCalibration = false
    @State private var scanningMode: ScanningMode = .bodyMapping
    @State private var showingResults = false
    @State private var hapticFeedbackEnabled = true
    @State private var voiceGuidanceEnabled = true
    
    private enum ScanningMode: String, CaseIterable {
        case bodyMapping = "Body Mapping"
        case painLocalization = "Pain Localization"
        case postureAnalysis = "Posture Analysis"
        case movementTracking = "Movement Tracking"
        
        var icon: String {
            switch self {
            case .bodyMapping: return "figure.stand"
            case .painLocalization: return "target"
            case .postureAnalysis: return "figure.walk"
            case .movementTracking: return "figure.run"
            }
        }
        
        var description: String {
            switch self {
            case .bodyMapping: return "Map your body structure for precise pain tracking"
            case .painLocalization: return "Point to specific areas of pain for accurate logging"
            case .postureAnalysis: return "Analyze your posture for pain correlation insights"
            case .movementTracking: return "Track movement patterns and pain triggers"
            }
        }
    }
    
    var body: some View {
        ZStack {
            // AR Camera View
            ARViewContainer(arManager: arManager)
                .edgesIgnoringSafeArea(.all)
            
            // Overlay UI
            VStack {
                // Top Controls
                HStack {
                    // Mode Selector
                    Menu {
                        ForEach(ScanningMode.allCases, id: \.self) { mode in
                            Button(action: { scanningMode = mode }) {
                                Label(mode.rawValue, systemImage: mode.icon)
                            }
                        }
                    } label: {
                        HStack {
                            Image(systemName: scanningMode.icon)
                            Text(scanningMode.rawValue)
                                .font(.caption)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 20)
                                .fill(.ultraThinMaterial)
                        )
                    }
                    
                    Spacer()
                    
                    // Settings Button
                    Button(action: { showingCalibration = true }) {
                        Image(systemName: "gearshape")
                            .font(.title3)
                            .foregroundColor(.white)
                            .padding(12)
                            .background(
                                Circle()
                                    .fill(.ultraThinMaterial)
                            )
                    }
                }
                .padding()
                
                Spacer()
                
                // Center Crosshair and Instructions
                VStack(spacing: 20) {
                    // Crosshair for targeting
                    if scanningMode == .painLocalization {
                        CrosshairView()
                            .frame(width: 50, height: 50)
                    }
                    
                    // Instructions
                    InstructionBubble(mode: scanningMode, isScanning: isScanning)
                }
                
                Spacer()
                
                // Bottom Controls
                VStack(spacing: 20) {
                    // Scan Progress
                    if isScanning {
                        ScanProgressView(progress: scanProgress, mode: scanningMode)
                    }
                    
                    // Detected Body Parts
                    if !detectedBodyParts.isEmpty {
                        DetectedPartsOverlay(parts: detectedBodyParts)
                    }
                    
                    // Control Buttons
                    HStack(spacing: 30) {
                        // Cancel/Back Button
                        Button(action: {
                            if isScanning {
                                stopScanning()
                            } else {
                                // Navigate back
                            }
                        }) {
                            Image(systemName: isScanning ? "xmark" : "chevron.left")
                                .font(.title2)
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(
                                    Circle()
                                        .fill(.ultraThinMaterial)
                                )
                        }
                        
                        // Main Scan Button
                        Button(action: toggleScanning) {
                            ZStack {
                                Circle()
                                    .fill(
                                        LinearGradient(
                                            colors: isScanning ? [.red, .pink] : [.blue, .purple],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .frame(width: 80, height: 80)
                                    .scaleEffect(isScanning ? 1.1 : 1.0)
                                    .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: isScanning)
                                
                                Image(systemName: isScanning ? "stop.fill" : "viewfinder")
                                    .font(.title)
                                    .foregroundColor(.white)
                            }
                        }
                        .disabled(!arManager.isSessionReady)
                        
                        // Results Button
                        Button(action: { showingResults = true }) {
                            Image(systemName: "doc.text")
                                .font(.title2)
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(
                                    Circle()
                                        .fill(.ultraThinMaterial)
                                )
                        }
                        .disabled(detectedBodyParts.isEmpty)
                    }
                    .padding(.bottom, 40)
                }
            }
            
            // AR Anchors and Overlays
            ForEach(detectedBodyParts, id: \.id) { part in
                ARBodyPartOverlay(part: part, screenSize: UIScreen.main.bounds.size)
            }
        }
        .onAppear {
            setupARSession()
        }
        .onDisappear {
            arManager.stopSession()
        }
        .sheet(isPresented: $showingCalibration) {
            ARCalibrationView(arManager: arManager)
        }
        .sheet(isPresented: $showingResults) {
            ARScanResultsView(
                detectedParts: detectedBodyParts,
                selectedRegions: $selectedRegions,
                painIntensity: $painIntensity
            )
        }
        .onChange(of: scanningMode) { mode in
            arManager.setScanningMode(mode)
        }
    }
    
    private func setupARSession() {
        arManager.startSession()
        arManager.onBodyPartsDetected = { parts in
            detectedBodyParts = parts
            
            if hapticFeedbackEnabled {
                let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                impactFeedback.impactOccurred()
            }
        }
        
        arManager.onScanProgressUpdated = { progress in
            scanProgress = progress
        }
    }
    
    private func toggleScanning() {
        if isScanning {
            stopScanning()
        } else {
            startScanning()
        }
    }
    
    private func startScanning() {
        isScanning = true
        scanProgress = 0.0
        detectedBodyParts.removeAll()
        
        arManager.startBodyScanning(mode: scanningMode)
        
        if hapticFeedbackEnabled {
            let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
            impactFeedback.impactOccurred()
        }
        
        if voiceGuidanceEnabled {
            provideVoiceGuidance(for: scanningMode)
        }
    }
    
    private func stopScanning() {
        isScanning = false
        arManager.stopBodyScanning()
        
        if hapticFeedbackEnabled {
            let impactFeedback = UIImpactFeedbackGenerator(style: .light)
            impactFeedback.impactOccurred()
        }
    }
    
    private func provideVoiceGuidance(for mode: ScanningMode) {
        let synthesizer = AVSpeechSynthesizer()
        let utterance: AVSpeechUtterance
        
        switch mode {
        case .bodyMapping:
            utterance = AVSpeechUtterance(string: "Hold your device steady and slowly move around your body to map all areas")
        case .painLocalization:
            utterance = AVSpeechUtterance(string: "Point the crosshair at the specific area where you feel pain")
        case .postureAnalysis:
            utterance = AVSpeechUtterance(string: "Stand naturally and hold still for posture analysis")
        case .movementTracking:
            utterance = AVSpeechUtterance(string: "Perform the movement that triggers your pain")
        }
        
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        synthesizer.speak(utterance)
    }
}

// MARK: - AR View Container

struct ARViewContainer: UIViewRepresentable {
    let arManager: ARBodyScanningManager
    
    func makeUIView(context: Context) -> ARView {
        return arManager.arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // Update AR view if needed
    }
}

// MARK: - Crosshair View

struct CrosshairView: View {
    @State private var isAnimating = false
    
    var body: some View {
        ZStack {
            // Outer ring
            Circle()
                .stroke(Color.red, lineWidth: 2)
                .scaleEffect(isAnimating ? 1.2 : 1.0)
                .opacity(isAnimating ? 0.3 : 0.8)
            
            // Inner crosshair
            VStack {
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 2, height: 15)
                Spacer()
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 2, height: 15)
            }
            
            HStack {
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 15, height: 2)
                Spacer()
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 15, height: 2)
            }
            
            // Center dot
            Circle()
                .fill(Color.red)
                .frame(width: 4, height: 4)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
                isAnimating = true
            }
        }
    }
}

// MARK: - Instruction Bubble

struct InstructionBubble: View {
    let mode: ARBodyScanningView.ScanningMode
    let isScanning: Bool
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: mode.icon)
                .font(.title2)
                .foregroundColor(.white)
            
            Text(isScanning ? "Scanning..." : mode.description)
                .font(.subheadline)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 16)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(.ultraThinMaterial)
        )
        .padding(.horizontal, 40)
    }
}

// MARK: - Scan Progress View

struct ScanProgressView: View {
    let progress: Double
    let mode: ARBodyScanningView.ScanningMode
    
    var body: some View {
        VStack(spacing: 12) {
            // Progress Ring
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.3), lineWidth: 4)
                    .frame(width: 60, height: 60)
                
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(Color.blue, lineWidth: 4)
                    .frame(width: 60, height: 60)
                    .rotationEffect(.degrees(-90))
                    .animation(.linear(duration: 0.3), value: progress)
                
                Text("\(Int(progress * 100))%")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
            }
            
            Text("Scanning \(mode.rawValue)")
                .font(.caption)
                .foregroundColor(.white)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(.ultraThinMaterial)
        )
    }
}

// MARK: - Detected Parts Overlay

struct DetectedPartsOverlay: View {
    let parts: [DetectedBodyPart]
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(parts, id: \.id) { part in
                    DetectedPartCard(part: part)
                }
            }
            .padding(.horizontal)
        }
    }
}

struct DetectedPartCard: View {
    let part: DetectedBodyPart
    
    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: part.bodyRegion.icon)
                .font(.title3)
                .foregroundColor(.white)
            
            Text(part.bodyRegion.displayName)
                .font(.caption2)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
            
            Text("\(Int(part.confidence * 100))%")
                .font(.caption2)
                .foregroundColor(.blue)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(.ultraThinMaterial)
        )
        .frame(width: 80)
    }
}

// MARK: - AR Body Part Overlay

struct ARBodyPartOverlay: View {
    let part: DetectedBodyPart
    let screenSize: CGSize
    
    var body: some View {
        VStack(spacing: 4) {
            // Pain intensity indicator
            Circle()
                .fill(Color.red.opacity(0.7))
                .frame(width: 20, height: 20)
                .overlay(
                    Circle()
                        .stroke(Color.white, lineWidth: 2)
                )
            
            // Body part label
            Text(part.bodyRegion.displayName)
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.black.opacity(0.7))
                )
        }
        .position(
            x: part.screenPosition.x * screenSize.width,
            y: part.screenPosition.y * screenSize.height
        )
    }
}

// MARK: - AR Calibration View

struct ARCalibrationView: View {
    let arManager: ARBodyScanningManager
    
    @State private var trackingQuality: ARCamera.TrackingState = .notAvailable
    @State private var lightingCondition: String = "Unknown"
    @State private var deviceOrientation: String = "Unknown"
    
    var body: some View {
        NavigationView {
            Form {
                Section("AR Session Status") {
                    HStack {
                        Image(systemName: trackingStatusIcon)
                            .foregroundColor(trackingStatusColor)
                        Text("Tracking Quality")
                        Spacer()
                        Text(trackingStatusText)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Image(systemName: "lightbulb")
                            .foregroundColor(.yellow)
                        Text("Lighting")
                        Spacer()
                        Text(lightingCondition)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Image(systemName: "iphone")
                            .foregroundColor(.blue)
                        Text("Device Orientation")
                        Spacer()
                        Text(deviceOrientation)
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Calibration Actions") {
                    Button("Reset AR Session") {
                        arManager.resetSession()
                    }
                    
                    Button("Recalibrate Body Tracking") {
                        arManager.recalibrateBodyTracking()
                    }
                    
                    Button("Clear Detected Anchors") {
                        arManager.clearAnchors()
                    }
                }
                
                Section("Performance Settings") {
                    Toggle("High Quality Tracking", isOn: .constant(true))
                    Toggle("Body Pose Detection", isOn: .constant(true))
                    Toggle("Hand Tracking", isOn: .constant(false))
                }
                
                Section("Tips") {
                    VStack(alignment: .leading, spacing: 8) {
                        TipRow(icon: "lightbulb", text: "Ensure good lighting conditions")
                        TipRow(icon: "figure.stand", text: "Stand 1-2 meters from the camera")
                        TipRow(icon: "hand.raised", text: "Move slowly for better tracking")
                        TipRow(icon: "iphone.landscape", text: "Hold device in landscape mode")
                    }
                }
            }
            .navigationTitle("AR Calibration")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    // Dismiss
                }
            )
        }
        .onAppear {
            updateCalibrationInfo()
        }
    }
    
    private var trackingStatusIcon: String {
        switch trackingQuality {
        case .normal: return "checkmark.circle.fill"
        case .limited: return "exclamationmark.triangle.fill"
        case .notAvailable: return "xmark.circle.fill"
        @unknown default: return "questionmark.circle.fill"
        }
    }
    
    private var trackingStatusColor: Color {
        switch trackingQuality {
        case .normal: return .green
        case .limited: return .orange
        case .notAvailable: return .red
        @unknown default: return .gray
        }
    }
    
    private var trackingStatusText: String {
        switch trackingQuality {
        case .normal: return "Good"
        case .limited: return "Limited"
        case .notAvailable: return "Not Available"
        @unknown default: return "Unknown"
        }
    }
    
    private func updateCalibrationInfo() {
        // Update calibration information from AR manager
        trackingQuality = arManager.currentTrackingState
        lightingCondition = arManager.lightingCondition
        deviceOrientation = arManager.deviceOrientation
    }
}

struct TipRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 20)
            
            Text(text)
                .font(.subheadline)
        }
    }
}

// MARK: - AR Scan Results View

struct ARScanResultsView: View {
    let detectedParts: [DetectedBodyPart]
    @Binding var selectedRegions: Set<BodyRegion>
    @Binding var painIntensity: [BodyRegion: Double]
    
    @State private var selectedPart: DetectedBodyPart?
    @State private var painLevel: Double = 5.0
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 10) {
                    Text("Scan Results")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("\(detectedParts.count) body parts detected")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()
                
                // Detected Parts List
                ScrollView {
                    LazyVStack(spacing: 15) {
                        ForEach(detectedParts, id: \.id) { part in
                            ARResultCard(
                                part: part,
                                isSelected: selectedPart?.id == part.id,
                                painLevel: painLevel
                            ) {
                                selectedPart = part
                            } onAddToPainTracking: {
                                selectedRegions.insert(part.bodyRegion)
                                painIntensity[part.bodyRegion] = painLevel
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                
                // Pain Level Slider
                if selectedPart != nil {
                    VStack(spacing: 10) {
                        Text("Pain Level: \(Int(painLevel))")
                            .font(.headline)
                        
                        Slider(value: $painLevel, in: 0...10, step: 1)
                            .accentColor(.red)
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 15)
                            .fill(Color(.systemGray6))
                    )
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    // Dismiss
                },
                trailing: Button("Save All") {
                    saveAllDetectedParts()
                }
            )
        }
    }
    
    private func saveAllDetectedParts() {
        for part in detectedParts {
            selectedRegions.insert(part.bodyRegion)
            painIntensity[part.bodyRegion] = painLevel
        }
    }
}

struct ARResultCard: View {
    let part: DetectedBodyPart
    let isSelected: Bool
    let painLevel: Double
    let onSelect: () -> Void
    let onAddToPainTracking: () -> Void
    
    var body: some View {
        HStack(spacing: 15) {
            // Body part icon
            Image(systemName: part.bodyRegion.icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 40)
            
            // Part info
            VStack(alignment: .leading, spacing: 4) {
                Text(part.bodyRegion.displayName)
                    .font(.headline)
                
                Text("Confidence: \(Int(part.confidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("Position: (\(String(format: "%.2f", part.worldPosition.x)), \(String(format: "%.2f", part.worldPosition.y)), \(String(format: "%.2f", part.worldPosition.z)))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Action buttons
            VStack(spacing: 8) {
                Button(action: onSelect) {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .foregroundColor(isSelected ? .green : .gray)
                }
                
                Button(action: onAddToPainTracking) {
                    Image(systemName: "plus.circle.fill")
                        .foregroundColor(.blue)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(isSelected ? Color.blue.opacity(0.1) : Color(.systemBackground))
                .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
        )
        .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
}

// MARK: - AR Body Scanning Manager

class ARBodyScanningManager: NSObject, ObservableObject {
    @Published var isSessionReady = false
    @Published var currentTrackingState: ARCamera.TrackingState = .notAvailable
    @Published var lightingCondition = "Unknown"
    @Published var deviceOrientation = "Unknown"
    
    let arView: ARView
    private var bodyTrackingConfiguration: ARBodyTrackingConfiguration?
    private var currentScanningMode: ARBodyScanningView.ScanningMode = .bodyMapping
    
    var onBodyPartsDetected: (([DetectedBodyPart]) -> Void)?
    var onScanProgressUpdated: ((Double) -> Void)?
    
    override init() {
        arView = ARView(frame: .zero)
        super.init()
        
        setupARView()
    }
    
    private func setupARView() {
        arView.session.delegate = self
        
        // Configure AR session
        if ARBodyTrackingConfiguration.isSupported {
            bodyTrackingConfiguration = ARBodyTrackingConfiguration()
            bodyTrackingConfiguration?.automaticImageScaleEstimationEnabled = true
        }
    }
    
    func startSession() {
        guard let configuration = bodyTrackingConfiguration else {
            print("Body tracking not supported")
            return
        }
        
        arView.session.run(configuration)
        isSessionReady = true
    }
    
    func stopSession() {
        arView.session.pause()
        isSessionReady = false
    }
    
    func resetSession() {
        guard let configuration = bodyTrackingConfiguration else { return }
        arView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    func setScanningMode(_ mode: ARBodyScanningView.ScanningMode) {
        currentScanningMode = mode
    }
    
    func startBodyScanning(mode: ARBodyScanningView.ScanningMode) {
        currentScanningMode = mode
        // Start specific scanning logic based on mode
    }
    
    func stopBodyScanning() {
        // Stop scanning logic
    }
    
    func recalibrateBodyTracking() {
        resetSession()
    }
    
    func clearAnchors() {
        arView.scene.anchors.removeAll()
    }
}

// MARK: - ARSessionDelegate

extension ARBodyScanningManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        currentTrackingState = frame.camera.trackingState
        
        // Update lighting condition
        let lightEstimate = frame.lightEstimate
        if let ambientIntensity = lightEstimate?.ambientIntensity {
            if ambientIntensity < 500 {
                lightingCondition = "Too Dark"
            } else if ambientIntensity > 2000 {
                lightingCondition = "Too Bright"
            } else {
                lightingCondition = "Good"
            }
        }
        
        // Process body tracking if available
        if let bodyAnchor = frame.anchors.first(where: { $0 is ARBodyAnchor }) as? ARBodyAnchor {
            processBodyAnchor(bodyAnchor)
        }
    }
    
    private func processBodyAnchor(_ bodyAnchor: ARBodyAnchor) {
        let skeleton = bodyAnchor.skeleton
        var detectedParts: [DetectedBodyPart] = []
        
        // Map skeleton joints to body regions
        let jointMappings: [(ARSkeleton.JointName, BodyRegion)] = [
            (.head, .head),
            (.neck1, .neck),
            (.leftShoulder, .leftShoulder),
            (.rightShoulder, .rightShoulder),
            (.leftArm, .leftUpperArm),
            (.rightArm, .rightUpperArm),
            (.leftForearm, .leftForearm),
            (.rightForearm, .rightForearm),
            (.leftHand, .leftHand),
            (.rightHand, .rightHand),
            (.spine7, .upperBack),
            (.spine4, .midBack),
            (.spine1, .lowerBack),
            (.leftUpLeg, .leftThigh),
            (.rightUpLeg, .rightThigh),
            (.leftLeg, .leftCalf),
            (.rightLeg, .rightCalf),
            (.leftFoot, .leftFoot),
            (.rightFoot, .rightFoot)
        ]
        
        for (jointName, bodyRegion) in jointMappings {
            if skeleton.isJointTracked(jointName) {
                let transform = skeleton.modelTransform(for: jointName)
                let worldPosition = simd_make_float3(transform.columns.3)
                
                // Convert to screen coordinates
                let screenPosition = arView.project(worldPosition) ?? CGPoint.zero
                let normalizedPosition = CGPoint(
                    x: screenPosition.x / arView.bounds.width,
                    y: screenPosition.y / arView.bounds.height
                )
                
                let detectedPart = DetectedBodyPart(
                    bodyRegion: bodyRegion,
                    worldPosition: worldPosition,
                    screenPosition: normalizedPosition,
                    confidence: 0.9, // High confidence for tracked joints
                    timestamp: Date()
                )
                
                detectedParts.append(detectedPart)
            }
        }
        
        DispatchQueue.main.async {
            self.onBodyPartsDetected?(detectedParts)
        }
    }
}

// MARK: - Data Models

struct DetectedBodyPart {
    let id = UUID()
    let bodyRegion: BodyRegion
    let worldPosition: simd_float3
    let screenPosition: CGPoint
    let confidence: Double
    let timestamp: Date
}

extension BodyRegion {
    var icon: String {
        switch self {
        case .head: return "head.profile"
        case .neck: return "figure.stand"
        case .leftShoulder, .rightShoulder: return "figure.arms.open"
        case .leftUpperArm, .rightUpperArm: return "arm"
        case .leftForearm, .rightForearm: return "hand.raised"
        case .leftHand, .rightHand: return "hand.point.up"
        case .chest: return "heart"
        case .upperBack, .midBack, .lowerBack: return "figure.stand"
        case .leftThigh, .rightThigh: return "figure.walk"
        case .leftKnee, .rightKnee: return "figure.walk.circle"
        case .leftCalf, .rightCalf: return "figure.run"
        case .leftAnkle, .rightAnkle: return "figure.walk"
        case .leftFoot, .rightFoot: return "shoe"
        default: return "circle"
        }
    }
}