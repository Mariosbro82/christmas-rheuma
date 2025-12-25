//
//  ARScanningView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import ARKit
import SceneKit

struct ARScanningView: View {
    @ObservedObject var manager: BodyMappingManager
    @State private var showingInstructions = true
    @State private var scanningPhase: ScanningPhase = .preparation
    @State private var showingResults = false
    @State private var arViewCoordinator: ARViewCoordinator?
    
    enum ScanningPhase {
        case preparation
        case scanning
        case processing
        case completed
        
        var title: String {
            switch self {
            case .preparation: return "Prepare for Scan"
            case .scanning: return "Scanning in Progress"
            case .processing: return "Processing Results"
            case .completed: return "Scan Complete"
            }
        }
        
        var description: String {
            switch self {
            case .preparation: return "Position yourself in good lighting and ensure your full body is visible"
            case .scanning: return "Hold still while we capture your posture and movement patterns"
            case .processing: return "Analyzing your scan data and generating insights"
            case .completed: return "Your scan has been processed successfully"
            }
        }
    }
    
    var body: some View {
        VStack {
            if showingInstructions {
                ScanInstructionsView {
                    showingInstructions = false
                    startScanning()
                }
            } else {
                scanningContent
            }
        }
        .navigationTitle("AR Body Scan")
        .sheet(isPresented: $showingResults) {
            if let scan = manager.currentScan {
                ScanResultsView(scan: scan, manager: manager)
            }
        }
    }
    
    @ViewBuilder
    private var scanningContent: View {
        VStack {
            // AR Camera View
            ARCameraView(manager: manager, phase: $scanningPhase)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .cornerRadius(12)
                .overlay(
                    scanningOverlay,
                    alignment: .bottom
                )
            
            // Controls
            scanningControls
        }
        .padding()
    }
    
    @ViewBuilder
    private var scanningOverlay: some View {
        VStack(spacing: 16) {
            // Progress indicator
            if scanningPhase == .scanning || scanningPhase == .processing {
                ProgressView(value: manager.scanProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                    .frame(width: 200)
            }
            
            // Phase info
            VStack(spacing: 8) {
                Text(scanningPhase.title)
                    .font(.headline)
                    .foregroundColor(.white)
                
                Text(scanningPhase.description)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
            }
            .padding()
            .background(Color.black.opacity(0.7))
            .cornerRadius(12)
        }
        .padding()
    }
    
    @ViewBuilder
    private var scanningControls: some View {
        HStack(spacing: 20) {
            Button("Cancel") {
                cancelScanning()
            }
            .buttonStyle(.bordered)
            .disabled(scanningPhase == .processing)
            
            Spacer()
            
            if scanningPhase == .preparation {
                Button("Start Scan") {
                    startScanning()
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
            } else if scanningPhase == .scanning {
                Button("Stop Scan") {
                    stopScanning()
                }
                .buttonStyle(.bordered)
            } else if scanningPhase == .completed {
                Button("View Results") {
                    showingResults = true
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
            }
        }
        .padding()
    }
    
    private func startScanning() {
        scanningPhase = .scanning
        manager.startARSession()
        
        // Simulate scanning progress
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
            if manager.scanProgress < 1.0 {
                manager.scanProgress += 0.02
            } else {
                timer.invalidate()
                scanningPhase = .processing
                
                // Simulate processing delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    scanningPhase = .completed
                }
            }
        }
    }
    
    private func stopScanning() {
        manager.stopARSession()
        scanningPhase = .processing
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            scanningPhase = .completed
        }
    }
    
    private func cancelScanning() {
        manager.stopARSession()
        scanningPhase = .preparation
        manager.scanProgress = 0.0
        showingInstructions = true
    }
}

struct ScanInstructionsView: View {
    let onStart: () -> Void
    
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "figure.stand")
                .font(.system(size: 80))
                .foregroundColor(.blue)
            
            VStack(spacing: 16) {
                Text("AR Body Scan Instructions")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                VStack(alignment: .leading, spacing: 12) {
                    InstructionRow(
                        icon: "lightbulb",
                        title: "Good Lighting",
                        description: "Ensure you're in a well-lit area"
                    )
                    
                    InstructionRow(
                        icon: "figure.walk",
                        title: "Full Body Visible",
                        description: "Stand 6-8 feet from your device"
                    )
                    
                    InstructionRow(
                        icon: "hand.raised",
                        title: "Stay Still",
                        description: "Remain stationary during the scan"
                    )
                    
                    InstructionRow(
                        icon: "clock",
                        title: "2-3 Minutes",
                        description: "The scan will take a few minutes"
                    )
                }
            }
            
            Button("Start AR Scan") {
                onStart()
            }
            .buttonStyle(.borderedProminent)
            .tint(Colors.Primary.p500)
            .controlSize(.large)
        }
        .padding()
    }
}

struct InstructionRow: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
    }
}

struct ARCameraView: UIViewRepresentable {
    @ObservedObject var manager: BodyMappingManager
    @Binding var phase: ARScanningView.ScanningPhase
    
    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView()
        arView.delegate = context.coordinator
        arView.session.delegate = context.coordinator
        
        // Configure AR session
        let configuration = ARBodyTrackingConfiguration()
        arView.session.run(configuration)
        
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {
        // Update AR view based on scanning phase
    }
    
    func makeCoordinator() -> ARViewCoordinator {
        ARViewCoordinator(manager: manager, phase: $phase)
    }
}

class ARViewCoordinator: NSObject, ARSCNViewDelegate, ARSessionDelegate {
    let manager: BodyMappingManager
    @Binding var phase: ARScanningView.ScanningPhase
    
    init(manager: BodyMappingManager, phase: Binding<ARScanningView.ScanningPhase>) {
        self.manager = manager
        self._phase = phase
    }
    
    // MARK: - ARSCNViewDelegate
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard let bodyAnchor = anchor as? ARBodyAnchor else { return }
        
        // Process body anchor data
        DispatchQueue.main.async {
            // Update scanning progress or process body data
        }
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let bodyAnchor = anchor as? ARBodyAnchor else { return }
        
        // Update body tracking data
    }
    
    // MARK: - ARSessionDelegate
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        print("AR session failed: \(error)")
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        print("AR session was interrupted")
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        print("AR session interruption ended")
    }
}

struct ScanResultsView: View {
    let scan: BodyScan
    @ObservedObject var manager: BodyMappingManager
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab = 0
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Overall Results
                OverallResultsView(scan: scan)
                    .tabItem {
                        Image(systemName: "chart.bar")
                        Text("Overview")
                    }
                    .tag(0)
                
                // Posture Analysis
                if let arData = scan.arData {
                    PostureAnalysisView(postureAnalysis: arData.postureAnalysis)
                        .tabItem {
                            Image(systemName: "figure.stand")
                            Text("Posture")
                        }
                        .tag(1)
                    
                    // Movement Patterns
                    MovementPatternsView(movements: arData.movementPatterns)
                        .tabItem {
                            Image(systemName: "figure.walk")
                            Text("Movement")
                        }
                        .tag(2)
                }
                
                // Recommendations
                RecommendationsView(scan: scan)
                    .tabItem {
                        Image(systemName: "lightbulb")
                        Text("Tips")
                    }
                    .tag(3)
            }
            .navigationTitle("Scan Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        // Save scan results
                        dismiss()
                    }
                }
            }
        }
    }
}

struct OverallResultsView: View {
    let scan: BodyScan
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Scan Summary
                VStack(alignment: .leading, spacing: 8) {
                    Text("Scan Summary")
                        .font(.headline)
                    
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Scan Type")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(scan.scanType.displayName)
                                .font(.body)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .trailing) {
                            Text("Overall Pain Score")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f/4.0", scan.overallPainScore))
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(painScoreColor)
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                }
                
                // Affected Regions
                VStack(alignment: .leading, spacing: 8) {
                    Text("Affected Regions")
                        .font(.headline)
                    
                    let painfulRegions = scan.regions.filter { $0.painLevel.rawValue > 0 }
                    
                    if painfulRegions.isEmpty {
                        Text("No pain reported in any regions")
                            .foregroundColor(.secondary)
                            .italic()
                    } else {
                        ForEach(painfulRegions, id: \.id) { region in
                            HStack {
                                Circle()
                                    .fill(region.painLevel.color)
                                    .frame(width: 16, height: 16)
                                
                                Text(region.name)
                                    .font(.body)
                                
                                Spacer()
                                
                                Text(region.painLevel.displayName)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 2)
                        }
                    }
                }
                
                // Scan Notes
                if !scan.notes.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Notes")
                            .font(.headline)
                        
                        Text(scan.notes)
                            .font(.body)
                    }
                }
            }
            .padding()
        }
    }
    
    private var painScoreColor: Color {
        switch scan.overallPainScore {
        case 0...1: return .green
        case 1...2: return .yellow
        case 2...3: return .orange
        default: return .red
        }
    }
}

struct PostureAnalysisView: View {
    let postureAnalysis: PostureAnalysis
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Overall Score
                VStack(spacing: 8) {
                    Text("Overall Posture Score")
                        .font(.headline)
                    
                    ZStack {
                        Circle()
                            .stroke(Color.gray.opacity(0.3), lineWidth: 8)
                            .frame(width: 120, height: 120)
                        
                        Circle()
                            .trim(from: 0, to: CGFloat(postureAnalysis.overallScore))
                            .stroke(scoreColor, lineWidth: 8)
                            .frame(width: 120, height: 120)
                            .rotationEffect(.degrees(-90))
                        
                        VStack {
                            Text(String(format: "%.0f%%", postureAnalysis.overallScore * 100))
                                .font(.title)
                                .fontWeight(.bold)
                            Text("Score")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                
                // Individual Metrics
                VStack(alignment: .leading, spacing: 12) {
                    Text("Detailed Analysis")
                        .font(.headline)
                    
                    PostureMetricRow(
                        title: "Spinal Alignment",
                        score: postureAnalysis.spinalAlignment,
                        icon: "figure.stand"
                    )
                    
                    PostureMetricRow(
                        title: "Shoulder Level",
                        score: postureAnalysis.shoulderLevel,
                        icon: "figure.arms.open"
                    )
                    
                    PostureMetricRow(
                        title: "Hip Alignment",
                        score: postureAnalysis.hipAlignment,
                        icon: "figure.walk"
                    )
                    
                    PostureMetricRow(
                        title: "Head Position",
                        score: postureAnalysis.headPosition,
                        icon: "head.profile"
                    )
                }
                
                // Recommendations
                if !postureAnalysis.recommendations.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recommendations")
                            .font(.headline)
                        
                        ForEach(postureAnalysis.recommendations, id: \.self) { recommendation in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "lightbulb")
                                    .foregroundColor(.yellow)
                                    .font(.caption)
                                
                                Text(recommendation)
                                    .font(.body)
                            }
                        }
                    }
                }
            }
            .padding()
        }
    }
    
    private var scoreColor: Color {
        switch postureAnalysis.overallScore {
        case 0.8...1.0: return .green
        case 0.6...0.8: return .yellow
        case 0.4...0.6: return .orange
        default: return .red
        }
    }
}

struct PostureMetricRow: View {
    let title: String
    let score: Float
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body)
                
                ProgressView(value: score)
                    .progressViewStyle(LinearProgressViewStyle(tint: scoreColor))
            }
            
            Text(String(format: "%.0f%%", score * 100))
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
    }
    
    private var scoreColor: Color {
        switch score {
        case 0.8...1.0: return .green
        case 0.6...0.8: return .yellow
        case 0.4...0.6: return .orange
        default: return .red
        }
    }
}

struct MovementPatternsView: View {
    let movements: [MovementPattern]
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Movement Analysis")
                    .font(.headline)
                    .padding(.horizontal)
                
                if movements.isEmpty {
                    Text("No movement patterns detected")
                        .foregroundColor(.secondary)
                        .italic()
                        .padding()
                } else {
                    ForEach(movements.indices, id: \.self) { index in
                        MovementPatternCard(movement: movements[index])
                    }
                    .padding(.horizontal)
                }
            }
        }
    }
}

struct MovementPatternCard: View {
    let movement: MovementPattern
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(movement.jointName)
                    .font(.headline)
                
                Spacer()
                
                if movement.compensation {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(.orange)
                }
            }
            
            VStack(spacing: 8) {
                HStack {
                    Text("Range of Motion")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.0f%%", movement.rangeOfMotion * 100))
                        .font(.caption)
                }
                ProgressView(value: movement.rangeOfMotion)
                    .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                
                HStack {
                    Text("Smoothness")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.0f%%", movement.smoothness * 100))
                        .font(.caption)
                }
                ProgressView(value: movement.smoothness)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
            }
            
            if !movement.recommendations.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Recommendations:")
                        .font(.caption)
                        .fontWeight(.medium)
                    
                    ForEach(movement.recommendations, id: \.self) { recommendation in
                        Text("• \(recommendation)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }
}

struct RecommendationsView: View {
    let scan: BodyScan
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Personalized Recommendations")
                    .font(.headline)
                    .padding(.horizontal)
                
                // General recommendations based on scan
                LazyVStack(spacing: 12) {
                    RecommendationCard(
                        icon: "figure.strengthtraining.functional",
                        title: "Exercise Recommendations",
                        description: "Based on your posture analysis, focus on core strengthening and flexibility exercises.",
                        priority: .high
                    )
                    
                    RecommendationCard(
                        icon: "desktopcomputer",
                        title: "Ergonomic Improvements",
                        description: "Consider adjusting your workspace setup to improve posture throughout the day.",
                        priority: .medium
                    )
                    
                    RecommendationCard(
                        icon: "bell",
                        title: "Movement Reminders",
                        description: "Set regular reminders to take breaks and perform stretching exercises.",
                        priority: .medium
                    )
                    
                    if scan.overallPainScore > 2.0 {
                        RecommendationCard(
                            icon: "stethoscope",
                            title: "Consult Healthcare Provider",
                            description: "Your pain levels suggest you should discuss these findings with your healthcare provider.",
                            priority: .high
                        )
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct RecommendationCard: View {
    let icon: String
    let title: String
    let description: String
    let priority: Priority
    
    enum Priority {
        case low, medium, high
        
        var color: Color {
            switch self {
            case .low: return .green
            case .medium: return .yellow
            case .high: return .red
            }
        }
    }
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(title)
                        .font(.headline)
                    
                    Spacer()
                    
                    Circle()
                        .fill(priority.color)
                        .frame(width: 8, height: 8)
                }
                
                Text(description)
                    .font(.body)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }
}

#Preview {
    ARScanningView(manager: BodyMappingManager())
}