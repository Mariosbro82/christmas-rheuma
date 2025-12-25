//
//  ARBodyScanningView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import ARKit
import RealityKit

struct ARBodyScanningView: View {
    @StateObject private var arManager = ARBodyScanningManager.shared
    @State private var selectedTab = 0
    @State private var showingPermissionAlert = false
    @State private var showingHelpSheet = false
    @State private var showingScanResults = false
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Scanning Tab
                ScanningView()
                    .tabItem {
                        Image(systemName: "camera.viewfinder")
                        Text("Scan")
                    }
                    .tag(0)
                
                // Results Tab
                ScanResultsView()
                    .tabItem {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("Results")
                    }
                    .tag(1)
                
                // History Tab
                ScanHistoryView()
                    .tabItem {
                        Image(systemName: "clock.arrow.circlepath")
                        Text("History")
                    }
                    .tag(2)
                
                // Settings Tab
                ARScanSettingsView()
                    .tabItem {
                        Image(systemName: "gearshape.fill")
                        Text("Settings")
                    }
                    .tag(3)
            }
            .navigationTitle("Body Scanning")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelpSheet = true
                    }
                }
            }
        }
        .onAppear {
            checkPermissions()
        }
        .alert("Camera Permission Required", isPresented: $showingPermissionAlert) {
            Button("Settings") {
                if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(settingsUrl)
                }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Camera access is required for body scanning. Please enable it in Settings.")
        }
        .sheet(isPresented: $showingHelpSheet) {
            ARScanHelpView()
        }
        .sheet(isPresented: $showingScanResults) {
            if let scan = arManager.currentScan {
                ScanResultDetailView(scan: scan)
            }
        }
        .onChange(of: arManager.currentScan) { scan in
            if scan != nil && !arManager.isScanning {
                showingScanResults = true
            }
        }
    }
    
    private func checkPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if !granted {
                    showingPermissionAlert = true
                }
            }
        }
    }
}

struct ScanningView: View {
    @StateObject private var arManager = ARBodyScanningManager.shared
    @State private var showingInstructions = true
    
    var body: some View {
        ZStack {
            if arManager.isSupported {
                ARViewContainer()
                    .edgesIgnoringSafeArea(.all)
                
                VStack {
                    // Top overlay
                    ScanningOverlayView()
                        .padding(.top)
                    
                    Spacer()
                    
                    // Bottom controls
                    ScanningControlsView()
                        .padding(.bottom, 50)
                }
            } else {
                ARNotSupportedView()
            }
        }
        .sheet(isPresented: $showingInstructions) {
            ScanInstructionsView(isPresented: $showingInstructions)
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    @StateObject private var arManager = ARBodyScanningManager.shared
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arManager.setupARSession(in: arView)
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // Update AR view if needed
    }
}

struct ScanningOverlayView: View {
    @StateObject private var arManager = ARBodyScanningManager.shared
    
    var body: some View {
        VStack(spacing: 16) {
            // Status card
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 12, height: 12)
                
                Text(statusText)
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
                
                if arManager.isScanning {
                    Text("\(Int(arManager.scanProgress * 100))%")
                        .font(.caption)
                        .foregroundColor(.white)
                }
            }
            .padding()
            .background(Color.black.opacity(0.7))
            .cornerRadius(12)
            .padding(.horizontal)
            
            // Progress bar
            if arManager.isScanning {
                ProgressView(value: arManager.scanProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                    .padding(.horizontal)
            }
            
            // Quality indicators
            if let scan = arManager.currentScan {
                QualityIndicatorsView(quality: scan.scanQuality)
            }
            
            // Error message
            if let error = arManager.error {
                ErrorMessageView(error: error)
            }
        }
    }
    
    private var statusColor: Color {
        if arManager.isScanning {
            return .green
        } else if arManager.error != nil {
            return .red
        } else {
            return .gray
        }
    }
    
    private var statusText: String {
        if arManager.isScanning {
            return "Scanning..."
        } else if arManager.error != nil {
            return "Error"
        } else {
            return "Ready to Scan"
        }
    }
}

struct QualityIndicatorsView: View {
    let quality: ScanQuality
    
    var body: some View {
        HStack(spacing: 20) {
            QualityIndicator(
                title: "Lighting",
                value: quality.lightingQuality,
                icon: "sun.max.fill"
            )
            
            QualityIndicator(
                title: "Tracking",
                value: quality.trackingStability,
                icon: "target"
            )
            
            QualityIndicator(
                title: "Visibility",
                value: quality.bodyVisibility,
                icon: "eye.fill"
            )
        }
        .padding()
        .background(Color.black.opacity(0.7))
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

struct QualityIndicator: View {
    let title: String
    let value: Float
    let icon: String
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .foregroundColor(qualityColor)
                .font(.caption)
            
            Text(title)
                .font(.caption2)
                .foregroundColor(.white)
            
            Circle()
                .fill(qualityColor)
                .frame(width: 8, height: 8)
        }
    }
    
    private var qualityColor: Color {
        if value > 0.8 {
            return .green
        } else if value > 0.6 {
            return .yellow
        } else if value > 0.4 {
            return .orange
        } else {
            return .red
        }
    }
}

struct ErrorMessageView: View {
    let error: ARBodyScanError
    
    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            
            Text(error.localizedDescription)
                .font(.caption)
                .foregroundColor(.white)
        }
        .padding()
        .background(Color.red.opacity(0.8))
        .cornerRadius(8)
        .padding(.horizontal)
    }
}

struct ScanningControlsView: View {
    @StateObject private var arManager = ARBodyScanningManager.shared
    
    var body: some View {
        VStack(spacing: 20) {
            // Main scan button
            Button(action: toggleScanning) {
                ZStack {
                    Circle()
                        .fill(buttonColor)
                        .frame(width: 80, height: 80)
                    
                    Image(systemName: buttonIcon)
                        .font(.system(size: 30, weight: .bold))
                        .foregroundColor(.white)
                }
            }
            .scaleEffect(arManager.isScanning ? 1.1 : 1.0)
            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: arManager.isScanning)
            
            // Control buttons
            HStack(spacing: 30) {
                if arManager.isScanning {
                    Button("Pause") {
                        arManager.pauseScanning()
                    }
                    .buttonStyle(SecondaryButtonStyle())
                    
                    Button("Stop") {
                        arManager.stopScanning()
                    }
                    .buttonStyle(DestructiveButtonStyle())
                } else {
                    Button("Instructions") {
                        // Show instructions
                    }
                    .buttonStyle(SecondaryButtonStyle())
                    
                    Button("Settings") {
                        // Show settings
                    }
                    .buttonStyle(SecondaryButtonStyle())
                }
            }
        }
    }
    
    private func toggleScanning() {
        if arManager.isScanning {
            arManager.stopScanning()
        } else {
            arManager.startScanning()
        }
    }
    
    private var buttonColor: Color {
        if arManager.isScanning {
            return .red
        } else {
            return .blue
        }
    }
    
    private var buttonIcon: String {
        if arManager.isScanning {
            return "stop.fill"
        } else {
            return "play.fill"
        }
    }
}

struct ScanResultsView: View {
    @StateObject private var arManager = ARBodyScanningManager.shared
    
    var body: some View {
        ScrollView {
            if let scan = arManager.currentScan {
                VStack(spacing: 20) {
                    // Overall score
                    PostureScoreCard(score: scan.postureAnalysis.overallScore)
                    
                    // Posture analysis
                    PostureAnalysisCard(analysis: scan.postureAnalysis)
                    
                    // Balance metrics
                    BalanceMetricsCard(metrics: scan.balanceMetrics)
                    
                    // Recommendations
                    RecommendationsCard(recommendations: scan.recommendations)
                    
                    // Detailed metrics
                    DetailedMetricsCard(scan: scan)
                }
                .padding()
            } else {
                NoScanDataView()
            }
        }
        .navigationTitle("Scan Results")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct PostureScoreCard: View {
    let score: Float
    
    var body: some View {
        VStack(spacing: 16) {
            Text("Overall Posture Score")
                .font(.headline)
                .foregroundColor(.primary)
            
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 8)
                    .frame(width: 120, height: 120)
                
                Circle()
                    .trim(from: 0, to: CGFloat(score))
                    .stroke(scoreColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                
                VStack {
                    Text("\(Int(score * 100))")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(scoreColor)
                    
                    Text("/ 100")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Text(scoreDescription)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var scoreColor: Color {
        if score > 0.8 {
            return .green
        } else if score > 0.6 {
            return .yellow
        } else if score > 0.4 {
            return .orange
        } else {
            return .red
        }
    }
    
    private var scoreDescription: String {
        if score > 0.8 {
            return "Excellent posture! Keep up the good work."
        } else if score > 0.6 {
            return "Good posture with room for improvement."
        } else if score > 0.4 {
            return "Fair posture. Consider the recommendations below."
        } else {
            return "Poor posture. Please follow the recommendations."
        }
    }
}

struct PostureAnalysisCard: View {
    let analysis: PostureAnalysis
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Posture Analysis")
                .font(.headline)
                .foregroundColor(.primary)
            
            VStack(spacing: 12) {
                PostureMetricRow(title: "Head Position", metric: analysis.headPosition)
                PostureMetricRow(title: "Shoulder Alignment", metric: analysis.shoulderAlignment)
                PostureMetricRow(title: "Spinal Curvature", metric: analysis.spinalCurvature)
                PostureMetricRow(title: "Hip Alignment", metric: analysis.hipAlignment)
            }
            
            if analysis.postureType != .neutral {
                HStack {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(.blue)
                    
                    Text("Detected: \(analysis.postureType.description)")
                        .font(.subheadline)
                        .foregroundColor(.primary)
                }
                .padding(.top, 8)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct PostureMetricRow: View {
    let title: String
    let metric: PostureMetric
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.primary)
                
                Text(metric.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                HStack {
                    Circle()
                        .fill(metric.severity.color)
                        .frame(width: 8, height: 8)
                    
                    Text(metric.severity.description)
                        .font(.caption)
                        .foregroundColor(.primary)
                }
                
                if metric.trend != .stable {
                    HStack {
                        Image(systemName: metric.trend.systemImage)
                            .foregroundColor(metric.trend.color)
                            .font(.caption)
                        
                        Text(metric.trend.rawValue.capitalized)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
}

struct BalanceMetricsCard: View {
    let metrics: BalanceMetrics
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Balance Analysis")
                .font(.headline)
                .foregroundColor(.primary)
            
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Stability Score")
                        .font(.subheadline)
                        .foregroundColor(.primary)
                    
                    Text("\(Int(metrics.stabilityScore * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(stabilityColor)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 8) {
                    Text("Fall Risk")
                        .font(.subheadline)
                        .foregroundColor(.primary)
                    
                    HStack {
                        Circle()
                            .fill(metrics.fallRisk.color)
                            .frame(width: 12, height: 12)
                        
                        Text(metrics.fallRisk.description)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                    }
                }
            }
            
            // Weight distribution
            WeightDistributionView(distribution: metrics.weightDistribution)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var stabilityColor: Color {
        if metrics.stabilityScore > 0.8 {
            return .green
        } else if metrics.stabilityScore > 0.6 {
            return .yellow
        } else if metrics.stabilityScore > 0.4 {
            return .orange
        } else {
            return .red
        }
    }
}

struct WeightDistributionView: View {
    let distribution: WeightDistribution
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Weight Distribution")
                .font(.subheadline)
                .foregroundColor(.primary)
            
            HStack {
                VStack {
                    Text("Left")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("\(Int(distribution.leftFoot))%")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                
                Spacer()
                
                VStack {
                    Text("Right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("\(Int(distribution.rightFoot))%")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
            }
            
            // Visual representation
            HStack(spacing: 4) {
                Rectangle()
                    .fill(Color.blue)
                    .frame(width: CGFloat(distribution.leftFoot) * 2, height: 8)
                
                Rectangle()
                    .fill(Color.orange)
                    .frame(width: CGFloat(distribution.rightFoot) * 2, height: 8)
            }
            .cornerRadius(4)
        }
    }
}

struct RecommendationsCard: View {
    let recommendations: [PostureRecommendation]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recommendations")
                .font(.headline)
                .foregroundColor(.primary)
            
            if recommendations.isEmpty {
                Text("No specific recommendations at this time. Your posture looks good!")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .italic()
            } else {
                ForEach(recommendations.prefix(3), id: \.id) { recommendation in
                    RecommendationRow(recommendation: recommendation)
                }
                
                if recommendations.count > 3 {
                    NavigationLink("View All Recommendations") {
                        AllRecommendationsView(recommendations: recommendations)
                    }
                    .font(.subheadline)
                    .foregroundColor(.blue)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct RecommendationRow: View {
    let recommendation: PostureRecommendation
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: recommendation.type.systemImage)
                .foregroundColor(recommendation.type.color)
                .font(.title3)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(recommendation.title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
                
                Text(recommendation.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            Circle()
                .fill(recommendation.priority.color)
                .frame(width: 8, height: 8)
        }
        .padding(.vertical, 4)
    }
}

struct DetailedMetricsCard: View {
    let scan: BodyScanData
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Button(action: { isExpanded.toggle() }) {
                HStack {
                    Text("Detailed Metrics")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Spacer()
                    
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                }
            }
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    MetricDetailRow(title: "Scan Quality", value: "\(Int(scan.scanQuality.overallScore * 100))%")
                    MetricDetailRow(title: "Scan Duration", value: formatDuration(scan.timestamp))
                    MetricDetailRow(title: "Joints Tracked", value: "\(scan.jointPositions.count)")
                    
                    if !scan.bodyMeasurements.asymmetries.isEmpty {
                        Text("Body Asymmetries Detected: \(scan.bodyMeasurements.asymmetries.count)")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func formatDuration(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

struct MetricDetailRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.primary)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
        }
    }
}

struct NoScanDataView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Scan Data")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
            
            Text("Perform a body scan to see your posture analysis and recommendations.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }
}

struct ScanHistoryView: View {
    @StateObject private var arManager = ARBodyScanningManager.shared
    @State private var selectedScan: BodyScanData?
    
    var body: some View {
        List {
            ForEach(arManager.scanHistory.reversed(), id: \.id) { scan in
                ScanHistoryRow(scan: scan)
                    .onTapGesture {
                        selectedScan = scan
                    }
            }
            .onDelete(perform: deleteScans)
        }
        .navigationTitle("Scan History")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selectedScan) { scan in
            ScanResultDetailView(scan: scan)
        }
    }
    
    private func deleteScans(at offsets: IndexSet) {
        let reversedHistory = arManager.scanHistory.reversed()
        for index in offsets {
            let scan = Array(reversedHistory)[index]
            arManager.deleteScan(scan)
        }
    }
}

struct ScanHistoryRow: View {
    let scan: BodyScanData
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(formatDate(scan.timestamp))
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text("Score: \(Int(scan.postureAnalysis.overallScore * 100))%")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                if scan.postureAnalysis.postureType != .neutral {
                    Text(scan.postureAnalysis.postureType.description)
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Circle()
                    .fill(scoreColor(scan.postureAnalysis.overallScore))
                    .frame(width: 12, height: 12)
                
                Text("\(scan.recommendations.count) tips")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    private func scoreColor(_ score: Float) -> Color {
        if score > 0.8 {
            return .green
        } else if score > 0.6 {
            return .yellow
        } else if score > 0.4 {
            return .orange
        } else {
            return .red
        }
    }
}

struct ARScanSettingsView: View {
    @AppStorage("arScanAutoSave") private var autoSave = true
    @AppStorage("arScanHapticFeedback") private var hapticFeedback = true
    @AppStorage("arScanVoiceGuidance") private var voiceGuidance = false
    @AppStorage("arScanQualityThreshold") private var qualityThreshold = 0.7
    
    var body: some View {
        Form {
            Section("Scanning") {
                Toggle("Auto-save Scans", isOn: $autoSave)
                Toggle("Haptic Feedback", isOn: $hapticFeedback)
                Toggle("Voice Guidance", isOn: $voiceGuidance)
            }
            
            Section("Quality") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Quality Threshold: \(Int(qualityThreshold * 100))%")
                        .font(.subheadline)
                    
                    Slider(value: $qualityThreshold, in: 0.5...1.0, step: 0.05)
                }
            }
            
            Section("Data") {
                Button("Export All Scans") {
                    // Export functionality
                }
                
                Button("Clear History", role: .destructive) {
                    // Clear history
                }
            }
        }
        .navigationTitle("AR Scan Settings")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ARNotSupportedView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.orange)
            
            Text("AR Not Supported")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
            
            Text("Body scanning requires a device with A12 Bionic chip or later and iOS 13.0 or later.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }
}

struct ScanInstructionsView: View {
    @Binding var isPresented: Bool
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("How to Perform a Body Scan")
                        .font(.title2)
                        .fontWeight(.bold)
                        .padding(.bottom)
                    
                    InstructionStep(
                        number: 1,
                        title: "Prepare Your Space",
                        description: "Find a well-lit area with at least 6 feet of space around you. Remove any obstacles."
                    )
                    
                    InstructionStep(
                        number: 2,
                        title: "Position Yourself",
                        description: "Stand 3-4 feet away from your device. Make sure your entire body is visible in the camera."
                    )
                    
                    InstructionStep(
                        number: 3,
                        title: "Start Scanning",
                        description: "Tap the scan button and remain still. The scan will take about 30 seconds."
                    )
                    
                    InstructionStep(
                        number: 4,
                        title: "Review Results",
                        description: "After scanning, review your posture analysis and follow the recommendations."
                    )
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Tips for Best Results")
                            .font(.headline)
                            .padding(.top)
                        
                        Text("• Wear form-fitting clothes")
                        Text("• Stand naturally with arms at your sides")
                        Text("• Look straight ahead")
                        Text("• Avoid moving during the scan")
                        Text("• Ensure good lighting")
                    }
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                }
                .padding()
            }
            .navigationTitle("Instructions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        isPresented = false
                    }
                }
            }
        }
    }
}

struct InstructionStep: View {
    let number: Int
    let title: String
    let description: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            ZStack {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 32, height: 32)
                
                Text("\(number)")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct ARScanHelpView: View {
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("AR Body Scanning Help")
                        .font(.title2)
                        .fontWeight(.bold)
                        .padding(.bottom)
                    
                    HelpSection(
                        title: "What is Body Scanning?",
                        content: "Body scanning uses advanced AR technology to analyze your posture, balance, and movement patterns in real-time."
                    )
                    
                    HelpSection(
                        title: "How Accurate is it?",
                        content: "Our scanning technology is highly accurate when used in optimal conditions. Results are for informational purposes and should not replace professional medical advice."
                    )
                    
                    HelpSection(
                        title: "Troubleshooting",
                        content: "If you're experiencing issues, ensure good lighting, clear the area of obstacles, and make sure your device supports AR body tracking."
                    )
                    
                    HelpSection(
                        title: "Privacy",
                        content: "All scan data is stored locally on your device. No video or images are saved, only the analyzed posture data."
                    )
                }
                .padding()
            }
            .navigationTitle("Help")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        // Dismiss
                    }
                }
            }
        }
    }
}

struct HelpSection: View {
    let title: String
    let content: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(.primary)
            
            Text(content)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

struct ScanResultDetailView: View {
    let scan: BodyScanData
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    PostureScoreCard(score: scan.postureAnalysis.overallScore)
                    PostureAnalysisCard(analysis: scan.postureAnalysis)
                    BalanceMetricsCard(metrics: scan.balanceMetrics)
                    RecommendationsCard(recommendations: scan.recommendations)
                    DetailedMetricsCard(scan: scan)
                }
                .padding()
            }
            .navigationTitle("Scan Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Share") {
                        // Share functionality
                    }
                }
            }
        }
    }
}

struct AllRecommendationsView: View {
    let recommendations: [PostureRecommendation]
    
    var body: some View {
        List {
            ForEach(recommendations, id: \.id) { recommendation in
                RecommendationDetailRow(recommendation: recommendation)
            }
        }
        .navigationTitle("All Recommendations")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct RecommendationDetailRow: View {
    let recommendation: PostureRecommendation
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button(action: { isExpanded.toggle() }) {
                HStack {
                    Image(systemName: recommendation.type.systemImage)
                        .foregroundColor(recommendation.type.color)
                        .font(.title3)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text(recommendation.title)
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        Text(recommendation.description)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .lineLimit(isExpanded ? nil : 2)
                    }
                    
                    Spacer()
                    
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                }
            }
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    if !recommendation.exercises.isEmpty {
                        Text("Recommended Exercises:")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        
                        ForEach(recommendation.exercises.prefix(3), id: \.name) { exercise in
                            Text("• \(exercise.name)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    if !recommendation.expectedImprovement.isEmpty {
                        Text("Expected Improvement: \(recommendation.expectedImprovement)")
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                }
                .padding(.leading, 32)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Button Styles

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.gray.opacity(0.2))
            .foregroundColor(.primary)
            .cornerRadius(8)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
    }
}

struct DestructiveButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.red)
            .foregroundColor(.white)
            .cornerRadius(8)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
    }
}

#Preview {
    ARBodyScanningView()
}