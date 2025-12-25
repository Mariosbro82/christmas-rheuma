//
//  AppleWatchPainMonitoringView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import WatchConnectivity
import HealthKit
import CoreMotion
import UserNotifications

struct AppleWatchPainMonitoringView: View {
    @StateObject private var watchManager = AppleWatchManager.shared
    @StateObject private var healthManager = HealthKitManager.shared
    @StateObject private var aiEngine = AIMLEngine.shared
    
    @State private var isWatchConnected = false
    @State private var watchBatteryLevel: Double = 0.0
    @State private var lastSyncTime: Date?
    @State private var realTimePainData: [WatchPainReading] = []
    @State private var showingWatchSetup = false
    @State private var enabledFeatures: Set<WatchFeature> = []
    @State private var syncStatus: SyncStatus = .idle
    @State private var watchAppInstalled = false
    @State private var heartRateData: [HeartRateReading] = []
    @State private var movementData: [MovementReading] = []
    @State private var selectedTimeRange: TimeRange = .today
    
    private enum TimeRange: String, CaseIterable {
        case today = "Today"
        case week = "This Week"
        case month = "This Month"
        
        var dateRange: (start: Date, end: Date) {
            let calendar = Calendar.current
            let now = Date()
            
            switch self {
            case .today:
                let start = calendar.startOfDay(for: now)
                return (start, now)
            case .week:
                let start = calendar.dateInterval(of: .weekOfYear, for: now)?.start ?? now
                return (start, now)
            case .month:
                let start = calendar.dateInterval(of: .month, for: now)?.start ?? now
                return (start, now)
            }
        }
    }
    
    private enum SyncStatus {
        case idle
        case syncing
        case success
        case failed(String)
        
        var color: Color {
            switch self {
            case .idle: return .gray
            case .syncing: return .blue
            case .success: return .green
            case .failed: return .red
            }
        }
        
        var icon: String {
            switch self {
            case .idle: return "circle"
            case .syncing: return "arrow.triangle.2.circlepath"
            case .success: return "checkmark.circle.fill"
            case .failed: return "exclamationmark.triangle.fill"
            }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Watch Connection Status
                    WatchConnectionCard(
                        isConnected: isWatchConnected,
                        batteryLevel: watchBatteryLevel,
                        lastSync: lastSyncTime,
                        syncStatus: syncStatus,
                        watchAppInstalled: watchAppInstalled
                    ) {
                        if !watchAppInstalled {
                            showingWatchSetup = true
                        } else {
                            syncWithWatch()
                        }
                    }
                    
                    // Time Range Selector
                    TimeRangeSelector(selectedRange: $selectedTimeRange)
                    
                    // Real-Time Pain Monitoring
                    RealTimePainCard(painData: realTimePainData)
                    
                    // Health Metrics Integration
                    HealthMetricsCard(
                        heartRateData: heartRateData,
                        movementData: movementData,
                        timeRange: selectedTimeRange
                    )
                    
                    // Watch Features Configuration
                    WatchFeaturesCard(
                        enabledFeatures: $enabledFeatures,
                        isWatchConnected: isWatchConnected
                    )
                    
                    // AI Insights from Watch Data
                    WatchAIInsightsCard()
                    
                    // Emergency Features
                    EmergencyFeaturesCard()
                }
                .padding()
            }
            .navigationTitle("Apple Watch")
            .navigationBarTitleDisplayMode(.large)
            .navigationBarItems(
                trailing: Menu {
                    Button("Sync Now", action: syncWithWatch)
                    Button("Watch Settings") { showingWatchSetup = true }
                    Button("Export Data", action: exportWatchData)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            )
        }
        .onAppear {
            setupWatchConnection()
            loadWatchData()
        }
        .onChange(of: selectedTimeRange) { _ in
            loadWatchData()
        }
        .sheet(isPresented: $showingWatchSetup) {
            WatchSetupView(watchManager: watchManager)
        }
    }
    
    private func setupWatchConnection() {
        watchManager.delegate = self
        watchManager.startSession()
        
        // Check connection status
        isWatchConnected = watchManager.isWatchConnected
        watchAppInstalled = watchManager.isWatchAppInstalled
        
        // Setup health data monitoring
        healthManager.requestAuthorization { success in
            if success {
                startHealthDataMonitoring()
            }
        }
    }
    
    private func startHealthDataMonitoring() {
        // Start real-time health monitoring
        healthManager.startHeartRateMonitoring { readings in
            DispatchQueue.main.async {
                self.heartRateData = readings
            }
        }
        
        healthManager.startMovementMonitoring { readings in
            DispatchQueue.main.async {
                self.movementData = readings
            }
        }
    }
    
    private func loadWatchData() {
        let range = selectedTimeRange.dateRange
        
        // Load pain data from watch
        watchManager.loadPainData(from: range.start, to: range.end) { data in
            DispatchQueue.main.async {
                self.realTimePainData = data
            }
        }
        
        // Load health metrics
        healthManager.loadHeartRateData(from: range.start, to: range.end) { data in
            DispatchQueue.main.async {
                self.heartRateData = data
            }
        }
        
        healthManager.loadMovementData(from: range.start, to: range.end) { data in
            DispatchQueue.main.async {
                self.movementData = data
            }
        }
    }
    
    private func syncWithWatch() {
        syncStatus = .syncing
        
        watchManager.syncData { result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self.syncStatus = .success
                    self.lastSyncTime = Date()
                    self.loadWatchData()
                case .failure(let error):
                    self.syncStatus = .failed(error.localizedDescription)
                }
                
                // Reset status after 3 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                    self.syncStatus = .idle
                }
            }
        }
    }
    
    private func exportWatchData() {
        // Export watch data functionality
        let exporter = DataExporter()
        exporter.exportWatchData(
            painData: realTimePainData,
            heartRateData: heartRateData,
            movementData: movementData
        )
    }
}

// MARK: - Watch Connection Card

struct WatchConnectionCard: View {
    let isConnected: Bool
    let batteryLevel: Double
    let lastSync: Date?
    let syncStatus: AppleWatchPainMonitoringView.SyncStatus
    let watchAppInstalled: Bool
    let onAction: () -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Image(systemName: "applewatch")
                    .font(.title2)
                    .foregroundColor(isConnected ? .green : .gray)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Apple Watch")
                        .font(.headline)
                    
                    Text(connectionStatusText)
                        .font(.subheadline)
                        .foregroundColor(isConnected ? .green : .red)
                }
                
                Spacer()
                
                // Sync Status
                VStack(spacing: 4) {
                    Image(systemName: syncStatus.icon)
                        .foregroundColor(syncStatus.color)
                    
                    if case .syncing = syncStatus {
                        ProgressView()
                            .scaleEffect(0.7)
                    }
                }
            }
            
            // Connection Details
            if isConnected {
                VStack(spacing: 12) {
                    // Battery Level
                    HStack {
                        Image(systemName: "battery.100")
                            .foregroundColor(batteryColor)
                        
                        Text("Battery")
                        
                        Spacer()
                        
                        Text("\(Int(batteryLevel * 100))%")
                            .fontWeight(.medium)
                    }
                    
                    // Last Sync
                    if let lastSync = lastSync {
                        HStack {
                            Image(systemName: "arrow.triangle.2.circlepath")
                                .foregroundColor(.blue)
                            
                            Text("Last Sync")
                            
                            Spacer()
                            
                            Text(lastSync, style: .relative)
                                .fontWeight(.medium)
                        }
                    }
                    
                    // Watch App Status
                    HStack {
                        Image(systemName: watchAppInstalled ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundColor(watchAppInstalled ? .green : .orange)
                        
                        Text("InflamAI Watch")
                        
                        Spacer()
                        
                        Text(watchAppInstalled ? "Installed" : "Not Installed")
                            .fontWeight(.medium)
                            .foregroundColor(watchAppInstalled ? .green : .orange)
                    }
                }
                .font(.subheadline)
            }
            
            // Action Button
            Button(action: onAction) {
                HStack {
                    Image(systemName: actionButtonIcon)
                    Text(actionButtonText)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(actionButtonColor)
                )
                .foregroundColor(.white)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
    
    private var connectionStatusText: String {
        if !watchAppInstalled {
            return "App Not Installed"
        }
        return isConnected ? "Connected" : "Disconnected"
    }
    
    private var batteryColor: Color {
        if batteryLevel > 0.5 {
            return .green
        } else if batteryLevel > 0.2 {
            return .orange
        } else {
            return .red
        }
    }
    
    private var actionButtonIcon: String {
        if !watchAppInstalled {
            return "square.and.arrow.down"
        }
        return "arrow.triangle.2.circlepath"
    }
    
    private var actionButtonText: String {
        if !watchAppInstalled {
            return "Install Watch App"
        }
        return "Sync Now"
    }
    
    private var actionButtonColor: Color {
        if !watchAppInstalled {
            return .blue
        }
        return .green
    }
}

// MARK: - Time Range Selector

struct TimeRangeSelector: View {
    @Binding var selectedRange: AppleWatchPainMonitoringView.TimeRange
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Time Range")
                .font(.headline)
            
            HStack(spacing: 12) {
                ForEach(AppleWatchPainMonitoringView.TimeRange.allCases, id: \.self) { range in
                    Button(action: { selectedRange = range }) {
                        Text(range.rawValue)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 20)
                                    .fill(selectedRange == range ? Color.blue : Color(.systemGray5))
                            )
                            .foregroundColor(selectedRange == range ? .white : .primary)
                    }
                }
                
                Spacer()
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

// MARK: - Real-Time Pain Card

struct RealTimePainCard: View {
    let painData: [WatchPainReading]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .font(.title2)
                    .foregroundColor(.red)
                
                Text("Real-Time Pain Monitoring")
                    .font(.headline)
                
                Spacer()
                
                if !painData.isEmpty {
                    Text("\(painData.count) readings")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            if painData.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.title)
                        .foregroundColor(.gray)
                    
                    Text("No pain data available")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Text("Start tracking pain on your Apple Watch")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 20)
            } else {
                // Pain Level Chart
                PainLevelChart(data: painData)
                    .frame(height: 120)
                
                // Current Pain Status
                if let latestReading = painData.last {
                    CurrentPainStatus(reading: latestReading)
                }
                
                // Quick Stats
                PainQuickStats(data: painData)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

struct PainLevelChart: View {
    let data: [WatchPainReading]
    
    var body: some View {
        // Simplified chart implementation
        GeometryReader { geometry in
            Path { path in
                guard !data.isEmpty else { return }
                
                let width = geometry.size.width
                let height = geometry.size.height
                let stepX = width / CGFloat(max(data.count - 1, 1))
                
                for (index, reading) in data.enumerated() {
                    let x = CGFloat(index) * stepX
                    let y = height - (CGFloat(reading.painLevel) / 10.0 * height)
                    
                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(Color.red, lineWidth: 2)
        }
    }
}

struct CurrentPainStatus: View {
    let reading: WatchPainReading
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Current Pain Level")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Text("\(Int(reading.painLevel))/10")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(painLevelColor)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("Last Updated")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Text(reading.timestamp, style: .time)
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
    
    private var painLevelColor: Color {
        switch reading.painLevel {
        case 0...3: return .green
        case 4...6: return .orange
        case 7...10: return .red
        default: return .gray
        }
    }
}

struct PainQuickStats: View {
    let data: [WatchPainReading]
    
    private var averagePain: Double {
        guard !data.isEmpty else { return 0 }
        return data.map { $0.painLevel }.reduce(0, +) / Double(data.count)
    }
    
    private var maxPain: Double {
        data.map { $0.painLevel }.max() ?? 0
    }
    
    private var painTrend: String {
        guard data.count >= 2 else { return "—" }
        let recent = data.suffix(3).map { $0.painLevel }
        let earlier = data.prefix(3).map { $0.painLevel }
        
        let recentAvg = recent.reduce(0, +) / Double(recent.count)
        let earlierAvg = earlier.reduce(0, +) / Double(earlier.count)
        
        if recentAvg > earlierAvg + 0.5 {
            return "↗️ Increasing"
        } else if recentAvg < earlierAvg - 0.5 {
            return "↘️ Decreasing"
        } else {
            return "→ Stable"
        }
    }
    
    var body: some View {
        HStack(spacing: 20) {
            StatItem(title: "Average", value: String(format: "%.1f", averagePain), color: .blue)
            StatItem(title: "Peak", value: String(format: "%.0f", maxPain), color: .red)
            StatItem(title: "Trend", value: painTrend, color: .purple)
        }
    }
}

struct StatItem: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Health Metrics Card

struct HealthMetricsCard: View {
    let heartRateData: [HeartRateReading]
    let movementData: [MovementReading]
    let timeRange: AppleWatchPainMonitoringView.TimeRange
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "heart.fill")
                    .font(.title2)
                    .foregroundColor(.red)
                
                Text("Health Metrics")
                    .font(.headline)
                
                Spacer()
            }
            
            // Heart Rate Section
            VStack(alignment: .leading, spacing: 8) {
                Text("Heart Rate")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                if heartRateData.isEmpty {
                    Text("No heart rate data available")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    HeartRateMetrics(data: heartRateData)
                }
            }
            
            Divider()
            
            // Movement Section
            VStack(alignment: .leading, spacing: 8) {
                Text("Movement & Activity")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                if movementData.isEmpty {
                    Text("No movement data available")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    MovementMetrics(data: movementData)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

struct HeartRateMetrics: View {
    let data: [HeartRateReading]
    
    private var averageHeartRate: Double {
        guard !data.isEmpty else { return 0 }
        return data.map { $0.beatsPerMinute }.reduce(0, +) / Double(data.count)
    }
    
    private var restingHeartRate: Double {
        data.map { $0.beatsPerMinute }.min() ?? 0
    }
    
    private var maxHeartRate: Double {
        data.map { $0.beatsPerMinute }.max() ?? 0
    }
    
    var body: some View {
        HStack(spacing: 15) {
            MetricBox(title: "Average", value: "\(Int(averageHeartRate))", unit: "BPM", color: .red)
            MetricBox(title: "Resting", value: "\(Int(restingHeartRate))", unit: "BPM", color: .green)
            MetricBox(title: "Peak", value: "\(Int(maxHeartRate))", unit: "BPM", color: .orange)
        }
    }
}

struct MovementMetrics: View {
    let data: [MovementReading]
    
    private var totalSteps: Int {
        data.map { $0.stepCount }.reduce(0, +)
    }
    
    private var averageActivity: Double {
        guard !data.isEmpty else { return 0 }
        return data.map { $0.activityLevel }.reduce(0, +) / Double(data.count)
    }
    
    var body: some View {
        HStack(spacing: 15) {
            MetricBox(title: "Steps", value: "\(totalSteps)", unit: "", color: .blue)
            MetricBox(title: "Activity", value: String(format: "%.1f", averageActivity), unit: "/10", color: .purple)
            MetricBox(title: "Status", value: activityStatus, unit: "", color: activityColor)
        }
    }
    
    private var activityStatus: String {
        switch averageActivity {
        case 0...3: return "Low"
        case 4...6: return "Moderate"
        case 7...10: return "High"
        default: return "—"
        }
    }
    
    private var activityColor: Color {
        switch averageActivity {
        case 0...3: return .red
        case 4...6: return .orange
        case 7...10: return .green
        default: return .gray
        }
    }
}

struct MetricBox: View {
    let title: String
    let value: String
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            
            HStack(alignment: .bottom, spacing: 2) {
                Text(value)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                
                if !unit.isEmpty {
                    Text(unit)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6))
        )
    }
}

// MARK: - Watch Features Card

struct WatchFeaturesCard: View {
    @Binding var enabledFeatures: Set<WatchFeature>
    let isWatchConnected: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "gearshape.2.fill")
                    .font(.title2)
                    .foregroundColor(.blue)
                
                Text("Watch Features")
                    .font(.headline)
                
                Spacer()
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                ForEach(WatchFeature.allCases, id: \.self) { feature in
                    WatchFeatureToggle(
                        feature: feature,
                        isEnabled: enabledFeatures.contains(feature),
                        isWatchConnected: isWatchConnected
                    ) { enabled in
                        if enabled {
                            enabledFeatures.insert(feature)
                        } else {
                            enabledFeatures.remove(feature)
                        }
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

struct WatchFeatureToggle: View {
    let feature: WatchFeature
    let isEnabled: Bool
    let isWatchConnected: Bool
    let onToggle: (Bool) -> Void
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: feature.icon)
                .font(.title3)
                .foregroundColor(isEnabled ? .blue : .gray)
            
            Text(feature.displayName)
                .font(.caption)
                .fontWeight(.medium)
                .multilineTextAlignment(.center)
            
            Toggle("", isOn: Binding(
                get: { isEnabled },
                set: onToggle
            ))
            .labelsHidden()
            .disabled(!isWatchConnected)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isEnabled ? Color.blue.opacity(0.1) : Color(.systemGray6))
                .stroke(isEnabled ? Color.blue : Color.clear, lineWidth: 1)
        )
    }
}

// MARK: - Watch AI Insights Card

struct WatchAIInsightsCard: View {
    @StateObject private var aiEngine = AIMLEngine.shared
    @State private var insights: [WatchAIInsight] = []
    @State private var isLoading = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.purple)
                
                Text("AI Insights")
                    .font(.headline)
                
                Spacer()
                
                if isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            
            if insights.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "lightbulb")
                        .font(.title)
                        .foregroundColor(.gray)
                    
                    Text("No insights available yet")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Text("AI will analyze your watch data to provide personalized insights")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 20)
            } else {
                ForEach(insights, id: \.id) { insight in
                    WatchInsightRow(insight: insight)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .onAppear {
            loadAIInsights()
        }
    }
    
    private func loadAIInsights() {
        isLoading = true
        
        // Simulate AI analysis
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.insights = [
                WatchAIInsight(
                    type: .painPattern,
                    title: "Pain Pattern Detected",
                    description: "Your pain levels tend to increase in the afternoon. Consider scheduling rest breaks.",
                    confidence: 0.85,
                    actionable: true
                ),
                WatchAIInsight(
                    type: .heartRateCorrelation,
                    title: "Heart Rate Correlation",
                    description: "Elevated heart rate often precedes pain episodes by 30 minutes.",
                    confidence: 0.72,
                    actionable: true
                )
            ]
            self.isLoading = false
        }
    }
}

struct WatchInsightRow: View {
    let insight: WatchAIInsight
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: insight.type.icon)
                .font(.title3)
                .foregroundColor(insight.type.color)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(insight.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(insight.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                HStack {
                    Text("Confidence: \(Int(insight.confidence * 100))%")
                        .font(.caption2)
                        .foregroundColor(.blue)
                    
                    if insight.actionable {
                        Text("• Actionable")
                            .font(.caption2)
                            .foregroundColor(.green)
                    }
                }
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

// MARK: - Emergency Features Card

struct EmergencyFeaturesCard: View {
    @State private var emergencyContactsEnabled = true
    @State private var fallDetectionEnabled = true
    @State private var severePainAlertsEnabled = true
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundColor(.red)
                
                Text("Emergency Features")
                    .font(.headline)
                
                Spacer()
            }
            
            VStack(spacing: 12) {
                EmergencyFeatureRow(
                    icon: "phone.fill",
                    title: "Emergency Contacts",
                    description: "Auto-notify contacts during severe pain episodes",
                    isEnabled: $emergencyContactsEnabled
                )
                
                EmergencyFeatureRow(
                    icon: "figure.fall",
                    title: "Fall Detection",
                    description: "Detect falls and send automatic alerts",
                    isEnabled: $fallDetectionEnabled
                )
                
                EmergencyFeatureRow(
                    icon: "bell.fill",
                    title: "Severe Pain Alerts",
                    description: "Alert when pain levels exceed threshold",
                    isEnabled: $severePainAlertsEnabled
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

struct EmergencyFeatureRow: View {
    let icon: String
    let title: String
    let description: String
    @Binding var isEnabled: Bool
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(.red)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Toggle("", isOn: $isEnabled)
                .labelsHidden()
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
    }
}

// MARK: - Watch Setup View

struct WatchSetupView: View {
    let watchManager: AppleWatchManager
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 16) {
                    Image(systemName: "applewatch")
                        .font(.system(size: 60))
                        .foregroundColor(.blue)
                    
                    Text("Set Up Apple Watch")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    Text("Connect your Apple Watch to enable real-time pain monitoring and health tracking")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                // Setup Steps
                VStack(spacing: 20) {
                    SetupStepView(
                        number: 1,
                        title: "Install InflamAI on Apple Watch",
                        description: "Open the Watch app and install InflamAI",
                        isCompleted: watchManager.isWatchAppInstalled
                    )
                    
                    SetupStepView(
                        number: 2,
                        title: "Enable Health Permissions",
                        description: "Allow access to heart rate and activity data",
                        isCompleted: false
                    )
                    
                    SetupStepView(
                        number: 3,
                        title: "Configure Notifications",
                        description: "Set up pain tracking reminders and alerts",
                        isCompleted: false
                    )
                }
                
                Spacer()
                
                // Action Buttons
                VStack(spacing: 12) {
                    Button("Open Watch App") {
                        // Open Watch app
                        if let url = URL(string: "watch://") {
                            UIApplication.shared.open(url)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                    
                    Button("Skip for Now") {
                        presentationMode.wrappedValue.dismiss()
                    }
                    .foregroundColor(.secondary)
                }
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
    }
}

struct SetupStepView: View {
    let number: Int
    let title: String
    let description: String
    let isCompleted: Bool
    
    var body: some View {
        HStack(spacing: 16) {
            // Step number/checkmark
            ZStack {
                Circle()
                    .fill(isCompleted ? Color.green : Color.blue)
                    .frame(width: 30, height: 30)
                
                if isCompleted {
                    Image(systemName: "checkmark")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                } else {
                    Text("\(number)")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
            }
            
            // Step content
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
    }
}

// MARK: - AppleWatchPainMonitoringView Extension

extension AppleWatchPainMonitoringView: AppleWatchManagerDelegate {
    func watchConnectionDidChange(_ isConnected: Bool) {
        DispatchQueue.main.async {
            self.isWatchConnected = isConnected
        }
    }
    
    func watchBatteryLevelDidUpdate(_ level: Double) {
        DispatchQueue.main.async {
            self.watchBatteryLevel = level
        }
    }
    
    func watchAppInstallationDidChange(_ isInstalled: Bool) {
        DispatchQueue.main.async {
            self.watchAppInstalled = isInstalled
        }
    }
    
    func didReceivePainData(_ data: [WatchPainReading]) {
        DispatchQueue.main.async {
            self.realTimePainData = data
        }
    }
}

// MARK: - Data Models

struct WatchPainReading {
    let id = UUID()
    let painLevel: Double
    let bodyRegion: BodyRegion?
    let timestamp: Date
    let heartRate: Double?
    let activityLevel: Double?
}

struct HeartRateReading {
    let id = UUID()
    let beatsPerMinute: Double
    let timestamp: Date
}

struct MovementReading {
    let id = UUID()
    let stepCount: Int
    let activityLevel: Double
    let timestamp: Date
}

struct WatchAIInsight {
    let id = UUID()
    let type: InsightType
    let title: String
    let description: String
    let confidence: Double
    let actionable: Bool
    
    enum InsightType {
        case painPattern
        case heartRateCorrelation
        case activityCorrelation
        case medicationEffectiveness
        
        var icon: String {
            switch self {
            case .painPattern: return "waveform.path.ecg"
            case .heartRateCorrelation: return "heart.fill"
            case .activityCorrelation: return "figure.walk"
            case .medicationEffectiveness: return "pills.fill"
            }
        }
        
        var color: Color {
            switch self {
            case .painPattern: return .red
            case .heartRateCorrelation: return .pink
            case .activityCorrelation: return .blue
            case .medicationEffectiveness: return .green
            }
        }
    }
}

enum WatchFeature: String, CaseIterable {
    case painTracking = "Pain Tracking"
    case heartRateMonitoring = "Heart Rate"
    case activityTracking = "Activity"
    case medicationReminders = "Medication"
    case fallDetection = "Fall Detection"
    case emergencyAlerts = "Emergency"
    
    var displayName: String {
        return self.rawValue
    }
    
    var icon: String {
        switch self {
        case .painTracking: return "waveform.path.ecg"
        case .heartRateMonitoring: return "heart.fill"
        case .activityTracking: return "figure.walk"
        case .medicationReminders: return "pills.fill"
        case .fallDetection: return "figure.fall"
        case .emergencyAlerts: return "exclamationmark.triangle.fill"
        }
    }
}

// MARK: - Apple Watch Manager

class AppleWatchManager: NSObject, ObservableObject {
    static let shared = AppleWatchManager()
    
    @Published var isWatchConnected = false
    @Published var isWatchAppInstalled = false
    
    private var session: WCSession?
    weak var delegate: AppleWatchManagerDelegate?
    
    override init() {
        super.init()
        
        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
        }
    }
    
    func startSession() {
        session?.activate()
    }
    
    func syncData(completion: @escaping (Result<Void, Error>) -> Void) {
        guard let session = session, session.isReachable else {
            completion(.failure(WatchError.notReachable))
            return
        }
        
        let message = ["action": "sync_data"]
        session.sendMessage(message, replyHandler: { _ in
            completion(.success(()))
        }, errorHandler: { error in
            completion(.failure(error))
        })
    }
    
    func loadPainData(from startDate: Date, to endDate: Date, completion: @escaping ([WatchPainReading]) -> Void) {
        // Load pain data from watch or local storage
        // This is a placeholder implementation
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            let sampleData = self.generateSamplePainData(from: startDate, to: endDate)
            completion(sampleData)
        }
    }
    
    private func generateSamplePainData(from startDate: Date, to endDate: Date) -> [WatchPainReading] {
        var data: [WatchPainReading] = []
        let calendar = Calendar.current
        var currentDate = startDate
        
        while currentDate <= endDate {
            let painLevel = Double.random(in: 1...8)
            let heartRate = Double.random(in: 60...100)
            let activityLevel = Double.random(in: 1...10)
            
            let reading = WatchPainReading(
                painLevel: painLevel,
                bodyRegion: BodyRegion.allCases.randomElement(),
                timestamp: currentDate,
                heartRate: heartRate,
                activityLevel: activityLevel
            )
            
            data.append(reading)
            currentDate = calendar.date(byAdding: .hour, value: 2, to: currentDate) ?? currentDate
        }
        
        return data
    }
}

// MARK: - WCSessionDelegate

extension AppleWatchManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.isWatchConnected = session.isPaired && session.isWatchAppInstalled
            self.isWatchAppInstalled = session.isWatchAppInstalled
            
            self.delegate?.watchConnectionDidChange(self.isWatchConnected)
            self.delegate?.watchAppInstallationDidChange(self.isWatchAppInstalled)
        }
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        // Handle session becoming inactive
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        // Handle session deactivation
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        // Handle messages from watch
        if let painData = message["painData"] as? [[String: Any]] {
            let readings = painData.compactMap { dict -> WatchPainReading? in
                guard let painLevel = dict["painLevel"] as? Double,
                      let timestamp = dict["timestamp"] as? Date else {
                    return nil
                }
                
                return WatchPainReading(
                    painLevel: painLevel,
                    bodyRegion: nil,
                    timestamp: timestamp,
                    heartRate: dict["heartRate"] as? Double,
                    activityLevel: dict["activityLevel"] as? Double
                )
            }
            
            DispatchQueue.main.async {
                self.delegate?.didReceivePainData(readings)
            }
        }
    }
}

// MARK: - Protocols

protocol AppleWatchManagerDelegate: AnyObject {
    func watchConnectionDidChange(_ isConnected: Bool)
    func watchBatteryLevelDidUpdate(_ level: Double)
    func watchAppInstallationDidChange(_ isInstalled: Bool)
    func didReceivePainData(_ data: [WatchPainReading])
}

// MARK: - Errors

enum WatchError: Error {
    case notReachable
    case notInstalled
    case syncFailed
    
    var localizedDescription: String {
        switch self {
        case .notReachable:
            return "Apple Watch is not reachable"
        case .notInstalled:
            return "InflamAI is not installed on Apple Watch"
        case .syncFailed:
            return "Failed to sync data with Apple Watch"
        }
    }
}