//
//  RealTimeHealthDashboard.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-20.
//

import SwiftUI
import Charts
import HealthKit

struct RealTimeHealthDashboard: View {
    @StateObject private var healthMonitor = RealTimeHealthMonitor.shared
    @State private var selectedTimeRange: TimeRange = .hour
    @State private var selectedMetric: HealthMetric = .heartRate
    @State private var showingAlerts = false
    @State private var isRefreshing = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Connection Status
                    connectionStatusCard
                    
                    // Real-time Vitals
                    vitalsOverviewCard
                    
                    // Live Charts
                    liveChartsSection
                    
                    // Activity Metrics
                    activityMetricsCard
                    
                    // Health Alerts
                    healthAlertsCard
                    
                    // Quick Actions
                    quickActionsCard
                }
                .padding(.horizontal)
            }
            .navigationTitle("Live Health")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Refresh Data") {
                            refreshData()
                        }
                        
                        Button("View All Alerts") {
                            showingAlerts = true
                        }
                        
                        Divider()
                        
                        Button(healthMonitor.isMonitoring ? "Stop Monitoring" : "Start Monitoring") {
                            toggleMonitoring()
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .refreshable {
                await refreshData()
            }
            .sheet(isPresented: $showingAlerts) {
                HealthAlertsView()
            }
        }
        .task {
            if !healthMonitor.isMonitoring {
                await healthMonitor.startRealTimeMonitoring()
            }
        }
    }
    
    // MARK: - Connection Status Card
    
    private var connectionStatusCard: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: connectionStatusIcon)
                    .foregroundColor(connectionStatusColor)
                    .font(.title2)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("HealthKit Connection")
                        .font(.headline)
                    
                    Text(connectionStatusText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                if healthMonitor.isMonitoring {
                    LiveIndicator()
                }
            }
            
            if let lastUpdate = healthMonitor.lastUpdateTime {
                HStack {
                    Text("Last updated: \(lastUpdate, style: .relative) ago")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    // MARK: - Vitals Overview Card
    
    private var vitalsOverviewCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Vital Signs")
                    .font(.headline)
                
                Spacer()
                
                Button("Details") {
                    // Navigate to detailed vitals view
                }
                .font(.caption)
                .foregroundColor(.blue)
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                VitalSignCard(
                    title: "Heart Rate",
                    value: "\(Int(healthMonitor.currentVitals.heartRate))",
                    unit: "BPM",
                    icon: "heart.fill",
                    color: .red,
                    isNormal: isHeartRateNormal
                )
                
                VitalSignCard(
                    title: "Blood Pressure",
                    value: "\(Int(healthMonitor.currentVitals.bloodPressure.systolic))/\(Int(healthMonitor.currentVitals.bloodPressure.diastolic))",
                    unit: "mmHg",
                    icon: "drop.fill",
                    color: .blue,
                    isNormal: isBloodPressureNormal
                )
                
                VitalSignCard(
                    title: "Oxygen Sat",
                    value: "\(Int(healthMonitor.currentVitals.oxygenSaturation))",
                    unit: "%",
                    icon: "lungs.fill",
                    color: .cyan,
                    isNormal: isOxygenSaturationNormal
                )
                
                VitalSignCard(
                    title: "Temperature",
                    value: String(format: "%.1f", healthMonitor.currentVitals.bodyTemperature),
                    unit: "°F",
                    icon: "thermometer",
                    color: .orange,
                    isNormal: isTemperatureNormal
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    // MARK: - Live Charts Section
    
    private var liveChartsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Live Monitoring")
                    .font(.headline)
                
                Spacer()
                
                Picker("Metric", selection: $selectedMetric) {
                    ForEach(HealthMetric.allCases, id: \.self) { metric in
                        Text(metric.displayName)
                            .tag(metric)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }
            
            // Time Range Selector
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.displayName)
                        .tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            
            // Live Chart
            LiveHealthChart(
                metric: selectedMetric,
                timeRange: selectedTimeRange
            )
            .frame(height: 200)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    // MARK: - Activity Metrics Card
    
    private var activityMetricsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Today's Activity")
                .font(.headline)
            
            HStack(spacing: 20) {
                ActivityMetricView(
                    title: "Steps",
                    value: "\(healthMonitor.realtimeMetrics.dailySteps)",
                    goal: "10,000",
                    progress: Double(healthMonitor.realtimeMetrics.dailySteps) / 10000.0,
                    color: .green
                )
                
                ActivityMetricView(
                    title: "Calories",
                    value: "\(Int(healthMonitor.realtimeMetrics.caloriesBurned))",
                    goal: "2,000",
                    progress: healthMonitor.realtimeMetrics.caloriesBurned / 2000.0,
                    color: .orange
                )
                
                ActivityMetricView(
                    title: "Active Min",
                    value: "\(healthMonitor.realtimeMetrics.activeMinutes)",
                    goal: "30",
                    progress: Double(healthMonitor.realtimeMetrics.activeMinutes) / 30.0,
                    color: .blue
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    // MARK: - Health Alerts Card
    
    private var healthAlertsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Health Alerts")
                    .font(.headline)
                
                Spacer()
                
                if !healthMonitor.healthAlerts.isEmpty {
                    Button("View All") {
                        showingAlerts = true
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
            }
            
            if healthMonitor.healthAlerts.isEmpty {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    
                    Text("No active alerts")
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
                .padding(.vertical, 8)
            } else {
                ForEach(healthMonitor.healthAlerts.prefix(3), id: \.id) { alert in
                    HealthAlertRow(alert: alert)
                }
                
                if healthMonitor.healthAlerts.count > 3 {
                    Button("View \(healthMonitor.healthAlerts.count - 3) more alerts") {
                        showingAlerts = true
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                    .padding(.top, 4)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    // MARK: - Quick Actions Card
    
    private var quickActionsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick Actions")
                .font(.headline)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                QuickActionButton(
                    title: "Record Symptoms",
                    icon: "plus.circle.fill",
                    color: .blue
                ) {
                    // Navigate to symptom recording
                }
                
                QuickActionButton(
                    title: "Take Medication",
                    icon: "pills.fill",
                    color: .green
                ) {
                    // Navigate to medication tracking
                }
                
                QuickActionButton(
                    title: "Export Data",
                    icon: "square.and.arrow.up",
                    color: .orange
                ) {
                    // Export health data
                }
                
                QuickActionButton(
                    title: "Settings",
                    icon: "gear",
                    color: .gray
                ) {
                    // Navigate to settings
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
    
    // MARK: - Computed Properties
    
    private var connectionStatusIcon: String {
        switch healthMonitor.connectionStatus {
        case .connected:
            return "checkmark.circle.fill"
        case .connecting:
            return "arrow.clockwise.circle"
        case .disconnected:
            return "xmark.circle.fill"
        case .error:
            return "exclamationmark.triangle.fill"
        }
    }
    
    private var connectionStatusColor: Color {
        switch healthMonitor.connectionStatus {
        case .connected:
            return .green
        case .connecting:
            return .orange
        case .disconnected:
            return .red
        case .error:
            return .red
        }
    }
    
    private var connectionStatusText: String {
        switch healthMonitor.connectionStatus {
        case .connected:
            return "Connected and monitoring"
        case .connecting:
            return "Connecting to HealthKit..."
        case .disconnected:
            return "Not connected"
        case .error(let message):
            return "Error: \(message)"
        }
    }
    
    private var isHeartRateNormal: Bool {
        let hr = healthMonitor.currentVitals.heartRate
        return hr >= 60 && hr <= 100
    }
    
    private var isBloodPressureNormal: Bool {
        let systolic = healthMonitor.currentVitals.bloodPressure.systolic
        let diastolic = healthMonitor.currentVitals.bloodPressure.diastolic
        return systolic < 120 && diastolic < 80
    }
    
    private var isOxygenSaturationNormal: Bool {
        return healthMonitor.currentVitals.oxygenSaturation >= 95
    }
    
    private var isTemperatureNormal: Bool {
        let temp = healthMonitor.currentVitals.bodyTemperature
        return temp >= 97.0 && temp <= 99.5
    }
    
    // MARK: - Methods
    
    private func refreshData() {
        Task {
            isRefreshing = true
            await healthMonitor.refreshVitals()
            isRefreshing = false
        }
    }
    
    private func toggleMonitoring() {
        Task {
            if healthMonitor.isMonitoring {
                healthMonitor.stopRealTimeMonitoring()
            } else {
                await healthMonitor.startRealTimeMonitoring()
            }
        }
    }
}

// MARK: - Supporting Views

struct LiveIndicator: View {
    @State private var isAnimating = false
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Color.red)
                .frame(width: 8, height: 8)
                .opacity(isAnimating ? 1.0 : 0.3)
                .animation(
                    Animation.easeInOut(duration: 1.0).repeatForever(autoreverses: true),
                    value: isAnimating
                )
            
            Text("LIVE")
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(.red)
        }
        .onAppear {
            isAnimating = true
        }
    }
}

struct VitalSignCard: View {
    let title: String
    let value: String
    let unit: String
    let icon: String
    let color: Color
    let isNormal: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title3)
                
                Spacer()
                
                Image(systemName: isNormal ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundColor(isNormal ? .green : .orange)
                    .font(.caption)
            }
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

struct ActivityMetricView: View {
    let title: String
    let value: String
    let goal: String
    let progress: Double
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            ZStack {
                Circle()
                    .stroke(color.opacity(0.2), lineWidth: 4)
                
                Circle()
                    .trim(from: 0, to: min(progress, 1.0))
                    .stroke(color, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                
                VStack(spacing: 2) {
                    Text(value)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("/\(goal)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 60, height: 60)
        }
    }
}

struct HealthAlertRow: View {
    let alert: HealthAlert
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: alertIcon)
                .foregroundColor(alertColor)
                .font(.title3)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(alert.message)
                    .font(.subheadline)
                    .lineLimit(2)
                
                Text(alert.timestamp, style: .relative)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if !alert.isRead {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 8, height: 8)
            }
        }
        .padding(.vertical, 4)
    }
    
    private var alertIcon: String {
        switch alert.severity {
        case .low:
            return "info.circle.fill"
        case .medium:
            return "exclamationmark.triangle.fill"
        case .high:
            return "exclamationmark.octagon.fill"
        }
    }
    
    private var alertColor: Color {
        switch alert.severity {
        case .low:
            return .blue
        case .medium:
            return .orange
        case .high:
            return .red
        }
    }
}

struct QuickActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct LiveHealthChart: View {
    let metric: HealthMetric
    let timeRange: TimeRange
    
    @State private var chartData: [HealthDataPoint] = []
    
    var body: some View {
        Chart(chartData, id: \.timestamp) { dataPoint in
            LineMark(
                x: .value("Time", dataPoint.timestamp),
                y: .value(metric.displayName, dataPoint.value)
            )
            .foregroundStyle(metric.color)
            .lineStyle(StrokeStyle(lineWidth: 2))
            
            AreaMark(
                x: .value("Time", dataPoint.timestamp),
                y: .value(metric.displayName, dataPoint.value)
            )
            .foregroundStyle(metric.color.opacity(0.1))
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXAxis {
            AxisMarks(values: .stride(by: timeRange.axisStride)) { _ in
                AxisGridLine()
                AxisValueLabel(format: timeRange.axisFormat)
            }
        }
        .onAppear {
            generateMockData()
        }
        .onChange(of: metric) { _ in
            generateMockData()
        }
        .onChange(of: timeRange) { _ in
            generateMockData()
        }
    }
    
    private func generateMockData() {
        let now = Date()
        let interval = timeRange.dataInterval
        let count = timeRange.dataPointCount
        
        chartData = (0..<count).map { index in
            let timestamp = now.addingTimeInterval(-Double(count - index) * interval)
            let baseValue = metric.baseValue
            let variation = metric.variation
            let value = baseValue + Double.random(in: -variation...variation)
            
            return HealthDataPoint(timestamp: timestamp, value: value)
        }
    }
}

struct HealthAlertsView: View {
    @StateObject private var healthMonitor = RealTimeHealthMonitor.shared
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List {
                ForEach(healthMonitor.healthAlerts, id: \.id) { alert in
                    HealthAlertDetailRow(alert: alert)
                }
            }
            .navigationTitle("Health Alerts")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct HealthAlertDetailRow: View {
    let alert: HealthAlert
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: alertIcon)
                    .foregroundColor(alertColor)
                
                Text(alert.message)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                if !alert.isRead {
                    Circle()
                        .fill(Color.blue)
                        .frame(width: 8, height: 8)
                }
            }
            
            HStack {
                Text(alert.timestamp, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text("•")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(alert.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(alert.severity.displayName)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(alertColor.opacity(0.2))
                    .foregroundColor(alertColor)
                    .cornerRadius(4)
            }
        }
        .padding(.vertical, 4)
    }
    
    private var alertIcon: String {
        switch alert.severity {
        case .low:
            return "info.circle.fill"
        case .medium:
            return "exclamationmark.triangle.fill"
        case .high:
            return "exclamationmark.octagon.fill"
        }
    }
    
    private var alertColor: Color {
        switch alert.severity {
        case .low:
            return .blue
        case .medium:
            return .orange
        case .high:
            return .red
        }
    }
}

// MARK: - Supporting Types

struct HealthDataPoint {
    let timestamp: Date
    let value: Double
}

enum HealthMetric: CaseIterable {
    case heartRate
    case bloodPressure
    case oxygenSaturation
    case temperature
    case steps
    case calories
    
    var displayName: String {
        switch self {
        case .heartRate: return "Heart Rate"
        case .bloodPressure: return "Blood Pressure"
        case .oxygenSaturation: return "Oxygen Saturation"
        case .temperature: return "Temperature"
        case .steps: return "Steps"
        case .calories: return "Calories"
        }
    }
    
    var color: Color {
        switch self {
        case .heartRate: return .red
        case .bloodPressure: return .blue
        case .oxygenSaturation: return .cyan
        case .temperature: return .orange
        case .steps: return .green
        case .calories: return .purple
        }
    }
    
    var baseValue: Double {
        switch self {
        case .heartRate: return 75
        case .bloodPressure: return 120
        case .oxygenSaturation: return 98
        case .temperature: return 98.6
        case .steps: return 500
        case .calories: return 100
        }
    }
    
    var variation: Double {
        switch self {
        case .heartRate: return 15
        case .bloodPressure: return 20
        case .oxygenSaturation: return 2
        case .temperature: return 1
        case .steps: return 200
        case .calories: return 50
        }
    }
}

enum TimeRange: CaseIterable {
    case hour
    case day
    case week
    
    var displayName: String {
        switch self {
        case .hour: return "1H"
        case .day: return "1D"
        case .week: return "1W"
        }
    }
    
    var dataInterval: TimeInterval {
        switch self {
        case .hour: return 60 // 1 minute
        case .day: return 3600 // 1 hour
        case .week: return 86400 // 1 day
        }
    }
    
    var dataPointCount: Int {
        switch self {
        case .hour: return 60
        case .day: return 24
        case .week: return 7
        }
    }
    
    var axisStride: Calendar.Component {
        switch self {
        case .hour: return .minute
        case .day: return .hour
        case .week: return .day
        }
    }
    
    var axisFormat: Date.FormatStyle {
        switch self {
        case .hour: return .dateTime.minute()
        case .day: return .dateTime.hour()
        case .week: return .dateTime.weekday(.abbreviated)
        }
    }
}

extension HealthAlertSeverity {
    var displayName: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }
}

#Preview {
    RealTimeHealthDashboard()
}