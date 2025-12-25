//
//  AppleWatchView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import WatchConnectivity

// MARK: - Main Apple Watch View

struct AppleWatchView: View {
    @StateObject private var watchManager = AppleWatchManager.shared
    @StateObject private var syncManager = WatchDataSyncManager()
    @StateObject private var settingsManager = WatchSettingsManager()
    @State private var selectedTab = 0
    
    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from SettingsView,
        // which is already wrapped in NavigationView in MainTabView.
        VStack(spacing: 0) {
            // Header
            WatchConnectionHeader()

            // Tab View
            TabView(selection: $selectedTab) {
                WatchOverviewView()
                    .tabItem {
                        Image(systemName: "heart.fill")
                        Text("Overview")
                    }
                    .tag(0)

                WatchHealthDataView()
                    .tabItem {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("Health")
                    }
                    .tag(1)

                WatchComplicationsView()
                    .tabItem {
                        Image(systemName: "apps.iphone")
                        Text("Complications")
                    }
                    .tag(2)

                WatchSyncView()
                    .tabItem {
                        Image(systemName: "arrow.triangle.2.circlepath")
                        Text("Sync")
                    }
                    .tag(3)

                WatchSettingsView()
                    .tabItem {
                        Image(systemName: "gear")
                        Text("Settings")
                    }
                    .tag(4)
            }
        }
        .navigationTitle("Apple Watch")
        .navigationBarTitleDisplayMode(.large)
        .environmentObject(watchManager)
        .environmentObject(syncManager)
        .environmentObject(settingsManager)
    }
}

// MARK: - Watch Connection Header

struct WatchConnectionHeader: View {
    @EnvironmentObject var watchManager: AppleWatchManager
    
    var body: some View {
        HStack {
            Image(systemName: "applewatch")
                .foregroundColor(watchManager.isWatchConnected ? .green : .gray)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(watchManager.isWatchConnected ? "Connected" : "Disconnected")
                    .font(.headline)
                    .foregroundColor(watchManager.isWatchConnected ? .green : .red)
                
                if watchManager.isWatchConnected {
                    HStack {
                        Image(systemName: "battery.100")
                            .foregroundColor(batteryColor)
                        Text("\(Int(watchManager.watchBatteryLevel * 100))%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            if let lastSync = watchManager.lastSyncDate {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Last Sync")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(lastSync, style: .time)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .padding(.horizontal)
    }
    
    private var batteryColor: Color {
        if watchManager.watchBatteryLevel > 0.5 {
            return .green
        } else if watchManager.watchBatteryLevel > 0.2 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - Watch Overview View

struct WatchOverviewView: View {
    @EnvironmentObject var watchManager: AppleWatchManager
    @State private var showingInstallInstructions = false
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                // Connection Status Card
                ConnectionStatusCard()
                
                // Quick Actions
                QuickActionsCard()
                
                // Recent Activity
                RecentActivityCard()
                
                // Watch App Status
                WatchAppStatusCard(showingInstructions: $showingInstallInstructions)
            }
            .padding()
        }
        .sheet(isPresented: $showingInstallInstructions) {
            WatchAppInstallInstructionsView()
        }
    }
}

struct ConnectionStatusCard: View {
    @EnvironmentObject var watchManager: AppleWatchManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "applewatch.side.right")
                    .foregroundColor(.blue)
                Text("Connection Status")
                    .font(.headline)
                Spacer()
            }
            
            VStack(spacing: 8) {
                StatusRow(title: "Watch Connected", 
                         status: watchManager.isWatchConnected,
                         icon: "checkmark.circle.fill")
                
                StatusRow(title: "App Installed", 
                         status: watchManager.isWatchAppInstalled,
                         icon: "app.badge.checkmark")
                
                if watchManager.isWatchConnected {
                    HStack {
                        Text("Battery Level")
                        Spacer()
                        ProgressView(value: watchManager.watchBatteryLevel)
                            .frame(width: 60)
                        Text("\(Int(watchManager.watchBatteryLevel * 100))%")
                            .font(.caption)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct StatusRow: View {
    let title: String
    let status: Bool
    let icon: String
    
    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Image(systemName: icon)
                .foregroundColor(status ? .green : .red)
            Text(status ? "Yes" : "No")
                .foregroundColor(status ? .green : .red)
                .font(.caption)
        }
    }
}

struct QuickActionsCard: View {
    @EnvironmentObject var watchManager: AppleWatchManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "bolt.fill")
                    .foregroundColor(.orange)
                Text("Quick Actions")
                    .font(.headline)
                Spacer()
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                QuickActionButton(
                    title: "Sync Now",
                    icon: "arrow.triangle.2.circlepath",
                    color: .blue
                ) {
                    watchManager.syncAllData()
                }
                
                QuickActionButton(
                    title: "Send Reminder",
                    icon: "bell.fill",
                    color: .green
                ) {
                    sendTestReminder()
                }
                
                QuickActionButton(
                    title: "Update Complications",
                    icon: "apps.iphone",
                    color: .purple
                ) {
                    watchManager.updateComplications()
                }
                
                QuickActionButton(
                    title: "Request Symptoms",
                    icon: "heart.text.square",
                    color: .red
                ) {
                    watchManager.requestSymptomUpdate()
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func sendTestReminder() {
        let reminder = WatchMedicationReminder(
            id: UUID(),
            medicationName: "Test Medication",
            dosage: "10mg",
            scheduledTime: Date().addingTimeInterval(60),
            isTaken: false,
            reminderType: .daily
        )
        watchManager.sendMedicationReminder(reminder)
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
                    .multilineTextAlignment(.center)
                    .foregroundColor(.primary)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct RecentActivityCard: View {
    @State private var recentActivities: [WatchActivity] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "clock.fill")
                    .foregroundColor(.green)
                Text("Recent Activity")
                    .font(.headline)
                Spacer()
            }
            
            if recentActivities.isEmpty {
                Text("No recent activity")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(recentActivities) { activity in
                    ActivityRow(activity: activity)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
        .onAppear {
            loadRecentActivities()
        }
    }
    
    private func loadRecentActivities() {
        // Mock data - replace with actual implementation
        recentActivities = [
            WatchActivity(id: UUID(), type: "Heart Rate", value: "72 BPM", timestamp: Date().addingTimeInterval(-300)),
            WatchActivity(id: UUID(), type: "Steps", value: "1,234", timestamp: Date().addingTimeInterval(-600)),
            WatchActivity(id: UUID(), type: "Medication", value: "Taken", timestamp: Date().addingTimeInterval(-1800))
        ]
    }
}

struct WatchActivity: Identifiable {
    let id: UUID
    let type: String
    let value: String
    let timestamp: Date
}

struct ActivityRow: View {
    let activity: WatchActivity
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(activity.type)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(activity.value)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(activity.timestamp, style: .time)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

struct WatchAppStatusCard: View {
    @EnvironmentObject var watchManager: AppleWatchManager
    @Binding var showingInstructions: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "app.badge")
                    .foregroundColor(.blue)
                Text("Watch App")
                    .font(.headline)
                Spacer()
            }
            
            if watchManager.isWatchAppInstalled {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("InflamAI is installed on your Apple Watch")
                        .font(.subheadline)
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("App not installed on Apple Watch")
                            .font(.subheadline)
                    }
                    
                    Button("View Installation Instructions") {
                        showingInstructions = true
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

// MARK: - Watch Health Data View

struct WatchHealthDataView: View {
    @State private var healthMetrics: [HealthMetric] = []
    @State private var selectedTimeRange: TimeRange = .today
    
    enum TimeRange: String, CaseIterable {
        case today = "Today"
        case week = "This Week"
        case month = "This Month"
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Time Range Picker
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)
            
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(healthMetrics) { metric in
                        HealthMetricCard(metric: metric)
                    }
                }
                .padding()
            }
        }
        .onAppear {
            loadHealthMetrics()
        }
        .onChange(of: selectedTimeRange) { _ in
            loadHealthMetrics()
        }
    }
    
    private func loadHealthMetrics() {
        // Mock data - replace with actual HealthKit data
        healthMetrics = [
            HealthMetric(id: UUID(), name: "Heart Rate", value: "72", unit: "BPM", icon: "heart.fill", color: .red, trend: .stable),
            HealthMetric(id: UUID(), name: "Steps", value: "8,432", unit: "steps", icon: "figure.walk", color: .green, trend: .up),
            HealthMetric(id: UUID(), name: "Active Calories", value: "245", unit: "cal", icon: "flame.fill", color: .orange, trend: .up),
            HealthMetric(id: UUID(), name: "Stand Hours", value: "6", unit: "hours", icon: "figure.stand", color: .blue, trend: .down)
        ]
    }
}

struct HealthMetric: Identifiable {
    let id: UUID
    let name: String
    let value: String
    let unit: String
    let icon: String
    let color: Color
    let trend: Trend
    
    enum Trend {
        case up, down, stable
        
        var icon: String {
            switch self {
            case .up: return "arrow.up"
            case .down: return "arrow.down"
            case .stable: return "minus"
            }
        }
        
        var color: Color {
            switch self {
            case .up: return .green
            case .down: return .red
            case .stable: return .gray
            }
        }
    }
}

struct HealthMetricCard: View {
    let metric: HealthMetric
    
    var body: some View {
        HStack {
            Image(systemName: metric.icon)
                .foregroundColor(metric.color)
                .font(.title2)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(metric.name)
                    .font(.headline)
                Text("\(metric.value) \(metric.unit)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Image(systemName: metric.trend.icon)
                .foregroundColor(metric.trend.color)
                .font(.caption)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

// MARK: - Watch Complications View

struct WatchComplicationsView: View {
    @EnvironmentObject var watchManager: AppleWatchManager
    @StateObject private var complicationManager = WatchComplicationManager()
    
    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Text("Watch Complications")
                    .font(.title2)
                    .fontWeight(.bold)
                Spacer()
                Button("Update All") {
                    complicationManager.updateAllComplications()
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
                .controlSize(.small)
            }
            .padding(.horizontal)
            
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(complicationManager.activeComplications) { complication in
                        ComplicationCard(complication: complication)
                    }
                    
                    if complicationManager.activeComplications.isEmpty {
                        EmptyComplicationsView()
                    }
                }
                .padding()
            }
        }
        .onAppear {
            complicationManager.updateAllComplications()
        }
    }
}

struct ComplicationCard: View {
    let complication: WatchComplication
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(complication.type.rawValue.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.headline)
                Spacer()
                Text(complication.lastUpdated, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            HStack {
                Text(complication.displayText)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(Color(complication.color))
                
                Spacer()
                
                if let value = complication.value {
                    Text(String(format: "%.1f", value))
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(Color(complication.color))
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct EmptyComplicationsView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "apps.iphone")
                .font(.system(size: 50))
                .foregroundColor(.gray)
            
            Text("No Complications Available")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Complications will appear here when your Apple Watch is connected and the app is installed.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }
}

// MARK: - Watch Sync View

struct WatchSyncView: View {
    @EnvironmentObject var syncManager: WatchDataSyncManager
    @EnvironmentObject var watchManager: AppleWatchManager
    
    var body: some View {
        VStack(spacing: 20) {
            // Sync Status
            SyncStatusCard()
            
            // Sync Controls
            SyncControlsCard()
            
            // Sync History
            SyncHistoryCard()
            
            Spacer()
        }
        .padding()
    }
}

struct SyncStatusCard: View {
    @EnvironmentObject var syncManager: WatchDataSyncManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .foregroundColor(.blue)
                Text("Sync Status")
                    .font(.headline)
                Spacer()
            }
            
            HStack {
                Text("Status:")
                Spacer()
                Text(syncStatusText)
                    .foregroundColor(syncStatusColor)
                    .fontWeight(.medium)
            }
            
            if let lastSync = syncManager.lastSyncDate {
                HStack {
                    Text("Last Sync:")
                    Spacer()
                    Text(lastSync, style: .relative)
                        .foregroundColor(.secondary)
                }
            }
            
            if syncManager.pendingUploads > 0 {
                HStack {
                    Text("Pending:")
                    Spacer()
                    Text("\(syncManager.pendingUploads) items")
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var syncStatusText: String {
        switch syncManager.syncStatus {
        case .idle:
            return "Ready"
        case .syncing:
            return "Syncing..."
        case .completed:
            return "Completed"
        case .failed:
            return "Failed"
        }
    }
    
    private var syncStatusColor: Color {
        switch syncManager.syncStatus {
        case .idle:
            return .blue
        case .syncing:
            return .orange
        case .completed:
            return .green
        case .failed:
            return .red
        }
    }
}

struct SyncControlsCard: View {
    @EnvironmentObject var syncManager: WatchDataSyncManager
    @EnvironmentObject var watchManager: AppleWatchManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "gear")
                    .foregroundColor(.orange)
                Text("Sync Controls")
                    .font(.headline)
                Spacer()
            }
            
            VStack(spacing: 8) {
                Button("Sync All Data") {
                    Task {
                        await syncManager.performFullSync()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
                .disabled(!watchManager.isWatchConnected || syncManager.syncStatus == .syncing)
                
                Button("Force Sync") {
                    watchManager.syncAllData()
                }
                .buttonStyle(.bordered)
                .disabled(!watchManager.isWatchConnected)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct SyncHistoryCard: View {
    @State private var syncHistory: [SyncHistoryItem] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .foregroundColor(.green)
                Text("Recent Syncs")
                    .font(.headline)
                Spacer()
            }
            
            if syncHistory.isEmpty {
                Text("No sync history available")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(syncHistory) { item in
                    SyncHistoryRow(item: item)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
        .onAppear {
            loadSyncHistory()
        }
    }
    
    private func loadSyncHistory() {
        // Mock data - replace with actual implementation
        syncHistory = [
            SyncHistoryItem(id: UUID(), date: Date().addingTimeInterval(-300), status: .completed, itemCount: 15),
            SyncHistoryItem(id: UUID(), date: Date().addingTimeInterval(-1800), status: .completed, itemCount: 8),
            SyncHistoryItem(id: UUID(), date: Date().addingTimeInterval(-3600), status: .failed, itemCount: 0)
        ]
    }
}

struct SyncHistoryItem: Identifiable {
    let id: UUID
    let date: Date
    let status: SyncStatus
    let itemCount: Int
    
    enum SyncStatus {
        case completed, failed
        
        var color: Color {
            switch self {
            case .completed: return .green
            case .failed: return .red
            }
        }
        
        var icon: String {
            switch self {
            case .completed: return "checkmark.circle.fill"
            case .failed: return "xmark.circle.fill"
            }
        }
    }
}

struct SyncHistoryRow: View {
    let item: SyncHistoryItem
    
    var body: some View {
        HStack {
            Image(systemName: item.status.icon)
                .foregroundColor(item.status.color)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(item.date, style: .time)
                    .font(.subheadline)
                if item.status == .completed {
                    Text("\(item.itemCount) items synced")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text("Sync failed")
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }
            
            Spacer()
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Watch Settings View

struct WatchSettingsView: View {
    @EnvironmentObject var settingsManager: WatchSettingsManager
    
    var body: some View {
        Form {
            Section("Health Monitoring") {
                Toggle("Heart Rate Monitoring", isOn: $settingsManager.settings.enableHeartRateMonitoring)
                Toggle("Step Tracking", isOn: $settingsManager.settings.enableStepTracking)
                Toggle("Workout Tracking", isOn: $settingsManager.settings.enableWorkoutTracking)
            }
            
            Section("Notifications") {
                Toggle("Medication Reminders", isOn: $settingsManager.settings.medicationReminderEnabled)
                Toggle("Symptom Reminders", isOn: $settingsManager.settings.symptomReminderEnabled)
                Toggle("Emergency Contacts", isOn: $settingsManager.settings.emergencyContactsEnabled)
            }
            
            Section("Interface") {
                Toggle("Haptic Feedback", isOn: $settingsManager.settings.hapticFeedbackEnabled)
            }
            
            Section("Data Management") {
                Toggle("Auto Sync", isOn: $settingsManager.settings.autoSyncEnabled)
                Toggle("Battery Optimization", isOn: $settingsManager.settings.batteryOptimizationEnabled)
                
                Stepper("Data Retention: \(settingsManager.settings.dataRetentionDays) days", 
                       value: $settingsManager.settings.dataRetentionDays, 
                       in: 7...90, 
                       step: 7)
            }
        }
        .navigationTitle("Watch Settings")
        .onChange(of: settingsManager.settings) { newSettings in
            settingsManager.updateSettings(newSettings)
        }
    }
}

// MARK: - Watch App Install Instructions

struct WatchAppInstallInstructionsView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Installing InflamAI on Apple Watch")
                        .font(.title)
                        .fontWeight(.bold)
                    
                    InstructionStep(
                        number: 1,
                        title: "Open the Watch App",
                        description: "On your iPhone, open the Apple Watch app."
                    )
                    
                    InstructionStep(
                        number: 2,
                        title: "Find InflamAI",
                        description: "Scroll down to find InflamAI in the list of available apps."
                    )
                    
                    InstructionStep(
                        number: 3,
                        title: "Install the App",
                        description: "Tap the 'Install' button next to InflamAI to install it on your Apple Watch."
                    )
                    
                    InstructionStep(
                        number: 4,
                        title: "Wait for Installation",
                        description: "The app will automatically install on your Apple Watch. This may take a few minutes."
                    )
                    
                    InstructionStep(
                        number: 5,
                        title: "Open on Watch",
                        description: "Once installed, you can open InflamAI directly on your Apple Watch from the app grid."
                    )
                    
                    Text("Note: Make sure your Apple Watch is connected and has sufficient battery before installing.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.top)
                }
                .padding()
            }
            .navigationTitle("Installation Guide")
            .navigationBarTitleDisplayMode(.inline)
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

struct InstructionStep: View {
    let number: Int
    let title: String
    let description: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.headline)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .frame(width: 24, height: 24)
                .background(Color.blue)
                .clipShape(Circle())
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

#Preview {
    AppleWatchView()
}