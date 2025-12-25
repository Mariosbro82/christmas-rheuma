//
//  WeatherNotificationSettingsView.swift
//  InflamAI-Swift
//
//  User interface for configuring weather-based flare alerts
//  Allows customization of thresholds and advance warning times
//

import SwiftUI

struct WeatherNotificationSettingsView: View {
    @AppStorage("weatherNotificationsEnabled") private var notificationsEnabled = false
    @AppStorage("weatherNotificationThreshold") private var pressureThreshold = 10.0
    @AppStorage("weatherNotificationAdvanceHours") private var advanceHours = 6

    @State private var showingPermissionAlert = false
    @State private var showingTestSuccess = false

    private let notificationService = WeatherNotificationService.shared

    var body: some View {
        Form {
            // Enable/Disable Section
            Section {
                Toggle("Enable Weather Alerts", isOn: $notificationsEnabled)
                    .onChange(of: notificationsEnabled) { newValue in
                        handleToggleChange(newValue)
                    }

                if notificationsEnabled {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.caption)

                        Text("Alerts are active")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            } header: {
                Text("Weather Flare Alerts")
            } footer: {
                Text("Receive notifications when weather conditions may trigger symptoms based on barometric pressure changes.")
            }

            if notificationsEnabled {
                // Threshold Configuration
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Pressure Drop Threshold")
                                .font(.subheadline)

                            Spacer()

                            Text("\(Int(pressureThreshold)) hPa")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.blue)
                        }

                        Slider(value: $pressureThreshold, in: 5...20, step: 1)
                            .onChange(of: pressureThreshold) { _ in
                                updateSettings()
                            }

                        Text("Alert me when pressure drops by \(Int(pressureThreshold)) hPa or more within 12 hours")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("Sensitivity")
                } footer: {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("• Lower values (5-8 hPa): More sensitive, more alerts")
                        Text("• Medium values (9-12 hPa): Balanced sensitivity")
                        Text("• Higher values (13-20 hPa): Less sensitive, fewer alerts")
                    }
                    .font(.caption)
                }

                // Advance Warning Configuration
                Section {
                    Picker("Advance Warning", selection: $advanceHours) {
                        Text("3 hours").tag(3)
                        Text("6 hours").tag(6)
                        Text("12 hours").tag(12)
                        Text("24 hours").tag(24)
                    }
                    .onChange(of: advanceHours) { _ in
                        updateSettings()
                    }
                } header: {
                    Text("Timing")
                } footer: {
                    Text("How far in advance to receive alerts before pressure drops begin")
                }

                // Test Notification
                Section {
                    Button {
                        sendTestNotification()
                    } label: {
                        HStack {
                            Image(systemName: "bell.badge")
                                .foregroundColor(.blue)

                            Text("Send Test Notification")
                                .foregroundColor(.primary)

                            Spacer()

                            if showingTestSuccess {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .transition(.scale.combined(with: .opacity))
                            }
                        }
                    }
                } footer: {
                    Text("Verify that notifications are working correctly")
                }

                // Info Section
                Section {
                    InfoRow(
                        icon: "clock",
                        title: "Background Updates",
                        description: "Weather data is checked every 12 hours automatically"
                    )

                    InfoRow(
                        icon: "location",
                        title: "Location Access",
                        description: "Required to get accurate local weather data"
                    )

                    InfoRow(
                        icon: "battery.100",
                        title: "Battery Impact",
                        description: "Minimal battery usage with efficient background checks"
                    )
                } header: {
                    Text("How It Works")
                }
            }

            // Personalization Tips
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 12) {
                        Image(systemName: "lightbulb.fill")
                            .foregroundColor(.yellow)
                            .font(.title3)

                        Text("Personalization Tip")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                    }

                    Text("The app learns your personal weather sensitivity over time. After 30+ days of tracking, your alert threshold will be automatically adjusted based on your individual patterns.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle("Weather Alerts")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Notification Permission Required", isPresented: $showingPermissionAlert) {
            Button("Open Settings", role: .none) {
                openAppSettings()
            }
            Button("Cancel", role: .cancel) {
                notificationsEnabled = false
            }
        } message: {
            Text("Please enable notifications in Settings to receive weather alerts.")
        }
    }

    // MARK: - Private Methods

    private func handleToggleChange(_ enabled: Bool) {
        if enabled {
            Task {
                do {
                    try await notificationService.requestPermissions()
                    updateSettings()
                } catch {
                    // Permission denied
                    await MainActor.run {
                        showingPermissionAlert = true
                    }
                }
            }
        } else {
            updateSettings()
        }
    }

    private func updateSettings() {
        notificationService.updateSettings(
            enabled: notificationsEnabled,
            threshold: pressureThreshold,
            advanceHours: advanceHours
        )
    }

    private func sendTestNotification() {
        Task {
            await notificationService.sendTestNotification()

            // Show checkmark briefly
            await MainActor.run {
                withAnimation {
                    showingTestSuccess = true
                }
            }

            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

            await MainActor.run {
                withAnimation {
                    showingTestSuccess = false
                }
            }
        }
    }

    private func openAppSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(url)
        }
    }
}

// MARK: - Supporting Views

struct InfoRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .font(.title3)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Preview

struct WeatherNotificationSettingsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            WeatherNotificationSettingsView()
        }
    }
}
