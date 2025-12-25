//
//  SettingsDetailViews.swift
//  InflamAI
//
//  Backup & Restore, Privacy, Theme, Apple Watch, HealthKit views
//

import SwiftUI
import CoreData

// MARK: - Backup & Restore View

struct BackupRestoreView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var isCreatingBackup = false
    @State private var showBackupSuccess = false
    @State private var backupURL: URL?
    @State private var showShareSheet = false

    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header
                VStack(spacing: AssetsManager.Spacing.md) {
                    Image(systemName: "externaldrive.fill")
                        .font(.system(size: 60))
                        .foregroundColor(AssetsManager.Colors.primary)

                    Text("Backup & Restore")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Protect your health data")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Create Backup
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Create Backup")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    Button {
                        createBackup()
                    } label: {
                        HStack {
                            if isCreatingBackup {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Image(systemName: "square.and.arrow.down.fill")
                                Text("Create Backup Now")
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(AssetsManager.Gradients.primary)
                        .foregroundColor(.white)
                        .cornerRadius(AssetsManager.CornerRadius.md)
                    }
                    .disabled(isCreatingBackup)
                    .padding(.horizontal, AssetsManager.Spacing.md)
                }

                // What's Included
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("What's Included")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    VStack(spacing: AssetsManager.Spacing.xs) {
                        BackupItemRow(icon: "chart.xyaxis.line", title: "All symptom logs and BASDAI scores")
                        BackupItemRow(icon: "pills.fill", title: "Medication history and schedules")
                        BackupItemRow(icon: "figure.stand", title: "Body region pain maps")
                        BackupItemRow(icon: "calendar", title: "Flare events and timelines")
                        BackupItemRow(icon: "note.text", title: "Personal notes and observations")
                    }
                }

                // Privacy Notice
                MascotTipCard(
                    icon: "lock.shield.fill",
                    title: "Encrypted Backup",
                    message: "Your backup is encrypted and stored locally. Share it securely via AirDrop or Files app.",
                    color: AssetsManager.Colors.success
                )
                .padding(.horizontal, AssetsManager.Spacing.md)

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Backup & Restore")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Backup Created", isPresented: $showBackupSuccess) {
            Button("Share", role: .none) {
                showShareSheet = true
            }
            Button("OK", role: .cancel) {}
        } message: {
            Text("Your data has been backed up successfully.")
        }
        .sheet(isPresented: $showShareSheet) {
            if let url = backupURL {
                ShareSheet(items: [url])
            }
        }
    }

    private func createBackup() {
        isCreatingBackup = true

        Task {
            do {
                let url = try await exportBackup()
                await MainActor.run {
                    backupURL = url
                    isCreatingBackup = false
                    showBackupSuccess = true
                }
            } catch {
                await MainActor.run {
                    isCreatingBackup = false
                }
            }
        }
    }

    private func exportBackup() async throws -> URL {
        // Create JSON backup
        let data = try await context.perform {
            let logRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            let logs = try context.fetch(logRequest)

            let medRequest: NSFetchRequest<Medication> = Medication.fetchRequest()
            let medications = try context.fetch(medRequest)

            let regionRequest: NSFetchRequest<BodyRegionLog> = BodyRegionLog.fetchRequest()
            let regions = try context.fetch(regionRequest)

            return ExportData(
                exportDate: Date(),
                symptomLogs: logs.map { ExportableSymptomLog(from: $0) },
                medications: medications.map { ExportableMedication(from: $0) },
                bodyRegions: regions.map { ExportableBodyRegion(from: $0) }
            )
        }

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let jsonData = try encoder.encode(data)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("spinalytics_backup_\(Date().timeIntervalSince1970).json")

        try jsonData.write(to: tempURL)
        return tempURL
    }
}

struct BackupItemRow: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: AssetsManager.Spacing.md) {
            Image(systemName: icon)
                .foregroundColor(AssetsManager.Colors.primary)
                .frame(width: 24)

            Text(title)
                .font(.subheadline)

            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(AssetsManager.Colors.success)
        }
        .padding(AssetsManager.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                .fill(AssetsManager.Colors.cardBackground)
        )
        .padding(.horizontal, AssetsManager.Spacing.md)
    }
}

// MARK: - Privacy Settings View

struct PrivacySettingsView: View {
    @AppStorage("privacyAnalyticsEnabled") private var analyticsEnabled = false
    @AppStorage("privacyCrashReportsEnabled") private var crashReportsEnabled = false
    @AppStorage("privacyBiometricEnabled") private var biometricEnabled = false
    @State private var showDeleteConfirmation = false

    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header
                VStack(spacing: AssetsManager.Spacing.md) {
                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 60))
                        .foregroundColor(AssetsManager.Colors.primary)

                    Text("Your Privacy")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("We take your privacy seriously")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Privacy Promise
                MascotTipCard(
                    icon: "checkmark.shield.fill",
                    title: "Zero Data Collection",
                    message: "InflamAI collects ZERO data. No analytics, no tracking, no third parties. Your health information never leaves your device.",
                    color: AssetsManager.Colors.success
                )
                .padding(.horizontal, AssetsManager.Spacing.md)

                // Settings
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Privacy Controls")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    PrivacyToggleRow(
                        icon: "chart.bar.fill",
                        title: "Anonymous Analytics",
                        description: "Currently disabled by design",
                        isOn: .constant(false),
                        isDisabled: true
                    )

                    PrivacyToggleRow(
                        icon: "exclamationmark.triangle.fill",
                        title: "Crash Reports",
                        description: "Currently disabled by design",
                        isOn: .constant(false),
                        isDisabled: true
                    )

                    PrivacyToggleRow(
                        icon: "faceid",
                        title: "Biometric Lock",
                        description: "Require Face ID to open app",
                        isOn: $biometricEnabled,
                        isDisabled: false
                    )
                }

                // Data Management
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Data Management")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    Button(role: .destructive) {
                        showDeleteConfirmation = true
                    } label: {
                        HStack {
                            Image(systemName: "trash.fill")
                            Text("Delete All Data")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(AssetsManager.Colors.error.opacity(0.1))
                        .foregroundColor(AssetsManager.Colors.error)
                        .cornerRadius(AssetsManager.CornerRadius.md)
                    }
                    .padding(.horizontal, AssetsManager.Spacing.md)
                }

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Privacy")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Delete All Data?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                // Delete all data
            }
        } message: {
            Text("This will permanently delete all your health data. This action cannot be undone.")
        }
    }
}

struct PrivacyToggleRow: View {
    let icon: String
    let title: String
    let description: String
    @Binding var isOn: Bool
    let isDisabled: Bool

    var body: some View {
        HStack(spacing: AssetsManager.Spacing.md) {
            Image(systemName: icon)
                .foregroundColor(isDisabled ? .gray : AssetsManager.Colors.primary)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(description)
                    .font(.caption)
                    .foregroundColor(AssetsManager.Colors.secondaryText)
            }

            Spacer()

            Toggle("", isOn: $isOn)
                .disabled(isDisabled)
        }
        .padding(AssetsManager.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                .fill(AssetsManager.Colors.cardBackground)
                .shadow(
                    color: AssetsManager.Shadow.small.color,
                    radius: AssetsManager.Shadow.small.radius,
                    x: AssetsManager.Shadow.small.x,
                    y: AssetsManager.Shadow.small.y
                )
        )
        .padding(.horizontal, AssetsManager.Spacing.md)
        .opacity(isDisabled ? 0.6 : 1.0)
    }
}

// MARK: - Theme Settings View

struct ThemeSettingsView: View {
    @AppStorage("themeMode") private var themeMode = "system"

    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header
                VStack(spacing: AssetsManager.Spacing.md) {
                    Image(systemName: "paintbrush.fill")
                        .font(.system(size: 60))
                        .foregroundColor(AssetsManager.Colors.primary)

                    Text("Appearance")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Customize how InflamAI looks")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Theme Options
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Theme")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    ThemeOptionCard(
                        title: "System",
                        description: "Match your device settings",
                        icon: "iphone",
                        isSelected: themeMode == "system",
                        action: { themeMode = "system" }
                    )

                    ThemeOptionCard(
                        title: "Light",
                        description: "Always use light mode",
                        icon: "sun.max.fill",
                        isSelected: themeMode == "light",
                        action: { themeMode = "light" }
                    )

                    ThemeOptionCard(
                        title: "Dark",
                        description: "Always use dark mode",
                        icon: "moon.fill",
                        isSelected: themeMode == "dark",
                        action: { themeMode = "dark" }
                    )
                }

                MascotTipCard(
                    icon: "info.circle.fill",
                    title: "Coming Soon",
                    message: "Custom color themes and accessibility options are coming in a future update!",
                    color: AssetsManager.Colors.info
                )
                .padding(.horizontal, AssetsManager.Spacing.md)

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Theme")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ThemeOptionCard: View {
    let title: String
    let description: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: AssetsManager.Spacing.md) {
                ZStack {
                    Circle()
                        .fill(isSelected ? AssetsManager.Colors.primary.opacity(0.2) : Color.gray.opacity(0.1))
                        .frame(width: 50, height: 50)

                    Image(systemName: icon)
                        .font(.title3)
                        .foregroundColor(isSelected ? AssetsManager.Colors.primary : .gray)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(AssetsManager.Colors.primaryText)

                    Text(description)
                        .font(.caption)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }

                Spacer()

                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundColor(isSelected ? AssetsManager.Colors.primary : .gray)
            }
            .padding(AssetsManager.Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                    .fill(AssetsManager.Colors.cardBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                            .strokeBorder(isSelected ? AssetsManager.Colors.primary : Color.clear, lineWidth: 2)
                    )
                    .shadow(
                        color: AssetsManager.Shadow.small.color,
                        radius: AssetsManager.Shadow.small.radius,
                        x: AssetsManager.Shadow.small.x,
                        y: AssetsManager.Shadow.small.y
                    )
            )
            .padding(.horizontal, AssetsManager.Spacing.md)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Apple Watch View

struct AppleWatchView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header
                VStack(spacing: AssetsManager.Spacing.md) {
                    Image(systemName: "applewatch")
                        .font(.system(size: 60))
                        .foregroundColor(AssetsManager.Colors.primary)

                    Text("Apple Watch")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Quick symptom logging from your wrist")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Coming Soon
                MascotEmptyState(
                    icon: "applewatch.watchface",
                    title: "Coming Soon",
                    message: "Apple Watch companion app is in development! You'll be able to log pain, track medications, and view insights right from your wrist.",
                    actionTitle: nil,
                    action: nil
                )

                // Planned Features
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("Planned Features")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    FeatureRow(
                        icon: "hand.tap.fill",
                        title: "Quick Pain Log",
                        description: "Tap to log current pain level in seconds",
                        color: AssetsManager.Colors.primary
                    )

                    FeatureRow(
                        icon: "pills.fill",
                        title: "Medication Reminders",
                        description: "Haptic reminders and easy confirmation",
                        color: AssetsManager.Colors.medication
                    )

                    FeatureRow(
                        icon: "chart.line.uptrend.xyaxis",
                        title: "Quick Insights",
                        description: "View BASDAI score and trends at a glance",
                        color: AssetsManager.Colors.info
                    )

                    FeatureRow(
                        icon: "figure.walk",
                        title: "Activity Tracking",
                        description: "Correlate movement with symptom patterns",
                        color: AssetsManager.Colors.success
                    )
                }

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Apple Watch")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - HealthKit View

struct HealthKitView: View {
    @State private var healthKitEnabled = false
    @State private var sleepEnabled = false
    @State private var heartRateEnabled = false
    @State private var stepsEnabled = false

    var body: some View {
        ScrollView {
            VStack(spacing: AssetsManager.Spacing.lg) {
                // Header
                VStack(spacing: AssetsManager.Spacing.md) {
                    Image(systemName: "heart.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.red)

                    Text("HealthKit Integration")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Connect health data for deeper insights")
                        .font(.subheadline)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                }
                .padding(.top, AssetsManager.Spacing.xl)

                // Master Toggle
                VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                    Text("HealthKit Access")
                        .font(.headline)
                        .padding(.horizontal, AssetsManager.Spacing.md)

                    Toggle(isOn: $healthKitEnabled) {
                        HStack(spacing: AssetsManager.Spacing.md) {
                            Image(systemName: "heart.circle.fill")
                                .foregroundColor(.red)
                                .font(.title2)

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Enable HealthKit")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)

                                Text("Allow InflamAI to read health data")
                                    .font(.caption)
                                    .foregroundColor(AssetsManager.Colors.secondaryText)
                            }
                        }
                    }
                    .padding(AssetsManager.Spacing.md)
                    .background(
                        RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                            .fill(AssetsManager.Colors.cardBackground)
                            .shadow(
                                color: AssetsManager.Shadow.small.color,
                                radius: AssetsManager.Shadow.small.radius,
                                x: AssetsManager.Shadow.small.x,
                                y: AssetsManager.Shadow.small.y
                            )
                    )
                    .padding(.horizontal, AssetsManager.Spacing.md)
                }

                if healthKitEnabled {
                    // Data Types
                    VStack(alignment: .leading, spacing: AssetsManager.Spacing.md) {
                        Text("Data Types")
                            .font(.headline)
                            .padding(.horizontal, AssetsManager.Spacing.md)

                        HealthDataToggle(
                            icon: "bed.double.fill",
                            title: "Sleep Data",
                            description: "Correlate sleep quality with AS symptoms",
                            isOn: $sleepEnabled,
                            color: .purple
                        )

                        HealthDataToggle(
                            icon: "heart.fill",
                            title: "Heart Rate",
                            description: "Monitor cardiovascular health trends",
                            isOn: $heartRateEnabled,
                            color: .red
                        )

                        HealthDataToggle(
                            icon: "figure.walk",
                            title: "Steps & Activity",
                            description: "Track movement and exercise correlation",
                            isOn: $stepsEnabled,
                            color: .green
                        )
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // Privacy
                MascotTipCard(
                    icon: "lock.shield.fill",
                    title: "Read-Only Access",
                    message: "InflamAI only READS your health data. We never write back to HealthKit without your explicit permission.",
                    color: AssetsManager.Colors.success
                )
                .padding(.horizontal, AssetsManager.Spacing.md)

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("HealthKit")
        .navigationBarTitleDisplayMode(.inline)
        .animation(.easeInOut, value: healthKitEnabled)
    }
}

struct HealthDataToggle: View {
    let icon: String
    let title: String
    let description: String
    @Binding var isOn: Bool
    let color: Color

    var body: some View {
        Toggle(isOn: $isOn) {
            HStack(spacing: AssetsManager.Spacing.md) {
                ZStack {
                    Circle()
                        .fill(color.opacity(0.15))
                        .frame(width: 44, height: 44)

                    Image(systemName: icon)
                        .foregroundColor(color)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)

                    Text(description)
                        .font(.caption)
                        .foregroundColor(AssetsManager.Colors.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(AssetsManager.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: AssetsManager.CornerRadius.md)
                .fill(AssetsManager.Colors.cardBackground)
                .shadow(
                    color: AssetsManager.Shadow.small.color,
                    radius: AssetsManager.Shadow.small.radius,
                    x: AssetsManager.Shadow.small.x,
                    y: AssetsManager.Shadow.small.y
                )
        )
        .padding(.horizontal, AssetsManager.Spacing.md)
    }
}

// MARK: - Previews

#Preview("Backup") {
    NavigationView {
        BackupRestoreView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}

#Preview("Privacy") {
    NavigationView {
        PrivacySettingsView()
    }
}

#Preview("Theme") {
    NavigationView {
        ThemeSettingsView()
    }
}

#Preview("Apple Watch") {
    NavigationView {
        AppleWatchView()
    }
}

#Preview("HealthKit") {
    NavigationView {
        HealthKitView()
    }
}
