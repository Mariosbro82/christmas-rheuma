//
//  SettingsView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import LocalAuthentication
import CoreData

struct SettingsView: View {
    // MARK: - Neural Engine
    @ObservedObject private var neuralEngine = UnifiedNeuralEngine.shared
    @ObservedObject private var mlIntegration = MLIntegrationService.shared

    // MARK: - User Profile Fetch
    @FetchRequest(
        entity: UserProfile.entity(),
        sortDescriptors: [NSSortDescriptor(keyPath: \UserProfile.createdAt, ascending: false)],
        animation: .default
    )
    private var profiles: FetchedResults<UserProfile>

    private var userProfile: UserProfile? { profiles.first }

    @StateObject private var questionnairePrefs = QuestionnaireUserPreferences.shared
    @State private var showingThemeSettings = false
    @State private var showingExportView = false
    @State private var showingAbout = false
    @State private var showingNeuralEngineSettings = false
    @State private var showingProfileEdit = false
    @State private var notificationsEnabled = true
    @AppStorage("biometricLockEnabled") private var biometricAuthEnabled = false
    @State private var cloudSyncEnabled = false
    @State private var showingBiometricError = false
    @State private var biometricErrorMessage = ""
    @State private var showingReseedConfirmation = false
    @State private var isReseedingData = false

    // MARK: - Computed Properties
    private var userAge: Int? {
        guard let dob = userProfile?.dateOfBirth else { return nil }
        return Calendar.current.dateComponents([.year], from: dob, to: Date()).year
    }

    private var diseaseYears: Int? {
        guard let diagnosis = userProfile?.diagnosisDate else { return nil }
        return Calendar.current.dateComponents([.year], from: diagnosis, to: Date()).year
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from MoreView and HomeView,
        // which are already wrapped in NavigationView in MainTabView.
        List {
                // User Profile Section (NEW - at the top for visibility)
                Section {
                    HStack {
                        Image(systemName: "person.crop.circle.fill")
                            .foregroundColor(.blue)
                            .font(.title2)
                            .frame(width: 40, height: 40)

                        VStack(alignment: .leading, spacing: 4) {
                            // Display name or "Your Profile"
                            Text(userProfile?.name ?? "Your Profile")
                                .font(.headline)

                            // Display age and disease info if available
                            HStack(spacing: 12) {
                                if let age = userAge {
                                    Label("\(age) yrs", systemImage: "calendar")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                if let bmi = userProfile?.bmi, bmi > 0 {
                                    Label(String(format: "BMI %.1f", bmi), systemImage: "scalemass")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                if let years = diseaseYears {
                                    Label("AS \(years)y", systemImage: "cross.circle")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }

                        Spacer()

                        Image(systemName: "chevron.right")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingProfileEdit = true
                    }
                } header: {
                    Text("Your Profile")
                } footer: {
                    if userProfile == nil || userAge == nil {
                        Text("Complete your profile to improve ML prediction accuracy.")
                    } else {
                        Text("Tap to edit your health profile.")
                    }
                }

                // Questionnaires Section
                Section("Questionnaires") {
                    NavigationLink(destination: QuestionnaireSettingsView()) {
                        HStack {
                            Image(systemName: "list.clipboard")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            VStack(alignment: .leading, spacing: 2) {
                                Text("Manage Questionnaires")
                                    .font(.body)
                                Text("\(questionnairePrefs.enabledCount) active")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }

                // Theme Section
                Section("Appearance") {
                    HStack {
                        Image(systemName: "paintbrush.fill")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Theme")
                                .font(.body)
                            Text("System")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Image(systemName: "chevron.right")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingThemeSettings = true
                    }
                }
                
                // Notifications Section
                Section("Notifications") {
                    NavigationLink(destination: WeatherNotificationSettingsView()) {
                        HStack {
                            Image(systemName: "cloud.sun.rain.fill")
                                .foregroundColor(.orange)
                                .frame(width: 24)

                            VStack(alignment: .leading, spacing: 2) {
                                Text("Weather Alerts")
                                    .font(.body)
                                Text("Pressure-based flare warnings")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    HStack {
                        Image(systemName: "brain.head.profile")
                            .foregroundColor(.purple)
                            .frame(width: 24)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("AI Flare Alerts")
                                .font(.body)
                            Text("ML-based risk notifications")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Toggle("", isOn: Binding(
                            get: { FlareNotificationService.shared.notificationsEnabled },
                            set: { FlareNotificationService.shared.notificationsEnabled = $0 }
                        ))
                            .toggleStyle(SwitchToggleStyle(tint: .purple))
                    }

                    HStack {
                        Image(systemName: "bell.fill")
                            .foregroundColor(.blue)
                            .frame(width: 24)

                        Text("Medication Reminders")
                            .font(.body)

                        Spacer()

                        Toggle("", isOn: $notificationsEnabled)
                            .toggleStyle(SwitchToggleStyle(tint: .blue))
                    }
                    
                    HStack {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        Text("Pain Tracking Reminders")
                            .font(.body)
                        
                        Spacer()
                        
                        Toggle("", isOn: $notificationsEnabled)
                            .toggleStyle(SwitchToggleStyle(tint: .blue))
                    }
                }
                
                // Security Section
                Section("Security & Privacy") {
                    HStack {
                        Image(systemName: "faceid")
                            .foregroundColor(.blue)
                            .frame(width: 24)

                        Text("Biometric Authentication")
                            .font(.body)

                        Spacer()

                        Toggle("", isOn: $biometricAuthEnabled)
                            .toggleStyle(SwitchToggleStyle(tint: .blue))
                            .onChange(of: biometricAuthEnabled) { newValue in
                                if newValue {
                                    requestBiometricAuthentication()
                                }
                            }
                    }
                    
                    NavigationLink(destination: PrivacySettingsView()) {
                        HStack {
                            Image(systemName: "lock.shield")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            Text("Privacy Settings")
                                .font(.body)
                        }
                    }
                }
                
                // Data Section
                Section("Data Management") {
                    HStack {
                        Image(systemName: "icloud.fill")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        Text("Cloud Sync")
                            .font(.body)
                        
                        Spacer()
                        
                        Toggle("", isOn: $cloudSyncEnabled)
                            .toggleStyle(SwitchToggleStyle(tint: .blue))
                    }
                    
                    HStack {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        Text("Export Data")
                            .font(.body)
                        
                        Spacer()
                        
                        Image(systemName: "chevron.right")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingExportView = true
                    }
                    
                    NavigationLink(destination: BackupRestoreView()) {
                        HStack {
                            Image(systemName: "externaldrive")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            Text("Backup & Restore")
                                .font(.body)
                        }
                    }
                }
                
                // Neural Engine Section
                Section {
                    NavigationLink(destination: UnifiedNeuralEngineView()) {
                        HStack {
                            ZStack {
                                Circle()
                                    .fill(neuralEngineStatusColor.opacity(0.2))
                                    .frame(width: 32, height: 32)

                                Image(systemName: "brain.head.profile")
                                    .foregroundColor(neuralEngineStatusColor)
                                    .font(.system(size: 16))
                            }
                            .frame(width: 24)

                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text("Neural Engine")
                                        .font(.body)

                                    if neuralEngine.isPersonalized {
                                        Text("PERSONALIZED")
                                            .font(.caption2)
                                            .fontWeight(.bold)
                                            .foregroundColor(.white)
                                            .padding(.horizontal, 4)
                                            .padding(.vertical, 2)
                                            .background(Color.green)
                                            .cornerRadius(4)
                                    }
                                }

                                Text(neuralEngine.engineStatus.displayMessage)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            // Mini status indicator
                            if let prediction = neuralEngine.currentPrediction {
                                Text("\(Int(prediction.probability * 100))%")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(prediction.willFlare ? .orange : .green)
                            }
                        }
                    }

                    // Model Stats Row
                    HStack {
                        Image(systemName: "chart.bar.doc.horizontal")
                            .foregroundColor(.purple)
                            .frame(width: 24)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Model Statistics")
                                .font(.body)
                            Text("v\(neuralEngine.modelVersion) • \(neuralEngine.daysOfUserData) days • \(Int(neuralEngine.learningProgress * 100))% learned")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        // Personalization progress mini-bar
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Color(.systemGray5))
                                    .frame(width: 40, height: 4)

                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Color.purple)
                                    .frame(width: 40 * CGFloat(neuralEngine.learningProgress), height: 4)
                            }
                        }
                        .frame(width: 40, height: 4)
                    }

                    // Auto-Personalization Toggle
                    HStack {
                        Image(systemName: "brain")
                            .foregroundColor(.purple)
                            .frame(width: 24)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Auto-Personalization")
                                .font(.body)
                            Text("Learn from your check-ins")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Toggle("", isOn: $mlIntegration.autoPersonalizationEnabled)
                            .toggleStyle(SwitchToggleStyle(tint: .purple))
                    }

                    // Training Stats Row
                    if mlIntegration.trainingSamplesCollected > 0 {
                        HStack {
                            Image(systemName: "chart.bar.doc.horizontal")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            VStack(alignment: .leading, spacing: 2) {
                                Text("Training Samples")
                                    .font(.body)
                                Text("\(mlIntegration.trainingSamplesCollected) samples collected")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            Text(mlIntegration.getAccuracyMetrics().summary)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Reset Learning Button
                    Button(role: .destructive) {
                        showingNeuralEngineSettings = true
                    } label: {
                        HStack {
                            Image(systemName: "arrow.counterclockwise")
                                .foregroundColor(.red)
                                .frame(width: 24)

                            Text("Reset Neural Engine")
                                .font(.body)
                                .foregroundColor(.red)
                        }
                    }
                } header: {
                    Text("Self-Learning AI")
                } footer: {
                    Text("The Neural Engine learns from your data to provide personalized flare predictions. All processing happens on your device. \(mlIntegration.autoPersonalizationEnabled ? "Auto-personalization is ON." : "Auto-personalization is OFF.")")
                }

                // Health Section
                Section("Health Integration") {
                    NavigationLink(destination: MLFeaturesDisplayView()) {
                        HStack {
                            Image(systemName: "waveform.path.ecg")
                                .foregroundColor(.purple)
                                .frame(width: 24)

                            VStack(alignment: .leading, spacing: 2) {
                                Text("92 Data Streams")
                                    .font(.body)
                                Text("Real-time ML feature tracking")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    NavigationLink(destination: HealthKitView()) {
                        HStack {
                            Image(systemName: "heart.fill")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            Text("HealthKit Integration")
                                .font(.body)
                        }
                    }

                    NavigationLink(destination: AppleWatchView()) {
                        HStack {
                            Image(systemName: "applewatch")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            Text("Apple Watch")
                                .font(.body)
                        }
                    }
                }
                
                // Demo Data Section
                Section {
                    Button {
                        showingReseedConfirmation = true
                    } label: {
                        HStack {
                            Image(systemName: "arrow.counterclockwise")
                                .foregroundColor(.orange)
                                .frame(width: 24)

                            VStack(alignment: .leading, spacing: 2) {
                                Text("Reseed Demo Data")
                                    .font(.body)
                                    .foregroundColor(.primary)
                                Text("Reset to Anna's 200-day sample data")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            if isReseedingData {
                                ProgressView()
                                    .scaleEffect(0.8)
                            }
                        }
                    }
                    .disabled(isReseedingData)

                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundColor(.blue)
                            .frame(width: 24)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Demo Mode Active")
                                .font(.body)
                            Text("Showing Anna's AS journey")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Text("200 days")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("Demo Data")
                } footer: {
                    Text("This is a demo app showing realistic data for 'Anna', a 30-year-old woman with AS. Reseed to reset all data.")
                }

                // Support Section
                Section("Support") {
                    NavigationLink(destination: HelpView()) {
                        HStack {
                            Image(systemName: "questionmark.circle")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            Text("Help & FAQ")
                                .font(.body)
                        }
                    }

                    NavigationLink(destination: CommunityView()) {
                        HStack {
                            Image(systemName: "person.3")
                                .foregroundColor(.blue)
                                .frame(width: 24)

                            Text("Community Support")
                                .font(.body)
                        }
                    }
                    
                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        Text("About")
                            .font(.body)
                        
                        Spacer()
                        
                        Image(systemName: "chevron.right")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingAbout = true
                    }
                }
            }
            .navigationTitle("Settings")
            .sheet(isPresented: $showingThemeSettings) {
                NavigationView {
                    ThemeSettingsView()
                        .toolbar {
                            ToolbarItem(placement: .navigationBarTrailing) {
                                Button("Done") {
                                    showingThemeSettings = false
                                }
                                .foregroundColor(.blue)
                            }
                        }
                }
            }
            .sheet(isPresented: $showingExportView) {
                NavigationView {
                    ExportDataView()
                        .toolbar {
                            ToolbarItem(placement: .navigationBarTrailing) {
                                Button("Done") {
                                    showingExportView = false
                                }
                                .foregroundColor(.blue)
                            }
                        }
                }
            }
            .sheet(isPresented: $showingAbout) {
                NavigationView {
                    AboutView()
                        .toolbar {
                            ToolbarItem(placement: .navigationBarTrailing) {
                                Button("Done") {
                                    showingAbout = false
                                }
                                .foregroundColor(.blue)
                            }
                        }
                }
            }
            .sheet(isPresented: $showingProfileEdit) {
                SettingsProfileEditView()
            }
            .alert("Biometric Authentication", isPresented: $showingBiometricError) {
                Button("OK") {
                    biometricAuthEnabled = false
                }
            } message: {
                Text(biometricErrorMessage)
            }
            .confirmationDialog(
                "Reset Neural Engine",
                isPresented: $showingNeuralEngineSettings,
                titleVisibility: .visible
            ) {
                Button("Reset All Learning", role: .destructive) {
                    // Reset the neural engine to baseline
                    UserDefaults.standard.removeObject(forKey: "neural_engine_model_version")
                    UserDefaults.standard.removeObject(forKey: "neural_engine_last_update")
                    UserDefaults.standard.removeObject(forKey: "neural_engine_accuracy")
                    UserDefaults.standard.removeObject(forKey: "neural_engine_prediction_logs")
                    // Force reload
                    Task {
                        await neuralEngine.refresh()
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will reset all personalized learning and return to the baseline model. Your symptom data will not be affected.")
            }
            .confirmationDialog(
                "Reseed Demo Data",
                isPresented: $showingReseedConfirmation,
                titleVisibility: .visible
            ) {
                Button("Reseed All Data", role: .destructive) {
                    reseedDemoData()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will delete all current data and regenerate 200 days of demo data for Anna. This action cannot be undone.")
            }
    }

    // MARK: - Demo Data Reseed

    @MainActor
    private func reseedDemoData() {
        isReseedingData = true

        Task {
            do {
                let context = InflamAIPersistenceController.shared.container.viewContext
                try await DemoDataSeeder.shared.seedDemoData(context: context)
                UserDefaults.standard.hasDemoDataBeenSeeded = true

                // Refresh neural engine after reseed
                await neuralEngine.refresh()

                print("✅ Demo data reseeded successfully")
            } catch {
                print("❌ Failed to reseed demo data: \(error)")
            }

            isReseedingData = false
        }
    }

    // MARK: - Neural Engine Helpers

    private var neuralEngineStatusColor: Color {
        switch neuralEngine.engineStatus {
        case .ready:
            if let prediction = neuralEngine.currentPrediction {
                return prediction.willFlare ? .orange : .green
            }
            return .green
        case .initializing:
            return .orange
        case .learning:
            return .blue
        case .error:
            return .red
        }
    }

    // MARK: - Biometric Authentication

    private func requestBiometricAuthentication() {
        let context = LAContext()
        var error: NSError?

        // Check if biometric authentication is available
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            biometricErrorMessage = error?.localizedDescription ?? "Biometric authentication is not available on this device."
            showingBiometricError = true
            biometricAuthEnabled = false
            return
        }

        // Request biometric authentication
        context.evaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics,
            localizedReason: "Enable biometric authentication to protect your health data"
        ) { success, authError in
            DispatchQueue.main.async {
                if success {
                    // Authentication successful - toggle stays on
                    print("✅ Biometric authentication enabled")
                } else {
                    // Authentication failed - turn toggle off
                    biometricErrorMessage = authError?.localizedDescription ?? "Authentication failed"
                    showingBiometricError = true
                    biometricAuthEnabled = false
                }
            }
        }
    }
}

// MARK: - Placeholder Views (TODO: Implement)

struct PrivacySettingsView: View {
    var body: some View {
        Text("Privacy Settings - Coming Soon")
            .navigationTitle("Privacy")
    }
}

struct BackupRestoreView: View {
    var body: some View {
        Text("Backup & Restore - Coming Soon")
            .navigationTitle("Backup")
    }
}

struct HealthKitView: View {
    var body: some View {
        Text("HealthKit Integration - Coming Soon")
            .navigationTitle("HealthKit")
    }
}

struct AppleWatchView: View {
    var body: some View {
        Text("Apple Watch - Coming Soon")
            .navigationTitle("Apple Watch")
    }
}

struct HelpView: View {
    var body: some View {
        Text("Help & FAQ - Coming Soon")
            .navigationTitle("Help")
    }
}

struct CommunityView: View {
    var body: some View {
        Text("Community Support - Coming Soon")
            .navigationTitle("Community")
    }
}

struct ThemeSettingsView: View {
    var body: some View {
        Text("Theme Settings - Coming Soon")
            .navigationTitle("Theme")
    }
}

struct AboutView: View {
    var body: some View {
        VStack(spacing: 20) {
            Text("InflamAI")
                .font(.largeTitle)
                .fontWeight(.bold)
            Text("Version 1.0")
                .foregroundColor(.secondary)
            Text("Ankylosing Spondylitis Management")
                .foregroundColor(.secondary)
        }
        .navigationTitle("About")
    }
}

// MARK: - Profile Edit View for Settings

struct SettingsProfileEditView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel = UserProfileEditViewModel()  // CONSOLIDATED: was SettingsProfileEditViewModel

    var body: some View {
        NavigationView {
            Form {
                // Personal Information Section
                Section {
                    TextField("Name (optional)", text: $viewModel.name)

                    DatePicker("Date of Birth",
                               selection: $viewModel.dateOfBirth,
                               in: ...Date(),
                               displayedComponents: .date)

                    Picker("Gender", selection: $viewModel.gender) {
                        Text("Select...").tag("")
                        Text("Male").tag("male")
                        Text("Female").tag("female")
                        Text("Other").tag("other")
                        Text("Prefer not to say").tag("unknown")
                    }
                } header: {
                    Text("Personal Information")
                } footer: {
                    Text("This information helps personalize your health insights.")
                }

                // Body Measurements Section
                Section {
                    HStack {
                        Text("Height")
                        Spacer()
                        TextField("cm", value: $viewModel.heightCm, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Text("cm")
                            .foregroundColor(.secondary)
                    }

                    HStack {
                        Text("Weight")
                        Spacer()
                        TextField("kg", value: $viewModel.weightKg, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Text("kg")
                            .foregroundColor(.secondary)
                    }

                    if viewModel.calculatedBMI > 0 {
                        HStack {
                            Text("BMI")
                            Spacer()
                            Text(String(format: "%.1f", viewModel.calculatedBMI))
                                .foregroundColor(viewModel.bmiColor)
                                .fontWeight(.medium)
                            Text(viewModel.bmiCategory)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                } header: {
                    Text("Body Measurements")
                } footer: {
                    Text("BMI is used to assess disease risk patterns.")
                }

                // AS-Specific Information Section
                Section {
                    DatePicker("AS Diagnosis Date",
                               selection: $viewModel.diagnosisDate,
                               in: ...Date(),
                               displayedComponents: .date)

                    Toggle("HLA-B27 Positive", isOn: $viewModel.hlaB27Positive)

                    Toggle("Previous Biologic Treatment", isOn: $viewModel.biologicExperienced)
                } header: {
                    Text("AS Information")
                } footer: {
                    Text("Disease duration and HLA-B27 status affect flare prediction accuracy.")
                }

                // Lifestyle Section
                Section {
                    Picker("Smoking Status", selection: $viewModel.smokingStatus) {
                        Text("Never smoked").tag("never")
                        Text("Former smoker").tag("former")
                        Text("Current smoker").tag("current")
                    }
                } header: {
                    Text("Lifestyle")
                } footer: {
                    Text("Smoking affects disease progression and treatment response.")
                }
            }
            .navigationTitle("Edit Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        viewModel.saveProfile()
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
}

// NOTE: SettingsProfileEditViewModel DELETED - now using consolidated UserProfileEditViewModel
// from Features/Settings/UserProfileEditView.swift (eliminates ~90 lines of duplicate code)

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            SettingsView()
                .preferredColorScheme(.light)

            SettingsView()
                .preferredColorScheme(.dark)
        }
    }
}