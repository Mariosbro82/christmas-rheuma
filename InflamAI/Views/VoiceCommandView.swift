//
//  VoiceCommandView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Speech

struct VoiceCommandView: View {
    @StateObject private var voiceManager = VoiceCommandManager.shared
    @State private var showingSettings = false
    @State private var showingHelp = false
    @State private var showingPermissions = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "mic.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(voiceManager.isListening ? .red : .blue)
                        .scaleEffect(voiceManager.isListening ? 1.2 : 1.0)
                        .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: voiceManager.isListening)
                    
                    Text("Voice Commands")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text(voiceManager.isListening ? "Listening..." : "Tap to speak")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                // Status Card
                VoiceStatusCard()
                
                // Main Voice Button
                VoiceControlButton()
                
                // Recognized Text Display
                if !voiceManager.recognizedText.isEmpty {
                    RecognizedTextView()
                }
                
                // Last Command Display
                if let lastCommand = voiceManager.lastCommand {
                    LastCommandView(command: lastCommand)
                }
                
                // Quick Actions
                QuickVoiceActionsView()
                
                Spacer()
                
                // Bottom Actions
                HStack(spacing: 20) {
                    Button("Help") {
                        showingHelp = true
                    }
                    .buttonStyle(.bordered)
                    
                    Button("Settings") {
                        showingSettings = true
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding()
            .navigationTitle("Voice Commands")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showingSettings) {
                VoiceSettingsView()
            }
            .sheet(isPresented: $showingHelp) {
                VoiceHelpView()
            }
            .sheet(isPresented: $showingPermissions) {
                VoicePermissionsView()
            }
            .onAppear {
                if !voiceManager.isAvailable {
                    showingPermissions = true
                }
            }
        }
    }
}

struct VoiceStatusCard: View {
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: statusIcon)
                    .foregroundColor(statusColor)
                Text(statusText)
                    .font(.headline)
                Spacer()
            }
            
            if let error = voiceManager.error {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error.localizedDescription)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Processing indicator
            if voiceManager.isProcessing {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Processing command...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var statusIcon: String {
        if voiceManager.isListening {
            return "mic.fill"
        } else if voiceManager.isAvailable {
            return "checkmark.circle.fill"
        } else {
            return "xmark.circle.fill"
        }
    }
    
    private var statusColor: Color {
        if voiceManager.isListening {
            return .red
        } else if voiceManager.isAvailable {
            return .green
        } else {
            return .red
        }
    }
    
    private var statusText: String {
        if voiceManager.isListening {
            return "Listening for commands"
        } else if voiceManager.isAvailable {
            return "Ready for voice commands"
        } else {
            return "Voice commands unavailable"
        }
    }
}

struct VoiceControlButton: View {
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    var body: some View {
        Button(action: toggleListening) {
            ZStack {
                Circle()
                    .fill(voiceManager.isListening ? Color.red : Color.blue)
                    .frame(width: 120, height: 120)
                    .scaleEffect(voiceManager.isListening ? 1.1 : 1.0)
                    .animation(.easeInOut(duration: 0.3), value: voiceManager.isListening)
                
                Image(systemName: voiceManager.isListening ? "stop.fill" : "mic.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.white)
            }
        }
        .disabled(!voiceManager.isAvailable)
        .opacity(voiceManager.isAvailable ? 1.0 : 0.5)
    }
    
    private func toggleListening() {
        if voiceManager.isListening {
            voiceManager.stopListening()
        } else {
            voiceManager.startListening()
        }
    }
}

struct RecognizedTextView: View {
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "quote.bubble.fill")
                    .foregroundColor(.blue)
                Text("Recognized Speech")
                    .font(.headline)
                Spacer()
            }
            
            Text(voiceManager.recognizedText)
                .font(.body)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct LastCommandView: View {
    let command: VoiceCommand
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "command")
                    .foregroundColor(.green)
                Text("Last Command")
                    .font(.headline)
                Spacer()
                Text(command.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text("\"\(command.phrase)\"")
                    .font(.body)
                    .italic()
                
                Text(actionDescription(for: command.action))
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                if let response = voiceManager.lastResponse {
                    Text("Response: \(response.text)")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func actionDescription(for action: VoiceAction) -> String {
        switch action {
        case .navigateToSymptoms:
            return "Navigate to Symptoms"
        case .navigateToMedications:
            return "Navigate to Medications"
        case .logPain(let location, let severity):
            return "Log Pain - Location: \(location), Severity: \(severity)"
        case .logMood(let level):
            return "Log Mood - Level: \(level)"
        case .takeMedication(let name):
            return "Take Medication - \(name)"
        case .help:
            return "Show Help"
        default:
            return "Unknown Action"
        }
    }
}

struct QuickVoiceActionsView: View {
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    private let quickActions = [
        ("Log Pain", "figure.walk.circle", "log pain in my knee level 5"),
        ("Take Medication", "pills.circle", "take methotrexate"),
        ("Check Schedule", "calendar.circle", "check medication schedule"),
        ("Emergency", "phone.circle.fill", "emergency call")
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "bolt.circle.fill")
                    .foregroundColor(.orange)
                Text("Quick Actions")
                    .font(.headline)
                Spacer()
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                ForEach(quickActions, id: \.0) { action in
                    QuickActionButton(
                        title: action.0,
                        icon: action.1,
                        command: action.2
                    )
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct QuickActionButton: View {
    let title: String
    let icon: String
    let command: String
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    var body: some View {
        Button(action: executeCommand) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(.blue)
                
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(8)
        }
        .disabled(!voiceManager.isAvailable)
    }
    
    private func executeCommand() {
        // Simulate voice command execution
        Task {
            let nlpProcessor = NaturalLanguageProcessor()
            do {
                let voiceCommand = try await nlpProcessor.processCommand(command)
                let processor = VoiceCommandProcessor()
                let response = await processor.executeCommand(voiceCommand)
                
                await MainActor.run {
                    voiceManager.lastCommand = voiceCommand
                    voiceManager.lastResponse = response
                    
                    if response.shouldSpeak {
                        voiceManager.speak(response.text)
                    }
                    
                    response.action?()
                }
            } catch {
                print("Error executing quick action: \(error)")
            }
        }
    }
}

struct VoiceSettingsView: View {
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    @Environment(\.dismiss) private var dismiss
    @State private var settings: VoiceSettings
    
    init() {
        _settings = State(initialValue: VoiceCommandManager.shared.settings)
    }
    
    var body: some View {
        NavigationView {
            Form {
                Section("General") {
                    Toggle("Enable Voice Commands", isOn: $settings.isEnabled)
                    
                    Picker("Language", selection: $settings.language) {
                        ForEach(voiceManager.getAvailableLanguages(), id: \.self) { language in
                            Text(Locale(identifier: language).localizedString(forIdentifier: language) ?? language)
                                .tag(language)
                        }
                    }
                }
                
                Section("Voice Settings") {
                    VStack {
                        HStack {
                            Text("Speech Speed")
                            Spacer()
                            Text("\(settings.voiceSpeed, specifier: "%.1f")")
                        }
                        Slider(value: $settings.voiceSpeed, in: 0.1...1.0, step: 0.1)
                    }
                    
                    VStack {
                        HStack {
                            Text("Voice Pitch")
                            Spacer()
                            Text("\(settings.voicePitch, specifier: "%.1f")")
                        }
                        Slider(value: $settings.voicePitch, in: 0.5...2.0, step: 0.1)
                    }
                    
                    VStack {
                        HStack {
                            Text("Voice Volume")
                            Spacer()
                            Text("\(settings.voiceVolume, specifier: "%.1f")")
                        }
                        Slider(value: $settings.voiceVolume, in: 0.1...1.0, step: 0.1)
                    }
                }
                
                Section("Wake Word") {
                    Toggle("Enable Wake Word", isOn: $settings.wakeWordEnabled)
                    
                    if settings.wakeWordEnabled {
                        TextField("Wake Word", text: $settings.wakeWord)
                    }
                }
                
                Section("Behavior") {
                    Toggle("Continuous Listening", isOn: $settings.continuousListening)
                    Toggle("Haptic Feedback", isOn: $settings.hapticFeedbackEnabled)
                    Toggle("Confirmation Required", isOn: $settings.confirmationRequired)
                    Toggle("Privacy Mode (On-Device)", isOn: $settings.privacyMode)
                }
                
                Section("Test Voice") {
                    Button("Test Speech Synthesis") {
                        voiceManager.speak("This is a test of the voice synthesis system.")
                    }
                    
                    Button("Stop Speaking") {
                        voiceManager.stopSpeaking()
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Voice Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        voiceManager.updateSettings(settings)
                        dismiss()
                    }
                }
            }
        }
    }
}

struct VoiceHelpView: View {
    @Environment(\.dismiss) private var dismiss
    
    private let commandCategories = [
        ("Navigation", [
            "Go to symptoms",
            "Navigate to medications",
            "Show appointments",
            "Open journal",
            "Go to analytics",
            "Go back",
            "Go home"
        ]),
        ("Symptom Tracking", [
            "Log pain in my knee level 7",
            "Record mood level 5",
            "Add fatigue level 8",
            "Log stiffness severity 6"
        ]),
        ("Medication Management", [
            "Take methotrexate",
            "Skip prednisone",
            "Set reminder for humira at 8 AM",
            "Check medication schedule"
        ]),
        ("Emergency", [
            "Emergency call",
            "Contact doctor",
            "Call for help"
        ]),
        ("General", [
            "Help",
            "What can you do",
            "Repeat",
            "Cancel"
        ])
    ]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Introduction
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Voice Commands Help")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        
                        Text("Use natural language to control the app. Here are some examples of what you can say:")
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    // Command Categories
                    ForEach(commandCategories, id: \.0) { category in
                        VStack(alignment: .leading, spacing: 12) {
                            Text(category.0)
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(category.1, id: \.self) { command in
                                    HStack {
                                        Image(systemName: "quote.bubble")
                                            .foregroundColor(.blue)
                                            .font(.caption)
                                        Text("\"\(command)\"")
                                            .font(.body)
                                            .italic()
                                    }
                                }
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                        }
                    }
                    
                    // Tips
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Tips")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            TipRow(icon: "lightbulb", text: "Speak clearly and at a normal pace")
                            TipRow(icon: "speaker.wave.2", text: "Use the app in a quiet environment for best results")
                            TipRow(icon: "number", text: "Include severity levels (1-10) when logging symptoms")
                            TipRow(icon: "clock", text: "Specify times when setting medication reminders")
                            TipRow(icon: "person.2", text: "Use medication names or body parts for better recognition")
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    }
                }
                .padding()
            }
            .navigationTitle("Help")
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

struct TipRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.orange)
                .frame(width: 20)
            Text(text)
                .font(.body)
        }
    }
}

struct VoicePermissionsView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var voiceManager = VoiceCommandManager.shared
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                Spacer()
                
                // Icon
                Image(systemName: "mic.circle.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.blue)
                
                // Title and Description
                VStack(spacing: 16) {
                    Text("Enable Voice Commands")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.center)
                    
                    Text("Voice commands allow you to control the app hands-free. You can log symptoms, take medications, and navigate the app using natural speech.")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                // Permissions Status
                VStack(spacing: 12) {
                    PermissionRow(
                        title: "Speech Recognition",
                        icon: "mic.fill",
                        isGranted: voiceManager.isAvailable
                    )
                    
                    PermissionRow(
                        title: "Microphone Access",
                        icon: "waveform",
                        isGranted: voiceManager.isAvailable
                    )
                }
                
                Spacer()
                
                // Action Buttons
                VStack(spacing: 12) {
                    Button("Enable Voice Commands") {
                        openSettings()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                    .disabled(voiceManager.isAvailable)
                    
                    Button("Maybe Later") {
                        dismiss()
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding()
            .navigationTitle("Permissions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Skip") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func openSettings() {
        if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(settingsUrl)
        }
    }
}

struct PermissionRow: View {
    let title: String
    let icon: String
    let isGranted: Bool
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(isGranted ? .green : .orange)
                .frame(width: 24)
            
            Text(title)
                .font(.body)
            
            Spacer()
            
            Image(systemName: isGranted ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                .foregroundColor(isGranted ? .green : .orange)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

#Preview {
    VoiceCommandView()
}