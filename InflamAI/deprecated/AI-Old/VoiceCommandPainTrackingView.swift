//
//  VoiceCommandPainTrackingView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import Speech
import AVFoundation

struct VoiceCommandPainTrackingView: View {
    @StateObject private var voiceEngine = VoiceCommandEngine.shared
    @StateObject private var speechRecognizer = SpeechRecognizer()
    @StateObject private var aiEngine = AIMLEngine.shared
    
    @Binding var selectedRegions: Set<BodyRegion>
    @Binding var painIntensity: [BodyRegion: Double]
    
    @State private var isListening = false
    @State private var recognizedText = ""
    @State private var lastCommand = ""
    @State private var commandHistory: [VoiceCommand] = []
    @State private var showingPermissionAlert = false
    @State private var voiceFeedbackEnabled = true
    @State private var hapticFeedbackEnabled = true
    @State private var smartSuggestionsEnabled = true
    
    private let synthesizer = AVSpeechSynthesizer()
    
    var body: some View {
        VStack(spacing: 25) {
            // Header
            VStack(spacing: 15) {
                HStack {
                    Image(systemName: "mic.circle.fill")
                        .font(.largeTitle)
                        .foregroundColor(isListening ? .red : .blue)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Voice Pain Tracking")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        Text(isListening ? "Listening..." : "Tap to start")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    VoiceSettingsButton()
                }
                
                // Voice Status Indicator
                VoiceStatusIndicator(isListening: isListening, recognizedText: recognizedText)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(.systemBackground))
                    .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
            )
            
            // Main Voice Control
            VStack(spacing: 20) {
                // Large Voice Button
                Button(action: toggleListening) {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: isListening ? [.red, .pink] : [.blue, .purple],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 120, height: 120)
                            .scaleEffect(isListening ? 1.1 : 1.0)
                            .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: isListening)
                        
                        Image(systemName: isListening ? "stop.circle" : "mic")
                            .font(.system(size: 40, weight: .medium))
                            .foregroundColor(.white)
                    }
                }
                .buttonStyle(PlainButtonStyle())
                .disabled(!speechRecognizer.isAvailable)
                
                // Voice Command Examples
                VoiceCommandExamplesView()
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(.systemBackground))
                    .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
            )
            
            // Recent Commands
            if !commandHistory.isEmpty {
                RecentCommandsView(commands: commandHistory)
            }
            
            // Quick Voice Actions
            QuickVoiceActionsView()
            
            Spacer()
        }
        .padding()
        .onAppear {
            setupVoiceRecognition()
        }
        .onChange(of: speechRecognizer.transcript) { transcript in
            recognizedText = transcript
            processVoiceCommand(transcript)
        }
        .alert("Microphone Permission Required", isPresented: $showingPermissionAlert) {
            Button("Settings") {
                if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(settingsUrl)
                }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Please enable microphone access in Settings to use voice commands.")
        }
    }
    
    private func setupVoiceRecognition() {
        speechRecognizer.requestPermission { granted in
            if !granted {
                showingPermissionAlert = true
            }
        }
    }
    
    private func toggleListening() {
        if isListening {
            stopListening()
        } else {
            startListening()
        }
    }
    
    private func startListening() {
        guard speechRecognizer.isAvailable else { return }
        
        isListening = true
        recognizedText = ""
        speechRecognizer.startTranscribing()
        
        if hapticFeedbackEnabled {
            let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
            impactFeedback.impactOccurred()
        }
        
        if voiceFeedbackEnabled {
            speakText("Listening for pain tracking command")
        }
    }
    
    private func stopListening() {
        isListening = false
        speechRecognizer.stopTranscribing()
        
        if hapticFeedbackEnabled {
            let impactFeedback = UIImpactFeedbackGenerator(style: .light)
            impactFeedback.impactOccurred()
        }
    }
    
    private func processVoiceCommand(_ text: String) {
        guard !text.isEmpty else { return }
        
        Task {
            let command = await aiEngine.parseVoiceCommand(text)
            
            await MainActor.run {
                executeCommand(command)
                addToCommandHistory(command)
                lastCommand = text
            }
        }
    }
    
    private func executeCommand(_ command: VoiceCommand) {
        switch command.type {
        case .setPainLevel:
            if let region = command.bodyRegion, let intensity = command.intensity {
                painIntensity[region] = intensity
                selectedRegions.insert(region)
                
                if voiceFeedbackEnabled {
                    speakText("Set \(region.displayName) pain to level \(Int(intensity))")
                }
                
                if hapticFeedbackEnabled {
                    let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                    impactFeedback.impactOccurred()
                }
            }
            
        case .addPainRegion:
            if let region = command.bodyRegion {
                selectedRegions.insert(region)
                painIntensity[region] = command.intensity ?? 5.0
                
                if voiceFeedbackEnabled {
                    speakText("Added \(region.displayName) to pain tracking")
                }
            }
            
        case .removePainRegion:
            if let region = command.bodyRegion {
                selectedRegions.remove(region)
                painIntensity.removeValue(forKey: region)
                
                if voiceFeedbackEnabled {
                    speakText("Removed \(region.displayName) from pain tracking")
                }
            }
            
        case .clearAll:
            selectedRegions.removeAll()
            painIntensity.removeAll()
            
            if voiceFeedbackEnabled {
                speakText("Cleared all pain tracking")
            }
            
        case .savePainEntry:
            savePainEntry()
            
            if voiceFeedbackEnabled {
                speakText("Pain entry saved successfully")
            }
            
        case .getPainSummary:
            providePainSummary()
            
        case .unknown:
            if voiceFeedbackEnabled {
                speakText("Sorry, I didn't understand that command. Try saying something like 'Set back pain to level 7'")
            }
        }
    }
    
    private func addToCommandHistory(_ command: VoiceCommand) {
        commandHistory.insert(command, at: 0)
        if commandHistory.count > 10 {
            commandHistory.removeLast()
        }
    }
    
    private func savePainEntry() {
        // Save current pain data
        let entry = PainEntry(
            timestamp: Date(),
            regions: selectedRegions,
            intensities: painIntensity,
            notes: "Voice command entry"
        )
        
        // Save to Core Data or health data manager
        HealthDataManager.shared.savePainEntry(entry)
    }
    
    private func providePainSummary() {
        let totalRegions = selectedRegions.count
        let averagePain = painIntensity.values.reduce(0, +) / Double(max(painIntensity.count, 1))
        
        let summary = "You have \(totalRegions) pain regions tracked with an average intensity of \(String(format: "%.1f", averagePain))"
        
        if voiceFeedbackEnabled {
            speakText(summary)
        }
    }
    
    private func speakText(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        utterance.volume = 0.8
        
        synthesizer.speak(utterance)
    }
}

// MARK: - Voice Status Indicator

struct VoiceStatusIndicator: View {
    let isListening: Bool
    let recognizedText: String
    
    var body: some View {
        VStack(spacing: 10) {
            // Audio Wave Animation
            HStack(spacing: 4) {
                ForEach(0..<5) { index in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(isListening ? .blue : .gray.opacity(0.3))
                        .frame(width: 4, height: isListening ? CGFloat.random(in: 10...30) : 10)
                        .animation(
                            isListening ? 
                                .easeInOut(duration: 0.5).repeatForever(autoreverses: true).delay(Double(index) * 0.1) :
                                .none,
                            value: isListening
                        )
                }
            }
            .frame(height: 40)
            
            // Recognized Text
            if !recognizedText.isEmpty {
                Text(recognizedText)
                    .font(.subheadline)
                    .foregroundColor(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color.blue.opacity(0.1))
                    )
                    .transition(.opacity.combined(with: .scale))
            }
        }
    }
}

// MARK: - Voice Command Examples

struct VoiceCommandExamplesView: View {
    private let examples = [
        "Set back pain to level 7",
        "Add shoulder pain",
        "Remove knee pain",
        "Save pain entry",
        "Clear all pain",
        "What's my pain summary?"
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "lightbulb")
                    .foregroundColor(.yellow)
                Text("Voice Command Examples")
                    .font(.headline)
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 10) {
                ForEach(examples, id: \.self) { example in
                    VoiceExampleCard(text: example)
                }
            }
        }
    }
}

struct VoiceExampleCard: View {
    let text: String
    
    var body: some View {
        Text("\"\(text)\"")
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray6))
            )
            .multilineTextAlignment(.center)
    }
}

// MARK: - Recent Commands

struct RecentCommandsView: View {
    let commands: [VoiceCommand]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .foregroundColor(.blue)
                Text("Recent Commands")
                    .font(.headline)
            }
            .padding(.horizontal)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(commands.prefix(5), id: \.id) { command in
                        RecentCommandCard(command: command)
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
        )
    }
}

struct RecentCommandCard: View {
    let command: VoiceCommand
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: command.type.icon)
                    .foregroundColor(command.type.color)
                    .font(.caption)
                
                Text(command.type.displayName)
                    .font(.caption2)
                    .fontWeight(.medium)
            }
            
            if let region = command.bodyRegion {
                Text(region.displayName)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            if let intensity = command.intensity {
                Text("Level \(Int(intensity))")
                    .font(.caption2)
                    .foregroundColor(.blue)
            }
            
            Text(command.timestamp.formatted(.dateTime.hour().minute()))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
        .frame(width: 120)
    }
}

// MARK: - Quick Voice Actions

struct QuickVoiceActionsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "bolt.circle")
                    .foregroundColor(.orange)
                Text("Quick Actions")
                    .font(.headline)
            }
            .padding(.horizontal)
            
            HStack(spacing: 15) {
                QuickActionButton(
                    title: "Emergency",
                    subtitle: "High Pain",
                    icon: "exclamationmark.triangle.fill",
                    color: .red
                ) {
                    // Handle emergency pain logging
                }
                
                QuickActionButton(
                    title: "Medication",
                    subtitle: "Log Taken",
                    icon: "pills.fill",
                    color: .blue
                ) {
                    // Handle medication logging
                }
                
                QuickActionButton(
                    title: "Summary",
                    subtitle: "Get Report",
                    icon: "doc.text",
                    color: .green
                ) {
                    // Handle summary request
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
        )
    }
}

struct QuickActionButton: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                
                VStack(spacing: 2) {
                    Text(title)
                        .font(.caption)
                        .fontWeight(.medium)
                    
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(color.opacity(0.1))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Voice Settings Button

struct VoiceSettingsButton: View {
    @State private var showingSettings = false
    
    var body: some View {
        Button(action: { showingSettings = true }) {
            Image(systemName: "gearshape")
                .font(.title3)
                .foregroundColor(.blue)
        }
        .sheet(isPresented: $showingSettings) {
            VoiceSettingsView()
        }
    }
}

struct VoiceSettingsView: View {
    @State private var voiceFeedbackEnabled = true
    @State private var hapticFeedbackEnabled = true
    @State private var smartSuggestionsEnabled = true
    @State private var autoSaveEnabled = false
    @State private var selectedVoice = "en-US"
    @State private var speechRate: Double = 0.5
    
    private let availableVoices = [
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "es-ES": "Spanish",
        "fr-FR": "French",
        "de-DE": "German"
    ]
    
    var body: some View {
        NavigationView {
            Form {
                Section("Voice Feedback") {
                    Toggle("Enable Voice Feedback", isOn: $voiceFeedbackEnabled)
                    
                    if voiceFeedbackEnabled {
                        Picker("Voice Language", selection: $selectedVoice) {
                            ForEach(availableVoices.keys.sorted(), id: \.self) { key in
                                Text(availableVoices[key] ?? key).tag(key)
                            }
                        }
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Speech Rate")
                            Slider(value: $speechRate, in: 0.1...1.0, step: 0.1)
                            Text("\(String(format: "%.1f", speechRate))x")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                
                Section("Haptic Feedback") {
                    Toggle("Enable Haptic Feedback", isOn: $hapticFeedbackEnabled)
                }
                
                Section("Smart Features") {
                    Toggle("Smart Suggestions", isOn: $smartSuggestionsEnabled)
                    Toggle("Auto-Save Entries", isOn: $autoSaveEnabled)
                }
                
                Section("Privacy") {
                    HStack {
                        Image(systemName: "lock.shield")
                            .foregroundColor(.green)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Voice data is processed locally")
                                .font(.subheadline)
                            Text("Your voice commands are not stored or transmitted")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Voice Settings")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    // Dismiss
                }
            )
        }
    }
}

// MARK: - Speech Recognizer

class SpeechRecognizer: ObservableObject {
    @Published var transcript = ""
    @Published var isRecording = false
    @Published var isAvailable = false
    
    private var audioEngine: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private let recognizer: SFSpeechRecognizer?
    
    init() {
        recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        
        Task {
            await requestPermission { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAvailable = granted
                }
            }
        }
    }
    
    func requestPermission(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                completion(status == .authorized)
            }
        }
    }
    
    func startTranscribing() {
        guard let recognizer = recognizer, recognizer.isAvailable else {
            return
        }
        
        request = SFSpeechAudioBufferRecognitionRequest()
        guard let request = request else { return }
        
        request.shouldReportPartialResults = true
        
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else { return }
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            request.append(buffer)
        }
        
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
        } catch {
            return
        }
        
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            DispatchQueue.main.async {
                if let result = result {
                    self?.transcript = result.bestTranscription.formattedString
                }
                
                if error != nil || result?.isFinal == true {
                    self?.stopTranscribing()
                }
            }
        }
        
        isRecording = true
    }
    
    func stopTranscribing() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        
        audioEngine = nil
        request = nil
        task = nil
        isRecording = false
    }
}

// MARK: - Voice Command Models

struct VoiceCommand {
    let id = UUID()
    let type: VoiceCommandType
    let originalText: String
    let bodyRegion: BodyRegion?
    let intensity: Double?
    let timestamp: Date
    let confidence: Double
}

enum VoiceCommandType {
    case setPainLevel
    case addPainRegion
    case removePainRegion
    case clearAll
    case savePainEntry
    case getPainSummary
    case unknown
    
    var displayName: String {
        switch self {
        case .setPainLevel: return "Set Pain"
        case .addPainRegion: return "Add Region"
        case .removePainRegion: return "Remove Region"
        case .clearAll: return "Clear All"
        case .savePainEntry: return "Save Entry"
        case .getPainSummary: return "Get Summary"
        case .unknown: return "Unknown"
        }
    }
    
    var icon: String {
        switch self {
        case .setPainLevel: return "slider.horizontal.3"
        case .addPainRegion: return "plus.circle"
        case .removePainRegion: return "minus.circle"
        case .clearAll: return "trash"
        case .savePainEntry: return "checkmark.circle"
        case .getPainSummary: return "doc.text"
        case .unknown: return "questionmark.circle"
        }
    }
    
    var color: Color {
        switch self {
        case .setPainLevel: return .blue
        case .addPainRegion: return .green
        case .removePainRegion: return .red
        case .clearAll: return .orange
        case .savePainEntry: return .purple
        case .getPainSummary: return .indigo
        case .unknown: return .gray
        }
    }
}

struct PainEntry {
    let id = UUID()
    let timestamp: Date
    let regions: Set<BodyRegion>
    let intensities: [BodyRegion: Double]
    let notes: String
    let source: String = "voice_command"
}