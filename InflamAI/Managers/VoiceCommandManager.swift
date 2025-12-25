//
//  VoiceCommandManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import Speech
import AVFoundation
import Combine

// MARK: - Voice Command Manager

class VoiceCommandManager: NSObject, ObservableObject {
    static let shared = VoiceCommandManager()
    
    @Published var isListening = false
    @Published var isAvailable = false
    @Published var lastCommand: String = ""
    @Published var recognizedText: String = ""
    @Published var errorMessage: String? = nil
    
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private var cancellables = Set<AnyCancellable>()
    
    // Voice command patterns
    private let commandPatterns: [VoiceCommand] = [
        // Pain tracking
        VoiceCommand(patterns: ["log pain", "record pain", "add pain"], action: .logPain),
        VoiceCommand(patterns: ["pain level", "rate pain"], action: .ratePain),
        
        // Medication
        VoiceCommand(patterns: ["take medication", "log medication", "record medication"], action: .logMedication),
        VoiceCommand(patterns: ["medication reminder", "set reminder"], action: .setMedicationReminder),
        
        // Journal
        VoiceCommand(patterns: ["add journal", "write journal", "journal entry"], action: .addJournalEntry),
        
        // Navigation
        VoiceCommand(patterns: ["go to dashboard", "show dashboard", "home"], action: .navigateToDashboard),
        VoiceCommand(patterns: ["go to analytics", "show analytics", "view analytics"], action: .navigateToAnalytics),
        VoiceCommand(patterns: ["go to medications", "show medications"], action: .navigateToMedications),
        VoiceCommand(patterns: ["go to journal", "show journal"], action: .navigateToJournal),
        
        // Settings
        VoiceCommand(patterns: ["toggle dark mode", "switch theme", "dark mode"], action: .toggleDarkMode),
        VoiceCommand(patterns: ["increase text size", "larger text"], action: .increaseTextSize),
        VoiceCommand(patterns: ["decrease text size", "smaller text"], action: .decreaseTextSize),
        
        // Emergency
        VoiceCommand(patterns: ["emergency", "help", "call doctor"], action: .emergency),
        
        // General
        VoiceCommand(patterns: ["stop listening", "stop", "cancel"], action: .stopListening),
        VoiceCommand(patterns: ["help", "what can you do", "commands"], action: .showHelp)
    ]
    
    override init() {
        super.init()
        setupSpeechRecognizer()
        requestPermissions()
    }
    
    // MARK: - Public Methods
    
    func startListening() {
        guard isAvailable else {
            errorMessage = "Voice recognition is not available"
            return
        }
        
        guard !isListening else { return }
        
        do {
            try startRecognition()
            isListening = true
            errorMessage = nil
            HapticManager.shared.notification(.success)
        } catch {
            errorMessage = "Failed to start voice recognition: \(error.localizedDescription)"
            HapticManager.shared.notification(.error)
        }
    }
    
    func stopListening() {
        guard isListening else { return }
        
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        recognitionRequest = nil
        recognitionTask = nil
        
        isListening = false
        HapticManager.shared.impact(.light)
    }
    
    func toggleListening() {
        if isListening {
            stopListening()
        } else {
            startListening()
        }
    }
    
    func processCommand(_ text: String) {
        let lowercaseText = text.lowercased()
        
        for command in commandPatterns {
            for pattern in command.patterns {
                if lowercaseText.contains(pattern.lowercased()) {
                    executeCommand(command.action, with: text)
                    lastCommand = text
                    return
                }
            }
        }
        
        // If no command matched, try to extract pain level
        if let painLevel = extractPainLevel(from: lowercaseText) {
            executeCommand(.ratePain, with: "\(painLevel)")
            lastCommand = text
        } else {
            errorMessage = "Command not recognized: \(text)"
            HapticManager.shared.notification(.warning)
        }
    }
    
    // MARK: - Private Methods
    
    private func setupSpeechRecognizer() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        speechRecognizer?.delegate = self
        
        isAvailable = speechRecognizer?.isAvailable ?? false
    }
    
    private func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                switch status {
                case .authorized:
                    self?.requestMicrophonePermission()
                case .denied, .restricted, .notDetermined:
                    self?.isAvailable = false
                    self?.errorMessage = "Speech recognition permission denied"
                @unknown default:
                    self?.isAvailable = false
                }
            }
        }
    }
    
    private func requestMicrophonePermission() {
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                self?.isAvailable = granted && (self?.speechRecognizer?.isAvailable ?? false)
                if !granted {
                    self?.errorMessage = "Microphone permission denied"
                }
            }
        }
    }
    
    private func startRecognition() throws {
        // Cancel any previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw VoiceCommandError.recognitionRequestFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        
        // Configure audio engine
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            DispatchQueue.main.async {
                if let result = result {
                    let recognizedText = result.bestTranscription.formattedString
                    self?.recognizedText = recognizedText
                    
                    if result.isFinal {
                        self?.processCommand(recognizedText)
                        self?.stopListening()
                    }
                }
                
                if let error = error {
                    self?.errorMessage = "Recognition error: \(error.localizedDescription)"
                    self?.stopListening()
                }
            }
        }
    }
    
    private func executeCommand(_ action: VoiceCommandAction, with parameter: String = "") {
        DispatchQueue.main.async {
            switch action {
            case .logPain:
                NotificationCenter.default.post(name: .voiceCommandLogPain, object: nil)
                
            case .ratePain:
                if let level = Int(parameter), level >= 0 && level <= 10 {
                    NotificationCenter.default.post(name: .voiceCommandRatePain, object: level)
                } else {
                    NotificationCenter.default.post(name: .voiceCommandLogPain, object: nil)
                }
                
            case .logMedication:
                NotificationCenter.default.post(name: .voiceCommandLogMedication, object: nil)
                
            case .setMedicationReminder:
                NotificationCenter.default.post(name: .voiceCommandSetReminder, object: nil)
                
            case .addJournalEntry:
                NotificationCenter.default.post(name: .voiceCommandAddJournal, object: nil)
                
            case .navigateToDashboard:
                NotificationCenter.default.post(name: .voiceCommandNavigate, object: "dashboard")
                
            case .navigateToAnalytics:
                NotificationCenter.default.post(name: .voiceCommandNavigate, object: "analytics")
                
            case .navigateToMedications:
                NotificationCenter.default.post(name: .voiceCommandNavigate, object: "medications")
                
            case .navigateToJournal:
                NotificationCenter.default.post(name: .voiceCommandNavigate, object: "journal")
                
            case .toggleDarkMode:
                ThemeManager.shared.toggleDarkMode()
                
            case .increaseTextSize:
                let currentScale = ThemeManager.shared.textScale
                let scales = TextScale.allCases
                if let currentIndex = scales.firstIndex(of: currentScale),
                   currentIndex < scales.count - 1 {
                    ThemeManager.shared.setTextScale(scales[currentIndex + 1])
                }
                
            case .decreaseTextSize:
                let currentScale = ThemeManager.shared.textScale
                let scales = TextScale.allCases
                if let currentIndex = scales.firstIndex(of: currentScale),
                   currentIndex > 0 {
                    ThemeManager.shared.setTextScale(scales[currentIndex - 1])
                }
                
            case .emergency:
                NotificationCenter.default.post(name: .voiceCommandEmergency, object: nil)
                
            case .stopListening:
                self.stopListening()
                
            case .showHelp:
                NotificationCenter.default.post(name: .voiceCommandShowHelp, object: nil)
            }
            
            HapticManager.shared.notification(.success)
        }
    }
    
    private func extractPainLevel(from text: String) -> Int? {
        let patterns = [
            "pain level (\\d+)",
            "rate (\\d+)",
            "level (\\d+)",
            "(\\d+) out of",
            "(\\d+)/10",
            "(\\d+)"
        ]
        
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                let range = NSRange(location: 0, length: text.utf16.count)
                if let match = regex.firstMatch(in: text, options: [], range: range) {
                    let numberRange = match.range(at: 1)
                    if let swiftRange = Range(numberRange, in: text) {
                        if let level = Int(text[swiftRange]), level >= 0 && level <= 10 {
                            return level
                        }
                    }
                }
            }
        }
        
        return nil
    }
}

// MARK: - Speech Recognizer Delegate

extension VoiceCommandManager: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        DispatchQueue.main.async {
            self.isAvailable = available
            if !available {
                self.stopListening()
            }
        }
    }
}

// MARK: - Supporting Types

struct VoiceCommand {
    let patterns: [String]
    let action: VoiceCommandAction
}

enum VoiceCommandAction {
    case logPain
    case ratePain
    case logMedication
    case setMedicationReminder
    case addJournalEntry
    case navigateToDashboard
    case navigateToAnalytics
    case navigateToMedications
    case navigateToJournal
    case toggleDarkMode
    case increaseTextSize
    case decreaseTextSize
    case emergency
    case stopListening
    case showHelp
}

enum VoiceCommandError: Error {
    case recognitionRequestFailed
    case audioEngineFailed
    case permissionDenied
    
    var localizedDescription: String {
        switch self {
        case .recognitionRequestFailed:
            return "Failed to create recognition request"
        case .audioEngineFailed:
            return "Audio engine failed to start"
        case .permissionDenied:
            return "Permission denied for speech recognition"
        }
    }
}

// MARK: - Notification Extensions

extension Notification.Name {
    static let voiceCommandLogPain = Notification.Name("voiceCommandLogPain")
    static let voiceCommandRatePain = Notification.Name("voiceCommandRatePain")
    static let voiceCommandLogMedication = Notification.Name("voiceCommandLogMedication")
    static let voiceCommandSetReminder = Notification.Name("voiceCommandSetReminder")
    static let voiceCommandAddJournal = Notification.Name("voiceCommandAddJournal")
    static let voiceCommandNavigate = Notification.Name("voiceCommandNavigate")
    static let voiceCommandEmergency = Notification.Name("voiceCommandEmergency")
    static let voiceCommandShowHelp = Notification.Name("voiceCommandShowHelp")
}

// MARK: - Voice Command Button

struct VoiceCommandButton: View {
    @StateObject private var voiceManager = VoiceCommandManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        Button(action: {
            voiceManager.toggleListening()
        }) {
            Image(systemName: voiceManager.isListening ? "mic.fill" : "mic")
                .font(.title2)
                .foregroundColor(voiceManager.isListening ? themeManager.colors.accent : themeManager.colors.textSecondary)
                .scaleEffect(voiceManager.isListening ? 1.2 : 1.0)
                .animation(themeManager.animations.spring, value: voiceManager.isListening)
        }
        .disabled(!voiceManager.isAvailable)
        .opacity(voiceManager.isAvailable ? 1.0 : 0.5)
    }
}

// MARK: - Voice Command Overlay

struct VoiceCommandOverlay: View {
    @StateObject private var voiceManager = VoiceCommandManager.shared
    @EnvironmentObject private var themeManager: ThemeManager
    
    var body: some View {
        if voiceManager.isListening {
            VStack(spacing: 16) {
                // Listening indicator
                HStack(spacing: 8) {
                    Image(systemName: "mic.fill")
                        .foregroundColor(themeManager.colors.accent)
                        .scaleEffect(1.2)
                    
                    Text("Listening...")
                        .font(themeManager.typography.headline)
                        .foregroundColor(themeManager.colors.textPrimary)
                }
                
                // Recognized text
                if !voiceManager.recognizedText.isEmpty {
                    Text(voiceManager.recognizedText)
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                // Stop button
                Button("Stop") {
                    voiceManager.stopListening()
                }
                .themedButton(style: .secondary)
            }
            .padding(24)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(themeManager.colors.surface)
                    .shadow(color: themeManager.colors.shadow, radius: 8)
            )
            .transition(.scale.combined(with: .opacity))
            .animation(themeManager.animations.spring, value: voiceManager.isListening)
        }
    }
}

// MARK: - Voice Command Help View

struct VoiceCommandHelpView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    
    private let commandCategories: [CommandCategory] = [
        CommandCategory(
            title: "Pain Tracking",
            commands: ["Log pain", "Pain level 5", "Rate pain"]
        ),
        CommandCategory(
            title: "Medication",
            commands: ["Take medication", "Set reminder", "Log medication"]
        ),
        CommandCategory(
            title: "Navigation",
            commands: ["Go to dashboard", "Show analytics", "Go to medications"]
        ),
        CommandCategory(
            title: "Settings",
            commands: ["Toggle dark mode", "Increase text size", "Decrease text size"]
        ),
        CommandCategory(
            title: "General",
            commands: ["Help", "Stop listening", "Emergency"]
        )
    ]
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(commandCategories, id: \.title) { category in
                        VStack(alignment: .leading, spacing: 12) {
                            Text(category.title)
                                .font(themeManager.typography.headline)
                                .foregroundColor(themeManager.colors.textPrimary)
                            
                            ForEach(category.commands, id: \.self) { command in
                                HStack {
                                    Text("\"\(command)\"")
                                        .font(themeManager.typography.body)
                                        .foregroundColor(themeManager.colors.textSecondary)
                                    
                                    Spacer()
                                }
                                .padding(.vertical, 4)
                            }
                        }
                        .themedCard()
                    }
                }
                .padding()
            }
            .navigationTitle("Voice Commands")
            .themedBackground()
        }
    }
}

struct CommandCategory {
    let title: String
    let commands: [String]
}

#Preview {
    VoiceCommandHelpView()
        .environmentObject(ThemeManager.shared)
}