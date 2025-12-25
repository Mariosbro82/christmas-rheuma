//
//  VoiceCommandEngine.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import Foundation
import Speech
import AVFoundation
import Combine
import NaturalLanguage

class VoiceCommandEngine: NSObject, ObservableObject {
    static let shared = VoiceCommandEngine()
    
    @Published var isListening = false
    @Published var isProcessing = false
    @Published var lastCommand = ""
    @Published var commandHistory: [VoiceCommand] = []
    @Published var recognizedText = ""
    @Published var confidence: Float = 0.0
    @Published var isEnabled = true
    @Published var voiceFeedbackEnabled = true
    @Published var hapticFeedbackEnabled = true
    @Published var smartSuggestionsEnabled = true
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    
    private var cancellables = Set<AnyCancellable>()
    private let nlProcessor = NLLanguageRecognizer()
    private let intentClassifier = NLModel()
    
    // Command patterns and responses
    private let commandPatterns: [CommandPattern] = [
        // Pain level commands
        CommandPattern(
            patterns: ["pain level", "set pain", "my pain is", "pain at"],
            intent: .setPainLevel,
            parameters: ["level"],
            examples: ["Set pain level to 7", "My pain is 5", "Pain level 8"]
        ),
        
        // Body region commands
        CommandPattern(
            patterns: ["pain in", "hurts in", "add pain", "mark pain"],
            intent: .addPainRegion,
            parameters: ["region", "level"],
            examples: ["Pain in lower back", "My knee hurts", "Add pain in shoulder"]
        ),
        
        // Remove pain commands
        CommandPattern(
            patterns: ["remove pain", "clear pain", "no pain", "pain gone"],
            intent: .removePainRegion,
            parameters: ["region"],
            examples: ["Remove pain from back", "Clear shoulder pain", "No more knee pain"]
        ),
        
        // Save and log commands
        CommandPattern(
            patterns: ["save entry", "log pain", "record pain", "save this"],
            intent: .saveEntry,
            parameters: [],
            examples: ["Save pain entry", "Log this pain", "Record current pain"]
        ),
        
        // Summary commands
        CommandPattern(
            patterns: ["pain summary", "how am i", "pain report", "show summary"],
            intent: .getSummary,
            parameters: [],
            examples: ["Give me pain summary", "How am I doing?", "Show pain report"]
        ),
        
        // Medication commands
        CommandPattern(
            patterns: ["take medication", "took medicine", "medication taken", "took pill"],
            intent: .logMedication,
            parameters: ["medication", "dose"],
            examples: ["Took ibuprofen", "Take medication", "Took 200mg ibuprofen"]
        ),
        
        // Emergency commands
        CommandPattern(
            patterns: ["emergency", "help me", "severe pain", "call doctor"],
            intent: .emergency,
            parameters: [],
            examples: ["This is an emergency", "Help me", "Severe pain emergency"]
        ),
        
        // Analysis commands
        CommandPattern(
            patterns: ["analyze pain", "pain patterns", "what patterns", "pain analysis"],
            intent: .analyzePatterns,
            parameters: [],
            examples: ["Analyze my pain patterns", "What patterns do you see?", "Pain analysis"]
        )
    ]
    
    // Body region mappings
    private let bodyRegionMappings: [String: BodyRegion] = [
        // Head and neck
        "head": .head,
        "neck": .neck,
        "cervical": .cervicalSpine,
        "cervical spine": .cervicalSpine,
        
        // Torso
        "chest": .chest,
        "upper back": .upperBack,
        "middle back": .middleBack,
        "lower back": .lowerBack,
        "thoracic": .thoracicSpine,
        "thoracic spine": .thoracicSpine,
        "lumbar": .lumbarSpine,
        "lumbar spine": .lumbarSpine,
        "sacral": .sacralSpine,
        "sacral spine": .sacralSpine,
        
        // Arms
        "left shoulder": .leftShoulder,
        "right shoulder": .rightShoulder,
        "shoulder": .leftShoulder, // Default to left
        "left arm": .leftArm,
        "right arm": .rightArm,
        "arm": .leftArm, // Default to left
        "left elbow": .leftElbow,
        "right elbow": .rightElbow,
        "elbow": .leftElbow, // Default to left
        "left wrist": .leftWrist,
        "right wrist": .rightWrist,
        "wrist": .leftWrist, // Default to left
        "left hand": .leftHand,
        "right hand": .rightHand,
        "hand": .leftHand, // Default to left
        
        // Legs
        "left hip": .leftHip,
        "right hip": .rightHip,
        "hip": .leftHip, // Default to left
        "left thigh": .leftThigh,
        "right thigh": .rightThigh,
        "thigh": .leftThigh, // Default to left
        "left knee": .leftKnee,
        "right knee": .rightKnee,
        "knee": .leftKnee, // Default to left
        "left calf": .leftCalf,
        "right calf": .rightCalf,
        "calf": .leftCalf, // Default to left
        "left ankle": .leftAnkle,
        "right ankle": .rightAnkle,
        "ankle": .leftAnkle, // Default to left
        "left foot": .leftFoot,
        "right foot": .rightFoot,
        "foot": .leftFoot // Default to left
    ]
    
    override init() {
        super.init()
        setupSpeechRecognition()
        setupAudioSession()
        loadUserPreferences()
    }
    
    // MARK: - Setup
    
    private func setupSpeechRecognition() {
        guard let speechRecognizer = speechRecognizer else {
            print("Speech recognizer not available")
            return
        }
        
        speechRecognizer.delegate = self
        
        SFSpeechRecognizer.requestAuthorization { [weak self] authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    self?.isEnabled = true
                case .denied, .restricted, .notDetermined:
                    self?.isEnabled = false
                @unknown default:
                    self?.isEnabled = false
                }
            }
        }
    }
    
    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session setup failed: \(error)")
        }
    }
    
    private func loadUserPreferences() {
        voiceFeedbackEnabled = UserDefaults.standard.bool(forKey: "voiceFeedbackEnabled")
        hapticFeedbackEnabled = UserDefaults.standard.bool(forKey: "hapticFeedbackEnabled")
        smartSuggestionsEnabled = UserDefaults.standard.bool(forKey: "smartSuggestionsEnabled")
    }
    
    // MARK: - Voice Recognition
    
    func startListening() {
        guard isEnabled && !isListening else { return }
        
        // Cancel any ongoing recognition
        stopListening()
        
        do {
            // Create recognition request
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else { return }
            
            recognitionRequest.shouldReportPartialResults = true
            recognitionRequest.requiresOnDeviceRecognition = true
            
            // Create audio input node
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
            
            // Start audio engine
            audioEngine.prepare()
            try audioEngine.start()
            
            // Start recognition task
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                self?.handleRecognitionResult(result: result, error: error)
            }
            
            DispatchQueue.main.async {
                self.isListening = true
                self.recognizedText = ""
            }
            
            // Provide haptic feedback
            if hapticFeedbackEnabled {
                let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                impactFeedback.impactOccurred()
            }
            
        } catch {
            print("Failed to start listening: \(error)")
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        recognitionRequest = nil
        recognitionTask = nil
        
        DispatchQueue.main.async {
            self.isListening = false
        }
    }
    
    private func handleRecognitionResult(result: SFSpeechRecognitionResult?, error: Error?) {
        if let error = error {
            print("Recognition error: \(error)")
            stopListening()
            return
        }
        
        guard let result = result else { return }
        
        DispatchQueue.main.async {
            self.recognizedText = result.bestTranscription.formattedString
            self.confidence = result.bestTranscription.averageConfidence
            
            if result.isFinal {
                self.processVoiceCommand(self.recognizedText)
                self.stopListening()
            }
        }
    }
    
    // MARK: - Command Processing
    
    func processVoiceCommand(_ text: String) {
        guard !text.isEmpty else { return }
        
        isProcessing = true
        lastCommand = text
        
        // Find matching command pattern
        let matchedCommand = findMatchingCommand(text)
        
        // Create voice command record
        let voiceCommand = VoiceCommand(
            text: text,
            intent: matchedCommand?.intent ?? .unknown,
            confidence: confidence,
            timestamp: Date(),
            processed: matchedCommand != nil
        )
        
        commandHistory.append(voiceCommand)
        
        // Execute command
        if let command = matchedCommand {
            executeCommand(command, originalText: text)
        } else {
            handleUnknownCommand(text)
        }
        
        isProcessing = false
    }
    
    private func findMatchingCommand(_ text: String) -> MatchedCommand? {
        let lowercaseText = text.lowercased()
        
        for pattern in commandPatterns {
            for patternText in pattern.patterns {
                if lowercaseText.contains(patternText.lowercased()) {
                    let parameters = extractParameters(from: lowercaseText, pattern: pattern)
                    return MatchedCommand(
                        pattern: pattern,
                        parameters: parameters,
                        confidence: confidence
                    )
                }
            }
        }
        
        return nil
    }
    
    private func extractParameters(from text: String, pattern: CommandPattern) -> [String: String] {
        var parameters: [String: String] = [:]
        
        switch pattern.intent {
        case .setPainLevel:
            if let level = extractPainLevel(from: text) {
                parameters["level"] = String(level)
            }
            
        case .addPainRegion, .removePainRegion:
            if let region = extractBodyRegion(from: text) {
                parameters["region"] = region.rawValue
            }
            if let level = extractPainLevel(from: text) {
                parameters["level"] = String(level)
            }
            
        case .logMedication:
            if let medication = extractMedication(from: text) {
                parameters["medication"] = medication
            }
            if let dose = extractDose(from: text) {
                parameters["dose"] = dose
            }
            
        default:
            break
        }
        
        return parameters
    }
    
    private func extractPainLevel(from text: String) -> Int? {
        // Look for numbers 0-10
        let numberRegex = try! NSRegularExpression(pattern: "\\b([0-9]|10)\\b")
        let matches = numberRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        
        for match in matches {
            if let range = Range(match.range, in: text) {
                if let level = Int(String(text[range])), level >= 0 && level <= 10 {
                    return level
                }
            }
        }
        
        // Look for word numbers
        let wordNumbers = [
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        ]
        
        for (word, number) in wordNumbers {
            if text.lowercased().contains(word) {
                return number
            }
        }
        
        return nil
    }
    
    private func extractBodyRegion(from text: String) -> BodyRegion? {
        let lowercaseText = text.lowercased()
        
        // Find the longest matching region name
        var bestMatch: (region: BodyRegion, length: Int)?
        
        for (regionName, region) in bodyRegionMappings {
            if lowercaseText.contains(regionName.lowercased()) {
                if bestMatch == nil || regionName.count > bestMatch!.length {
                    bestMatch = (region, regionName.count)
                }
            }
        }
        
        return bestMatch?.region
    }
    
    private func extractMedication(from text: String) -> String? {
        let commonMedications = [
            "ibuprofen", "advil", "motrin", "tylenol", "acetaminophen",
            "aspirin", "naproxen", "aleve", "tramadol", "codeine",
            "morphine", "oxycodone", "hydrocodone", "gabapentin", "lyrica"
        ]
        
        let lowercaseText = text.lowercased()
        
        for medication in commonMedications {
            if lowercaseText.contains(medication) {
                return medication.capitalized
            }
        }
        
        return nil
    }
    
    private func extractDose(from text: String) -> String? {
        // Look for dose patterns like "200mg", "2 tablets", "one pill"
        let doseRegex = try! NSRegularExpression(pattern: "\\b(\\d+)\\s*(mg|milligrams?|tablets?|pills?|capsules?)\\b")
        let matches = doseRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        
        if let match = matches.first {
            if let range = Range(match.range, in: text) {
                return String(text[range])
            }
        }
        
        return nil
    }
    
    // MARK: - Command Execution
    
    private func executeCommand(_ command: MatchedCommand, originalText: String) {
        switch command.pattern.intent {
        case .setPainLevel:
            executePainLevelCommand(command)
            
        case .addPainRegion:
            executeAddPainRegionCommand(command)
            
        case .removePainRegion:
            executeRemovePainRegionCommand(command)
            
        case .saveEntry:
            executeSaveEntryCommand()
            
        case .getSummary:
            executeGetSummaryCommand()
            
        case .logMedication:
            executeLogMedicationCommand(command)
            
        case .emergency:
            executeEmergencyCommand()
            
        case .analyzePatterns:
            executeAnalyzePatternsCommand()
            
        case .unknown:
            handleUnknownCommand(originalText)
        }
    }
    
    private func executePainLevelCommand(_ command: MatchedCommand) {
        guard let levelString = command.parameters["level"],
              let level = Int(levelString) else {
            speakResponse("I couldn't understand the pain level. Please say a number from 0 to 10.")
            return
        }
        
        // Update global pain level
        NotificationCenter.default.post(
            name: .voiceCommandSetPainLevel,
            object: nil,
            userInfo: ["level": level]
        )
        
        speakResponse("Pain level set to \(level) out of 10.")
        
        // Provide contextual feedback
        if level >= 8 {
            speakResponse("That's a high pain level. Consider taking medication or contacting your healthcare provider.")
        } else if level <= 2 {
            speakResponse("That's a low pain level. Great to hear you're feeling better.")
        }
    }
    
    private func executeAddPainRegionCommand(_ command: MatchedCommand) {
        guard let regionString = command.parameters["region"],
              let region = BodyRegion(rawValue: regionString) else {
            speakResponse("I couldn't identify the body region. Please try again.")
            return
        }
        
        let level = command.parameters["level"].flatMap(Int.init) ?? 5
        
        NotificationCenter.default.post(
            name: .voiceCommandAddPainRegion,
            object: nil,
            userInfo: ["region": region, "level": level]
        )
        
        speakResponse("Added pain in \(region.displayName) at level \(level).")
    }
    
    private func executeRemovePainRegionCommand(_ command: MatchedCommand) {
        guard let regionString = command.parameters["region"],
              let region = BodyRegion(rawValue: regionString) else {
            speakResponse("I couldn't identify the body region to remove. Please try again.")
            return
        }
        
        NotificationCenter.default.post(
            name: .voiceCommandRemovePainRegion,
            object: nil,
            userInfo: ["region": region]
        )
        
        speakResponse("Removed pain from \(region.displayName).")
    }
    
    private func executeSaveEntryCommand() {
        NotificationCenter.default.post(name: .voiceCommandSaveEntry, object: nil)
        speakResponse("Pain entry saved successfully.")
    }
    
    private func executeGetSummaryCommand() {
        // Get current pain summary
        let painData = PainDataStore.shared.painEntries
        
        if painData.isEmpty {
            speakResponse("No pain data available yet. Start tracking your pain to get insights.")
            return
        }
        
        let recentData = Array(painData.suffix(7))
        let averagePain = recentData.map { $0.painLevel }.reduce(0, +) / Double(recentData.count)
        let maxPain = recentData.map { $0.painLevel }.max() ?? 0
        
        let summary = "Your average pain level this week is \(String(format: "%.1f", averagePain)) out of 10, with a maximum of \(String(format: "%.1f", maxPain))."
        
        speakResponse(summary)
        
        // Add AI insights if available
        if smartSuggestionsEnabled {
            AIMLEngine.shared.generatePainPredictions(painData: painData) { [weak self] predictions in
                if let nextPrediction = predictions.first {
                    let predictionText = "Based on your patterns, I predict your pain level will be around \(String(format: "%.1f", nextPrediction.predictedLevel)) in the next 24 hours."
                    self?.speakResponse(predictionText)
                }
            }
        }
    }
    
    private func executeLogMedicationCommand(_ command: MatchedCommand) {
        let medication = command.parameters["medication"] ?? "medication"
        let dose = command.parameters["dose"] ?? ""
        
        NotificationCenter.default.post(
            name: .voiceCommandLogMedication,
            object: nil,
            userInfo: ["medication": medication, "dose": dose]
        )
        
        let response = dose.isEmpty ? "Logged \(medication)." : "Logged \(dose) of \(medication)."
        speakResponse(response)
    }
    
    private func executeEmergencyCommand() {
        NotificationCenter.default.post(name: .voiceCommandEmergency, object: nil)
        speakResponse("Emergency mode activated. Consider contacting your healthcare provider or emergency services if needed.")
    }
    
    private func executeAnalyzePatternsCommand() {
        let painData = PainDataStore.shared.painEntries
        
        if painData.count < 3 {
            speakResponse("Not enough data for pattern analysis. Keep tracking your pain for better insights.")
            return
        }
        
        AIMLEngine.shared.analyzePainPatterns(painData: painData) { [weak self] patterns in
            if patterns.isEmpty {
                self?.speakResponse("No significant patterns detected in your pain data.")
            } else {
                let topPattern = patterns.first!
                self?.speakResponse("I found a \(topPattern.type.displayName.lowercased()): \(topPattern.description)")
            }
        }
    }
    
    private func handleUnknownCommand(_ text: String) {
        speakResponse("I didn't understand that command. Try saying something like 'set pain level to 5' or 'pain in lower back'.")
        
        // Suggest similar commands if smart suggestions are enabled
        if smartSuggestionsEnabled {
            let suggestions = generateCommandSuggestions(for: text)
            if !suggestions.isEmpty {
                let suggestionText = "Did you mean: \(suggestions.first!)"
                speakResponse(suggestionText)
            }
        }
    }
    
    private func generateCommandSuggestions(for text: String) -> [String] {
        let lowercaseText = text.lowercased()
        var suggestions: [String] = []
        
        // Simple keyword matching for suggestions
        if lowercaseText.contains("pain") {
            suggestions.append("Set pain level to 5")
            suggestions.append("Pain in lower back")
        }
        
        if lowercaseText.contains("medication") || lowercaseText.contains("medicine") {
            suggestions.append("Took ibuprofen")
            suggestions.append("Log medication")
        }
        
        if lowercaseText.contains("save") || lowercaseText.contains("record") {
            suggestions.append("Save pain entry")
        }
        
        return suggestions
    }
    
    // MARK: - Voice Feedback
    
    private func speakResponse(_ text: String) {
        guard voiceFeedbackEnabled else { return }
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.0
        utterance.volume = 0.8
        
        synthesizer.speak(utterance)
        
        // Also provide haptic feedback
        if hapticFeedbackEnabled {
            let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
            impactFeedback.impactOccurred()
        }
    }
    
    // MARK: - Settings
    
    func updateVoiceFeedback(_ enabled: Bool) {
        voiceFeedbackEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "voiceFeedbackEnabled")
    }
    
    func updateHapticFeedback(_ enabled: Bool) {
        hapticFeedbackEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "hapticFeedbackEnabled")
    }
    
    func updateSmartSuggestions(_ enabled: Bool) {
        smartSuggestionsEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "smartSuggestionsEnabled")
    }
    
    // MARK: - Quick Commands
    
    func quickPainLevel(_ level: Int) {
        processVoiceCommand("Set pain level to \(level)")
    }
    
    func quickSaveEntry() {
        processVoiceCommand("Save pain entry")
    }
    
    func quickSummary() {
        processVoiceCommand("Give me pain summary")
    }
    
    func quickEmergency() {
        processVoiceCommand("Emergency")
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension VoiceCommandEngine: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        DispatchQueue.main.async {
            self.isEnabled = available
        }
    }
}

// MARK: - Data Models

struct VoiceCommand {
    let id = UUID()
    let text: String
    let intent: CommandIntent
    let confidence: Float
    let timestamp: Date
    let processed: Bool
}

struct CommandPattern {
    let patterns: [String]
    let intent: CommandIntent
    let parameters: [String]
    let examples: [String]
}

struct MatchedCommand {
    let pattern: CommandPattern
    let parameters: [String: String]
    let confidence: Float
}

enum CommandIntent {
    case setPainLevel
    case addPainRegion
    case removePainRegion
    case saveEntry
    case getSummary
    case logMedication
    case emergency
    case analyzePatterns
    case unknown
    
    var displayName: String {
        switch self {
        case .setPainLevel: return "Set Pain Level"
        case .addPainRegion: return "Add Pain Region"
        case .removePainRegion: return "Remove Pain Region"
        case .saveEntry: return "Save Entry"
        case .getSummary: return "Get Summary"
        case .logMedication: return "Log Medication"
        case .emergency: return "Emergency"
        case .analyzePatterns: return "Analyze Patterns"
        case .unknown: return "Unknown"
        }
    }
    
    var icon: String {
        switch self {
        case .setPainLevel: return "slider.horizontal.3"
        case .addPainRegion: return "plus.circle.fill"
        case .removePainRegion: return "minus.circle.fill"
        case .saveEntry: return "square.and.arrow.down.fill"
        case .getSummary: return "chart.bar.fill"
        case .logMedication: return "pills.fill"
        case .emergency: return "exclamationmark.triangle.fill"
        case .analyzePatterns: return "brain.head.profile"
        case .unknown: return "questionmark.circle.fill"
        }
    }
}

// MARK: - Notification Names

extension Notification.Name {
    static let voiceCommandSetPainLevel = Notification.Name("voiceCommandSetPainLevel")
    static let voiceCommandAddPainRegion = Notification.Name("voiceCommandAddPainRegion")
    static let voiceCommandRemovePainRegion = Notification.Name("voiceCommandRemovePainRegion")
    static let voiceCommandSaveEntry = Notification.Name("voiceCommandSaveEntry")
    static let voiceCommandLogMedication = Notification.Name("voiceCommandLogMedication")
    static let voiceCommandEmergency = Notification.Name("voiceCommandEmergency")
}