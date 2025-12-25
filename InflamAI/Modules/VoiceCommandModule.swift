//
//  VoiceCommandModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import Speech
import AVFoundation
import Combine
import SwiftUI

// MARK: - Voice Command Models

struct VoiceCommand {
    let id: UUID
    let phrase: String
    let action: VoiceAction
    let parameters: [String: Any]?
    let confidence: Float
    let timestamp: Date
    
    init(phrase: String, action: VoiceAction, parameters: [String: Any]? = nil, confidence: Float = 1.0) {
        self.id = UUID()
        self.phrase = phrase
        self.action = action
        self.parameters = parameters
        self.confidence = confidence
        self.timestamp = Date()
    }
}

enum VoiceAction {
    // Navigation
    case navigateToSymptoms
    case navigateToMedications
    case navigateToAppointments
    case navigateToJournal
    case navigateToAnalytics
    case navigateToSettings
    case goBack
    case goHome
    
    // Symptom tracking
    case logSymptom(type: String, severity: Int)
    case logPain(location: String, severity: Int)
    case logMood(level: Int)
    case logFatigue(level: Int)
    case logStiffness(severity: Int)
    
    // Medication management
    case takeMedication(name: String)
    case skipMedication(name: String)
    case setMedicationReminder(name: String, time: String)
    case checkMedicationSchedule
    
    // Appointments
    case scheduleAppointment(doctor: String, date: String)
    case checkNextAppointment
    case cancelAppointment(id: String)
    
    // Journal entries
    case addJournalEntry(content: String)
    case searchJournal(query: String)
    
    // Health data
    case recordVitals(type: String, value: Double)
    case checkHealthTrends
    case exportHealthData
    
    // Emergency
    case emergencyCall
    case contactDoctor
    
    // General
    case help
    case repeat
    case cancel
    case unknown(phrase: String)
}

struct VoiceResponse {
    let text: String
    let shouldSpeak: Bool
    let action: (() -> Void)?
    
    init(text: String, shouldSpeak: Bool = true, action: (() -> Void)? = nil) {
        self.text = text
        self.shouldSpeak = shouldSpeak
        self.action = action
    }
}

struct VoiceSettings {
    var isEnabled: Bool = true
    var language: String = "en-US"
    var voiceSpeed: Float = 0.5
    var voicePitch: Float = 1.0
    var voiceVolume: Float = 1.0
    var wakeWordEnabled: Bool = true
    var wakeWord: String = "Hey InflamAI"
    var continuousListening: Bool = false
    var hapticFeedbackEnabled: Bool = true
    var confirmationRequired: Bool = true
    var privacyMode: Bool = false
}

enum VoiceCommandError: Error, LocalizedError {
    case speechRecognitionNotAvailable
    case speechRecognitionDenied
    case speechRecognitionRestricted
    case audioSessionError
    case recognitionTaskError
    case synthesisError
    case commandNotRecognized
    case networkError
    case timeout
    
    var errorDescription: String? {
        switch self {
        case .speechRecognitionNotAvailable:
            return "Speech recognition is not available on this device"
        case .speechRecognitionDenied:
            return "Speech recognition permission denied"
        case .speechRecognitionRestricted:
            return "Speech recognition is restricted on this device"
        case .audioSessionError:
            return "Audio session configuration error"
        case .recognitionTaskError:
            return "Speech recognition task error"
        case .synthesisError:
            return "Speech synthesis error"
        case .commandNotRecognized:
            return "Voice command not recognized"
        case .networkError:
            return "Network error during speech processing"
        case .timeout:
            return "Speech recognition timeout"
        }
    }
}

// MARK: - Voice Command Manager

@MainActor
class VoiceCommandManager: NSObject, ObservableObject {
    static let shared = VoiceCommandManager()
    
    @Published var isListening = false
    @Published var isProcessing = false
    @Published var lastCommand: VoiceCommand?
    @Published var lastResponse: VoiceResponse?
    @Published var settings = VoiceSettings()
    @Published var recognizedText = ""
    @Published var isAvailable = false
    @Published var error: VoiceCommandError?
    
    private let speechRecognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioSession: AVAudioSession?
    
    private let commandProcessor = VoiceCommandProcessor()
    private let nlpProcessor = NaturalLanguageProcessor()
    
    private var cancellables = Set<AnyCancellable>()
    private var listeningTimer: Timer?
    private var wakeWordDetector: WakeWordDetector?
    
    override init() {
        self.speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: settings.language))
        super.init()
        
        setupAudioSession()
        setupSpeechRecognizer()
        setupWakeWordDetector()
        checkAvailability()
    }
    
    // MARK: - Setup Methods
    
    private func setupAudioSession() {
        audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession?.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
            try audioSession?.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            self.error = .audioSessionError
        }
    }
    
    private func setupSpeechRecognizer() {
        speechRecognizer?.delegate = self
        speechSynthesizer.delegate = self
    }
    
    private func setupWakeWordDetector() {
        if settings.wakeWordEnabled {
            wakeWordDetector = WakeWordDetector(wakeWord: settings.wakeWord)
            wakeWordDetector?.delegate = self
        }
    }
    
    private func checkAvailability() {
        guard let speechRecognizer = speechRecognizer else {
            isAvailable = false
            error = .speechRecognitionNotAvailable
            return
        }
        
        isAvailable = speechRecognizer.isAvailable
        
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                switch status {
                case .authorized:
                    self?.isAvailable = true
                    self?.error = nil
                case .denied:
                    self?.isAvailable = false
                    self?.error = .speechRecognitionDenied
                case .restricted:
                    self?.isAvailable = false
                    self?.error = .speechRecognitionRestricted
                case .notDetermined:
                    self?.isAvailable = false
                @unknown default:
                    self?.isAvailable = false
                }
            }
        }
    }
    
    // MARK: - Voice Recognition Methods
    
    func startListening() {
        guard isAvailable && !isListening else { return }
        
        stopListening()
        
        do {
            let inputNode = audioEngine.inputNode
            
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else {
                error = .recognitionTaskError
                return
            }
            
            recognitionRequest.shouldReportPartialResults = true
            
            if #available(iOS 13, *) {
                recognitionRequest.requiresOnDeviceRecognition = settings.privacyMode
            }
            
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                DispatchQueue.main.async {
                    if let result = result {
                        self?.recognizedText = result.bestTranscription.formattedString
                        
                        if result.isFinal {
                            self?.processVoiceCommand(self?.recognizedText ?? "")
                            self?.stopListening()
                        }
                    }
                    
                    if let error = error {
                        self?.error = .recognitionTaskError
                        self?.stopListening()
                    }
                }
            }
            
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            
            isListening = true
            recognizedText = ""
            
            // Provide haptic feedback
            if settings.hapticFeedbackEnabled {
                let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                impactFeedback.impactOccurred()
            }
            
            // Set timeout for listening
            listeningTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: false) { [weak self] _ in
                self?.stopListening()
            }
            
        } catch {
            self.error = .audioSessionError
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        recognitionTask?.cancel()
        recognitionTask = nil
        
        isListening = false
        listeningTimer?.invalidate()
        listeningTimer = nil
    }
    
    private func processVoiceCommand(_ text: String) {
        isProcessing = true
        
        Task {
            do {
                let command = try await nlpProcessor.processCommand(text)
                let response = await commandProcessor.executeCommand(command)
                
                await MainActor.run {
                    self.lastCommand = command
                    self.lastResponse = response
                    self.isProcessing = false
                    
                    if response.shouldSpeak {
                        self.speak(response.text)
                    }
                    
                    response.action?()
                }
            } catch {
                await MainActor.run {
                    self.error = .commandNotRecognized
                    self.isProcessing = false
                }
            }
        }
    }
    
    // MARK: - Speech Synthesis
    
    func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: settings.language)
        utterance.rate = settings.voiceSpeed
        utterance.pitchMultiplier = settings.voicePitch
        utterance.volume = settings.voiceVolume
        
        speechSynthesizer.speak(utterance)
    }
    
    func stopSpeaking() {
        speechSynthesizer.stopSpeaking(at: .immediate)
    }
    
    // MARK: - Settings Management
    
    func updateSettings(_ newSettings: VoiceSettings) {
        settings = newSettings
        
        // Update speech recognizer language if changed
        if let newRecognizer = SFSpeechRecognizer(locale: Locale(identifier: settings.language)) {
            speechRecognizer?.delegate = nil
            newRecognizer.delegate = self
        }
        
        // Update wake word detector
        if settings.wakeWordEnabled {
            setupWakeWordDetector()
        } else {
            wakeWordDetector = nil
        }
    }
    
    // MARK: - Utility Methods
    
    func getAvailableLanguages() -> [String] {
        return SFSpeechRecognizer.supportedLocales().map { $0.identifier }
    }
    
    func getAvailableVoices() -> [AVSpeechSynthesisVoice] {
        return AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix(String(settings.language.prefix(2))) }
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension VoiceCommandManager: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        isAvailable = available
        if !available {
            stopListening()
        }
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension VoiceCommandManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        // Speech started
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        // Speech finished
        if settings.continuousListening {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.startListening()
            }
        }
    }
}

// MARK: - Natural Language Processor

class NaturalLanguageProcessor {
    private let commandPatterns: [String: VoiceAction] = [
        // Navigation patterns
        "go to symptoms|navigate to symptoms|show symptoms": .navigateToSymptoms,
        "go to medications|navigate to medications|show medications|show meds": .navigateToMedications,
        "go to appointments|navigate to appointments|show appointments": .navigateToAppointments,
        "go to journal|navigate to journal|show journal|open diary": .navigateToJournal,
        "go to analytics|navigate to analytics|show analytics|show reports": .navigateToAnalytics,
        "go to settings|navigate to settings|show settings|open settings": .navigateToSettings,
        "go back|navigate back|back": .goBack,
        "go home|navigate home|home": .goHome,
        
        // Symptom tracking patterns
        "log pain|record pain|add pain": .logPain(location: "", severity: 0),
        "log mood|record mood|add mood": .logMood(level: 0),
        "log fatigue|record fatigue|add fatigue|feeling tired": .logFatigue(level: 0),
        "log stiffness|record stiffness|add stiffness|feeling stiff": .logStiffness(severity: 0),
        
        // Medication patterns
        "take medication|took medication|mark medication taken": .takeMedication(name: ""),
        "skip medication|skip dose|missed medication": .skipMedication(name: ""),
        "set reminder|medication reminder|remind me": .setMedicationReminder(name: "", time: ""),
        "check medication|medication schedule|what medications": .checkMedicationSchedule,
        
        // Emergency patterns
        "emergency|call for help|urgent|help me": .emergencyCall,
        "contact doctor|call doctor|reach doctor": .contactDoctor,
        
        // General patterns
        "help|what can you do|commands": .help,
        "repeat|say again|what did you say": .repeat,
        "cancel|stop|never mind": .cancel
    ]
    
    func processCommand(_ text: String) async throws -> VoiceCommand {
        let normalizedText = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Extract parameters from the text
        let parameters = extractParameters(from: normalizedText)
        
        // Find matching action
        for (pattern, baseAction) in commandPatterns {
            let patterns = pattern.split(separator: "|").map { String($0) }
            
            for patternOption in patterns {
                if normalizedText.contains(patternOption) {
                    let action = enrichAction(baseAction, with: parameters, from: normalizedText)
                    return VoiceCommand(phrase: text, action: action, parameters: parameters)
                }
            }
        }
        
        // If no pattern matches, try to extract intent using more advanced NLP
        let action = try await extractIntentUsingNLP(from: normalizedText, parameters: parameters)
        return VoiceCommand(phrase: text, action: action, parameters: parameters)
    }
    
    private func extractParameters(from text: String) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        // Extract severity levels (1-10)
        let severityPattern = #"(\d+)\s*(?:out of 10|/10|level|severity)"#
        if let severityMatch = text.range(of: severityPattern, options: .regularExpression) {
            let severityString = String(text[severityMatch])
            if let severity = Int(severityString.components(separatedBy: CharacterSet.decimalDigits.inverted).joined()) {
                parameters["severity"] = min(max(severity, 1), 10)
            }
        }
        
        // Extract body locations
        let bodyParts = ["head", "neck", "shoulder", "arm", "elbow", "wrist", "hand", "finger", "back", "chest", "hip", "leg", "knee", "ankle", "foot", "toe"]
        for bodyPart in bodyParts {
            if text.contains(bodyPart) {
                parameters["location"] = bodyPart
                break
            }
        }
        
        // Extract medication names (common rheumatoid arthritis medications)
        let medications = ["methotrexate", "prednisone", "humira", "enbrel", "remicade", "plaquenil", "sulfasalazine", "leflunomide"]
        for medication in medications {
            if text.contains(medication) {
                parameters["medication"] = medication
                break
            }
        }
        
        // Extract time expressions
        let timePattern = #"(\d{1,2})\s*:?\s*(\d{2})?\s*(am|pm)?"#
        if let timeMatch = text.range(of: timePattern, options: .regularExpression) {
            parameters["time"] = String(text[timeMatch])
        }
        
        return parameters
    }
    
    private func enrichAction(_ baseAction: VoiceAction, with parameters: [String: Any], from text: String) -> VoiceAction {
        // FIXED: Use 0 for unspecified severity, not fake "5"
        // 0 = "severity not specified" - downstream should prompt user or filter from ML
        switch baseAction {
        case .logPain(_, _):
            let location = parameters["location"] as? String ?? ""
            let severity = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logPain(location: location, severity: severity)

        case .logMood(_):
            let level = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logMood(level: level)

        case .logFatigue(_):
            let level = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logFatigue(level: level)

        case .logStiffness(_):
            let severity = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logStiffness(severity: severity)
            
        case .takeMedication(_):
            let name = parameters["medication"] as? String ?? ""
            return .takeMedication(name: name)
            
        case .skipMedication(_):
            let name = parameters["medication"] as? String ?? ""
            return .skipMedication(name: name)
            
        case .setMedicationReminder(_, _):
            let name = parameters["medication"] as? String ?? ""
            let time = parameters["time"] as? String ?? ""
            return .setMedicationReminder(name: name, time: time)
            
        default:
            return baseAction
        }
    }
    
    private func extractIntentUsingNLP(from text: String, parameters: [String: Any]) async throws -> VoiceAction {
        // This is a simplified NLP approach. In a real app, you might use Core ML or a cloud-based NLP service
        // FIXED: Use 0 for unspecified severity, not fake "5"

        // Check for symptom-related keywords
        if text.contains("pain") || text.contains("hurt") || text.contains("ache") {
            let location = parameters["location"] as? String ?? ""
            let severity = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logPain(location: location, severity: severity)
        }

        if text.contains("tired") || text.contains("exhausted") || text.contains("fatigue") {
            let level = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logFatigue(level: level)
        }

        if text.contains("stiff") || text.contains("rigid") {
            let severity = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logStiffness(severity: severity)
        }

        if text.contains("mood") || text.contains("feeling") || text.contains("emotion") {
            let level = parameters["severity"] as? Int ?? 0  // 0 = unspecified
            return .logMood(level: level)
        }

        // If no intent is recognized, return unknown
        return .unknown(phrase: text)
    }
}

// MARK: - Voice Command Processor

class VoiceCommandProcessor {
    func executeCommand(_ command: VoiceCommand) async -> VoiceResponse {
        switch command.action {
        case .navigateToSymptoms:
            return VoiceResponse(text: "Navigating to symptoms tracking") {
                // Navigation logic here
            }
            
        case .navigateToMedications:
            return VoiceResponse(text: "Opening medications management")
            
        case .navigateToAppointments:
            return VoiceResponse(text: "Showing your appointments")
            
        case .navigateToJournal:
            return VoiceResponse(text: "Opening your health journal")
            
        case .navigateToAnalytics:
            return VoiceResponse(text: "Displaying health analytics")
            
        case .navigateToSettings:
            return VoiceResponse(text: "Opening settings")
            
        case .goBack:
            return VoiceResponse(text: "Going back")
            
        case .goHome:
            return VoiceResponse(text: "Returning to home screen")
            
        case .logPain(let location, let severity):
            let locationText = location.isEmpty ? "" : " in your \(location)"
            return VoiceResponse(text: "Logged pain\(locationText) with severity \(severity) out of 10")
            
        case .logMood(let level):
            return VoiceResponse(text: "Recorded mood level \(level) out of 10")
            
        case .logFatigue(let level):
            return VoiceResponse(text: "Logged fatigue level \(level) out of 10")
            
        case .logStiffness(let severity):
            return VoiceResponse(text: "Recorded stiffness severity \(severity) out of 10")
            
        case .takeMedication(let name):
            let medicationText = name.isEmpty ? "medication" : name
            return VoiceResponse(text: "Marked \(medicationText) as taken")
            
        case .skipMedication(let name):
            let medicationText = name.isEmpty ? "medication" : name
            return VoiceResponse(text: "Marked \(medicationText) as skipped")
            
        case .setMedicationReminder(let name, let time):
            let medicationText = name.isEmpty ? "medication" : name
            let timeText = time.isEmpty ? "" : " at \(time)"
            return VoiceResponse(text: "Set reminder for \(medicationText)\(timeText)")
            
        case .checkMedicationSchedule:
            return VoiceResponse(text: "You have 2 medications due today: Methotrexate at 8 AM and Prednisone at 6 PM")
            
        case .emergencyCall:
            return VoiceResponse(text: "Initiating emergency call") {
                // Emergency call logic
            }
            
        case .contactDoctor:
            return VoiceResponse(text: "Contacting your doctor")
            
        case .help:
            return VoiceResponse(text: "You can say things like 'log pain in my knee level 7', 'take methotrexate', 'go to symptoms', or 'check medication schedule'")
            
        case .repeat:
            return VoiceResponse(text: "I'm sorry, could you please repeat that?")
            
        case .cancel:
            return VoiceResponse(text: "Cancelled", shouldSpeak: false)
            
        case .unknown(let phrase):
            return VoiceResponse(text: "I didn't understand '\(phrase)'. Try saying 'help' to see what I can do")
            
        default:
            return VoiceResponse(text: "Command not implemented yet")
        }
    }
}

// MARK: - Wake Word Detector

class WakeWordDetector {
    weak var delegate: WakeWordDetectorDelegate?
    private let wakeWord: String
    private var isListening = false
    
    init(wakeWord: String) {
        self.wakeWord = wakeWord
    }
    
    func startListening() {
        isListening = true
        // Implement wake word detection logic
        // This would typically use a lightweight speech recognition model
        // that runs continuously in the background
    }
    
    func stopListening() {
        isListening = false
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        // Process audio buffer for wake word detection
        // This is a simplified implementation
        // In a real app, you would use a specialized wake word detection library
    }
}

protocol WakeWordDetectorDelegate: AnyObject {
    func wakeWordDetected()
}

extension VoiceCommandManager: WakeWordDetectorDelegate {
    func wakeWordDetected() {
        if settings.wakeWordEnabled && !isListening {
            startListening()
        }
    }
}