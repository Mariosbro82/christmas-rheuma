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
import HealthKit
import UserNotifications

// MARK: - Voice Command Models

struct VoiceCommand {
    let id: UUID
    let phrase: String
    let action: VoiceAction
    let parameters: [String: Any]?
    let confidence: Float
    let timestamp: Date
    let language: String
    let context: VoiceContext?
}

enum VoiceAction: String, CaseIterable {
    // Pain Management
    case logPain = "log_pain"
    case updatePainLevel = "update_pain_level"
    case describePain = "describe_pain"
    case painLocation = "pain_location"
    
    // Medication
    case takeMedication = "take_medication"
    case skipMedication = "skip_medication"
    case setMedicationReminder = "set_medication_reminder"
    case checkMedicationSchedule = "check_medication_schedule"
    
    // Mood & Symptoms
    case logMood = "log_mood"
    case logSymptom = "log_symptom"
    case logFatigue = "log_fatigue"
    case logStiffness = "log_stiffness"
    
    // Exercise & Activity
    case startExercise = "start_exercise"
    case stopExercise = "stop_exercise"
    case logActivity = "log_activity"
    case setExerciseReminder = "set_exercise_reminder"
    
    // Sleep
    case logSleep = "log_sleep"
    case setSleepGoal = "set_sleep_goal"
    case checkSleepQuality = "check_sleep_quality"
    
    // Navigation
    case openSection = "open_section"
    case goBack = "go_back"
    case showDashboard = "show_dashboard"
    case showReports = "show_reports"
    
    // Emergency
    case emergencyCall = "emergency_call"
    case contactDoctor = "contact_doctor"
    case findNearestHospital = "find_nearest_hospital"
    
    // Settings
    case enableNotifications = "enable_notifications"
    case disableNotifications = "disable_notifications"
    case changeLanguage = "change_language"
    case adjustVolume = "adjust_volume"
    
    // Data & Reports
    case generateReport = "generate_report"
    case exportData = "export_data"
    case shareWithDoctor = "share_with_doctor"
    case showTrends = "show_trends"
    
    // Meditation & Wellness
    case startMeditation = "start_meditation"
    case playRelaxationMusic = "play_relaxation_music"
    case breathingExercise = "breathing_exercise"
    
    // Help & Information
    case getHelp = "get_help"
    case explainFeature = "explain_feature"
    case showTutorial = "show_tutorial"
    case repeatLastAction = "repeat_last_action"
    
    var displayName: String {
        switch self {
        case .logPain: return "Log Pain"
        case .updatePainLevel: return "Update Pain Level"
        case .describePain: return "Describe Pain"
        case .painLocation: return "Pain Location"
        case .takeMedication: return "Take Medication"
        case .skipMedication: return "Skip Medication"
        case .setMedicationReminder: return "Set Medication Reminder"
        case .checkMedicationSchedule: return "Check Medication Schedule"
        case .logMood: return "Log Mood"
        case .logSymptom: return "Log Symptom"
        case .logFatigue: return "Log Fatigue"
        case .logStiffness: return "Log Stiffness"
        case .startExercise: return "Start Exercise"
        case .stopExercise: return "Stop Exercise"
        case .logActivity: return "Log Activity"
        case .setExerciseReminder: return "Set Exercise Reminder"
        case .logSleep: return "Log Sleep"
        case .setSleepGoal: return "Set Sleep Goal"
        case .checkSleepQuality: return "Check Sleep Quality"
        case .openSection: return "Open Section"
        case .goBack: return "Go Back"
        case .showDashboard: return "Show Dashboard"
        case .showReports: return "Show Reports"
        case .emergencyCall: return "Emergency Call"
        case .contactDoctor: return "Contact Doctor"
        case .findNearestHospital: return "Find Nearest Hospital"
        case .enableNotifications: return "Enable Notifications"
        case .disableNotifications: return "Disable Notifications"
        case .changeLanguage: return "Change Language"
        case .adjustVolume: return "Adjust Volume"
        case .generateReport: return "Generate Report"
        case .exportData: return "Export Data"
        case .shareWithDoctor: return "Share with Doctor"
        case .showTrends: return "Show Trends"
        case .startMeditation: return "Start Meditation"
        case .playRelaxationMusic: return "Play Relaxation Music"
        case .breathingExercise: return "Breathing Exercise"
        case .getHelp: return "Get Help"
        case .explainFeature: return "Explain Feature"
        case .showTutorial: return "Show Tutorial"
        case .repeatLastAction: return "Repeat Last Action"
        }
    }
    
    var category: VoiceActionCategory {
        switch self {
        case .logPain, .updatePainLevel, .describePain, .painLocation:
            return .pain
        case .takeMedication, .skipMedication, .setMedicationReminder, .checkMedicationSchedule:
            return .medication
        case .logMood, .logSymptom, .logFatigue, .logStiffness:
            return .symptoms
        case .startExercise, .stopExercise, .logActivity, .setExerciseReminder:
            return .exercise
        case .logSleep, .setSleepGoal, .checkSleepQuality:
            return .sleep
        case .openSection, .goBack, .showDashboard, .showReports:
            return .navigation
        case .emergencyCall, .contactDoctor, .findNearestHospital:
            return .emergency
        case .enableNotifications, .disableNotifications, .changeLanguage, .adjustVolume:
            return .settings
        case .generateReport, .exportData, .shareWithDoctor, .showTrends:
            return .data
        case .startMeditation, .playRelaxationMusic, .breathingExercise:
            return .wellness
        case .getHelp, .explainFeature, .showTutorial, .repeatLastAction:
            return .help
        }
    }
}

enum VoiceActionCategory: String, CaseIterable {
    case pain = "pain"
    case medication = "medication"
    case symptoms = "symptoms"
    case exercise = "exercise"
    case sleep = "sleep"
    case navigation = "navigation"
    case emergency = "emergency"
    case settings = "settings"
    case data = "data"
    case wellness = "wellness"
    case help = "help"
    
    var displayName: String {
        switch self {
        case .pain: return "Pain Management"
        case .medication: return "Medication"
        case .symptoms: return "Symptoms"
        case .exercise: return "Exercise"
        case .sleep: return "Sleep"
        case .navigation: return "Navigation"
        case .emergency: return "Emergency"
        case .settings: return "Settings"
        case .data: return "Data & Reports"
        case .wellness: return "Wellness"
        case .help: return "Help"
        }
    }
}

struct VoiceContext {
    let currentScreen: String?
    let userState: UserState?
    let timeOfDay: TimeOfDay
    let recentActions: [VoiceAction]
    let environmentalFactors: EnvironmentalFactors?
}

enum UserState {
    case active
    case resting
    case exercising
    case sleeping
    case inPain
    case takingMedication
}

enum TimeOfDay {
    case morning
    case afternoon
    case evening
    case night
    
    static func current() -> TimeOfDay {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 6..<12: return .morning
        case 12..<17: return .afternoon
        case 17..<22: return .evening
        default: return .night
        }
    }
}

struct EnvironmentalFactors {
    let noiseLevel: Double
    let location: String?
    let isMoving: Bool
    let batteryLevel: Double
}

struct VoiceCommandPattern {
    let action: VoiceAction
    let patterns: [String]
    let parameters: [VoiceParameter]
    let requiredConfidence: Float
    let contextSensitive: Bool
}

struct VoiceParameter {
    let name: String
    let type: ParameterType
    let required: Bool
    let defaultValue: Any?
    let validValues: [String]?
}

enum ParameterType {
    case number
    case text
    case boolean
    case date
    case time
    case duration
    case painLevel
    case mood
    case bodyPart
    case medication
}

// MARK: - Voice Recognition States

enum VoiceRecognitionState {
    case idle
    case listening
    case processing
    case responding
    case error(VoiceError)
}

enum VoiceError: Error, LocalizedError {
    case speechRecognitionNotAvailable
    case microphonePermissionDenied
    case speechRecognitionDenied
    case audioSessionError
    case recognitionFailed
    case commandNotRecognized
    case parameterMissing(String)
    case actionExecutionFailed
    
    var errorDescription: String? {
        switch self {
        case .speechRecognitionNotAvailable:
            return "Speech recognition is not available on this device"
        case .microphonePermissionDenied:
            return "Microphone permission is required for voice commands"
        case .speechRecognitionDenied:
            return "Speech recognition permission is required"
        case .audioSessionError:
            return "Audio session error occurred"
        case .recognitionFailed:
            return "Failed to recognize speech"
        case .commandNotRecognized:
            return "Command not recognized. Try saying 'help' for available commands"
        case .parameterMissing(let parameter):
            return "Missing required parameter: \(parameter)"
        case .actionExecutionFailed:
            return "Failed to execute the requested action"
        }
    }
}

// MARK: - Voice Response Models

struct VoiceResponse {
    let text: String
    let audioURL: URL?
    let followUpQuestions: [String]?
    let suggestedActions: [VoiceAction]?
    let requiresUserInput: Bool
    let priority: ResponsePriority
}

enum ResponsePriority {
    case low
    case normal
    case high
    case urgent
}

// MARK: - Voice Command Manager

@MainActor
class VoiceCommandManager: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var recognitionState: VoiceRecognitionState = .idle
    @Published var isListening: Bool = false
    @Published var recognizedText: String = ""
    @Published var lastCommand: VoiceCommand?
    @Published var lastResponse: VoiceResponse?
    @Published var isVoiceEnabled: Bool = true
    @Published var currentLanguage: String = "en-US"
    @Published var voiceSettings: VoiceSettings = VoiceSettings()
    @Published var commandHistory: [VoiceCommand] = []
    @Published var availableCommands: [VoiceCommandPattern] = []
    
    // MARK: - Private Properties
    private let speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    // Command Processing
    private let commandProcessor = VoiceCommandProcessor()
    private let nlpEngine = NaturalLanguageProcessor()
    private let contextAnalyzer = VoiceContextAnalyzer()
    private let responseGenerator = VoiceResponseGenerator()
    
    // Integration Managers
    private weak var healthDataManager: HealthDataManager?
    private weak var medicationManager: MedicationManager?
    private weak var painTrackingManager: PainTrackingManager?
    private weak var navigationController: UINavigationController?
    
    // Settings
    private var cancellables = Set<AnyCancellable>()
    
    override init() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: currentLanguage))
        super.init()
        
        setupVoiceCommands()
        setupAudioSession()
        loadSettings()
        
        speechRecognizer?.delegate = self
        speechSynthesizer.delegate = self
    }
    
    // MARK: - Setup
    
    private func setupVoiceCommands() {
        availableCommands = VoiceCommandPatterns.allPatterns
    }
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session setup failed: \(error)")
        }
    }
    
    private func loadSettings() {
        if let data = UserDefaults.standard.data(forKey: "voiceSettings"),
           let settings = try? JSONDecoder().decode(VoiceSettings.self, from: data) {
            voiceSettings = settings
            currentLanguage = settings.language
            isVoiceEnabled = settings.enabled
        }
    }
    
    private func saveSettings() {
        do {
            let data = try JSONEncoder().encode(voiceSettings)
            UserDefaults.standard.set(data, forKey: "voiceSettings")
        } catch {
            print("Failed to save voice settings: \(error)")
        }
    }
    
    // MARK: - Permission Management
    
    func requestPermissions() async -> Bool {
        let speechPermission = await requestSpeechRecognitionPermission()
        let microphonePermission = await requestMicrophonePermission()
        
        return speechPermission && microphonePermission
    }
    
    private func requestSpeechRecognitionPermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }
    
    private func requestMicrophonePermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
    
    // MARK: - Voice Recognition Control
    
    func startListening() async {
        guard isVoiceEnabled else { return }
        
        guard await requestPermissions() else {
            recognitionState = .error(.speechRecognitionDenied)
            return
        }
        
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            recognitionState = .error(.speechRecognitionNotAvailable)
            return
        }
        
        do {
            try startRecognition()
        } catch {
            recognitionState = .error(.recognitionFailed)
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        recognitionRequest = nil
        recognitionTask = nil
        
        isListening = false
        recognitionState = .idle
    }
    
    private func startRecognition() throws {
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw VoiceError.recognitionFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = voiceSettings.useOnDeviceRecognition
        
        // Setup audio input
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        isListening = true
        recognitionState = .listening
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            Task { @MainActor in
                if let result = result {
                    self.recognizedText = result.bestTranscription.formattedString
                    
                    if result.isFinal {
                        await self.processRecognizedText(result.bestTranscription.formattedString)
                    }
                }
                
                if let error = error {
                    self.recognitionState = .error(.recognitionFailed)
                    self.stopListening()
                }
            }
        }
    }
    
    // MARK: - Command Processing
    
    private func processRecognizedText(_ text: String) async {
        recognitionState = .processing
        
        do {
            // Analyze context
            let context = await contextAnalyzer.analyzeContext()
            
            // Process natural language
            let nlpResult = await nlpEngine.process(text: text, context: context)
            
            // Match command pattern
            guard let commandPattern = matchCommandPattern(text: text, nlpResult: nlpResult) else {
                await handleUnrecognizedCommand(text: text)
                return
            }
            
            // Extract parameters
            let parameters = extractParameters(from: text, pattern: commandPattern, nlpResult: nlpResult)
            
            // Create voice command
            let command = VoiceCommand(
                id: UUID(),
                phrase: text,
                action: commandPattern.action,
                parameters: parameters,
                confidence: nlpResult.confidence,
                timestamp: Date(),
                language: currentLanguage,
                context: context
            )
            
            // Execute command
            await executeCommand(command)
            
        } catch {
            recognitionState = .error(.actionExecutionFailed)
        }
        
        stopListening()
    }
    
    private func matchCommandPattern(text: String, nlpResult: NLPResult) -> VoiceCommandPattern? {
        let lowercaseText = text.lowercased()
        
        for pattern in availableCommands {
            for patternString in pattern.patterns {
                if lowercaseText.contains(patternString.lowercased()) {
                    return pattern
                }
            }
        }
        
        // Try fuzzy matching
        return fuzzyMatchCommand(text: lowercaseText, nlpResult: nlpResult)
    }
    
    private func fuzzyMatchCommand(text: String, nlpResult: NLPResult) -> VoiceCommandPattern? {
        // Implementation would use more sophisticated NLP matching
        // For now, return nil
        return nil
    }
    
    private func extractParameters(from text: String, pattern: VoiceCommandPattern, nlpResult: NLPResult) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        for parameter in pattern.parameters {
            if let value = extractParameter(name: parameter.name, type: parameter.type, from: text, nlpResult: nlpResult) {
                parameters[parameter.name] = value
            } else if parameter.required {
                // Handle missing required parameter
                parameters[parameter.name] = parameter.defaultValue
            }
        }
        
        return parameters
    }
    
    private func extractParameter(name: String, type: ParameterType, from text: String, nlpResult: NLPResult) -> Any? {
        switch type {
        case .number:
            return extractNumber(from: text)
        case .text:
            return text
        case .boolean:
            return extractBoolean(from: text)
        case .date:
            return extractDate(from: text)
        case .time:
            return extractTime(from: text)
        case .duration:
            return extractDuration(from: text)
        case .painLevel:
            return extractPainLevel(from: text)
        case .mood:
            return extractMood(from: text)
        case .bodyPart:
            return extractBodyPart(from: text)
        case .medication:
            return extractMedication(from: text)
        }
    }
    
    // MARK: - Parameter Extraction Helpers
    
    private func extractNumber(from text: String) -> Double? {
        let numberFormatter = NumberFormatter()
        let words = text.components(separatedBy: .whitespaces)
        
        for word in words {
            if let number = numberFormatter.number(from: word) {
                return number.doubleValue
            }
        }
        
        // Handle word numbers (one, two, three, etc.)
        let wordToNumber: [String: Double] = [
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        ]
        
        for word in words {
            if let number = wordToNumber[word.lowercased()] {
                return number
            }
        }
        
        return nil
    }
    
    private func extractBoolean(from text: String) -> Bool? {
        let lowercaseText = text.lowercased()
        
        if lowercaseText.contains("yes") || lowercaseText.contains("true") || lowercaseText.contains("on") {
            return true
        } else if lowercaseText.contains("no") || lowercaseText.contains("false") || lowercaseText.contains("off") {
            return false
        }
        
        return nil
    }
    
    private func extractDate(from text: String) -> Date? {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        
        // Try various date formats
        let formats = ["today", "tomorrow", "yesterday"]
        let lowercaseText = text.lowercased()
        
        if lowercaseText.contains("today") {
            return Date()
        } else if lowercaseText.contains("tomorrow") {
            return Calendar.current.date(byAdding: .day, value: 1, to: Date())
        } else if lowercaseText.contains("yesterday") {
            return Calendar.current.date(byAdding: .day, value: -1, to: Date())
        }
        
        return nil
    }
    
    private func extractTime(from text: String) -> Date? {
        // Implementation would extract time from text
        return nil
    }
    
    private func extractDuration(from text: String) -> TimeInterval? {
        // Implementation would extract duration from text
        return nil
    }
    
    private func extractPainLevel(from text: String) -> Int? {
        if let number = extractNumber(from: text) {
            let level = Int(number)
            return (1...10).contains(level) ? level : nil
        }
        
        let painWords: [String: Int] = [
            "no": 0, "none": 0, "minimal": 1, "mild": 2, "low": 2,
            "moderate": 5, "medium": 5, "high": 7, "severe": 8,
            "extreme": 9, "unbearable": 10, "worst": 10
        ]
        
        let words = text.lowercased().components(separatedBy: .whitespaces)
        for word in words {
            if let level = painWords[word] {
                return level
            }
        }
        
        return nil
    }
    
    private func extractMood(from text: String) -> String? {
        let moodWords = ["happy", "sad", "angry", "anxious", "calm", "stressed", "tired", "energetic"]
        let words = text.lowercased().components(separatedBy: .whitespaces)
        
        for word in words {
            if moodWords.contains(word) {
                return word
            }
        }
        
        return nil
    }
    
    private func extractBodyPart(from text: String) -> String? {
        let bodyParts = ["head", "neck", "shoulder", "arm", "elbow", "wrist", "hand", "finger",
                        "chest", "back", "spine", "hip", "leg", "knee", "ankle", "foot", "toe"]
        let words = text.lowercased().components(separatedBy: .whitespaces)
        
        for word in words {
            if bodyParts.contains(word) {
                return word
            }
        }
        
        return nil
    }
    
    private func extractMedication(from text: String) -> String? {
        // Implementation would extract medication names from text
        // This would typically involve a database of medication names
        return nil
    }
    
    // MARK: - Command Execution
    
    private func executeCommand(_ command: VoiceCommand) async {
        lastCommand = command
        commandHistory.append(command)
        
        do {
            let result = try await commandProcessor.execute(command: command)
            let response = await responseGenerator.generateResponse(for: command, result: result)
            
            await deliverResponse(response)
            
            recognitionState = .responding
            
        } catch {
            let errorResponse = VoiceResponse(
                text: "Sorry, I couldn't complete that action. \(error.localizedDescription)",
                audioURL: nil,
                followUpQuestions: nil,
                suggestedActions: nil,
                requiresUserInput: false,
                priority: .normal
            )
            
            await deliverResponse(errorResponse)
            recognitionState = .error(.actionExecutionFailed)
        }
    }
    
    private func handleUnrecognizedCommand(text: String) async {
        let response = VoiceResponse(
            text: "I didn't understand that command. You can say things like 'log my pain level as 5' or 'take my medication'.",
            audioURL: nil,
            followUpQuestions: ["What would you like to do?"],
            suggestedActions: [.getHelp, .logPain, .takeMedication],
            requiresUserInput: true,
            priority: .normal
        )
        
        await deliverResponse(response)
        recognitionState = .error(.commandNotRecognized)
    }
    
    // MARK: - Response Delivery
    
    private func deliverResponse(_ response: VoiceResponse) async {
        lastResponse = response
        
        if voiceSettings.enableSpeechOutput {
            await speakResponse(response.text)
        }
        
        if voiceSettings.enableHapticFeedback {
            await provideFeedback(for: response)
        }
        
        // Show visual response if needed
        if response.requiresUserInput {
            // Implementation would show UI for user input
        }
    }
    
    private func speakResponse(_ text: String) async {
        return await withCheckedContinuation { continuation in
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = AVSpeechSynthesisVoice(language: currentLanguage)
            utterance.rate = Float(voiceSettings.speechRate)
            utterance.volume = Float(voiceSettings.speechVolume)
            
            speechSynthesizer.speak(utterance)
            
            // For simplicity, we'll continue immediately
            // In a real implementation, you'd wait for speech to complete
            continuation.resume()
        }
    }
    
    private func provideFeedback(for response: VoiceResponse) async {
        let feedbackGenerator = UINotificationFeedbackGenerator()
        
        switch response.priority {
        case .low, .normal:
            feedbackGenerator.notificationOccurred(.success)
        case .high:
            feedbackGenerator.notificationOccurred(.warning)
        case .urgent:
            feedbackGenerator.notificationOccurred(.error)
        }
    }
    
    // MARK: - Settings Management
    
    func updateSettings(_ settings: VoiceSettings) {
        voiceSettings = settings
        currentLanguage = settings.language
        isVoiceEnabled = settings.enabled
        saveSettings()
    }
    
    func setLanguage(_ language: String) {
        currentLanguage = language
        voiceSettings.language = language
        saveSettings()
        
        // Recreate speech recognizer with new language
        // Implementation would handle this
    }
    
    // MARK: - Integration
    
    func setHealthDataManager(_ manager: HealthDataManager) {
        healthDataManager = manager
        commandProcessor.setHealthDataManager(manager)
    }
    
    func setMedicationManager(_ manager: MedicationManager) {
        medicationManager = manager
        commandProcessor.setMedicationManager(manager)
    }
    
    func setPainTrackingManager(_ manager: PainTrackingManager) {
        painTrackingManager = manager
        commandProcessor.setPainTrackingManager(manager)
    }
    
    func setNavigationController(_ controller: UINavigationController) {
        navigationController = controller
        commandProcessor.setNavigationController(controller)
    }
    
    // MARK: - Public API
    
    func getAvailableCommands(for category: VoiceActionCategory? = nil) -> [VoiceCommandPattern] {
        if let category = category {
            return availableCommands.filter { $0.action.category == category }
        }
        return availableCommands
    }
    
    func simulateCommand(_ action: VoiceAction, parameters: [String: Any]? = nil) async {
        let command = VoiceCommand(
            id: UUID(),
            phrase: "Simulated command",
            action: action,
            parameters: parameters,
            confidence: 1.0,
            timestamp: Date(),
            language: currentLanguage,
            context: nil
        )
        
        await executeCommand(command)
    }
    
    func clearHistory() {
        commandHistory.removeAll()
        UserDefaults.standard.removeObject(forKey: "voiceCommandHistory")
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension VoiceCommandManager: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if !available {
            recognitionState = .error(.speechRecognitionNotAvailable)
        }
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension VoiceCommandManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        if recognitionState == .responding {
            recognitionState = .idle
        }
    }
}

// MARK: - Supporting Classes

struct NLPResult {
    let intent: String?
    let entities: [String: Any]
    let confidence: Float
    let sentiment: Double
}

class NaturalLanguageProcessor {
    func process(text: String, context: VoiceContext?) async -> NLPResult {
        // Implementation would use Core ML or external NLP service
        return NLPResult(
            intent: "log_pain",
            entities: [:],
            confidence: 0.8,
            sentiment: 0.0
        )
    }
}

class VoiceContextAnalyzer {
    func analyzeContext() async -> VoiceContext {
        return VoiceContext(
            currentScreen: nil,
            userState: .active,
            timeOfDay: TimeOfDay.current(),
            recentActions: [],
            environmentalFactors: nil
        )
    }
}

class VoiceResponseGenerator {
    func generateResponse(for command: VoiceCommand, result: Any?) async -> VoiceResponse {
        let responseText = generateResponseText(for: command, result: result)
        
        return VoiceResponse(
            text: responseText,
            audioURL: nil,
            followUpQuestions: generateFollowUpQuestions(for: command),
            suggestedActions: generateSuggestedActions(for: command),
            requiresUserInput: false,
            priority: .normal
        )
    }
    
    private func generateResponseText(for command: VoiceCommand, result: Any?) -> String {
        switch command.action {
        case .logPain:
            return "I've logged your pain level. Is there anything else you'd like to record?"
        case .takeMedication:
            return "I've marked your medication as taken. Great job staying on track!"
        case .logMood:
            return "I've recorded your mood. Thank you for sharing."
        default:
            return "Done! Is there anything else I can help you with?"
        }
    }
    
    private func generateFollowUpQuestions(for command: VoiceCommand) -> [String]? {
        switch command.action {
        case .logPain:
            return ["Would you like to describe the pain?", "Should I set a reminder to check on you later?"]
        case .takeMedication:
            return ["How are you feeling?", "Any side effects?"]
        default:
            return nil
        }
    }
    
    private func generateSuggestedActions(for command: VoiceCommand) -> [VoiceAction]? {
        switch command.action {
        case .logPain:
            return [.describePain, .logMood, .startMeditation]
        case .takeMedication:
            return [.logMood, .setMedicationReminder]
        default:
            return nil
        }
    }
}

class VoiceCommandProcessor {
    private weak var healthDataManager: HealthDataManager?
    private weak var medicationManager: MedicationManager?
    private weak var painTrackingManager: PainTrackingManager?
    private weak var navigationController: UINavigationController?
    
    func execute(command: VoiceCommand) async throws -> Any? {
        switch command.action {
        case .logPain:
            return try await executePainLogging(command: command)
        case .takeMedication:
            return try await executeMedicationTaking(command: command)
        case .logMood:
            return try await executeMoodLogging(command: command)
        case .openSection:
            return try await executeNavigation(command: command)
        default:
            throw VoiceError.actionExecutionFailed
        }
    }
    
    private func executePainLogging(command: VoiceCommand) async throws -> Any? {
        guard let painManager = painTrackingManager else {
            throw VoiceError.actionExecutionFailed
        }
        
        let painLevel = command.parameters?["painLevel"] as? Int ?? 5
        let bodyPart = command.parameters?["bodyPart"] as? String
        
        // Implementation would log pain
        return "Pain logged successfully"
    }
    
    private func executeMedicationTaking(command: VoiceCommand) async throws -> Any? {
        guard let medicationManager = medicationManager else {
            throw VoiceError.actionExecutionFailed
        }
        
        // Implementation would mark medication as taken
        return "Medication marked as taken"
    }
    
    private func executeMoodLogging(command: VoiceCommand) async throws -> Any? {
        guard let healthManager = healthDataManager else {
            throw VoiceError.actionExecutionFailed
        }
        
        let mood = command.parameters?["mood"] as? String ?? "neutral"
        
        // Implementation would log mood
        return "Mood logged successfully"
    }
    
    private func executeNavigation(command: VoiceCommand) async throws -> Any? {
        guard let navigationController = navigationController else {
            throw VoiceError.actionExecutionFailed
        }
        
        let section = command.parameters?["section"] as? String
        
        // Implementation would navigate to section
        return "Navigation completed"
    }
    
    // MARK: - Manager Setters
    
    func setHealthDataManager(_ manager: HealthDataManager) {
        healthDataManager = manager
    }
    
    func setMedicationManager(_ manager: MedicationManager) {
        medicationManager = manager
    }
    
    func setPainTrackingManager(_ manager: PainTrackingManager) {
        painTrackingManager = manager
    }
    
    func setNavigationController(_ controller: UINavigationController) {
        navigationController = controller
    }
}

// MARK: - Voice Settings

struct VoiceSettings: Codable {
    var enabled: Bool = true
    var language: String = "en-US"
    var enableSpeechOutput: Bool = true
    var enableHapticFeedback: Bool = true
    var useOnDeviceRecognition: Bool = true
    var speechRate: Double = 0.5
    var speechVolume: Double = 1.0
    var minimumConfidence: Float = 0.6
    var autoStopListening: Bool = true
    var autoStopTimeout: TimeInterval = 5.0
    var wakeWordEnabled: Bool = false
    var wakeWord: String = "Hey InflamAI"
    var contextAwareCommands: Bool = true
    var personalizedResponses: Bool = true
}

// MARK: - Voice Command Patterns

struct VoiceCommandPatterns {
    static let allPatterns: [VoiceCommandPattern] = [
        // Pain Management
        VoiceCommandPattern(
            action: .logPain,
            patterns: ["log pain", "record pain", "my pain is", "pain level"],
            parameters: [
                VoiceParameter(name: "painLevel", type: .painLevel, required: false, defaultValue: 5, validValues: nil),
                VoiceParameter(name: "bodyPart", type: .bodyPart, required: false, defaultValue: nil, validValues: nil)
            ],
            requiredConfidence: 0.7,
            contextSensitive: true
        ),
        
        // Medication
        VoiceCommandPattern(
            action: .takeMedication,
            patterns: ["take medication", "took my medicine", "medication taken"],
            parameters: [
                VoiceParameter(name: "medication", type: .medication, required: false, defaultValue: nil, validValues: nil),
                VoiceParameter(name: "time", type: .time, required: false, defaultValue: nil, validValues: nil)
            ],
            requiredConfidence: 0.8,
            contextSensitive: false
        ),
        
        // Mood
        VoiceCommandPattern(
            action: .logMood,
            patterns: ["log mood", "I feel", "my mood is", "feeling"],
            parameters: [
                VoiceParameter(name: "mood", type: .mood, required: true, defaultValue: nil, validValues: nil)
            ],
            requiredConfidence: 0.6,
            contextSensitive: true
        ),
        
        // Navigation
        VoiceCommandPattern(
            action: .openSection,
            patterns: ["open", "go to", "show me", "navigate to"],
            parameters: [
                VoiceParameter(name: "section", type: .text, required: true, defaultValue: nil, validValues: ["dashboard", "reports", "medications", "exercises"])
            ],
            requiredConfidence: 0.7,
            contextSensitive: false
        ),
        
        // Help
        VoiceCommandPattern(
            action: .getHelp,
            patterns: ["help", "what can you do", "commands", "how to"],
            parameters: [],
            requiredConfidence: 0.5,
            contextSensitive: false
        )
    ]
}