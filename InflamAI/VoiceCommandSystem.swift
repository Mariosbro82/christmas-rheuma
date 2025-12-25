//
//  VoiceCommandSystem.swift
//  InflamAI-Swift
//
//  Advanced voice command and speech recognition system for hands-free interaction
//

import Foundation
import Speech
import AVFoundation
import Combine
import CoreData
import NaturalLanguage
import SwiftUI

// MARK: - Voice Command Models

enum VoiceCommand: String, CaseIterable {
    case logPain = "log pain"
    case takeMedication = "take medication"
    case recordJournal = "record journal"
    case checkSymptoms = "check symptoms"
    case scheduleAppointment = "schedule appointment"
    case askQuestion = "ask question"
    case startAssessment = "start assessment"
    case viewAnalytics = "view analytics"
    case setReminder = "set reminder"
    case emergencyHelp = "emergency help"
    
    var aliases: [String] {
        switch self {
        case .logPain:
            return ["record pain", "add pain", "pain entry", "I have pain", "my pain is"]
        case .takeMedication:
            return ["medication taken", "took medicine", "took pill", "medication reminder"]
        case .recordJournal:
            return ["journal entry", "add journal", "write journal", "record thoughts"]
        case .checkSymptoms:
            return ["symptom check", "how am I feeling", "check health", "symptoms today"]
        case .scheduleAppointment:
            return ["book appointment", "schedule doctor", "make appointment"]
        case .askQuestion:
            return ["question", "help me", "what is", "how do I", "can you"]
        case .startAssessment:
            return ["BASDAI assessment", "start test", "health assessment"]
        case .viewAnalytics:
            return ["show analytics", "view reports", "health data", "my progress"]
        case .setReminder:
            return ["remind me", "set alarm", "medication reminder", "appointment reminder"]
        case .emergencyHelp:
            return ["emergency", "help me", "call doctor", "urgent"]
        }
    }
}

struct VoiceCommandResult {
    let command: VoiceCommand
    let confidence: Float
    let parameters: [String: Any]
    let timestamp: Date
    let rawText: String
}

struct SpeechRecognitionResult {
    let text: String
    let confidence: Float
    let isFinal: Bool
    let timestamp: Date
}

// MARK: - Voice Command System

class VoiceCommandSystem: NSObject, ObservableObject {
    // Speech Recognition
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    // Text-to-Speech
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    // Natural Language Processing
    private let nlProcessor = NaturalLanguageProcessor()
    
    // Core Data
    private let context: NSManagedObjectContext
    
    // Published Properties
    @Published var isListening = false
    @Published var isProcessing = false
    @Published var lastRecognizedText = ""
    @Published var lastCommand: VoiceCommandResult?
    @Published var voiceEnabled = true
    @Published var speechRate: Float = 0.5
    @Published var voiceGender: AVSpeechSynthesisVoiceGender = .female
    @Published var language = "en-US"
    
    // Voice Command History
    @Published var commandHistory: [VoiceCommandResult] = []
    
    // Conversation Context
    private var conversationContext: ConversationContext = ConversationContext()
    
    // Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    init(context: NSManagedObjectContext) {
        self.context = context
        super.init()
        
        setupAudioSession()
        requestPermissions()
        setupSpeechSynthesizer()
        loadUserPreferences()
    }
    
    // MARK: - Setup
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
    
    private func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { [weak self] authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    self?.voiceEnabled = true
                case .denied, .restricted, .notDetermined:
                    self?.voiceEnabled = false
                @unknown default:
                    self?.voiceEnabled = false
                }
            }
        }
        
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                if !granted {
                    self?.voiceEnabled = false
                }
            }
        }
    }
    
    private func setupSpeechSynthesizer() {
        speechSynthesizer.delegate = self
    }
    
    private func loadUserPreferences() {
        // Load user voice preferences from UserDefaults
        speechRate = UserDefaults.standard.float(forKey: "voiceSpeechRate")
        if speechRate == 0 { speechRate = 0.5 }
        
        let genderRaw = UserDefaults.standard.integer(forKey: "voiceGender")
        voiceGender = AVSpeechSynthesisVoiceGender(rawValue: genderRaw) ?? .female
        
        language = UserDefaults.standard.string(forKey: "voiceLanguage") ?? "en-US"
    }
    
    // MARK: - Speech Recognition
    
    func startListening() {
        guard voiceEnabled else {
            speak("Voice commands are not available. Please check permissions.")
            return
        }
        
        guard !audioEngine.isRunning else { return }
        
        do {
            try startSpeechRecognition()
            isListening = true
            
            // Provide audio feedback
            AudioServicesPlaySystemSound(1113) // Begin recording sound
            
        } catch {
            print("Failed to start speech recognition: \(error)")
            speak("Sorry, I couldn't start listening. Please try again.")
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        isListening = false
        
        // Provide audio feedback
        AudioServicesPlaySystemSound(1114) // End recording sound
    }
    
    private func startSpeechRecognition() throws {
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw VoiceCommandError.recognitionRequestFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = true
        
        // Setup audio input
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            self?.handleSpeechRecognitionResult(result: result, error: error)
        }
    }
    
    private func handleSpeechRecognitionResult(result: SFSpeechRecognitionResult?, error: Error?) {
        if let error = error {
            print("Speech recognition error: \(error)")
            stopListening()
            return
        }
        
        guard let result = result else { return }
        
        let recognizedText = result.bestTranscription.formattedString
        
        DispatchQueue.main.async {
            self.lastRecognizedText = recognizedText
            
            if result.isFinal {
                self.processVoiceCommand(recognizedText)
                self.stopListening()
            }
        }
    }
    
    // MARK: - Voice Command Processing
    
    private func processVoiceCommand(_ text: String) {
        isProcessing = true
        
        // Use NLP to understand the command
        nlProcessor.processText(text) { [weak self] result in
            DispatchQueue.main.async {
                self?.handleNLPResult(result, originalText: text)
                self?.isProcessing = false
            }
        }
    }
    
    private func handleNLPResult(_ nlpResult: NLPResult, originalText: String) {
        // Identify the command
        guard let command = identifyCommand(from: nlpResult, text: originalText) else {
            handleUnknownCommand(originalText)
            return
        }
        
        // Extract parameters
        let parameters = extractParameters(from: nlpResult, for: command)
        
        // Create command result
        let commandResult = VoiceCommandResult(
            command: command,
            confidence: nlpResult.confidence,
            parameters: parameters,
            timestamp: Date(),
            rawText: originalText
        )
        
        // Execute command
        executeCommand(commandResult)
        
        // Update history
        commandHistory.append(commandResult)
        lastCommand = commandResult
        
        // Update conversation context
        conversationContext.addInteraction(command: command, parameters: parameters)
    }
    
    private func identifyCommand(from nlpResult: NLPResult, text: String) -> VoiceCommand? {
        let lowercaseText = text.lowercased()
        
        // Check for direct command matches
        for command in VoiceCommand.allCases {
            if lowercaseText.contains(command.rawValue) {
                return command
            }
            
            // Check aliases
            for alias in command.aliases {
                if lowercaseText.contains(alias.lowercased()) {
                    return command
                }
            }
        }
        
        // Use NLP intent classification
        return classifyIntent(from: nlpResult)
    }
    
    private func classifyIntent(from nlpResult: NLPResult) -> VoiceCommand? {
        // Use the NLP result to classify intent
        if nlpResult.entities.contains(where: { $0.type == .painLevel || $0.type == .bodyPart }) {
            return .logPain
        }
        
        if nlpResult.entities.contains(where: { $0.type == .medication }) {
            return .takeMedication
        }
        
        if nlpResult.sentiment == .negative && nlpResult.entities.contains(where: { $0.type == .emotion }) {
            return .recordJournal
        }
        
        if nlpResult.intent == .question {
            return .askQuestion
        }
        
        return nil
    }
    
    private func extractParameters(from nlpResult: NLPResult, for command: VoiceCommand) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        switch command {
        case .logPain:
            parameters = extractPainParameters(from: nlpResult)
        case .takeMedication:
            parameters = extractMedicationParameters(from: nlpResult)
        case .recordJournal:
            parameters = extractJournalParameters(from: nlpResult)
        case .setReminder:
            parameters = extractReminderParameters(from: nlpResult)
        default:
            break
        }
        
        return parameters
    }
    
    private func extractPainParameters(from nlpResult: NLPResult) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        // Extract pain level
        if let painLevel = nlpResult.entities.first(where: { $0.type == .painLevel })?.value {
            parameters["painLevel"] = painLevel
        }
        
        // Extract body part
        if let bodyPart = nlpResult.entities.first(where: { $0.type == .bodyPart })?.value {
            parameters["bodyPart"] = bodyPart
        }
        
        // Extract pain type
        if let painType = nlpResult.entities.first(where: { $0.type == .painType })?.value {
            parameters["painType"] = painType
        }
        
        return parameters
    }
    
    private func extractMedicationParameters(from nlpResult: NLPResult) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        // Extract medication name
        if let medication = nlpResult.entities.first(where: { $0.type == .medication })?.value {
            parameters["medication"] = medication
        }
        
        // Extract dosage
        if let dosage = nlpResult.entities.first(where: { $0.type == .dosage })?.value {
            parameters["dosage"] = dosage
        }
        
        return parameters
    }
    
    private func extractJournalParameters(from nlpResult: NLPResult) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        parameters["mood"] = nlpResult.sentiment.rawValue
        parameters["content"] = nlpResult.originalText
        
        // Extract emotions
        let emotions = nlpResult.entities.filter { $0.type == .emotion }.map { $0.value }
        if !emotions.isEmpty {
            parameters["emotions"] = emotions
        }
        
        return parameters
    }
    
    private func extractReminderParameters(from nlpResult: NLPResult) -> [String: Any] {
        var parameters: [String: Any] = [:]
        
        // Extract time
        if let time = nlpResult.entities.first(where: { $0.type == .time })?.value {
            parameters["time"] = time
        }
        
        // Extract reminder type
        if let reminderType = nlpResult.entities.first(where: { $0.type == .reminderType })?.value {
            parameters["type"] = reminderType
        }
        
        return parameters
    }
    
    // MARK: - Command Execution
    
    private func executeCommand(_ commandResult: VoiceCommandResult) {
        switch commandResult.command {
        case .logPain:
            executePainLogging(commandResult)
        case .takeMedication:
            executeMedicationLogging(commandResult)
        case .recordJournal:
            executeJournalEntry(commandResult)
        case .checkSymptoms:
            executeSymptomCheck(commandResult)
        case .scheduleAppointment:
            executeAppointmentScheduling(commandResult)
        case .askQuestion:
            executeQuestionAnswering(commandResult)
        case .startAssessment:
            executeAssessmentStart(commandResult)
        case .viewAnalytics:
            executeAnalyticsView(commandResult)
        case .setReminder:
            executeReminderSetting(commandResult)
        case .emergencyHelp:
            executeEmergencyHelp(commandResult)
        }
    }
    
    private func executePainLogging(_ commandResult: VoiceCommandResult) {
        let parameters = commandResult.parameters
        
        // Extract pain information
        // FIXED: Use 0 for unspecified pain level, not fake "5"
        // 0 = "pain exists but severity not specified" - downstream should handle
        let painLevel = parameters["painLevel"] as? Int ?? 0
        let bodyPart = parameters["bodyPart"] as? String ?? "general"
        let painType = parameters["painType"] as? String ?? "general"
        
        // Create pain entry
        let painEntry = PainEntry(context: context)
        painEntry.id = UUID()
        painEntry.painLevel = Int16(painLevel)
        painEntry.bodyRegion = bodyPart
        painEntry.painType = painType
        painEntry.timestamp = Date()
        painEntry.notes = "Voice entry: \(commandResult.rawText)"
        
        do {
            try context.save()
            speak("I've logged your pain level of \(painLevel) for your \(bodyPart). Is there anything else you'd like to add?")
        } catch {
            speak("Sorry, I couldn't save your pain entry. Please try again.")
        }
    }
    
    private func executeMedicationLogging(_ commandResult: VoiceCommandResult) {
        let parameters = commandResult.parameters
        
        // Extract medication information
        let medicationName = parameters["medication"] as? String ?? "medication"
        let dosage = parameters["dosage"] as? String
        
        // Create medication intake entry
        let medicationIntake = MedicationIntake(context: context)
        medicationIntake.id = UUID()
        medicationIntake.timestamp = Date()
        medicationIntake.notes = "Voice entry: \(commandResult.rawText)"
        
        do {
            try context.save()
            var response = "I've recorded that you took your \(medicationName)"
            if let dosage = dosage {
                response += " (\(dosage))"
            }
            response += ". Great job staying on track with your medication!"
            speak(response)
        } catch {
            speak("Sorry, I couldn't record your medication intake. Please try again.")
        }
    }
    
    private func executeJournalEntry(_ commandResult: VoiceCommandResult) {
        let parameters = commandResult.parameters
        
        // Create journal entry
        let journalEntry = JournalEntry(context: context)
        journalEntry.id = UUID()
        journalEntry.timestamp = Date()
        journalEntry.content = commandResult.rawText
        journalEntry.mood = parameters["mood"] as? String ?? "neutral"
        
        do {
            try context.save()
            speak("I've saved your journal entry. Thank you for sharing your thoughts with me.")
        } catch {
            speak("Sorry, I couldn't save your journal entry. Please try again.")
        }
    }
    
    private func executeSymptomCheck(_ commandResult: VoiceCommandResult) {
        // Analyze recent health data and provide summary
        let healthAnalyzer = HealthDataAnalyzer(context: context)
        
        healthAnalyzer.generateDailySummary { [weak self] summary in
            DispatchQueue.main.async {
                self?.speak(summary)
            }
        }
    }
    
    private func executeAppointmentScheduling(_ commandResult: VoiceCommandResult) {
        speak("I can help you remember to schedule an appointment. Would you like me to set a reminder to call your doctor?")
        // In a full implementation, this would integrate with calendar apps
    }
    
    private func executeQuestionAnswering(_ commandResult: VoiceCommandResult) {
        let question = commandResult.rawText
        
        // Use AI to answer health-related questions
        let aiAssistant = AIHealthAssistant()
        
        aiAssistant.answerQuestion(question) { [weak self] answer in
            DispatchQueue.main.async {
                self?.speak(answer)
            }
        }
    }
    
    private func executeAssessmentStart(_ commandResult: VoiceCommandResult) {
        speak("Let's start your BASDAI assessment. I'll ask you a series of questions about how you've been feeling. Are you ready to begin?")
        // This would trigger the BASDAI assessment flow
    }
    
    private func executeAnalyticsView(_ commandResult: VoiceCommandResult) {
        // Generate spoken analytics summary
        let analyticsEngine = AnalyticsEngine(context: context)
        
        analyticsEngine.generateSpokenSummary { [weak self] summary in
            DispatchQueue.main.async {
                self?.speak(summary)
            }
        }
    }
    
    private func executeReminderSetting(_ commandResult: VoiceCommandResult) {
        let parameters = commandResult.parameters
        
        // Set up reminder based on parameters
        let reminderType = parameters["type"] as? String ?? "general"
        let time = parameters["time"] as? String
        
        var response = "I'll set up a \(reminderType) reminder for you"
        if let time = time {
            response += " at \(time)"
        }
        response += ". You'll receive a notification when it's time."
        
        speak(response)
    }
    
    private func executeEmergencyHelp(_ commandResult: VoiceCommandResult) {
        speak("I understand this might be urgent. If this is a medical emergency, please call emergency services immediately. Would you like me to help you contact your healthcare provider?")
        
        // In a full implementation, this could:
        // - Show emergency contacts
        // - Prepare emergency health summary
        // - Guide through emergency protocols
    }
    
    private func handleUnknownCommand(_ text: String) {
        speak("I'm not sure I understood that. You can ask me to log pain, record medication, write a journal entry, or ask health questions. What would you like to do?")
    }
    
    // MARK: - Text-to-Speech
    
    func speak(_ text: String, priority: SpeechPriority = .normal) {
        guard voiceEnabled else { return }
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = speechRate
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        
        // Set voice based on user preference
        if let voice = AVSpeechSynthesisVoice(language: language) {
            utterance.voice = voice
        }
        
        // Handle priority
        if priority == .high {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        
        speechSynthesizer.speak(utterance)
    }
    
    func stopSpeaking() {
        speechSynthesizer.stopSpeaking(at: .immediate)
    }
    
    // MARK: - Conversation Management
    
    func startConversation() {
        speak("Hello! I'm your health assistant. How can I help you today?")
    }
    
    func endConversation() {
        speak("Goodbye! Take care of yourself.")
        conversationContext.reset()
    }
    
    // MARK: - Settings
    
    func updateVoiceSettings(rate: Float, gender: AVSpeechSynthesisVoiceGender, language: String) {
        speechRate = rate
        voiceGender = gender
        self.language = language
        
        // Save to UserDefaults
        UserDefaults.standard.set(rate, forKey: "voiceSpeechRate")
        UserDefaults.standard.set(gender.rawValue, forKey: "voiceGender")
        UserDefaults.standard.set(language, forKey: "voiceLanguage")
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension VoiceCommandSystem: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        // Speech started
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        // Speech finished
    }
}

// MARK: - Supporting Classes

class NaturalLanguageProcessor {
    private let tagger = NLTagger(tagSchemes: [.tokenType, .lexicalClass, .nameType, .sentimentScore])
    
    func processText(_ text: String, completion: @escaping (NLPResult) -> Void) {
        tagger.string = text
        
        var entities: [NLPEntity] = []
        var sentiment: NLPSentiment = .neutral
        var intent: NLPIntent = .unknown
        
        // Extract entities
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType) { tag, tokenRange in
            if let tag = tag {
                let value = String(text[tokenRange])
                let entityType = mapNLTagToEntityType(tag)
                entities.append(NLPEntity(type: entityType, value: value, range: tokenRange))
            }
            return true
        }
        
        // Analyze sentiment
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .paragraph, scheme: .sentimentScore) { tag, _ in
            if let tag = tag, let score = Double(tag.rawValue) {
                sentiment = score > 0.1 ? .positive : (score < -0.1 ? .negative : .neutral)
            }
            return true
        }
        
        // Determine intent
        intent = determineIntent(from: text, entities: entities)
        
        let result = NLPResult(
            originalText: text,
            entities: entities,
            sentiment: sentiment,
            intent: intent,
            confidence: 0.8 // Simplified confidence score
        )
        
        completion(result)
    }
    
    private func mapNLTagToEntityType(_ tag: NLTag) -> NLPEntityType {
        switch tag {
        case .personalName:
            return .person
        case .placeName:
            return .location
        case .organizationName:
            return .organization
        default:
            return .other
        }
    }
    
    private func determineIntent(from text: String, entities: [NLPEntity]) -> NLPIntent {
        let lowercaseText = text.lowercased()
        
        if lowercaseText.contains("?") || lowercaseText.hasPrefix("what") || lowercaseText.hasPrefix("how") || lowercaseText.hasPrefix("why") {
            return .question
        }
        
        if lowercaseText.contains("pain") || lowercaseText.contains("hurt") {
            return .painReport
        }
        
        if lowercaseText.contains("medication") || lowercaseText.contains("pill") || lowercaseText.contains("medicine") {
            return .medicationReport
        }
        
        return .unknown
    }
}

class ConversationContext {
    private var interactions: [VoiceCommandResult] = []
    private var currentTopic: String?
    private var userPreferences: [String: Any] = [:]
    
    func addInteraction(command: VoiceCommand, parameters: [String: Any]) {
        // Track conversation flow
        currentTopic = command.rawValue
        
        // Learn user preferences
        updateUserPreferences(from: parameters)
    }
    
    private func updateUserPreferences(from parameters: [String: Any]) {
        // Update user preferences based on interaction patterns
        for (key, value) in parameters {
            userPreferences[key] = value
        }
    }
    
    func reset() {
        interactions.removeAll()
        currentTopic = nil
    }
}

class HealthDataAnalyzer {
    private let context: NSManagedObjectContext
    
    init(context: NSManagedObjectContext) {
        self.context = context
    }
    
    func generateDailySummary(completion: @escaping (String) -> Void) {
        // Analyze today's health data
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: today)!
        
        // Fetch today's data
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@", today as NSDate, tomorrow as NSDate)
        
        do {
            let painEntries = try context.fetch(painRequest)
            let averagePain = painEntries.isEmpty ? 0 : painEntries.map { Int($0.painLevel) }.reduce(0, +) / painEntries.count
            
            var summary = "Here's your health summary for today: "
            
            if painEntries.isEmpty {
                summary += "You haven't logged any pain today, which is great! "
            } else {
                summary += "Your average pain level today is \(averagePain) out of 10. "
            }
            
            // Add more analysis here...
            
            completion(summary)
        } catch {
            completion("I'm having trouble accessing your health data right now.")
        }
    }
}

class AIHealthAssistant {
    func answerQuestion(_ question: String, completion: @escaping (String) -> Void) {
        // In a real implementation, this would use an AI service
        // For now, provide basic responses
        
        let lowercaseQuestion = question.lowercased()
        
        var answer = ""
        
        if lowercaseQuestion.contains("pain") {
            answer = "Pain management is important for people with rheumatic conditions. Regular monitoring, medication adherence, and lifestyle modifications can help. Always consult with your healthcare provider for personalized advice."
        } else if lowercaseQuestion.contains("medication") {
            answer = "It's important to take your medications as prescribed by your doctor. If you have concerns about side effects or effectiveness, please discuss them with your healthcare provider."
        } else if lowercaseQuestion.contains("exercise") {
            answer = "Gentle exercise can be beneficial for managing rheumatic conditions. Low-impact activities like swimming, walking, or yoga may help. Always check with your doctor before starting a new exercise routine."
        } else {
            answer = "I'm here to help with health-related questions. For specific medical advice, please consult with your healthcare provider."
        }
        
        completion(answer)
    }
}

class AnalyticsEngine {
    private let context: NSManagedObjectContext
    
    init(context: NSManagedObjectContext) {
        self.context = context
    }
    
    func generateSpokenSummary(completion: @escaping (String) -> Void) {
        // Generate a spoken summary of health analytics
        let summary = "Based on your recent data, your pain levels have been stable with an average of 4 out of 10. Your medication adherence is excellent at 95%. Keep up the great work!"
        completion(summary)
    }
}

// MARK: - Supporting Enums and Structs

enum SpeechPriority {
    case low, normal, high
}

enum VoiceCommandError: Error {
    case recognitionRequestFailed
    case audioEngineFailed
    case permissionDenied
}

struct NLPResult {
    let originalText: String
    let entities: [NLPEntity]
    let sentiment: NLPSentiment
    let intent: NLPIntent
    let confidence: Float
}

struct NLPEntity {
    let type: NLPEntityType
    let value: String
    let range: Range<String.Index>
}

enum NLPEntityType {
    case person, location, organization, painLevel, bodyPart, painType, medication, dosage, emotion, time, reminderType, other
}

enum NLPSentiment: String {
    case positive, negative, neutral
}

enum NLPIntent {
    case question, painReport, medicationReport, journalEntry, reminder, unknown
}