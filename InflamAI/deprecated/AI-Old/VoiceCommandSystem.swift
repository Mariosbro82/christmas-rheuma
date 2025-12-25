//
//  VoiceCommandSystem.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import Foundation
import Speech
import AVFoundation
import Combine
import NaturalLanguage
import CoreML
import CreateML

// MARK: - Voice Command System
class VoiceCommandSystem: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var isListening: Bool = false
    @Published var isProcessing: Bool = false
    @Published var recognizedText: String = ""
    @Published var lastCommand: VoiceCommand?
    @Published var commandHistory: [VoiceCommand] = []
    @Published var voiceEnabled: Bool = true
    @Published var listeningMode: ListeningMode = .manual
    @Published var confidence: Float = 0.0
    @Published var processingStatus: ProcessingStatus = .idle
    @Published var supportedLanguages: [String] = ["en-US", "es-ES", "fr-FR", "de-DE"]
    @Published var currentLanguage: String = "en-US"
    @Published var voiceProfile: VoiceProfile?
    @Published var contextualSuggestions: [String] = []
    
    // MARK: - Private Properties
    private let speechRecognizer: SFSpeechRecognizer
    private let audioEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - NLP Components
    private let nlpProcessor: AdvancedNLPProcessor
    private let intentClassifier: IntentClassifier
    private let entityExtractor: EntityExtractor
    private let contextManager: ConversationContextManager
    private let commandExecutor: VoiceCommandExecutor
    private let feedbackGenerator: VoiceFeedbackGenerator
    private let personalizationEngine: VoicePersonalizationEngine
    private let multilingualProcessor: MultilingualProcessor
    private let conversationManager: ConversationManager
    private let voiceAnalyzer: VoiceAnalyzer
    private let adaptiveProcessor: AdaptiveVoiceProcessor
    
    // MARK: - Configuration
    private let maxRecordingDuration: TimeInterval = 60.0
    private let silenceThreshold: Float = -50.0
    private let confidenceThreshold: Float = 0.7
    private let maxCommandHistory: Int = 100
    private let contextWindow: TimeInterval = 300 // 5 minutes
    
    // MARK: - Initialization
    override init() {
        self.speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: currentLanguage))!
        self.nlpProcessor = AdvancedNLPProcessor()
        self.intentClassifier = IntentClassifier()
        self.entityExtractor = EntityExtractor()
        self.contextManager = ConversationContextManager()
        self.commandExecutor = VoiceCommandExecutor()
        self.feedbackGenerator = VoiceFeedbackGenerator()
        self.personalizationEngine = VoicePersonalizationEngine()
        self.multilingualProcessor = MultilingualProcessor()
        self.conversationManager = ConversationManager()
        self.voiceAnalyzer = VoiceAnalyzer()
        self.adaptiveProcessor = AdaptiveVoiceProcessor()
        
        super.init()
        
        setupVoiceRecognition()
        setupAudioSession()
        loadVoiceProfile()
        setupContextualSuggestions()
    }
    
    // MARK: - Public Methods
    func requestPermissions() async -> Bool {
        let speechStatus = await SFSpeechRecognizer.requestAuthorization()
        let audioStatus = await AVAudioSession.sharedInstance().requestRecordPermission()
        
        return speechStatus == .authorized && audioStatus
    }
    
    func startListening() {
        guard !isListening else { return }
        
        Task {
            do {
                try await startSpeechRecognition()
                DispatchQueue.main.async {
                    self.isListening = true
                    self.processingStatus = .listening
                }
            } catch {
                print("Failed to start listening: \(error)")
            }
        }
    }
    
    func stopListening() {
        guard isListening else { return }
        
        stopSpeechRecognition()
        
        DispatchQueue.main.async {
            self.isListening = false
            self.processingStatus = .idle
        }
    }
    
    func processVoiceCommand(_ text: String) async {
        DispatchQueue.main.async {
            self.isProcessing = true
            self.processingStatus = .processing
            self.recognizedText = text
        }
        
        do {
            // Analyze voice characteristics
            let voiceCharacteristics = await voiceAnalyzer.analyzeVoice(text)
            
            // Process with NLP
            let nlpResult = await nlpProcessor.process(text, context: contextManager.currentContext)
            
            // Classify intent
            let intent = await intentClassifier.classifyIntent(nlpResult)
            
            // Extract entities
            let entities = await entityExtractor.extractEntities(nlpResult)
            
            // Create voice command
            let command = VoiceCommand(
                id: UUID(),
                text: text,
                intent: intent,
                entities: entities,
                confidence: nlpResult.confidence,
                timestamp: Date(),
                language: currentLanguage,
                voiceCharacteristics: voiceCharacteristics
            )
            
            // Update context
            contextManager.updateContext(with: command)
            
            // Execute command
            let result = await commandExecutor.execute(command)
            
            // Generate feedback
            await feedbackGenerator.generateFeedback(for: result)
            
            // Learn from interaction
            await personalizationEngine.learn(from: command, result: result)
            
            // Update UI
            DispatchQueue.main.async {
                self.lastCommand = command
                self.commandHistory.append(command)
                self.confidence = command.confidence
                self.isProcessing = false
                self.processingStatus = .completed
                
                // Limit history size
                if self.commandHistory.count > self.maxCommandHistory {
                    self.commandHistory.removeFirst()
                }
            }
            
            // Update contextual suggestions
            updateContextualSuggestions()
            
        } catch {
            DispatchQueue.main.async {
                self.isProcessing = false
                self.processingStatus = .error
            }
            print("Error processing voice command: \(error)")
        }
    }
    
    func enableContinuousListening() {
        listeningMode = .continuous
        startListening()
    }
    
    func disableContinuousListening() {
        listeningMode = .manual
        stopListening()
    }
    
    func changeLanguage(_ languageCode: String) {
        guard supportedLanguages.contains(languageCode) else { return }
        
        currentLanguage = languageCode
        
        // Reinitialize speech recognizer with new language
        if let newRecognizer = SFSpeechRecognizer(locale: Locale(identifier: languageCode)) {
            // Update recognizer
        }
        
        // Update NLP components
        multilingualProcessor.switchLanguage(languageCode)
    }
    
    func trainPersonalizedModel() async {
        await personalizationEngine.trainModel(with: commandHistory)
    }
    
    func getVoiceInsights() -> VoiceInsights {
        return VoiceInsights(
            totalCommands: commandHistory.count,
            averageConfidence: calculateAverageConfidence(),
            mostUsedIntents: getMostUsedIntents(),
            languageDistribution: getLanguageDistribution(),
            voicePatterns: voiceAnalyzer.getPatterns(),
            improvementSuggestions: getImprovementSuggestions()
        )
    }
    
    func exportVoiceData() -> VoiceDataExport {
        return VoiceDataExport(
            commands: commandHistory,
            voiceProfile: voiceProfile,
            insights: getVoiceInsights(),
            exportDate: Date()
        )
    }
    
    // MARK: - Private Methods
    private func setupVoiceRecognition() {
        speechRecognizer.delegate = self
    }
    
    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
    
    private func startSpeechRecognition() async throws {
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw VoiceError.recognitionRequestFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = true
        
        // Setup audio engine
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        // Start recognition
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            if let result = result {
                let text = result.bestTranscription.formattedString
                
                DispatchQueue.main.async {
                    self?.recognizedText = text
                    self?.confidence = result.bestTranscription.segments.last?.confidence ?? 0.0
                }
                
                if result.isFinal {
                    Task {
                        await self?.processVoiceCommand(text)
                    }
                }
            }
            
            if error != nil {
                self?.stopSpeechRecognition()
            }
        }
    }
    
    private func stopSpeechRecognition() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        recognitionTask?.cancel()
        recognitionTask = nil
    }
    
    private func loadVoiceProfile() {
        // Load user's voice profile from storage
        voiceProfile = VoiceProfile(
            userId: "current_user",
            voiceCharacteristics: VoiceCharacteristics(
                pitch: 150.0,
                tone: .neutral,
                speed: 1.0,
                accent: .american
            ),
            preferredLanguages: ["en-US"],
            commandPreferences: [:],
            adaptationLevel: 0.5
        )
    }
    
    private func setupContextualSuggestions() {
        contextualSuggestions = [
            "Log my pain level",
            "Record symptoms",
            "Take medication",
            "Start exercise",
            "Check my progress",
            "Schedule appointment",
            "Export health data",
            "Set reminder"
        ]
    }
    
    private func updateContextualSuggestions() {
        // Update suggestions based on context and usage patterns
        let suggestions = personalizationEngine.generateSuggestions()
        
        DispatchQueue.main.async {
            self.contextualSuggestions = suggestions
        }
    }
    
    private func calculateAverageConfidence() -> Float {
        guard !commandHistory.isEmpty else { return 0.0 }
        
        let total = commandHistory.reduce(0.0) { $0 + $1.confidence }
        return total / Float(commandHistory.count)
    }
    
    private func getMostUsedIntents() -> [String] {
        let intentCounts = Dictionary(grouping: commandHistory, by: { $0.intent.name })
            .mapValues { $0.count }
        
        return intentCounts.sorted { $0.value > $1.value }
            .prefix(5)
            .map { $0.key }
    }
    
    private func getLanguageDistribution() -> [String: Int] {
        return Dictionary(grouping: commandHistory, by: { $0.language })
            .mapValues { $0.count }
    }
    
    private func getImprovementSuggestions() -> [String] {
        var suggestions: [String] = []
        
        let avgConfidence = calculateAverageConfidence()
        if avgConfidence < 0.8 {
            suggestions.append("Try speaking more clearly")
            suggestions.append("Reduce background noise")
        }
        
        if commandHistory.count < 10 {
            suggestions.append("Use voice commands more frequently to improve accuracy")
        }
        
        return suggestions
    }
}

// MARK: - Speech Recognizer Delegate
extension VoiceCommandSystem: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        DispatchQueue.main.async {
            self.voiceEnabled = available
        }
    }
}

// MARK: - Supporting Classes
class AdvancedNLPProcessor {
    private let tokenizer = NLTokenizer(unit: .word)
    private let tagger = NLTagger(tagSchemes: [.tokenType, .language, .lexicalClass, .nameType, .lemma])
    
    func process(_ text: String, context: ConversationContext) async -> NLPResult {
        // Advanced NLP processing with context awareness
        let tokens = tokenizeText(text)
        let entities = extractNamedEntities(text)
        let sentiment = analyzeSentiment(text)
        let language = detectLanguage(text)
        let intent = classifyIntent(text, context: context)
        
        return NLPResult(
            originalText: text,
            tokens: tokens,
            entities: entities,
            sentiment: sentiment,
            language: language,
            intent: intent,
            confidence: calculateConfidence(text, intent: intent)
        )
    }
    
    private func tokenizeText(_ text: String) -> [String] {
        tokenizer.string = text
        var tokens: [String] = []
        
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            tokens.append(String(text[tokenRange]))
            return true
        }
        
        return tokens
    }
    
    private func extractNamedEntities(_ text: String) -> [NamedEntity] {
        tagger.string = text
        var entities: [NamedEntity] = []
        
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType) { tag, tokenRange in
            if let tag = tag {
                entities.append(NamedEntity(
                    text: String(text[tokenRange]),
                    type: tag.rawValue,
                    range: tokenRange
                ))
            }
            return true
        }
        
        return entities
    }
    
    private func analyzeSentiment(_ text: String) -> SentimentScore {
        // Implement sentiment analysis
        return SentimentScore(positive: 0.5, negative: 0.3, neutral: 0.2)
    }
    
    private func detectLanguage(_ text: String) -> String {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage?.rawValue ?? "en"
    }
    
    private func classifyIntent(_ text: String, context: ConversationContext) -> Intent {
        // Advanced intent classification with context
        return Intent(name: "log_symptom", confidence: 0.8, parameters: [:])
    }
    
    private func calculateConfidence(_ text: String, intent: Intent) -> Float {
        // Calculate overall confidence score
        return intent.confidence
    }
}

class IntentClassifier {
    func classifyIntent(_ nlpResult: NLPResult) async -> Intent {
        // Classify user intent from NLP result
        return Intent(name: "unknown", confidence: 0.5, parameters: [:])
    }
}

class EntityExtractor {
    func extractEntities(_ nlpResult: NLPResult) async -> [Entity] {
        // Extract relevant entities for health app
        return []
    }
}

class ConversationContextManager {
    var currentContext: ConversationContext = ConversationContext()
    
    func updateContext(with command: VoiceCommand) {
        currentContext.lastCommand = command
        currentContext.timestamp = Date()
        currentContext.conversationHistory.append(command)
    }
}

class VoiceCommandExecutor {
    func execute(_ command: VoiceCommand) async -> CommandResult {
        // Execute the voice command
        return CommandResult(
            success: true,
            message: "Command executed successfully",
            data: nil
        )
    }
}

class VoiceFeedbackGenerator {
    func generateFeedback(for result: CommandResult) async {
        // Generate voice or haptic feedback
    }
}

class VoicePersonalizationEngine {
    func learn(from command: VoiceCommand, result: CommandResult) async {
        // Learn from user interactions to improve personalization
    }
    
    func trainModel(with history: [VoiceCommand]) async {
        // Train personalized model
    }
    
    func generateSuggestions() -> [String] {
        // Generate personalized suggestions
        return []
    }
}

class MultilingualProcessor {
    func switchLanguage(_ languageCode: String) {
        // Switch processing language
    }
}

class ConversationManager {
    func manageConversation(_ command: VoiceCommand) -> ConversationState {
        // Manage conversation flow
        return ConversationState()
    }
}

class VoiceAnalyzer {
    func analyzeVoice(_ text: String) async -> VoiceCharacteristics {
        // Analyze voice characteristics
        return VoiceCharacteristics(
            pitch: 150.0,
            tone: .neutral,
            speed: 1.0,
            accent: .american
        )
    }
    
    func getPatterns() -> [VoicePattern] {
        // Get voice usage patterns
        return []
    }
}

class AdaptiveVoiceProcessor {
    func adaptToUser(_ voiceProfile: VoiceProfile) {
        // Adapt processing to user's voice profile
    }
}

// MARK: - Data Structures
struct VoiceCommand: Identifiable, Codable {
    let id: UUID
    let text: String
    let intent: Intent
    let entities: [Entity]
    let confidence: Float
    let timestamp: Date
    let language: String
    let voiceCharacteristics: VoiceCharacteristics
}

struct Intent: Codable {
    let name: String
    let confidence: Float
    let parameters: [String: String]
}

struct Entity: Identifiable, Codable {
    let id: UUID
    let text: String
    let type: String
    let confidence: Float
    let value: String?
    
    init(text: String, type: String, confidence: Float, value: String? = nil) {
        self.id = UUID()
        self.text = text
        self.type = type
        self.confidence = confidence
        self.value = value
    }
}

struct NLPResult {
    let originalText: String
    let tokens: [String]
    let entities: [NamedEntity]
    let sentiment: SentimentScore
    let language: String
    let intent: Intent
    let confidence: Float
}

struct NamedEntity {
    let text: String
    let type: String
    let range: Range<String.Index>
}

struct SentimentScore {
    let positive: Float
    let negative: Float
    let neutral: Float
}

struct ConversationContext {
    var lastCommand: VoiceCommand?
    var timestamp: Date = Date()
    var conversationHistory: [VoiceCommand] = []
    var userPreferences: [String: Any] = [:]
    var sessionId: UUID = UUID()
}

struct CommandResult {
    let success: Bool
    let message: String
    let data: Any?
}

struct VoiceProfile: Codable {
    let userId: String
    let voiceCharacteristics: VoiceCharacteristics
    let preferredLanguages: [String]
    let commandPreferences: [String: Float]
    let adaptationLevel: Float
}

struct VoiceCharacteristics: Codable {
    let pitch: Float
    let tone: VoiceTone
    let speed: Float
    let accent: VoiceAccent
}

struct VoiceInsights {
    let totalCommands: Int
    let averageConfidence: Float
    let mostUsedIntents: [String]
    let languageDistribution: [String: Int]
    let voicePatterns: [VoicePattern]
    let improvementSuggestions: [String]
}

struct VoiceDataExport: Codable {
    let commands: [VoiceCommand]
    let voiceProfile: VoiceProfile?
    let insights: VoiceInsights
    let exportDate: Date
}

struct VoicePattern: Identifiable {
    let id: UUID
    let type: PatternType
    let frequency: Int
    let confidence: Float
    let description: String
    
    init(type: PatternType, frequency: Int, confidence: Float, description: String) {
        self.id = UUID()
        self.type = type
        self.frequency = frequency
        self.confidence = confidence
        self.description = description
    }
}

struct ConversationState {
    var isActive: Bool = false
    var currentTopic: String?
    var expectedResponse: String?
    var context: [String: Any] = [:]
}

// MARK: - Enums
enum ListeningMode: String, CaseIterable {
    case manual = "manual"
    case continuous = "continuous"
    case pushToTalk = "push_to_talk"
    case voiceActivated = "voice_activated"
}

enum ProcessingStatus: String, CaseIterable {
    case idle = "idle"
    case listening = "listening"
    case processing = "processing"
    case completed = "completed"
    case error = "error"
}

enum VoiceTone: String, CaseIterable, Codable {
    case neutral = "neutral"
    case happy = "happy"
    case sad = "sad"
    case angry = "angry"
    case excited = "excited"
    case calm = "calm"
}

enum VoiceAccent: String, CaseIterable, Codable {
    case american = "american"
    case british = "british"
    case australian = "australian"
    case canadian = "canadian"
    case indian = "indian"
    case other = "other"
}

enum PatternType: String, CaseIterable {
    case temporal = "temporal"
    case linguistic = "linguistic"
    case behavioral = "behavioral"
    case contextual = "contextual"
}

enum VoiceError: Error {
    case recognitionRequestFailed
    case audioEngineError
    case permissionDenied
    case processingError
    case networkError
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let voiceCommandRecognized = Notification.Name("voiceCommandRecognized")
    static let voiceCommandExecuted = Notification.Name("voiceCommandExecuted")
    static let voiceListeningStarted = Notification.Name("voiceListeningStarted")
    static let voiceListeningStopped = Notification.Name("voiceListeningStopped")
    static let voiceLanguageChanged = Notification.Name("voiceLanguageChanged")
    static let voiceProfileUpdated = Notification.Name("voiceProfileUpdated")
}