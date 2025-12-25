//
//  AccessibilityManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import UIKit
import AVFoundation
import Speech
import Combine
import os.log

// MARK: - Accessibility Manager
class AccessibilityManager: NSObject, ObservableObject {
    static let shared = AccessibilityManager()
    
    // MARK: - Properties
    @Published var isVoiceOverEnabled = false
    @Published var isHighContrastEnabled = false
    @Published var textScaleFactor: CGFloat = 1.0
    @Published var isHapticFeedbackEnabled = true
    @Published var isVoiceCommandsEnabled = false
    @Published var currentLanguage: SupportedLanguage = .english
    @Published var accessibilitySettings: AccessibilitySettings = AccessibilitySettings()
    @Published var isListening = false
    @Published var lastVoiceCommand: VoiceCommand?
    @Published var speechRecognitionStatus: SpeechRecognitionStatus = .notDetermined
    
    // Voice and Speech
    private let speechSynthesizer = AVSpeechSynthesizer()
    private let speechRecognizer = SFSpeechRecognizer()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    // Haptic Feedback
    private let impactFeedbackLight = UIImpactFeedbackGenerator(style: .light)
    private let impactFeedbackMedium = UIImpactFeedbackGenerator(style: .medium)
    private let impactFeedbackHeavy = UIImpactFeedbackGenerator(style: .heavy)
    private let selectionFeedback = UISelectionFeedbackGenerator()
    private let notificationFeedback = UINotificationFeedbackGenerator()
    
    // Voice Command Processing
    private let voiceCommandProcessor = VoiceCommandProcessor()
    private let speechToTextProcessor = SpeechToTextProcessor()
    private let textToSpeechProcessor = TextToSpeechProcessor()
    
    // Localization
    private let localizationManager = LocalizationManager()
    
    // Accessibility Observers
    private var accessibilityObservers: [NSObjectProtocol] = []
    private var cancellables = Set<AnyCancellable>()
    
    private let logger = Logger(subsystem: "com.inflamai.accessibility", category: "AccessibilityManager")
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupAccessibilityObservers()
        setupSpeechRecognition()
        setupTextToSpeech()
        loadAccessibilitySettings()
        updateAccessibilityStatus()
    }
    
    deinit {
        removeAccessibilityObservers()
        stopVoiceCommands()
    }
    
    // MARK: - Setup
    private func setupAccessibilityObservers() {
        // VoiceOver status observer
        let voiceOverObserver = NotificationCenter.default.addObserver(
            forName: UIAccessibility.voiceOverStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateVoiceOverStatus()
        }
        accessibilityObservers.append(voiceOverObserver)
        
        // High contrast observer
        let contrastObserver = NotificationCenter.default.addObserver(
            forName: UIAccessibility.darkerSystemColorsStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateHighContrastStatus()
        }
        accessibilityObservers.append(contrastObserver)
        
        // Text size observer
        let textSizeObserver = NotificationCenter.default.addObserver(
            forName: UIContentSizeCategory.didChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateTextScaleFactor()
        }
        accessibilityObservers.append(textSizeObserver)
        
        // Reduce motion observer
        let motionObserver = NotificationCenter.default.addObserver(
            forName: UIAccessibility.reduceMotionStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateMotionSettings()
        }
        accessibilityObservers.append(motionObserver)
    }
    
    private func removeAccessibilityObservers() {
        accessibilityObservers.forEach { observer in
            NotificationCenter.default.removeObserver(observer)
        }
        accessibilityObservers.removeAll()
    }
    
    private func setupSpeechRecognition() {
        guard let speechRecognizer = speechRecognizer else {
            logger.error("Speech recognizer not available")
            return
        }
        
        speechRecognizer.delegate = self
        
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                self?.speechRecognitionStatus = SpeechRecognitionStatus(from: status)
                self?.logger.info("Speech recognition authorization: \(status.rawValue)")
            }
        }
    }
    
    private func setupTextToSpeech() {
        speechSynthesizer.delegate = self
    }
    
    // MARK: - Accessibility Status Updates
    private func updateAccessibilityStatus() {
        updateVoiceOverStatus()
        updateHighContrastStatus()
        updateTextScaleFactor()
        updateMotionSettings()
    }
    
    private func updateVoiceOverStatus() {
        isVoiceOverEnabled = UIAccessibility.isVoiceOverRunning
        logger.info("VoiceOver status: \(isVoiceOverEnabled)")
    }
    
    private func updateHighContrastStatus() {
        isHighContrastEnabled = UIAccessibility.isDarkerSystemColorsEnabled
        logger.info("High contrast status: \(isHighContrastEnabled)")
    }
    
    private func updateTextScaleFactor() {
        let contentSizeCategory = UIApplication.shared.preferredContentSizeCategory
        textScaleFactor = getScaleFactor(for: contentSizeCategory)
        logger.info("Text scale factor: \(textScaleFactor)")
    }
    
    private func updateMotionSettings() {
        let isReduceMotionEnabled = UIAccessibility.isReduceMotionEnabled
        accessibilitySettings.isReduceMotionEnabled = isReduceMotionEnabled
        logger.info("Reduce motion: \(isReduceMotionEnabled)")
    }
    
    private func getScaleFactor(for category: UIContentSizeCategory) -> CGFloat {
        switch category {
        case .extraSmall:
            return 0.8
        case .small:
            return 0.9
        case .medium:
            return 1.0
        case .large:
            return 1.1
        case .extraLarge:
            return 1.2
        case .extraExtraLarge:
            return 1.3
        case .extraExtraExtraLarge:
            return 1.4
        case .accessibilityMedium:
            return 1.6
        case .accessibilityLarge:
            return 1.8
        case .accessibilityExtraLarge:
            return 2.0
        case .accessibilityExtraExtraLarge:
            return 2.2
        case .accessibilityExtraExtraExtraLarge:
            return 2.4
        default:
            return 1.0
        }
    }
    
    // MARK: - VoiceOver Support
    func announceForVoiceOver(_ text: String, priority: UIAccessibility.AnnouncementPriority = .medium) {
        guard isVoiceOverEnabled else { return }
        
        let localizedText = localizationManager.localizedString(for: text, language: currentLanguage)
        
        DispatchQueue.main.async {
            UIAccessibility.post(notification: .announcement, argument: localizedText)
        }
        
        logger.info("VoiceOver announcement: \(localizedText)")
    }
    
    func setAccessibilityLabel(for view: UIView, text: String) {
        let localizedText = localizationManager.localizedString(for: text, language: currentLanguage)
        view.accessibilityLabel = localizedText
    }
    
    func setAccessibilityHint(for view: UIView, hint: String) {
        let localizedHint = localizationManager.localizedString(for: hint, language: currentLanguage)
        view.accessibilityHint = localizedHint
    }
    
    func setAccessibilityValue(for view: UIView, value: String) {
        let localizedValue = localizationManager.localizedString(for: value, language: currentLanguage)
        view.accessibilityValue = localizedValue
    }
    
    func configureAccessibilityTraits(for view: UIView, traits: UIAccessibilityTraits) {
        view.accessibilityTraits = traits
    }
    
    func makeViewAccessible(_ view: UIView, label: String, hint: String? = nil, traits: UIAccessibilityTraits = .none) {
        view.isAccessibilityElement = true
        setAccessibilityLabel(for: view, text: label)
        
        if let hint = hint {
            setAccessibilityHint(for: view, hint: hint)
        }
        
        if traits != .none {
            configureAccessibilityTraits(for: view, traits: traits)
        }
    }
    
    // MARK: - Text-to-Speech
    func speak(_ text: String, rate: Float = 0.5, pitch: Float = 1.0, volume: Float = 1.0) {
        let localizedText = localizationManager.localizedString(for: text, language: currentLanguage)
        
        let utterance = AVSpeechUtterance(string: localizedText)
        utterance.rate = rate
        utterance.pitchMultiplier = pitch
        utterance.volume = volume
        utterance.voice = getVoiceForCurrentLanguage()
        
        speechSynthesizer.speak(utterance)
        logger.info("Speaking: \(localizedText)")
    }
    
    func stopSpeaking() {
        speechSynthesizer.stopSpeaking(at: .immediate)
    }
    
    func pauseSpeaking() {
        speechSynthesizer.pauseSpeaking(at: .immediate)
    }
    
    func continueSpeaking() {
        speechSynthesizer.continueSpeaking()
    }
    
    private func getVoiceForCurrentLanguage() -> AVSpeechSynthesisVoice? {
        let languageCode = currentLanguage.languageCode
        return AVSpeechSynthesisVoice(language: languageCode)
    }
    
    // MARK: - Voice Commands
    func startVoiceCommands() {
        guard speechRecognitionStatus == .authorized else {
            logger.error("Speech recognition not authorized")
            return
        }
        
        guard !audioEngine.isRunning else {
            logger.warning("Audio engine already running")
            return
        }
        
        do {
            try startSpeechRecognition()
            isVoiceCommandsEnabled = true
            isListening = true
            logger.info("Voice commands started")
        } catch {
            logger.error("Failed to start voice commands: \(error.localizedDescription)")
        }
    }
    
    func stopVoiceCommands() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        recognitionRequest = nil
        recognitionTask = nil
        
        isVoiceCommandsEnabled = false
        isListening = false
        
        logger.info("Voice commands stopped")
    }
    
    private func startSpeechRecognition() throws {
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw AccessibilityError.speechRecognitionUnavailable
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
            self?.handleSpeechRecognitionResult(result: result, error: error)
        }
    }
    
    private func handleSpeechRecognitionResult(result: SFSpeechRecognitionResult?, error: Error?) {
        if let error = error {
            logger.error("Speech recognition error: \(error.localizedDescription)")
            stopVoiceCommands()
            return
        }
        
        guard let result = result else { return }
        
        let transcription = result.bestTranscription.formattedString
        
        if result.isFinal {
            processVoiceCommand(transcription)
        }
    }
    
    private func processVoiceCommand(_ transcription: String) {
        logger.info("Processing voice command: \(transcription)")
        
        let command = voiceCommandProcessor.processCommand(transcription, language: currentLanguage)
        
        DispatchQueue.main.async {
            self.lastVoiceCommand = command
            self.executeVoiceCommand(command)
        }
    }
    
    private func executeVoiceCommand(_ command: VoiceCommand) {
        logger.info("Executing voice command: \(command.type.rawValue)")
        
        switch command.type {
        case .navigateToScreen:
            handleNavigationCommand(command)
        case .recordPain:
            handlePainRecordingCommand(command)
        case .takeMedication:
            handleMedicationCommand(command)
        case .readLastEntry:
            handleReadLastEntryCommand()
        case .emergencyHelp:
            handleEmergencyCommand()
        case .repeatLastAnnouncement:
            handleRepeatCommand()
        case .unknown:
            handleUnknownCommand(command)
        }
        
        // Provide haptic feedback
        provideHapticFeedback(.success)
    }
    
    private func handleNavigationCommand(_ command: VoiceCommand) {
        // Implementation would depend on your navigation system
        speak("Navigating to \(command.parameters["screen"] ?? "unknown screen")")
    }
    
    private func handlePainRecordingCommand(_ command: VoiceCommand) {
        if let painLevel = command.parameters["level"] {
            speak("Recording pain level \(painLevel)")
            // Trigger pain recording with the specified level
        } else {
            speak("Please specify a pain level from 1 to 10")
        }
    }
    
    private func handleMedicationCommand(_ command: VoiceCommand) {
        if let medication = command.parameters["medication"] {
            speak("Marking \(medication) as taken")
            // Trigger medication recording
        } else {
            speak("Please specify which medication you took")
        }
    }
    
    private func handleReadLastEntryCommand() {
        // Read the last journal entry or pain record
        speak("Reading your last entry")
        // Implementation would fetch and read the last entry
    }
    
    private func handleEmergencyCommand() {
        speak("Activating emergency assistance")
        // Trigger emergency protocols
    }
    
    private func handleRepeatCommand() {
        // Repeat the last announcement
        speak("Repeating last announcement")
    }
    
    private func handleUnknownCommand(_ command: VoiceCommand) {
        speak("I didn't understand that command. Please try again.")
    }
    
    // MARK: - Haptic Feedback
    func provideHapticFeedback(_ type: HapticFeedbackType) {
        guard isHapticFeedbackEnabled else { return }
        
        DispatchQueue.main.async {
            switch type {
            case .light:
                self.impactFeedbackLight.impactOccurred()
            case .medium:
                self.impactFeedbackMedium.impactOccurred()
            case .heavy:
                self.impactFeedbackHeavy.impactOccurred()
            case .selection:
                self.selectionFeedback.selectionChanged()
            case .success:
                self.notificationFeedback.notificationOccurred(.success)
            case .warning:
                self.notificationFeedback.notificationOccurred(.warning)
            case .error:
                self.notificationFeedback.notificationOccurred(.error)
            }
        }
        
        logger.debug("Haptic feedback: \(type.rawValue)")
    }
    
    func prepareHapticFeedback() {
        impactFeedbackLight.prepare()
        impactFeedbackMedium.prepare()
        impactFeedbackHeavy.prepare()
        selectionFeedback.prepare()
        notificationFeedback.prepare()
    }
    
    // MARK: - Localization
    func changeLanguage(_ language: SupportedLanguage) {
        currentLanguage = language
        localizationManager.setCurrentLanguage(language)
        saveAccessibilitySettings()
        
        // Update speech recognizer for new language
        if isVoiceCommandsEnabled {
            stopVoiceCommands()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.startVoiceCommands()
            }
        }
        
        logger.info("Language changed to: \(language.rawValue)")
    }
    
    func getLocalizedString(_ key: String) -> String {
        return localizationManager.localizedString(for: key, language: currentLanguage)
    }
    
    func getAvailableLanguages() -> [SupportedLanguage] {
        return SupportedLanguage.allCases
    }
    
    // MARK: - High Contrast Mode
    func enableHighContrastMode(_ enabled: Bool) {
        accessibilitySettings.isHighContrastModeEnabled = enabled
        saveAccessibilitySettings()
        
        // Post notification for UI updates
        NotificationCenter.default.post(
            name: .accessibilityHighContrastChanged,
            object: nil,
            userInfo: ["enabled": enabled]
        )
        
        logger.info("High contrast mode: \(enabled)")
    }
    
    func getHighContrastColors() -> AccessibilityColors {
        return AccessibilityColors(
            primaryText: isHighContrastEnabled ? .black : .label,
            secondaryText: isHighContrastEnabled ? .darkGray : .secondaryLabel,
            background: isHighContrastEnabled ? .white : .systemBackground,
            secondaryBackground: isHighContrastEnabled ? .lightGray : .secondarySystemBackground,
            accent: isHighContrastEnabled ? .blue : .systemBlue,
            destructive: isHighContrastEnabled ? .red : .systemRed
        )
    }
    
    // MARK: - Text Scaling
    func setCustomTextScaleFactor(_ factor: CGFloat) {
        textScaleFactor = max(0.5, min(3.0, factor)) // Clamp between 0.5x and 3.0x
        accessibilitySettings.customTextScaleFactor = textScaleFactor
        saveAccessibilitySettings()
        
        // Post notification for UI updates
        NotificationCenter.default.post(
            name: .accessibilityTextScaleChanged,
            object: nil,
            userInfo: ["scaleFactor": textScaleFactor]
        )
        
        logger.info("Text scale factor set to: \(textScaleFactor)")
    }
    
    func getScaledFont(_ font: UIFont) -> UIFont {
        let scaledSize = font.pointSize * textScaleFactor
        return font.withSize(scaledSize)
    }
    
    func getScaledValue(_ value: CGFloat) -> CGFloat {
        return value * textScaleFactor
    }
    
    // MARK: - Settings Management
    func updateAccessibilitySettings(_ settings: AccessibilitySettings) {
        accessibilitySettings = settings
        
        // Apply settings
        isHapticFeedbackEnabled = settings.isHapticFeedbackEnabled
        
        if settings.isVoiceCommandsEnabled && !isVoiceCommandsEnabled {
            startVoiceCommands()
        } else if !settings.isVoiceCommandsEnabled && isVoiceCommandsEnabled {
            stopVoiceCommands()
        }
        
        if let customScale = settings.customTextScaleFactor {
            setCustomTextScaleFactor(customScale)
        }
        
        enableHighContrastMode(settings.isHighContrastModeEnabled)
        
        saveAccessibilitySettings()
        logger.info("Accessibility settings updated")
    }
    
    func resetAccessibilitySettings() {
        accessibilitySettings = AccessibilitySettings()
        textScaleFactor = 1.0
        isHapticFeedbackEnabled = true
        
        if isVoiceCommandsEnabled {
            stopVoiceCommands()
        }
        
        saveAccessibilitySettings()
        logger.info("Accessibility settings reset to defaults")
    }
    
    // MARK: - Accessibility Testing
    func performAccessibilityAudit() -> AccessibilityAuditReport {
        var issues: [AccessibilityIssue] = []
        
        // Check VoiceOver support
        if !isVoiceOverEnabled {
            issues.append(AccessibilityIssue(
                type: .voiceOverSupport,
                severity: .medium,
                description: "VoiceOver is not enabled",
                recommendation: "Enable VoiceOver in Settings > Accessibility"
            ))
        }
        
        // Check text scaling
        if textScaleFactor < 1.0 {
            issues.append(AccessibilityIssue(
                type: .textScaling,
                severity: .low,
                description: "Text scaling is below recommended size",
                recommendation: "Consider increasing text size for better readability"
            ))
        }
        
        // Check high contrast
        if !isHighContrastEnabled && UIAccessibility.isDarkerSystemColorsEnabled {
            issues.append(AccessibilityIssue(
                type: .colorContrast,
                severity: .medium,
                description: "High contrast is enabled in system but not in app",
                recommendation: "Enable high contrast mode in app settings"
            ))
        }
        
        return AccessibilityAuditReport(
            timestamp: Date(),
            issues: issues,
            overallScore: calculateAccessibilityScore(issues: issues)
        )
    }
    
    private func calculateAccessibilityScore(issues: [AccessibilityIssue]) -> Double {
        let totalPossibleScore = 100.0
        let deductions = issues.reduce(0.0) { total, issue in
            switch issue.severity {
            case .low: return total + 5.0
            case .medium: return total + 15.0
            case .high: return total + 30.0
            case .critical: return total + 50.0
            }
        }
        
        return max(0.0, totalPossibleScore - deductions)
    }
    
    // MARK: - Data Persistence
    private func loadAccessibilitySettings() {
        if let data = UserDefaults.standard.data(forKey: "accessibility_settings"),
           let settings = try? JSONDecoder().decode(AccessibilitySettings.self, from: data) {
            self.accessibilitySettings = settings
            
            // Apply loaded settings
            if let language = SupportedLanguage(rawValue: settings.preferredLanguage) {
                currentLanguage = language
            }
            
            isHapticFeedbackEnabled = settings.isHapticFeedbackEnabled
            
            if let customScale = settings.customTextScaleFactor {
                textScaleFactor = customScale
            }
        }
    }
    
    private func saveAccessibilitySettings() {
        accessibilitySettings.preferredLanguage = currentLanguage.rawValue
        
        if let data = try? JSONEncoder().encode(accessibilitySettings) {
            UserDefaults.standard.set(data, forKey: "accessibility_settings")
        }
    }
}

// MARK: - SFSpeechRecognizerDelegate
extension AccessibilityManager: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        DispatchQueue.main.async {
            if !available && self.isVoiceCommandsEnabled {
                self.stopVoiceCommands()
            }
        }
    }
}

// MARK: - AVSpeechSynthesizerDelegate
extension AccessibilityManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        logger.debug("Speech synthesis started")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        logger.debug("Speech synthesis finished")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        logger.debug("Speech synthesis cancelled")
    }
}

// MARK: - Supporting Classes
class VoiceCommandProcessor {
    private let commandPatterns: [SupportedLanguage: [VoiceCommandType: [String]]] = [
        .english: [
            .navigateToScreen: ["go to", "open", "navigate to", "show me"],
            .recordPain: ["record pain", "log pain", "pain level", "my pain is"],
            .takeMedication: ["took medication", "took medicine", "took", "medication taken"],
            .readLastEntry: ["read last", "what was my last", "last entry"],
            .emergencyHelp: ["emergency", "help", "call for help", "need help"],
            .repeatLastAnnouncement: ["repeat", "say again", "what did you say"]
        ],
        .spanish: [
            .navigateToScreen: ["ir a", "abrir", "navegar a", "muéstrame"],
            .recordPain: ["registrar dolor", "anotar dolor", "nivel de dolor", "mi dolor es"],
            .takeMedication: ["tomé medicamento", "tomé medicina", "tomé", "medicamento tomado"],
            .readLastEntry: ["leer último", "cuál fue mi último", "última entrada"],
            .emergencyHelp: ["emergencia", "ayuda", "pedir ayuda", "necesito ayuda"],
            .repeatLastAnnouncement: ["repetir", "di otra vez", "qué dijiste"]
        ],
        .german: [
            .navigateToScreen: ["gehe zu", "öffnen", "navigiere zu", "zeige mir"],
            .recordPain: ["schmerz aufzeichnen", "schmerz notieren", "schmerzniveau", "mein schmerz ist"],
            .takeMedication: ["medikament genommen", "medizin genommen", "genommen", "medikament eingenommen"],
            .readLastEntry: ["letzten eintrag lesen", "was war mein letzter", "letzter eintrag"],
            .emergencyHelp: ["notfall", "hilfe", "um hilfe rufen", "brauche hilfe"],
            .repeatLastAnnouncement: ["wiederholen", "nochmal sagen", "was hast du gesagt"]
        ],
        .french: [
            .navigateToScreen: ["aller à", "ouvrir", "naviguer vers", "montre-moi"],
            .recordPain: ["enregistrer douleur", "noter douleur", "niveau de douleur", "ma douleur est"],
            .takeMedication: ["pris médicament", "pris médecine", "pris", "médicament pris"],
            .readLastEntry: ["lire dernier", "quel était mon dernier", "dernière entrée"],
            .emergencyHelp: ["urgence", "aide", "appeler à l'aide", "besoin d'aide"],
            .repeatLastAnnouncement: ["répéter", "dire encore", "qu'as-tu dit"]
        ]
    ]
    
    func processCommand(_ transcription: String, language: SupportedLanguage) -> VoiceCommand {
        let lowercaseTranscription = transcription.lowercased()
        
        guard let patterns = commandPatterns[language] else {
            return VoiceCommand(type: .unknown, confidence: 0.0, originalText: transcription, parameters: [:])
        }
        
        for (commandType, keywords) in patterns {
            for keyword in keywords {
                if lowercaseTranscription.contains(keyword) {
                    let parameters = extractParameters(from: transcription, for: commandType, language: language)
                    let confidence = calculateConfidence(transcription: lowercaseTranscription, keyword: keyword)
                    
                    return VoiceCommand(
                        type: commandType,
                        confidence: confidence,
                        originalText: transcription,
                        parameters: parameters
                    )
                }
            }
        }
        
        return VoiceCommand(type: .unknown, confidence: 0.0, originalText: transcription, parameters: [:])
    }
    
    private func extractParameters(from transcription: String, for commandType: VoiceCommandType, language: SupportedLanguage) -> [String: String] {
        var parameters: [String: String] = [:]
        
        switch commandType {
        case .recordPain:
            if let painLevel = extractPainLevel(from: transcription) {
                parameters["level"] = String(painLevel)
            }
        case .takeMedication:
            if let medication = extractMedication(from: transcription, language: language) {
                parameters["medication"] = medication
            }
        case .navigateToScreen:
            if let screen = extractScreenName(from: transcription, language: language) {
                parameters["screen"] = screen
            }
        default:
            break
        }
        
        return parameters
    }
    
    private func extractPainLevel(from transcription: String) -> Int? {
        let numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        let lowercaseTranscription = transcription.lowercased()
        
        // Check for numeric words
        for (index, number) in numbers.enumerated() {
            if lowercaseTranscription.contains(number) {
                return index
            }
        }
        
        // Check for digits
        let regex = try? NSRegularExpression(pattern: "\\b([0-9]|10)\\b")
        if let match = regex?.firstMatch(in: transcription, range: NSRange(transcription.startIndex..., in: transcription)) {
            let matchString = String(transcription[Range(match.range, in: transcription)!])
            return Int(matchString)
        }
        
        return nil
    }
    
    private func extractMedication(from transcription: String, language: SupportedLanguage) -> String? {
        // This would be more sophisticated in a real implementation
        // For now, return a simple extraction
        let words = transcription.components(separatedBy: .whitespaces)
        
        // Look for common medication patterns
        for word in words {
            if word.lowercased().hasSuffix("in") || word.lowercased().hasSuffix("ol") {
                return word
            }
        }
        
        return nil
    }
    
    private func extractScreenName(from transcription: String, language: SupportedLanguage) -> String? {
        let screenNames: [SupportedLanguage: [String]] = [
            .english: ["dashboard", "pain", "medication", "journal", "settings"],
            .spanish: ["tablero", "dolor", "medicamento", "diario", "configuración"],
            .german: ["dashboard", "schmerz", "medikament", "tagebuch", "einstellungen"],
            .french: ["tableau de bord", "douleur", "médicament", "journal", "paramètres"]
        ]
        
        guard let screens = screenNames[language] else { return nil }
        
        let lowercaseTranscription = transcription.lowercased()
        
        for screen in screens {
            if lowercaseTranscription.contains(screen) {
                return screen
            }
        }
        
        return nil
    }
    
    private func calculateConfidence(transcription: String, keyword: String) -> Double {
        let transcriptionLength = transcription.count
        let keywordLength = keyword.count
        
        if transcriptionLength == 0 {
            return 0.0
        }
        
        // Simple confidence calculation based on keyword length vs transcription length
        let baseConfidence = Double(keywordLength) / Double(transcriptionLength)
        
        // Boost confidence if keyword appears at the beginning
        if transcription.lowercased().hasPrefix(keyword) {
            return min(1.0, baseConfidence * 1.5)
        }
        
        return min(1.0, baseConfidence)
    }
}

class SpeechToTextProcessor {
    // Additional speech-to-text processing functionality
}

class TextToSpeechProcessor {
    // Additional text-to-speech processing functionality
}

class LocalizationManager {
    private var currentLanguage: SupportedLanguage = .english
    
    // This would typically load from localization files
    private let localizations: [SupportedLanguage: [String: String]] = [
        .english: [
            "pain_recorded": "Pain level recorded",
            "medication_taken": "Medication marked as taken",
            "emergency_activated": "Emergency assistance activated",
            "navigation_complete": "Navigation completed"
        ],
        .spanish: [
            "pain_recorded": "Nivel de dolor registrado",
            "medication_taken": "Medicamento marcado como tomado",
            "emergency_activated": "Asistencia de emergencia activada",
            "navigation_complete": "Navegación completada"
        ],
        .german: [
            "pain_recorded": "Schmerzniveau aufgezeichnet",
            "medication_taken": "Medikament als eingenommen markiert",
            "emergency_activated": "Notfallhilfe aktiviert",
            "navigation_complete": "Navigation abgeschlossen"
        ],
        .french: [
            "pain_recorded": "Niveau de douleur enregistré",
            "medication_taken": "Médicament marqué comme pris",
            "emergency_activated": "Assistance d'urgence activée",
            "navigation_complete": "Navigation terminée"
        ]
    ]
    
    func setCurrentLanguage(_ language: SupportedLanguage) {
        currentLanguage = language
    }
    
    func localizedString(for key: String, language: SupportedLanguage) -> String {
        return localizations[language]?[key] ?? key
    }
}

// MARK: - Supporting Types
struct AccessibilitySettings: Codable {
    var isVoiceOverEnabled = false
    var isVoiceCommandsEnabled = false
    var isHapticFeedbackEnabled = true
    var isHighContrastModeEnabled = false
    var isReduceMotionEnabled = false
    var customTextScaleFactor: CGFloat?
    var preferredLanguage = SupportedLanguage.english.rawValue
    var speechRate: Float = 0.5
    var speechPitch: Float = 1.0
    var speechVolume: Float = 1.0
}

enum SupportedLanguage: String, CaseIterable, Codable {
    case english = "en"
    case spanish = "es"
    case german = "de"
    case french = "fr"
    
    var languageCode: String {
        return self.rawValue
    }
    
    var displayName: String {
        switch self {
        case .english: return "English"
        case .spanish: return "Español"
        case .german: return "Deutsch"
        case .french: return "Français"
        }
    }
}

enum HapticFeedbackType: String, CaseIterable {
    case light = "Light"
    case medium = "Medium"
    case heavy = "Heavy"
    case selection = "Selection"
    case success = "Success"
    case warning = "Warning"
    case error = "Error"
}

enum VoiceCommandType: String, CaseIterable {
    case navigateToScreen = "Navigate to Screen"
    case recordPain = "Record Pain"
    case takeMedication = "Take Medication"
    case readLastEntry = "Read Last Entry"
    case emergencyHelp = "Emergency Help"
    case repeatLastAnnouncement = "Repeat Last Announcement"
    case unknown = "Unknown"
}

struct VoiceCommand {
    var type: VoiceCommandType
    var confidence: Double
    var originalText: String
    var parameters: [String: String]
    var timestamp = Date()
}

enum SpeechRecognitionStatus {
    case notDetermined
    case denied
    case restricted
    case authorized
    
    init(from authorizationStatus: SFSpeechRecognizerAuthorizationStatus) {
        switch authorizationStatus {
        case .notDetermined:
            self = .notDetermined
        case .denied:
            self = .denied
        case .restricted:
            self = .restricted
        case .authorized:
            self = .authorized
        @unknown default:
            self = .notDetermined
        }
    }
}

struct AccessibilityColors {
    var primaryText: UIColor
    var secondaryText: UIColor
    var background: UIColor
    var secondaryBackground: UIColor
    var accent: UIColor
    var destructive: UIColor
}

struct AccessibilityAuditReport {
    var timestamp: Date
    var issues: [AccessibilityIssue]
    var overallScore: Double
}

struct AccessibilityIssue {
    var type: AccessibilityIssueType
    var severity: AccessibilityIssueSeverity
    var description: String
    var recommendation: String
}

enum AccessibilityIssueType: String, CaseIterable {
    case voiceOverSupport = "VoiceOver Support"
    case textScaling = "Text Scaling"
    case colorContrast = "Color Contrast"
    case hapticFeedback = "Haptic Feedback"
    case voiceCommands = "Voice Commands"
}

enum AccessibilityIssueSeverity: String, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
    case critical = "Critical"
}

enum AccessibilityError: LocalizedError {
    case speechRecognitionUnavailable
    case speechSynthesisUnavailable
    case microphoneAccessDenied
    case languageNotSupported
    
    var errorDescription: String? {
        switch self {
        case .speechRecognitionUnavailable:
            return "Speech recognition is not available"
        case .speechSynthesisUnavailable:
            return "Speech synthesis is not available"
        case .microphoneAccessDenied:
            return "Microphone access was denied"
        case .languageNotSupported:
            return "The selected language is not supported"
        }
    }
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let accessibilityHighContrastChanged = Notification.Name("accessibilityHighContrastChanged")
    static let accessibilityTextScaleChanged = Notification.Name("accessibilityTextScaleChanged")
    static let accessibilityVoiceCommandReceived = Notification.Name("accessibilityVoiceCommandReceived")
}