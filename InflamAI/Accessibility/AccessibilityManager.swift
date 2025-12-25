//
//  AccessibilityManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import SwiftUI
import UIKit
import AVFoundation
import os.log
import Combine

// MARK: - Accessibility Manager
class AccessibilityManager: NSObject, ObservableObject {
    
    static let shared = AccessibilityManager()
    
    private let logger = Logger(subsystem: "InflamAI", category: "Accessibility")
    private let speechSynthesizer = AVSpeechSynthesizer()
    private let notificationCenter = NotificationCenter.default
    
    // Published properties
    @Published var isVoiceOverEnabled = false
    @Published var isHighContrastEnabled = false
    @Published var isReducedMotionEnabled = false
    @Published var isReducedTransparencyEnabled = false
    @Published var isBoldTextEnabled = false
    @Published var isLargerTextEnabled = false
    @Published var isButtonShapesEnabled = false
    @Published var isOnOffLabelsEnabled = false
    @Published var isDifferentiateWithoutColorEnabled = false
    @Published var isInvertColorsEnabled = false
    @Published var isGrayscaleEnabled = false
    @Published var isAssistiveTouchEnabled = false
    @Published var isSwitchControlEnabled = false
    @Published var isGuidedAccessEnabled = false
    
    // Custom accessibility settings
    @Published var accessibilitySettings = AccessibilitySettings()
    @Published var voiceSettings = VoiceSettings()
    @Published var visualSettings = VisualSettings()
    @Published var motorSettings = MotorSettings()
    @Published var cognitiveSettings = CognitiveSettings()
    
    // Language and localization
    @Published var currentLanguage: AppLanguage = .english
    @Published var supportedLanguages: [AppLanguage] = AppLanguage.allCases
    @Published var isRTLLanguage = false
    
    // Font and text settings
    @Published var currentFontSize: FontSize = .medium
    @Published var lineSpacing: LineSpacing = .normal
    @Published var letterSpacing: LetterSpacing = .normal
    
    // Color and contrast settings
    @Published var contrastLevel: ContrastLevel = .normal
    @Published var colorBlindnessType: ColorBlindnessType = .none
    @Published var customColorScheme: CustomColorScheme?
    
    // Navigation and interaction
    @Published var navigationStyle: NavigationStyle = .standard
    @Published var hapticFeedbackEnabled = true
    @Published var hapticFeedbackIntensity: HapticIntensity = .medium
    @Published var speechPriority: SpeechPriority = .normal
    @Published var gestureTimeout: TimeInterval = 3.0
    @Published var gestureSensitivity: GestureSensitivity = .normal
    
    // Internal state
    private var cancellables = Set<AnyCancellable>()
    private var accessibilityObservers: [NSObjectProtocol] = []
    private var currentSpeechUtterance: AVSpeechUtterance?
    private var speechQueue: [String] = []
    private var isSpeaking = false
    
    override init() {
        super.init()
        setupAccessibilityObservers()
        loadAccessibilitySettings()
        updateAccessibilityStatus()
        setupSpeechSynthesizer()
    }
    
    deinit {
        removeAccessibilityObservers()
    }
    
    // MARK: - Public Methods
    
    func updateAccessibilitySettings(_ settings: AccessibilitySettings) {
        accessibilitySettings = settings
        saveAccessibilitySettings()
        applyAccessibilitySettings()
        
        logger.info("Accessibility settings updated")
    }
    
    func updateVoiceSettings(_ settings: VoiceSettings) {
        voiceSettings = settings
        saveAccessibilitySettings()
        configureSpeechSynthesizer()
        
        logger.info("Voice settings updated")
    }
    
    func updateVisualSettings(_ settings: VisualSettings) {
        visualSettings = settings
        saveAccessibilitySettings()
        applyVisualSettings()
        
        logger.info("Visual settings updated")
    }
    
    func updateMotorSettings(_ settings: MotorSettings) {
        motorSettings = settings
        saveAccessibilitySettings()
        applyMotorSettings()
        
        logger.info("Motor settings updated")
    }
    
    func updateCognitiveSettings(_ settings: CognitiveSettings) {
        cognitiveSettings = settings
        saveAccessibilitySettings()
        applyCognitiveSettings()
        
        logger.info("Cognitive settings updated")
    }
    
    func changeLanguage(to language: AppLanguage) {
        currentLanguage = language
        isRTLLanguage = language.isRTL
        saveAccessibilitySettings()
        
        // Update app language
        UserDefaults.standard.set([language.code], forKey: "AppleLanguages")
        UserDefaults.standard.synchronize()
        
        logger.info("Language changed to \(language.displayName)")
    }
    
    func updateFontSize(_ size: FontSize) {
        currentFontSize = size
        saveAccessibilitySettings()
        applyFontSettings()
        
        logger.info("Font size updated to \(size)")
    }
    
    func updateContrastLevel(_ level: ContrastLevel) {
        contrastLevel = level
        saveAccessibilitySettings()
        applyContrastSettings()
        
        logger.info("Contrast level updated to \(level)")
    }
    
    func updateColorBlindnessSupport(_ type: ColorBlindnessType) {
        colorBlindnessType = type
        saveAccessibilitySettings()
        applyColorBlindnessSettings()
        
        logger.info("Color blindness support updated to \(type)")
    }
    
    func speak(_ text: String, priority: SpeechPriority = .normal, interrupt: Bool = false) {
        guard accessibilitySettings.enableSpeech else { return }
        
        if interrupt {
            stopSpeaking()
        }
        
        if priority == .urgent || !isSpeaking {
            speakImmediately(text)
        } else {
            speechQueue.append(text)
        }
        
        logger.debug("Speech requested: \(text)")
    }
    
    func stopSpeaking() {
        speechSynthesizer.stopSpeaking(at: .immediate)
        speechQueue.removeAll()
        isSpeaking = false
        
        logger.debug("Speech stopped")
    }
    
    func pauseSpeaking() {
        speechSynthesizer.pauseSpeaking(at: .immediate)
        
        logger.debug("Speech paused")
    }
    
    func continueSpeaking() {
        speechSynthesizer.continueSpeaking()
        
        logger.debug("Speech continued")
    }
    
    func provideHapticFeedback(_ type: HapticFeedbackType) {
        guard hapticFeedbackEnabled else { return }
        
        switch type {
        case .light:
            let feedback = UIImpactFeedbackGenerator(style: .light)
            feedback.impactOccurred(intensity: hapticFeedbackIntensity.value)
            
        case .medium:
            let feedback = UIImpactFeedbackGenerator(style: .medium)
            feedback.impactOccurred(intensity: hapticFeedbackIntensity.value)
            
        case .heavy:
            let feedback = UIImpactFeedbackGenerator(style: .heavy)
            feedback.impactOccurred(intensity: hapticFeedbackIntensity.value)
            
        case .success:
            let feedback = UINotificationFeedbackGenerator()
            feedback.notificationOccurred(.success)
            
        case .warning:
            let feedback = UINotificationFeedbackGenerator()
            feedback.notificationOccurred(.warning)
            
        case .error:
            let feedback = UINotificationFeedbackGenerator()
            feedback.notificationOccurred(.error)
            
        case .selection:
            let feedback = UISelectionFeedbackGenerator()
            feedback.selectionChanged()
        }
        
        logger.debug("Haptic feedback provided: \(type)")
    }
    
    func announceForAccessibility(_ announcement: String, priority: SpeechPriority = .normal) {
        // Post accessibility announcement
        UIAccessibility.post(notification: .announcement, argument: announcement)
        
        // Also speak if speech is enabled
        if accessibilitySettings.enableSpeech {
            speak(announcement, priority: priority)
        }
        
        logger.debug("Accessibility announcement: \(announcement)")
    }
    
    func createAccessibilityElement(label: String, hint: String? = nil, traits: UIAccessibilityTraits = [], frame: CGRect = .zero) -> AccessibilityElement {
        return AccessibilityElement(
            label: label,
            hint: hint,
            traits: traits,
            frame: frame,
            isAccessibilityElement: true
        )
    }
    
    func getAccessibilityLabel(for key: String, arguments: [String] = []) -> String {
        let localizedString = NSLocalizedString(key, comment: "")
        
        if arguments.isEmpty {
            return localizedString
        } else {
            return String(format: localizedString, arguments: arguments)
        }
    }
    
    func getAccessibilityHint(for key: String) -> String {
        return NSLocalizedString("\(key)_hint", comment: "")
    }
    
    func isAccessibilityFeatureEnabled(_ feature: AccessibilityFeature) -> Bool {
        switch feature {
        case .voiceOver:
            return isVoiceOverEnabled
        case .highContrast:
            return isHighContrastEnabled
        case .reducedMotion:
            return isReducedMotionEnabled
        case .reducedTransparency:
            return isReducedTransparencyEnabled
        case .boldText:
            return isBoldTextEnabled
        case .largerText:
            return isLargerTextEnabled
        case .buttonShapes:
            return isButtonShapesEnabled
        case .onOffLabels:
            return isOnOffLabelsEnabled
        case .differentiateWithoutColor:
            return isDifferentiateWithoutColorEnabled
        case .invertColors:
            return isInvertColorsEnabled
        case .grayscale:
            return isGrayscaleEnabled
        case .assistiveTouch:
            return isAssistiveTouchEnabled
        case .switchControl:
            return isSwitchControlEnabled
        case .guidedAccess:
            return isGuidedAccessEnabled
        }
    }
    
    func getRecommendedSettings() -> AccessibilityRecommendations {
        var recommendations = AccessibilityRecommendations()
        
        // Analyze current accessibility status and provide recommendations
        if isVoiceOverEnabled {
            recommendations.voiceOverOptimizations = [
                "Enable detailed descriptions for images",
                "Use semantic headings for better navigation",
                "Provide clear button labels and hints"
            ]
        }
        
        if isHighContrastEnabled || contrastLevel != .normal {
            recommendations.visualOptimizations = [
                "Use high contrast color schemes",
                "Increase border thickness",
                "Avoid color-only information"
            ]
        }
        
        if isReducedMotionEnabled {
            recommendations.motionOptimizations = [
                "Disable auto-playing animations",
                "Use fade transitions instead of sliding",
                "Provide static alternatives to animated content"
            ]
        }
        
        if motorSettings.enableLargerTouchTargets {
            recommendations.motorOptimizations = [
                "Increase button sizes",
                "Add more spacing between interactive elements",
                "Enable gesture alternatives"
            ]
        }
        
        return recommendations
    }
    
    func exportAccessibilitySettings() -> Data? {
        do {
            let export = AccessibilityExport(
                timestamp: Date(),
                accessibilitySettings: accessibilitySettings,
                voiceSettings: voiceSettings,
                visualSettings: visualSettings,
                motorSettings: motorSettings,
                cognitiveSettings: cognitiveSettings,
                language: currentLanguage,
                fontSize: currentFontSize,
                contrastLevel: contrastLevel,
                colorBlindnessType: colorBlindnessType
            )
            
            return try JSONEncoder().encode(export)
        } catch {
            logger.error("Failed to export accessibility settings: \(error.localizedDescription)")
            return nil
        }
    }
    
    func importAccessibilitySettings(from data: Data) -> Bool {
        do {
            let export = try JSONDecoder().decode(AccessibilityExport.self, from: data)
            
            accessibilitySettings = export.accessibilitySettings
            voiceSettings = export.voiceSettings
            visualSettings = export.visualSettings
            motorSettings = export.motorSettings
            cognitiveSettings = export.cognitiveSettings
            currentLanguage = export.language
            currentFontSize = export.fontSize
            contrastLevel = export.contrastLevel
            colorBlindnessType = export.colorBlindnessType
            
            saveAccessibilitySettings()
            applyAccessibilitySettings()
            
            logger.info("Accessibility settings imported successfully")
            return true
        } catch {
            logger.error("Failed to import accessibility settings: \(error.localizedDescription)")
            return false
        }
    }
    
    // MARK: - Private Methods
    
    private func setupAccessibilityObservers() {
        // VoiceOver
        let voiceOverObserver = notificationCenter.addObserver(
            forName: UIAccessibility.voiceOverStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isVoiceOverEnabled = UIAccessibility.isVoiceOverRunning
        }
        accessibilityObservers.append(voiceOverObserver)
        
        // High Contrast
        let contrastObserver = notificationCenter.addObserver(
            forName: UIAccessibility.darkerSystemColorsStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isHighContrastEnabled = UIAccessibility.isDarkerSystemColorsEnabled
        }
        accessibilityObservers.append(contrastObserver)
        
        // Reduced Motion
        let motionObserver = notificationCenter.addObserver(
            forName: UIAccessibility.reduceMotionStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isReducedMotionEnabled = UIAccessibility.isReduceMotionEnabled
        }
        accessibilityObservers.append(motionObserver)
        
        // Reduced Transparency
        let transparencyObserver = notificationCenter.addObserver(
            forName: UIAccessibility.reduceTransparencyStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isReducedTransparencyEnabled = UIAccessibility.isReduceTransparencyEnabled
        }
        accessibilityObservers.append(transparencyObserver)
        
        // Bold Text
        let boldTextObserver = notificationCenter.addObserver(
            forName: UIAccessibility.boldTextStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isBoldTextEnabled = UIAccessibility.isBoldTextEnabled
        }
        accessibilityObservers.append(boldTextObserver)
        
        // Button Shapes
        let buttonShapesObserver = notificationCenter.addObserver(
            forName: UIAccessibility.buttonShapesEnabledStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isButtonShapesEnabled = UIAccessibility.isButtonShapesEnabled
        }
        accessibilityObservers.append(buttonShapesObserver)
        
        // On/Off Labels
        let onOffLabelsObserver = notificationCenter.addObserver(
            forName: UIAccessibility.onOffSwitchLabelsDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isOnOffLabelsEnabled = UIAccessibility.isOnOffSwitchLabelsEnabled
        }
        accessibilityObservers.append(onOffLabelsObserver)
        
        // Differentiate Without Color
        let colorObserver = notificationCenter.addObserver(
            forName: UIAccessibility.differentiateWithoutColorDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isDifferentiateWithoutColorEnabled = UIAccessibility.shouldDifferentiateWithoutColor
        }
        accessibilityObservers.append(colorObserver)
        
        // Invert Colors
        let invertColorsObserver = notificationCenter.addObserver(
            forName: UIAccessibility.invertColorsStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isInvertColorsEnabled = UIAccessibility.isInvertColorsEnabled
        }
        accessibilityObservers.append(invertColorsObserver)
        
        // Grayscale
        let grayscaleObserver = notificationCenter.addObserver(
            forName: UIAccessibility.grayscaleStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isGrayscaleEnabled = UIAccessibility.isGrayscaleEnabled
        }
        accessibilityObservers.append(grayscaleObserver)
        
        // Assistive Touch
        let assistiveTouchObserver = notificationCenter.addObserver(
            forName: UIAccessibility.assistiveTouchStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isAssistiveTouchEnabled = UIAccessibility.isAssistiveTouchRunning
        }
        accessibilityObservers.append(assistiveTouchObserver)
        
        // Switch Control
        let switchControlObserver = notificationCenter.addObserver(
            forName: UIAccessibility.switchControlStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isSwitchControlEnabled = UIAccessibility.isSwitchControlRunning
        }
        accessibilityObservers.append(switchControlObserver)
        
        // Guided Access
        let guidedAccessObserver = notificationCenter.addObserver(
            forName: UIAccessibility.guidedAccessStatusDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.isGuidedAccessEnabled = UIAccessibility.isGuidedAccessEnabled
        }
        accessibilityObservers.append(guidedAccessObserver)
    }
    
    private func removeAccessibilityObservers() {
        accessibilityObservers.forEach { observer in
            notificationCenter.removeObserver(observer)
        }
        accessibilityObservers.removeAll()
    }
    
    private func updateAccessibilityStatus() {
        isVoiceOverEnabled = UIAccessibility.isVoiceOverRunning
        isHighContrastEnabled = UIAccessibility.isDarkerSystemColorsEnabled
        isReducedMotionEnabled = UIAccessibility.isReduceMotionEnabled
        isReducedTransparencyEnabled = UIAccessibility.isReduceTransparencyEnabled
        isBoldTextEnabled = UIAccessibility.isBoldTextEnabled
        isLargerTextEnabled = UIApplication.shared.preferredContentSizeCategory.isAccessibilityCategory
        isButtonShapesEnabled = UIAccessibility.isButtonShapesEnabled
        isOnOffLabelsEnabled = UIAccessibility.isOnOffSwitchLabelsEnabled
        isDifferentiateWithoutColorEnabled = UIAccessibility.shouldDifferentiateWithoutColor
        isInvertColorsEnabled = UIAccessibility.isInvertColorsEnabled
        isGrayscaleEnabled = UIAccessibility.isGrayscaleEnabled
        isAssistiveTouchEnabled = UIAccessibility.isAssistiveTouchRunning
        isSwitchControlEnabled = UIAccessibility.isSwitchControlRunning
        isGuidedAccessEnabled = UIAccessibility.isGuidedAccessEnabled
    }
    
    private func setupSpeechSynthesizer() {
        speechSynthesizer.delegate = self
        configureSpeechSynthesizer()
    }
    
    private func configureSpeechSynthesizer() {
        // Configure speech settings based on user preferences
        // This will be applied when creating speech utterances
    }
    
    private func speakImmediately(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: currentLanguage.code)
        utterance.rate = voiceSettings.speechRate
        utterance.pitchMultiplier = voiceSettings.pitchMultiplier
        utterance.volume = voiceSettings.volume
        
        currentSpeechUtterance = utterance
        speechSynthesizer.speak(utterance)
        isSpeaking = true
    }
    
    private func speakNextInQueue() {
        guard !speechQueue.isEmpty else {
            isSpeaking = false
            return
        }
        
        let text = speechQueue.removeFirst()
        speakImmediately(text)
    }
    
    private func applyAccessibilitySettings() {
        applyVisualSettings()
        applyMotorSettings()
        applyCognitiveSettings()
        applyFontSettings()
        applyContrastSettings()
        applyColorBlindnessSettings()
    }
    
    private func applyVisualSettings() {
        // Apply visual accessibility settings
        // This would typically involve updating UI elements
    }
    
    private func applyMotorSettings() {
        // Apply motor accessibility settings
        gestureTimeout = motorSettings.gestureTimeout
        gestureSensitivity = motorSettings.gestureSensitivity
    }
    
    private func applyCognitiveSettings() {
        // Apply cognitive accessibility settings
        // This might involve simplifying UI or providing additional guidance
    }
    
    private func applyFontSettings() {
        // Apply font size settings
        // This would typically involve updating the app's font scale
    }
    
    private func applyContrastSettings() {
        // Apply contrast settings
        // This would involve updating color schemes
    }
    
    private func applyColorBlindnessSettings() {
        // Apply color blindness accommodations
        // This would involve adjusting color palettes
    }
    
    private func saveAccessibilitySettings() {
        let settings = AccessibilitySettingsData(
            accessibilitySettings: accessibilitySettings,
            voiceSettings: voiceSettings,
            visualSettings: visualSettings,
            motorSettings: motorSettings,
            cognitiveSettings: cognitiveSettings,
            language: currentLanguage,
            fontSize: currentFontSize,
            lineSpacing: lineSpacing,
            letterSpacing: letterSpacing,
            contrastLevel: contrastLevel,
            colorBlindnessType: colorBlindnessType,
            navigationStyle: navigationStyle,
            hapticFeedbackEnabled: hapticFeedbackEnabled,
            hapticFeedbackIntensity: hapticFeedbackIntensity,
            speechPriority: speechPriority,
            gestureTimeout: gestureTimeout,
            gestureSensitivity: gestureSensitivity
        )
        
        do {
            let data = try JSONEncoder().encode(settings)
            UserDefaults.standard.set(data, forKey: "AccessibilitySettings")
        } catch {
            logger.error("Failed to save accessibility settings: \(error.localizedDescription)")
        }
    }
    
    private func loadAccessibilitySettings() {
        guard let data = UserDefaults.standard.data(forKey: "AccessibilitySettings"),
              let settings = try? JSONDecoder().decode(AccessibilitySettingsData.self, from: data) else {
            return
        }
        
        accessibilitySettings = settings.accessibilitySettings
        voiceSettings = settings.voiceSettings
        visualSettings = settings.visualSettings
        motorSettings = settings.motorSettings
        cognitiveSettings = settings.cognitiveSettings
        currentLanguage = settings.language
        currentFontSize = settings.fontSize
        lineSpacing = settings.lineSpacing
        letterSpacing = settings.letterSpacing
        contrastLevel = settings.contrastLevel
        colorBlindnessType = settings.colorBlindnessType
        navigationStyle = settings.navigationStyle
        hapticFeedbackEnabled = settings.hapticFeedbackEnabled
        hapticFeedbackIntensity = settings.hapticFeedbackIntensity
        speechPriority = settings.speechPriority
        gestureTimeout = settings.gestureTimeout
        gestureSensitivity = settings.gestureSensitivity
        
        isRTLLanguage = currentLanguage.isRTL
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension AccessibilityManager: AVSpeechSynthesizerDelegate {
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        logger.debug("Speech started")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        logger.debug("Speech finished")
        speakNextInQueue()
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        logger.debug("Speech cancelled")
        isSpeaking = false
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didPause utterance: AVSpeechUtterance) {
        logger.debug("Speech paused")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didContinue utterance: AVSpeechUtterance) {
        logger.debug("Speech continued")
    }
}

// MARK: - Supporting Types

struct AccessibilitySettings: Codable {
    var enableSpeech = true
    var enableHapticFeedback = true
    var enableHighContrast = false
    var enableLargeText = false
    var enableSimplifiedUI = false
    var enableVoiceControl = false
    var enableGestureAlternatives = true
    var enableAudioDescriptions = true
    var enableCaptions = true
    var enableReducedMotion = false
    var enableFocusIndicators = true
    var enableKeyboardNavigation = true
}

struct VoiceSettings: Codable {
    var speechRate: Float = 0.5
    var pitchMultiplier: Float = 1.0
    var volume: Float = 1.0
    var voiceIdentifier: String?
    var enableSpeechInterruption = true
    var enableSpeechQueue = true
    var speechLanguage: String = "en-US"
}

struct VisualSettings: Codable {
    var enableHighContrast = false
    var enableDarkMode = false
    var enableColorInversion = false
    var enableGrayscale = false
    var contrastRatio: Double = 1.0
    var brightnessAdjustment: Double = 0.0
    var enableCustomColors = false
    var enablePatternFills = false
    var enableBorderHighlights = true
    var enableFocusRing = true
}

struct MotorSettings: Codable {
    var enableLargerTouchTargets = false
    var touchTargetMinimumSize: CGFloat = 44.0
    var enableGestureAlternatives = true
    var gestureTimeout: TimeInterval = 3.0
    var gestureSensitivity: GestureSensitivity = .normal
    var enableStickyKeys = false
    var enableSlowKeys = false
    var enableBounceKeys = false
    var enableMouseKeys = false
    var enableDwellControl = false
    var dwellTime: TimeInterval = 1.0
}

struct CognitiveSettings: Codable {
    var enableSimplifiedLanguage = false
    var enableStepByStepGuidance = false
    var enableProgressIndicators = true
    var enableConfirmationDialogs = true
    var enableUndoActions = true
    var enableAutoSave = true
    var enableReminders = true
    var enableContextualHelp = true
    var enableErrorPrevention = true
    var enableConsistentNavigation = true
}

enum AppLanguage: String, CaseIterable, Codable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case russian = "ru"
    case chinese = "zh"
    case japanese = "ja"
    case korean = "ko"
    case arabic = "ar"
    case hebrew = "he"
    
    var displayName: String {
        switch self {
        case .english: return "English"
        case .spanish: return "Español"
        case .french: return "Français"
        case .german: return "Deutsch"
        case .italian: return "Italiano"
        case .portuguese: return "Português"
        case .russian: return "Русский"
        case .chinese: return "中文"
        case .japanese: return "日本語"
        case .korean: return "한국어"
        case .arabic: return "العربية"
        case .hebrew: return "עברית"
        }
    }
    
    var code: String {
        return rawValue
    }
    
    var isRTL: Bool {
        return self == .arabic || self == .hebrew
    }
}

enum FontSize: String, CaseIterable, Codable {
    case extraSmall = "extra_small"
    case small = "small"
    case medium = "medium"
    case large = "large"
    case extraLarge = "extra_large"
    case accessibility1 = "accessibility_1"
    case accessibility2 = "accessibility_2"
    case accessibility3 = "accessibility_3"
    case accessibility4 = "accessibility_4"
    case accessibility5 = "accessibility_5"
    
    var scaleFactor: CGFloat {
        switch self {
        case .extraSmall: return 0.8
        case .small: return 0.9
        case .medium: return 1.0
        case .large: return 1.1
        case .extraLarge: return 1.2
        case .accessibility1: return 1.3
        case .accessibility2: return 1.4
        case .accessibility3: return 1.5
        case .accessibility4: return 1.6
        case .accessibility5: return 1.7
        }
    }
}

enum LineSpacing: String, CaseIterable, Codable {
    case tight = "tight"
    case normal = "normal"
    case loose = "loose"
    case extraLoose = "extra_loose"
    
    var multiplier: CGFloat {
        switch self {
        case .tight: return 1.0
        case .normal: return 1.2
        case .loose: return 1.4
        case .extraLoose: return 1.6
        }
    }
}

enum LetterSpacing: String, CaseIterable, Codable {
    case tight = "tight"
    case normal = "normal"
    case loose = "loose"
    case extraLoose = "extra_loose"
    
    var points: CGFloat {
        switch self {
        case .tight: return -0.5
        case .normal: return 0.0
        case .loose: return 0.5
        case .extraLoose: return 1.0
        }
    }
}

enum ContrastLevel: String, CaseIterable, Codable {
    case low = "low"
    case normal = "normal"
    case high = "high"
    case maximum = "maximum"
    
    var ratio: Double {
        switch self {
        case .low: return 3.0
        case .normal: return 4.5
        case .high: return 7.0
        case .maximum: return 21.0
        }
    }
}

enum ColorBlindnessType: String, CaseIterable, Codable {
    case none = "none"
    case protanopia = "protanopia"
    case deuteranopia = "deuteranopia"
    case tritanopia = "tritanopia"
    case protanomaly = "protanomaly"
    case deuteranomaly = "deuteranomaly"
    case tritanomaly = "tritanomaly"
    case monochromacy = "monochromacy"
    
    var displayName: String {
        switch self {
        case .none: return "None"
        case .protanopia: return "Protanopia (Red-blind)"
        case .deuteranopia: return "Deuteranopia (Green-blind)"
        case .tritanopia: return "Tritanopia (Blue-blind)"
        case .protanomaly: return "Protanomaly (Red-weak)"
        case .deuteranomaly: return "Deuteranomaly (Green-weak)"
        case .tritanomaly: return "Tritanomaly (Blue-weak)"
        case .monochromacy: return "Monochromacy (Color-blind)"
        }
    }
}

struct CustomColorScheme: Codable {
    var primaryColor: String
    var secondaryColor: String
    var backgroundColor: String
    var textColor: String
    var accentColor: String
    var errorColor: String
    var warningColor: String
    var successColor: String
}

enum NavigationStyle: String, CaseIterable, Codable {
    case standard = "standard"
    case simplified = "simplified"
    case tabBased = "tab_based"
    case listBased = "list_based"
    case gestureOnly = "gesture_only"
    case voiceControlled = "voice_controlled"
}

enum HapticFeedbackType {
    case light
    case medium
    case heavy
    case success
    case warning
    case error
    case selection
}

enum HapticIntensity: String, CaseIterable, Codable {
    case light = "light"
    case medium = "medium"
    case strong = "strong"
    
    var value: CGFloat {
        switch self {
        case .light: return 0.5
        case .medium: return 1.0
        case .strong: return 1.5
        }
    }
}

enum SpeechPriority: String, CaseIterable, Codable {
    case low = "low"
    case normal = "normal"
    case high = "high"
    case urgent = "urgent"
}

enum GestureSensitivity: String, CaseIterable, Codable {
    case low = "low"
    case normal = "normal"
    case high = "high"
    
    var threshold: CGFloat {
        switch self {
        case .low: return 20.0
        case .normal: return 10.0
        case .high: return 5.0
        }
    }
}

enum AccessibilityFeature {
    case voiceOver
    case highContrast
    case reducedMotion
    case reducedTransparency
    case boldText
    case largerText
    case buttonShapes
    case onOffLabels
    case differentiateWithoutColor
    case invertColors
    case grayscale
    case assistiveTouch
    case switchControl
    case guidedAccess
}

struct AccessibilityElement {
    let label: String
    let hint: String?
    let traits: UIAccessibilityTraits
    let frame: CGRect
    let isAccessibilityElement: Bool
}

struct AccessibilityRecommendations {
    var voiceOverOptimizations: [String] = []
    var visualOptimizations: [String] = []
    var motionOptimizations: [String] = []
    var motorOptimizations: [String] = []
    var cognitiveOptimizations: [String] = []
}

struct AccessibilitySettingsData: Codable {
    let accessibilitySettings: AccessibilitySettings
    let voiceSettings: VoiceSettings
    let visualSettings: VisualSettings
    let motorSettings: MotorSettings
    let cognitiveSettings: CognitiveSettings
    let language: AppLanguage
    let fontSize: FontSize
    let lineSpacing: LineSpacing
    let letterSpacing: LetterSpacing
    let contrastLevel: ContrastLevel
    let colorBlindnessType: ColorBlindnessType
    let navigationStyle: NavigationStyle
    let hapticFeedbackEnabled: Bool
    let hapticFeedbackIntensity: HapticIntensity
    let speechPriority: SpeechPriority
    let gestureTimeout: TimeInterval
    let gestureSensitivity: GestureSensitivity
}

struct AccessibilityExport: Codable {
    let timestamp: Date
    let accessibilitySettings: AccessibilitySettings
    let voiceSettings: VoiceSettings
    let visualSettings: VisualSettings
    let motorSettings: MotorSettings
    let cognitiveSettings: CognitiveSettings
    let language: AppLanguage
    let fontSize: FontSize
    let contrastLevel: ContrastLevel
    let colorBlindnessType: ColorBlindnessType
}