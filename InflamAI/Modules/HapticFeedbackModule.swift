//
//  HapticFeedbackModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import UIKit
import CoreHaptics
import Combine

// MARK: - Haptic Feedback Models

enum HapticFeedbackType {
    // Basic feedback types
    case success
    case warning
    case error
    case selection
    case impact(intensity: HapticIntensity)
    
    // Health-specific feedback
    case symptomLogged
    case medicationTaken
    case medicationMissed
    case appointmentReminder
    case emergencyAlert
    case painLevelChange(level: Int)
    case moodChange(level: Int)
    
    // Navigation feedback
    case pageTransition
    case tabSwitch
    case modalPresent
    case modalDismiss
    
    // Interaction feedback
    case buttonPress
    case longPress
    case swipeAction
    case pullToRefresh
    case dataSync
    
    // Voice command feedback
    case voiceCommandStart
    case voiceCommandEnd
    case voiceCommandRecognized
    case voiceCommandError
    
    // Custom patterns
    case heartbeat
    case breathingPattern
    case progressComplete
    case timerTick
    case custom(pattern: HapticPattern)
}

enum HapticIntensity: Float, CaseIterable {
    case light = 0.3
    case medium = 0.6
    case heavy = 1.0
    
    var description: String {
        switch self {
        case .light: return "Light"
        case .medium: return "Medium"
        case .heavy: return "Heavy"
        }
    }
}

struct HapticPattern {
    let events: [HapticEvent]
    let duration: TimeInterval
    
    init(events: [HapticEvent]) {
        self.events = events
        self.duration = events.last?.time ?? 0
    }
}

struct HapticEvent {
    let time: TimeInterval
    let intensity: Float
    let sharpness: Float
    let duration: TimeInterval
    
    init(time: TimeInterval, intensity: Float = 1.0, sharpness: Float = 1.0, duration: TimeInterval = 0.1) {
        self.time = time
        self.intensity = intensity
        self.sharpness = sharpness
        self.duration = duration
    }
}

struct HapticSettings {
    var isEnabled: Bool = true
    var intensity: HapticIntensity = .medium
    var enableSystemHaptics: Bool = true
    var enableCustomPatterns: Bool = true
    var enableHealthFeedback: Bool = true
    var enableNavigationFeedback: Bool = true
    var enableVoiceFeedback: Bool = true
    var enableAccessibilityEnhancements: Bool = true
    var respectSystemSettings: Bool = true
    var customIntensityMultiplier: Float = 1.0
}

enum HapticError: Error, LocalizedError {
    case notSupported
    case engineNotStarted
    case patternCreationFailed
    case playbackFailed
    case permissionDenied
    
    var errorDescription: String? {
        switch self {
        case .notSupported:
            return "Haptic feedback is not supported on this device"
        case .engineNotStarted:
            return "Haptic engine failed to start"
        case .patternCreationFailed:
            return "Failed to create haptic pattern"
        case .playbackFailed:
            return "Haptic playback failed"
        case .permissionDenied:
            return "Haptic feedback permission denied"
        }
    }
}

// MARK: - Haptic Feedback Manager

@MainActor
class HapticFeedbackManager: ObservableObject {
    static let shared = HapticFeedbackManager()
    
    @Published var settings = HapticSettings()
    @Published var isSupported = false
    @Published var isEngineRunning = false
    @Published var error: HapticError?
    
    private var hapticEngine: CHHapticEngine?
    private var impactFeedbackLight = UIImpactFeedbackGenerator(style: .light)
    private var impactFeedbackMedium = UIImpactFeedbackGenerator(style: .medium)
    private var impactFeedbackHeavy = UIImpactFeedbackGenerator(style: .heavy)
    private var selectionFeedback = UISelectionFeedbackGenerator()
    private var notificationFeedback = UINotificationFeedbackGenerator()
    
    private var cancellables = Set<AnyCancellable>()
    private var activePatterns: [String: CHHapticPatternPlayer] = [:]
    
    // Predefined patterns
    private lazy var predefinedPatterns: [HapticFeedbackType: HapticPattern] = {
        return [
            .heartbeat: createHeartbeatPattern(),
            .breathingPattern: createBreathingPattern(),
            .progressComplete: createProgressCompletePattern(),
            .timerTick: createTimerTickPattern()
        ]
    }()
    
    init() {
        setupHapticEngine()
        checkSupport()
        prepareGenerators()
        observeSystemSettings()
    }
    
    // MARK: - Setup Methods
    
    private func setupHapticEngine() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else {
            isSupported = false
            error = .notSupported
            return
        }
        
        do {
            hapticEngine = try CHHapticEngine()
            hapticEngine?.stoppedHandler = { [weak self] reason in
                DispatchQueue.main.async {
                    self?.isEngineRunning = false
                    if reason == .systemError {
                        self?.error = .engineNotStarted
                    }
                }
            }
            
            hapticEngine?.resetHandler = { [weak self] in
                DispatchQueue.main.async {
                    self?.restartEngine()
                }
            }
            
            isSupported = true
            startEngine()
        } catch {
            self.error = .engineNotStarted
            isSupported = false
        }
    }
    
    private func startEngine() {
        guard let hapticEngine = hapticEngine else { return }
        
        do {
            try hapticEngine.start()
            isEngineRunning = true
            error = nil
        } catch {
            self.error = .engineNotStarted
            isEngineRunning = false
        }
    }
    
    private func restartEngine() {
        hapticEngine?.stop()
        startEngine()
    }
    
    private func checkSupport() {
        isSupported = CHHapticEngine.capabilitiesForHardware().supportsHaptics
    }
    
    private func prepareGenerators() {
        impactFeedbackLight.prepare()
        impactFeedbackMedium.prepare()
        impactFeedbackHeavy.prepare()
        selectionFeedback.prepare()
        notificationFeedback.prepare()
    }
    
    private func observeSystemSettings() {
        // Observe system haptic settings changes
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.checkSupport()
                if self?.settings.respectSystemSettings == true {
                    self?.updateSettingsFromSystem()
                }
            }
            .store(in: &cancellables)
    }
    
    private func updateSettingsFromSystem() {
        // Update settings based on system preferences
        // This would check system accessibility settings
    }
    
    // MARK: - Public Methods
    
    func playFeedback(_ type: HapticFeedbackType) {
        guard settings.isEnabled && isSupported else { return }
        
        switch type {
        case .success:
            playNotificationFeedback(.success)
            
        case .warning:
            playNotificationFeedback(.warning)
            
        case .error:
            playNotificationFeedback(.error)
            
        case .selection:
            playSelectionFeedback()
            
        case .impact(let intensity):
            playImpactFeedback(intensity)
            
        case .symptomLogged:
            if settings.enableHealthFeedback {
                playCustomPattern(createSymptomLoggedPattern())
            }
            
        case .medicationTaken:
            if settings.enableHealthFeedback {
                playCustomPattern(createMedicationTakenPattern())
            }
            
        case .medicationMissed:
            if settings.enableHealthFeedback {
                playCustomPattern(createMedicationMissedPattern())
            }
            
        case .appointmentReminder:
            if settings.enableHealthFeedback {
                playCustomPattern(createAppointmentReminderPattern())
            }
            
        case .emergencyAlert:
            playCustomPattern(createEmergencyAlertPattern())
            
        case .painLevelChange(let level):
            if settings.enableHealthFeedback {
                playCustomPattern(createPainLevelPattern(level: level))
            }
            
        case .moodChange(let level):
            if settings.enableHealthFeedback {
                playCustomPattern(createMoodChangePattern(level: level))
            }
            
        case .pageTransition, .tabSwitch, .modalPresent, .modalDismiss:
            if settings.enableNavigationFeedback {
                playImpactFeedback(.light)
            }
            
        case .buttonPress, .longPress, .swipeAction:
            playImpactFeedback(.light)
            
        case .pullToRefresh, .dataSync:
            playImpactFeedback(.medium)
            
        case .voiceCommandStart, .voiceCommandEnd, .voiceCommandRecognized, .voiceCommandError:
            if settings.enableVoiceFeedback {
                playVoiceFeedback(type)
            }
            
        case .heartbeat, .breathingPattern, .progressComplete, .timerTick:
            if let pattern = predefinedPatterns[type] {
                playCustomPattern(pattern)
            }
            
        case .custom(let pattern):
            if settings.enableCustomPatterns {
                playCustomPattern(pattern)
            }
        }
    }
    
    func playSequence(_ types: [HapticFeedbackType], interval: TimeInterval = 0.1) {
        guard settings.isEnabled && isSupported else { return }
        
        for (index, type) in types.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * interval) {
                self.playFeedback(type)
            }
        }
    }
    
    func playRepeatingPattern(_ type: HapticFeedbackType, count: Int, interval: TimeInterval = 1.0) {
        guard settings.isEnabled && isSupported else { return }
        
        for i in 0..<count {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * interval) {
                self.playFeedback(type)
            }
        }
    }
    
    func stopAllFeedback() {
        hapticEngine?.stop()
        activePatterns.removeAll()
    }
    
    func updateSettings(_ newSettings: HapticSettings) {
        settings = newSettings
        
        if !settings.isEnabled {
            stopAllFeedback()
        }
        
        if settings.respectSystemSettings {
            updateSettingsFromSystem()
        }
    }
    
    // MARK: - Private Feedback Methods
    
    private func playNotificationFeedback(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        guard settings.enableSystemHaptics else { return }
        notificationFeedback.notificationOccurred(type)
    }
    
    private func playSelectionFeedback() {
        guard settings.enableSystemHaptics else { return }
        selectionFeedback.selectionChanged()
    }
    
    private func playImpactFeedback(_ intensity: HapticIntensity) {
        guard settings.enableSystemHaptics else { return }
        
        let adjustedIntensity = intensity.rawValue * settings.customIntensityMultiplier
        
        switch intensity {
        case .light:
            impactFeedbackLight.impactOccurred(intensity: adjustedIntensity)
        case .medium:
            impactFeedbackMedium.impactOccurred(intensity: adjustedIntensity)
        case .heavy:
            impactFeedbackHeavy.impactOccurred(intensity: adjustedIntensity)
        }
    }
    
    private func playCustomPattern(_ pattern: HapticPattern) {
        guard settings.enableCustomPatterns && isEngineRunning else { return }
        
        do {
            let hapticPattern = try createCHHapticPattern(from: pattern)
            let player = try hapticEngine?.makePlayer(with: hapticPattern)
            try player?.start(atTime: 0)
        } catch {
            self.error = .playbackFailed
        }
    }
    
    private func playVoiceFeedback(_ type: HapticFeedbackType) {
        switch type {
        case .voiceCommandStart:
            playImpactFeedback(.light)
        case .voiceCommandEnd:
            playImpactFeedback(.medium)
        case .voiceCommandRecognized:
            playNotificationFeedback(.success)
        case .voiceCommandError:
            playNotificationFeedback(.error)
        default:
            break
        }
    }
    
    // MARK: - Pattern Creation Methods
    
    private func createCHHapticPattern(from pattern: HapticPattern) throws -> CHHapticPattern {
        var events: [CHHapticEvent] = []
        
        for hapticEvent in pattern.events {
            let intensity = CHHapticEventParameter(parameterID: .hapticIntensity, value: hapticEvent.intensity * settings.customIntensityMultiplier)
            let sharpness = CHHapticEventParameter(parameterID: .hapticSharpness, value: hapticEvent.sharpness)
            
            let event = CHHapticEvent(
                eventType: .hapticTransient,
                parameters: [intensity, sharpness],
                relativeTime: hapticEvent.time,
                duration: hapticEvent.duration
            )
            
            events.append(event)
        }
        
        return try CHHapticPattern(events: events, parameters: [])
    }
    
    private func createHeartbeatPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.8, sharpness: 0.5, duration: 0.1),
            HapticEvent(time: 0.15, intensity: 1.0, sharpness: 0.8, duration: 0.1),
            HapticEvent(time: 0.8, intensity: 0.8, sharpness: 0.5, duration: 0.1),
            HapticEvent(time: 0.95, intensity: 1.0, sharpness: 0.8, duration: 0.1)
        ]
        return HapticPattern(events: events)
    }
    
    private func createBreathingPattern() -> HapticPattern {
        var events: [HapticEvent] = []
        let breathDuration: TimeInterval = 4.0
        let steps = 20
        
        for i in 0..<steps {
            let time = Double(i) * (breathDuration / Double(steps))
            let progress = Double(i) / Double(steps - 1)
            let intensity = Float(0.3 + 0.4 * sin(progress * .pi))
            
            events.append(HapticEvent(time: time, intensity: intensity, sharpness: 0.3, duration: 0.1))
        }
        
        return HapticPattern(events: events)
    }
    
    private func createProgressCompletePattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.5, sharpness: 0.5, duration: 0.1),
            HapticEvent(time: 0.1, intensity: 0.7, sharpness: 0.7, duration: 0.1),
            HapticEvent(time: 0.2, intensity: 1.0, sharpness: 1.0, duration: 0.2)
        ]
        return HapticPattern(events: events)
    }
    
    private func createTimerTickPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.6, sharpness: 1.0, duration: 0.05)
        ]
        return HapticPattern(events: events)
    }
    
    private func createSymptomLoggedPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.7, sharpness: 0.5, duration: 0.1),
            HapticEvent(time: 0.15, intensity: 0.5, sharpness: 0.3, duration: 0.1)
        ]
        return HapticPattern(events: events)
    }
    
    private func createMedicationTakenPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.8, sharpness: 0.8, duration: 0.1),
            HapticEvent(time: 0.2, intensity: 0.6, sharpness: 0.6, duration: 0.1),
            HapticEvent(time: 0.4, intensity: 0.4, sharpness: 0.4, duration: 0.1)
        ]
        return HapticPattern(events: events)
    }
    
    private func createMedicationMissedPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 1.0, sharpness: 1.0, duration: 0.1),
            HapticEvent(time: 0.3, intensity: 1.0, sharpness: 1.0, duration: 0.1),
            HapticEvent(time: 0.6, intensity: 1.0, sharpness: 1.0, duration: 0.1)
        ]
        return HapticPattern(events: events)
    }
    
    private func createAppointmentReminderPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.6, sharpness: 0.5, duration: 0.2),
            HapticEvent(time: 0.5, intensity: 0.8, sharpness: 0.7, duration: 0.2),
            HapticEvent(time: 1.0, intensity: 0.6, sharpness: 0.5, duration: 0.2)
        ]
        return HapticPattern(events: events)
    }
    
    private func createEmergencyAlertPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 1.0, sharpness: 1.0, duration: 0.2),
            HapticEvent(time: 0.3, intensity: 1.0, sharpness: 1.0, duration: 0.2),
            HapticEvent(time: 0.6, intensity: 1.0, sharpness: 1.0, duration: 0.2),
            HapticEvent(time: 0.9, intensity: 1.0, sharpness: 1.0, duration: 0.2)
        ]
        return HapticPattern(events: events)
    }
    
    private func createPainLevelPattern(level: Int) -> HapticPattern {
        let intensity = Float(min(max(level, 1), 10)) / 10.0
        let events = [
            HapticEvent(time: 0.0, intensity: intensity, sharpness: intensity, duration: 0.1 + Double(intensity) * 0.1)
        ]
        return HapticPattern(events: events)
    }
    
    private func createMoodChangePattern(level: Int) -> HapticPattern {
        let normalizedLevel = Float(min(max(level, 1), 10)) / 10.0
        let intensity = normalizedLevel > 0.5 ? normalizedLevel : 1.0 - normalizedLevel
        let sharpness = normalizedLevel > 0.5 ? 0.3 : 0.8
        
        let events = [
            HapticEvent(time: 0.0, intensity: intensity, sharpness: sharpness, duration: 0.15)
        ]
        return HapticPattern(events: events)
    }
}

// MARK: - Haptic Feedback Extensions

extension View {
    func hapticFeedback(_ type: HapticFeedbackType, on trigger: some Equatable) -> some View {
        self.onChange(of: trigger) { _ in
            HapticFeedbackManager.shared.playFeedback(type)
        }
    }
    
    func hapticFeedback(_ type: HapticFeedbackType) -> some View {
        self.onTapGesture {
            HapticFeedbackManager.shared.playFeedback(type)
        }
    }
}

// MARK: - Accessibility Haptic Helpers

struct AccessibilityHapticHelper {
    static func playNavigationFeedback() {
        HapticFeedbackManager.shared.playFeedback(.pageTransition)
    }
    
    static func playSelectionFeedback() {
        HapticFeedbackManager.shared.playFeedback(.selection)
    }
    
    static func playErrorFeedback() {
        HapticFeedbackManager.shared.playFeedback(.error)
    }
    
    static func playSuccessFeedback() {
        HapticFeedbackManager.shared.playFeedback(.success)
    }
    
    static func playHealthActionFeedback(_ action: HealthAction) {
        switch action {
        case .symptomLogged:
            HapticFeedbackManager.shared.playFeedback(.symptomLogged)
        case .medicationTaken:
            HapticFeedbackManager.shared.playFeedback(.medicationTaken)
        case .appointmentScheduled:
            HapticFeedbackManager.shared.playFeedback(.appointmentReminder)
        }
    }
}

enum HealthAction {
    case symptomLogged
    case medicationTaken
    case appointmentScheduled
}