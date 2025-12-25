//
//  HapticFeedbackSystem.swift
//  InflamAI-Swift
//
//  Advanced haptic feedback system for enhanced accessibility and user experience
//

import Foundation
import UIKit
import CoreHaptics
import Combine
import SwiftUI

// MARK: - Haptic Feedback Types

enum HapticFeedbackType {
    // Health Events
    case painLevelChange(from: Int, to: Int)
    case medicationReminder
    case medicationTaken
    case flareUpWarning
    case improvementDetected
    case assessmentComplete
    
    // UI Interactions
    case buttonPress
    case toggleSwitch
    case sliderChange
    case navigationTransition
    case errorOccurred
    case successAction
    
    // Voice Commands
    case voiceCommandStart
    case voiceCommandEnd
    case voiceCommandRecognized
    case voiceCommandFailed
    
    // Accessibility
    case focusChange
    case importantAlert
    case guidanceHaptic
    case confirmationRequest
    
    // Custom Patterns
    case breathingGuide(phase: BreathingPhase)
    case heartbeat
    case progressUpdate(progress: Float)
    case timeReminder
}

enum BreathingPhase {
    case inhale, hold, exhale, pause
}

enum HapticIntensity: Float {
    case light = 0.3
    case medium = 0.6
    case strong = 1.0
}

enum HapticDuration: TimeInterval {
    case short = 0.1
    case medium = 0.3
    case long = 0.6
    case extended = 1.0
}

// MARK: - Haptic Pattern Definitions

struct HapticPattern {
    let events: [HapticEvent]
    let duration: TimeInterval
    let repeats: Int
    let intensity: HapticIntensity
}

struct HapticEvent {
    let type: CHHapticEvent.EventType
    let intensity: Float
    let sharpness: Float
    let relativeTime: TimeInterval
    let duration: TimeInterval
}

// MARK: - Haptic Feedback System

class HapticFeedbackSystem: ObservableObject {
    // Core Haptics Engine
    private var hapticEngine: CHHapticEngine?
    private var supportsHaptics: Bool = false
    
    // Legacy Haptic Generators
    private let impactGenerator = UIImpactFeedbackGenerator()
    private let selectionGenerator = UISelectionFeedbackGenerator()
    private let notificationGenerator = UINotificationFeedbackGenerator()
    
    // Settings
    @Published var hapticEnabled = true
    @Published var hapticIntensity: HapticIntensity = .medium
    @Published var accessibilityHapticsEnabled = true
    @Published var healthEventHapticsEnabled = true
    @Published var voiceCommandHapticsEnabled = true
    
    // State
    @Published var isPlaying = false
    private var currentPattern: CHHapticPattern?
    private var currentPlayer: CHHapticPatternPlayer?
    
    // Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    // Breathing Guide State
    private var breathingTimer: Timer?
    private var breathingPhase: BreathingPhase = .inhale
    
    init() {
        setupHapticEngine()
        loadUserPreferences()
        setupNotifications()
    }
    
    // MARK: - Setup
    
    private func setupHapticEngine() {
        // Check if device supports haptics
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else {
            print("Device doesn't support haptics")
            return
        }
        
        supportsHaptics = true
        
        do {
            hapticEngine = try CHHapticEngine()
            try hapticEngine?.start()
            
            // Handle engine reset
            hapticEngine?.resetHandler = { [weak self] in
                print("Haptic engine reset")
                do {
                    try self?.hapticEngine?.start()
                } catch {
                    print("Failed to restart haptic engine: \(error)")
                }
            }
            
            // Handle engine stopped
            hapticEngine?.stoppedHandler = { reason in
                print("Haptic engine stopped: \(reason)")
            }
            
        } catch {
            print("Failed to create haptic engine: \(error)")
            supportsHaptics = false
        }
    }
    
    private func loadUserPreferences() {
        hapticEnabled = UserDefaults.standard.bool(forKey: "hapticEnabled")
        if hapticEnabled == false && UserDefaults.standard.object(forKey: "hapticEnabled") == nil {
            hapticEnabled = true // Default to enabled
        }
        
        let intensityRaw = UserDefaults.standard.float(forKey: "hapticIntensity")
        hapticIntensity = HapticIntensity(rawValue: intensityRaw) ?? .medium
        
        accessibilityHapticsEnabled = UserDefaults.standard.bool(forKey: "accessibilityHapticsEnabled")
        if accessibilityHapticsEnabled == false && UserDefaults.standard.object(forKey: "accessibilityHapticsEnabled") == nil {
            accessibilityHapticsEnabled = true
        }
        
        healthEventHapticsEnabled = UserDefaults.standard.bool(forKey: "healthEventHapticsEnabled")
        if healthEventHapticsEnabled == false && UserDefaults.standard.object(forKey: "healthEventHapticsEnabled") == nil {
            healthEventHapticsEnabled = true
        }
        
        voiceCommandHapticsEnabled = UserDefaults.standard.bool(forKey: "voiceCommandHapticsEnabled")
        if voiceCommandHapticsEnabled == false && UserDefaults.standard.object(forKey: "voiceCommandHapticsEnabled") == nil {
            voiceCommandHapticsEnabled = true
        }
    }
    
    private func setupNotifications() {
        // Listen for app state changes
        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { [weak self] _ in
                self?.restartHapticEngine()
            }
            .store(in: &cancellables)
    }
    
    private func restartHapticEngine() {
        guard supportsHaptics else { return }
        
        do {
            try hapticEngine?.start()
        } catch {
            print("Failed to restart haptic engine: \(error)")
        }
    }
    
    // MARK: - Public Interface
    
    func playHaptic(_ type: HapticFeedbackType) {
        guard hapticEnabled else { return }
        
        // Check category-specific settings
        switch type {
        case .painLevelChange, .medicationReminder, .medicationTaken, .flareUpWarning, .improvementDetected, .assessmentComplete:
            guard healthEventHapticsEnabled else { return }
        case .voiceCommandStart, .voiceCommandEnd, .voiceCommandRecognized, .voiceCommandFailed:
            guard voiceCommandHapticsEnabled else { return }
        case .focusChange, .importantAlert, .guidanceHaptic, .confirmationRequest:
            guard accessibilityHapticsEnabled else { return }
        default:
            break
        }
        
        if supportsHaptics {
            playAdvancedHaptic(type)
        } else {
            playLegacyHaptic(type)
        }
    }
    
    func startBreathingGuide(inhaleTime: TimeInterval = 4.0, holdTime: TimeInterval = 4.0, exhaleTime: TimeInterval = 6.0, pauseTime: TimeInterval = 2.0) {
        guard hapticEnabled && accessibilityHapticsEnabled else { return }
        
        stopBreathingGuide()
        
        let totalCycleTime = inhaleTime + holdTime + exhaleTime + pauseTime
        
        breathingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            
            let elapsed = timer.timeInterval * Double(timer.fireCount)
            let cyclePosition = elapsed.truncatingRemainder(dividingBy: totalCycleTime)
            
            var newPhase: BreathingPhase
            
            if cyclePosition < inhaleTime {
                newPhase = .inhale
            } else if cyclePosition < inhaleTime + holdTime {
                newPhase = .hold
            } else if cyclePosition < inhaleTime + holdTime + exhaleTime {
                newPhase = .exhale
            } else {
                newPhase = .pause
            }
            
            if newPhase != self.breathingPhase {
                self.breathingPhase = newPhase
                self.playHaptic(.breathingGuide(phase: newPhase))
            }
        }
    }
    
    func stopBreathingGuide() {
        breathingTimer?.invalidate()
        breathingTimer = nil
    }
    
    func playProgressHaptic(progress: Float) {
        guard hapticEnabled else { return }
        playHaptic(.progressUpdate(progress: progress))
    }
    
    func playCustomPattern(_ pattern: HapticPattern) {
        guard hapticEnabled && supportsHaptics else { return }
        
        do {
            let hapticPattern = try createCHHapticPattern(from: pattern)
            let player = try hapticEngine?.makePlayer(with: hapticPattern)
            try player?.start(atTime: 0)
        } catch {
            print("Failed to play custom haptic pattern: \(error)")
        }
    }
    
    // MARK: - Advanced Haptic Implementation
    
    private func playAdvancedHaptic(_ type: HapticFeedbackType) {
        do {
            let pattern = createHapticPattern(for: type)
            let hapticPattern = try createCHHapticPattern(from: pattern)
            
            currentPlayer?.stop(atTime: 0)
            currentPlayer = try hapticEngine?.makePlayer(with: hapticPattern)
            
            try currentPlayer?.start(atTime: 0)
            isPlaying = true
            
            // Auto-stop after pattern duration
            DispatchQueue.main.asyncAfter(deadline: .now() + pattern.duration) {
                self.isPlaying = false
            }
            
        } catch {
            print("Failed to play advanced haptic: \(error)")
            // Fallback to legacy haptics
            playLegacyHaptic(type)
        }
    }
    
    private func createHapticPattern(for type: HapticFeedbackType) -> HapticPattern {
        let baseIntensity = hapticIntensity.rawValue
        
        switch type {
        case .painLevelChange(let from, let to):
            return createPainLevelChangePattern(from: from, to: to, baseIntensity: baseIntensity)
            
        case .medicationReminder:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.8, sharpness: 0.5, relativeTime: 0, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.6, sharpness: 0.5, relativeTime: 0.3, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 0.7, relativeTime: 0.6, duration: 0.2)
                ],
                duration: 0.8,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .medicationTaken:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 0.8, relativeTime: 0, duration: 0.1),
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.4, sharpness: 0.3, relativeTime: 0.1, duration: 0.3)
                ],
                duration: 0.4,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .flareUpWarning:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 1.0, relativeTime: 0, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.8, sharpness: 0.8, relativeTime: 0.2, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 1.0, relativeTime: 0.4, duration: 0.1)
                ],
                duration: 0.5,
                repeats: 2,
                intensity: hapticIntensity
            )
            
        case .improvementDetected:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.3, sharpness: 0.2, relativeTime: 0, duration: 0.2),
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.5, sharpness: 0.4, relativeTime: 0.2, duration: 0.2),
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.7, sharpness: 0.6, relativeTime: 0.4, duration: 0.2)
                ],
                duration: 0.6,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .breathingGuide(let phase):
            return createBreathingPattern(for: phase, baseIntensity: baseIntensity)
            
        case .heartbeat:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.6, sharpness: 0.8, relativeTime: 0, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.4, sharpness: 0.6, relativeTime: 0.15, duration: 0.05)
                ],
                duration: 0.2,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .progressUpdate(let progress):
            return createProgressPattern(progress: progress, baseIntensity: baseIntensity)
            
        case .voiceCommandStart:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.7, sharpness: 0.5, relativeTime: 0, duration: 0.1)
                ],
                duration: 0.1,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .voiceCommandEnd:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.5, sharpness: 0.3, relativeTime: 0, duration: 0.15)
                ],
                duration: 0.15,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .importantAlert:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 1.0, relativeTime: 0, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 1.0, relativeTime: 0.3, duration: 0.1),
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 1.0, relativeTime: 0.6, duration: 0.1)
                ],
                duration: 0.7,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        default:
            return createDefaultPattern(baseIntensity: baseIntensity)
        }
    }
    
    private func createPainLevelChangePattern(from: Int, to: Int, baseIntensity: Float) -> HapticPattern {
        let difference = abs(to - from)
        let increasing = to > from
        
        var events: [HapticEvent] = []
        
        if increasing {
            // Pain increasing - more intense, sharper haptics
            for i in 0..<difference {
                let intensity = baseIntensity * (0.5 + Float(i) * 0.1)
                let sharpness = 0.7 + Float(i) * 0.1
                events.append(HapticEvent(
                    type: .hapticTransient,
                    intensity: min(intensity, 1.0),
                    sharpness: min(sharpness, 1.0),
                    relativeTime: TimeInterval(i) * 0.2,
                    duration: 0.1
                ))
            }
        } else {
            // Pain decreasing - gentler, softer haptics
            for i in 0..<difference {
                let intensity = baseIntensity * (0.8 - Float(i) * 0.1)
                let sharpness = 0.5 - Float(i) * 0.05
                events.append(HapticEvent(
                    type: .hapticContinuous,
                    intensity: max(intensity, 0.2),
                    sharpness: max(sharpness, 0.1),
                    relativeTime: TimeInterval(i) * 0.3,
                    duration: 0.2
                ))
            }
        }
        
        return HapticPattern(
            events: events,
            duration: TimeInterval(difference) * (increasing ? 0.2 : 0.3) + 0.2,
            repeats: 1,
            intensity: hapticIntensity
        )
    }
    
    private func createBreathingPattern(for phase: BreathingPhase, baseIntensity: Float) -> HapticPattern {
        switch phase {
        case .inhale:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.3, sharpness: 0.2, relativeTime: 0, duration: 0.5)
                ],
                duration: 0.5,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .hold:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.2, sharpness: 0.1, relativeTime: 0, duration: 0.3)
                ],
                duration: 0.3,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .exhale:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticContinuous, intensity: baseIntensity * 0.4, sharpness: 0.3, relativeTime: 0, duration: 0.6)
                ],
                duration: 0.6,
                repeats: 1,
                intensity: hapticIntensity
            )
            
        case .pause:
            return HapticPattern(
                events: [
                    HapticEvent(type: .hapticTransient, intensity: baseIntensity * 0.1, sharpness: 0.1, relativeTime: 0, duration: 0.1)
                ],
                duration: 0.1,
                repeats: 1,
                intensity: hapticIntensity
            )
        }
    }
    
    private func createProgressPattern(progress: Float, baseIntensity: Float) -> HapticPattern {
        let intensity = baseIntensity * progress
        let sharpness = 0.3 + (progress * 0.4)
        
        return HapticPattern(
            events: [
                HapticEvent(type: .hapticTransient, intensity: intensity, sharpness: sharpness, relativeTime: 0, duration: 0.1)
            ],
            duration: 0.1,
            repeats: 1,
            intensity: hapticIntensity
        )
    }
    
    private func createDefaultPattern(baseIntensity: Float) -> HapticPattern {
        return HapticPattern(
            events: [
                HapticEvent(type: .hapticTransient, intensity: baseIntensity, sharpness: 0.5, relativeTime: 0, duration: 0.1)
            ],
            duration: 0.1,
            repeats: 1,
            intensity: hapticIntensity
        )
    }
    
    private func createCHHapticPattern(from pattern: HapticPattern) throws -> CHHapticPattern {
        var hapticEvents: [CHHapticEvent] = []
        
        for event in pattern.events {
            let hapticEvent = CHHapticEvent(
                eventType: event.type,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: event.intensity * pattern.intensity.rawValue),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: event.sharpness)
                ],
                relativeTime: event.relativeTime,
                duration: event.duration
            )
            hapticEvents.append(hapticEvent)
        }
        
        return try CHHapticPattern(events: hapticEvents, parameters: [])
    }
    
    // MARK: - Legacy Haptic Implementation
    
    private func playLegacyHaptic(_ type: HapticFeedbackType) {
        switch type {
        case .painLevelChange(let from, let to):
            if to > from {
                notificationGenerator.notificationOccurred(.warning)
            } else {
                notificationGenerator.notificationOccurred(.success)
            }
            
        case .medicationReminder:
            notificationGenerator.notificationOccurred(.warning)
            
        case .medicationTaken:
            notificationGenerator.notificationOccurred(.success)
            
        case .flareUpWarning:
            notificationGenerator.notificationOccurred(.error)
            
        case .improvementDetected:
            notificationGenerator.notificationOccurred(.success)
            
        case .buttonPress:
            impactGenerator.impactOccurred(intensity: hapticIntensity.rawValue)
            
        case .toggleSwitch, .sliderChange:
            selectionGenerator.selectionChanged()
            
        case .errorOccurred:
            notificationGenerator.notificationOccurred(.error)
            
        case .successAction:
            notificationGenerator.notificationOccurred(.success)
            
        case .voiceCommandStart, .voiceCommandEnd:
            selectionGenerator.selectionChanged()
            
        case .importantAlert:
            notificationGenerator.notificationOccurred(.warning)
            
        default:
            impactGenerator.impactOccurred(intensity: hapticIntensity.rawValue)
        }
    }
    
    // MARK: - Settings Management
    
    func updateSettings(
        enabled: Bool,
        intensity: HapticIntensity,
        accessibilityEnabled: Bool,
        healthEventsEnabled: Bool,
        voiceCommandsEnabled: Bool
    ) {
        hapticEnabled = enabled
        hapticIntensity = intensity
        accessibilityHapticsEnabled = accessibilityEnabled
        healthEventHapticsEnabled = healthEventsEnabled
        voiceCommandHapticsEnabled = voiceCommandsEnabled
        
        // Save to UserDefaults
        UserDefaults.standard.set(enabled, forKey: "hapticEnabled")
        UserDefaults.standard.set(intensity.rawValue, forKey: "hapticIntensity")
        UserDefaults.standard.set(accessibilityEnabled, forKey: "accessibilityHapticsEnabled")
        UserDefaults.standard.set(healthEventsEnabled, forKey: "healthEventHapticsEnabled")
        UserDefaults.standard.set(voiceCommandsEnabled, forKey: "voiceCommandHapticsEnabled")
    }
    
    func testHaptic(_ type: HapticFeedbackType) {
        playHaptic(type)
    }
    
    // MARK: - Accessibility Support
    
    func playNavigationHaptic() {
        guard accessibilityHapticsEnabled else { return }
        playHaptic(.navigationTransition)
    }
    
    func playFocusChangeHaptic() {
        guard accessibilityHapticsEnabled else { return }
        playHaptic(.focusChange)
    }
    
    func playGuidanceHaptic() {
        guard accessibilityHapticsEnabled else { return }
        playHaptic(.guidanceHaptic)
    }
    
    func playConfirmationRequestHaptic() {
        guard accessibilityHapticsEnabled else { return }
        playHaptic(.confirmationRequest)
    }
}

// MARK: - SwiftUI Integration

struct HapticModifier: ViewModifier {
    let hapticSystem: HapticFeedbackSystem
    let feedbackType: HapticFeedbackType
    let trigger: Bool
    
    func body(content: Content) -> some View {
        content
            .onChange(of: trigger) { _ in
                hapticSystem.playHaptic(feedbackType)
            }
    }
}

extension View {
    func hapticFeedback(
        _ hapticSystem: HapticFeedbackSystem,
        type: HapticFeedbackType,
        trigger: Bool
    ) -> some View {
        modifier(HapticModifier(hapticSystem: hapticSystem, feedbackType: type, trigger: trigger))
    }
    
    func onTapHaptic(
        _ hapticSystem: HapticFeedbackSystem,
        type: HapticFeedbackType = .buttonPress
    ) -> some View {
        onTapGesture {
            hapticSystem.playHaptic(type)
        }
    }
}

// MARK: - Haptic Settings View

struct HapticSettingsView: View {
    @ObservedObject var hapticSystem: HapticFeedbackSystem
    
    var body: some View {
        Form {
            Section("General Settings") {
                Toggle("Enable Haptic Feedback", isOn: $hapticSystem.hapticEnabled)
                
                if hapticSystem.hapticEnabled {
                    VStack(alignment: .leading) {
                        Text("Intensity")
                        Picker("Intensity", selection: $hapticSystem.hapticIntensity) {
                            Text("Light").tag(HapticIntensity.light)
                            Text("Medium").tag(HapticIntensity.medium)
                            Text("Strong").tag(HapticIntensity.strong)
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                }
            }
            
            if hapticSystem.hapticEnabled {
                Section("Category Settings") {
                    Toggle("Health Events", isOn: $hapticSystem.healthEventHapticsEnabled)
                    Toggle("Voice Commands", isOn: $hapticSystem.voiceCommandHapticsEnabled)
                    Toggle("Accessibility", isOn: $hapticSystem.accessibilityHapticsEnabled)
                }
                
                Section("Test Haptics") {
                    Button("Test Pain Level Change") {
                        hapticSystem.testHaptic(.painLevelChange(from: 3, to: 6))
                    }
                    
                    Button("Test Medication Reminder") {
                        hapticSystem.testHaptic(.medicationReminder)
                    }
                    
                    Button("Test Flare-up Warning") {
                        hapticSystem.testHaptic(.flareUpWarning)
                    }
                    
                    Button("Test Improvement") {
                        hapticSystem.testHaptic(.improvementDetected)
                    }
                    
                    Button("Test Breathing Guide") {
                        hapticSystem.startBreathingGuide()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                            hapticSystem.stopBreathingGuide()
                        }
                    }
                }
            }
        }
        .navigationTitle("Haptic Settings")
        .onChange(of: hapticSystem.hapticEnabled) { _ in
            saveSettings()
        }
        .onChange(of: hapticSystem.hapticIntensity) { _ in
            saveSettings()
        }
        .onChange(of: hapticSystem.healthEventHapticsEnabled) { _ in
            saveSettings()
        }
        .onChange(of: hapticSystem.voiceCommandHapticsEnabled) { _ in
            saveSettings()
        }
        .onChange(of: hapticSystem.accessibilityHapticsEnabled) { _ in
            saveSettings()
        }
    }
    
    private func saveSettings() {
        hapticSystem.updateSettings(
            enabled: hapticSystem.hapticEnabled,
            intensity: hapticSystem.hapticIntensity,
            accessibilityEnabled: hapticSystem.accessibilityHapticsEnabled,
            healthEventsEnabled: hapticSystem.healthEventHapticsEnabled,
            voiceCommandsEnabled: hapticSystem.voiceCommandHapticsEnabled
        )
    }
}