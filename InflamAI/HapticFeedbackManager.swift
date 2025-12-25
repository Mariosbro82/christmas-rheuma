//
//  HapticFeedbackManager.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import UIKit
import CoreHaptics
import Combine
import SwiftUI

// MARK: - Haptic Feedback Models

enum HapticFeedbackType {
    // Basic Feedback
    case success
    case warning
    case error
    case selection
    case impact(UIImpactFeedbackGenerator.FeedbackStyle)
    
    // Custom Patterns
    case painAlert
    case medicationReminder
    case exerciseStart
    case exerciseComplete
    case flareUpWarning
    case emergencyAlert
    case dataSync
    case voiceCommandReceived
    case voiceCommandProcessed
    case navigationFeedback
    case buttonPress
    case swipeGesture
    case longPress
    case pullToRefresh
    case pageTransition
    case modalPresentation
    case modalDismissal
    case tabSwitch
    case settingsChange
    case achievementUnlocked
    case progressMilestone
    case reminderGentle
    case reminderUrgent
    case heartRateAlert
    case sleepReminder
    case breathingExercise
    case meditationStart
    case meditationEnd
    case socialNotification
    case messageReceived
    case callIncoming
    case timerStart
    case timerEnd
    case alarmGentle
    case alarmUrgent
    
    var displayName: String {
        switch self {
        case .success: return "Success"
        case .warning: return "Warning"
        case .error: return "Error"
        case .selection: return "Selection"
        case .impact(let style):
            switch style {
            case .light: return "Light Impact"
            case .medium: return "Medium Impact"
            case .heavy: return "Heavy Impact"
            case .soft: return "Soft Impact"
            case .rigid: return "Rigid Impact"
            @unknown default: return "Impact"
            }
        case .painAlert: return "Pain Alert"
        case .medicationReminder: return "Medication Reminder"
        case .exerciseStart: return "Exercise Start"
        case .exerciseComplete: return "Exercise Complete"
        case .flareUpWarning: return "Flare-up Warning"
        case .emergencyAlert: return "Emergency Alert"
        case .dataSync: return "Data Sync"
        case .voiceCommandReceived: return "Voice Command Received"
        case .voiceCommandProcessed: return "Voice Command Processed"
        case .navigationFeedback: return "Navigation"
        case .buttonPress: return "Button Press"
        case .swipeGesture: return "Swipe Gesture"
        case .longPress: return "Long Press"
        case .pullToRefresh: return "Pull to Refresh"
        case .pageTransition: return "Page Transition"
        case .modalPresentation: return "Modal Presentation"
        case .modalDismissal: return "Modal Dismissal"
        case .tabSwitch: return "Tab Switch"
        case .settingsChange: return "Settings Change"
        case .achievementUnlocked: return "Achievement Unlocked"
        case .progressMilestone: return "Progress Milestone"
        case .reminderGentle: return "Gentle Reminder"
        case .reminderUrgent: return "Urgent Reminder"
        case .heartRateAlert: return "Heart Rate Alert"
        case .sleepReminder: return "Sleep Reminder"
        case .breathingExercise: return "Breathing Exercise"
        case .meditationStart: return "Meditation Start"
        case .meditationEnd: return "Meditation End"
        case .socialNotification: return "Social Notification"
        case .messageReceived: return "Message Received"
        case .callIncoming: return "Incoming Call"
        case .timerStart: return "Timer Start"
        case .timerEnd: return "Timer End"
        case .alarmGentle: return "Gentle Alarm"
        case .alarmUrgent: return "Urgent Alarm"
        }
    }
    
    var category: HapticCategory {
        switch self {
        case .success, .warning, .error, .selection, .impact:
            return .system
        case .painAlert, .medicationReminder, .flareUpWarning, .heartRateAlert:
            return .health
        case .exerciseStart, .exerciseComplete, .breathingExercise, .meditationStart, .meditationEnd:
            return .exercise
        case .emergencyAlert, .reminderUrgent, .alarmUrgent:
            return .emergency
        case .dataSync, .voiceCommandReceived, .voiceCommandProcessed:
            return .system
        case .navigationFeedback, .buttonPress, .swipeGesture, .longPress, .pullToRefresh, .pageTransition, .modalPresentation, .modalDismissal, .tabSwitch:
            return .interaction
        case .settingsChange, .achievementUnlocked, .progressMilestone:
            return .feedback
        case .reminderGentle, .sleepReminder:
            return .notification
        case .socialNotification, .messageReceived, .callIncoming:
            return .communication
        case .timerStart, .timerEnd, .alarmGentle:
            return .timer
        }
    }
}

enum HapticCategory: String, CaseIterable {
    case system = "system"
    case health = "health"
    case exercise = "exercise"
    case emergency = "emergency"
    case interaction = "interaction"
    case feedback = "feedback"
    case notification = "notification"
    case communication = "communication"
    case timer = "timer"
    
    var displayName: String {
        switch self {
        case .system: return "System"
        case .health: return "Health"
        case .exercise: return "Exercise"
        case .emergency: return "Emergency"
        case .interaction: return "Interaction"
        case .feedback: return "Feedback"
        case .notification: return "Notification"
        case .communication: return "Communication"
        case .timer: return "Timer"
        }
    }
}

struct HapticPattern {
    let events: [HapticEvent]
    let duration: TimeInterval
    let intensity: Float
    let sharpness: Float
    let repeatCount: Int
    let repeatDelay: TimeInterval
}

struct HapticEvent {
    let time: TimeInterval
    let intensity: Float
    let sharpness: Float
    let duration: TimeInterval
    let type: HapticEventType
}

enum HapticEventType {
    case transient
    case continuous
    case parameter
}

struct HapticSettings {
    var enabled: Bool = true
    var intensity: Float = 1.0
    var categorySettings: [HapticCategory: HapticCategorySettings] = [:]
    var accessibilityMode: Bool = false
    var reducedMotion: Bool = false
    var customPatterns: [String: HapticPattern] = [:]
    
    init() {
        // Initialize default category settings
        for category in HapticCategory.allCases {
            categorySettings[category] = HapticCategorySettings()
        }
    }
}

struct HapticCategorySettings {
    var enabled: Bool = true
    var intensity: Float = 1.0
    var customIntensity: Bool = false
}

// MARK: - Haptic Feedback Manager

@MainActor
class HapticFeedbackManager: ObservableObject {
    // MARK: - Published Properties
    @Published var settings: HapticSettings = HapticSettings()
    @Published var isHapticsSupported: Bool = false
    @Published var isAdvancedHapticsSupported: Bool = false
    @Published var currentPattern: HapticPattern?
    @Published var isPlaying: Bool = false
    
    // MARK: - Private Properties
    private var hapticEngine: CHHapticEngine?
    private var hapticPlayer: CHHapticPatternPlayer?
    
    // Basic Feedback Generators
    private let notificationGenerator = UINotificationFeedbackGenerator()
    private let selectionGenerator = UISelectionFeedbackGenerator()
    private let lightImpactGenerator = UIImpactFeedbackGenerator(style: .light)
    private let mediumImpactGenerator = UIImpactFeedbackGenerator(style: .medium)
    private let heavyImpactGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private let softImpactGenerator: UIImpactFeedbackGenerator?
    private let rigidImpactGenerator: UIImpactFeedbackGenerator?
    
    // Pattern Library
    private var patternLibrary: [HapticFeedbackType: HapticPattern] = [:]
    
    // Settings
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        // Initialize iOS 13+ impact generators if available
        if #available(iOS 13.0, *) {
            softImpactGenerator = UIImpactFeedbackGenerator(style: .soft)
            rigidImpactGenerator = UIImpactFeedbackGenerator(style: .rigid)
        } else {
            softImpactGenerator = nil
            rigidImpactGenerator = nil
        }
        
        setupHapticEngine()
        loadSettings()
        createPatternLibrary()
        observeAccessibilitySettings()
    }
    
    // MARK: - Setup
    
    private func setupHapticEngine() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else {
            isHapticsSupported = false
            isAdvancedHapticsSupported = false
            return
        }
        
        isHapticsSupported = true
        
        do {
            hapticEngine = try CHHapticEngine()
            isAdvancedHapticsSupported = true
            
            hapticEngine?.stoppedHandler = { [weak self] reason in
                Task { @MainActor in
                    self?.handleEngineStop(reason: reason)
                }
            }
            
            hapticEngine?.resetHandler = { [weak self] in
                Task { @MainActor in
                    self?.handleEngineReset()
                }
            }
            
            try hapticEngine?.start()
            
        } catch {
            print("Failed to create haptic engine: \(error)")
            isAdvancedHapticsSupported = false
        }
    }
    
    private func loadSettings() {
        if let data = UserDefaults.standard.data(forKey: "hapticSettings"),
           let loadedSettings = try? JSONDecoder().decode(HapticSettings.self, from: data) {
            settings = loadedSettings
        }
    }
    
    private func saveSettings() {
        do {
            let data = try JSONEncoder().encode(settings)
            UserDefaults.standard.set(data, forKey: "hapticSettings")
        } catch {
            print("Failed to save haptic settings: \(error)")
        }
    }
    
    private func observeAccessibilitySettings() {
        NotificationCenter.default.publisher(for: UIAccessibility.reduceMotionStatusDidChangeNotification)
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.updateAccessibilitySettings()
                }
            }
            .store(in: &cancellables)
    }
    
    private func updateAccessibilitySettings() {
        settings.reducedMotion = UIAccessibility.isReduceMotionEnabled
        
        // Adjust haptic intensity for accessibility
        if settings.reducedMotion {
            settings.intensity = min(settings.intensity, 0.7)
        }
    }
    
    // MARK: - Pattern Library
    
    private func createPatternLibrary() {
        // Basic patterns
        patternLibrary[.success] = createSuccessPattern()
        patternLibrary[.warning] = createWarningPattern()
        patternLibrary[.error] = createErrorPattern()
        
        // Health-related patterns
        patternLibrary[.painAlert] = createPainAlertPattern()
        patternLibrary[.medicationReminder] = createMedicationReminderPattern()
        patternLibrary[.flareUpWarning] = createFlareUpWarningPattern()
        patternLibrary[.heartRateAlert] = createHeartRateAlertPattern()
        
        // Exercise patterns
        patternLibrary[.exerciseStart] = createExerciseStartPattern()
        patternLibrary[.exerciseComplete] = createExerciseCompletePattern()
        patternLibrary[.breathingExercise] = createBreathingExercisePattern()
        
        // Meditation patterns
        patternLibrary[.meditationStart] = createMeditationStartPattern()
        patternLibrary[.meditationEnd] = createMeditationEndPattern()
        
        // Emergency patterns
        patternLibrary[.emergencyAlert] = createEmergencyAlertPattern()
        
        // Interaction patterns
        patternLibrary[.buttonPress] = createButtonPressPattern()
        patternLibrary[.swipeGesture] = createSwipeGesturePattern()
        patternLibrary[.longPress] = createLongPressPattern()
        
        // Notification patterns
        patternLibrary[.reminderGentle] = createGentleReminderPattern()
        patternLibrary[.reminderUrgent] = createUrgentReminderPattern()
        patternLibrary[.socialNotification] = createSocialNotificationPattern()
        
        // Achievement patterns
        patternLibrary[.achievementUnlocked] = createAchievementPattern()
        patternLibrary[.progressMilestone] = createProgressMilestonePattern()
        
        // Voice command patterns
        patternLibrary[.voiceCommandReceived] = createVoiceCommandReceivedPattern()
        patternLibrary[.voiceCommandProcessed] = createVoiceCommandProcessedPattern()
    }
    
    // MARK: - Pattern Creation Methods
    
    private func createSuccessPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.8, sharpness: 0.5, duration: 0.1, type: .transient),
            HapticEvent(time: 0.15, intensity: 1.0, sharpness: 0.8, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.25, intensity: 0.9, sharpness: 0.65, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createWarningPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.6, sharpness: 0.3, duration: 0.2, type: .continuous),
            HapticEvent(time: 0.3, intensity: 0.8, sharpness: 0.5, duration: 0.2, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 0.5, intensity: 0.7, sharpness: 0.4, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createErrorPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 1.0, sharpness: 1.0, duration: 0.1, type: .transient),
            HapticEvent(time: 0.2, intensity: 1.0, sharpness: 1.0, duration: 0.1, type: .transient),
            HapticEvent(time: 0.4, intensity: 1.0, sharpness: 1.0, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.5, intensity: 1.0, sharpness: 1.0, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createPainAlertPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.3, sharpness: 0.2, duration: 0.5, type: .continuous),
            HapticEvent(time: 0.6, intensity: 0.5, sharpness: 0.3, duration: 0.3, type: .continuous),
            HapticEvent(time: 1.0, intensity: 0.7, sharpness: 0.4, duration: 0.2, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 1.2, intensity: 0.5, sharpness: 0.3, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createMedicationReminderPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.6, sharpness: 0.4, duration: 0.1, type: .transient),
            HapticEvent(time: 0.5, intensity: 0.6, sharpness: 0.4, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.6, intensity: 0.6, sharpness: 0.4, repeatCount: 2, repeatDelay: 1.0)
    }
    
    private func createFlareUpWarningPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.4, sharpness: 0.2, duration: 0.8, type: .continuous),
            HapticEvent(time: 1.0, intensity: 0.6, sharpness: 0.4, duration: 0.4, type: .continuous),
            HapticEvent(time: 1.5, intensity: 0.8, sharpness: 0.6, duration: 0.2, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 1.7, intensity: 0.6, sharpness: 0.4, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createHeartRateAlertPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.8, sharpness: 0.6, duration: 0.1, type: .transient),
            HapticEvent(time: 0.8, intensity: 0.8, sharpness: 0.6, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.9, intensity: 0.8, sharpness: 0.6, repeatCount: 3, repeatDelay: 0.5)
    }
    
    private func createExerciseStartPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.5, sharpness: 0.3, duration: 0.2, type: .continuous),
            HapticEvent(time: 0.3, intensity: 0.7, sharpness: 0.5, duration: 0.2, type: .continuous),
            HapticEvent(time: 0.6, intensity: 0.9, sharpness: 0.7, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.7, intensity: 0.7, sharpness: 0.5, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createExerciseCompletePattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.8, sharpness: 0.6, duration: 0.1, type: .transient),
            HapticEvent(time: 0.2, intensity: 0.9, sharpness: 0.7, duration: 0.1, type: .transient),
            HapticEvent(time: 0.4, intensity: 1.0, sharpness: 0.8, duration: 0.15, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.55, intensity: 0.9, sharpness: 0.7, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createBreathingExercisePattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.3, sharpness: 0.1, duration: 2.0, type: .continuous),
            HapticEvent(time: 2.5, intensity: 0.3, sharpness: 0.1, duration: 2.0, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 4.5, intensity: 0.3, sharpness: 0.1, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createMeditationStartPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.2, sharpness: 0.1, duration: 1.0, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 1.0, intensity: 0.2, sharpness: 0.1, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createMeditationEndPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.4, sharpness: 0.2, duration: 0.5, type: .continuous),
            HapticEvent(time: 0.7, intensity: 0.2, sharpness: 0.1, duration: 0.8, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 1.5, intensity: 0.3, sharpness: 0.15, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createEmergencyAlertPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 1.0, sharpness: 1.0, duration: 0.2, type: .transient),
            HapticEvent(time: 0.3, intensity: 1.0, sharpness: 1.0, duration: 0.2, type: .transient),
            HapticEvent(time: 0.6, intensity: 1.0, sharpness: 1.0, duration: 0.2, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.8, intensity: 1.0, sharpness: 1.0, repeatCount: 5, repeatDelay: 0.5)
    }
    
    private func createButtonPressPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.6, sharpness: 0.5, duration: 0.05, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.05, intensity: 0.6, sharpness: 0.5, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createSwipeGesturePattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.4, sharpness: 0.3, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.1, intensity: 0.4, sharpness: 0.3, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createLongPressPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.5, sharpness: 0.4, duration: 0.3, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 0.3, intensity: 0.5, sharpness: 0.4, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createGentleReminderPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.4, sharpness: 0.2, duration: 0.2, type: .continuous)
        ]
        return HapticPattern(events: events, duration: 0.2, intensity: 0.4, sharpness: 0.2, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createUrgentReminderPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.8, sharpness: 0.6, duration: 0.1, type: .transient),
            HapticEvent(time: 0.3, intensity: 0.8, sharpness: 0.6, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.4, intensity: 0.8, sharpness: 0.6, repeatCount: 3, repeatDelay: 0.8)
    }
    
    private func createSocialNotificationPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.5, sharpness: 0.4, duration: 0.1, type: .transient),
            HapticEvent(time: 0.15, intensity: 0.3, sharpness: 0.2, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.25, intensity: 0.4, sharpness: 0.3, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createAchievementPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.6, sharpness: 0.4, duration: 0.1, type: .transient),
            HapticEvent(time: 0.15, intensity: 0.8, sharpness: 0.6, duration: 0.1, type: .transient),
            HapticEvent(time: 0.3, intensity: 1.0, sharpness: 0.8, duration: 0.2, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.5, intensity: 0.8, sharpness: 0.6, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createProgressMilestonePattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.7, sharpness: 0.5, duration: 0.15, type: .transient),
            HapticEvent(time: 0.25, intensity: 0.9, sharpness: 0.7, duration: 0.15, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.4, intensity: 0.8, sharpness: 0.6, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createVoiceCommandReceivedPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.3, sharpness: 0.2, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.1, intensity: 0.3, sharpness: 0.2, repeatCount: 1, repeatDelay: 0.0)
    }
    
    private func createVoiceCommandProcessedPattern() -> HapticPattern {
        let events = [
            HapticEvent(time: 0.0, intensity: 0.5, sharpness: 0.4, duration: 0.1, type: .transient),
            HapticEvent(time: 0.15, intensity: 0.3, sharpness: 0.2, duration: 0.1, type: .transient)
        ]
        return HapticPattern(events: events, duration: 0.25, intensity: 0.4, sharpness: 0.3, repeatCount: 1, repeatDelay: 0.0)
    }
    
    // MARK: - Engine Management
    
    private func handleEngineStop(reason: CHHapticEngine.StoppedReason) {
        print("Haptic engine stopped: \(reason)")
        isPlaying = false
        
        switch reason {
        case .audioSessionInterrupt, .applicationSuspended, .idleTimeout:
            // Try to restart the engine
            restartEngine()
        case .systemError, .notifyWhenFinished:
            // Handle error or completion
            break
        @unknown default:
            break
        }
    }
    
    private func handleEngineReset() {
        print("Haptic engine reset")
        isPlaying = false
        restartEngine()
    }
    
    private func restartEngine() {
        do {
            try hapticEngine?.start()
        } catch {
            print("Failed to restart haptic engine: \(error)")
        }
    }
    
    // MARK: - Public API
    
    func playHaptic(_ type: HapticFeedbackType, intensity: Float? = nil) {
        guard settings.enabled else { return }
        
        let category = type.category
        guard let categorySettings = settings.categorySettings[category],
              categorySettings.enabled else { return }
        
        let finalIntensity = intensity ?? (categorySettings.customIntensity ? categorySettings.intensity : settings.intensity)
        
        if settings.reducedMotion {
            playReducedMotionHaptic(type, intensity: finalIntensity)
        } else {
            playFullHaptic(type, intensity: finalIntensity)
        }
    }
    
    private func playFullHaptic(_ type: HapticFeedbackType, intensity: Float) {
        if isAdvancedHapticsSupported, let pattern = patternLibrary[type] {
            playAdvancedHaptic(pattern: pattern, intensity: intensity)
        } else {
            playBasicHaptic(type, intensity: intensity)
        }
    }
    
    private func playReducedMotionHaptic(_ type: HapticFeedbackType, intensity: Float) {
        // Use gentler haptics for reduced motion
        let reducedIntensity = intensity * 0.6
        playBasicHaptic(type, intensity: reducedIntensity)
    }
    
    private func playBasicHaptic(_ type: HapticFeedbackType, intensity: Float) {
        switch type {
        case .success:
            notificationGenerator.notificationOccurred(.success)
        case .warning:
            notificationGenerator.notificationOccurred(.warning)
        case .error:
            notificationGenerator.notificationOccurred(.error)
        case .selection:
            selectionGenerator.selectionChanged()
        case .impact(let style):
            playImpactHaptic(style: style, intensity: intensity)
        default:
            // Map other types to basic haptics
            mapToBasicHaptic(type)
        }
    }
    
    private func playImpactHaptic(style: UIImpactFeedbackGenerator.FeedbackStyle, intensity: Float) {
        switch style {
        case .light:
            lightImpactGenerator.impactOccurred(intensity: CGFloat(intensity))
        case .medium:
            mediumImpactGenerator.impactOccurred(intensity: CGFloat(intensity))
        case .heavy:
            heavyImpactGenerator.impactOccurred(intensity: CGFloat(intensity))
        case .soft:
            if #available(iOS 13.0, *) {
                softImpactGenerator?.impactOccurred(intensity: CGFloat(intensity))
            } else {
                lightImpactGenerator.impactOccurred(intensity: CGFloat(intensity))
            }
        case .rigid:
            if #available(iOS 13.0, *) {
                rigidImpactGenerator?.impactOccurred(intensity: CGFloat(intensity))
            } else {
                heavyImpactGenerator.impactOccurred(intensity: CGFloat(intensity))
            }
        @unknown default:
            mediumImpactGenerator.impactOccurred(intensity: CGFloat(intensity))
        }
    }
    
    private func mapToBasicHaptic(_ type: HapticFeedbackType) {
        switch type.category {
        case .health, .emergency:
            notificationGenerator.notificationOccurred(.warning)
        case .exercise, .feedback:
            notificationGenerator.notificationOccurred(.success)
        case .interaction:
            selectionGenerator.selectionChanged()
        default:
            mediumImpactGenerator.impactOccurred()
        }
    }
    
    private func playAdvancedHaptic(pattern: HapticPattern, intensity: Float) {
        guard let engine = hapticEngine else { return }
        
        do {
            let adjustedPattern = adjustPatternIntensity(pattern, intensity: intensity)
            let hapticPattern = try createCHHapticPattern(from: adjustedPattern)
            
            hapticPlayer = try engine.makePlayer(with: hapticPattern)
            
            isPlaying = true
            currentPattern = adjustedPattern
            
            try hapticPlayer?.start(atTime: 0)
            
            // Schedule completion
            DispatchQueue.main.asyncAfter(deadline: .now() + adjustedPattern.duration) {
                self.isPlaying = false
                self.currentPattern = nil
            }
            
        } catch {
            print("Failed to play advanced haptic: \(error)")
            // Fallback to basic haptic
            playBasicHaptic(.impact(.medium), intensity: intensity)
        }
    }
    
    private func adjustPatternIntensity(_ pattern: HapticPattern, intensity: Float) -> HapticPattern {
        let adjustedEvents = pattern.events.map { event in
            HapticEvent(
                time: event.time,
                intensity: event.intensity * intensity,
                sharpness: event.sharpness,
                duration: event.duration,
                type: event.type
            )
        }
        
        return HapticPattern(
            events: adjustedEvents,
            duration: pattern.duration,
            intensity: pattern.intensity * intensity,
            sharpness: pattern.sharpness,
            repeatCount: pattern.repeatCount,
            repeatDelay: pattern.repeatDelay
        )
    }
    
    private func createCHHapticPattern(from pattern: HapticPattern) throws -> CHHapticPattern {
        var hapticEvents: [CHHapticEvent] = []
        
        for event in pattern.events {
            let hapticEvent: CHHapticEvent
            
            switch event.type {
            case .transient:
                hapticEvent = CHHapticEvent(
                    eventType: .hapticTransient,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: event.intensity),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: event.sharpness)
                    ],
                    relativeTime: event.time
                )
            case .continuous:
                hapticEvent = CHHapticEvent(
                    eventType: .hapticContinuous,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: event.intensity),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: event.sharpness)
                    ],
                    relativeTime: event.time,
                    duration: event.duration
                )
            case .parameter:
                // Handle parameter events if needed
                continue
            }
            
            hapticEvents.append(hapticEvent)
        }
        
        return try CHHapticPattern(events: hapticEvents, parameters: [])
    }
    
    // MARK: - Custom Patterns
    
    func createCustomPattern(name: String, events: [HapticEvent]) {
        let pattern = HapticPattern(
            events: events,
            duration: events.map { $0.time + $0.duration }.max() ?? 0.0,
            intensity: events.map { $0.intensity }.max() ?? 1.0,
            sharpness: events.map { $0.sharpness }.max() ?? 1.0,
            repeatCount: 1,
            repeatDelay: 0.0
        )
        
        settings.customPatterns[name] = pattern
        saveSettings()
    }
    
    func playCustomPattern(name: String, intensity: Float? = nil) {
        guard let pattern = settings.customPatterns[name] else { return }
        
        let finalIntensity = intensity ?? settings.intensity
        playAdvancedHaptic(pattern: pattern, intensity: finalIntensity)
    }
    
    func deleteCustomPattern(name: String) {
        settings.customPatterns.removeValue(forKey: name)
        saveSettings()
    }
    
    // MARK: - Settings Management
    
    func updateSettings(_ newSettings: HapticSettings) {
        settings = newSettings
        saveSettings()
    }
    
    func updateCategorySettings(_ category: HapticCategory, settings: HapticCategorySettings) {
        self.settings.categorySettings[category] = settings
        saveSettings()
    }
    
    func resetToDefaults() {
        settings = HapticSettings()
        saveSettings()
    }
    
    // MARK: - Utility Methods
    
    func stopCurrentHaptic() {
        hapticPlayer?.stop(atTime: 0)
        isPlaying = false
        currentPattern = nil
    }
    
    func testHaptic(_ type: HapticFeedbackType) {
        playHaptic(type, intensity: 1.0)
    }
    
    func getAvailablePatterns() -> [HapticFeedbackType] {
        return Array(patternLibrary.keys)
    }
    
    func getCustomPatternNames() -> [String] {
        return Array(settings.customPatterns.keys)
    }
}

// MARK: - SwiftUI Integration

struct HapticFeedbackModifier: ViewModifier {
    let type: HapticFeedbackType
    let intensity: Float?
    let manager: HapticFeedbackManager
    
    func body(content: Content) -> some View {
        content
            .onTapGesture {
                manager.playHaptic(type, intensity: intensity)
            }
    }
}

extension View {
    func hapticFeedback(_ type: HapticFeedbackType, intensity: Float? = nil, manager: HapticFeedbackManager) -> some View {
        modifier(HapticFeedbackModifier(type: type, intensity: intensity, manager: manager))
    }
}

// MARK: - Haptic Settings View

struct HapticSettingsView: View {
    @ObservedObject var hapticManager: HapticFeedbackManager
    @State private var selectedCategory: HapticCategory = .system
    
    var body: some View {
        NavigationView {
            Form {
                Section("General Settings") {
                    Toggle("Enable Haptic Feedback", isOn: $hapticManager.settings.enabled)
                    
                    VStack(alignment: .leading) {
                        Text("Overall Intensity")
                        Slider(value: Binding(
                            get: { Double(hapticManager.settings.intensity) },
                            set: { hapticManager.settings.intensity = Float($0) }
                        ), in: 0...1, step: 0.1)
                        Text("\(Int(hapticManager.settings.intensity * 100))%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Toggle("Accessibility Mode", isOn: $hapticManager.settings.accessibilityMode)
                }
                
                Section("Category Settings") {
                    Picker("Category", selection: $selectedCategory) {
                        ForEach(HapticCategory.allCases, id: \.self) { category in
                            Text(category.displayName).tag(category)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    if let categorySettings = hapticManager.settings.categorySettings[selectedCategory] {
                        CategorySettingsView(
                            category: selectedCategory,
                            settings: categorySettings,
                            hapticManager: hapticManager
                        )
                    }
                }
                
                Section("Test Haptics") {
                    ForEach(hapticManager.getAvailablePatterns().filter { $0.category == selectedCategory }, id: \.displayName) { type in
                        Button(action: {
                            hapticManager.testHaptic(type)
                        }) {
                            HStack {
                                Text(type.displayName)
                                Spacer()
                                Image(systemName: "play.circle")
                            }
                        }
                    }
                }
                
                if !hapticManager.getCustomPatternNames().isEmpty {
                    Section("Custom Patterns") {
                        ForEach(hapticManager.getCustomPatternNames(), id: \.self) { name in
                            HStack {
                                Text(name)
                                Spacer()
                                Button("Test") {
                                    hapticManager.playCustomPattern(name: name)
                                }
                                Button("Delete") {
                                    hapticManager.deleteCustomPattern(name: name)
                                }
                                .foregroundColor(.red)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Haptic Settings")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct CategorySettingsView: View {
    let category: HapticCategory
    @State var settings: HapticCategorySettings
    let hapticManager: HapticFeedbackManager
    
    var body: some View {
        Toggle("Enable \(category.displayName)", isOn: $settings.enabled)
        
        Toggle("Custom Intensity", isOn: $settings.customIntensity)
        
        if settings.customIntensity {
            VStack(alignment: .leading) {
                Text("Intensity")
                Slider(value: Binding(
                    get: { Double(settings.intensity) },
                    set: { settings.intensity = Float($0) }
                ), in: 0...1, step: 0.1)
                Text("\(Int(settings.intensity * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}