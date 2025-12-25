//
//  AdaptiveUIModule.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import Foundation
import SwiftUI
import Combine
import UIKit

// MARK: - Adaptive UI Models

struct UserPreferences {
    var preferredFontSize: FontSize = .medium
    var preferredColorScheme: ColorScheme? = nil
    var preferredContrast: ContrastLevel = .standard
    var preferredAnimationSpeed: AnimationSpeed = .normal
    var preferredLayoutDensity: LayoutDensity = .comfortable
    var preferredNavigationStyle: NavigationStyle = .tabs
    var enableReducedMotion: Bool = false
    var enableLargeText: Bool = false
    var enableHighContrast: Bool = false
    var enableVoiceOver: Bool = false
    var preferredLanguage: String = "en"
    var timeFormat: TimeFormat = .twelveHour
    var dateFormat: DateFormat = .medium
    var measurementUnit: MeasurementUnit = .metric
}

enum FontSize: String, CaseIterable {
    case extraSmall = "xs"
    case small = "s"
    case medium = "m"
    case large = "l"
    case extraLarge = "xl"
    case accessibility = "accessibility"
    
    var scaleFactor: CGFloat {
        switch self {
        case .extraSmall: return 0.8
        case .small: return 0.9
        case .medium: return 1.0
        case .large: return 1.2
        case .extraLarge: return 1.4
        case .accessibility: return 1.8
        }
    }
    
    var description: String {
        switch self {
        case .extraSmall: return "Extra Small"
        case .small: return "Small"
        case .medium: return "Medium"
        case .large: return "Large"
        case .extraLarge: return "Extra Large"
        case .accessibility: return "Accessibility"
        }
    }
}

enum ContrastLevel: String, CaseIterable {
    case low = "low"
    case standard = "standard"
    case high = "high"
    case maximum = "maximum"
    
    var description: String {
        switch self {
        case .low: return "Low Contrast"
        case .standard: return "Standard Contrast"
        case .high: return "High Contrast"
        case .maximum: return "Maximum Contrast"
        }
    }
}

enum AnimationSpeed: String, CaseIterable {
    case slow = "slow"
    case normal = "normal"
    case fast = "fast"
    case none = "none"
    
    var multiplier: Double {
        switch self {
        case .slow: return 1.5
        case .normal: return 1.0
        case .fast: return 0.7
        case .none: return 0.0
        }
    }
    
    var description: String {
        switch self {
        case .slow: return "Slow"
        case .normal: return "Normal"
        case .fast: return "Fast"
        case .none: return "No Animation"
        }
    }
}

enum LayoutDensity: String, CaseIterable {
    case compact = "compact"
    case comfortable = "comfortable"
    case spacious = "spacious"
    
    var spacing: CGFloat {
        switch self {
        case .compact: return 8
        case .comfortable: return 16
        case .spacious: return 24
        }
    }
    
    var padding: CGFloat {
        switch self {
        case .compact: return 12
        case .comfortable: return 16
        case .spacious: return 20
        }
    }
    
    var description: String {
        switch self {
        case .compact: return "Compact"
        case .comfortable: return "Comfortable"
        case .spacious: return "Spacious"
        }
    }
}

enum NavigationStyle: String, CaseIterable {
    case tabs = "tabs"
    case sidebar = "sidebar"
    case drawer = "drawer"
    case minimal = "minimal"
    
    var description: String {
        switch self {
        case .tabs: return "Tab Navigation"
        case .sidebar: return "Sidebar Navigation"
        case .drawer: return "Drawer Navigation"
        case .minimal: return "Minimal Navigation"
        }
    }
}

enum TimeFormat: String, CaseIterable {
    case twelveHour = "12h"
    case twentyFourHour = "24h"
    
    var description: String {
        switch self {
        case .twelveHour: return "12 Hour"
        case .twentyFourHour: return "24 Hour"
        }
    }
}

enum DateFormat: String, CaseIterable {
    case short = "short"
    case medium = "medium"
    case long = "long"
    case full = "full"
    
    var description: String {
        switch self {
        case .short: return "Short (1/1/24)"
        case .medium: return "Medium (Jan 1, 2024)"
        case .long: return "Long (January 1, 2024)"
        case .full: return "Full (Monday, January 1, 2024)"
        }
    }
}

enum MeasurementUnit: String, CaseIterable {
    case metric = "metric"
    case imperial = "imperial"
    
    var description: String {
        switch self {
        case .metric: return "Metric (kg, cm)"
        case .imperial: return "Imperial (lbs, ft)"
        }
    }
}

struct UserBehaviorData {
    var screenTimeBySection: [String: TimeInterval] = [:]
    var mostUsedFeatures: [String: Int] = [:]
    var preferredTimeOfDay: [String: Int] = [:]
    var interactionPatterns: [String: Any] = [:]
    var errorEncounters: [String: Int] = [:]
    var helpRequestsByTopic: [String: Int] = [:]
    var customizationChanges: [String: Date] = [:]
    var accessibilityUsage: [String: Bool] = [:]
    var deviceOrientationPreference: UIDeviceOrientation?
    var averageSessionDuration: TimeInterval = 0
    var lastActiveDate: Date = Date()
}

struct AdaptiveTheme {
    var primaryColor: Color
    var secondaryColor: Color
    var accentColor: Color
    var backgroundColor: Color
    var surfaceColor: Color
    var textColor: Color
    var secondaryTextColor: Color
    var borderColor: Color
    var errorColor: Color
    var warningColor: Color
    var successColor: Color
    var infoColor: Color
    
    static let light = AdaptiveTheme(
        primaryColor: .blue,
        secondaryColor: .gray,
        accentColor: .orange,
        backgroundColor: .white,
        surfaceColor: Color(.systemBackground),
        textColor: .primary,
        secondaryTextColor: .secondary,
        borderColor: Color(.separator),
        errorColor: .red,
        warningColor: .orange,
        successColor: .green,
        infoColor: .blue
    )
    
    static let dark = AdaptiveTheme(
        primaryColor: .blue,
        secondaryColor: .gray,
        accentColor: .orange,
        backgroundColor: .black,
        surfaceColor: Color(.systemBackground),
        textColor: .primary,
        secondaryTextColor: .secondary,
        borderColor: Color(.separator),
        errorColor: .red,
        warningColor: .orange,
        successColor: .green,
        infoColor: .blue
    )
    
    static let highContrast = AdaptiveTheme(
        primaryColor: .black,
        secondaryColor: .gray,
        accentColor: .yellow,
        backgroundColor: .white,
        surfaceColor: .white,
        textColor: .black,
        secondaryTextColor: .black,
        borderColor: .black,
        errorColor: .red,
        warningColor: .orange,
        successColor: .green,
        infoColor: .blue
    )
}

struct AdaptiveLayout {
    var cardCornerRadius: CGFloat
    var buttonHeight: CGFloat
    var minimumTouchTarget: CGFloat
    var horizontalPadding: CGFloat
    var verticalSpacing: CGFloat
    var sectionSpacing: CGFloat
    var iconSize: CGFloat
    var imageCornerRadius: CGFloat
    
    static func create(for density: LayoutDensity, fontSize: FontSize) -> AdaptiveLayout {
        let baseMultiplier = fontSize.scaleFactor
        
        return AdaptiveLayout(
            cardCornerRadius: 12 * baseMultiplier,
            buttonHeight: max(44, 40 * baseMultiplier),
            minimumTouchTarget: max(44, 40 * baseMultiplier),
            horizontalPadding: density.padding * baseMultiplier,
            verticalSpacing: density.spacing * baseMultiplier,
            sectionSpacing: density.spacing * 1.5 * baseMultiplier,
            iconSize: 24 * baseMultiplier,
            imageCornerRadius: 8 * baseMultiplier
        )
    }
}

enum AdaptiveUIError: Error, LocalizedError {
    case preferencesLoadFailed
    case preferencesSaveFailed
    case behaviorDataCorrupted
    case themeApplicationFailed
    
    var errorDescription: String? {
        switch self {
        case .preferencesLoadFailed:
            return "Failed to load user preferences"
        case .preferencesSaveFailed:
            return "Failed to save user preferences"
        case .behaviorDataCorrupted:
            return "User behavior data is corrupted"
        case .themeApplicationFailed:
            return "Failed to apply adaptive theme"
        }
    }
}

// MARK: - Adaptive UI Manager

@MainActor
class AdaptiveUIManager: ObservableObject {
    static let shared = AdaptiveUIManager()
    
    @Published var userPreferences = UserPreferences()
    @Published var currentTheme = AdaptiveTheme.light
    @Published var currentLayout = AdaptiveLayout.create(for: .comfortable, fontSize: .medium)
    @Published var behaviorData = UserBehaviorData()
    @Published var isLearningEnabled = true
    @Published var adaptationLevel: AdaptationLevel = .moderate
    @Published var error: AdaptiveUIError?
    
    private var cancellables = Set<AnyCancellable>()
    private let userDefaults = UserDefaults.standard
    private let behaviorTracker = BehaviorTracker()
    private let themeGenerator = ThemeGenerator()
    private let layoutOptimizer = LayoutOptimizer()
    private let accessibilityMonitor = AccessibilityMonitor()
    
    // Learning algorithms
    private let preferencePredictor = PreferencePredictor()
    private let usageAnalyzer = UsageAnalyzer()
    private let adaptationEngine = AdaptationEngine()
    
    init() {
        loadPreferences()
        loadBehaviorData()
        setupSystemObservers()
        startBehaviorTracking()
        applyCurrentPreferences()
    }
    
    // MARK: - Setup Methods
    
    private func loadPreferences() {
        if let data = userDefaults.data(forKey: "AdaptiveUIPreferences"),
           let preferences = try? JSONDecoder().decode(UserPreferences.self, from: data) {
            userPreferences = preferences
        } else {
            // Load system preferences as defaults
            loadSystemPreferences()
        }
    }
    
    private func loadBehaviorData() {
        if let data = userDefaults.data(forKey: "AdaptiveUIBehaviorData"),
           let behavior = try? JSONDecoder().decode(UserBehaviorData.self, from: data) {
            behaviorData = behavior
        }
    }
    
    private func loadSystemPreferences() {
        userPreferences.preferredColorScheme = UITraitCollection.current.userInterfaceStyle == .dark ? .dark : .light
        userPreferences.enableReducedMotion = UIAccessibility.isReduceMotionEnabled
        userPreferences.enableLargeText = UIApplication.shared.preferredContentSizeCategory.isAccessibilityCategory
        userPreferences.enableHighContrast = UIAccessibility.isDarkerSystemColorsEnabled
        userPreferences.enableVoiceOver = UIAccessibility.isVoiceOverRunning
        
        // Adjust font size based on system settings
        if userPreferences.enableLargeText {
            userPreferences.preferredFontSize = .accessibility
        }
        
        // Adjust contrast based on system settings
        if userPreferences.enableHighContrast {
            userPreferences.preferredContrast = .high
        }
        
        // Adjust animation speed based on reduce motion
        if userPreferences.enableReducedMotion {
            userPreferences.preferredAnimationSpeed = .none
        }
    }
    
    private func setupSystemObservers() {
        // Observe system accessibility changes
        NotificationCenter.default.publisher(for: UIAccessibility.reduceMotionStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.handleAccessibilityChange()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIAccessibility.darkerSystemColorsStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.handleAccessibilityChange()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIAccessibility.voiceOverStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.handleAccessibilityChange()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIContentSizeCategory.didChangeNotification)
            .sink { [weak self] _ in
                self?.handleContentSizeChange()
            }
            .store(in: &cancellables)
        
        // Observe app lifecycle
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { [weak self] _ in
                self?.savePreferences()
                self?.saveBehaviorData()
            }
            .store(in: &cancellables)
    }
    
    private func startBehaviorTracking() {
        guard isLearningEnabled else { return }
        
        behaviorTracker.startTracking { [weak self] behaviorUpdate in
            self?.processBehaviorUpdate(behaviorUpdate)
        }
    }
    
    private func applyCurrentPreferences() {
        updateTheme()
        updateLayout()
    }
    
    // MARK: - Public Methods
    
    func updatePreferences(_ newPreferences: UserPreferences) {
        userPreferences = newPreferences
        applyCurrentPreferences()
        savePreferences()
        
        // Track preference changes for learning
        if isLearningEnabled {
            trackPreferenceChange(newPreferences)
        }
    }
    
    func adaptToUserBehavior() {
        guard isLearningEnabled else { return }
        
        let suggestions = adaptationEngine.generateAdaptations(
            preferences: userPreferences,
            behaviorData: behaviorData,
            level: adaptationLevel
        )
        
        applySuggestions(suggestions)
    }
    
    func resetToDefaults() {
        userPreferences = UserPreferences()
        behaviorData = UserBehaviorData()
        loadSystemPreferences()
        applyCurrentPreferences()
        savePreferences()
        saveBehaviorData()
    }
    
    func exportPreferences() -> Data? {
        return try? JSONEncoder().encode(userPreferences)
    }
    
    func importPreferences(from data: Data) -> Bool {
        guard let preferences = try? JSONDecoder().decode(UserPreferences.self, from: data) else {
            return false
        }
        
        updatePreferences(preferences)
        return true
    }
    
    func trackFeatureUsage(_ feature: String) {
        guard isLearningEnabled else { return }
        
        behaviorData.mostUsedFeatures[feature, default: 0] += 1
        
        // Track time of day usage
        let hour = Calendar.current.component(.hour, from: Date())
        let timeKey = "\(hour):00"
        behaviorData.preferredTimeOfDay[timeKey, default: 0] += 1
        
        saveBehaviorData()
    }
    
    func trackScreenTime(for section: String, duration: TimeInterval) {
        guard isLearningEnabled else { return }
        
        behaviorData.screenTimeBySection[section, default: 0] += duration
        saveBehaviorData()
    }
    
    func trackError(_ error: String) {
        guard isLearningEnabled else { return }
        
        behaviorData.errorEncounters[error, default: 0] += 1
        saveBehaviorData()
    }
    
    func trackHelpRequest(_ topic: String) {
        guard isLearningEnabled else { return }
        
        behaviorData.helpRequestsByTopic[topic, default: 0] += 1
        saveBehaviorData()
    }
    
    // MARK: - Private Methods
    
    private func updateTheme() {
        currentTheme = themeGenerator.generateTheme(
            colorScheme: userPreferences.preferredColorScheme,
            contrast: userPreferences.preferredContrast,
            accessibility: userPreferences.enableHighContrast
        )
    }
    
    private func updateLayout() {
        currentLayout = AdaptiveLayout.create(
            for: userPreferences.preferredLayoutDensity,
            fontSize: userPreferences.preferredFontSize
        )
    }
    
    private func handleAccessibilityChange() {
        loadSystemPreferences()
        applyCurrentPreferences()
        savePreferences()
    }
    
    private func handleContentSizeChange() {
        if UIApplication.shared.preferredContentSizeCategory.isAccessibilityCategory {
            userPreferences.preferredFontSize = .accessibility
        } else {
            // Map system content size to our font sizes
            let category = UIApplication.shared.preferredContentSizeCategory
            switch category {
            case .extraSmall, .small:
                userPreferences.preferredFontSize = .small
            case .medium:
                userPreferences.preferredFontSize = .medium
            case .large, .extraLarge:
                userPreferences.preferredFontSize = .large
            case .extraExtraLarge, .extraExtraExtraLarge:
                userPreferences.preferredFontSize = .extraLarge
            default:
                userPreferences.preferredFontSize = .medium
            }
        }
        
        applyCurrentPreferences()
        savePreferences()
    }
    
    private func processBehaviorUpdate(_ update: BehaviorUpdate) {
        // Process behavior updates from the tracker
        switch update.type {
        case .featureUsage:
            trackFeatureUsage(update.feature)
        case .screenTime:
            trackScreenTime(for: update.section, duration: update.duration)
        case .error:
            trackError(update.errorType)
        case .helpRequest:
            trackHelpRequest(update.topic)
        }
    }
    
    private func trackPreferenceChange(_ preferences: UserPreferences) {
        let changeKey = "preference_change_\(Date().timeIntervalSince1970)"
        behaviorData.customizationChanges[changeKey] = Date()
        saveBehaviorData()
    }
    
    private func applySuggestions(_ suggestions: [AdaptationSuggestion]) {
        for suggestion in suggestions {
            switch suggestion.type {
            case .fontSizeIncrease:
                if userPreferences.preferredFontSize != .accessibility {
                    let currentIndex = FontSize.allCases.firstIndex(of: userPreferences.preferredFontSize) ?? 2
                    let newIndex = min(currentIndex + 1, FontSize.allCases.count - 1)
                    userPreferences.preferredFontSize = FontSize.allCases[newIndex]
                }
            case .contrastIncrease:
                if userPreferences.preferredContrast != .maximum {
                    let currentIndex = ContrastLevel.allCases.firstIndex(of: userPreferences.preferredContrast) ?? 1
                    let newIndex = min(currentIndex + 1, ContrastLevel.allCases.count - 1)
                    userPreferences.preferredContrast = ContrastLevel.allCases[newIndex]
                }
            case .animationSpeedDecrease:
                if userPreferences.preferredAnimationSpeed != .none {
                    let currentIndex = AnimationSpeed.allCases.firstIndex(of: userPreferences.preferredAnimationSpeed) ?? 1
                    let newIndex = min(currentIndex + 1, AnimationSpeed.allCases.count - 1)
                    userPreferences.preferredAnimationSpeed = AnimationSpeed.allCases[newIndex]
                }
            case .layoutDensityIncrease:
                if userPreferences.preferredLayoutDensity != .spacious {
                    let currentIndex = LayoutDensity.allCases.firstIndex(of: userPreferences.preferredLayoutDensity) ?? 1
                    let newIndex = min(currentIndex + 1, LayoutDensity.allCases.count - 1)
                    userPreferences.preferredLayoutDensity = LayoutDensity.allCases[newIndex]
                }
            }
        }
        
        applyCurrentPreferences()
        savePreferences()
    }
    
    private func savePreferences() {
        do {
            let data = try JSONEncoder().encode(userPreferences)
            userDefaults.set(data, forKey: "AdaptiveUIPreferences")
        } catch {
            self.error = .preferencesSaveFailed
        }
    }
    
    private func saveBehaviorData() {
        do {
            let data = try JSONEncoder().encode(behaviorData)
            userDefaults.set(data, forKey: "AdaptiveUIBehaviorData")
        } catch {
            self.error = .behaviorDataCorrupted
        }
    }
}

// MARK: - Supporting Classes

enum AdaptationLevel: String, CaseIterable {
    case minimal = "minimal"
    case moderate = "moderate"
    case aggressive = "aggressive"
    
    var description: String {
        switch self {
        case .minimal: return "Minimal Adaptation"
        case .moderate: return "Moderate Adaptation"
        case .aggressive: return "Aggressive Adaptation"
        }
    }
}

struct BehaviorUpdate {
    let type: BehaviorUpdateType
    let feature: String
    let section: String
    let duration: TimeInterval
    let errorType: String
    let topic: String
    let timestamp: Date
    
    init(type: BehaviorUpdateType, feature: String = "", section: String = "", duration: TimeInterval = 0, errorType: String = "", topic: String = "") {
        self.type = type
        self.feature = feature
        self.section = section
        self.duration = duration
        self.errorType = errorType
        self.topic = topic
        self.timestamp = Date()
    }
}

enum BehaviorUpdateType {
    case featureUsage
    case screenTime
    case error
    case helpRequest
}

struct AdaptationSuggestion {
    let type: AdaptationSuggestionType
    let confidence: Float
    let reason: String
}

enum AdaptationSuggestionType {
    case fontSizeIncrease
    case contrastIncrease
    case animationSpeedDecrease
    case layoutDensityIncrease
}

class BehaviorTracker {
    private var updateHandler: ((BehaviorUpdate) -> Void)?
    
    func startTracking(updateHandler: @escaping (BehaviorUpdate) -> Void) {
        self.updateHandler = updateHandler
    }
    
    func trackFeatureUsage(_ feature: String) {
        updateHandler?(BehaviorUpdate(type: .featureUsage, feature: feature))
    }
    
    func trackScreenTime(section: String, duration: TimeInterval) {
        updateHandler?(BehaviorUpdate(type: .screenTime, section: section, duration: duration))
    }
    
    func trackError(_ errorType: String) {
        updateHandler?(BehaviorUpdate(type: .error, errorType: errorType))
    }
    
    func trackHelpRequest(_ topic: String) {
        updateHandler?(BehaviorUpdate(type: .helpRequest, topic: topic))
    }
}

class ThemeGenerator {
    func generateTheme(colorScheme: ColorScheme?, contrast: ContrastLevel, accessibility: Bool) -> AdaptiveTheme {
        if accessibility || contrast == .maximum {
            return AdaptiveTheme.highContrast
        }
        
        switch colorScheme {
        case .dark:
            return AdaptiveTheme.dark
        case .light, .none:
            return AdaptiveTheme.light
        @unknown default:
            return AdaptiveTheme.light
        }
    }
}

class LayoutOptimizer {
    func optimizeLayout(for preferences: UserPreferences, behaviorData: UserBehaviorData) -> AdaptiveLayout {
        return AdaptiveLayout.create(for: preferences.preferredLayoutDensity, fontSize: preferences.preferredFontSize)
    }
}

class AccessibilityMonitor {
    func getCurrentAccessibilitySettings() -> [String: Bool] {
        return [
            "reduceMotion": UIAccessibility.isReduceMotionEnabled,
            "largeText": UIApplication.shared.preferredContentSizeCategory.isAccessibilityCategory,
            "highContrast": UIAccessibility.isDarkerSystemColorsEnabled,
            "voiceOver": UIAccessibility.isVoiceOverRunning,
            "switchControl": UIAccessibility.isSwitchControlRunning,
            "assistiveTouch": UIAccessibility.isAssistiveTouchRunning
        ]
    }
}

class PreferencePredictor {
    func predictPreferences(from behaviorData: UserBehaviorData) -> UserPreferences {
        var predictions = UserPreferences()
        
        // Predict font size based on error encounters
        if behaviorData.errorEncounters.values.reduce(0, +) > 10 {
            predictions.preferredFontSize = .large
        }
        
        // Predict layout density based on screen time patterns
        let totalScreenTime = behaviorData.screenTimeBySection.values.reduce(0, +)
        if totalScreenTime > 3600 { // More than 1 hour
            predictions.preferredLayoutDensity = .spacious
        }
        
        return predictions
    }
}

class UsageAnalyzer {
    func analyzeUsagePatterns(_ behaviorData: UserBehaviorData) -> [String: Any] {
        var analysis: [String: Any] = [:]
        
        // Analyze most used features
        let sortedFeatures = behaviorData.mostUsedFeatures.sorted { $0.value > $1.value }
        analysis["topFeatures"] = Array(sortedFeatures.prefix(5))
        
        // Analyze time patterns
        let sortedTimes = behaviorData.preferredTimeOfDay.sorted { $0.value > $1.value }
        analysis["peakUsageTimes"] = Array(sortedTimes.prefix(3))
        
        // Analyze error patterns
        analysis["errorRate"] = behaviorData.errorEncounters.values.reduce(0, +)
        
        return analysis
    }
}

class AdaptationEngine {
    func generateAdaptations(preferences: UserPreferences, behaviorData: UserBehaviorData, level: AdaptationLevel) -> [AdaptationSuggestion] {
        var suggestions: [AdaptationSuggestion] = []
        
        let errorCount = behaviorData.errorEncounters.values.reduce(0, +)
        let helpRequestCount = behaviorData.helpRequestsByTopic.values.reduce(0, +)
        
        // Suggest font size increase if many errors or help requests
        if errorCount > 5 || helpRequestCount > 3 {
            suggestions.append(AdaptationSuggestion(
                type: .fontSizeIncrease,
                confidence: 0.8,
                reason: "High error rate suggests text may be too small"
            ))
        }
        
        // Suggest contrast increase if accessibility features are used
        if behaviorData.accessibilityUsage["highContrast"] == true {
            suggestions.append(AdaptationSuggestion(
                type: .contrastIncrease,
                confidence: 0.9,
                reason: "User has enabled high contrast accessibility"
            ))
        }
        
        // Suggest slower animations if reduce motion is enabled
        if behaviorData.accessibilityUsage["reduceMotion"] == true {
            suggestions.append(AdaptationSuggestion(
                type: .animationSpeedDecrease,
                confidence: 1.0,
                reason: "User has enabled reduce motion accessibility"
            ))
        }
        
        return suggestions
    }
}

// MARK: - SwiftUI Extensions

extension View {
    func adaptiveFont(_ style: Font.TextStyle = .body) -> some View {
        let manager = AdaptiveUIManager.shared
        let scaleFactor = manager.userPreferences.preferredFontSize.scaleFactor
        
        return self.font(.system(style).weight(.regular))
            .scaleEffect(scaleFactor)
    }
    
    func adaptivePadding() -> some View {
        let manager = AdaptiveUIManager.shared
        return self.padding(manager.currentLayout.horizontalPadding)
    }
    
    func adaptiveSpacing() -> some View {
        let manager = AdaptiveUIManager.shared
        return self.padding(.vertical, manager.currentLayout.verticalSpacing)
    }
    
    func adaptiveAnimation<V: Equatable>(_ animation: Animation?, value: V) -> some View {
        let manager = AdaptiveUIManager.shared
        let speed = manager.userPreferences.preferredAnimationSpeed
        
        if speed == .none {
            return self
        } else {
            let adjustedAnimation = animation?.speed(speed.multiplier)
            return self.animation(adjustedAnimation, value: value)
        }
    }
    
    func trackFeatureUsage(_ feature: String) -> some View {
        self.onAppear {
            AdaptiveUIManager.shared.trackFeatureUsage(feature)
        }
    }
    
    func trackScreenTime(_ section: String) -> some View {
        self.onAppear {
            let startTime = Date()
            return self.onDisappear {
                let duration = Date().timeIntervalSince(startTime)
                AdaptiveUIManager.shared.trackScreenTime(for: section, duration: duration)
            }
        }
    }
}