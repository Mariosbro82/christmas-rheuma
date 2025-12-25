//
//  AdaptiveUISystem.swift
//  InflamAI-Swift
//
//  Adaptive UI system that learns user preferences and adjusts interface automatically
//

import Foundation
import SwiftUI
import Combine
import CoreData
import UIKit

// MARK: - UI Adaptation Types

enum UIAdaptationType {
    case fontSize(CGFloat)
    case colorScheme(ColorScheme)
    case contrast(ContrastLevel)
    case buttonSize(ButtonSize)
    case spacing(SpacingLevel)
    case animationSpeed(AnimationSpeed)
    case navigationStyle(NavigationStyle)
    case inputMethod(InputMethod)
    case layout(LayoutStyle)
    case accessibility(AccessibilityLevel)
}

enum ContrastLevel: String, CaseIterable {
    case low = "Low"
    case normal = "Normal"
    case high = "High"
    case maximum = "Maximum"
    
    var multiplier: Double {
        switch self {
        case .low: return 0.7
        case .normal: return 1.0
        case .high: return 1.3
        case .maximum: return 1.6
        }
    }
}

enum ButtonSize: String, CaseIterable {
    case small = "Small"
    case medium = "Medium"
    case large = "Large"
    case extraLarge = "Extra Large"
    
    var height: CGFloat {
        switch self {
        case .small: return 36
        case .medium: return 44
        case .large: return 52
        case .extraLarge: return 60
        }
    }
    
    var padding: CGFloat {
        switch self {
        case .small: return 8
        case .medium: return 12
        case .large: return 16
        case .extraLarge: return 20
        }
    }
}

enum SpacingLevel: String, CaseIterable {
    case compact = "Compact"
    case normal = "Normal"
    case comfortable = "Comfortable"
    case spacious = "Spacious"
    
    var multiplier: CGFloat {
        switch self {
        case .compact: return 0.7
        case .normal: return 1.0
        case .comfortable: return 1.3
        case .spacious: return 1.6
        }
    }
}

enum AnimationSpeed: String, CaseIterable {
    case slow = "Slow"
    case normal = "Normal"
    case fast = "Fast"
    case none = "None"
    
    var duration: Double {
        switch self {
        case .slow: return 0.6
        case .normal: return 0.3
        case .fast: return 0.15
        case .none: return 0.0
        }
    }
}

enum NavigationStyle: String, CaseIterable {
    case standard = "Standard"
    case simplified = "Simplified"
    case gestureOnly = "Gesture Only"
    case voiceFirst = "Voice First"
}

enum InputMethod: String, CaseIterable {
    case touch = "Touch"
    case voice = "Voice"
    case hybrid = "Hybrid"
    case accessibility = "Accessibility"
}

enum LayoutStyle: String, CaseIterable {
    case compact = "Compact"
    case standard = "Standard"
    case comfortable = "Comfortable"
    case singleColumn = "Single Column"
}

enum AccessibilityLevel: String, CaseIterable {
    case standard = "Standard"
    case enhanced = "Enhanced"
    case maximum = "Maximum"
}

// MARK: - User Behavior Tracking

struct UserInteraction {
    let timestamp: Date
    let action: String
    let screen: String
    let duration: TimeInterval
    let success: Bool
    let difficulty: DifficultyLevel
    let context: [String: Any]
}

enum DifficultyLevel: Int {
    case easy = 1
    case moderate = 2
    case difficult = 3
    case veryDifficult = 4
}

struct UsagePattern {
    let feature: String
    let frequency: Double
    let averageDuration: TimeInterval
    let successRate: Double
    let preferredTime: DateComponents
    let context: String
}

struct AccessibilityNeed {
    let type: AccessibilityType
    let severity: AccessibilitySeverity
    let adaptations: [UIAdaptationType]
    let detectedAt: Date
}

enum AccessibilityType {
    case visualImpairment
    case motorImpairment
    case cognitiveImpairment
    case hearingImpairment
    case temporaryImpairment
}

enum AccessibilitySeverity: Int {
    case mild = 1
    case moderate = 2
    case severe = 3
}

// MARK: - Adaptive UI System

class AdaptiveUISystem: ObservableObject {
    // Core Data
    private let context: NSManagedObjectContext
    
    // Current UI State
    @Published var currentTheme: AdaptiveTheme = AdaptiveTheme()
    @Published var fontSize: CGFloat = 16
    @Published var colorScheme: ColorScheme = .light
    @Published var contrastLevel: ContrastLevel = .normal
    @Published var buttonSize: ButtonSize = .medium
    @Published var spacingLevel: SpacingLevel = .normal
    @Published var animationSpeed: AnimationSpeed = .normal
    @Published var navigationStyle: NavigationStyle = .standard
    @Published var inputMethod: InputMethod = .touch
    @Published var layoutStyle: LayoutStyle = .standard
    @Published var accessibilityLevel: AccessibilityLevel = .standard
    
    // Learning System
    private var userInteractions: [UserInteraction] = []
    private var usagePatterns: [String: UsagePattern] = [:]
    private var accessibilityNeeds: [AccessibilityNeed] = []
    
    // Adaptation Engine
    private let adaptationEngine = UIAdaptationEngine()
    private let behaviorAnalyzer = UserBehaviorAnalyzer()
    private let accessibilityDetector = AccessibilityNeedsDetector()
    
    // Settings
    @Published var adaptiveUIEnabled = true
    @Published var learningEnabled = true
    @Published var autoAdjustEnabled = true
    @Published var accessibilityAutoDetection = true
    
    // State
    @Published var isLearning = false
    @Published var adaptationSuggestions: [AdaptationSuggestion] = []
    
    // Timers
    private var analysisTimer: Timer?
    private var adaptationTimer: Timer?
    
    // Cancellables
    private var cancellables = Set<AnyCancellable>()
    
    init(context: NSManagedObjectContext) {
        self.context = context
        
        loadUserPreferences()
        setupPeriodicAnalysis()
        setupSystemObservers()
        startLearning()
    }
    
    // MARK: - Setup
    
    private func loadUserPreferences() {
        // Load saved UI preferences
        fontSize = CGFloat(UserDefaults.standard.float(forKey: "adaptiveUI_fontSize"))
        if fontSize == 0 { fontSize = 16 }
        
        let colorSchemeRaw = UserDefaults.standard.string(forKey: "adaptiveUI_colorScheme") ?? "light"
        colorScheme = colorSchemeRaw == "dark" ? .dark : .light
        
        let contrastRaw = UserDefaults.standard.string(forKey: "adaptiveUI_contrast") ?? "normal"
        contrastLevel = ContrastLevel(rawValue: contrastRaw) ?? .normal
        
        let buttonSizeRaw = UserDefaults.standard.string(forKey: "adaptiveUI_buttonSize") ?? "medium"
        buttonSize = ButtonSize(rawValue: buttonSizeRaw) ?? .medium
        
        let spacingRaw = UserDefaults.standard.string(forKey: "adaptiveUI_spacing") ?? "normal"
        spacingLevel = SpacingLevel(rawValue: spacingRaw) ?? .normal
        
        let animationRaw = UserDefaults.standard.string(forKey: "adaptiveUI_animation") ?? "normal"
        animationSpeed = AnimationSpeed(rawValue: animationRaw) ?? .normal
        
        let navigationRaw = UserDefaults.standard.string(forKey: "adaptiveUI_navigation") ?? "standard"
        navigationStyle = NavigationStyle(rawValue: navigationRaw) ?? .standard
        
        let inputRaw = UserDefaults.standard.string(forKey: "adaptiveUI_input") ?? "touch"
        inputMethod = InputMethod(rawValue: inputRaw) ?? .touch
        
        let layoutRaw = UserDefaults.standard.string(forKey: "adaptiveUI_layout") ?? "standard"
        layoutStyle = LayoutStyle(rawValue: layoutRaw) ?? .standard
        
        let accessibilityRaw = UserDefaults.standard.string(forKey: "adaptiveUI_accessibility") ?? "standard"
        accessibilityLevel = AccessibilityLevel(rawValue: accessibilityRaw) ?? .standard
        
        // Load system settings
        adaptiveUIEnabled = UserDefaults.standard.bool(forKey: "adaptiveUI_enabled")
        if !adaptiveUIEnabled && UserDefaults.standard.object(forKey: "adaptiveUI_enabled") == nil {
            adaptiveUIEnabled = true
        }
        
        learningEnabled = UserDefaults.standard.bool(forKey: "adaptiveUI_learning")
        if !learningEnabled && UserDefaults.standard.object(forKey: "adaptiveUI_learning") == nil {
            learningEnabled = true
        }
        
        autoAdjustEnabled = UserDefaults.standard.bool(forKey: "adaptiveUI_autoAdjust")
        if !autoAdjustEnabled && UserDefaults.standard.object(forKey: "adaptiveUI_autoAdjust") == nil {
            autoAdjustEnabled = true
        }
        
        accessibilityAutoDetection = UserDefaults.standard.bool(forKey: "adaptiveUI_accessibilityDetection")
        if !accessibilityAutoDetection && UserDefaults.standard.object(forKey: "adaptiveUI_accessibilityDetection") == nil {
            accessibilityAutoDetection = true
        }
        
        // Load historical data
        loadHistoricalData()
        
        // Update theme
        updateCurrentTheme()
    }
    
    private func loadHistoricalData() {
        // Load user interactions from Core Data or UserDefaults
        if let data = UserDefaults.standard.data(forKey: "adaptiveUI_interactions"),
           let interactions = try? JSONDecoder().decode([UserInteractionData].self, from: data) {
            userInteractions = interactions.map { $0.toUserInteraction() }
        }
        
        // Load usage patterns
        if let data = UserDefaults.standard.data(forKey: "adaptiveUI_patterns"),
           let patterns = try? JSONDecoder().decode([String: UsagePatternData].self, from: data) {
            usagePatterns = patterns.mapValues { $0.toUsagePattern() }
        }
        
        // Load accessibility needs
        if let data = UserDefaults.standard.data(forKey: "adaptiveUI_accessibilityNeeds"),
           let needs = try? JSONDecoder().decode([AccessibilityNeedData].self, from: data) {
            accessibilityNeeds = needs.map { $0.toAccessibilityNeed() }
        }
    }
    
    private func setupPeriodicAnalysis() {
        // Analyze user behavior every 5 minutes
        analysisTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak self] _ in
            self?.analyzeUserBehavior()
        }
        
        // Check for adaptations every hour
        adaptationTimer = Timer.scheduledTimer(withTimeInterval: 3600, repeats: true) { [weak self] _ in
            self?.checkForAdaptations()
        }
    }
    
    private func setupSystemObservers() {
        // Listen for accessibility changes
        NotificationCenter.default.publisher(for: UIAccessibility.voiceOverStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.detectAccessibilityChanges()
            }
            .store(in: &cancellables)
        
        NotificationCenter.default.publisher(for: UIAccessibility.switchControlStatusDidChangeNotification)
            .sink { [weak self] _ in
                self?.detectAccessibilityChanges()
            }
            .store(in: &cancellables)
        
        // Listen for app state changes
        NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)
            .sink { [weak self] _ in
                self?.onAppBecameActive()
            }
            .store(in: &cancellables)
    }
    
    private func startLearning() {
        guard learningEnabled else { return }
        isLearning = true
    }
    
    // MARK: - User Interaction Tracking
    
    func trackInteraction(
        action: String,
        screen: String,
        duration: TimeInterval = 0,
        success: Bool = true,
        difficulty: DifficultyLevel = .easy,
        context: [String: Any] = [:]
    ) {
        guard learningEnabled else { return }
        
        let interaction = UserInteraction(
            timestamp: Date(),
            action: action,
            screen: screen,
            duration: duration,
            success: success,
            difficulty: difficulty,
            context: context
        )
        
        userInteractions.append(interaction)
        
        // Keep only recent interactions (last 30 days)
        let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
        userInteractions = userInteractions.filter { $0.timestamp > thirtyDaysAgo }
        
        // Save to persistent storage
        saveInteractionData()
        
        // Trigger immediate analysis for critical interactions
        if difficulty.rawValue >= 3 || !success {
            analyzeUserBehavior()
        }
    }
    
    func trackScreenTime(screen: String, duration: TimeInterval) {
        trackInteraction(
            action: "screen_view",
            screen: screen,
            duration: duration,
            success: true,
            difficulty: .easy,
            context: ["duration": duration]
        )
    }
    
    func trackGestureFailure(gesture: String, screen: String, attempts: Int) {
        let difficulty: DifficultyLevel = attempts > 3 ? .veryDifficult : (attempts > 2 ? .difficult : .moderate)
        
        trackInteraction(
            action: "gesture_failure",
            screen: screen,
            success: false,
            difficulty: difficulty,
            context: ["gesture": gesture, "attempts": attempts]
        )
    }
    
    func trackAccessibilityUsage(feature: String, screen: String) {
        trackInteraction(
            action: "accessibility_feature",
            screen: screen,
            success: true,
            difficulty: .easy,
            context: ["feature": feature]
        )
    }
    
    // MARK: - Behavior Analysis
    
    private func analyzeUserBehavior() {
        guard learningEnabled && !userInteractions.isEmpty else { return }
        
        // Analyze usage patterns
        analyzeUsagePatterns()
        
        // Detect accessibility needs
        if accessibilityAutoDetection {
            detectAccessibilityNeeds()
        }
        
        // Generate adaptation suggestions
        generateAdaptationSuggestions()
        
        // Auto-apply adaptations if enabled
        if autoAdjustEnabled {
            applyAutoAdaptations()
        }
    }
    
    private func analyzeUsagePatterns() {
        let analyzer = behaviorAnalyzer
        
        // Group interactions by feature
        let groupedInteractions = Dictionary(grouping: userInteractions) { $0.action }
        
        for (feature, interactions) in groupedInteractions {
            let pattern = analyzer.analyzePattern(for: feature, interactions: interactions)
            usagePatterns[feature] = pattern
        }
        
        // Save patterns
        savePatternData()
    }
    
    private func detectAccessibilityNeeds() {
        let detector = accessibilityDetector
        let newNeeds = detector.detectNeeds(from: userInteractions)
        
        for need in newNeeds {
            if !accessibilityNeeds.contains(where: { $0.type == need.type }) {
                accessibilityNeeds.append(need)
            }
        }
        
        // Save accessibility needs
        saveAccessibilityData()
    }
    
    private func generateAdaptationSuggestions() {
        let engine = adaptationEngine
        let suggestions = engine.generateSuggestions(
            patterns: usagePatterns,
            accessibilityNeeds: accessibilityNeeds,
            currentSettings: getCurrentSettings()
        )
        
        DispatchQueue.main.async {
            self.adaptationSuggestions = suggestions
        }
    }
    
    private func applyAutoAdaptations() {
        let engine = adaptationEngine
        let adaptations = engine.generateAutoAdaptations(
            patterns: usagePatterns,
            accessibilityNeeds: accessibilityNeeds,
            currentSettings: getCurrentSettings()
        )
        
        for adaptation in adaptations {
            applyAdaptation(adaptation)
        }
    }
    
    // MARK: - Adaptation Application
    
    func applyAdaptation(_ adaptation: UIAdaptationType) {
        DispatchQueue.main.async {
            switch adaptation {
            case .fontSize(let size):
                self.fontSize = size
            case .colorScheme(let scheme):
                self.colorScheme = scheme
            case .contrast(let level):
                self.contrastLevel = level
            case .buttonSize(let size):
                self.buttonSize = size
            case .spacing(let level):
                self.spacingLevel = level
            case .animationSpeed(let speed):
                self.animationSpeed = speed
            case .navigationStyle(let style):
                self.navigationStyle = style
            case .inputMethod(let method):
                self.inputMethod = method
            case .layout(let style):
                self.layoutStyle = style
            case .accessibility(let level):
                self.accessibilityLevel = level
            }
            
            self.updateCurrentTheme()
            self.saveUserPreferences()
        }
    }
    
    func applySuggestion(_ suggestion: AdaptationSuggestion) {
        for adaptation in suggestion.adaptations {
            applyAdaptation(adaptation)
        }
        
        // Remove applied suggestion
        adaptationSuggestions.removeAll { $0.id == suggestion.id }
    }
    
    func dismissSuggestion(_ suggestion: AdaptationSuggestion) {
        adaptationSuggestions.removeAll { $0.id == suggestion.id }
    }
    
    // MARK: - Theme Management
    
    private func updateCurrentTheme() {
        currentTheme = AdaptiveTheme(
            fontSize: fontSize,
            colorScheme: colorScheme,
            contrastLevel: contrastLevel,
            buttonSize: buttonSize,
            spacingLevel: spacingLevel,
            animationSpeed: animationSpeed,
            navigationStyle: navigationStyle,
            inputMethod: inputMethod,
            layoutStyle: layoutStyle,
            accessibilityLevel: accessibilityLevel
        )
    }
    
    // MARK: - Accessibility Detection
    
    private func detectAccessibilityChanges() {
        guard accessibilityAutoDetection else { return }
        
        var detectedNeeds: [AccessibilityNeed] = []
        
        // Check VoiceOver
        if UIAccessibility.isVoiceOverRunning {
            detectedNeeds.append(AccessibilityNeed(
                type: .visualImpairment,
                severity: .severe,
                adaptations: [
                    .fontSize(20),
                    .buttonSize(.extraLarge),
                    .spacing(.spacious),
                    .contrast(.high),
                    .accessibility(.maximum)
                ],
                detectedAt: Date()
            ))
        }
        
        // Check Switch Control
        if UIAccessibility.isSwitchControlRunning {
            detectedNeeds.append(AccessibilityNeed(
                type: .motorImpairment,
                severity: .severe,
                adaptations: [
                    .buttonSize(.extraLarge),
                    .spacing(.spacious),
                    .animationSpeed(.slow),
                    .navigationStyle(.simplified)
                ],
                detectedAt: Date()
            ))
        }
        
        // Check Reduce Motion
        if UIAccessibility.isReduceMotionEnabled {
            detectedNeeds.append(AccessibilityNeed(
                type: .cognitiveImpairment,
                severity: .moderate,
                adaptations: [
                    .animationSpeed(.none),
                    .navigationStyle(.simplified)
                ],
                detectedAt: Date()
            ))
        }
        
        // Add new needs
        for need in detectedNeeds {
            if !accessibilityNeeds.contains(where: { $0.type == need.type }) {
                accessibilityNeeds.append(need)
                
                // Auto-apply critical accessibility adaptations
                if autoAdjustEnabled {
                    for adaptation in need.adaptations {
                        applyAdaptation(adaptation)
                    }
                }
            }
        }
    }
    
    private func onAppBecameActive() {
        // Check for system accessibility changes
        detectAccessibilityChanges()
        
        // Resume learning if it was paused
        if learningEnabled && !isLearning {
            startLearning()
        }
    }
    
    // MARK: - Data Persistence
    
    private func saveUserPreferences() {
        UserDefaults.standard.set(Float(fontSize), forKey: "adaptiveUI_fontSize")
        UserDefaults.standard.set(colorScheme == .dark ? "dark" : "light", forKey: "adaptiveUI_colorScheme")
        UserDefaults.standard.set(contrastLevel.rawValue, forKey: "adaptiveUI_contrast")
        UserDefaults.standard.set(buttonSize.rawValue, forKey: "adaptiveUI_buttonSize")
        UserDefaults.standard.set(spacingLevel.rawValue, forKey: "adaptiveUI_spacing")
        UserDefaults.standard.set(animationSpeed.rawValue, forKey: "adaptiveUI_animation")
        UserDefaults.standard.set(navigationStyle.rawValue, forKey: "adaptiveUI_navigation")
        UserDefaults.standard.set(inputMethod.rawValue, forKey: "adaptiveUI_input")
        UserDefaults.standard.set(layoutStyle.rawValue, forKey: "adaptiveUI_layout")
        UserDefaults.standard.set(accessibilityLevel.rawValue, forKey: "adaptiveUI_accessibility")
        
        UserDefaults.standard.set(adaptiveUIEnabled, forKey: "adaptiveUI_enabled")
        UserDefaults.standard.set(learningEnabled, forKey: "adaptiveUI_learning")
        UserDefaults.standard.set(autoAdjustEnabled, forKey: "adaptiveUI_autoAdjust")
        UserDefaults.standard.set(accessibilityAutoDetection, forKey: "adaptiveUI_accessibilityDetection")
    }
    
    private func saveInteractionData() {
        let data = userInteractions.map { UserInteractionData(from: $0) }
        if let encoded = try? JSONEncoder().encode(data) {
            UserDefaults.standard.set(encoded, forKey: "adaptiveUI_interactions")
        }
    }
    
    private func savePatternData() {
        let data = usagePatterns.mapValues { UsagePatternData(from: $0) }
        if let encoded = try? JSONEncoder().encode(data) {
            UserDefaults.standard.set(encoded, forKey: "adaptiveUI_patterns")
        }
    }
    
    private func saveAccessibilityData() {
        let data = accessibilityNeeds.map { AccessibilityNeedData(from: $0) }
        if let encoded = try? JSONEncoder().encode(data) {
            UserDefaults.standard.set(encoded, forKey: "adaptiveUI_accessibilityNeeds")
        }
    }
    
    // MARK: - Utility Methods
    
    private func getCurrentSettings() -> [String: Any] {
        return [
            "fontSize": fontSize,
            "colorScheme": colorScheme == .dark ? "dark" : "light",
            "contrastLevel": contrastLevel.rawValue,
            "buttonSize": buttonSize.rawValue,
            "spacingLevel": spacingLevel.rawValue,
            "animationSpeed": animationSpeed.rawValue,
            "navigationStyle": navigationStyle.rawValue,
            "inputMethod": inputMethod.rawValue,
            "layoutStyle": layoutStyle.rawValue,
            "accessibilityLevel": accessibilityLevel.rawValue
        ]
    }
    
    func resetToDefaults() {
        fontSize = 16
        colorScheme = .light
        contrastLevel = .normal
        buttonSize = .medium
        spacingLevel = .normal
        animationSpeed = .normal
        navigationStyle = .standard
        inputMethod = .touch
        layoutStyle = .standard
        accessibilityLevel = .standard
        
        updateCurrentTheme()
        saveUserPreferences()
    }
    
    func clearLearningData() {
        userInteractions.removeAll()
        usagePatterns.removeAll()
        accessibilityNeeds.removeAll()
        adaptationSuggestions.removeAll()
        
        UserDefaults.standard.removeObject(forKey: "adaptiveUI_interactions")
        UserDefaults.standard.removeObject(forKey: "adaptiveUI_patterns")
        UserDefaults.standard.removeObject(forKey: "adaptiveUI_accessibilityNeeds")
    }
}

// MARK: - Supporting Classes

struct AdaptiveTheme {
    let fontSize: CGFloat
    let colorScheme: ColorScheme
    let contrastLevel: ContrastLevel
    let buttonSize: ButtonSize
    let spacingLevel: SpacingLevel
    let animationSpeed: AnimationSpeed
    let navigationStyle: NavigationStyle
    let inputMethod: InputMethod
    let layoutStyle: LayoutStyle
    let accessibilityLevel: AccessibilityLevel
    
    init(
        fontSize: CGFloat = 16,
        colorScheme: ColorScheme = .light,
        contrastLevel: ContrastLevel = .normal,
        buttonSize: ButtonSize = .medium,
        spacingLevel: SpacingLevel = .normal,
        animationSpeed: AnimationSpeed = .normal,
        navigationStyle: NavigationStyle = .standard,
        inputMethod: InputMethod = .touch,
        layoutStyle: LayoutStyle = .standard,
        accessibilityLevel: AccessibilityLevel = .standard
    ) {
        self.fontSize = fontSize
        self.colorScheme = colorScheme
        self.contrastLevel = contrastLevel
        self.buttonSize = buttonSize
        self.spacingLevel = spacingLevel
        self.animationSpeed = animationSpeed
        self.navigationStyle = navigationStyle
        self.inputMethod = inputMethod
        self.layoutStyle = layoutStyle
        self.accessibilityLevel = accessibilityLevel
    }
}

struct AdaptationSuggestion: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let reason: String
    let adaptations: [UIAdaptationType]
    let priority: SuggestionPriority
    let confidence: Double
}

enum SuggestionPriority: Int {
    case low = 1
    case medium = 2
    case high = 3
    case critical = 4
}

// MARK: - Data Transfer Objects

struct UserInteractionData: Codable {
    let timestamp: Date
    let action: String
    let screen: String
    let duration: TimeInterval
    let success: Bool
    let difficulty: Int
    
    init(from interaction: UserInteraction) {
        self.timestamp = interaction.timestamp
        self.action = interaction.action
        self.screen = interaction.screen
        self.duration = interaction.duration
        self.success = interaction.success
        self.difficulty = interaction.difficulty.rawValue
    }
    
    func toUserInteraction() -> UserInteraction {
        return UserInteraction(
            timestamp: timestamp,
            action: action,
            screen: screen,
            duration: duration,
            success: success,
            difficulty: DifficultyLevel(rawValue: difficulty) ?? .easy,
            context: [:]
        )
    }
}

struct UsagePatternData: Codable {
    let feature: String
    let frequency: Double
    let averageDuration: TimeInterval
    let successRate: Double
    let context: String
    
    init(from pattern: UsagePattern) {
        self.feature = pattern.feature
        self.frequency = pattern.frequency
        self.averageDuration = pattern.averageDuration
        self.successRate = pattern.successRate
        self.context = pattern.context
    }
    
    func toUsagePattern() -> UsagePattern {
        return UsagePattern(
            feature: feature,
            frequency: frequency,
            averageDuration: averageDuration,
            successRate: successRate,
            preferredTime: DateComponents(),
            context: context
        )
    }
}

struct AccessibilityNeedData: Codable {
    let type: String
    let severity: Int
    let detectedAt: Date
    
    init(from need: AccessibilityNeed) {
        self.type = String(describing: need.type)
        self.severity = need.severity.rawValue
        self.detectedAt = need.detectedAt
    }
    
    func toAccessibilityNeed() -> AccessibilityNeed {
        let accessibilityType: AccessibilityType
        switch type {
        case "visualImpairment": accessibilityType = .visualImpairment
        case "motorImpairment": accessibilityType = .motorImpairment
        case "cognitiveImpairment": accessibilityType = .cognitiveImpairment
        case "hearingImpairment": accessibilityType = .hearingImpairment
        default: accessibilityType = .temporaryImpairment
        }
        
        return AccessibilityNeed(
            type: accessibilityType,
            severity: AccessibilitySeverity(rawValue: severity) ?? .mild,
            adaptations: [],
            detectedAt: detectedAt
        )
    }
}