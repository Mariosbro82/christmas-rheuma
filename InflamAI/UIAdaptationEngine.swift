//
//  UIAdaptationEngine.swift
//  InflamAI-Swift
//
//  Supporting classes for the Adaptive UI System
//

import Foundation
import SwiftUI
import UIKit
import CoreML
import NaturalLanguage

// MARK: - User Behavior Analyzer

class UserBehaviorAnalyzer {
    
    func analyzePattern(for feature: String, interactions: [UserInteraction]) -> UsagePattern {
        guard !interactions.isEmpty else {
            return UsagePattern(
                feature: feature,
                frequency: 0,
                averageDuration: 0,
                successRate: 0,
                preferredTime: DateComponents(),
                context: "No data"
            )
        }
        
        // Calculate frequency (interactions per day)
        let timeSpan = interactions.last!.timestamp.timeIntervalSince(interactions.first!.timestamp)
        let days = max(timeSpan / 86400, 1) // At least 1 day
        let frequency = Double(interactions.count) / days
        
        // Calculate average duration
        let totalDuration = interactions.reduce(0) { $0 + $1.duration }
        let averageDuration = totalDuration / Double(interactions.count)
        
        // Calculate success rate
        let successfulInteractions = interactions.filter { $0.success }.count
        let successRate = Double(successfulInteractions) / Double(interactions.count)
        
        // Find preferred time
        let preferredTime = findPreferredTime(interactions: interactions)
        
        // Determine context
        let context = determineContext(interactions: interactions)
        
        return UsagePattern(
            feature: feature,
            frequency: frequency,
            averageDuration: averageDuration,
            successRate: successRate,
            preferredTime: preferredTime,
            context: context
        )
    }
    
    private func findPreferredTime(interactions: [UserInteraction]) -> DateComponents {
        let calendar = Calendar.current
        let hourCounts = Dictionary(grouping: interactions) { interaction in
            calendar.component(.hour, from: interaction.timestamp)
        }.mapValues { $0.count }
        
        let mostCommonHour = hourCounts.max(by: { $0.value < $1.value })?.key ?? 12
        
        var components = DateComponents()
        components.hour = mostCommonHour
        return components
    }
    
    private func determineContext(interactions: [UserInteraction]) -> String {
        let difficultyLevels = interactions.map { $0.difficulty.rawValue }
        let averageDifficulty = Double(difficultyLevels.reduce(0, +)) / Double(difficultyLevels.count)
        
        let successRate = Double(interactions.filter { $0.success }.count) / Double(interactions.count)
        
        if averageDifficulty >= 3.0 {
            return "High difficulty"
        } else if successRate < 0.7 {
            return "Low success rate"
        } else if successRate > 0.95 {
            return "High proficiency"
        } else {
            return "Normal usage"
        }
    }
    
    func detectUsabilityIssues(interactions: [UserInteraction]) -> [UsabilityIssue] {
        var issues: [UsabilityIssue] = []
        
        // Group by screen and action
        let groupedInteractions = Dictionary(grouping: interactions) { "\($0.screen)_\($0.action)" }
        
        for (key, screenInteractions) in groupedInteractions {
            let components = key.split(separator: "_")
            guard components.count == 2 else { continue }
            
            let screen = String(components[0])
            let action = String(components[1])
            
            // Check for high failure rate
            let failureRate = 1.0 - (Double(screenInteractions.filter { $0.success }.count) / Double(screenInteractions.count))
            if failureRate > 0.3 {
                issues.append(UsabilityIssue(
                    type: .highFailureRate,
                    screen: screen,
                    action: action,
                    severity: failureRate > 0.5 ? .high : .medium,
                    description: "High failure rate (\(Int(failureRate * 100))%) for \(action) on \(screen)",
                    suggestedFixes: generateFailureRateFixes(action: action, failureRate: failureRate)
                ))
            }
            
            // Check for excessive duration
            let averageDuration = screenInteractions.reduce(0) { $0 + $1.duration } / Double(screenInteractions.count)
            if averageDuration > 10.0 && action != "screen_view" {
                issues.append(UsabilityIssue(
                    type: .excessiveDuration,
                    screen: screen,
                    action: action,
                    severity: averageDuration > 20.0 ? .high : .medium,
                    description: "Excessive duration (\(Int(averageDuration))s) for \(action) on \(screen)",
                    suggestedFixes: generateDurationFixes(action: action, duration: averageDuration)
                ))
            }
            
            // Check for high difficulty
            let averageDifficulty = screenInteractions.reduce(0) { $0 + Double($1.difficulty.rawValue) } / Double(screenInteractions.count)
            if averageDifficulty > 2.5 {
                issues.append(UsabilityIssue(
                    type: .highDifficulty,
                    screen: screen,
                    action: action,
                    severity: averageDifficulty > 3.0 ? .high : .medium,
                    description: "High difficulty (\(String(format: "%.1f", averageDifficulty))/4) for \(action) on \(screen)",
                    suggestedFixes: generateDifficultyFixes(action: action, difficulty: averageDifficulty)
                ))
            }
        }
        
        return issues
    }
    
    private func generateFailureRateFixes(action: String, failureRate: Double) -> [UIAdaptationType] {
        var fixes: [UIAdaptationType] = []
        
        if action.contains("button") || action.contains("tap") {
            fixes.append(.buttonSize(.large))
            fixes.append(.spacing(.comfortable))
        }
        
        if failureRate > 0.5 {
            fixes.append(.navigationStyle(.simplified))
            fixes.append(.animationSpeed(.slow))
        }
        
        return fixes
    }
    
    private func generateDurationFixes(action: String, duration: TimeInterval) -> [UIAdaptationType] {
        var fixes: [UIAdaptationType] = []
        
        if duration > 15.0 {
            fixes.append(.navigationStyle(.simplified))
            fixes.append(.layout(.singleColumn))
        }
        
        if action.contains("input") || action.contains("form") {
            fixes.append(.fontSize(18))
            fixes.append(.buttonSize(.large))
        }
        
        return fixes
    }
    
    private func generateDifficultyFixes(action: String, difficulty: Double) -> [UIAdaptationType] {
        var fixes: [UIAdaptationType] = []
        
        if difficulty > 3.0 {
            fixes.append(.buttonSize(.extraLarge))
            fixes.append(.spacing(.spacious))
            fixes.append(.fontSize(20))
            fixes.append(.contrast(.high))
        } else {
            fixes.append(.buttonSize(.large))
            fixes.append(.spacing(.comfortable))
            fixes.append(.fontSize(18))
        }
        
        return fixes
    }
}

// MARK: - Accessibility Needs Detector

class AccessibilityNeedsDetector {
    
    func detectNeeds(from interactions: [UserInteraction]) -> [AccessibilityNeed] {
        var detectedNeeds: [AccessibilityNeed] = []
        
        // Analyze interaction patterns for accessibility indicators
        detectedNeeds.append(contentsOf: detectVisualImpairment(interactions: interactions))
        detectedNeeds.append(contentsOf: detectMotorImpairment(interactions: interactions))
        detectedNeeds.append(contentsOf: detectCognitiveImpairment(interactions: interactions))
        detectedNeeds.append(contentsOf: detectTemporaryImpairment(interactions: interactions))
        
        return detectedNeeds
    }
    
    private func detectVisualImpairment(interactions: [UserInteraction]) -> [AccessibilityNeed] {
        var needs: [AccessibilityNeed] = []
        
        // Look for patterns indicating visual difficulties
        let gestureFailures = interactions.filter { $0.action.contains("gesture_failure") }
        let highFailureRate = Double(gestureFailures.count) / Double(interactions.count) > 0.2
        
        let longDurations = interactions.filter { $0.duration > 15.0 && $0.action != "screen_view" }
        let frequentLongDurations = Double(longDurations.count) / Double(interactions.count) > 0.3
        
        if highFailureRate || frequentLongDurations {
            let severity: AccessibilitySeverity = (highFailureRate && frequentLongDurations) ? .severe : .moderate
            
            needs.append(AccessibilityNeed(
                type: .visualImpairment,
                severity: severity,
                adaptations: [
                    .fontSize(severity == .severe ? 22 : 18),
                    .contrast(severity == .severe ? .maximum : .high),
                    .buttonSize(severity == .severe ? .extraLarge : .large),
                    .spacing(.comfortable)
                ],
                detectedAt: Date()
            ))
        }
        
        return needs
    }
    
    private func detectMotorImpairment(interactions: [UserInteraction]) -> [AccessibilityNeed] {
        var needs: [AccessibilityNeed] = []
        
        // Look for patterns indicating motor difficulties
        let gestureFailures = interactions.filter { 
            $0.action.contains("gesture_failure") && 
            ($0.context["gesture"] as? String)?.contains("swipe") == true ||
            ($0.context["gesture"] as? String)?.contains("pinch") == true
        }
        
        let multipleAttempts = interactions.filter {
            ($0.context["attempts"] as? Int ?? 1) > 2
        }
        
        if !gestureFailures.isEmpty || !multipleAttempts.isEmpty {
            let severity: AccessibilitySeverity = gestureFailures.count > 5 ? .severe : .moderate
            
            needs.append(AccessibilityNeed(
                type: .motorImpairment,
                severity: severity,
                adaptations: [
                    .buttonSize(.extraLarge),
                    .spacing(.spacious),
                    .navigationStyle(.simplified),
                    .animationSpeed(.slow)
                ],
                detectedAt: Date()
            ))
        }
        
        return needs
    }
    
    private func detectCognitiveImpairment(interactions: [UserInteraction]) -> [AccessibilityNeed] {
        var needs: [AccessibilityNeed] = []
        
        // Look for patterns indicating cognitive difficulties
        let screenSwitching = interactions.filter { $0.action == "screen_view" }
        let rapidSwitching = screenSwitching.filter { $0.duration < 3.0 }.count
        let frequentRapidSwitching = Double(rapidSwitching) / Double(screenSwitching.count) > 0.4
        
        let highDifficultyActions = interactions.filter { $0.difficulty.rawValue >= 3 }
        let frequentDifficulty = Double(highDifficultyActions.count) / Double(interactions.count) > 0.3
        
        if frequentRapidSwitching || frequentDifficulty {
            needs.append(AccessibilityNeed(
                type: .cognitiveImpairment,
                severity: (frequentRapidSwitching && frequentDifficulty) ? .severe : .moderate,
                adaptations: [
                    .navigationStyle(.simplified),
                    .animationSpeed(.slow),
                    .layout(.singleColumn),
                    .spacing(.comfortable)
                ],
                detectedAt: Date()
            ))
        }
        
        return needs
    }
    
    private func detectTemporaryImpairment(interactions: [UserInteraction]) -> [AccessibilityNeed] {
        var needs: [AccessibilityNeed] = []
        
        // Look for recent changes in interaction patterns
        let recentInteractions = interactions.filter {
            $0.timestamp > Calendar.current.date(byAdding: .day, value: -3, to: Date()) ?? Date()
        }
        
        let olderInteractions = interactions.filter {
            $0.timestamp <= Calendar.current.date(byAdding: .day, value: -3, to: Date()) ?? Date()
        }
        
        guard !recentInteractions.isEmpty && !olderInteractions.isEmpty else { return needs }
        
        let recentSuccessRate = Double(recentInteractions.filter { $0.success }.count) / Double(recentInteractions.count)
        let olderSuccessRate = Double(olderInteractions.filter { $0.success }.count) / Double(olderInteractions.count)
        
        let significantDecrease = olderSuccessRate - recentSuccessRate > 0.2
        
        if significantDecrease {
            needs.append(AccessibilityNeed(
                type: .temporaryImpairment,
                severity: .moderate,
                adaptations: [
                    .buttonSize(.large),
                    .spacing(.comfortable),
                    .animationSpeed(.slow),
                    .fontSize(18)
                ],
                detectedAt: Date()
            ))
        }
        
        return needs
    }
}

// MARK: - UI Adaptation Engine

class UIAdaptationEngine {
    
    func generateSuggestions(
        patterns: [String: UsagePattern],
        accessibilityNeeds: [AccessibilityNeed],
        currentSettings: [String: Any]
    ) -> [AdaptationSuggestion] {
        var suggestions: [AdaptationSuggestion] = []
        
        // Generate accessibility-based suggestions
        suggestions.append(contentsOf: generateAccessibilitySuggestions(needs: accessibilityNeeds, currentSettings: currentSettings))
        
        // Generate usage pattern-based suggestions
        suggestions.append(contentsOf: generatePatternBasedSuggestions(patterns: patterns, currentSettings: currentSettings))
        
        // Generate performance-based suggestions
        suggestions.append(contentsOf: generatePerformanceSuggestions(patterns: patterns, currentSettings: currentSettings))
        
        // Sort by priority and confidence
        return suggestions.sorted { first, second in
            if first.priority.rawValue != second.priority.rawValue {
                return first.priority.rawValue > second.priority.rawValue
            }
            return first.confidence > second.confidence
        }
    }
    
    func generateAutoAdaptations(
        patterns: [String: UsagePattern],
        accessibilityNeeds: [AccessibilityNeed],
        currentSettings: [String: Any]
    ) -> [UIAdaptationType] {
        var adaptations: [UIAdaptationType] = []
        
        // Auto-apply critical accessibility adaptations
        for need in accessibilityNeeds where need.severity == .severe {
            adaptations.append(contentsOf: need.adaptations)
        }
        
        // Auto-apply high-confidence pattern adaptations
        let suggestions = generateSuggestions(patterns: patterns, accessibilityNeeds: accessibilityNeeds, currentSettings: currentSettings)
        for suggestion in suggestions where suggestion.confidence > 0.8 && suggestion.priority.rawValue >= 3 {
            adaptations.append(contentsOf: suggestion.adaptations)
        }
        
        return Array(Set(adaptations.map { String(describing: $0) })).compactMap { _ in adaptations.first }
    }
    
    private func generateAccessibilitySuggestions(
        needs: [AccessibilityNeed],
        currentSettings: [String: Any]
    ) -> [AdaptationSuggestion] {
        var suggestions: [AdaptationSuggestion] = []
        
        for need in needs {
            let priority: SuggestionPriority = need.severity == .severe ? .critical : (need.severity == .moderate ? .high : .medium)
            let confidence = need.severity == .severe ? 0.95 : (need.severity == .moderate ? 0.8 : 0.6)
            
            let suggestion = AdaptationSuggestion(
                title: getAccessibilityTitle(for: need.type),
                description: getAccessibilityDescription(for: need.type, severity: need.severity),
                reason: "Detected \(need.type) with \(need.severity) severity",
                adaptations: need.adaptations,
                priority: priority,
                confidence: confidence
            )
            
            suggestions.append(suggestion)
        }
        
        return suggestions
    }
    
    private func generatePatternBasedSuggestions(
        patterns: [String: UsagePattern],
        currentSettings: [String: Any]
    ) -> [AdaptationSuggestion] {
        var suggestions: [AdaptationSuggestion] = []
        
        for (feature, pattern) in patterns {
            // Suggest improvements for low success rate features
            if pattern.successRate < 0.7 {
                let adaptations = generateSuccessRateImprovements(for: feature, pattern: pattern)
                if !adaptations.isEmpty {
                    suggestions.append(AdaptationSuggestion(
                        title: "Improve \(feature) Usability",
                        description: "Make \(feature) easier to use",
                        reason: "Low success rate (\(Int(pattern.successRate * 100))%) detected",
                        adaptations: adaptations,
                        priority: .medium,
                        confidence: 0.7
                    ))
                }
            }
            
            // Suggest optimizations for frequently used features
            if pattern.frequency > 5.0 { // More than 5 times per day
                let adaptations = generateFrequencyOptimizations(for: feature, pattern: pattern)
                if !adaptations.isEmpty {
                    suggestions.append(AdaptationSuggestion(
                        title: "Optimize \(feature)",
                        description: "Streamline frequently used feature",
                        reason: "High usage frequency (\(String(format: "%.1f", pattern.frequency)) times/day)",
                        adaptations: adaptations,
                        priority: .low,
                        confidence: 0.6
                    ))
                }
            }
        }
        
        return suggestions
    }
    
    private func generatePerformanceSuggestions(
        patterns: [String: UsagePattern],
        currentSettings: [String: Any]
    ) -> [AdaptationSuggestion] {
        var suggestions: [AdaptationSuggestion] = []
        
        // Analyze overall performance
        let averageSuccessRate = patterns.values.reduce(0) { $0 + $1.successRate } / Double(patterns.count)
        
        if averageSuccessRate < 0.8 {
            suggestions.append(AdaptationSuggestion(
                title: "Improve Overall Usability",
                description: "General interface improvements",
                reason: "Overall success rate is \(Int(averageSuccessRate * 100))%",
                adaptations: [
                    .buttonSize(.large),
                    .spacing(.comfortable),
                    .fontSize(18),
                    .contrast(.high)
                ],
                priority: .medium,
                confidence: 0.7
            ))
        }
        
        return suggestions
    }
    
    private func generateSuccessRateImprovements(for feature: String, pattern: UsagePattern) -> [UIAdaptationType] {
        var adaptations: [UIAdaptationType] = []
        
        if feature.contains("button") || feature.contains("tap") {
            adaptations.append(.buttonSize(.large))
            adaptations.append(.spacing(.comfortable))
        }
        
        if feature.contains("input") || feature.contains("form") {
            adaptations.append(.fontSize(18))
            adaptations.append(.contrast(.high))
        }
        
        if feature.contains("navigation") {
            adaptations.append(.navigationStyle(.simplified))
        }
        
        return adaptations
    }
    
    private func generateFrequencyOptimizations(for feature: String, pattern: UsagePattern) -> [UIAdaptationType] {
        var adaptations: [UIAdaptationType] = []
        
        // For frequently used features, optimize for speed
        if pattern.frequency > 10.0 {
            adaptations.append(.animationSpeed(.fast))
            adaptations.append(.navigationStyle(.simplified))
        }
        
        return adaptations
    }
    
    private func getAccessibilityTitle(for type: AccessibilityType) -> String {
        switch type {
        case .visualImpairment:
            return "Improve Visual Accessibility"
        case .motorImpairment:
            return "Improve Motor Accessibility"
        case .cognitiveImpairment:
            return "Simplify Interface"
        case .hearingImpairment:
            return "Improve Audio Accessibility"
        case .temporaryImpairment:
            return "Temporary Assistance"
        }
    }
    
    private func getAccessibilityDescription(for type: AccessibilityType, severity: AccessibilitySeverity) -> String {
        let severityText = severity == .severe ? "significant" : (severity == .moderate ? "moderate" : "mild")
        
        switch type {
        case .visualImpairment:
            return "Adjust interface for \(severityText) visual needs"
        case .motorImpairment:
            return "Make interface easier to navigate with \(severityText) motor limitations"
        case .cognitiveImpairment:
            return "Simplify interface for \(severityText) cognitive support"
        case .hearingImpairment:
            return "Enhance visual feedback for \(severityText) hearing support"
        case .temporaryImpairment:
            return "Provide temporary assistance for current needs"
        }
    }
}

// MARK: - Supporting Structures

struct UsabilityIssue {
    let type: UsabilityIssueType
    let screen: String
    let action: String
    let severity: IssueSeverity
    let description: String
    let suggestedFixes: [UIAdaptationType]
}

enum UsabilityIssueType {
    case highFailureRate
    case excessiveDuration
    case highDifficulty
    case frequentErrors
    case poorPerformance
}

enum IssueSeverity {
    case low
    case medium
    case high
    case critical
}

// MARK: - Machine Learning Integration

class AdaptationMLModel {
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // In a real implementation, this would load a trained Core ML model
        // For now, we'll use rule-based logic
    }
    
    func predictOptimalSettings(
        userProfile: UserProfile,
        usagePatterns: [String: UsagePattern],
        accessibilityNeeds: [AccessibilityNeed]
    ) -> [UIAdaptationType] {
        // This would use the ML model to predict optimal settings
        // For now, return rule-based predictions
        return generateRuleBasedPredictions(userProfile: userProfile, usagePatterns: usagePatterns, accessibilityNeeds: accessibilityNeeds)
    }
    
    private func generateRuleBasedPredictions(
        userProfile: UserProfile,
        usagePatterns: [String: UsagePattern],
        accessibilityNeeds: [AccessibilityNeed]
    ) -> [UIAdaptationType] {
        var predictions: [UIAdaptationType] = []
        
        // Age-based adaptations
        if userProfile.age > 65 {
            predictions.append(.fontSize(20))
            predictions.append(.buttonSize(.large))
            predictions.append(.contrast(.high))
        }
        
        // Health condition-based adaptations
        if userProfile.hasRheumatoidArthritis {
            predictions.append(.buttonSize(.extraLarge))
            predictions.append(.spacing(.spacious))
            predictions.append(.animationSpeed(.slow))
        }
        
        // Usage pattern-based adaptations
        let averageSuccessRate = usagePatterns.values.reduce(0) { $0 + $1.successRate } / Double(usagePatterns.count)
        if averageSuccessRate < 0.8 {
            predictions.append(.navigationStyle(.simplified))
            predictions.append(.layout(.singleColumn))
        }
        
        return predictions
    }
}

struct UserProfile {
    let age: Int
    let hasRheumatoidArthritis: Bool
    let hasVisualImpairment: Bool
    let hasMotorImpairment: Bool
    let preferredLanguage: String
    let experienceLevel: ExperienceLevel
}

enum ExperienceLevel {
    case beginner
    case intermediate
    case advanced
    case expert
}