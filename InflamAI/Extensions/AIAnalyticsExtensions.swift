//
//  AIAnalyticsExtensions.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI

// MARK: - HealthPredictionModel Extensions

extension HealthPredictionModel.ModelType {
    var displayName: String {
        switch self {
        case .flareUpPrediction:
            return "Flare-up Prediction"
        case .painIntensityForecast:
            return "Pain Intensity Forecast"
        case .medicationEffectiveness:
            return "Medication Effectiveness"
        case .symptomClassification:
            return "Symptom Classification"
        case .treatmentRecommendation:
            return "Treatment Recommendation"
        case .riskAssessment:
            return "Risk Assessment"
        case .progressPrediction:
            return "Progress Prediction"
        }
    }
    
    var icon: String {
        switch self {
        case .flareUpPrediction:
            return "exclamationmark.triangle.fill"
        case .painIntensityForecast:
            return "waveform.path.ecg"
        case .medicationEffectiveness:
            return "pills.fill"
        case .symptomClassification:
            return "list.bullet.clipboard"
        case .treatmentRecommendation:
            return "stethoscope"
        case .riskAssessment:
            return "shield.fill"
        case .progressPrediction:
            return "chart.line.uptrend.xyaxis"
        }
    }
    
    static var allCases: [HealthPredictionModel.ModelType] {
        return [.flareUpPrediction, .painIntensityForecast, .medicationEffectiveness, .symptomClassification, .treatmentRecommendation, .riskAssessment, .progressPrediction]
    }
}

// MARK: - PredictionResult Extensions

extension PredictionResult.PredictionTimeframe {
    var displayName: String {
        switch self {
        case .shortTerm:
            return "Short Term"
        case .mediumTerm:
            return "Medium Term"
        case .longTerm:
            return "Long Term"
        }
    }
    
    static var allCases: [PredictionResult.PredictionTimeframe] {
        return [.shortTerm, .mediumTerm, .longTerm]
    }
}

// MARK: - HealthInsight Extensions

extension HealthInsight.InsightCategory {
    var displayName: String {
        switch self {
        case .symptomPattern:
            return "Symptom Pattern"
        case .medicationResponse:
            return "Medication Response"
        case .lifestyleImpact:
            return "Lifestyle Impact"
        case .environmentalFactor:
            return "Environmental Factor"
        case .treatmentEffectiveness:
            return "Treatment Effectiveness"
        case .riskIdentification:
            return "Risk Identification"
        case .progressTracking:
            return "Progress Tracking"
        }
    }
    
    var icon: String {
        switch self {
        case .symptomPattern:
            return "waveform.path.ecg"
        case .medicationResponse:
            return "pills.fill"
        case .lifestyleImpact:
            return "figure.walk"
        case .environmentalFactor:
            return "cloud.sun.fill"
        case .treatmentEffectiveness:
            return "stethoscope"
        case .riskIdentification:
            return "exclamationmark.triangle.fill"
        case .progressTracking:
            return "chart.line.uptrend.xyaxis"
        }
    }
    
    static var allCases: [HealthInsight.InsightCategory] {
        return [.symptomPattern, .medicationResponse, .lifestyleImpact, .environmentalFactor, .treatmentEffectiveness, .riskIdentification, .progressTracking]
    }
}

extension HealthInsight.InsightImportance {
    var color: Color {
        switch self {
        case .low:
            return .green
        case .medium:
            return .orange
        case .high:
            return .red
        case .critical:
            return .purple
        }
    }
    
    var displayName: String {
        switch self {
        case .low:
            return "Low"
        case .medium:
            return "Medium"
        case .high:
            return "High"
        case .critical:
            return "Critical"
        }
    }
}

// MARK: - AIRecommendation Extensions

extension AIRecommendation.RecommendationCategory {
    var displayName: String {
        switch self {
        case .medication:
            return "Medication"
        case .lifestyle:
            return "Lifestyle"
        case .exercise:
            return "Exercise"
        case .diet:
            return "Diet"
        case .stress:
            return "Stress Management"
        case .sleep:
            return "Sleep"
        case .medical:
            return "Medical Care"
        }
    }
    
    var icon: String {
        switch self {
        case .medication:
            return "pills.fill"
        case .lifestyle:
            return "figure.walk"
        case .exercise:
            return "figure.run"
        case .diet:
            return "leaf.fill"
        case .stress:
            return "brain.head.profile"
        case .sleep:
            return "bed.double.fill"
        case .medical:
            return "stethoscope"
        }
    }
}

extension AIRecommendation.RecommendationPriority {
    var color: Color {
        switch self {
        case .low:
            return .green
        case .medium:
            return .orange
        case .high:
            return .red
        case .critical:
            return .purple
        }
    }
    
    var displayName: String {
        switch self {
        case .low:
            return "Low"
        case .medium:
            return "Medium"
        case .high:
            return "High"
        case .critical:
            return "Critical"
        }
    }
}

// MARK: - HealthTrend Extensions

extension HealthTrend.TrendDirection {
    var color: Color {
        switch self {
        case .improving:
            return .green
        case .stable:
            return .blue
        case .declining:
            return .red
        case .fluctuating:
            return .orange
        }
    }
    
    var icon: String {
        switch self {
        case .improving:
            return "arrow.up.right"
        case .stable:
            return "arrow.right"
        case .declining:
            return "arrow.down.right"
        case .fluctuating:
            return "arrow.up.and.down"
        }
    }
    
    var displayName: String {
        switch self {
        case .improving:
            return "Improving"
        case .stable:
            return "Stable"
        case .declining:
            return "Declining"
        case .fluctuating:
            return "Fluctuating"
        }
    }
}

// MARK: - HealthCorrelation Extensions

extension HealthCorrelation.CorrelationStrength {
    var displayName: String {
        switch self {
        case .weak:
            return "Weak"
        case .moderate:
            return "Moderate"
        case .strong:
            return "Strong"
        case .veryStrong:
            return "Very Strong"
        }
    }
    
    var color: Color {
        switch self {
        case .weak:
            return .gray
        case .moderate:
            return .orange
        case .strong:
            return .blue
        case .veryStrong:
            return .purple
        }
    }
}

extension HealthCorrelation.CorrelationType {
    var color: Color {
        switch self {
        case .positive:
            return .green
        case .negative:
            return .red
        case .nonLinear:
            return .orange
        }
    }
    
    var displayName: String {
        switch self {
        case .positive:
            return "Positive"
        case .negative:
            return "Negative"
        case .nonLinear:
            return "Non-linear"
        }
    }
}

// MARK: - RiskFactor Extensions

extension RiskFactor.RiskLevel {
    var color: Color {
        switch self {
        case .low:
            return .green
        case .moderate:
            return .yellow
        case .high:
            return .orange
        case .critical:
            return .red
        }
    }
    
    var displayName: String {
        switch self {
        case .low:
            return "Low"
        case .moderate:
            return "Moderate"
        case .high:
            return "High"
        case .critical:
            return "Critical"
        }
    }
    
    static var allCases: [RiskFactor.RiskLevel] {
        return [.low, .moderate, .high, .critical]
    }
}

extension RiskFactor.RiskImpact {
    var displayName: String {
        switch self {
        case .minimal:
            return "Minimal"
        case .minor:
            return "Minor"
        case .moderate:
            return "Moderate"
        case .major:
            return "Major"
        case .severe:
            return "Severe"
        }
    }
    
    var color: Color {
        switch self {
        case .minimal:
            return .green
        case .minor:
            return .yellow
        case .moderate:
            return .orange
        case .major:
            return .red
        case .severe:
            return .purple
        }
    }
}

extension RiskFactor.RiskCategory {
    var displayName: String {
        switch self {
        case .medical:
            return "Medical"
        case .lifestyle:
            return "Lifestyle"
        case .environmental:
            return "Environmental"
        case .genetic:
            return "Genetic"
        case .behavioral:
            return "Behavioral"
        }
    }
    
    var icon: String {
        switch self {
        case .medical:
            return "stethoscope"
        case .lifestyle:
            return "figure.walk"
        case .environmental:
            return "cloud.sun.fill"
        case .genetic:
            return "dna"
        case .behavioral:
            return "brain.head.profile"
        }
    }
}

// MARK: - HealthAnalytics Extensions

extension HealthAnalytics.AnalyticsTimeRange {
    var displayName: String {
        switch self {
        case .week:
            return "Past Week"
        case .month:
            return "Past Month"
        case .threeMonths:
            return "Past 3 Months"
        case .sixMonths:
            return "Past 6 Months"
        case .year:
            return "Past Year"
        case .allTime:
            return "All Time"
        }
    }
    
    var days: Int {
        switch self {
        case .week:
            return 7
        case .month:
            return 30
        case .threeMonths:
            return 90
        case .sixMonths:
            return 180
        case .year:
            return 365
        case .allTime:
            return Int.max
        }
    }
    
    static var allCases: [HealthAnalytics.AnalyticsTimeRange] {
        return [.week, .month, .threeMonths, .sixMonths, .year, .allTime]
    }
}

// MARK: - ProgressMetric Extensions

extension ProgressMetric.MetricCategory {
    var displayName: String {
        switch self {
        case .symptom:
            return "Symptoms"
        case .medication:
            return "Medication"
        case .activity:
            return "Activity"
        case .mood:
            return "Mood"
        case .sleep:
            return "Sleep"
        case .pain:
            return "Pain"
        case .fatigue:
            return "Fatigue"
        }
    }
    
    var icon: String {
        switch self {
        case .symptom:
            return "waveform.path.ecg"
        case .medication:
            return "pills.fill"
        case .activity:
            return "figure.walk"
        case .mood:
            return "face.smiling"
        case .sleep:
            return "bed.double.fill"
        case .pain:
            return "exclamationmark.triangle.fill"
        case .fatigue:
            return "battery.25"
        }
    }
    
    var color: Color {
        switch self {
        case .symptom:
            return .red
        case .medication:
            return .blue
        case .activity:
            return .green
        case .mood:
            return .purple
        case .sleep:
            return .indigo
        case .pain:
            return .orange
        case .fatigue:
            return .yellow
        }
    }
}

// MARK: - String Extensions for UI Formatting (CRIT-002, CRIT-003)

extension String {
    /// Converts snake_case to Title Case for user-facing display (CRIT-003 Fix)
    /// Examples:
    /// - "poor_sleep" → "Poor Sleep"
    /// - "weather_change" → "Weather Change"
    /// - "menstrual_cycle" → "Menstrual Cycle"
    var displayName: String {
        return self
            .replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    /// Provides localized string with fallback to display name if key not found (CRIT-002 Fix)
    /// This prevents raw localization keys from being displayed to users
    var localizedWithFallback: String {
        let localized = NSLocalizedString(self, comment: "")
        if localized == self {
            // Key not found, return formatted version
            #if DEBUG
            print("⚠️ Missing translation for key: \(self)")
            #endif
            return self.displayName
        }
        return localized
    }
}