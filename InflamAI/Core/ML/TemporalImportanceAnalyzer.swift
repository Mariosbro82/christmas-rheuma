//
//  TemporalImportanceAnalyzer.swift
//  InflamAI
//
//  Temporal importance analysis for 30-day prediction window
//  Shows which days in the sequence had the most impact on prediction
//

import Foundation
import CoreML

@MainActor
class TemporalImportanceAnalyzer {

    private let model: ASFlarePredictor
    private let scaler: FeatureScaler

    struct DayImportance: Identifiable {
        let id = UUID()
        let dayOffset: Int              // 0 = today, 1 = yesterday, etc.
        let importance: Float           // 0-1 (normalized importance score)
        let date: Date
        let aggregateFeatures: [Float]  // Feature values for that day

        var relativeImportance: String {
            if importance >= 0.8 {
                return "Critical"
            } else if importance >= 0.6 {
                return "High"
            } else if importance >= 0.4 {
                return "Moderate"
            } else if importance >= 0.2 {
                return "Low"
            } else {
                return "Minimal"
            }
        }

        var color: String {
            if importance >= 0.8 { return "red" }
            if importance >= 0.6 { return "orange" }
            if importance >= 0.4 { return "yellow" }
            if importance >= 0.2 { return "blue" }
            return "gray"
        }

        var description: String {
            let formatter = DateFormatter()
            formatter.dateFormat = "MMM d"
            let dateStr = formatter.string(from: date)

            return "\(dateStr) (\(dayOffset) days ago) - \(relativeImportance) impact"
        }
    }

    struct TemporalPattern {
        let recentDaysBias: Float       // 0-1: how much recent days dominate
        let longerTermInfluence: Float  // 0-1: how much older days matter
        let peakImportanceDayOffset: Int
        let pattern: PatternType

        enum PatternType {
            case recentFocused          // Most important days are recent (0-7)
            case balanced               // Importance spread across window
            case historicalFocused      // Older days (14-30) more important
            case spikePattern           // One or two days dominate

            var description: String {
                switch self {
                case .recentFocused:
                    return "Recent days (last week) have the strongest influence"
                case .balanced:
                    return "Importance is balanced across the entire window"
                case .historicalFocused:
                    return "Historical patterns (2-4 weeks ago) drive prediction"
                case .spikePattern:
                    return "One specific day has unusually high impact"
                }
            }

            var icon: String {
                switch self {
                case .recentFocused: return "clock.fill"
                case .balanced: return "chart.bar.fill"
                case .historicalFocused: return "calendar.badge.clock"
                case .spikePattern: return "exclamationmark.triangle.fill"
                }
            }
        }
    }

    init(model: ASFlarePredictor, scaler: FeatureScaler) {
        self.model = model
        self.scaler = scaler
    }

    // MARK: - Temporal Importance Analysis

    /// Compute importance of each day in the 30-day window
    /// Uses leave-one-out perturbation: remove each day and measure impact
    func analyzeDayImportance(
        features: [[Float]],
        currentDate: Date
    ) async throws -> [DayImportance] {

        guard features.count == 30 else {
            throw AnalysisError.invalidSequenceLength(expected: 30, got: features.count)
        }

        // Get baseline prediction
        let baselineProbability = try await getPrediction(features: features)

        var dayImportances: [DayImportance] = []
        let calendar = Calendar.current

        // For each day in the sequence
        for dayOffset in 0..<30 {
            // Create modified sequence where this day is zeroed out
            var perturbedFeatures = features
            perturbedFeatures[29 - dayOffset] = Array(repeating: 0.0, count: 92)  // Neutral values

            // Get perturbed prediction
            let perturbedProbability = try await getPrediction(features: perturbedFeatures)

            // Importance = how much prediction changed when day removed
            let importanceScore = abs(baselineProbability - perturbedProbability)

            // Calculate date for this day
            let date = calendar.date(byAdding: .day, value: -dayOffset, to: currentDate)!

            dayImportances.append(DayImportance(
                dayOffset: dayOffset,
                importance: importanceScore,
                date: date,
                aggregateFeatures: features[29 - dayOffset]
            ))
        }

        // Normalize importance scores to 0-1 range
        let maxImportance = dayImportances.map { $0.importance }.max() ?? 1.0
        if maxImportance > 0 {
            for i in 0..<dayImportances.count {
                let normalized = dayImportances[i].importance / maxImportance
                dayImportances[i] = DayImportance(
                    dayOffset: dayImportances[i].dayOffset,
                    importance: normalized,
                    date: dayImportances[i].date,
                    aggregateFeatures: dayImportances[i].aggregateFeatures
                )
            }
        }

        return dayImportances
    }

    /// Identify the temporal pattern (recent vs historical focus)
    func identifyTemporalPattern(
        dayImportances: [DayImportance]
    ) -> TemporalPattern {

        // Split into recent (0-7 days) and longer-term (8-30 days)
        let recentImportance = dayImportances
            .filter { $0.dayOffset <= 7 }
            .map { $0.importance }
            .reduce(0, +)

        let longerTermImportance = dayImportances
            .filter { $0.dayOffset > 7 }
            .map { $0.importance }
            .reduce(0, +)

        let totalImportance = recentImportance + longerTermImportance

        let recentBias = totalImportance > 0 ? recentImportance / totalImportance : 0.5
        let longerTermInfluence = totalImportance > 0 ? longerTermImportance / totalImportance : 0.5

        // Find peak importance day
        let peakDay = dayImportances.max(by: { $0.importance < $1.importance })?.dayOffset ?? 0

        // Determine pattern type
        let pattern: TemporalPattern.PatternType

        // Check for spike pattern (one day has >60% of total importance)
        let peakImportance = dayImportances.max(by: { $0.importance < $1.importance })?.importance ?? 0
        if peakImportance > 0.6 {
            pattern = .spikePattern
        } else if recentBias > 0.65 {
            pattern = .recentFocused
        } else if longerTermInfluence > 0.65 {
            pattern = .historicalFocused
        } else {
            pattern = .balanced
        }

        return TemporalPattern(
            recentDaysBias: recentBias,
            longerTermInfluence: longerTermInfluence,
            peakImportanceDayOffset: peakDay,
            pattern: pattern
        )
    }

    /// Get top N most important days
    func getTopImportantDays(
        dayImportances: [DayImportance],
        topN: Int = 5
    ) -> [DayImportance] {
        return dayImportances
            .sorted { $0.importance > $1.importance }
            .prefix(topN)
            .map { $0 }
    }

    /// Generate natural language explanation of temporal pattern
    func explainTemporalPattern(
        dayImportances: [DayImportance],
        pattern: TemporalPattern
    ) -> String {
        let topDays = getTopImportantDays(dayImportances: dayImportances, topN: 3)

        var explanation = "\(pattern.pattern.description)\n\n"
        explanation += "Most influential days:\n"

        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"

        for (index, day) in topDays.enumerated() {
            let dateStr = formatter.string(from: day.date)
            explanation += "\(index + 1). \(dateStr) (\(day.dayOffset) days ago) - \(String(format: "%.0f%%", day.importance * 100)) impact\n"
        }

        // Add contextual advice
        switch pattern.pattern {
        case .recentFocused:
            explanation += "\n💡 Focus on current habits - recent changes matter most."
        case .balanced:
            explanation += "\n💡 Your patterns span weeks - consistency is key."
        case .historicalFocused:
            explanation += "\n💡 Past trends are catching up - review last 2-4 weeks."
        case .spikePattern:
            let spikeDay = formatter.string(from: topDays.first!.date)
            explanation += "\n⚠️ Unusual spike on \(spikeDay) - investigate what happened."
        }

        return explanation
    }

    // MARK: - Visualization Data

    /// Generate data for chart visualization
    func generateChartData(
        dayImportances: [DayImportance]
    ) -> [(x: Int, y: Float, date: Date)] {
        return dayImportances.map { day in
            (x: day.dayOffset, y: day.importance, date: day.date)
        }
    }

    /// Generate heatmap data (7x4 grid for 28 days + 2 extras)
    func generateHeatmapData(
        dayImportances: [DayImportance]
    ) -> [[Float]] {
        var heatmap: [[Float]] = Array(repeating: Array(repeating: 0.0, count: 7), count: 5)

        for day in dayImportances.prefix(35) {  // Up to 35 days (5 weeks)
            let row = day.dayOffset / 7
            let col = day.dayOffset % 7

            if row < 5 && col < 7 {
                heatmap[row][col] = day.importance
            }
        }

        return heatmap
    }

    // MARK: - Feature Aggregation for Day

    /// Get summary of key features for a specific day
    func getKeyFeaturesForDay(
        dayImportance: DayImportance,
        featureNames: [String]
    ) -> [(feature: String, value: Float)] {
        let features = dayImportance.aggregateFeatures

        // Get top 5 non-zero features
        var featureValues: [(String, Float)] = []

        for (index, value) in features.enumerated() where abs(value) > 0.01 {
            let name = index < featureNames.count ? featureNames[index] : "Feature \(index)"
            featureValues.append((name, value))
        }

        // Sort by absolute value
        featureValues.sort { abs($0.1) > abs($1.1) }

        return Array(featureValues.prefix(5))
    }

    // MARK: - Helper Methods

    private func getPrediction(features: [[Float]]) async throws -> Float {
        // Normalize each day's features separately
        let normalizedFeatures = features.map { dayFeatures in
            scaler.transform(dayFeatures)
        }

        // Create MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 30, 92], dataType: .float32)
        for (i, timestep) in normalizedFeatures.enumerated() {
            for (j, value) in timestep.enumerated() {
                inputArray[[0, i, j] as [NSNumber]] = NSNumber(value: value)
            }
        }

        // Run prediction
        let input = ASFlarePredictorInput(features: inputArray)
        let output = try await model.prediction(input: input)

        // Extract probability
        let probabilities = output.probabilities
        return probabilities[1].floatValue
    }

    // MARK: - Errors

    enum AnalysisError: LocalizedError {
        case invalidSequenceLength(expected: Int, got: Int)
        case insufficientData

        var errorDescription: String? {
            switch self {
            case .invalidSequenceLength(let expected, let got):
                return "Expected sequence length \(expected), got \(got)"
            case .insufficientData:
                return "Insufficient data for temporal analysis"
            }
        }
    }
}

// MARK: - Visualization Helpers

extension TemporalImportanceAnalyzer {

    /// Generate human-readable timeline summary
    func generateTimelineSummary(
        dayImportances: [DayImportance]
    ) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"

        // Group by week
        var weekSummaries: [String] = []

        for weekIndex in 0..<5 {  // 5 weeks (35 days)
            let weekDays = dayImportances.filter { day in
                day.dayOffset >= weekIndex * 7 && day.dayOffset < (weekIndex + 1) * 7
            }

            guard !weekDays.isEmpty else { continue }

            let avgImportance = weekDays.map { $0.importance }.reduce(0, +) / Float(weekDays.count)
            let weekStart = weekDays.first!.date
            let weekLabel = weekIndex == 0 ? "This week" : "\(weekIndex + 1) weeks ago"

            let summary = "\(weekLabel): \(String(format: "%.0f%%", avgImportance * 100)) avg impact"
            weekSummaries.append(summary)
        }

        return weekSummaries.joined(separator: "\n")
    }

    /// Detect anomalies in temporal importance
    func detectAnomalies(
        dayImportances: [DayImportance]
    ) -> [String] {
        var anomalies: [String] = []

        // Check for sudden spikes
        for i in 1..<dayImportances.count {
            let current = dayImportances[i].importance
            let previous = dayImportances[i - 1].importance

            if current > 0.8 && previous < 0.3 {
                let formatter = DateFormatter()
                formatter.dateFormat = "MMM d"
                let dateStr = formatter.string(from: dayImportances[i].date)
                anomalies.append("Sudden spike on \(dateStr) - investigate triggers")
            }
        }

        // Check for gaps (consecutive days with near-zero importance)
        var gapStart: Int?
        for i in 0..<dayImportances.count {
            if dayImportances[i].importance < 0.1 {
                if gapStart == nil {
                    gapStart = i
                }
            } else {
                if let start = gapStart, i - start >= 3 {
                    anomalies.append("Low data quality: \(i - start) days with minimal impact")
                }
                gapStart = nil
            }
        }

        return anomalies
    }
}
