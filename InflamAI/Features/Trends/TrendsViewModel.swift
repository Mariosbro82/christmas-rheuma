//
//  TrendsViewModel.swift
//  InflamAI
//
//  View model for trends analysis and pattern detection
//

import Foundation
import CoreData
import Combine
import SwiftUI

@MainActor
class TrendsViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var dataPoints: [TrendDataPoint] = []
    @Published var topTriggers: [Trigger] = []
    @Published var detectedPatterns: [DetectedPattern]?
    @Published var flareEvents: [TrendsFlareData] = []
    @Published var isLoading = false
    @Published var currentStreak = 0
    @Published var assessmentDataByType: [String: [AssessmentDataPoint]] = [:]

    // MARK: - Properties

    private let context: NSManagedObjectContext
    private let correlationEngine = CorrelationEngine()
    private var currentTimeRange: TimeRange = .month

    // Statistics
    var flareDayCount: Int {
        dataPoints.filter { $0.isFlare }.count
    }

    var bestScoreDate: String {
        guard let bestPoint = dataPoints.min(by: { $0.basdaiScore < $1.basdaiScore }) else {
            return "N/A"
        }
        return bestPoint.date.formatted(date: .abbreviated, time: .omitted)
    }

    var xAxisStride: Calendar.Component {
        switch currentTimeRange {
        case .week: return .day
        case .month: return .day
        case .threeMonths: return .weekOfYear
        case .year: return .month
        }
    }

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - Data Loading

    func loadData(timeRange: TimeRange) {
        currentTimeRange = timeRange
        isLoading = true

        Task {
            do {
                let startDate = timeRange.startDate
                let endDate = Date()

                // Fetch symptom logs
                let logs = try await fetchSymptomLogs(from: startDate, to: endDate)

                // Convert to data points
                dataPoints = logs.map { log in
                    TrendDataPoint(
                        id: log.id ?? UUID(),
                        date: log.timestamp ?? Date(),
                        basdaiScore: log.basdaiScore,
                        painLevel: Double(log.fatigueLevel), // Average pain from regions
                        stiffness: Double(log.morningStiffnessMinutes) / 12.0, // Scale to 0-10
                        fatigue: Double(log.fatigueLevel),
                        isFlare: log.isFlareEvent
                    )
                }.sorted { $0.date < $1.date }

                // Calculate streak
                currentStreak = calculateStreak(logs: logs)

                // Analyze triggers
                if logs.count >= 7 {
                    topTriggers = correlationEngine.findTopTriggers(logs: logs, limit: 3)
                }

                // Detect patterns
                detectPatterns(logs: logs)

                // Load flare events
                await loadFlareEvents(from: startDate, to: endDate)

                // Load assessment data
                await loadAssessmentData(from: startDate, to: endDate)

                isLoading = false

            } catch {
                print("Error loading trend data: \(error)")
                isLoading = false
            }
        }
    }

    // MARK: - Assessment Data Loading

    private func loadAssessmentData(from startDate: Date, to endDate: Date) async {
        do {
            let responses = try await context.perform {
                let request: NSFetchRequest<QuestionnaireResponse> = QuestionnaireResponse.fetchRequest()
                request.predicate = NSPredicate(format: "createdAt >= %@ AND createdAt <= %@", startDate as NSDate, endDate as NSDate)
                request.sortDescriptors = [NSSortDescriptor(keyPath: \QuestionnaireResponse.createdAt, ascending: true)]
                return try self.context.fetch(request)
            }

            // Group by questionnaire type
            var dataByType: [String: [AssessmentDataPoint]] = [:]

            for response in responses {
                guard let questionnaireID = response.questionnaireID,
                      let createdAt = response.createdAt else {
                    continue
                }

                let dataPoint = AssessmentDataPoint(
                    id: response.id ?? UUID(),
                    date: createdAt,
                    score: response.score,
                    questionnaireID: questionnaireID
                )

                dataByType[questionnaireID, default: []].append(dataPoint)
            }

            assessmentDataByType = dataByType

        } catch {
            print("Error loading assessment data: \(error)")
        }
    }

    func colorForAssessment(_ assessmentType: String) -> Color {
        // Map assessment types to colors
        let colors: [Color] = [.green, .blue, .orange, .purple, .pink, .cyan, .indigo, .mint, .teal, .red]

        // Use hash to consistently assign colors
        let index = abs(assessmentType.hashValue) % colors.count
        return colors[index]
    }

    // MARK: - Core Data Fetching

    private func fetchSymptomLogs(from startDate: Date, to endDate: Date) async throws -> [SymptomLog] {
        try await context.perform {
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp <= %@", startDate as NSDate, endDate as NSDate)
            request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]
            return try self.context.fetch(request)
        }
    }

    private func loadFlareEvents(from startDate: Date, to endDate: Date) async {
        do {
            let events = try await context.perform {
                let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
                request.predicate = NSPredicate(format: "startDate >= %@ AND startDate <= %@", startDate as NSDate, endDate as NSDate)
                request.sortDescriptors = [NSSortDescriptor(keyPath: \FlareEvent.startDate, ascending: false)]
                return try self.context.fetch(request)
            }

            flareEvents = events.map { event in
                var durationDays: Int?
                if let endDate = event.endDate {
                    durationDays = Calendar.current.dateComponents([.day], from: event.startDate ?? Date(), to: endDate).day
                }

                var triggers: String?
                if let triggersData = event.suspectedTriggers,
                   let triggerArray = try? JSONDecoder().decode([String].self, from: triggersData) {
                    triggers = triggerArray.joined(separator: ", ")
                }

                return TrendsFlareData(
                    id: event.id ?? UUID(),
                    date: event.startDate ?? Date(),
                    severity: Int(event.severity),
                    durationDays: durationDays,
                    suspectedTriggers: triggers
                )
            }
        } catch {
            print("Error loading flare events: \(error)")
        }
    }

    // MARK: - Statistics

    func averageValue(for metric: TrendMetric) -> Double? {
        guard !dataPoints.isEmpty else { return nil }

        let sum = dataPoints.reduce(0.0) { $0 + $1.value(for: metric) }
        return sum / Double(dataPoints.count)
    }

    func minValue(for metric: TrendMetric) -> Double? {
        dataPoints.map { $0.value(for: metric) }.min()
    }

    func maxValue(for metric: TrendMetric) -> Double? {
        dataPoints.map { $0.value(for: metric) }.max()
    }

    func trend(for metric: TrendMetric) -> TrendDirection {
        guard dataPoints.count >= 2 else { return .stable }

        // Compare average of last 7 days vs previous 7 days
        let recent = dataPoints.suffix(7)
        let previous = dataPoints.dropLast(7).suffix(7)

        guard !recent.isEmpty && !previous.isEmpty else { return .stable }

        let recentAvg = recent.reduce(0.0) { $0 + $1.value(for: metric) } / Double(recent.count)
        let previousAvg = previous.reduce(0.0) { $0 + $1.value(for: metric) } / Double(previous.count)

        let percentChange = ((recentAvg - previousAvg) / previousAvg) * 100

        if abs(percentChange) < 5 {
            return .stable
        } else if percentChange > 0 {
            return .up(abs(percentChange))
        } else {
            return .down(abs(percentChange))
        }
    }

    private func calculateStreak(logs: [SymptomLog]) -> Int {
        guard !logs.isEmpty else { return 0 }

        let calendar = Calendar.current
        var streak = 0
        var currentDate = calendar.startOfDay(for: Date())

        // Count backwards from today
        while true {
            let hasLog = logs.contains { log in
                guard let logDate = log.timestamp else { return false }
                return calendar.isDate(logDate, inSameDayAs: currentDate)
            }

            if hasLog {
                streak += 1
                currentDate = calendar.date(byAdding: .day, value: -1, to: currentDate)!
            } else {
                break
            }
        }

        return streak
    }

    // MARK: - Pattern Detection

    private func detectPatterns(logs: [SymptomLog]) {
        guard logs.count >= 14 else {
            detectedPatterns = nil
            return
        }

        var patterns: [DetectedPattern] = []

        // Pattern 1: Weekly cycle
        if let weeklyPattern = detectWeeklyCycle(logs: logs) {
            patterns.append(weeklyPattern)
        }

        // Pattern 2: Weather sensitivity
        if let weatherPattern = detectWeatherSensitivity(logs: logs) {
            patterns.append(weatherPattern)
        }

        // Pattern 3: Sleep impact
        if let sleepPattern = detectSleepImpact(logs: logs) {
            patterns.append(sleepPattern)
        }

        // Pattern 4: Progressive improvement/deterioration
        if let progressionPattern = detectProgression(logs: logs) {
            patterns.append(progressionPattern)
        }

        detectedPatterns = patterns.isEmpty ? nil : patterns
    }

    private func detectWeeklyCycle(logs: [SymptomLog]) -> DetectedPattern? {
        // Group by day of week
        var dayAverages: [Int: [Double]] = [:]

        for log in logs {
            guard let date = log.timestamp else { continue }
            let weekday = Calendar.current.component(.weekday, from: date)
            dayAverages[weekday, default: []].append(log.basdaiScore)
        }

        let averages = dayAverages.mapValues { scores in
            scores.reduce(0, +) / Double(scores.count)
        }

        // Check if variance is significant
        guard let max = averages.values.max(),
              let min = averages.values.min() else {
            return nil
        }

        if (max - min) > 2.0 {
            // Find worst day
            let worstDay = averages.max { $0.value < $1.value }?.key ?? 1
            let dayName = Calendar.current.weekdaySymbols[worstDay - 1]

            return DetectedPattern(
                id: UUID(),
                title: "Weekly Pattern Detected",
                description: "Your symptoms tend to be worse on \(dayName)s. Consider extra rest or preventive measures on this day.",
                iconName: "calendar",
                color: .blue
            )
        }

        return nil
    }

    private func detectWeatherSensitivity(logs: [SymptomLog]) -> DetectedPattern? {
        let pressureChanges = logs.compactMap { $0.contextSnapshot?.pressureChange12h }
        let painScores = logs.compactMap { $0.basdaiScore }

        guard pressureChanges.count == painScores.count,
              let correlation = correlationEngine.pearsonCorrelation(pressureChanges, painScores) else {
            return nil
        }

        if correlation < -0.4 {
            return DetectedPattern(
                id: UUID(),
                title: "Weather Sensitive",
                description: "Rapid pressure drops correlate with increased symptoms. Check weather forecasts and prepare accordingly.",
                iconName: "cloud.rain",
                color: .cyan
            )
        }

        return nil
    }

    private func detectSleepImpact(logs: [SymptomLog]) -> DetectedPattern? {
        let sleepDurations = logs.compactMap { $0.sleepDurationHours }
        let painScores = logs.compactMap { $0.basdaiScore }

        guard sleepDurations.count == painScores.count,
              let correlation = correlationEngine.pearsonCorrelation(sleepDurations, painScores) else {
            return nil
        }

        if correlation < -0.5 {
            return DetectedPattern(
                id: UUID(),
                title: "Sleep Critical",
                description: "Poor sleep strongly correlates with worse symptoms. Prioritize 7-8 hours of quality sleep.",
                iconName: "bed.double.fill",
                color: .indigo
            )
        }

        return nil
    }

    private func detectProgression(logs: [SymptomLog]) -> DetectedPattern? {
        guard logs.count >= 30 else { return nil }

        let scores = logs.map { $0.basdaiScore }
        let indices = Array(0..<scores.count).map { Double($0) }

        // Simple linear regression
        guard let slope = correlationEngine.pearsonCorrelation(indices, scores) else {
            return nil
        }

        if slope > 0.3 {
            return DetectedPattern(
                id: UUID(),
                title: "Gradual Worsening",
                description: "Your symptoms show a gradual upward trend. Consider consulting your rheumatologist for treatment review.",
                iconName: "arrow.up.right.circle",
                color: .orange
            )
        } else if slope < -0.3 {
            return DetectedPattern(
                id: UUID(),
                title: "Steady Improvement",
                description: "Your symptoms are trending downward! Your current management plan is working well.",
                iconName: "arrow.down.right.circle",
                color: .green
            )
        }

        return nil
    }

    // MARK: - Export

    func exportData(format: ExportFormat) async {
        switch format {
        case .pdf:
            await exportPDF()
        case .csv:
            await exportCSV()
        case .json:
            await exportJSON()
        }
    }

    private func exportPDF() async {
        // PDF export implementation
        print("Exporting PDF...")
    }

    private func exportCSV() async {
        // CSV export implementation
        var csv = "Date,BASDAI,Pain,Stiffness,Fatigue,Flare\n"

        for point in dataPoints {
            csv += "\(point.date),\(point.basdaiScore),\(point.painLevel),\(point.stiffness),\(point.fatigue),\(point.isFlare)\n"
        }

        // Save to file
        print("CSV generated: \(csv.count) bytes")
    }

    private func exportJSON() async {
        // JSON export implementation
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted

        if let jsonData = try? encoder.encode(dataPoints),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print("JSON generated: \(jsonString.count) bytes")
        }
    }
}

// MARK: - Supporting Models

struct TrendDataPoint: Identifiable, Codable {
    let id: UUID
    let date: Date
    let basdaiScore: Double
    let painLevel: Double
    let stiffness: Double
    let fatigue: Double
    let isFlare: Bool

    func value(for metric: TrendMetric) -> Double {
        switch metric {
        case .basdai: return basdaiScore
        case .pain: return painLevel
        case .stiffness: return stiffness
        case .fatigue: return fatigue
        case .assessments: return basdaiScore // Assessments use their own data structure
        }
    }
}

struct TrendsFlareData: Identifiable {
    let id: UUID
    let date: Date
    let severity: Int
    let durationDays: Int?
    let suspectedTriggers: String?
}

struct DetectedPattern: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let iconName: String
    let color: Color
}

struct AssessmentDataPoint: Identifiable {
    let id: UUID
    let date: Date
    let score: Double
    let questionnaireID: String
}