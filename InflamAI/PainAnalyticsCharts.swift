//
//  PainAnalyticsCharts.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts
import CoreData

struct PainAnalyticsCharts: View {
    let painEntries: [PainEntry]
    let journalEntries: [JournalEntry]
    let bassdaiAssessments: [BASSDAIAssessment]
    let medications: [Medication]
    let medicationIntakes: [MedicationIntake]
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Pain Trend Chart
                PainTrendChart(painEntries: painEntries)
                
                // BASDAI Trend Chart
                BASSDAITrendChart(assessments: bassdaiAssessments)
                
                // Pain vs Mood Correlation
                PainMoodCorrelationChart(journalEntries: journalEntries)
                
                // Medication Adherence Chart
                MedicationAdherenceChart(
                    medications: medications,
                    intakes: medicationIntakes
                )
                
                // Pain Heat Map
                PainHeatMapChart(painEntries: painEntries)
                
                // Weekly Pain Summary
                WeeklyPainSummaryChart(painEntries: painEntries)
            }
            .padding()
        }
    }
}

struct PainTrendChart: View {
    let painEntries: [PainEntry]
    
    private var chartData: [PainDataPoint] {
        painEntries
            .sorted { ($0.timestamp ?? Date.distantPast) < ($1.timestamp ?? Date.distantPast) }
            .map { entry in
                PainDataPoint(
                    date: entry.timestamp ?? Date(),
                    painLevel: Double(entry.painLevel),
                    bodyRegion: entry.bodyRegions ?? "Unknown"
                )
            }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Pain Level Trends")
                .font(.title2)
                .fontWeight(.bold)
            
            if #available(iOS 16.0, *) {
                Chart(chartData, id: \.date) { dataPoint in
                    LineMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Pain Level", dataPoint.painLevel)
                    )
                    .foregroundStyle(.red)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    
                    PointMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Pain Level", dataPoint.painLevel)
                    )
                    .foregroundStyle(.red)
                    .symbolSize(30)
                }
                .frame(height: 200)
                .chartYScale(domain: 0...10)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: 7)) { _ in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel(format: .dateTime.month().day())
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .stride(by: 2)) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel()
                    }
                }
            } else {
                // Fallback for iOS 15
                LegacyPainTrendChart(data: chartData)
            }
            
            // Summary statistics
            HStack(spacing: 20) {
                StatCard(
                    title: "Average",
                    value: String(format: "%.1f", averagePainLevel),
                    color: .blue
                )
                StatCard(
                    title: "Peak",
                    value: String(format: "%.0f", maxPainLevel),
                    color: .red
                )
                StatCard(
                    title: "Entries",
                    value: "\(painEntries.count)",
                    color: .green
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private var averagePainLevel: Double {
        guard !painEntries.isEmpty else { return 0 }
        let total = painEntries.reduce(0) { $0 + Double($1.painLevel) }
        return total / Double(painEntries.count)
    }
    
    private var maxPainLevel: Double {
        painEntries.map { Double($0.painLevel) }.max() ?? 0
    }
}

struct BASSDAITrendChart: View {
    let assessments: [BASSDAIAssessment]
    
    private var chartData: [BASSDAIDataPoint] {
        assessments
            .sorted { ($0.date ?? Date.distantPast) < ($1.date ?? Date.distantPast) }
            .map { assessment in
                BASSDAIDataPoint(
                    date: assessment.date ?? Date(),
                    totalScore: assessment.totalScore,
                    fatigue: assessment.fatigue,
                    morningStiffness: assessment.morningStiffness,
                    spinalPain: assessment.spinalPain
                )
            }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("BASDAI Assessment Trends")
                .font(.title2)
                .fontWeight(.bold)
            
            if #available(iOS 16.0, *) {
                Chart(chartData, id: \.date) { dataPoint in
                    LineMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Total Score", dataPoint.totalScore)
                    )
                    .foregroundStyle(.orange)
                    .lineStyle(StrokeStyle(lineWidth: 3))
                    
                    LineMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Fatigue", dataPoint.fatigue)
                    )
                    .foregroundStyle(.purple)
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [5]))
                    
                    LineMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Morning Stiffness", dataPoint.morningStiffness)
                    )
                    .foregroundStyle(.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [3]))
                }
                .frame(height: 200)
                .chartYScale(domain: 0...10)
                .chartLegend(position: .bottom)
            } else {
                LegacyBASSDAIChart(data: chartData)
            }
            
            // BASDAI interpretation
            if let latestScore = chartData.last?.totalScore {
                BASSDAIInterpretationCard(score: latestScore)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct PainMoodCorrelationChart: View {
    let journalEntries: [JournalEntry]
    
    private var correlationData: [CorrelationDataPoint] {
        journalEntries.compactMap { entry in
            guard let mood = entry.mood,
                  let moodValue = moodToValue(mood) else { return nil }
            
            return CorrelationDataPoint(
                date: entry.date ?? Date(),
                painLevel: Double(entry.painLevel),
                moodValue: moodValue,
                sleepQuality: Double(entry.sleepQuality),
                energyLevel: Double(entry.energyLevel)
            )
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Pain vs Mood Correlation")
                .font(.title2)
                .fontWeight(.bold)
            
            if #available(iOS 16.0, *) {
                Chart(correlationData, id: \.date) { dataPoint in
                    PointMark(
                        x: .value("Pain Level", dataPoint.painLevel),
                        y: .value("Mood", dataPoint.moodValue)
                    )
                    .foregroundStyle(.red)
                    .symbolSize(50)
                    .opacity(0.7)
                }
                .frame(height: 200)
                .chartXScale(domain: 0...10)
                .chartYScale(domain: 1...5)
            } else {
                LegacyCorrelationChart(data: correlationData)
            }
            
            // Correlation insights
            CorrelationInsightsCard(data: correlationData)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func moodToValue(_ mood: String) -> Double? {
        switch mood.lowercased() {
        case "very poor", "terrible": return 1.0
        case "poor", "bad": return 2.0
        case "okay", "neutral": return 3.0
        case "good": return 4.0
        case "very good", "excellent": return 5.0
        default: return nil
        }
    }
}

struct MedicationAdherenceChart: View {
    let medications: [Medication]
    let intakes: [MedicationIntake]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Medication Adherence")
                .font(.title2)
                .fontWeight(.bold)
            
            ForEach(medications, id: \.objectID) { medication in
                MedicationAdherenceRow(
                    medication: medication,
                    intakes: intakes.filter { $0.medication == medication }
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct PainHeatMapChart: View {
    let painEntries: [PainEntry]
    
    private var heatMapData: [HeatMapDataPoint] {
        let calendar = Calendar.current
        let now = Date()
        let thirtyDaysAgo = calendar.date(byAdding: .day, value: -30, to: now) ?? now
        
        var dataPoints: [HeatMapDataPoint] = []
        
        for i in 0..<30 {
            let date = calendar.date(byAdding: .day, value: i, to: thirtyDaysAgo) ?? Date()
            let dayEntries = painEntries.filter { entry in
                calendar.isDate(entry.timestamp ?? Date(), inSameDayAs: date)
            }
            
            let averagePain = dayEntries.isEmpty ? 0 :
                Double(dayEntries.reduce(0) { $0 + $1.painLevel }) / Double(dayEntries.count)
            
            dataPoints.append(HeatMapDataPoint(
                date: date,
                dayOfWeek: calendar.component(.weekday, from: date),
                weekOfMonth: calendar.component(.weekOfYear, from: date),
                painLevel: averagePain
            ))
        }
        
        return dataPoints
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("30-Day Pain Heat Map")
                .font(.title2)
                .fontWeight(.bold)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 7), spacing: 4) {
                ForEach(heatMapData, id: \.date) { dataPoint in
                    Rectangle()
                        .fill(heatMapColor(for: dataPoint.painLevel))
                        .frame(height: 20)
                        .cornerRadius(4)
                        .overlay(
                            Text("\(Calendar.current.component(.day, from: dataPoint.date))")
                                .font(.caption2)
                                .foregroundColor(.white)
                        )
                }
            }
            
            HStack {
                Text("Low")
                    .font(.caption)
                Rectangle()
                    .fill(LinearGradient(
                        gradient: Gradient(colors: [.green, .yellow, .orange, .red]),
                        startPoint: .leading,
                        endPoint: .trailing
                    ))
                    .frame(height: 10)
                Text("High")
                    .font(.caption)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func heatMapColor(for painLevel: Double) -> Color {
        switch painLevel {
        case 0:
            return .gray.opacity(0.3)
        case 0..<2:
            return .green
        case 2..<4:
            return .yellow
        case 4..<6:
            return .orange
        case 6..<8:
            return .red
        default:
            return .purple
        }
    }
}

struct WeeklyPainSummaryChart: View {
    let painEntries: [PainEntry]
    
    private var weeklyData: [WeeklyPainData] {
        let calendar = Calendar.current
        let now = Date()
        var weeklyData: [WeeklyPainData] = []
        
        for i in 0..<8 {
            let weekStart = calendar.date(byAdding: .weekOfYear, value: -i, to: now) ?? now
            let weekEnd = calendar.date(byAdding: .day, value: 6, to: weekStart) ?? now
            
            let weekEntries = painEntries.filter { entry in
                guard let timestamp = entry.timestamp else { return false }
                return timestamp >= weekStart && timestamp <= weekEnd
            }
            
            let averagePain = weekEntries.isEmpty ? 0 :
                Double(weekEntries.reduce(0) { $0 + $1.painLevel }) / Double(weekEntries.count)
            
            weeklyData.append(WeeklyPainData(
                weekStart: weekStart,
                averagePain: averagePain,
                entryCount: weekEntries.count
            ))
        }
        
        return weeklyData.reversed()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Weekly Pain Summary")
                .font(.title2)
                .fontWeight(.bold)
            
            if #available(iOS 16.0, *) {
                Chart(weeklyData, id: \.weekStart) { data in
                    BarMark(
                        x: .value("Week", data.weekStart),
                        y: .value("Average Pain", data.averagePain)
                    )
                    .foregroundStyle(.blue)
                    .cornerRadius(4)
                }
                .frame(height: 150)
                .chartYScale(domain: 0...10)
            } else {
                LegacyWeeklyChart(data: weeklyData)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Data Models

struct PainDataPoint {
    let date: Date
    let painLevel: Double
    let bodyRegion: String
}

struct BASSDAIDataPoint {
    let date: Date
    let totalScore: Double
    let fatigue: Double
    let morningStiffness: Double
    let spinalPain: Double
}

struct CorrelationDataPoint {
    let date: Date
    let painLevel: Double
    let moodValue: Double
    let sleepQuality: Double
    let energyLevel: Double
}

struct HeatMapDataPoint {
    let date: Date
    let dayOfWeek: Int
    let weekOfMonth: Int
    let painLevel: Double
}

struct WeeklyPainData {
    let weekStart: Date
    let averagePain: Double
    let entryCount: Int
}

// MARK: - Supporting Views

struct StatCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
}

struct BASSDAIInterpretationCard: View {
    let score: Double
    
    private var interpretation: (String, Color) {
        switch score {
        case 0..<2:
            return ("Inactive Disease", .green)
        case 2..<4:
            return ("Mild Activity", .yellow)
        case 4..<6:
            return ("Moderate Activity", .orange)
        case 6..<8:
            return ("High Activity", .red)
        default:
            return ("Very High Activity", .purple)
        }
    }
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Current BASDAI Score")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(String(format: "%.1f", score))
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(interpretation.1)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("Disease Activity")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(interpretation.0)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(interpretation.1)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
}

struct CorrelationInsightsCard: View {
    let data: [CorrelationDataPoint]
    
    private var painMoodCorrelation: Double {
        guard data.count > 1 else { return 0 }
        
        let painValues = data.map { $0.painLevel }
        let moodValues = data.map { $0.moodValue }
        
        return calculateCorrelation(painValues, moodValues)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Insights")
                .font(.headline)
            
            HStack {
                Text("Pain-Mood Correlation:")
                    .font(.caption)
                Spacer()
                Text(String(format: "%.2f", painMoodCorrelation))
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(correlationColor)
            }
            
            Text(correlationInterpretation)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
    
    private var correlationColor: Color {
        let abs = Swift.abs(painMoodCorrelation)
        if abs > 0.7 { return .red }
        if abs > 0.5 { return .orange }
        if abs > 0.3 { return .yellow }
        return .green
    }
    
    private var correlationInterpretation: String {
        let abs = Swift.abs(painMoodCorrelation)
        if abs > 0.7 { return "Strong correlation between pain and mood" }
        if abs > 0.5 { return "Moderate correlation between pain and mood" }
        if abs > 0.3 { return "Weak correlation between pain and mood" }
        return "No significant correlation between pain and mood"
    }
    
    private func calculateCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count && x.count > 1 else { return 0 }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        return denominator == 0 ? 0 : numerator / denominator
    }
}

struct MedicationAdherenceRow: View {
    let medication: Medication
    let intakes: [MedicationIntake]
    
    private var adherencePercentage: Double {
        let calendar = Calendar.current
        let now = Date()
        let thirtyDaysAgo = calendar.date(byAdding: .day, value: -30, to: now) ?? now
        
        let recentIntakes = intakes.filter { intake in
            guard let timestamp = intake.timestamp else { return false }
            return timestamp >= thirtyDaysAgo
        }
        
        let expectedDoses = 30 // Assuming daily medication
        let actualDoses = recentIntakes.count
        
        return expectedDoses > 0 ? Double(actualDoses) / Double(expectedDoses) * 100 : 0
    }
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(medication.name ?? "Unknown Medication")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                Text("\(medication.dosage ?? "") \(medication.unit ?? "")")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(String(format: "%.0f%%", adherencePercentage))
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(adherenceColor)
                
                Text("30-day adherence")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 8)
    }
    
    private var adherenceColor: Color {
        switch adherencePercentage {
        case 90...100: return .green
        case 70..<90: return .yellow
        case 50..<70: return .orange
        default: return .red
        }
    }
}

// MARK: - Legacy Chart Views for iOS 15 compatibility

struct LegacyPainTrendChart: View {
    let data: [PainDataPoint]
    
    var body: some View {
        Text("Pain trend chart (iOS 16+ required for full visualization)")
            .frame(height: 200)
            .frame(maxWidth: .infinity)
            .background(Color(.systemGray5))
            .cornerRadius(8)
    }
}

struct LegacyBASSDAIChart: View {
    let data: [BASSDAIDataPoint]
    
    var body: some View {
        Text("BASDAI chart (iOS 16+ required for full visualization)")
            .frame(height: 200)
            .frame(maxWidth: .infinity)
            .background(Color(.systemGray5))
            .cornerRadius(8)
    }
}

struct LegacyCorrelationChart: View {
    let data: [CorrelationDataPoint]
    
    var body: some View {
        Text("Correlation chart (iOS 16+ required for full visualization)")
            .frame(height: 200)
            .frame(maxWidth: .infinity)
            .background(Color(.systemGray5))
            .cornerRadius(8)
    }
}

struct LegacyWeeklyChart: View {
    let data: [WeeklyPainData]
    
    var body: some View {
        Text("Weekly chart (iOS 16+ required for full visualization)")
            .frame(height: 150)
            .frame(maxWidth: .infinity)
            .background(Color(.systemGray5))
            .cornerRadius(8)
    }
}

struct PainAnalyticsCharts_Previews: PreviewProvider {
    static var previews: some View {
        PainAnalyticsCharts(
            painEntries: [],
            journalEntries: [],
            bassdaiAssessments: [],
            medications: [],
            medicationIntakes: []
        )
    }
}