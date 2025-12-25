//
//  AnalyticsView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import Charts
import CoreData

struct AnalyticsView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @State private var selectedTimeRange: TimeRange = .week
    @State private var selectedChart: ChartType = .pain
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)],
        animation: .default)
    private var painEntries: FetchedResults<PainEntry>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: true)],
        animation: .default)
    private var bassdaiAssessments: FetchedResults<BASSDAIAssessment>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)],
        animation: .default)
    private var journalEntries: FetchedResults<JournalEntry>
    
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \MedicationIntake.timestamp, ascending: true)],
        animation: .default)
    private var medicationIntakes: FetchedResults<MedicationIntake>
    
    enum TimeRange: String, CaseIterable {
        case week = "Week"
        case month = "Month"
        case threeMonths = "3 Months"
        
        var days: Int {
            switch self {
            case .week: return 7
            case .month: return 30
            case .threeMonths: return 90
            }
        }
    }
    
    enum ChartType: String, CaseIterable {
        case pain = "Pain Levels"
        case basdai = "BASDAI Scores"
        case energy = "Energy & Mood"
        case medication = "Medication Adherence"
        
        var icon: String {
            switch self {
            case .pain: return "figure.walk"
            case .basdai: return "chart.bar.fill"
            case .energy: return "bolt.fill"
            case .medication: return "pills.fill"
            }
        }
        
        var color: Color {
            switch self {
            case .pain: return .red
            case .basdai: return .orange
            case .energy: return .green
            case .medication: return .blue
            }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Time Range Selector
                    timeRangeSelector
                    
                    // Chart Type Selector
                    chartTypeSelector
                    
                    // Main Chart
                    mainChartView
                    
                    // Summary Statistics
                    summaryStatsView
                    
                    // Trend Insights
                    trendInsightsView
                }
                .padding()
            }
            .navigationTitle("Analytics")
            .navigationBarTitleDisplayMode(.large)
        }
    }
    
    private var timeRangeSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Time Range")
                .font(.headline)
                .fontWeight(.semibold)
            
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
        }
    }
    
    private var chartTypeSelector: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Chart Type")
                .font(.headline)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                ForEach(ChartType.allCases, id: \.self) { chartType in
                    ChartTypeCard(
                        type: chartType,
                        isSelected: selectedChart == chartType
                    ) {
                        selectedChart = chartType
                    }
                }
            }
        }
    }
    
    private var mainChartView: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: selectedChart.icon)
                    .foregroundColor(selectedChart.color)
                Text(selectedChart.rawValue)
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            Group {
                switch selectedChart {
                case .pain:
                    painTrendChart
                case .basdai:
                    bassdaiTrendChart
                case .energy:
                    energyMoodChart
                case .medication:
                    medicationAdherenceChart
                }
            }
            .frame(height: 250)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.white)
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
    
    private var painTrendChart: some View {
        let filteredData = filteredPainData
        
        return Chart(filteredData, id: \.date) { entry in
            LineMark(
                x: .value("Date", entry.date),
                y: .value("Pain Level", entry.painLevel)
            )
            .foregroundStyle(Color.red.gradient)
            .interpolationMethod(.catmullRom)
            
            AreaMark(
                x: .value("Date", entry.date),
                y: .value("Pain Level", entry.painLevel)
            )
            .foregroundStyle(Color.red.opacity(0.1))
            .interpolationMethod(.catmullRom)
        }
        .chartYScale(domain: 0...10)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: selectedTimeRange == .week ? 1 : 7)) {
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.weekday(.abbreviated))
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
    }
    
    private var bassdaiTrendChart: some View {
        let filteredData = filteredBASSDAIData
        
        return Chart(filteredData, id: \.date) { assessment in
            LineMark(
                x: .value("Date", assessment.date),
                y: .value("BASDAI Score", assessment.totalScore)
            )
            .foregroundStyle(Color.orange.gradient)
            .interpolationMethod(.catmullRom)
            
            PointMark(
                x: .value("Date", assessment.date),
                y: .value("BASDAI Score", assessment.totalScore)
            )
            .foregroundStyle(Color.orange)
        }
        .chartYScale(domain: 0...10)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: selectedTimeRange == .week ? 1 : 7)) {
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.weekday(.abbreviated))
            }
        }
    }
    
    private var energyMoodChart: some View {
        let filteredData = filteredJournalData
        
        return Chart {
            ForEach(filteredData, id: \.date) { entry in
                LineMark(
                    x: .value("Date", entry.date),
                    y: .value("Energy", entry.energyLevel)
                )
                .foregroundStyle(Color.green)
                .symbol(Circle())
                
                LineMark(
                    x: .value("Date", entry.date),
                    y: .value("Sleep", entry.sleepQuality)
                )
                .foregroundStyle(Color.blue)
                .symbol(Square())
            }
        }
        .chartYScale(domain: 0...10)
        .chartForegroundStyleScale([
            "Energy": Color.green,
            "Sleep": Color.blue
        ])
        .chartLegend(position: .top)
    }
    
    private var medicationAdherenceChart: some View {
        let adherenceData = calculateMedicationAdherence()
        
        return Chart(adherenceData, id: \.date) { data in
            BarMark(
                x: .value("Date", data.date),
                y: .value("Adherence %", data.adherencePercentage)
            )
            .foregroundStyle(Color.blue.gradient)
        }
        .chartYScale(domain: 0...100)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: selectedTimeRange == .week ? 1 : 7)) {
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.weekday(.abbreviated))
            }
        }
    }
    
    private var summaryStatsView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Summary Statistics")
                .font(.headline)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                StatCard(
                    title: "Avg Pain Level",
                    value: String(format: "%.1f", averagePainLevel),
                    icon: "figure.walk",
                    color: .red
                )
                
                StatCard(
                    title: "Avg BASDAI",
                    value: String(format: "%.1f", averageBASSDAI),
                    icon: "chart.bar.fill",
                    color: .orange
                )
                
                StatCard(
                    title: "Avg Energy",
                    value: String(format: "%.1f", averageEnergy),
                    icon: "bolt.fill",
                    color: .green
                )
                
                StatCard(
                    title: "Med Adherence",
                    value: "\(Int(medicationAdherenceRate))%",
                    icon: "pills.fill",
                    color: .blue
                )
            }
        }
    }
    
    private var trendInsightsView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Trend Insights")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(spacing: 12) {
                ForEach(generateTrendInsights(), id: \.id) { insight in
                    InsightCard(insight: insight)
                }
            }
        }
    }
}

// MARK: - Supporting Views

struct ChartTypeCard: View {
    let type: AnalyticsView.ChartType
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: type.icon)
                    .font(.system(size: 24))
                    .foregroundColor(isSelected ? .white : type.color)
                
                Text(type.rawValue)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(isSelected ? .white : .primary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isSelected ? type.color : Color.gray.opacity(0.1))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white)
                .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        )
    }
}

struct InsightCard: View {
    let insight: TrendInsight
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: insight.icon)
                .font(.system(size: 20))
                .foregroundColor(insight.color)
                .frame(width: 32, height: 32)
                .background(
                    Circle()
                        .fill(insight.color.opacity(0.1))
                )
            
            VStack(alignment: .leading, spacing: 4) {
                Text(insight.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(insight.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.1))
        )
    }
}

struct TrendInsight {
    let id = UUID()
    let title: String
    let description: String
    let icon: String
    let color: Color
}

// MARK: - Data Processing Extensions

extension AnalyticsView {
    private var filteredPainData: [(date: Date, painLevel: Double)] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        
        return painEntries
            .filter { ($0.timestamp ?? Date()) >= cutoffDate }
            .map { (date: $0.timestamp ?? Date(), painLevel: Double($0.painLevel)) }
    }
    
    private var filteredBASSDAIData: [(date: Date, totalScore: Double)] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        
        return bassdaiAssessments
            .filter { ($0.date ?? Date()) >= cutoffDate }
            .map { (date: $0.date ?? Date(), totalScore: $0.totalScore) }
    }
    
    private var filteredJournalData: [(date: Date, energyLevel: Double, sleepQuality: Double)] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        
        return journalEntries
            .filter { ($0.date ?? Date()) >= cutoffDate }
            .map { (date: $0.date ?? Date(), energyLevel: $0.energyLevel, sleepQuality: $0.sleepQuality) }
    }
    
    private func calculateMedicationAdherence() -> [(date: Date, adherencePercentage: Double)] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        let calendar = Calendar.current
        
        var adherenceData: [(date: Date, adherencePercentage: Double)] = []
        
        for i in 0..<selectedTimeRange.days {
            guard let date = calendar.date(byAdding: .day, value: -i, to: Date()) else { continue }
            let startOfDay = calendar.startOfDay(for: date)
            let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay) ?? Date()
            
            let dayIntakes = medicationIntakes.filter {
                guard let timestamp = $0.timestamp else { return false }
                return timestamp >= startOfDay && timestamp < endOfDay
            }
            
            // Simplified adherence calculation - in real app, this would be more sophisticated
            let adherencePercentage = min(Double(dayIntakes.count) * 25.0, 100.0) // Assuming 4 medications max
            adherenceData.append((date: startOfDay, adherencePercentage: adherencePercentage))
        }
        
        return adherenceData.reversed()
    }
    
    private var averagePainLevel: Double {
        let data = filteredPainData
        guard !data.isEmpty else { return 0 }
        return data.map { $0.painLevel }.reduce(0, +) / Double(data.count)
    }
    
    private var averageBASSDAI: Double {
        let data = filteredBASSDAIData
        guard !data.isEmpty else { return 0 }
        return data.map { $0.totalScore }.reduce(0, +) / Double(data.count)
    }
    
    private var averageEnergy: Double {
        let data = filteredJournalData
        guard !data.isEmpty else { return 0 }
        return data.map { $0.energyLevel }.reduce(0, +) / Double(data.count)
    }
    
    private var medicationAdherenceRate: Double {
        let adherenceData = calculateMedicationAdherence()
        guard !adherenceData.isEmpty else { return 0 }
        return adherenceData.map { $0.adherencePercentage }.reduce(0, +) / Double(adherenceData.count)
    }
    
    private func generateTrendInsights() -> [TrendInsight] {
        var insights: [TrendInsight] = []
        
        // Pain trend analysis
        let painData = filteredPainData
        if painData.count >= 3 {
            let recentPain = painData.suffix(3).map { $0.painLevel }.reduce(0, +) / 3
            let earlierPain = painData.prefix(3).map { $0.painLevel }.reduce(0, +) / 3
            
            if recentPain < earlierPain - 0.5 {
                insights.append(TrendInsight(
                    title: "Pain Improving",
                    description: "Your pain levels have decreased by \(String(format: "%.1f", earlierPain - recentPain)) points recently.",
                    icon: "arrow.down.circle.fill",
                    color: .green
                ))
            } else if recentPain > earlierPain + 0.5 {
                insights.append(TrendInsight(
                    title: "Pain Increasing",
                    description: "Your pain levels have increased by \(String(format: "%.1f", recentPain - earlierPain)) points recently.",
                    icon: "arrow.up.circle.fill",
                    color: .red
                ))
            }
        }
        
        // Medication adherence insight
        if medicationAdherenceRate > 80 {
            insights.append(TrendInsight(
                title: "Great Adherence",
                description: "You're maintaining excellent medication adherence at \(Int(medicationAdherenceRate))%.",
                icon: "checkmark.circle.fill",
                color: .green
            ))
        } else if medicationAdherenceRate < 60 {
            insights.append(TrendInsight(
                title: "Improve Adherence",
                description: "Consider setting more reminders to improve medication adherence.",
                icon: "exclamationmark.triangle.fill",
                color: .orange
            ))
        }
        
        // Energy correlation insight
        let journalData = filteredJournalData
        if !journalData.isEmpty {
            let avgEnergy = journalData.map { $0.energyLevel }.reduce(0, +) / Double(journalData.count)
            let avgSleep = journalData.map { $0.sleepQuality }.reduce(0, +) / Double(journalData.count)
            
            if avgSleep > 7 && avgEnergy > 6 {
                insights.append(TrendInsight(
                    title: "Sleep-Energy Connection",
                    description: "Good sleep quality is correlating with higher energy levels.",
                    icon: "moon.stars.fill",
                    color: .blue
                ))
            }
        }
        
        return insights
    }
}

struct AnalyticsView_Previews: PreviewProvider {
    static var previews: some View {
        AnalyticsView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
            .coreDataErrorAlert()
    }
}