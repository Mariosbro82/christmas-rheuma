//
//  TrendVisualizationComponents.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import Charts
import CoreData

// MARK: - Weekly Trend View

struct WeeklyTrendView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @State private var selectedMetric: TrendMetric = .pain
    @State private var weekOffset: Int = 0
    @State private var trendData: [TrendDataPoint] = []
    @State private var isLoading = true
    @State private var averageValue: Double = 0
    @State private var trendDirection: TrendDirection = .stable
    
    private var currentWeekStart: Date {
        let calendar = Calendar.current
        let today = Date()
        let weekStart = calendar.dateInterval(of: .weekOfYear, for: today)?.start ?? today
        return calendar.date(byAdding: .weekOfYear, value: weekOffset, to: weekStart) ?? weekStart
    }
    
    private var currentWeekEnd: Date {
        Calendar.current.date(byAdding: .day, value: 6, to: currentWeekStart) ?? currentWeekStart
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Header with metric selector
            VStack(spacing: 12) {
                HStack {
                    Text("Weekly Trends")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Spacer()
                    
                    Menu {
                        ForEach(TrendMetric.allCases, id: \.self) { metric in
                            Button(metric.displayName) {
                                selectedMetric = metric
                                loadWeeklyData()
                            }
                        }
                    } label: {
                        HStack {
                            Text(selectedMetric.displayName)
                            Image(systemName: "chevron.down")
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color(.systemGray5))
                        .cornerRadius(8)
                    }
                }
                
                // Week navigation
                HStack {
                    Button(action: {
                        weekOffset -= 1
                        loadWeeklyData()
                    }) {
                        Image(systemName: "chevron.left")
                            .foregroundColor(.blue)
                    }
                    
                    Spacer()
                    
                    VStack {
                        Text(weekRangeText)
                            .font(.headline)
                        Text(weekOffset == 0 ? "This Week" : "\(abs(weekOffset)) week\(abs(weekOffset) == 1 ? "" : "s") ago")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Button(action: {
                        weekOffset += 1
                        loadWeeklyData()
                    }) {
                        Image(systemName: "chevron.right")
                            .foregroundColor(weekOffset >= 0 ? .gray : .blue)
                    }
                    .disabled(weekOffset >= 0)
                }
            }
            
            if isLoading {
                ProgressView("Loading weekly data...")
                    .frame(height: 200)
            } else if trendData.isEmpty {
                VStack {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 40))
                        .foregroundColor(.gray)
                    Text("No data available for this week")
                        .foregroundColor(.secondary)
                }
                .frame(height: 200)
            } else {
                VStack(spacing: 12) {
                    // Summary stats
                    HStack(spacing: 20) {
                        StatCard(
                            title: "Average",
                            value: String(format: "%.1f", averageValue),
                            color: selectedMetric.color
                        )
                        
                        StatCard(
                            title: "Trend",
                            value: trendDirection.rawValue,
                            color: trendDirection.color
                        )
                        
                        StatCard(
                            title: "Days Tracked",
                            value: "\(trendData.count)",
                            color: .blue
                        )
                    }
                    
                    // Chart
                    Chart(trendData) { dataPoint in
                        LineMark(
                            x: .value("Day", dataPoint.date),
                            y: .value(selectedMetric.displayName, dataPoint.value)
                        )
                        .foregroundStyle(selectedMetric.color)
                        .lineStyle(StrokeStyle(lineWidth: 3))
                        
                        PointMark(
                            x: .value("Day", dataPoint.date),
                            y: .value(selectedMetric.displayName, dataPoint.value)
                        )
                        .foregroundStyle(selectedMetric.color)
                        .symbolSize(50)
                    }
                    .frame(height: 200)
                    .chartXAxis {
                        AxisMarks(values: .stride(by: .day)) { value in
                            if let date = value.as(Date.self) {
                                AxisValueLabel {
                                    Text(dayFormatter.string(from: date))
                                        .font(.caption)
                                }
                            }
                        }
                    }
                    .chartYAxis {
                        AxisMarks { value in
                            AxisValueLabel {
                                if let doubleValue = value.as(Double.self) {
                                    Text(selectedMetric.formatValue(doubleValue))
                                        .font(.caption)
                                }
                            }
                        }
                    }
                    .chartYScale(domain: selectedMetric.yAxisRange)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onAppear {
            loadWeeklyData()
        }
        .onChange(of: selectedMetric) { _ in
            loadWeeklyData()
        }
    }
    
    private var weekRangeText: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return "\(formatter.string(from: currentWeekStart)) - \(formatter.string(from: currentWeekEnd))"
    }
    
    private var dayFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "E"
        return formatter
    }
    
    private func loadWeeklyData() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let data = fetchWeeklyData(for: selectedMetric, startDate: currentWeekStart, endDate: currentWeekEnd)
            let average = data.isEmpty ? 0 : data.reduce(0) { $0 + $1.value } / Double(data.count)
            let trend = calculateTrend(data)
            
            DispatchQueue.main.async {
                self.trendData = data
                self.averageValue = average
                self.trendDirection = trend
                self.isLoading = false
            }
        }
    }
    
    private func fetchWeeklyData(for metric: TrendMetric, startDate: Date, endDate: Date) -> [TrendDataPoint] {
        switch metric {
        case .pain:
            return fetchPainData(startDate: startDate, endDate: endDate)
        case .energy:
            return fetchEnergyData(startDate: startDate, endDate: endDate)
        case .mood:
            return fetchMoodData(startDate: startDate, endDate: endDate)
        case .sleep:
            return fetchSleepData(startDate: startDate, endDate: endDate)
        case .basdai:
            return fetchBASSDAIData(startDate: startDate, endDate: endDate)
        }
    }
    
    private func fetchPainData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp <= %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let timestamp = entry.timestamp else { return nil }
                return TrendDataPoint(date: timestamp, value: entry.painLevel)
            }
        } catch {
            return []
        }
    }
    
    private func fetchEnergyData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date <= %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.energyLevel)
            }
        } catch {
            return []
        }
    }
    
    private func fetchMoodData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date <= %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.mood)
            }
        } catch {
            return []
        }
    }
    
    private func fetchSleepData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date <= %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.sleepQuality)
            }
        } catch {
            return []
        }
    }
    
    private func fetchBASSDAIData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date <= %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.overallWellbeing)
            }
        } catch {
            return []
        }
    }
    
    private func calculateTrend(_ data: [TrendDataPoint]) -> TrendDirection {
        guard data.count >= 2 else { return .stable }
        
        let firstHalf = data.prefix(data.count / 2)
        let secondHalf = data.suffix(data.count / 2)
        
        let firstAvg = firstHalf.reduce(0) { $0 + $1.value } / Double(firstHalf.count)
        let secondAvg = secondHalf.reduce(0) { $0 + $1.value } / Double(secondHalf.count)
        
        let difference = secondAvg - firstAvg
        
        if difference > 0.5 {
            return .increasing
        } else if difference < -0.5 {
            return .decreasing
        } else {
            return .stable
        }
    }
}

// MARK: - Monthly Trend View

struct MonthlyTrendView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @State private var selectedMetric: TrendMetric = .pain
    @State private var monthOffset: Int = 0
    @State private var trendData: [TrendDataPoint] = []
    @State private var weeklyAverages: [TrendDataPoint] = []
    @State private var isLoading = true
    @State private var monthlyStats: MonthlyStats = MonthlyStats()
    
    private var currentMonthStart: Date {
        let calendar = Calendar.current
        let today = Date()
        let monthStart = calendar.dateInterval(of: .month, for: today)?.start ?? today
        return calendar.date(byAdding: .month, value: monthOffset, to: monthStart) ?? monthStart
    }
    
    private var currentMonthEnd: Date {
        let calendar = Calendar.current
        return calendar.dateInterval(of: .month, for: currentMonthStart)?.end ?? currentMonthStart
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Header
            VStack(spacing: 12) {
                HStack {
                    Text("Monthly Trends")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Spacer()
                    
                    Menu {
                        ForEach(TrendMetric.allCases, id: \.self) { metric in
                            Button(metric.displayName) {
                                selectedMetric = metric
                                loadMonthlyData()
                            }
                        }
                    } label: {
                        HStack {
                            Text(selectedMetric.displayName)
                            Image(systemName: "chevron.down")
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color(.systemGray5))
                        .cornerRadius(8)
                    }
                }
                
                // Month navigation
                HStack {
                    Button(action: {
                        monthOffset -= 1
                        loadMonthlyData()
                    }) {
                        Image(systemName: "chevron.left")
                            .foregroundColor(.blue)
                    }
                    
                    Spacer()
                    
                    VStack {
                        Text(monthYearText)
                            .font(.headline)
                        Text(monthOffset == 0 ? "This Month" : "\(abs(monthOffset)) month\(abs(monthOffset) == 1 ? "" : "s") ago")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Button(action: {
                        monthOffset += 1
                        loadMonthlyData()
                    }) {
                        Image(systemName: "chevron.right")
                            .foregroundColor(monthOffset >= 0 ? .gray : .blue)
                    }
                    .disabled(monthOffset >= 0)
                }
            }
            
            if isLoading {
                ProgressView("Loading monthly data...")
                    .frame(height: 300)
            } else if trendData.isEmpty {
                VStack {
                    Image(systemName: "chart.bar")
                        .font(.system(size: 40))
                        .foregroundColor(.gray)
                    Text("No data available for this month")
                        .foregroundColor(.secondary)
                }
                .frame(height: 300)
            } else {
                VStack(spacing: 16) {
                    // Monthly stats
                    MonthlyStatsView(stats: monthlyStats, metric: selectedMetric)
                    
                    // Daily trend chart
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Daily Values")
                            .font(.headline)
                        
                        Chart(trendData) { dataPoint in
                            LineMark(
                                x: .value("Date", dataPoint.date),
                                y: .value(selectedMetric.displayName, dataPoint.value)
                            )
                            .foregroundStyle(selectedMetric.color.opacity(0.7))
                            .lineStyle(StrokeStyle(lineWidth: 2))
                            
                            PointMark(
                                x: .value("Date", dataPoint.date),
                                y: .value(selectedMetric.displayName, dataPoint.value)
                            )
                            .foregroundStyle(selectedMetric.color)
                            .symbolSize(30)
                        }
                        .frame(height: 150)
                        .chartXAxis {
                            AxisMarks(values: .stride(by: .day, count: 7)) { value in
                                if let date = value.as(Date.self) {
                                    AxisValueLabel {
                                        Text("\(Calendar.current.component(.day, from: date))")
                                            .font(.caption)
                                    }
                                }
                            }
                        }
                        .chartYScale(domain: selectedMetric.yAxisRange)
                    }
                    
                    // Weekly averages chart
                    if !weeklyAverages.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Weekly Averages")
                                .font(.headline)
                            
                            Chart(weeklyAverages) { dataPoint in
                                BarMark(
                                    x: .value("Week", weekFormatter.string(from: dataPoint.date)),
                                    y: .value("Average", dataPoint.value)
                                )
                                .foregroundStyle(selectedMetric.color)
                            }
                            .frame(height: 120)
                            .chartYScale(domain: selectedMetric.yAxisRange)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onAppear {
            loadMonthlyData()
        }
        .onChange(of: selectedMetric) { _ in
            loadMonthlyData()
        }
    }
    
    private var monthYearText: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMMM yyyy"
        return formatter.string(from: currentMonthStart)
    }
    
    private var weekFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "'Week' w"
        return formatter
    }
    
    private func loadMonthlyData() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let dailyData = fetchMonthlyData(for: selectedMetric, startDate: currentMonthStart, endDate: currentMonthEnd)
            let weeklyData = calculateWeeklyAverages(from: dailyData)
            let stats = calculateMonthlyStats(from: dailyData)
            
            DispatchQueue.main.async {
                self.trendData = dailyData
                self.weeklyAverages = weeklyData
                self.monthlyStats = stats
                self.isLoading = false
            }
        }
    }
    
    private func fetchMonthlyData(for metric: TrendMetric, startDate: Date, endDate: Date) -> [TrendDataPoint] {
        switch metric {
        case .pain:
            return fetchPainData(startDate: startDate, endDate: endDate)
        case .energy:
            return fetchEnergyData(startDate: startDate, endDate: endDate)
        case .mood:
            return fetchMoodData(startDate: startDate, endDate: endDate)
        case .sleep:
            return fetchSleepData(startDate: startDate, endDate: endDate)
        case .basdai:
            return fetchBASSDAIData(startDate: startDate, endDate: endDate)
        }
    }
    
    private func fetchPainData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp < %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let timestamp = entry.timestamp else { return nil }
                return TrendDataPoint(date: timestamp, value: entry.painLevel)
            }
        } catch {
            return []
        }
    }
    
    private func fetchEnergyData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date < %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.energyLevel)
            }
        } catch {
            return []
        }
    }
    
    private func fetchMoodData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date < %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.mood)
            }
        } catch {
            return []
        }
    }
    
    private func fetchSleepData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date < %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.sleepQuality)
            }
        } catch {
            return []
        }
    }
    
    private func fetchBASSDAIData(startDate: Date, endDate: Date) -> [TrendDataPoint] {
        let request: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
        request.predicate = NSPredicate(
            format: "date >= %@ AND date < %@",
            startDate as CVarArg,
            endDate as CVarArg
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: true)]
        
        do {
            let entries = try viewContext.fetch(request)
            return entries.compactMap { entry in
                guard let date = entry.date else { return nil }
                return TrendDataPoint(date: date, value: entry.overallWellbeing)
            }
        } catch {
            return []
        }
    }
    
    private func calculateWeeklyAverages(from dailyData: [TrendDataPoint]) -> [TrendDataPoint] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: dailyData) { dataPoint in
            calendar.dateInterval(of: .weekOfYear, for: dataPoint.date)?.start ?? dataPoint.date
        }
        
        return grouped.compactMap { (weekStart, dataPoints) in
            let average = dataPoints.reduce(0) { $0 + $1.value } / Double(dataPoints.count)
            return TrendDataPoint(date: weekStart, value: average)
        }.sorted { $0.date < $1.date }
    }
    
    private func calculateMonthlyStats(from data: [TrendDataPoint]) -> MonthlyStats {
        guard !data.isEmpty else { return MonthlyStats() }
        
        let values = data.map { $0.value }
        let average = values.reduce(0, +) / Double(values.count)
        let minimum = values.min() ?? 0
        let maximum = values.max() ?? 0
        
        let sortedValues = values.sorted()
        let median: Double
        if sortedValues.count % 2 == 0 {
            median = (sortedValues[sortedValues.count / 2 - 1] + sortedValues[sortedValues.count / 2]) / 2
        } else {
            median = sortedValues[sortedValues.count / 2]
        }
        
        let trend = calculateTrend(data)
        
        return MonthlyStats(
            average: average,
            minimum: minimum,
            maximum: maximum,
            median: median,
            trend: trend,
            daysTracked: data.count
        )
    }
    
    private func calculateTrend(_ data: [TrendDataPoint]) -> TrendDirection {
        guard data.count >= 4 else { return .stable }
        
        let firstQuarter = data.prefix(data.count / 4)
        let lastQuarter = data.suffix(data.count / 4)
        
        let firstAvg = firstQuarter.reduce(0) { $0 + $1.value } / Double(firstQuarter.count)
        let lastAvg = lastQuarter.reduce(0) { $0 + $1.value } / Double(lastQuarter.count)
        
        let difference = lastAvg - firstAvg
        
        if difference > 0.5 {
            return .increasing
        } else if difference < -0.5 {
            return .decreasing
        } else {
            return .stable
        }
    }
}

// MARK: - Supporting Views

struct StatCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
}

struct MonthlyStatsView: View {
    let stats: MonthlyStats
    let metric: TrendMetric
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Monthly Summary")
                .font(.headline)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                StatCard(
                    title: "Average",
                    value: metric.formatValue(stats.average),
                    color: metric.color
                )
                
                StatCard(
                    title: "Best",
                    value: metric.formatValue(metric == .pain ? stats.minimum : stats.maximum),
                    color: .green
                )
                
                StatCard(
                    title: "Worst",
                    value: metric.formatValue(metric == .pain ? stats.maximum : stats.minimum),
                    color: .red
                )
                
                StatCard(
                    title: "Median",
                    value: metric.formatValue(stats.median),
                    color: .blue
                )
                
                StatCard(
                    title: "Trend",
                    value: stats.trend.rawValue,
                    color: stats.trend.color
                )
                
                StatCard(
                    title: "Days Tracked",
                    value: "\(stats.daysTracked)",
                    color: .purple
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
}

// MARK: - Data Models

enum TrendMetric: String, CaseIterable {
    case pain = "pain"
    case energy = "energy"
    case mood = "mood"
    case sleep = "sleep"
    case basdai = "basdai"
    
    var displayName: String {
        switch self {
        case .pain: return "Pain Level"
        case .energy: return "Energy Level"
        case .mood: return "Mood"
        case .sleep: return "Sleep Quality"
        case .basdai: return "BASDAI Score"
        }
    }
    
    var color: Color {
        switch self {
        case .pain: return .red
        case .energy: return .green
        case .mood: return .blue
        case .sleep: return .purple
        case .basdai: return .orange
        }
    }
    
    var yAxisRange: ClosedRange<Double> {
        switch self {
        case .pain, .energy, .mood, .sleep, .basdai:
            return 0...10
        }
    }
    
    func formatValue(_ value: Double) -> String {
        return String(format: "%.1f", value)
    }
}

struct MonthlyStats {
    let average: Double
    let minimum: Double
    let maximum: Double
    let median: Double
    let trend: TrendDirection
    let daysTracked: Int
    
    init(
        average: Double = 0,
        minimum: Double = 0,
        maximum: Double = 0,
        median: Double = 0,
        trend: TrendDirection = .stable,
        daysTracked: Int = 0
    ) {
        self.average = average
        self.minimum = minimum
        self.maximum = maximum
        self.median = median
        self.trend = trend
        self.daysTracked = daysTracked
    }
}

extension TrendDirection {
    var color: Color {
        switch self {
        case .increasing: return .green
        case .decreasing: return .red
        case .stable: return .blue
        }
    }
}

#Preview {
    VStack {
        WeeklyTrendView()
        MonthlyTrendView()
    }
    let container = NSPersistentContainer(name: "InflamAI")
        container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        container.loadPersistentStores { _, _ in }
        let context = container.viewContext
        
        return PainTrendChart()
            .environment(\.managedObjectContext, context)
}