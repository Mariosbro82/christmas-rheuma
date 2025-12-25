//
//  PainIntensityHistoryView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import Charts

struct PainIntensityHistoryView: View {
    let selectedRegions: Set<BodyRegion>
    
    @StateObject private var aiEngine = AIMLEngine.shared
    @StateObject private var dataManager = HealthDataManager.shared
    
    @State private var selectedTimeRange: TimeRange = .week
    @State private var showingDetailedAnalysis = false
    @State private var selectedDataPoint: PainDataPoint?
    @State private var painHistory: [PainDataPoint] = []
    @State private var isLoading = true
    
    private enum TimeRange: String, CaseIterable {
        case day = "24H"
        case week = "7D"
        case month = "30D"
        case quarter = "3M"
        case year = "1Y"
        
        var displayName: String {
            switch self {
            case .day: return "Last 24 Hours"
            case .week: return "Last Week"
            case .month: return "Last Month"
            case .quarter: return "Last 3 Months"
            case .year: return "Last Year"
            }
        }
        
        var dateRange: DateInterval {
            let now = Date()
            switch self {
            case .day:
                return DateInterval(start: Calendar.current.date(byAdding: .hour, value: -24, to: now)!, end: now)
            case .week:
                return DateInterval(start: Calendar.current.date(byAdding: .day, value: -7, to: now)!, end: now)
            case .month:
                return DateInterval(start: Calendar.current.date(byAdding: .day, value: -30, to: now)!, end: now)
            case .quarter:
                return DateInterval(start: Calendar.current.date(byAdding: .month, value: -3, to: now)!, end: now)
            case .year:
                return DateInterval(start: Calendar.current.date(byAdding: .year, value: -1, to: now)!, end: now)
            }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header with Time Range Selector
                    VStack(spacing: 15) {
                        HStack {
                            Text("Pain History")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                            
                            Spacer()
                            
                            Button("Analysis") {
                                showingDetailedAnalysis = true
                            }
                            .font(.subheadline)
                            .foregroundColor(.blue)
                        }
                        
                        // Time Range Picker
                        Picker("Time Range", selection: $selectedTimeRange) {
                            ForEach(TimeRange.allCases, id: \.self) { range in
                                Text(range.rawValue)
                                    .tag(range)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .onChange(of: selectedTimeRange) { _ in
                            loadPainHistory()
                        }
                    }
                    .padding()
                    
                    if isLoading {
                        ProgressView("Loading pain history...")
                            .frame(height: 200)
                    } else if painHistory.isEmpty {
                        EmptyHistoryView()
                    } else {
                        // Pain Trend Chart
                        PainTrendChartView(
                            data: painHistory,
                            selectedDataPoint: $selectedDataPoint,
                            timeRange: selectedTimeRange
                        )
                        
                        // Statistics Cards
                        PainStatisticsView(
                            data: painHistory,
                            selectedRegions: selectedRegions
                        )
                        
                        // Region-Specific Analysis
                        if !selectedRegions.isEmpty {
                            RegionAnalysisView(
                                data: painHistory,
                                regions: selectedRegions
                            )
                        }
                        
                        // AI Insights
                        AIPainInsightsView(
                            data: painHistory,
                            insights: aiEngine.analyzePainTrends(painHistory)
                        )
                        
                        // Recent Pain Entries
                        RecentPainEntriesView(
                            entries: Array(painHistory.prefix(10))
                        )
                    }
                }
            }
            .navigationBarHidden(true)
            .onAppear {
                loadPainHistory()
            }
            .sheet(isPresented: $showingDetailedAnalysis) {
                DetailedPainAnalysisView(
                    data: painHistory,
                    selectedRegions: selectedRegions
                )
            }
        }
    }
    
    private func loadPainHistory() {
        isLoading = true
        
        Task {
            do {
                let history = try await dataManager.getPainHistory(
                    for: selectedTimeRange.dateRange,
                    regions: selectedRegions.isEmpty ? nil : selectedRegions
                )
                
                await MainActor.run {
                    self.painHistory = history
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.painHistory = []
                    self.isLoading = false
                }
            }
        }
    }
}

// MARK: - Supporting Views

struct PainTrendChartView: View {
    let data: [PainDataPoint]
    @Binding var selectedDataPoint: PainDataPoint?
    let timeRange: PainIntensityHistoryView.TimeRange
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Pain Trend")
                .font(.headline)
                .padding(.horizontal)
            
            Chart(data, id: \.timestamp) { dataPoint in
                LineMark(
                    x: .value("Time", dataPoint.timestamp),
                    y: .value("Pain Level", dataPoint.averagePainLevel)
                )
                .foregroundStyle(colorForPainLevel(dataPoint.averagePainLevel))
                .lineStyle(StrokeStyle(lineWidth: 3))
                
                AreaMark(
                    x: .value("Time", dataPoint.timestamp),
                    y: .value("Pain Level", dataPoint.averagePainLevel)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [colorForPainLevel(dataPoint.averagePainLevel).opacity(0.3), .clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                
                if let selected = selectedDataPoint, selected.id == dataPoint.id {
                    PointMark(
                        x: .value("Time", dataPoint.timestamp),
                        y: .value("Pain Level", dataPoint.averagePainLevel)
                    )
                    .foregroundStyle(.white)
                    .symbolSize(100)
                }
            }
            .frame(height: 200)
            .chartXAxis {
                AxisMarks(values: .automatic) { value in
                    AxisGridLine()
                    AxisValueLabel(format: timeAxisFormat)
                }
            }
            .chartYAxis {
                AxisMarks(values: .stride(by: 2)) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            .chartYScale(domain: 0...10)
            .chartAngleSelection(value: .constant(nil))
            .chartBackground { chartProxy in
                GeometryReader { geometry in
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            selectDataPoint(at: location, in: geometry, chartProxy: chartProxy)
                        }
                }
            }
            .padding(.horizontal)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var timeAxisFormat: Date.FormatStyle {
        switch timeRange {
        case .day:
            return .dateTime.hour()
        case .week:
            return .dateTime.weekday(.abbreviated)
        case .month:
            return .dateTime.day()
        case .quarter, .year:
            return .dateTime.month(.abbreviated)
        }
    }
    
    private func colorForPainLevel(_ level: Double) -> Color {
        switch level {
        case 0...2: return .green
        case 3...4: return .yellow
        case 5...6: return .orange
        case 7...8: return .red.opacity(0.8)
        default: return .red
        }
    }
    
    private func selectDataPoint(at location: CGPoint, in geometry: GeometryProxy, chartProxy: ChartProxy) {
        // Implementation for selecting data points on chart
        // This would require more complex chart interaction logic
    }
}

struct PainStatisticsView: View {
    let data: [PainDataPoint]
    let selectedRegions: Set<BodyRegion>
    
    private var statistics: PainStatistics {
        PainStatistics(from: data)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Statistics")
                .font(.headline)
                .padding(.horizontal)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 15) {
                StatCard(
                    title: "Average Pain",
                    value: String(format: "%.1f", statistics.averagePain),
                    subtitle: "out of 10",
                    color: colorForPainLevel(statistics.averagePain),
                    icon: "chart.line.uptrend.xyaxis"
                )
                
                StatCard(
                    title: "Peak Pain",
                    value: String(format: "%.0f", statistics.maxPain),
                    subtitle: "highest recorded",
                    color: .red,
                    icon: "exclamationmark.triangle.fill"
                )
                
                StatCard(
                    title: "Pain-Free Days",
                    value: "\(statistics.painFreeDays)",
                    subtitle: "in period",
                    color: .green,
                    icon: "checkmark.circle.fill"
                )
                
                StatCard(
                    title: "Trend",
                    value: statistics.trendDirection,
                    subtitle: statistics.trendPercentage,
                    color: statistics.trendColor,
                    icon: statistics.trendIcon
                )
            }
            .padding(.horizontal)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private func colorForPainLevel(_ level: Double) -> Color {
        switch level {
        case 0...2: return .green
        case 3...4: return .yellow
        case 5...6: return .orange
        case 7...8: return .red.opacity(0.8)
        default: return .red
        }
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let subtitle: String
    let color: Color
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title3)
                
                Spacer()
            }
            
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.primary)
            
            Text(subtitle)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(color.opacity(0.1))
        )
    }
}

struct RegionAnalysisView: View {
    let data: [PainDataPoint]
    let regions: Set<BodyRegion>
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Region Analysis")
                .font(.headline)
                .padding(.horizontal)
            
            ForEach(Array(regions), id: \.self) { region in
                RegionPainCard(
                    region: region,
                    data: data.compactMap { $0.regionIntensity[region] }
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
}

struct RegionPainCard: View {
    let region: BodyRegion
    let data: [Double]
    
    private var averagePain: Double {
        data.isEmpty ? 0 : data.reduce(0, +) / Double(data.count)
    }
    
    private var maxPain: Double {
        data.max() ?? 0
    }
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(region.displayName)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("Avg: \(String(format: "%.1f", averagePain))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(String(format: "%.0f", maxPain))")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(colorForPainLevel(maxPain))
                
                Text("peak")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.systemGray6))
        )
        .padding(.horizontal)
    }
    
    private func colorForPainLevel(_ level: Double) -> Color {
        switch level {
        case 0...2: return .green
        case 3...4: return .yellow
        case 5...6: return .orange
        case 7...8: return .red.opacity(0.8)
        default: return .red
        }
    }
}

struct AIPainInsightsView: View {
    let data: [PainDataPoint]
    let insights: [PainInsight]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.blue)
                Text("AI Insights")
                    .font(.headline)
            }
            .padding(.horizontal)
            
            ForEach(insights, id: \.id) { insight in
                InsightCard(insight: insight)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
}

struct InsightCard: View {
    let insight: PainInsight
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: insight.icon)
                .foregroundColor(insight.color)
                .font(.title3)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(insight.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(insight.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                
                if let recommendation = insight.recommendation {
                    Text(recommendation)
                        .font(.caption)
                        .foregroundColor(.blue)
                        .padding(.top, 2)
                }
            }
            
            Spacer()
            
            Text("\(Int(insight.confidence * 100))%")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(insight.color.opacity(0.1))
        )
        .padding(.horizontal)
    }
}

struct RecentPainEntriesView: View {
    let entries: [PainDataPoint]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Recent Entries")
                .font(.headline)
                .padding(.horizontal)
            
            ForEach(entries, id: \.id) { entry in
                PainEntryRow(entry: entry)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
}

struct PainEntryRow: View {
    let entry: PainDataPoint
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(entry.timestamp, style: .date)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(entry.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(String(format: "%.0f", entry.averagePainLevel))")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(colorForPainLevel(entry.averagePainLevel))
                
                Text("\(entry.affectedRegions.count) regions")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.systemGray6))
        )
        .padding(.horizontal)
    }
    
    private func colorForPainLevel(_ level: Double) -> Color {
        switch level {
        case 0...2: return .green
        case 3...4: return .yellow
        case 5...6: return .orange
        case 7...8: return .red.opacity(0.8)
        default: return .red
        }
    }
}

struct EmptyHistoryView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "chart.line.downtrend.xyaxis")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Pain History")
                .font(.title2)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
            
            Text("Start tracking your pain to see trends and insights here.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(height: 200)
    }
}

// MARK: - Data Models

struct PainDataPoint {
    let id = UUID()
    let timestamp: Date
    let averagePainLevel: Double
    let maxPainLevel: Double
    let affectedRegions: Set<BodyRegion>
    let regionIntensity: [BodyRegion: Double]
    let notes: String?
    let triggers: [String]
    let medications: [String]
}

struct PainStatistics {
    let averagePain: Double
    let maxPain: Double
    let minPain: Double
    let painFreeDays: Int
    let trendDirection: String
    let trendPercentage: String
    let trendColor: Color
    let trendIcon: String
    
    init(from data: [PainDataPoint]) {
        let painLevels = data.map { $0.averagePainLevel }
        
        self.averagePain = painLevels.isEmpty ? 0 : painLevels.reduce(0, +) / Double(painLevels.count)
        self.maxPain = painLevels.max() ?? 0
        self.minPain = painLevels.min() ?? 0
        self.painFreeDays = data.filter { $0.averagePainLevel == 0 }.count
        
        // Calculate trend
        if data.count >= 2 {
            let recentAvg = Array(painLevels.suffix(3)).reduce(0, +) / Double(min(3, painLevels.count))
            let olderAvg = Array(painLevels.prefix(3)).reduce(0, +) / Double(min(3, painLevels.count))
            let change = recentAvg - olderAvg
            
            if abs(change) < 0.5 {
                self.trendDirection = "Stable"
                self.trendColor = .blue
                self.trendIcon = "minus"
            } else if change > 0 {
                self.trendDirection = "Increasing"
                self.trendColor = .red
                self.trendIcon = "arrow.up"
            } else {
                self.trendDirection = "Decreasing"
                self.trendColor = .green
                self.trendIcon = "arrow.down"
            }
            
            self.trendPercentage = String(format: "%.1f%%", abs(change / olderAvg * 100))
        } else {
            self.trendDirection = "N/A"
            self.trendPercentage = ""
            self.trendColor = .gray
            self.trendIcon = "questionmark"
        }
    }
}

struct PainInsight {
    let id = UUID()
    let title: String
    let description: String
    let recommendation: String?
    let confidence: Double
    let color: Color
    let icon: String
    let priority: Int
}