//
//  BodyInsightsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Charts

struct BodyInsightsView: View {
    @ObservedObject var manager: BodyMappingManager
    @State private var selectedTimeRange: TimeRange = .week
    @State private var selectedInsightCategory: BodyMappingInsight.InsightCategory? = nil
    @State private var showingInsightDetail = false
    @State private var selectedInsight: BodyMappingInsight?
    
    enum TimeRange: String, CaseIterable {
        case week = "Week"
        case month = "Month"
        case quarter = "3 Months"
        case year = "Year"
        
        var days: Int {
            switch self {
            case .week: return 7
            case .month: return 30
            case .quarter: return 90
            case .year: return 365
            }
        }
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header with time range selector
                headerSection
                
                // Insights overview
                insightsOverview
                
                // Category filter
                categoryFilter
                
                // Insights list
                insightsList
                
                // Pain trends chart
                painTrendsChart
                
                // Regional analysis
                regionalAnalysis
                
                // AI recommendations
                aiRecommendations
            }
            .padding()
        }
        .navigationTitle("Body Insights")
        .sheet(isPresented: $showingInsightDetail) {
            if let insight = selectedInsight {
                InsightDetailView(insight: insight, manager: manager)
            }
        }
        .onAppear {
            manager.generateInsights()
        }
    }
    
    @ViewBuilder
    private var headerSection: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Health Insights")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: {
                    manager.generateInsights()
                }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.title2)
                }
            }
            
            // Time range picker
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
        }
    }
    
    @ViewBuilder
    private var insightsOverview: some View {
        VStack(spacing: 16) {
            Text("Overview")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                InsightMetricCard(
                    title: "Total Insights",
                    value: "\(filteredInsights.count)",
                    icon: "lightbulb",
                    color: .blue
                )
                
                InsightMetricCard(
                    title: "High Priority",
                    value: "\(highPriorityInsights.count)",
                    icon: "exclamationmark.triangle",
                    color: .red
                )
                
                InsightMetricCard(
                    title: "Avg Pain Level",
                    value: String(format: "%.1f", averagePainLevel),
                    icon: "chart.line.uptrend.xyaxis",
                    color: .orange
                )
                
                InsightMetricCard(
                    title: "Improvement",
                    value: "\(improvementPercentage)%",
                    icon: "arrow.up.right",
                    color: .green
                )
            }
        }
    }
    
    @ViewBuilder
    private var categoryFilter: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Categories")
                    .font(.headline)
                
                Spacer()
                
                if selectedInsightCategory != nil {
                    Button("Clear") {
                        selectedInsightCategory = nil
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
            }
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(BodyMappingInsight.InsightCategory.allCases, id: \.self) { category in
                        CategoryChip(
                            category: category,
                            isSelected: selectedInsightCategory == category
                        ) {
                            selectedInsightCategory = selectedInsightCategory == category ? nil : category
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
    }
    
    @ViewBuilder
    private var insightsList: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Recent Insights")
                    .font(.headline)
                
                Spacer()
                
                Text("\(filteredInsights.count) insights")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            if filteredInsights.isEmpty {
                EmptyInsightsView()
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(filteredInsights.prefix(10), id: \.id) { insight in
                        InsightRowView(insight: insight) {
                            selectedInsight = insight
                            showingInsightDetail = true
                        }
                    }
                }
            }
        }
    }
    
    @ViewBuilder
    private var painTrendsChart: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Pain Trends")
                    .font(.headline)
                
                Spacer()
                
                Text(selectedTimeRange.rawValue)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            if !painTrendData.isEmpty {
                Chart(painTrendData, id: \.date) { dataPoint in
                    LineMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Pain Level", dataPoint.painLevel)
                    )
                    .foregroundStyle(.red)
                    .interpolationMethod(.catmullRom)
                    
                    AreaMark(
                        x: .value("Date", dataPoint.date),
                        y: .value("Pain Level", dataPoint.painLevel)
                    )
                    .foregroundStyle(.red.opacity(0.1))
                    .interpolationMethod(.catmullRom)
                }
                .frame(height: 200)
                .chartYScale(domain: 0...4)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: selectedTimeRange == .week ? 1 : 7)) {
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.weekday(.abbreviated))
                    }
                }
                .chartYAxis {
                    AxisMarks(values: [0, 1, 2, 3, 4]) {
                        AxisGridLine()
                        AxisValueLabel()
                    }
                }
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.1))
                    .frame(height: 200)
                    .overlay(
                        Text("No pain data available")
                            .foregroundColor(.secondary)
                    )
            }
        }
    }
    
    @ViewBuilder
    private var regionalAnalysis: some View {
        VStack(spacing: 12) {
            Text("Regional Analysis")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            let regionData = getRegionalPainData()
            
            if !regionData.isEmpty {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: 12) {
                    ForEach(regionData.prefix(6), id: \.region) { data in
                        RegionalPainCard(
                            region: data.region,
                            averagePain: data.averagePain,
                            trend: data.trend
                        )
                    }
                }
            } else {
                Text("No regional data available")
                    .foregroundColor(.secondary)
                    .italic()
            }
        }
    }
    
    @ViewBuilder
    private var aiRecommendations: some View {
        VStack(spacing: 12) {
            Text("AI Recommendations")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            let recommendations = getAIRecommendations()
            
            if !recommendations.isEmpty {
                LazyVStack(spacing: 8) {
                    ForEach(recommendations.prefix(3), id: \.id) { recommendation in
                        AIRecommendationCard(recommendation: recommendation)
                    }
                }
            } else {
                Text("No recommendations available")
                    .foregroundColor(.secondary)
                    .italic()
            }
        }
    }
    
    // MARK: - Computed Properties
    
    private var filteredInsights: [BodyMappingInsight] {
        var insights = manager.insights
        
        if let category = selectedInsightCategory {
            insights = insights.filter { $0.category == category }
        }
        
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        insights = insights.filter { $0.timestamp >= cutoffDate }
        
        return insights.sorted { $0.timestamp > $1.timestamp }
    }
    
    private var highPriorityInsights: [BodyMappingInsight] {
        filteredInsights.filter { $0.severity == .high }
    }
    
    private var averagePainLevel: Double {
        let recentEntries = manager.painEntries.filter {
            let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
            return $0.timestamp >= cutoffDate
        }
        
        guard !recentEntries.isEmpty else { return 0.0 }
        
        let totalPain = recentEntries.reduce(0.0) { sum, entry in
            sum + Double(entry.painLevel.rawValue)
        }
        
        return totalPain / Double(recentEntries.count)
    }
    
    private var improvementPercentage: Int {
        // Calculate improvement based on pain trend
        let recentData = painTrendData.suffix(7)
        let olderData = painTrendData.prefix(7)
        
        guard !recentData.isEmpty && !olderData.isEmpty else { return 0 }
        
        let recentAvg = recentData.reduce(0.0) { $0 + $1.painLevel } / Double(recentData.count)
        let olderAvg = olderData.reduce(0.0) { $0 + $1.painLevel } / Double(olderData.count)
        
        guard olderAvg > 0 else { return 0 }
        
        let improvement = (olderAvg - recentAvg) / olderAvg * 100
        return max(0, Int(improvement))
    }
    
    private var painTrendData: [PainTrendDataPoint] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        let recentEntries = manager.painEntries.filter { $0.timestamp >= cutoffDate }
        
        // Group by day and calculate average pain
        let calendar = Calendar.current
        let groupedEntries = Dictionary(grouping: recentEntries) { entry in
            calendar.startOfDay(for: entry.timestamp)
        }
        
        return groupedEntries.map { date, entries in
            let averagePain = entries.reduce(0.0) { sum, entry in
                sum + Double(entry.painLevel.rawValue)
            } / Double(entries.count)
            
            return PainTrendDataPoint(date: date, painLevel: averagePain)
        }.sorted { $0.date < $1.date }
    }
    
    private func getRegionalPainData() -> [RegionalPainData] {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -selectedTimeRange.days, to: Date()) ?? Date()
        let recentEntries = manager.painEntries.filter { $0.timestamp >= cutoffDate }
        
        let regionGroups = Dictionary(grouping: recentEntries) { $0.bodyRegion.name }
        
        return regionGroups.compactMap { regionName, entries in
            guard !entries.isEmpty else { return nil }
            
            let averagePain = entries.reduce(0.0) { sum, entry in
                sum + Double(entry.painLevel.rawValue)
            } / Double(entries.count)
            
            // Calculate trend (simplified)
            let trend: TrendDirection = entries.count > 1 ? .stable : .stable
            
            return RegionalPainData(
                region: regionName,
                averagePain: averagePain,
                trend: trend
            )
        }.sorted { $0.averagePain > $1.averagePain }
    }
    
    private func getAIRecommendations() -> [AIRecommendation] {
        // Generate AI recommendations based on insights and pain data
        var recommendations: [AIRecommendation] = []
        
        if averagePainLevel > 2.0 {
            recommendations.append(AIRecommendation(
                id: UUID(),
                title: "Consider Pain Management Consultation",
                description: "Your average pain levels have been elevated. Consider discussing pain management strategies with your healthcare provider.",
                category: .medical,
                priority: .high,
                confidence: 0.85,
                actionItems: [
                    ActionItem(id: UUID(), title: "Schedule appointment with rheumatologist", completed: false),
                    ActionItem(id: UUID(), title: "Prepare pain diary for consultation", completed: false)
                ],
                timestamp: Date()
            ))
        }
        
        if improvementPercentage < 10 {
            recommendations.append(AIRecommendation(
                id: UUID(),
                title: "Adjust Treatment Approach",
                description: "Your pain levels haven't shown significant improvement. Consider adjusting your current treatment plan.",
                category: .treatment,
                priority: .medium,
                confidence: 0.75,
                actionItems: [
                    ActionItem(id: UUID(), title: "Review current medications", completed: false),
                    ActionItem(id: UUID(), title: "Explore alternative therapies", completed: false)
                ],
                timestamp: Date()
            ))
        }
        
        return recommendations
    }
}

struct InsightMetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }
}

struct CategoryChip: View {
    let category: BodyMappingInsight.InsightCategory
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(category.displayName)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(isSelected ? Color.blue : Color.gray.opacity(0.2))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(16)
        }
    }
}

struct InsightRowView: View {
    let insight: BodyMappingInsight
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                // Severity indicator
                Circle()
                    .fill(insight.severity.color)
                    .frame(width: 12, height: 12)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(insight.title)
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text(insight.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 2) {
                    Text(insight.category.displayName)
                        .font(.caption)
                        .foregroundColor(.blue)
                    
                    Text(insight.timestamp, style: .relative)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color.gray.opacity(0.05))
            .cornerRadius(8)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct EmptyInsightsView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "lightbulb")
                .font(.system(size: 48))
                .foregroundColor(.gray)
            
            VStack(spacing: 8) {
                Text("No Insights Available")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                Text("Keep tracking your symptoms to generate personalized insights")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
    }
}

struct RegionalPainCard: View {
    let region: String
    let averagePain: Double
    let trend: TrendDirection
    
    enum TrendDirection {
        case improving, stable, worsening
        
        var icon: String {
            switch self {
            case .improving: return "arrow.down.right"
            case .stable: return "arrow.right"
            case .worsening: return "arrow.up.right"
            }
        }
        
        var color: Color {
            switch self {
            case .improving: return .green
            case .stable: return .yellow
            case .worsening: return .red
            }
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(region)
                    .font(.caption)
                    .fontWeight(.medium)
                
                Spacer()
                
                Image(systemName: trend.icon)
                    .foregroundColor(trend.color)
                    .font(.caption)
            }
            
            HStack {
                Text(String(format: "%.1f", averagePain))
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Text("/4.0")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            ProgressView(value: averagePain / 4.0)
                .progressViewStyle(LinearProgressViewStyle(tint: painColor))
        }
        .padding()
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }
    
    private var painColor: Color {
        switch averagePain {
        case 0...1: return .green
        case 1...2: return .yellow
        case 2...3: return .orange
        default: return .red
        }
    }
}

struct AIRecommendationCard: View {
    let recommendation: AIRecommendation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.purple)
                    .font(.title2)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(recommendation.title)
                        .font(.headline)
                    
                    Text(recommendation.category.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Circle()
                    .fill(recommendation.priority.color)
                    .frame(width: 8, height: 8)
            }
            
            Text(recommendation.description)
                .font(.body)
                .foregroundColor(.secondary)
            
            HStack {
                Text("Confidence: \(Int(recommendation.confidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(recommendation.timestamp, style: .relative)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.purple.opacity(0.05))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.purple.opacity(0.2), lineWidth: 1)
        )
    }
}

struct InsightDetailView: View {
    let insight: BodyMappingInsight
    @ObservedObject var manager: BodyMappingManager
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Circle()
                                .fill(insight.severity.color)
                                .frame(width: 16, height: 16)
                            
                            Text(insight.category.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Spacer()
                            
                            Text(insight.timestamp, style: .date)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Text(insight.title)
                            .font(.largeTitle)
                            .fontWeight(.bold)
                    }
                    
                    // Description
                    Text(insight.description)
                        .font(.body)
                    
                    // Recommendations
                    if !insight.recommendations.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Recommendations")
                                .font(.headline)
                            
                            ForEach(insight.recommendations, id: \.self) { recommendation in
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "lightbulb")
                                        .foregroundColor(.yellow)
                                        .font(.caption)
                                    
                                    Text(recommendation)
                                        .font(.body)
                                }
                            }
                        }
                    }
                    
                    // Related data
                    if !insight.relatedDataPoints.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Related Data")
                                .font(.headline)
                            
                            ForEach(insight.relatedDataPoints, id: \.self) { dataPoint in
                                Text("• \(dataPoint)")
                                    .font(.body)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Insight Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct PainTrendDataPoint {
    let date: Date
    let painLevel: Double
}

struct RegionalPainData {
    let region: String
    let averagePain: Double
    let trend: RegionalPainCard.TrendDirection
}

struct AIRecommendation {
    let id: UUID
    let title: String
    let description: String
    let category: RecommendationCategory
    let priority: RecommendationPriority
    let confidence: Double
    let actionItems: [ActionItem]
    let timestamp: Date
    
    enum RecommendationCategory: String, CaseIterable {
        case medical = "Medical"
        case lifestyle = "Lifestyle"
        case treatment = "Treatment"
        case exercise = "Exercise"
        case nutrition = "Nutrition"
        
        var displayName: String { rawValue }
    }
    
    enum RecommendationPriority: String, CaseIterable {
        case low = "Low"
        case medium = "Medium"
        case high = "High"
        
        var color: Color {
            switch self {
            case .low: return .green
            case .medium: return .yellow
            case .high: return .red
            }
        }
    }
}

struct ActionItem {
    let id: UUID
    let title: String
    var completed: Bool
}

// MARK: - Extensions

extension BodyMappingInsight.InsightSeverity {
    var color: Color {
        switch self {
        case .low: return .green
        case .medium: return .yellow
        case .high: return .red
        }
    }
}

extension BodyMappingInsight.InsightCategory {
    var displayName: String {
        switch self {
        case .painPattern: return "Pain Pattern"
        case .postureAnalysis: return "Posture"
        case .movementTrend: return "Movement"
        case .correlationInsight: return "Correlation"
        case .predictionAlert: return "Prediction"
        case .treatmentEffectiveness: return "Treatment"
        }
    }
}

#Preview {
    BodyInsightsView(manager: BodyMappingManager())
}