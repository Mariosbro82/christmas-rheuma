//
//  InsightDetailView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct InsightDetailView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @Environment(\.dismiss) private var dismiss
    let insight: AIInsight
    @State private var selectedTab: DetailTab = .overview
    @State private var showingActionSheet = false
    @State private var showingShareSheet = false
    
    enum DetailTab: String, CaseIterable {
        case overview = "Overview"
        case analysis = "Analysis"
        case recommendations = "Actions"
        case history = "History"
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header section
                    headerSection
                    
                    // Tab selector
                    tabSelector
                    
                    // Content based on selected tab
                    switch selectedTab {
                    case .overview:
                        overviewSection
                    case .analysis:
                        analysisSection
                    case .recommendations:
                        recommendationsSection
                    case .history:
                        historySection
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Insight Details")
            .navigationBarTitleDisplayMode(.inline)
            .themedBackground()
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                    .foregroundColor(themeManager.colors.primary)
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Share Insight", systemImage: "square.and.arrow.up") {
                            showingShareSheet = true
                        }
                        
                        Button("Mark as Read", systemImage: "checkmark.circle") {
                            markAsRead()
                        }
                        
                        Button("Set Reminder", systemImage: "bell") {
                            setReminder()
                        }
                        
                        Button("Report Issue", systemImage: "exclamationmark.triangle") {
                            reportIssue()
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .foregroundColor(themeManager.colors.primary)
                    }
                }
            }
            .confirmationDialog("Share Insight", isPresented: $showingShareSheet) {
                Button("Export as PDF") { exportAsPDF() }
                Button("Share with Doctor") { shareWithDoctor() }
                Button("Copy to Clipboard") { copyToClipboard() }
                Button("Cancel", role: .cancel) { }
            }
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 16) {
            // Category and priority
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: categoryIcon(insight.category))
                        .foregroundColor(categoryColor(insight.category))
                        .font(.title2)
                    
                    Text(categoryName(insight.category))
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                }
                
                Spacer()
                
                // Priority badge
                Text(priorityText(insight.priority))
                    .font(themeManager.typography.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(priorityColor(insight.priority))
                    )
                    .foregroundColor(.white)
            }
            
            // Title and description
            VStack(alignment: .leading, spacing: 8) {
                Text(insight.title)
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                    .multilineTextAlignment(.leading)
                
                Text(insight.description)
                    .font(themeManager.typography.body)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(nil)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            
            // Metrics row
            HStack(spacing: 20) {
                MetricItem(
                    title: "Confidence",
                    value: "\(Int(insight.confidence * 100))%",
                    color: confidenceColor(insight.confidence)
                )
                
                MetricItem(
                    title: "Relevance",
                    value: relevanceScore(),
                    color: .blue
                )
                
                MetricItem(
                    title: "Impact",
                    value: impactLevel(),
                    color: .purple
                )
                
                MetricItem(
                    title: "Urgency",
                    value: urgencyLevel(),
                    color: urgencyColor()
                )
            }
        }
        .padding(.horizontal)
    }
    
    private var tabSelector: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 0) {
                ForEach(DetailTab.allCases, id: \.self) { tab in
                    Button {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            selectedTab = tab
                        }
                    } label: {
                        VStack(spacing: 8) {
                            Text(tab.rawValue)
                                .font(themeManager.typography.body)
                                .fontWeight(selectedTab == tab ? .semibold : .regular)
                                .foregroundColor(selectedTab == tab ? themeManager.colors.primary : themeManager.colors.textSecondary)
                            
                            Rectangle()
                                .fill(selectedTab == tab ? themeManager.colors.primary : Color.clear)
                                .frame(height: 2)
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
            }
            .padding(.horizontal)
        }
    }
    
    private var overviewSection: some View {
        VStack(spacing: 20) {
            // Key findings
            keyFindingsCard
            
            // Data sources
            dataSourcesCard
            
            // Timeline
            timelineCard
            
            // Related insights
            relatedInsightsCard
        }
        .padding(.horizontal)
    }
    
    private var analysisSection: some View {
        VStack(spacing: 20) {
            // Statistical analysis
            statisticalAnalysisCard
            
            // Data visualization
            dataVisualizationCard
            
            // Correlation matrix
            correlationMatrixCard
            
            // Trend analysis
            trendAnalysisCard
        }
        .padding(.horizontal)
    }
    
    private var recommendationsSection: some View {
        VStack(spacing: 20) {
            // Immediate actions
            immediateActionsCard
            
            // Long-term strategies
            longTermStrategiesCard
            
            // Monitoring recommendations
            monitoringRecommendationsCard
            
            // Follow-up schedule
            followUpScheduleCard
        }
        .padding(.horizontal)
    }
    
    private var historySection: some View {
        VStack(spacing: 20) {
            // Similar insights
            similarInsightsCard
            
            // Trend over time
            historicalTrendCard
            
            // Effectiveness tracking
            effectivenessTrackingCard
        }
        .padding(.horizontal)
    }
    
    // MARK: - Overview Cards
    
    private var keyFindingsCard: some View {
        CardContainer(title: "Key Findings", icon: "key.fill") {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(generateKeyFindings(), id: \.self) { finding in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.caption)
                            .padding(.top, 2)
                        
                        Text(finding)
                            .font(themeManager.typography.body)
                            .foregroundColor(themeManager.colors.textPrimary)
                            .lineLimit(nil)
                    }
                }
            }
        }
    }
    
    private var dataSourcesCard: some View {
        CardContainer(title: "Data Sources", icon: "doc.text.fill") {
            VStack(spacing: 12) {
                ForEach(generateDataSources(), id: \.name) { source in
                    HStack {
                        Image(systemName: source.icon)
                            .foregroundColor(source.color)
                            .font(.title3)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text(source.name)
                                .font(themeManager.typography.body)
                                .fontWeight(.medium)
                                .foregroundColor(themeManager.colors.textPrimary)
                            
                            Text(source.description)
                                .font(themeManager.typography.caption)
                                .foregroundColor(themeManager.colors.textSecondary)
                        }
                        
                        Spacer()
                        
                        Text(source.dataPoints)
                            .font(themeManager.typography.caption)
                            .foregroundColor(.blue)
                    }
                }
            }
        }
    }
    
    private var timelineCard: some View {
        CardContainer(title: "Timeline", icon: "clock.fill") {
            VStack(spacing: 16) {
                // Timeline chart
                Chart {
                    ForEach(generateTimelineData(), id: \.date) { point in
                        LineMark(
                            x: .value("Date", point.date),
                            y: .value("Relevance", point.relevance)
                        )
                        .foregroundStyle(.blue)
                        .lineStyle(StrokeStyle(lineWidth: 2))
                        
                        PointMark(
                            x: .value("Date", point.date),
                            y: .value("Relevance", point.relevance)
                        )
                        .foregroundStyle(.blue)
                        .symbolSize(30)
                    }
                }
                .frame(height: 120)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: 7)) { _ in
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.month().day())
                    }
                }
                .chartYAxis {
                    AxisMarks { _ in
                        AxisGridLine()
                        AxisValueLabel()
                    }
                }
                
                // Key events
                VStack(alignment: .leading, spacing: 8) {
                    Text("Key Events")
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    ForEach(generateKeyEvents(), id: \.date) { event in
                        HStack {
                            Circle()
                                .fill(event.color)
                                .frame(width: 8, height: 8)
                            
                            Text(event.description)
                                .font(themeManager.typography.caption)
                                .foregroundColor(themeManager.colors.textSecondary)
                            
                            Spacer()
                            
                            Text(event.date, style: .date)
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                        }
                    }
                }
            }
        }
    }
    
    private var relatedInsightsCard: some View {
        CardContainer(title: "Related Insights", icon: "link") {
            VStack(spacing: 12) {
                ForEach(generateRelatedInsights(), id: \.id) { relatedInsight in
                    HStack {
                        Image(systemName: categoryIcon(relatedInsight.category))
                            .foregroundColor(categoryColor(relatedInsight.category))
                            .font(.title3)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text(relatedInsight.title)
                                .font(themeManager.typography.body)
                                .fontWeight(.medium)
                                .foregroundColor(themeManager.colors.textPrimary)
                                .lineLimit(1)
                            
                            Text("\(Int(relatedInsight.confidence * 100))% confidence")
                                .font(themeManager.typography.caption)
                                .foregroundColor(confidenceColor(relatedInsight.confidence))
                        }
                        
                        Spacer()
                        
                        Image(systemName: "chevron.right")
                            .foregroundColor(themeManager.colors.textSecondary)
                            .font(.caption)
                    }
                }
            }
        }
    }
    
    // MARK: - Analysis Cards
    
    private var statisticalAnalysisCard: some View {
        CardContainer(title: "Statistical Analysis", icon: "chart.bar.fill") {
            VStack(spacing: 16) {
                // Statistical metrics
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                    StatMetric(title: "Sample Size", value: "1,247", unit: "data points")
                    StatMetric(title: "P-Value", value: "0.003", unit: "significant")
                    StatMetric(title: "Effect Size", value: "0.72", unit: "large")
                    StatMetric(title: "R-Squared", value: "0.68", unit: "variance")
                }
                
                // Significance indicator
                HStack {
                    Image(systemName: "checkmark.seal.fill")
                        .foregroundColor(.green)
                    
                    Text("Statistically significant result (p < 0.05)")
                        .font(themeManager.typography.caption)
                        .foregroundColor(.green)
                        .fontWeight(.medium)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.green.opacity(0.1))
                )
            }
        }
    }
    
    private var dataVisualizationCard: some View {
        CardContainer(title: "Data Visualization", icon: "chart.line.uptrend.xyaxis") {
            VStack(spacing: 16) {
                // Main chart
                Chart {
                    ForEach(generateVisualizationData(), id: \.category) { data in
                        BarMark(
                            x: .value("Category", data.category),
                            y: .value("Value", data.value)
                        )
                        .foregroundStyle(data.color)
                        .cornerRadius(4)
                    }
                }
                .frame(height: 150)
                .chartXAxis {
                    AxisMarks { _ in
                        AxisValueLabel()
                    }
                }
                .chartYAxis {
                    AxisMarks { _ in
                        AxisGridLine()
                        AxisValueLabel()
                    }
                }
                
                // Legend
                HStack {
                    ForEach(generateVisualizationData(), id: \.category) { data in
                        HStack(spacing: 4) {
                            Circle()
                                .fill(data.color)
                                .frame(width: 8, height: 8)
                            
                            Text(data.category)
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                        }
                    }
                }
            }
        }
    }
    
    private var correlationMatrixCard: some View {
        CardContainer(title: "Correlation Matrix", icon: "grid") {
            VStack(spacing: 12) {
                // Correlation grid
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 8) {
                    ForEach(generateCorrelationData(), id: \.id) { correlation in
                        VStack(spacing: 4) {
                            Text(correlation.factor1)
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                                .lineLimit(1)
                            
                            Text("vs")
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                            
                            Text(correlation.factor2)
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                                .lineLimit(1)
                            
                            Text(String(format: "%.2f", correlation.value))
                                .font(themeManager.typography.body)
                                .fontWeight(.bold)
                                .foregroundColor(correlationColor(correlation.value))
                        }
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(correlationColor(correlation.value).opacity(0.1))
                        )
                    }
                }
                
                // Interpretation guide
                VStack(alignment: .leading, spacing: 4) {
                    Text("Interpretation:")
                        .font(themeManager.typography.caption)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    HStack {
                        Text("Strong: |r| > 0.7")
                        Spacer()
                        Text("Moderate: 0.3 < |r| < 0.7")
                        Spacer()
                        Text("Weak: |r| < 0.3")
                    }
                    .font(themeManager.typography.caption2)
                    .foregroundColor(themeManager.colors.textSecondary)
                }
            }
        }
    }
    
    private var trendAnalysisCard: some View {
        CardContainer(title: "Trend Analysis", icon: "chart.line.uptrend.xyaxis") {
            VStack(spacing: 16) {
                // Trend chart
                Chart {
                    ForEach(generateTrendData(), id: \.date) { point in
                        LineMark(
                            x: .value("Date", point.date),
                            y: .value("Value", point.value)
                        )
                        .foregroundStyle(.blue)
                        .lineStyle(StrokeStyle(lineWidth: 2))
                        
                        // Trend line
                        LineMark(
                            x: .value("Date", point.date),
                            y: .value("Trend", point.trend)
                        )
                        .foregroundStyle(.red)
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [5]))
                    }
                }
                .frame(height: 120)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: 7)) { _ in
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.month().day())
                    }
                }
                .chartYAxis {
                    AxisMarks { _ in
                        AxisGridLine()
                        AxisValueLabel()
                    }
                }
                
                // Trend summary
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Trend Direction")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.up.right")
                                .foregroundColor(.green)
                            Text("Improving")
                                .font(themeManager.typography.body)
                                .fontWeight(.medium)
                                .foregroundColor(.green)
                        }
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("Rate of Change")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("+12.5% per week")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(.blue)
                    }
                }
            }
        }
    }
    
    // MARK: - Recommendations Cards
    
    private var immediateActionsCard: some View {
        CardContainer(title: "Immediate Actions", icon: "bolt.fill") {
            VStack(spacing: 12) {
                ForEach(generateImmediateActions(), id: \.id) { action in
                    ActionItem(action: action)
                }
            }
        }
    }
    
    private var longTermStrategiesCard: some View {
        CardContainer(title: "Long-term Strategies", icon: "target") {
            VStack(spacing: 12) {
                ForEach(generateLongTermStrategies(), id: \.id) { strategy in
                    StrategyItem(strategy: strategy)
                }
            }
        }
    }
    
    private var monitoringRecommendationsCard: some View {
        CardContainer(title: "Monitoring Recommendations", icon: "eye.fill") {
            VStack(spacing: 12) {
                ForEach(generateMonitoringRecommendations(), id: \.id) { recommendation in
                    MonitoringItem(recommendation: recommendation)
                }
            }
        }
    }
    
    private var followUpScheduleCard: some View {
        CardContainer(title: "Follow-up Schedule", icon: "calendar") {
            VStack(spacing: 12) {
                ForEach(generateFollowUpSchedule(), id: \.id) { followUp in
                    FollowUpItem(followUp: followUp)
                }
            }
        }
    }
    
    // MARK: - History Cards
    
    private var similarInsightsCard: some View {
        CardContainer(title: "Similar Insights", icon: "doc.on.doc.fill") {
            VStack(spacing: 12) {
                ForEach(generateSimilarInsights(), id: \.id) { similarInsight in
                    SimilarInsightItem(insight: similarInsight)
                }
            }
        }
    }
    
    private var historicalTrendCard: some View {
        CardContainer(title: "Historical Trend", icon: "chart.line.uptrend.xyaxis") {
            VStack(spacing: 16) {
                // Historical chart
                Chart {
                    ForEach(generateHistoricalData(), id: \.date) { point in
                        LineMark(
                            x: .value("Date", point.date),
                            y: .value("Frequency", point.frequency)
                        )
                        .foregroundStyle(.purple)
                        .lineStyle(StrokeStyle(lineWidth: 2))
                        
                        AreaMark(
                            x: .value("Date", point.date),
                            y: .value("Frequency", point.frequency)
                        )
                        .foregroundStyle(.purple.opacity(0.3))
                    }
                }
                .frame(height: 120)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .month, count: 1)) { _ in
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.month(.abbreviated))
                    }
                }
                .chartYAxis {
                    AxisMarks { _ in
                        AxisGridLine()
                        AxisValueLabel()
                    }
                }
                
                // Historical summary
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("First Occurrence")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("3 months ago")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("Frequency")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("2.3x per month")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                }
            }
        }
    }
    
    private var effectivenessTrackingCard: some View {
        CardContainer(title: "Effectiveness Tracking", icon: "chart.bar.xaxis") {
            VStack(spacing: 16) {
                // Effectiveness metrics
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                    EffectivenessMetric(title: "Actions Taken", value: "8/12", percentage: 67)
                    EffectivenessMetric(title: "Improvement", value: "23%", percentage: 23)
                    EffectivenessMetric(title: "Adherence", value: "85%", percentage: 85)
                    EffectivenessMetric(title: "Satisfaction", value: "4.2/5", percentage: 84)
                }
                
                // Overall effectiveness
                VStack(spacing: 8) {
                    HStack {
                        Text("Overall Effectiveness")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.textPrimary)
                        
                        Spacer()
                        
                        Text("Good")
                            .font(themeManager.typography.body)
                            .fontWeight(.bold)
                            .foregroundColor(.green)
                    }
                    
                    ProgressView(value: 0.75)
                        .progressViewStyle(LinearProgressViewStyle(tint: .green))
                        .scaleEffect(x: 1, y: 2, anchor: .center)
                }
            }
        }
    }
    
    // MARK: - Helper Functions
    
    private func categoryIcon(_ category: AIInsight.Category) -> String {
        switch category {
        case .pattern:
            return "chart.bar.fill"
        case .prediction:
            return "crystal.ball.fill"
        case .recommendation:
            return "lightbulb.fill"
        case .correlation:
            return "link"
        case .anomaly:
            return "exclamationmark.triangle.fill"
        }
    }
    
    private func categoryColor(_ category: AIInsight.Category) -> Color {
        switch category {
        case .pattern:
            return .blue
        case .prediction:
            return .purple
        case .recommendation:
            return .green
        case .correlation:
            return .orange
        case .anomaly:
            return .red
        }
    }
    
    private func categoryName(_ category: AIInsight.Category) -> String {
        switch category {
        case .pattern:
            return "Pattern Analysis"
        case .prediction:
            return "Prediction"
        case .recommendation:
            return "Recommendation"
        case .correlation:
            return "Correlation"
        case .anomaly:
            return "Anomaly Detection"
        }
    }
    
    private func priorityText(_ priority: AIInsight.Priority) -> String {
        switch priority {
        case .low:
            return "LOW PRIORITY"
        case .medium:
            return "MEDIUM PRIORITY"
        case .high:
            return "HIGH PRIORITY"
        }
    }
    
    private func priorityColor(_ priority: AIInsight.Priority) -> Color {
        switch priority {
        case .low:
            return .green
        case .medium:
            return .orange
        case .high:
            return .red
        }
    }
    
    private func confidenceColor(_ confidence: Double) -> Color {
        switch confidence {
        case 0.8...1.0:
            return .green
        case 0.6..<0.8:
            return .orange
        case 0.4..<0.6:
            return .yellow
        default:
            return .red
        }
    }
    
    private func correlationColor(_ value: Double) -> Color {
        let absValue = abs(value)
        switch absValue {
        case 0.7...1.0:
            return value > 0 ? .green : .red
        case 0.3..<0.7:
            return .orange
        default:
            return .gray
        }
    }
    
    private func relevanceScore() -> String {
        let score = Int.random(in: 75...95)
        return "\(score)%"
    }
    
    private func impactLevel() -> String {
        ["High", "Medium", "Low"].randomElement() ?? "Medium"
    }
    
    private func urgencyLevel() -> String {
        ["Urgent", "Soon", "Later"].randomElement() ?? "Soon"
    }
    
    private func urgencyColor() -> Color {
        switch urgencyLevel() {
        case "Urgent":
            return .red
        case "Soon":
            return .orange
        default:
            return .green
        }
    }
    
    // MARK: - Data Generation Functions
    
    private func generateKeyFindings() -> [String] {
        [
            "Pain levels show a 23% correlation with barometric pressure changes",
            "Morning stiffness duration has decreased by 15% over the past month",
            "Medication effectiveness is 18% higher when taken with food",
            "Sleep quality directly impacts next-day pain levels (r = -0.67)",
            "Exercise frequency correlates with reduced flare intensity"
        ]
    }
    
    private func generateDataSources() -> [DataSource] {
        [
            DataSource(name: "Pain Tracking", description: "Daily pain level recordings", icon: "heart.fill", color: .red, dataPoints: "247 entries"),
            DataSource(name: "Weather Data", description: "Barometric pressure & temperature", icon: "cloud.fill", color: .blue, dataPoints: "90 days"),
            DataSource(name: "Medication Log", description: "Medication timing & dosage", icon: "pills.fill", color: .green, dataPoints: "156 doses"),
            DataSource(name: "Sleep Tracking", description: "Sleep duration & quality", icon: "bed.double.fill", color: .purple, dataPoints: "85 nights")
        ]
    }
    
    private func generateTimelineData() -> [TimelinePoint] {
        let calendar = Calendar.current
        let endDate = Date()
        
        return (0..<30).compactMap { dayOffset in
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: endDate) else { return nil }
            return TimelinePoint(
                date: date,
                relevance: Double.random(in: 0.3...1.0)
            )
        }.reversed()
    }
    
    private func generateKeyEvents() -> [KeyEvent] {
        [
            KeyEvent(date: Date().addingTimeInterval(-86400 * 7), description: "High pain episode detected", color: .red),
            KeyEvent(date: Date().addingTimeInterval(-86400 * 14), description: "Medication adjustment", color: .blue),
            KeyEvent(date: Date().addingTimeInterval(-86400 * 21), description: "Weather pattern change", color: .orange),
            KeyEvent(date: Date().addingTimeInterval(-86400 * 28), description: "Exercise routine started", color: .green)
        ]
    }
    
    private func generateRelatedInsights() -> [AIInsight] {
        [
            AIInsight(
                id: UUID(),
                title: "Weather Impact on Joint Pain",
                description: "Barometric pressure changes correlate with increased stiffness",
                category: .correlation,
                priority: .medium,
                confidence: 0.78,
                timestamp: Date().addingTimeInterval(-86400 * 3),
                actionable: true
            ),
            AIInsight(
                id: UUID(),
                title: "Sleep Quality Prediction",
                description: "Poor sleep quality predicted for tomorrow based on stress levels",
                category: .prediction,
                priority: .high,
                confidence: 0.85,
                timestamp: Date().addingTimeInterval(-86400 * 1),
                actionable: true
            )
        ]
    }
    
    private func generateVisualizationData() -> [VisualizationData] {
        [
            VisualizationData(category: "Pain", value: 6.2, color: .red),
            VisualizationData(category: "Sleep", value: 7.8, color: .blue),
            VisualizationData(category: "Stress", value: 4.5, color: .orange),
            VisualizationData(category: "Activity", value: 8.1, color: .green)
        ]
    }
    
    private func generateCorrelationData() -> [CorrelationData] {
        [
            CorrelationData(id: UUID(), factor1: "Pain", factor2: "Weather", value: 0.73),
            CorrelationData(id: UUID(), factor1: "Sleep", factor2: "Pain", value: -0.65),
            CorrelationData(id: UUID(), factor1: "Stress", factor2: "Flares", value: 0.82),
            CorrelationData(id: UUID(), factor1: "Exercise", factor2: "Mood", value: 0.58),
            CorrelationData(id: UUID(), factor1: "Medication", factor2: "Relief", value: 0.71),
            CorrelationData(id: UUID(), factor1: "Diet", factor2: "Energy", value: 0.45)
        ]
    }
    
    private func generateTrendData() -> [TrendPoint] {
        let calendar = Calendar.current
        let endDate = Date()
        
        return (0..<30).compactMap { dayOffset in
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: endDate) else { return nil }
            let baseValue = 5.0 + sin(Double(dayOffset) * 0.2) * 2.0
            let trend = 5.0 + Double(dayOffset) * 0.05
            return TrendPoint(
                date: date,
                value: baseValue + Double.random(in: -0.5...0.5),
                trend: trend
            )
        }.reversed()
    }
    
    private func generateImmediateActions() -> [ActionItem] {
        [
            ActionItem(
                id: UUID(),
                title: "Take preventive medication",
                description: "Weather forecast shows pressure drop in 2 hours",
                priority: .high,
                estimatedTime: "5 minutes",
                completed: false
            ),
            ActionItem(
                id: UUID(),
                title: "Apply heat therapy",
                description: "Current stiffness level suggests immediate relief needed",
                priority: .medium,
                estimatedTime: "15 minutes",
                completed: false
            )
        ]
    }
    
    private func generateLongTermStrategies() -> [StrategyItem] {
        [
            StrategyItem(
                id: UUID(),
                title: "Optimize sleep schedule",
                description: "Consistent bedtime could reduce morning stiffness by 30%",
                timeframe: "2-4 weeks",
                expectedImpact: "High",
                difficulty: "Medium"
            ),
            StrategyItem(
                id: UUID(),
                title: "Weather-based medication timing",
                description: "Adjust medication schedule based on weather forecasts",
                timeframe: "1-2 weeks",
                expectedImpact: "Medium",
                difficulty: "Low"
            )
        ]
    }
    
    private func generateMonitoringRecommendations() -> [MonitoringItem] {
        [
            MonitoringItem(
                id: UUID(),
                title: "Track sleep quality",
                description: "Monitor sleep duration and quality for correlation analysis",
                frequency: "Daily",
                duration: "2 weeks",
                metrics: ["Duration", "Quality", "Interruptions"]
            ),
            MonitoringItem(
                id: UUID(),
                title: "Weather correlation tracking",
                description: "Record pain levels during weather changes",
                frequency: "As needed",
                duration: "1 month",
                metrics: ["Pain level", "Pressure", "Temperature"]
            )
        ]
    }
    
    private func generateFollowUpSchedule() -> [FollowUpItem] {
        [
            FollowUpItem(
                id: UUID(),
                title: "Review sleep optimization progress",
                date: Calendar.current.date(byAdding: .weekOfYear, value: 2, to: Date()) ?? Date(),
                type: "Progress Review",
                description: "Assess effectiveness of sleep schedule changes"
            ),
            FollowUpItem(
                id: UUID(),
                title: "Medication timing evaluation",
                date: Calendar.current.date(byAdding: .weekOfYear, value: 1, to: Date()) ?? Date(),
                type: "Medication Review",
                description: "Evaluate weather-based medication timing effectiveness"
            )
        ]
    }
    
    private func generateSimilarInsights() -> [SimilarInsightItem] {
        [
            SimilarInsightItem(
                id: UUID(),
                title: "Weather sensitivity pattern",
                similarity: 0.89,
                date: Date().addingTimeInterval(-86400 * 30),
                outcome: "Implemented successfully"
            ),
            SimilarInsightItem(
                id: UUID(),
                title: "Sleep-pain correlation",
                similarity: 0.76,
                date: Date().addingTimeInterval(-86400 * 45),
                outcome: "Partially effective"
            )
        ]
    }
    
    private func generateHistoricalData() -> [HistoricalPoint] {
        let calendar = Calendar.current
        let endDate = Date()
        
        return (0..<12).compactMap { monthOffset in
            guard let date = calendar.date(byAdding: .month, value: -monthOffset, to: endDate) else { return nil }
            return HistoricalPoint(
                date: date,
                frequency: Double.random(in: 1.0...5.0)
            )
        }.reversed()
    }
    
    // MARK: - Action Functions
    
    private func markAsRead() {
        // Implementation for marking insight as read
        print("Marking insight as read")
    }
    
    private func setReminder() {
        // Implementation for setting reminder
        print("Setting reminder")
    }
    
    private func reportIssue() {
        // Implementation for reporting issue
        print("Reporting issue")
    }
    
    private func exportAsPDF() {
        // Implementation for PDF export
        print("Exporting as PDF")
    }
    
    private func shareWithDoctor() {
        // Implementation for sharing with doctor
        print("Sharing with doctor")
    }
    
    private func copyToClipboard() {
        // Implementation for copying to clipboard
        print("Copying to clipboard")
    }
}

// MARK: - Supporting Views

struct CardContainer<Content: View>: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let icon: String
    let content: Content
    
    init(title: String, icon: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title3)
                
                Text(title)
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
            }
            
            content
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
    }
}

struct MetricItem: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(title)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
    }
}

struct StatMetric: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let unit: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
            
            Text(value)
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            Text(unit)
                .font(themeManager.typography.caption2)
                .foregroundColor(themeManager.colors.textSecondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(themeManager.colors.background)
        )
    }
}

struct EffectivenessMetric: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let percentage: Int
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
            
            Text(value)
                .font(themeManager.typography.body)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            ProgressView(value: Double(percentage) / 100.0)
                .progressViewStyle(LinearProgressViewStyle(tint: progressColor(percentage)))
                .scaleEffect(x: 1, y: 1.5, anchor: .center)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(themeManager.colors.background)
        )
    }
    
    private func progressColor(_ percentage: Int) -> Color {
        switch percentage {
        case 80...100:
            return .green
        case 60..<80:
            return .orange
        default:
            return .red
        }
    }
}

// MARK: - Data Models

struct DataSource {
    let name: String
    let description: String
    let icon: String
    let color: Color
    let dataPoints: String
}

struct TimelinePoint {
    let date: Date
    let relevance: Double
}

struct KeyEvent {
    let date: Date
    let description: String
    let color: Color
}

struct VisualizationData {
    let category: String
    let value: Double
    let color: Color
}

struct CorrelationData {
    let id: UUID
    let factor1: String
    let factor2: String
    let value: Double
}

struct TrendPoint {
    let date: Date
    let value: Double
    let trend: Double
}

struct ActionItem: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let priority: AIInsight.Priority
    let estimatedTime: String
    var completed: Bool
}

struct StrategyItem: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let timeframe: String
    let expectedImpact: String
    let difficulty: String
}

struct MonitoringItem: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let frequency: String
    let duration: String
    let metrics: [String]
}

struct FollowUpItem: Identifiable {
    let id: UUID
    let title: String
    let date: Date
    let type: String
    let description: String
}

struct SimilarInsightItem: Identifiable {
    let id: UUID
    let title: String
    let similarity: Double
    let date: Date
    let outcome: String
}

struct HistoricalPoint {
    let date: Date
    let frequency: Double
}

#Preview {
    InsightDetailView(
        insight: AIInsight(
            id: UUID(),
            title: "Weather Impact Analysis",
            description: "Your pain levels show a strong correlation with barometric pressure changes. This insight suggests implementing weather-based preventive measures.",
            category: .correlation,
            priority: .high,
            confidence: 0.85,
            timestamp: Date(),
            actionable: true
        )
    )
    .environmentObject(ThemeManager())
}