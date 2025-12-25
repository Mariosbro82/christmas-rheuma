//
//  InsightsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct InsightsView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @EnvironmentObject private var analyticsManager: AdvancedAnalyticsManager
    @State private var selectedTimeRange: TimeRange = .month
    @State private var selectedInsightType: InsightType = .all
    @State private var showingInsightDetail = false
    @State private var selectedInsight: AIInsight?
    @State private var isRefreshing = false
    
    enum TimeRange: String, CaseIterable {
        case week = "Week"
        case month = "Month"
        case quarter = "3 Months"
        case year = "Year"
    }
    
    enum InsightType: String, CaseIterable {
        case all = "All"
        case patterns = "Patterns"
        case predictions = "Predictions"
        case recommendations = "Recommendations"
        case correlations = "Correlations"
    }
    
    var filteredInsights: [AIInsight] {
        let insights = analyticsManager.insights
        
        switch selectedInsightType {
        case .all:
            return insights
        case .patterns:
            return insights.filter { $0.category == .pattern }
        case .predictions:
            return insights.filter { $0.category == .prediction }
        case .recommendations:
            return insights.filter { $0.category == .recommendation }
        case .correlations:
            return insights.filter { $0.category == .correlation }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header with controls
                    headerSection
                    
                    // Insights summary
                    insightsSummarySection
                    
                    // Quick actions
                    quickActionsSection
                    
                    // Insights list
                    insightsListSection
                    
                    // AI recommendations
                    aiRecommendationsSection
                }
                .padding(.vertical)
            }
            .navigationTitle("AI Insights")
            .themedBackground()
            .refreshable {
                await refreshInsights()
            }
            .sheet(item: $selectedInsight) { insight in
                InsightDetailView(insight: insight)
                    .environmentObject(themeManager)
            }
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 16) {
            // Time range selector
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .onChange(of: selectedTimeRange) { _ in
                loadInsights()
            }
            
            // Insight type filter
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(InsightType.allCases, id: \.self) { type in
                        Button {
                            selectedInsightType = type
                        } label: {
                            Text(type.rawValue)
                                .font(themeManager.typography.body)
                                .fontWeight(.medium)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                                        .fill(selectedInsightType == type ? themeManager.colors.primary : themeManager.colors.background)
                                )
                                .foregroundColor(selectedInsightType == type ? .white : themeManager.colors.textPrimary)
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding(.horizontal)
    }
    
    private var insightsSummarySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Insights Summary")
                .font(themeManager.typography.title2)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                SummaryCard(
                    title: "Total Insights",
                    value: "\(filteredInsights.count)",
                    icon: "lightbulb.fill",
                    color: .blue,
                    trend: .stable
                )
                
                SummaryCard(
                    title: "High Priority",
                    value: "\(filteredInsights.filter { $0.priority == .high }.count)",
                    icon: "exclamationmark.triangle.fill",
                    color: .red,
                    trend: .decreasing
                )
                
                SummaryCard(
                    title: "Actionable",
                    value: "\(filteredInsights.filter { $0.actionable }.count)",
                    icon: "checkmark.circle.fill",
                    color: .green,
                    trend: .increasing
                )
                
                SummaryCard(
                    title: "Confidence",
                    value: "\(Int(averageConfidence() * 100))%",
                    icon: "brain.head.profile",
                    color: .purple,
                    trend: .increasing
                )
            }
        }
        .padding(.horizontal)
    }
    
    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick Actions")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    QuickActionCard(
                        title: "Generate Report",
                        icon: "doc.text.fill",
                        color: .blue
                    ) {
                        generateInsightsReport()
                    }
                    
                    QuickActionCard(
                        title: "Export Data",
                        icon: "square.and.arrow.up.fill",
                        color: .green
                    ) {
                        exportInsights()
                    }
                    
                    QuickActionCard(
                        title: "Schedule Review",
                        icon: "calendar.badge.plus",
                        color: .orange
                    ) {
                        scheduleInsightReview()
                    }
                    
                    QuickActionCard(
                        title: "Share with Doctor",
                        icon: "person.badge.plus.fill",
                        color: .purple
                    ) {
                        shareWithDoctor()
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding(.horizontal)
    }
    
    private var insightsListSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Recent Insights")
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
                
                Button("View All") {
                    // Navigate to full insights list
                }
                .font(themeManager.typography.body)
                .foregroundColor(themeManager.colors.primary)
            }
            
            LazyVStack(spacing: 12) {
                ForEach(filteredInsights.prefix(5), id: \.id) { insight in
                    InsightCard(insight: insight) {
                        selectedInsight = insight
                        showingInsightDetail = true
                    }
                }
            }
        }
        .padding(.horizontal)
    }
    
    private var aiRecommendationsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.purple)
                    .font(.title2)
                
                Text("AI Recommendations")
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            VStack(spacing: 12) {
                ForEach(generateAIRecommendations(), id: \.id) { recommendation in
                    AIRecommendationCard(recommendation: recommendation)
                }
            }
        }
        .padding(.horizontal)
    }
    
    // MARK: - Helper Functions
    
    private func averageConfidence() -> Double {
        guard !filteredInsights.isEmpty else { return 0 }
        return filteredInsights.map { $0.confidence }.reduce(0, +) / Double(filteredInsights.count)
    }
    
    private func loadInsights() {
        let startDate: Date
        let endDate = Date()
        
        switch selectedTimeRange {
        case .week:
            startDate = Calendar.current.date(byAdding: .weekOfYear, value: -1, to: endDate) ?? endDate
        case .month:
            startDate = Calendar.current.date(byAdding: .month, value: -1, to: endDate) ?? endDate
        case .quarter:
            startDate = Calendar.current.date(byAdding: .month, value: -3, to: endDate) ?? endDate
        case .year:
            startDate = Calendar.current.date(byAdding: .year, value: -1, to: endDate) ?? endDate
        }
        
        analyticsManager.loadAnalytics(for: startDate...endDate)
    }
    
    @MainActor
    private func refreshInsights() async {
        isRefreshing = true
        
        // Simulate API call delay
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        
        loadInsights()
        isRefreshing = false
    }
    
    private func generateInsightsReport() {
        // Implementation for generating insights report
        print("Generating insights report")
    }
    
    private func exportInsights() {
        // Implementation for exporting insights
        print("Exporting insights")
    }
    
    private func scheduleInsightReview() {
        // Implementation for scheduling insight review
        print("Scheduling insight review")
    }
    
    private func shareWithDoctor() {
        // Implementation for sharing with doctor
        print("Sharing with doctor")
    }
    
    private func generateAIRecommendations() -> [AIRecommendation] {
        [
            AIRecommendation(
                id: UUID(),
                title: "Optimize Sleep Schedule",
                description: "Based on your pain patterns, maintaining a consistent sleep schedule could reduce morning stiffness by 30%.",
                category: .lifestyle,
                priority: .high,
                confidence: 0.85,
                estimatedImpact: "High",
                timeToImplement: "1-2 weeks",
                actions: [
                    "Set a consistent bedtime of 10:30 PM",
                    "Create a 30-minute wind-down routine",
                    "Avoid screens 1 hour before bed",
                    "Track sleep quality for 2 weeks"
                ]
            ),
            AIRecommendation(
                id: UUID(),
                title: "Weather-Based Medication Timing",
                description: "Your pain levels correlate with barometric pressure changes. Adjusting medication timing could help.",
                category: .medication,
                priority: .medium,
                confidence: 0.72,
                estimatedImpact: "Medium",
                timeToImplement: "Immediate",
                actions: [
                    "Monitor weather forecasts daily",
                    "Take preventive medication 2 hours before pressure drops",
                    "Discuss timing adjustments with your doctor",
                    "Track effectiveness for 1 month"
                ]
            ),
            AIRecommendation(
                id: UUID(),
                title: "Stress Management Protocol",
                description: "High stress levels precede 78% of your flare-ups. Implementing stress management could be preventive.",
                category: .mentalHealth,
                priority: .high,
                confidence: 0.78,
                estimatedImpact: "High",
                timeToImplement: "2-4 weeks",
                actions: [
                    "Practice 10 minutes of daily meditation",
                    "Use breathing exercises during stressful moments",
                    "Consider stress management counseling",
                    "Track stress levels and pain correlation"
                ]
            )
        ]
    }
}

// MARK: - Supporting Views

struct SummaryCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let icon: String
    let color: Color
    let trend: Trend
    
    enum Trend {
        case increasing, decreasing, stable
    }
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)
                
                Spacer()
                
                Image(systemName: trendIcon)
                    .foregroundColor(trendColor)
                    .font(.caption)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Text(title)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
    }
    
    private var trendIcon: String {
        switch trend {
        case .increasing:
            return "arrow.up.right"
        case .decreasing:
            return "arrow.down.right"
        case .stable:
            return "minus"
        }
    }
    
    private var trendColor: Color {
        switch trend {
        case .increasing:
            return .green
        case .decreasing:
            return .red
        case .stable:
            return .gray
        }
    }
}

struct QuickActionCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)
                
                Text(title)
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
            .frame(width: 100, height: 80)
            .background(
                RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                    .fill(themeManager.colors.cardBackground)
                    .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct InsightCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let insight: AIInsight
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                // Category icon
                Image(systemName: categoryIcon(insight.category))
                    .foregroundColor(categoryColor(insight.category))
                    .font(.title3)
                    .frame(width: 24)
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(insight.title)
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.textPrimary)
                            .lineLimit(1)
                        
                        Spacer()
                        
                        // Priority badge
                        Text(priorityText(insight.priority))
                            .font(themeManager.typography.caption2)
                            .fontWeight(.bold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(priorityColor(insight.priority))
                            )
                            .foregroundColor(.white)
                    }
                    
                    Text(insight.description)
                        .font(themeManager.typography.caption)
                        .foregroundColor(themeManager.colors.textSecondary)
                        .lineLimit(2)
                    
                    HStack {
                        // Confidence indicator
                        HStack(spacing: 4) {
                            Image(systemName: "brain.head.profile")
                                .font(.caption2)
                            Text("\(Int(insight.confidence * 100))%")
                                .font(themeManager.typography.caption2)
                        }
                        .foregroundColor(confidenceColor(insight.confidence))
                        
                        Spacer()
                        
                        // Timestamp
                        Text(timeAgo(insight.timestamp))
                            .font(themeManager.typography.caption2)
                            .foregroundColor(themeManager.colors.textSecondary)
                    }
                }
                
                Image(systemName: "chevron.right")
                    .foregroundColor(themeManager.colors.textSecondary)
                    .font(.caption)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                    .fill(themeManager.colors.cardBackground)
                    .shadow(color: themeManager.colors.shadow, radius: 2, x: 0, y: 1)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
    
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
    
    private func priorityText(_ priority: AIInsight.Priority) -> String {
        switch priority {
        case .low:
            return "LOW"
        case .medium:
            return "MED"
        case .high:
            return "HIGH"
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
    
    private func timeAgo(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct AIRecommendationCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let recommendation: AIRecommendation
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: categoryIcon(recommendation.category))
                    .foregroundColor(categoryColor(recommendation.category))
                    .font(.title3)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(recommendation.title)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    HStack(spacing: 12) {
                        Text("Impact: \(recommendation.estimatedImpact)")
                            .font(themeManager.typography.caption2)
                            .foregroundColor(.green)
                        
                        Text("Time: \(recommendation.timeToImplement)")
                            .font(themeManager.typography.caption2)
                            .foregroundColor(.blue)
                        
                        Text("\(Int(recommendation.confidence * 100))% confident")
                            .font(themeManager.typography.caption2)
                            .foregroundColor(confidenceColor(recommendation.confidence))
                    }
                }
                
                Spacer()
                
                Button {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        isExpanded.toggle()
                    }
                } label: {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(themeManager.colors.textSecondary)
                        .font(.caption)
                }
            }
            
            // Description
            Text(recommendation.description)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
                .lineLimit(isExpanded ? nil : 2)
            
            // Expanded content
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Action Steps:")
                        .font(themeManager.typography.caption)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    ForEach(Array(recommendation.actions.enumerated()), id: \.offset) { index, action in
                        HStack(alignment: .top, spacing: 8) {
                            Text("\(index + 1).")
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                                .frame(width: 16, alignment: .leading)
                            
                            Text(action)
                                .font(themeManager.typography.caption)
                                .foregroundColor(themeManager.colors.textSecondary)
                                .lineLimit(nil)
                        }
                    }
                    
                    HStack {
                        Button("Start Implementation") {
                            // Implementation action
                        }
                        .font(themeManager.typography.caption)
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(themeManager.colors.primary)
                        )
                        
                        Spacer()
                        
                        Button("Remind Later") {
                            // Reminder action
                        }
                        .font(themeManager.typography.caption)
                        .foregroundColor(themeManager.colors.primary)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
    }
    
    private func categoryIcon(_ category: AIRecommendation.Category) -> String {
        switch category {
        case .lifestyle:
            return "figure.walk"
        case .medication:
            return "pills.fill"
        case .exercise:
            return "figure.strengthtraining.traditional"
        case .nutrition:
            return "leaf.fill"
        case .mentalHealth:
            return "brain.head.profile"
        case .sleep:
            return "bed.double.fill"
        }
    }
    
    private func categoryColor(_ category: AIRecommendation.Category) -> Color {
        switch category {
        case .lifestyle:
            return .blue
        case .medication:
            return .red
        case .exercise:
            return .green
        case .nutrition:
            return .orange
        case .mentalHealth:
            return .purple
        case .sleep:
            return .indigo
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
}

// MARK: - Supporting Data Models

struct AIRecommendation: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let category: Category
    let priority: AIInsight.Priority
    let confidence: Double
    let estimatedImpact: String
    let timeToImplement: String
    let actions: [String]
    
    enum Category {
        case lifestyle, medication, exercise, nutrition, mentalHealth, sleep
    }
}

#Preview {
    InsightsView()
        .environmentObject(ThemeManager())
        .environmentObject(AdvancedAnalyticsManager())
}