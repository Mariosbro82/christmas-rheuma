//
//  AIAnalyticsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Charts

struct AIAnalyticsView: View {
    @StateObject private var analyticsManager = AIHealthAnalyticsManager()
    @State private var selectedTab = 0
    @State private var showingExportSheet = false
    @State private var showingModelTraining = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with analysis status
                AnalyticsHeaderView(
                    lastAnalysis: analyticsManager.lastAnalysisDate,
                    isLoading: analyticsManager.isLoading,
                    progress: analyticsManager.analysisProgress
                )
                
                // Tab selector
                Picker("Analytics Tab", selection: $selectedTab) {
                    Text("Overview").tag(0)
                    Text("Predictions").tag(1)
                    Text("Insights").tag(2)
                    Text("Trends").tag(3)
                    Text("Risks").tag(4)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                
                // Content based on selected tab
                TabView(selection: $selectedTab) {
                    AnalyticsOverviewView(manager: analyticsManager)
                        .tag(0)
                    
                    PredictionsView(manager: analyticsManager)
                        .tag(1)
                    
                    InsightsView(manager: analyticsManager)
                        .tag(2)
                    
                    TrendsView(manager: analyticsManager)
                        .tag(3)
                    
                    RiskFactorsView(manager: analyticsManager)
                        .tag(4)
                }
                .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
            }
            .navigationTitle("AI Analytics")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Refresh Analysis") {
                            Task {
                                try? await analyticsManager.generateComprehensiveAnalysis()
                            }
                        }
                        
                        Button("Export Data") {
                            showingExportSheet = true
                        }
                        
                        Button("Train Models") {
                            showingModelTraining = true
                        }
                        
                        Picker("Time Range", selection: $analyticsManager.selectedTimeRange) {
                            ForEach(HealthAnalytics.AnalyticsTimeRange.allCases, id: \.self) { range in
                                Text(range.displayName).tag(range)
                            }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $showingExportSheet) {
                ExportAnalyticsView(manager: analyticsManager)
            }
            .sheet(isPresented: $showingModelTraining) {
                ModelTrainingView(manager: analyticsManager)
            }
            .task {
                if analyticsManager.currentAnalytics == nil {
                    try? await analyticsManager.generateComprehensiveAnalysis()
                }
            }
        }
    }
}

struct AnalyticsHeaderView: View {
    let lastAnalysis: Date?
    let isLoading: Bool
    let progress: Double
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("AI Health Analytics")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    if let lastAnalysis = lastAnalysis {
                        Text("Last updated: \(lastAnalysis, style: .relative) ago")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("No analysis available")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                if isLoading {
                    VStack(spacing: 4) {
                        ProgressView(value: progress)
                            .frame(width: 60)
                        Text("\(Int(progress * 100))%")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                } else {
                    Image(systemName: "brain.head.profile")
                        .font(.title2)
                        .foregroundColor(.blue)
                }
            }
            
            if isLoading {
                ProgressView(value: progress)
                    .progressViewStyle(LinearProgressViewStyle())
            }
        }
        .padding()
        .background(Color(.systemGray6))
    }
}

struct AnalyticsOverviewView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                // Key metrics cards
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                    MetricCard(
                        title: "Insights",
                        value: "\(manager.insights.count)",
                        icon: "lightbulb.fill",
                        color: .blue
                    )
                    
                    MetricCard(
                        title: "Predictions",
                        value: "\(manager.predictions.count)",
                        icon: "crystal.ball.fill",
                        color: .purple
                    )
                    
                    MetricCard(
                        title: "Risk Factors",
                        value: "\(manager.riskFactors.count)",
                        icon: "exclamationmark.triangle.fill",
                        color: .orange
                    )
                    
                    MetricCard(
                        title: "Recommendations",
                        value: "\(manager.recommendations.count)",
                        icon: "checkmark.seal.fill",
                        color: .green
                    )
                }
                
                // Recent insights
                if !manager.insights.isEmpty {
                    SectionHeaderView(title: "Recent Insights", icon: "lightbulb.fill")
                    
                    ForEach(Array(manager.insights.prefix(3)), id: \.id) { insight in
                        InsightCardView(insight: insight)
                    }
                }
                
                // High priority recommendations
                if !manager.recommendations.isEmpty {
                    SectionHeaderView(title: "Priority Recommendations", icon: "star.fill")
                    
                    ForEach(manager.recommendations.filter { $0.priority == .high || $0.priority == .critical }.prefix(3), id: \.id) { recommendation in
                        RecommendationCardView(recommendation: recommendation)
                    }
                }
                
                // Risk assessment
                if !manager.riskFactors.isEmpty {
                    SectionHeaderView(title: "Risk Assessment", icon: "shield.fill")
                    
                    RiskOverviewView(riskFactors: manager.riskFactors)
                }
            }
            .padding()
        }
    }
}

struct PredictionsView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    @State private var selectedTimeframe: PredictionResult.PredictionTimeframe = .shortTerm
    @State private var selectedModelType: HealthPredictionModel.ModelType = .flareUpPrediction
    
    var body: some View {
        VStack(spacing: 0) {
            // Filters
            VStack(spacing: 12) {
                Picker("Model Type", selection: $selectedModelType) {
                    ForEach(HealthPredictionModel.ModelType.allCases, id: \.self) { type in
                        Text(type.displayName).tag(type)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                
                Picker("Timeframe", selection: $selectedTimeframe) {
                    ForEach(PredictionResult.PredictionTimeframe.allCases, id: \.self) { timeframe in
                        Text(timeframe.displayName).tag(timeframe)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
            }
            .padding()
            .background(Color(.systemGray6))
            
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(filteredPredictions, id: \.id) { prediction in
                        PredictionCardView(prediction: prediction)
                    }
                    
                    if filteredPredictions.isEmpty {
                        EmptyStateView(
                            icon: "crystal.ball",
                            title: "No Predictions",
                            message: "Generate new predictions to see results here."
                        )
                    }
                }
                .padding()
            }
        }
    }
    
    private var filteredPredictions: [PredictionResult] {
        manager.predictions.filter { prediction in
            prediction.modelType == selectedModelType &&
            prediction.timeframe == selectedTimeframe
        }
    }
}

struct InsightsView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    @State private var selectedCategory: HealthInsight.InsightCategory? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            // Category filter
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    FilterChip(
                        title: "All",
                        isSelected: selectedCategory == nil
                    ) {
                        selectedCategory = nil
                    }
                    
                    ForEach(HealthInsight.InsightCategory.allCases, id: \.self) { category in
                        FilterChip(
                            title: category.displayName,
                            isSelected: selectedCategory == category
                        ) {
                            selectedCategory = category
                        }
                    }
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
            
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(filteredInsights, id: \.id) { insight in
                        InsightCardView(insight: insight)
                    }
                    
                    if filteredInsights.isEmpty {
                        EmptyStateView(
                            icon: "lightbulb",
                            title: "No Insights",
                            message: "Insights will appear here as data is analyzed."
                        )
                    }
                }
                .padding()
            }
        }
    }
    
    private var filteredInsights: [HealthInsight] {
        if let category = selectedCategory {
            return manager.insights.filter { $0.category == category }
        }
        return manager.insights
    }
}

struct TrendsView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                ForEach(manager.trends, id: \.id) { trend in
                    TrendCardView(trend: trend)
                }
                
                if !manager.correlations.isEmpty {
                    SectionHeaderView(title: "Correlations", icon: "link")
                    
                    ForEach(manager.correlations, id: \.id) { correlation in
                        CorrelationCardView(correlation: correlation)
                    }
                }
                
                if manager.trends.isEmpty && manager.correlations.isEmpty {
                    EmptyStateView(
                        icon: "chart.line.uptrend.xyaxis",
                        title: "No Trends",
                        message: "Trends will appear here as more data is collected."
                    )
                }
            }
            .padding()
        }
    }
}

struct RiskFactorsView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    @State private var selectedRiskLevel: RiskFactor.RiskLevel? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            // Risk level filter
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    FilterChip(
                        title: "All",
                        isSelected: selectedRiskLevel == nil
                    ) {
                        selectedRiskLevel = nil
                    }
                    
                    ForEach(RiskFactor.RiskLevel.allCases, id: \.self) { level in
                        FilterChip(
                            title: level.displayName,
                            isSelected: selectedRiskLevel == level,
                            color: level.color
                        ) {
                            selectedRiskLevel = level
                        }
                    }
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
            
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(filteredRiskFactors, id: \.id) { riskFactor in
                        RiskFactorCardView(riskFactor: riskFactor)
                    }
                    
                    if filteredRiskFactors.isEmpty {
                        EmptyStateView(
                            icon: "shield",
                            title: "No Risk Factors",
                            message: "Risk assessment will appear here after analysis."
                        )
                    }
                }
                .padding()
            }
        }
    }
    
    private var filteredRiskFactors: [RiskFactor] {
        if let level = selectedRiskLevel {
            return manager.riskFactors.filter { $0.riskLevel == level }
        }
        return manager.riskFactors
    }
}

// MARK: - Card Views

struct MetricCard: View {
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
                    .foregroundColor(.primary)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct InsightCardView: View {
    let insight: HealthInsight
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(insight.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(insight.category.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color(.systemGray5))
                        .cornerRadius(4)
                }
                
                Spacer()
                
                VStack(spacing: 4) {
                    Circle()
                        .fill(insight.importance.color)
                        .frame(width: 12, height: 12)
                    
                    Text("\(Int(insight.confidence * 100))%")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            Text(insight.description)
                .font(.body)
                .foregroundColor(.primary)
                .fixedSize(horizontal: false, vertical: true)
            
            if insight.actionable {
                HStack {
                    Image(systemName: "lightbulb.fill")
                        .foregroundColor(.yellow)
                    Text("Actionable insight")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct PredictionCardView: View {
    let prediction: PredictionResult
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(prediction.modelType.displayName)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(prediction.timeframe.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(spacing: 4) {
                    Text("\(Int(prediction.confidence * 100))%")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(.blue)
                    
                    Text("Confidence")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            Text(prediction.prediction)
                .font(.body)
                .foregroundColor(.primary)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            
            if !prediction.recommendations.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recommendations")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    ForEach(Array(prediction.recommendations.prefix(2)), id: \.id) { recommendation in
                        HStack {
                            Image(systemName: recommendation.category.icon)
                                .foregroundColor(recommendation.priority.color)
                            
                            Text(recommendation.title)
                                .font(.caption)
                                .foregroundColor(.primary)
                            
                            Spacer()
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct RecommendationCardView: View {
    let recommendation: AIRecommendation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: recommendation.category.icon)
                    .foregroundColor(recommendation.priority.color)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(recommendation.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text(recommendation.category.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Text(recommendation.priority.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(recommendation.priority.color)
                    .cornerRadius(6)
            }
            
            Text(recommendation.description)
                .font(.body)
                .foregroundColor(.primary)
            
            if !recommendation.actionItems.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Action Items")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    ForEach(Array(recommendation.actionItems.prefix(3)), id: \.id) { item in
                        HStack {
                            Image(systemName: item.isCompleted ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(item.isCompleted ? .green : .gray)
                            
                            Text(item.title)
                                .font(.caption)
                                .foregroundColor(.primary)
                                .strikethrough(item.isCompleted)
                            
                            Spacer()
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct TrendCardView: View {
    let trend: HealthTrend
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(trend.metric)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                HStack(spacing: 4) {
                    Image(systemName: trend.direction.icon)
                        .foregroundColor(trend.direction.color)
                    
                    Text(trend.direction.displayName)
                        .font(.caption)
                        .foregroundColor(trend.direction.color)
                }
            }
            
            Text("Magnitude: \(String(format: "%.2f", trend.magnitude))")
                .font(.body)
                .foregroundColor(.secondary)
            
            Text("Timeframe: \(trend.timeframe)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            // Simple trend visualization
            if !trend.dataPoints.isEmpty {
                Chart(trend.dataPoints, id: \.date) { point in
                    LineMark(
                        x: .value("Date", point.date),
                        y: .value("Value", point.value)
                    )
                    .foregroundStyle(trend.direction.color)
                }
                .frame(height: 60)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct CorrelationCardView: View {
    let correlation: HealthCorrelation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("\(correlation.variable1) ↔ \(correlation.variable2)")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Text(correlation.strength.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(correlation.type.color)
                    .cornerRadius(6)
            }
            
            Text(correlation.description)
                .font(.body)
                .foregroundColor(.primary)
            
            HStack {
                Text("Correlation: \(String(format: "%.3f", correlation.correlation))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("Significance: \(String(format: "%.3f", correlation.significance))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct RiskFactorCardView: View {
    let riskFactor: RiskFactor
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(riskFactor.name)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Text(riskFactor.riskLevel.displayName)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(riskFactor.riskLevel.color)
                    .cornerRadius(6)
            }
            
            Text(riskFactor.description)
                .font(.body)
                .foregroundColor(.primary)
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Probability")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("\(Int(riskFactor.probability * 100))%")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Impact")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(riskFactor.impact.displayName)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                }
            }
            
            if !riskFactor.mitigationStrategies.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Mitigation Strategies")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    ForEach(Array(riskFactor.mitigationStrategies.prefix(3)), id: \.self) { strategy in
                        HStack {
                            Image(systemName: "shield.fill")
                                .foregroundColor(.blue)
                                .font(.caption)
                            
                            Text(strategy)
                                .font(.caption)
                                .foregroundColor(.primary)
                            
                            Spacer()
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

struct RiskOverviewView: View {
    let riskFactors: [RiskFactor]
    
    private var riskDistribution: [RiskFactor.RiskLevel: Int] {
        Dictionary(grouping: riskFactors, by: { $0.riskLevel })
            .mapValues { $0.count }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Risk Distribution")
                .font(.headline)
                .foregroundColor(.primary)
            
            HStack(spacing: 16) {
                ForEach(RiskFactor.RiskLevel.allCases, id: \.self) { level in
                    VStack(spacing: 4) {
                        Text("\(riskDistribution[level] ?? 0)")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(level.color)
                        
                        Text(level.displayName)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            
            // Risk level chart
            Chart(RiskFactor.RiskLevel.allCases, id: \.self) { level in
                BarMark(
                    x: .value("Level", level.displayName),
                    y: .value("Count", riskDistribution[level] ?? 0)
                )
                .foregroundStyle(level.color)
            }
            .frame(height: 100)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
    }
}

// MARK: - Helper Views

struct SectionHeaderView: View {
    let title: String
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
            
            Text(title)
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
            
            Spacer()
        }
        .padding(.vertical, 8)
    }
}

struct FilterChip: View {
    let title: String
    let isSelected: Bool
    var color: Color = .blue
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(isSelected ? .white : color)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(isSelected ? color : Color(.systemGray5))
                .cornerRadius(16)
        }
    }
}

struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 48))
                .foregroundColor(.gray)
            
            VStack(spacing: 8) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text(message)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Export and Training Views

struct ExportAnalyticsView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    @Environment(\.dismiss) private var dismiss
    @State private var isExporting = false
    @State private var exportedData: Data?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Export Analytics Data")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Export your health analytics data for backup or sharing with healthcare providers.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                if isExporting {
                    ProgressView("Exporting...")
                        .progressViewStyle(CircularProgressViewStyle())
                } else {
                    Button("Export Data") {
                        Task {
                            isExporting = true
                            do {
                                exportedData = try await manager.exportAnalytics()
                            } catch {
                                print("Export failed: \(error)")
                            }
                            isExporting = false
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                }
                
                if exportedData != nil {
                    Text("Data exported successfully!")
                        .foregroundColor(.green)
                        .font(.headline)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Export")
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

struct ModelTrainingView: View {
    @ObservedObject var manager: AIHealthAnalyticsManager
    @Environment(\.dismiss) private var dismiss
    @State private var selectedModelType: HealthPredictionModel.ModelType = .flareUpPrediction
    @State private var isTraining = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Train AI Models")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Retrain AI models with your latest health data to improve prediction accuracy.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                Picker("Model Type", selection: $selectedModelType) {
                    ForEach(HealthPredictionModel.ModelType.allCases, id: \.self) { type in
                        Text(type.displayName).tag(type)
                    }
                }
                .pickerStyle(WheelPickerStyle())
                
                if isTraining {
                    ProgressView("Training model...")
                        .progressViewStyle(CircularProgressViewStyle())
                } else {
                    Button("Start Training") {
                        Task {
                            isTraining = true
                            do {
                                try await manager.trainCustomModel(for: selectedModelType, with: [:])
                            } catch {
                                print("Training failed: \(error)")
                            }
                            isTraining = false
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Colors.Primary.p500)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Model Training")
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

#Preview {
    AIAnalyticsView()
}