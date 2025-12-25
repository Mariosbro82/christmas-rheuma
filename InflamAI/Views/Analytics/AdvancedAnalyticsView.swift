//
//  AdvancedAnalyticsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts
import CoreData
import CoreML

struct AdvancedAnalyticsView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @EnvironmentObject private var themeManager: ThemeManager
    @StateObject private var analyticsManager = AdvancedAnalyticsManager()
    @State private var selectedTimeRange: TimeRange = .month
    @State private var showingPredictions = false
    @State private var isLoading = false
    
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
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Time Range Selector
                    timeRangeSelector
                    
                    // AI Insights Card
                    aiInsightsCard
                    
                    // Pain Prediction Chart
                    painPredictionChart
                    
                    // Correlation Analysis
                    correlationAnalysis
                    
                    // Pattern Recognition
                    patternRecognition
                    
                    // Medication Effectiveness
                    medicationEffectiveness
                    
                    // Weather Correlation
                    weatherCorrelation
                }
                .padding(.vertical)
            }
            .navigationTitle("Advanced Analytics")
            .navigationBarTitleDisplayMode(.large)
            .themedBackground()
            .onAppear {
                loadAnalytics()
            }
            .refreshable {
                await refreshAnalytics()
            }
        }
    }
    
    private var timeRangeSelector: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Time Range")
                .font(themeManager.typography.headline)
                .fontWeight(.semibold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            Picker("Time Range", selection: $selectedTimeRange) {
                ForEach(TimeRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .onChange(of: selectedTimeRange) { _ in
                loadAnalytics()
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var aiInsightsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title2)
                
                Text("AI Insights")
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
                
                Button("View All") {
                    showingPredictions = true
                }
                .foregroundColor(themeManager.colors.primary)
            }
            
            if isLoading {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Analyzing patterns...")
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                }
            } else {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(analyticsManager.insights.prefix(3), id: \.id) { insight in
                        InsightRow(insight: insight)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
        .sheet(isPresented: $showingPredictions) {
            PredictionsView()
                .environmentObject(themeManager)
        }
    }
    
    private var painPredictionChart: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title2)
                
                Text("Pain Prediction")
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
                
                Text("Next 7 days")
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
            }
            
            Chart(analyticsManager.painPredictions) { prediction in
                LineMark(
                    x: .value("Date", prediction.date),
                    y: .value("Predicted Pain", prediction.predictedPain)
                )
                .foregroundStyle(themeManager.colors.primary)
                .lineStyle(StrokeStyle(lineWidth: 3))
                
                AreaMark(
                    x: .value("Date", prediction.date),
                    yStart: .value("Lower Bound", prediction.confidenceInterval.lowerBound),
                    yEnd: .value("Upper Bound", prediction.confidenceInterval.upperBound)
                )
                .foregroundStyle(themeManager.colors.primary.opacity(0.2))
            }
            .frame(height: 200)
            .chartYScale(domain: 0...10)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) { _ in
                    AxisGridLine()
                    AxisValueLabel(format: .dateTime.weekday(.abbreviated))
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var correlationAnalysis: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "link")
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title2)
                
                Text("Correlation Analysis")
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            LazyVStack(spacing: 12) {
                ForEach(analyticsManager.correlations, id: \.factor) { correlation in
                    CorrelationRow(correlation: correlation)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var patternRecognition: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "waveform.path.ecg")
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title2)
                
                Text("Pattern Recognition")
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            LazyVStack(spacing: 12) {
                ForEach(analyticsManager.patterns, id: \.id) { pattern in
                    PatternRow(pattern: pattern)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var medicationEffectiveness: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "pills")
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title2)
                
                Text("Medication Effectiveness")
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            Chart(analyticsManager.medicationEffectiveness) { effectiveness in
                BarMark(
                    x: .value("Medication", effectiveness.medicationName),
                    y: .value("Effectiveness", effectiveness.effectivenessScore)
                )
                .foregroundStyle(effectiveness.effectivenessScore > 7 ? Color.green : effectiveness.effectivenessScore > 4 ? Color.orange : Color.red)
            }
            .frame(height: 150)
            .chartYScale(domain: 0...10)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var weatherCorrelation: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "cloud.sun")
                    .foregroundColor(themeManager.colors.primary)
                    .font(.title2)
                
                Text("Weather Impact")
                    .font(themeManager.typography.title2)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            Chart(analyticsManager.weatherCorrelations) { weather in
                LineMark(
                    x: .value("Date", weather.date),
                    y: .value("Pain Level", weather.painLevel)
                )
                .foregroundStyle(Color.red)
                
                LineMark(
                    x: .value("Date", weather.date),
                    y: .value("Pressure", weather.barometricPressure / 10)
                )
                .foregroundStyle(Color.blue)
            }
            .frame(height: 150)
            .chartYScale(domain: 0...10)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private func loadAnalytics() {
        isLoading = true
        Task {
            await analyticsManager.loadAnalytics(for: selectedTimeRange)
            await MainActor.run {
                isLoading = false
            }
        }
    }
    
    private func refreshAnalytics() async {
        await analyticsManager.refreshAnalytics(for: selectedTimeRange)
    }
}

// MARK: - Supporting Views

struct InsightRow: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let insight: AIInsight
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: insight.icon)
                .foregroundColor(insight.priority.color)
                .font(.title3)
                .frame(width: 24, height: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(insight.title)
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Text(insight.description)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            Text("\(Int(insight.confidence * 100))%")
                .font(themeManager.typography.caption)
                .fontWeight(.medium)
                .foregroundColor(insight.priority.color)
        }
        .padding(.vertical, 8)
    }
}

struct CorrelationRow: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let correlation: CorrelationData
    
    var body: some View {
        HStack {
            Text(correlation.factor)
                .font(themeManager.typography.body)
                .foregroundColor(themeManager.colors.textPrimary)
            
            Spacer()
            
            HStack(spacing: 8) {
                ProgressView(value: abs(correlation.strength), total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle(tint: correlation.strength > 0 ? Color.red : Color.green))
                    .frame(width: 60)
                
                Text(String(format: "%.2f", correlation.strength))
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(correlation.strength > 0 ? Color.red : Color.green)
                    .frame(width: 40, alignment: .trailing)
            }
        }
    }
}

struct PatternRow: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let pattern: PatternData
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(pattern.name)
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
                
                Text("\(Int(pattern.frequency * 100))% frequency")
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
            }
            
            Text(pattern.description)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
                .lineLimit(2)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    AdvancedAnalyticsView()
        .environmentObject(ThemeManager())
}