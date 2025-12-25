//
//  DetailedPainAnalysisView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024-01-21.
//

import SwiftUI
import Charts

struct DetailedPainAnalysisView: View {
    let data: [PainDataPoint]
    let selectedRegions: Set<BodyRegion>
    
    @StateObject private var aiEngine = AIMLEngine.shared
    @StateObject private var dataManager = HealthDataManager.shared
    
    @State private var selectedAnalysisType: AnalysisType = .patterns
    @State private var showingExportOptions = false
    @State private var isGeneratingReport = false
    @State private var correlationData: [CorrelationInsight] = []
    @State private var predictiveAnalysis: PredictiveAnalysis?
    
    private enum AnalysisType: String, CaseIterable {
        case patterns = "Patterns"
        case correlations = "Correlations"
        case predictions = "Predictions"
        case triggers = "Triggers"
        case medications = "Medications"
        
        var icon: String {
            switch self {
            case .patterns: return "chart.line.uptrend.xyaxis"
            case .correlations: return "link"
            case .predictions: return "crystal.ball"
            case .triggers: return "exclamationmark.triangle"
            case .medications: return "pills"
            }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    VStack(spacing: 15) {
                        HStack {
                            Text("Detailed Analysis")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                            
                            Spacer()
                            
                            Button("Export") {
                                showingExportOptions = true
                            }
                            .font(.subheadline)
                            .foregroundColor(.blue)
                        }
                        
                        // Analysis Type Picker
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 15) {
                                ForEach(AnalysisType.allCases, id: \.self) { type in
                                    AnalysisTypeButton(
                                        type: type,
                                        isSelected: selectedAnalysisType == type
                                    ) {
                                        selectedAnalysisType = type
                                        loadAnalysisData()
                                    }
                                }
                            }
                            .padding(.horizontal)
                        }
                    }
                    .padding()
                    
                    // Analysis Content
                    Group {
                        switch selectedAnalysisType {
                        case .patterns:
                            PainPatternsAnalysisView(data: data)
                        case .correlations:
                            CorrelationAnalysisView(correlations: correlationData)
                        case .predictions:
                            PredictiveAnalysisView(analysis: predictiveAnalysis)
                        case .triggers:
                            TriggerAnalysisView(data: data)
                        case .medications:
                            MedicationEffectivenessView(data: data)
                        }
                    }
                    
                    // AI-Generated Summary
                    AISummaryView(data: data, analysisType: selectedAnalysisType)
                }
            }
            .navigationBarHidden(true)
            .onAppear {
                loadAnalysisData()
            }
            .sheet(isPresented: $showingExportOptions) {
                ExportOptionsView(data: data)
            }
        }
    }
    
    private func loadAnalysisData() {
        Task {
            switch selectedAnalysisType {
            case .correlations:
                let correlations = await aiEngine.analyzeCorrelations(data)
                await MainActor.run {
                    self.correlationData = correlations
                }
            case .predictions:
                let predictions = await aiEngine.generatePredictiveAnalysis(data)
                await MainActor.run {
                    self.predictiveAnalysis = predictions
                }
            default:
                break
            }
        }
    }
}

// MARK: - Analysis Type Button

struct AnalysisTypeButton: View {
    let type: DetailedPainAnalysisView.AnalysisType
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: type.icon)
                    .font(.caption)
                
                Text(type.rawValue)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(isSelected ? Color.blue : Color(.systemGray6))
            )
            .foregroundColor(isSelected ? .white : .primary)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Pain Patterns Analysis

struct PainPatternsAnalysisView: View {
    let data: [PainDataPoint]
    
    private var weeklyPatterns: [WeeklyPattern] {
        analyzeWeeklyPatterns()
    }
    
    private var dailyPatterns: [DailyPattern] {
        analyzeDailyPatterns()
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // Weekly Patterns
            PatternSectionView(
                title: "Weekly Patterns",
                icon: "calendar.badge.clock"
            ) {
                Chart(weeklyPatterns, id: \.dayOfWeek) { pattern in
                    BarMark(
                        x: .value("Day", pattern.dayName),
                        y: .value("Average Pain", pattern.averagePain)
                    )
                    .foregroundStyle(colorForPainLevel(pattern.averagePain))
                }
                .frame(height: 200)
                .chartYScale(domain: 0...10)
            }
            
            // Daily Patterns
            PatternSectionView(
                title: "Daily Patterns",
                icon: "clock.badge.checkmark"
            ) {
                Chart(dailyPatterns, id: \.hour) { pattern in
                    LineMark(
                        x: .value("Hour", pattern.hour),
                        y: .value("Average Pain", pattern.averagePain)
                    )
                    .foregroundStyle(.blue)
                    .lineStyle(StrokeStyle(lineWidth: 3))
                    
                    AreaMark(
                        x: .value("Hour", pattern.hour),
                        y: .value("Average Pain", pattern.averagePain)
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue.opacity(0.3), .clear],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
                .frame(height: 200)
                .chartXScale(domain: 0...23)
                .chartYScale(domain: 0...10)
            }
            
            // Pattern Insights
            PatternInsightsView(weeklyPatterns: weeklyPatterns, dailyPatterns: dailyPatterns)
        }
    }
    
    private func analyzeWeeklyPatterns() -> [WeeklyPattern] {
        let calendar = Calendar.current
        var patterns: [Int: [Double]] = [:]
        
        for dataPoint in data {
            let weekday = calendar.component(.weekday, from: dataPoint.timestamp)
            patterns[weekday, default: []].append(dataPoint.averagePainLevel)
        }
        
        return patterns.map { (weekday, painLevels) in
            WeeklyPattern(
                dayOfWeek: weekday,
                averagePain: painLevels.reduce(0, +) / Double(painLevels.count),
                dataPoints: painLevels.count
            )
        }.sorted { $0.dayOfWeek < $1.dayOfWeek }
    }
    
    private func analyzeDailyPatterns() -> [DailyPattern] {
        let calendar = Calendar.current
        var patterns: [Int: [Double]] = [:]
        
        for dataPoint in data {
            let hour = calendar.component(.hour, from: dataPoint.timestamp)
            patterns[hour, default: []].append(dataPoint.averagePainLevel)
        }
        
        return (0...23).map { hour in
            let painLevels = patterns[hour] ?? []
            return DailyPattern(
                hour: hour,
                averagePain: painLevels.isEmpty ? 0 : painLevels.reduce(0, +) / Double(painLevels.count),
                dataPoints: painLevels.count
            )
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
}

// MARK: - Correlation Analysis

struct CorrelationAnalysisView: View {
    let correlations: [CorrelationInsight]
    
    var body: some View {
        VStack(spacing: 20) {
            ForEach(correlations, id: \.id) { correlation in
                CorrelationCard(correlation: correlation)
            }
            
            if correlations.isEmpty {
                EmptyCorrelationView()
            }
        }
    }
}

struct CorrelationCard: View {
    let correlation: CorrelationInsight
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: correlation.icon)
                    .foregroundColor(correlation.color)
                    .font(.title2)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(correlation.title)
                        .font(.headline)
                        .fontWeight(.medium)
                    
                    Text("Correlation: \(String(format: "%.1f%%", correlation.strength * 100))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                CorrelationStrengthIndicator(strength: correlation.strength)
            }
            
            Text(correlation.description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            
            if let recommendation = correlation.recommendation {
                HStack {
                    Image(systemName: "lightbulb")
                        .foregroundColor(.yellow)
                    
                    Text(recommendation)
                        .font(.caption)
                        .foregroundColor(.blue)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 5)
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

struct CorrelationStrengthIndicator: View {
    let strength: Double
    
    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .stroke(Color(.systemGray5), lineWidth: 4)
                    .frame(width: 40, height: 40)
                
                Circle()
                    .trim(from: 0, to: strength)
                    .stroke(colorForStrength(strength), lineWidth: 4)
                    .frame(width: 40, height: 40)
                    .rotationEffect(.degrees(-90))
                
                Text("\(Int(strength * 100))")
                    .font(.caption2)
                    .fontWeight(.bold)
                    .foregroundColor(colorForStrength(strength))
            }
            
            Text(strengthLabel(strength))
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
    
    private func colorForStrength(_ strength: Double) -> Color {
        switch strength {
        case 0...0.3: return .red
        case 0.3...0.6: return .orange
        case 0.6...0.8: return .yellow
        default: return .green
        }
    }
    
    private func strengthLabel(_ strength: Double) -> String {
        switch strength {
        case 0...0.3: return "Weak"
        case 0.3...0.6: return "Moderate"
        case 0.6...0.8: return "Strong"
        default: return "Very Strong"
        }
    }
}

// MARK: - Predictive Analysis

struct PredictiveAnalysisView: View {
    let analysis: PredictiveAnalysis?
    
    var body: some View {
        VStack(spacing: 20) {
            if let analysis = analysis {
                // Prediction Chart
                PredictionChartView(predictions: analysis.predictions)
                
                // Risk Factors
                RiskFactorsView(riskFactors: analysis.riskFactors)
                
                // Recommendations
                PredictiveRecommendationsView(recommendations: analysis.recommendations)
            } else {
                ProgressView("Generating predictions...")
                    .frame(height: 200)
            }
        }
    }
}

// MARK: - Supporting Views

struct PatternSectionView<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
            }
            .padding(.horizontal)
            
            content
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
}

struct PatternInsightsView: View {
    let weeklyPatterns: [WeeklyPattern]
    let dailyPatterns: [DailyPattern]
    
    private var peakDay: WeeklyPattern? {
        weeklyPatterns.max { $0.averagePain < $1.averagePain }
    }
    
    private var peakHour: DailyPattern? {
        dailyPatterns.max { $0.averagePain < $1.averagePain }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "lightbulb")
                    .foregroundColor(.yellow)
                Text("Pattern Insights")
                    .font(.headline)
            }
            .padding(.horizontal)
            
            VStack(spacing: 10) {
                if let peakDay = peakDay {
                    InsightRow(
                        icon: "calendar",
                        title: "Peak Day",
                        description: "\(peakDay.dayName) shows highest average pain (\(String(format: "%.1f", peakDay.averagePain)))",
                        color: .red
                    )
                }
                
                if let peakHour = peakHour {
                    InsightRow(
                        icon: "clock",
                        title: "Peak Time",
                        description: "\(peakHour.timeString) shows highest average pain (\(String(format: "%.1f", peakHour.averagePain)))",
                        color: .orange
                    )
                }
                
                InsightRow(
                    icon: "chart.line.uptrend.xyaxis",
                    title: "Weekly Variation",
                    description: "Pain varies by \(String(format: "%.1f", weeklyVariation)) points throughout the week",
                    color: .blue
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
    
    private var weeklyVariation: Double {
        let painLevels = weeklyPatterns.map { $0.averagePain }
        guard let max = painLevels.max(), let min = painLevels.min() else { return 0 }
        return max - min
    }
}

struct InsightRow: View {
    let icon: String
    let title: String
    let description: String
    let color: Color
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.title3)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(color.opacity(0.1))
        )
    }
}

// MARK: - Data Models

struct WeeklyPattern {
    let dayOfWeek: Int
    let averagePain: Double
    let dataPoints: Int
    
    var dayName: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE"
        let date = Calendar.current.date(from: DateComponents(weekday: dayOfWeek))!
        return formatter.string(from: date)
    }
}

struct DailyPattern {
    let hour: Int
    let averagePain: Double
    let dataPoints: Int
    
    var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h a"
        let date = Calendar.current.date(from: DateComponents(hour: hour))!
        return formatter.string(from: date)
    }
}

struct CorrelationInsight {
    let id = UUID()
    let title: String
    let description: String
    let strength: Double
    let recommendation: String?
    let color: Color
    let icon: String
    let factors: [String]
}

struct PredictiveAnalysis {
    let predictions: [PainPrediction]
    let riskFactors: [RiskFactor]
    let recommendations: [PredictiveRecommendation]
    let confidence: Double
    let timeframe: String
}

struct PainPrediction {
    let date: Date
    let predictedPain: Double
    let confidence: Double
    let factors: [String]
}

struct RiskFactor {
    let name: String
    let impact: Double
    let description: String
    let color: Color
}

struct PredictiveRecommendation {
    let title: String
    let description: String
    let priority: Int
    let icon: String
    let color: Color
}

// MARK: - Additional Views

struct EmptyCorrelationView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "link.badge.plus")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Correlations Found")
                .font(.title2)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
            
            Text("Continue tracking to discover patterns and correlations in your pain data.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(height: 200)
        .padding()
    }
}

struct PredictionChartView: View {
    let predictions: [PainPrediction]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Pain Predictions")
                .font(.headline)
                .padding(.horizontal)
            
            Chart(predictions, id: \.date) { prediction in
                LineMark(
                    x: .value("Date", prediction.date),
                    y: .value("Predicted Pain", prediction.predictedPain)
                )
                .foregroundStyle(.blue)
                .lineStyle(StrokeStyle(lineWidth: 3, dash: [5, 5]))
                
                PointMark(
                    x: .value("Date", prediction.date),
                    y: .value("Predicted Pain", prediction.predictedPain)
                )
                .foregroundStyle(.blue)
                .symbolSize(prediction.confidence * 100)
            }
            .frame(height: 200)
            .chartYScale(domain: 0...10)
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
}

struct RiskFactorsView: View {
    let riskFactors: [RiskFactor]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Risk Factors")
                .font(.headline)
                .padding(.horizontal)
            
            ForEach(riskFactors, id: \.name) { factor in
                RiskFactorRow(factor: factor)
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

struct RiskFactorRow: View {
    let factor: RiskFactor
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(factor.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(factor.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(Int(factor.impact * 100))%")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(factor.color)
                
                Text("impact")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(factor.color.opacity(0.1))
        )
        .padding(.horizontal)
    }
}

struct PredictiveRecommendationsView: View {
    let recommendations: [PredictiveRecommendation]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Recommendations")
                .font(.headline)
                .padding(.horizontal)
            
            ForEach(recommendations.sorted { $0.priority < $1.priority }, id: \.title) { recommendation in
                RecommendationRow(recommendation: recommendation)
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

struct RecommendationRow: View {
    let recommendation: PredictiveRecommendation
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: recommendation.icon)
                .foregroundColor(recommendation.color)
                .font(.title3)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(recommendation.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text(recommendation.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
            
            Text("P\(recommendation.priority)")
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(recommendation.color)
                )
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(recommendation.color.opacity(0.1))
        )
        .padding(.horizontal)
    }
}

// MARK: - Trigger Analysis

struct TriggerAnalysisView: View {
    let data: [PainDataPoint]
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Trigger Analysis")
                .font(.headline)
            
            // Implementation for trigger analysis
            Text("Trigger analysis coming soon...")
                .foregroundColor(.secondary)
        }
        .frame(height: 200)
    }
}

// MARK: - Medication Effectiveness

struct MedicationEffectivenessView: View {
    let data: [PainDataPoint]
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Medication Effectiveness")
                .font(.headline)
            
            // Implementation for medication effectiveness analysis
            Text("Medication analysis coming soon...")
                .foregroundColor(.secondary)
        }
        .frame(height: 200)
    }
}

// MARK: - AI Summary

struct AISummaryView: View {
    let data: [PainDataPoint]
    let analysisType: DetailedPainAnalysisView.AnalysisType
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.blue)
                Text("AI Summary")
                    .font(.headline)
            }
            .padding(.horizontal)
            
            Text(generateSummary())
                .font(.subheadline)
                .foregroundColor(.secondary)
                .padding(.horizontal)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(Color.blue.opacity(0.1))
        )
        .padding(.horizontal)
    }
    
    private func generateSummary() -> String {
        // Generate AI summary based on analysis type and data
        switch analysisType {
        case .patterns:
            return "Your pain patterns show distinct weekly and daily variations. Consider tracking environmental factors and activities to identify potential triggers."
        case .correlations:
            return "Several correlations have been identified in your pain data. These insights can help optimize your treatment approach."
        case .predictions:
            return "Based on your historical data, we've generated predictions for your pain levels. Use these insights for proactive pain management."
        case .triggers:
            return "Trigger analysis helps identify factors that may be contributing to your pain episodes."
        case .medications:
            return "Medication effectiveness analysis shows how different treatments impact your pain levels over time."
        }
    }
}

// MARK: - Export Options

struct ExportOptionsView: View {
    let data: [PainDataPoint]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Export Options")
                    .font(.title2)
                    .fontWeight(.bold)
                
                // Export options implementation
                Text("Export functionality coming soon...")
                    .foregroundColor(.secondary)
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Done") {
                    // Dismiss
                }
            )
        }
    }
}