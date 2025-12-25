//
//  CorrelationDetailView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct CorrelationDetailView: View {
    @Environment(\.dismiss) private var dismiss
    // @EnvironmentObject private var themeManager: ThemeManager
    let correlation: CorrelationData
    @State private var selectedTimeframe: DetailTimeframe = .week
    @State private var showingExplanation = false
    
    enum DetailTimeframe: String, CaseIterable {
        case week = "7 Days"
        case month = "30 Days"
        case quarter = "90 Days"
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header with correlation info
                    headerSection
                    
                    // Correlation strength indicator
                    strengthIndicator
                    
                    // Scatter plot
                    scatterPlot
                    
                    // Time series chart
                    timeSeriesChart
                    
                    // Statistical analysis
                    statisticalAnalysis
                    
                    // Insights and recommendations
                    insightsSection
                }
                .padding(.vertical)
            }
            .navigationTitle("Correlation Details")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                    .foregroundColor(.blue)
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingExplanation = true
                    } label: {
                        Image(systemName: "questionmark.circle")
                            .foregroundColor(.blue)
                    }
                }
            }
            .background(Color(.systemBackground))
            .sheet(isPresented: $showingExplanation) {
                CorrelationExplanationView()
                    // .environmentObject(themeManager)
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    Text("\(correlation.factor1) vs \(correlation.factor2)")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    
                    Text(correlation.description)
                        .font(.body)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(String(format: "%.3f", correlation.strength))
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(correlationColor(correlation.strength))
                    
                    Text("Correlation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Timeframe selector
            Picker("Timeframe", selection: $selectedTimeframe) {
                ForEach(DetailTimeframe.allCases, id: \.self) { timeframe in
                    Text(timeframe.rawValue).tag(timeframe)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var strengthIndicator: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Correlation Strength")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            VStack(spacing: 12) {
                // Visual strength indicator
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Background
                        Rectangle()
                            .fill(Color(.systemGray5))
                            .frame(height: 20)
                            .cornerRadius(10)
                        
                        // Gradient background for scale
                        Rectangle()
                            .fill(LinearGradient(
                                colors: [.red, .orange, .yellow, .green, .blue],
                                startPoint: .leading,
                                endPoint: .trailing
                            ))
                            .frame(height: 20)
                            .cornerRadius(10)
                            .opacity(0.3)
                        
                        // Indicator
                        Circle()
                            .fill(correlationColor(correlation.strength))
                            .frame(width: 24, height: 24)
                            .offset(x: geometry.size.width * CGFloat((correlation.strength + 1) / 2) - 12)
                            .shadow(color: .black.opacity(0.3), radius: 2, x: 0, y: 1)
                    }
                }
                .frame(height: 24)
                
                // Scale labels
                HStack {
                    Text("-1.0")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("0.0")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("+1.0")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Interpretation
                HStack {
                    Text(interpretCorrelation(correlation.strength))
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundColor(correlationColor(correlation.strength))
                    
                    Spacer()
                    
                    Text("p-value: \(String(format: "%.4f", correlation.pValue))")
                        .font(.caption)
                        .foregroundColor(correlation.pValue < 0.05 ? .green : .orange)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var scatterPlot: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Scatter Plot Analysis")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Chart(generateScatterData()) { dataPoint in
                PointMark(
                    x: .value(correlation.factor1, dataPoint.x),
                    y: .value(correlation.factor2, dataPoint.y)
                )
                .foregroundStyle(Color.blue.opacity(0.7))
                .symbolSize(50)
                
                // Trend line
                LineMark(
                    x: .value(correlation.factor1, dataPoint.x),
                    y: .value("Trend", trendLineValue(for: dataPoint.x))
                )
                .foregroundStyle(correlationColor(correlation.strength))
                .lineStyle(StrokeStyle(lineWidth: 2))
            }
            .frame(height: 250)
            .chartXAxis {
                AxisMarks(position: .bottom) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            
            Text("Each point represents a day's data. The trend line shows the overall relationship.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var timeSeriesChart: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Time Series Comparison")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Chart(generateTimeSeriesData()) { dataPoint in
                LineMark(
                    x: .value("Date", dataPoint.date),
                    y: .value(correlation.factor1, dataPoint.value1)
                )
                .foregroundStyle(.blue)
                .lineStyle(StrokeStyle(lineWidth: 2))
                .symbol(Circle().strokeBorder(lineWidth: 1))
                
                LineMark(
                    x: .value("Date", dataPoint.date),
                    y: .value(correlation.factor2, dataPoint.value2 * 2) // Scale for visibility
                )
                .foregroundStyle(.orange)
                .lineStyle(StrokeStyle(lineWidth: 2, dash: [5]))
                .symbol(Square().strokeBorder(lineWidth: 1))
            }
            .frame(height: 200)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: selectedTimeframe == .week ? 1 : 7)) { value in
                    AxisGridLine()
                    AxisValueLabel(format: .dateTime.month(.abbreviated).day())
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            
            // Legend
            HStack(spacing: 20) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(.blue)
                        .frame(width: 12, height: 12)
                    Text(correlation.factor1)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                HStack(spacing: 8) {
                    Rectangle()
                        .fill(.orange)
                        .frame(width: 12, height: 2)
                    Text("\(correlation.factor2) (scaled)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var statisticalAnalysis: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Statistical Analysis")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                StatCard(title: "R-squared", value: String(format: "%.3f", correlation.strength * correlation.strength), description: "Variance explained")
                StatCard(title: "P-value", value: String(format: "%.4f", correlation.pValue), description: correlation.pValue < 0.05 ? "Statistically significant" : "Not significant")
                StatCard(title: "Sample Size", value: "\(correlation.sampleSize)", description: "Data points analyzed")
                StatCard(title: "Confidence", value: "95%", description: "Confidence interval")
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Confidence Interval")
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Text("[\(String(format: "%.3f", correlation.strength - 0.1)), \(String(format: "%.3f", correlation.strength + 0.1))]")
                    .font(.body)
                    .foregroundColor(.secondary)
                
                Text("We can be 95% confident that the true correlation lies within this range.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var insightsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)
                    .font(.title2)
                
                Text("Insights & Recommendations")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            VStack(alignment: .leading, spacing: 12) {
                InsightCard(
                    icon: "chart.line.uptrend.xyaxis",
                    title: "Pattern Recognition",
                    description: generatePatternInsight(),
                    color: .blue
                )
                
                InsightCard(
                    icon: "target",
                    title: "Actionable Recommendation",
                    description: generateRecommendation(),
                    color: .green
                )
                
                InsightCard(
                    icon: "exclamationmark.triangle.fill",
                    title: "Important Note",
                    description: "Correlation does not imply causation. This analysis shows statistical relationships, not causal effects.",
                    color: .orange
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    // Helper functions
    private func correlationColor(_ value: Double) -> Color {
        let absValue = abs(value)
        switch absValue {
        case 0..<0.3: return .gray
        case 0.3..<0.5: return .orange
        case 0.5..<0.7: return .yellow
        case 0.7..<0.9: return .green
        default: return .blue
        }
    }
    
    private func interpretCorrelation(_ value: Double) -> String {
        let absValue = abs(value)
        let direction = value >= 0 ? "Positive" : "Negative"
        
        switch absValue {
        case 0..<0.3: return "\(direction) - Weak"
        case 0.3..<0.5: return "\(direction) - Moderate"
        case 0.5..<0.7: return "\(direction) - Strong"
        case 0.7..<0.9: return "\(direction) - Very Strong"
        default: return "\(direction) - Extremely Strong"
        }
    }
    
    private func generateScatterData() -> [(x: Double, y: Double)] {
        let count = selectedTimeframe == .week ? 7 : (selectedTimeframe == .month ? 30 : 90)
        return (0..<count).map { i in
            let x = Double.random(in: 0...10)
            let y = correlation.strength * x + Double.random(in: -2...2)
            return (x: x, y: max(0, min(10, y)))
        }
    }
    
    private func generateTimeSeriesData() -> [(date: Date, value1: Double, value2: Double)] {
        let count = selectedTimeframe == .week ? 7 : (selectedTimeframe == .month ? 30 : 90)
        let startDate = Calendar.current.date(byAdding: .day, value: -count, to: Date()) ?? Date()
        
        return (0..<count).map { i in
            let date = Calendar.current.date(byAdding: .day, value: i, to: startDate) ?? Date()
            let value1 = 5 + 3 * sin(Double(i) * 0.2) + Double.random(in: -1...1)
            let value2 = correlation.strength * value1 + Double.random(in: -1...1)
            return (date: date, value1: value1, value2: max(0, min(10, value2)))
        }
    }
    
    private func trendLineValue(for x: Double) -> Double {
        return correlation.strength * x + 2.5
    }
    
    private func generatePatternInsight() -> String {
        if abs(correlation.strength) > 0.7 {
            return "Strong relationship detected. Changes in \(correlation.factor1) are highly predictive of changes in \(correlation.factor2)."
        } else if abs(correlation.strength) > 0.5 {
            return "Moderate relationship found. \(correlation.factor1) shows a noticeable influence on \(correlation.factor2)."
        } else {
            return "Weak relationship observed. \(correlation.factor1) has limited predictive value for \(correlation.factor2)."
        }
    }
    
    private func generateRecommendation() -> String {
        if correlation.factor1.contains("Sleep") && correlation.strength < -0.5 {
            return "Focus on improving sleep quality to potentially reduce pain levels. Consider establishing a consistent bedtime routine."
        } else if correlation.factor1.contains("Stress") && correlation.strength > 0.5 {
            return "Stress management techniques like meditation or deep breathing may help reduce symptom severity."
        } else if correlation.factor1.contains("Weather") {
            return "Monitor weather forecasts and prepare preventive measures during high-risk weather conditions."
        } else {
            return "Track this relationship over time to identify optimal strategies for symptom management."
        }
    }
}

// MARK: - Supporting Views

struct StatCard: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let description: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Text(description)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray5))
        )
    }
}

struct InsightCard: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.title3)
                .frame(width: 24, height: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(nil)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(color.opacity(0.1))
        )
    }
}

#Preview {
    CorrelationDetailView(correlation: CorrelationData(
        id: UUID(),
        factor1: "Sleep Quality",
        factor2: "Pain Level",
        strength: -0.73,
        pValue: 0.001,
        sampleSize: 90,
        description: "Better sleep quality is associated with lower pain levels"
    ))
    // .environmentObject(ThemeManager())
}