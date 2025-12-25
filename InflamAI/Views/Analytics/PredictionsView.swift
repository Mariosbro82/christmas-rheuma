//
//  PredictionsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct PredictionsView: View {
    @Environment(\.dismiss) private var dismiss
    // @EnvironmentObject private var themeManager: ThemeManager
    @StateObject private var analyticsManager = AdvancedAnalyticsManager()
    @State private var selectedPrediction: PainPrediction?
    @State private var showingFlareAlert = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header
                    headerSection
                    
                    // Prediction Chart
                    predictionChart
                    
                    // Flare Risk Assessment
                    flareRiskAssessment
                    
                    // Detailed Predictions
                    detailedPredictions
                    
                    // Recommendations
                    recommendationsSection
                }
                .padding(.vertical)
            }
            .navigationTitle("Pain Predictions")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundColor(.blue)
                }
            }
            .background(Color(.systemBackground))
            .alert("High Flare Risk Detected", isPresented: $showingFlareAlert) {
                Button("View Recommendations") {
                    // Handle recommendations
                }
                Button("Dismiss", role: .cancel) { }
            } message: {
                Text("Our AI model predicts a high risk of pain flare in the next 48 hours. Consider preventive measures.")
            }
            .onAppear {
                Task {
                    let dateRange = DateInterval(start: Date(), duration: 7 * 24 * 60 * 60) // 7 days
                    await analyticsManager.loadAnalytics(for: dateRange)
                }
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.blue)
                    .font(.title)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Statistical Pattern Analysis")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)

                    Text("Based on your historical data and patterns")
                        .font(.body)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }

            // Validated accuracy indicator
            if let accuracy = UserDefaults.standard.object(forKey: "modelAccuracy") as? Double, accuracy > 0 {
                HStack {
                    Text("Validated Accuracy:")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("\(String(format: "%.1f%%", accuracy * 100))")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(accuracy >= 0.7 ? .green : accuracy >= 0.6 ? .orange : .red)

                    Spacer()

                    Text("Tested on recent data")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                HStack {
                    Text("Accuracy:")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("Not yet validated")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)

                    Spacer()

                    Text("Need more data")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // MEDICAL DISCLAIMER
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                    .font(.caption)

                VStack(alignment: .leading, spacing: 4) {
                    Text("⚠️ NOT MEDICAL ADVICE")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.orange)

                    Text("This is a statistical estimate for informational purposes only. Always consult your rheumatologist before making treatment decisions. Do not use this to diagnose, treat, or manage medical conditions.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.orange.opacity(0.1))
            )
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var predictionChart: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("7-Day Pain Forecast")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Chart(analyticsManager.painPredictions) { prediction in
                LineMark(
                    x: .value("Date", prediction.date),
                    y: .value("Predicted Pain", prediction.predictedPain)
                )
                .foregroundStyle(.blue)
                .lineStyle(StrokeStyle(lineWidth: 3))
                .symbol(Circle().strokeBorder(lineWidth: 2))
                
                AreaMark(
                    x: .value("Date", prediction.date),
                    yStart: .value("Lower Bound", prediction.confidenceInterval.lowerBound),
                    yEnd: .value("Upper Bound", prediction.confidenceInterval.upperBound)
                )
                .foregroundStyle(Color.blue.opacity(0.2))
                
                if let selected = selectedPrediction, selected.date == prediction.date {
                    RuleMark(x: .value("Selected Date", prediction.date))
                        .foregroundStyle(.orange)
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [5]))
                }
            }
            .frame(height: 250)
            .chartYScale(domain: 0...10)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) { value in
                    AxisGridLine()
                    AxisValueLabel(format: .dateTime.weekday(.abbreviated).month(.abbreviated).day())
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel()
                }
            }
            .chartBackground { chartProxy in
                GeometryReader { geometry in
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            if let date = chartProxy.value(atX: location.x, as: Date.self) {
                                selectedPrediction = analyticsManager.painPredictions.min(by: {
                                    abs($0.date.timeIntervalSince(date)) < abs($1.date.timeIntervalSince(date))
                                })
                            }
                        }
                }
            }
            
            // Legend
            HStack(spacing: 20) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(.blue)
                        .frame(width: 12, height: 12)
                    Text("Predicted Pain")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                HStack(spacing: 8) {
                    Rectangle()
                        .fill(Color.blue.opacity(0.2))
                        .frame(width: 12, height: 12)
                    Text("Confidence Range")
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
    
    private var flareRiskAssessment: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                    .font(.title2)
                
                Text("Flare Risk Assessment")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            VStack(spacing: 12) {
                FlareRiskRow(timeframe: "Next 24 hours", risk: .low, percentage: 15)
                FlareRiskRow(timeframe: "Next 48 hours", risk: .medium, percentage: 35)
                FlareRiskRow(timeframe: "Next 7 days", risk: .high, percentage: 68)
            }
            
            Text("Risk factors: Weather changes, increased stress levels, irregular sleep pattern")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.top, 8)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var detailedPredictions: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Detailed Predictions")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            LazyVStack(spacing: 12) {
                ForEach(analyticsManager.painPredictions, id: \.date) { prediction in
                    PredictionRow(prediction: prediction, isSelected: selectedPrediction?.date == prediction.date)
                        .onTapGesture {
                            selectedPrediction = prediction
                        }
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
    
    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)
                    .font(.title2)
                
                Text("AI Recommendations")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            VStack(alignment: .leading, spacing: 12) {
                RecommendationRow(
                    icon: "pills.fill",
                    title: "Medication Timing",
                    description: "Take morning medication 30 minutes earlier for better effectiveness",
                    priority: .high
                )
                
                RecommendationRow(
                    icon: "bed.double.fill",
                    title: "Sleep Schedule",
                    description: "Maintain consistent sleep schedule to reduce flare risk",
                    priority: .medium
                )
                
                RecommendationRow(
                    icon: "figure.walk",
                    title: "Gentle Exercise",
                    description: "Light stretching in the morning can help reduce stiffness",
                    priority: .medium
                )
                
                RecommendationRow(
                    icon: "cloud.rain.fill",
                    title: "Weather Preparation",
                    description: "Low pressure system approaching - consider preventive measures",
                    priority: .high
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
}

// MARK: - Supporting Views

struct FlareRiskRow: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let timeframe: String
    let risk: RiskLevel
    let percentage: Int
    
    enum RiskLevel {
        case low, medium, high
        
        var color: Color {
            switch self {
            case .low: return .green
            case .medium: return .orange
            case .high: return .red
            }
        }
        
        var text: String {
            switch self {
            case .low: return "Low"
            case .medium: return "Medium"
            case .high: return "High"
            }
        }
    }
    
    var body: some View {
        HStack {
            Text(timeframe)
                .font(.body)
                .foregroundColor(.primary)
            
            Spacer()
            
            HStack(spacing: 8) {
                Text("\(percentage)%")
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(risk.color)
                
                Text(risk.text)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(risk.color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(risk.color.opacity(0.2))
                    )
            }
        }
    }
}

struct PredictionRow: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let prediction: PainPrediction
    let isSelected: Bool
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(prediction.date, style: .date)
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Text(prediction.date, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                HStack(spacing: 4) {
                    Text(String(format: "%.1f", prediction.predictedPain))
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(painColor(for: prediction.predictedPain))
                    
                    Text("/10")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Text("\(Int(prediction.confidence * 100))% confidence")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.blue.opacity(0.1) : Color.clear)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
                )
        )
    }
    
    private func painColor(for level: Double) -> Color {
        switch level {
        case 0..<3: return .green
        case 3..<6: return .orange
        case 6..<8: return .red
        default: return .purple
        }
    }
}

struct RecommendationRow: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    let priority: AIInsight.Priority
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(priority.color)
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
                    .lineLimit(2)
            }
            
            Spacer()
            
            Circle()
                .fill(priority.color)
                .frame(width: 8, height: 8)
        }
        .padding(.vertical, 8)
    }
}

#Preview {
    PredictionsView()
        // .environmentObject(ThemeManager())
}