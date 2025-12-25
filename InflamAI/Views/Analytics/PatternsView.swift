//
//  PatternsView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct PatternsView: View {
    // Removed ThemeManager dependency
    @EnvironmentObject private var analyticsManager: AdvancedAnalyticsManager
    @State private var selectedTimeRange: TimeRange = .month
    @State private var selectedPatternType: PatternType = .all
    @State private var showingPatternDetail = false
    @State private var selectedPattern: PatternData?
    
    enum PatternType: String, CaseIterable {
        case all = "All Patterns"
        case daily = "Daily Patterns"
        case weekly = "Weekly Patterns"
        case seasonal = "Seasonal Patterns"
        case trigger = "Trigger Patterns"
    }
    
    var filteredPatterns: [PatternData] {
        let patterns = analyticsManager.patterns
        
        switch selectedPatternType {
        case .all:
            return patterns
        case .daily:
            return patterns.filter { $0.type.contains("Daily") || $0.type.contains("Hourly") }
        case .weekly:
            return patterns.filter { $0.type.contains("Weekly") }
        case .seasonal:
            return patterns.filter { $0.type.contains("Seasonal") || $0.type.contains("Weather") }
        case .trigger:
            return patterns.filter { $0.type.contains("Trigger") || $0.type.contains("Flare") }
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header with controls
                    headerSection
                    
                    // Pattern summary cards
                    patternSummarySection
                    
                    // Pattern timeline
                    patternTimelineSection
                    
                    // Detected patterns list
                    detectedPatternsSection
                    
                    // Pattern insights
                    patternInsightsSection
                }
                .padding(.vertical)
            }
            .navigationTitle("Pattern Analysis")
            .navigationBarTitleDisplayMode(.large)
            .themedBackground()
            .onAppear {
                analyticsManager.loadAnalytics(for: selectedTimeRange)
            }
            .onChange(of: selectedTimeRange) { _ in
                analyticsManager.loadAnalytics(for: selectedTimeRange)
            }
            .sheet(item: $selectedPattern) { pattern in
                PatternDetailView(pattern: pattern)
                    // Removed ThemeManager environment object
            }
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 16) {
            // Time range selector
            HStack {
                Text("Time Range")
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
                
                Picker("Time Range", selection: $selectedTimeRange) {
                    ForEach(TimeRange.allCases, id: \.self) { range in
                        Text(range.rawValue).tag(range)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .foregroundColor(.blue)
            }
            
            // Pattern type filter
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(PatternType.allCases, id: \.self) { type in
                        Button {
                            selectedPatternType = type
                        } label: {
                            Text(type.rawValue)
                                .font(.caption)
                                .fontWeight(.medium)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                .fill(selectedPatternType == type ? Color.blue : Color(.systemBackground))
                                )
                                .foregroundColor(selectedPatternType == type ? .white : .secondary)
                        }
                    }
                }
                .padding(.horizontal)
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
    
    private var patternSummarySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Pattern Summary")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
                .padding(.horizontal)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    PatternSummaryCard(
                        title: "Total Patterns",
                        value: "\(filteredPatterns.count)",
                        subtitle: "Detected",
                        icon: "chart.bar.fill",
                        color: .blue
                    )
                    
                    PatternSummaryCard(
                        title: "Strong Patterns",
                        value: "\(filteredPatterns.filter { $0.confidence > 0.8 }.count)",
                        subtitle: "High Confidence",
                        icon: "star.fill",
                        color: .green
                    )
                    
                    PatternSummaryCard(
                        title: "Daily Patterns",
                        value: "\(filteredPatterns.filter { $0.type.contains("Daily") }.count)",
                        subtitle: "Recurring",
                        icon: "clock.fill",
                        color: .orange
                    )
                    
                    PatternSummaryCard(
                        title: "Trigger Patterns",
                        value: "\(filteredPatterns.filter { $0.type.contains("Trigger") }.count)",
                        subtitle: "Identified",
                        icon: "exclamationmark.triangle.fill",
                        color: .red
                    )
                }
                .padding(.horizontal)
            }
        }
    }
    
    private var patternTimelineSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Pattern Timeline")
                    .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
                
                Spacer()
                
                Button("View Details") {
                    // Show detailed timeline view
                }
                .font(.caption)
                    .foregroundColor(.blue)
            }
            
            Chart(generateTimelineData()) { dataPoint in
                // Pain level line
                LineMark(
                    x: .value("Time", dataPoint.time),
                    y: .value("Pain Level", dataPoint.painLevel)
                )
                .foregroundStyle(.red)
                .lineStyle(StrokeStyle(lineWidth: 2))
                
                // Pattern markers
                if dataPoint.hasPattern {
                    PointMark(
                        x: .value("Time", dataPoint.time),
                        y: .value("Pain Level", dataPoint.painLevel)
                    )
                    .foregroundStyle(.blue)
                    .symbolSize(100)
                    .symbol(Circle().strokeBorder(lineWidth: 2))
                }
                
                // Trigger events
                if dataPoint.hasTrigger {
                    RuleMark(
                        x: .value("Time", dataPoint.time)
                    )
                    .foregroundStyle(.orange)
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [5]))
                    .annotation(position: .top) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                            .font(.caption)
                    }
                }
            }
            .frame(height: 200)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: selectedTimeRange == .week ? 1 : 7)) { value in
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
                        .fill(.red)
                        .frame(width: 12, height: 12)
                    Text("Pain Level")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                HStack(spacing: 8) {
                    Circle()
                        .stroke(.blue, lineWidth: 2)
                        .frame(width: 12, height: 12)
                    Text("Pattern Detected")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                HStack(spacing: 8) {
                    Rectangle()
                        .fill(.orange)
                        .frame(width: 2, height: 12)
                    Text("Trigger Event")
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
    
    private var detectedPatternsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Detected Patterns")
                    .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.primary)
                
                Spacer()
                
                Text("\(filteredPatterns.count) patterns")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            LazyVStack(spacing: 12) {
                ForEach(filteredPatterns.prefix(10)) { pattern in
                    PatternCard(pattern: pattern) {
                        selectedPattern = pattern
                        showingPatternDetail = true
                    }
                }
                
                if filteredPatterns.count > 10 {
                    Button("View All \(filteredPatterns.count) Patterns") {
                        // Show all patterns view
                    }
                    .font(.body)
                    .foregroundColor(.blue)
                    .padding()
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
    
    private var patternInsightsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)
                    .font(.title2)
                
                Text("Pattern Insights")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
            }
            
            VStack(spacing: 12) {
                PatternInsightCard(
                    icon: "clock.fill",
                    title: "Peak Symptom Times",
                    description: generatePeakTimeInsight(),
                    color: .blue
                )
                
                PatternInsightCard(
                    icon: "calendar",
                    title: "Weekly Trends",
                    description: generateWeeklyTrendInsight(),
                    color: .green
                )
                
                PatternInsightCard(
                    icon: "exclamationmark.triangle.fill",
                    title: "Common Triggers",
                    description: generateTriggerInsight(),
                    color: .orange
                )
                
                PatternInsightCard(
                    icon: "heart.fill",
                    title: "Recovery Patterns",
                    description: generateRecoveryInsight(),
                    color: .red
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
    private func generateTimelineData() -> [TimelineDataPoint] {
        let days = selectedTimeRange == .week ? 7 : (selectedTimeRange == .month ? 30 : 90)
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        
        return (0..<days).map { i in
            let date = Calendar.current.date(byAdding: .day, value: i, to: startDate) ?? Date()
            let painLevel = 3 + 2 * sin(Double(i) * 0.3) + Double.random(in: -1...1)
            let hasPattern = i % 7 == 0 || i % 14 == 3 // Weekly pattern
            let hasTrigger = Double.random(in: 0...1) < 0.1 // 10% chance of trigger
            
            return TimelineDataPoint(
                time: date,
                painLevel: max(0, min(10, painLevel)),
                hasPattern: hasPattern,
                hasTrigger: hasTrigger
            )
        }
    }
    
    private func generatePeakTimeInsight() -> String {
        let morningPatterns = filteredPatterns.filter { $0.description.contains("morning") || $0.description.contains("AM") }
        let eveningPatterns = filteredPatterns.filter { $0.description.contains("evening") || $0.description.contains("PM") }
        
        if morningPatterns.count > eveningPatterns.count {
            return "Your symptoms tend to be worse in the morning. Consider gentle stretching or warm-up exercises upon waking."
        } else if eveningPatterns.count > morningPatterns.count {
            return "Evening symptoms are more common. End-of-day fatigue may be contributing to increased pain levels."
        } else {
            return "Your symptoms show consistent patterns throughout the day. Focus on maintaining steady activity levels."
        }
    }
    
    private func generateWeeklyTrendInsight() -> String {
        let weekendPatterns = filteredPatterns.filter { $0.description.contains("weekend") || $0.description.contains("Saturday") || $0.description.contains("Sunday") }
        
        if weekendPatterns.count > 2 {
            return "Weekend patterns detected. Changes in routine or activity levels may be affecting your symptoms."
        } else {
            return "Consistent weekly patterns observed. Your routine appears to provide good symptom stability."
        }
    }
    
    private func generateTriggerInsight() -> String {
        let triggerPatterns = filteredPatterns.filter { $0.type.contains("Trigger") }
        
        if triggerPatterns.isEmpty {
            return "No clear trigger patterns identified yet. Continue tracking to build a more complete picture."
        } else {
            let commonTrigger = triggerPatterns.first?.description.components(separatedBy: " ").first ?? "stress"
            return "\(commonTrigger.capitalized) appears to be a common trigger. Consider strategies to manage or avoid this factor."
        }
    }
    
    private func generateRecoveryInsight() -> String {
        let recoveryPatterns = filteredPatterns.filter { $0.description.contains("recovery") || $0.description.contains("improvement") }
        
        if recoveryPatterns.count > 2 {
            return "Good recovery patterns identified. Your body shows consistent healing responses to treatment."
        } else {
            return "Recovery patterns are still developing. Focus on consistent self-care and treatment adherence."
        }
    }
}

// MARK: - Supporting Views

struct PatternSummaryCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let subtitle: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                
                Text(title)
                    .font(.body)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)
                
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(width: 140, height: 120)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
    }
}

struct PatternCard: View {
    // Removed ThemeManager dependency
    let pattern: PatternData
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 16) {
                // Pattern type icon
                Image(systemName: patternIcon(for: pattern.type))
                    .foregroundColor(patternColor(for: pattern.type))
                    .font(.title2)
                    .frame(width: 32, height: 32)
                
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(pattern.type)
                            .font(.body)
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        // Confidence indicator
                        HStack(spacing: 4) {
                            ForEach(0..<5, id: \.self) { i in
                                Image(systemName: "star.fill")
                                    .foregroundColor(i < Int(pattern.confidence * 5) ? .yellow : .gray.opacity(0.3))
                                    .font(.caption2)
                            }
                        }
                    }
                    
                    Text(pattern.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                    
                    HStack {
                        Text("Confidence: \(Int(pattern.confidence * 100))%")
                            .font(.caption)
                            .foregroundColor(confidenceColor(pattern.confidence))
                        
                        Spacer()
                        
                        Text("\(pattern.occurrences) occurrences")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(.systemBackground))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private func patternIcon(for type: String) -> String {
        switch type {
        case let t where t.contains("Daily"):
            return "clock.fill"
        case let t where t.contains("Weekly"):
            return "calendar"
        case let t where t.contains("Seasonal"):
            return "leaf.fill"
        case let t where t.contains("Trigger"):
            return "exclamationmark.triangle.fill"
        case let t where t.contains("Weather"):
            return "cloud.fill"
        default:
            return "chart.bar.fill"
        }
    }
    
    private func patternColor(for type: String) -> Color {
        switch type {
        case let t where t.contains("Daily"):
            return .blue
        case let t where t.contains("Weekly"):
            return .green
        case let t where t.contains("Seasonal"):
            return .orange
        case let t where t.contains("Trigger"):
            return .red
        case let t where t.contains("Weather"):
            return .cyan
        default:
            return .purple
        }
    }
    
    private func confidenceColor(_ confidence: Double) -> Color {
        switch confidence {
        case 0.8...1.0:
            return .green
        case 0.6..<0.8:
            return .yellow
        case 0.4..<0.6:
            return .orange
        default:
            return .red
        }
    }
}

struct PatternInsightCard: View {
    // Removed ThemeManager dependency
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

struct TimelineDataPoint {
    let time: Date
    let painLevel: Double
    let hasPattern: Bool
    let hasTrigger: Bool
}

#Preview {
    PatternsView()
        // Removed ThemeManager environment object
        .environmentObject(AdvancedAnalyticsManager())
}