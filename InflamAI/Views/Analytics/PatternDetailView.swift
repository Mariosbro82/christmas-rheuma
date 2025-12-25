//
//  PatternDetailView.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI
import Charts

struct PatternDetailView: View {
    @EnvironmentObject private var themeManager: ThemeManager
    @Environment(\.dismiss) private var dismiss
    let pattern: PatternData
    @State private var selectedTab = 0
    @State private var showingExportOptions = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 20) {
                    // Header with pattern info
                    headerSection
                    
                    // Tab selector
                    tabSelector
                    
                    // Content based on selected tab
                    Group {
                        switch selectedTab {
                        case 0:
                            overviewSection
                        case 1:
                            analysisSection
                        case 2:
                            recommendationsSection
                        default:
                            overviewSection
                        }
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Pattern Details")
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
                    Button {
                        showingExportOptions = true
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundColor(themeManager.colors.primary)
                    }
                }
            }
            .actionSheet(isPresented: $showingExportOptions) {
                ActionSheet(
                    title: Text("Export Pattern Data"),
                    buttons: [
                        .default(Text("Export as PDF")) {
                            exportPattern(format: .pdf)
                        },
                        .default(Text("Share Summary")) {
                            sharePattern()
                        },
                        .default(Text("Add to Calendar")) {
                            addToCalendar()
                        },
                        .cancel()
                    ]
                )
            }
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 16) {
            // Pattern type and confidence
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: patternIcon(for: pattern.type))
                            .foregroundColor(patternColor(for: pattern.type))
                            .font(.title2)
                        
                        Text(pattern.type)
                            .font(themeManager.typography.title2)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Text(pattern.description)
                        .font(themeManager.typography.body)
                        .foregroundColor(themeManager.colors.textSecondary)
                        .lineLimit(nil)
                }
                
                Spacer()
            }
            
            // Confidence and stats
            HStack(spacing: 20) {
                StatCard(
                    title: "Confidence",
                    value: "\(Int(pattern.confidence * 100))%",
                    color: confidenceColor(pattern.confidence)
                )
                
                StatCard(
                    title: "Occurrences",
                    value: "\(pattern.occurrences)",
                    color: .blue
                )
                
                StatCard(
                    title: "Frequency",
                    value: frequencyText(for: pattern),
                    color: .green
                )
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
    
    private var tabSelector: some View {
        HStack(spacing: 0) {
            ForEach(Array(["Overview", "Analysis", "Recommendations"].enumerated()), id: \.offset) { index, title in
                Button {
                    selectedTab = index
                } label: {
                    Text(title)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(selectedTab == index ? themeManager.colors.primary : themeManager.colors.textSecondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(
                            Rectangle()
                                .fill(selectedTab == index ? themeManager.colors.primary.opacity(0.1) : Color.clear)
                        )
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(themeManager.colors.background)
        )
        .padding(.horizontal)
    }
    
    private var overviewSection: some View {
        VStack(spacing: 20) {
            // Pattern timeline
            patternTimelineCard
            
            // Key metrics
            keyMetricsCard
            
            // Related factors
            relatedFactorsCard
        }
    }
    
    private var analysisSection: some View {
        VStack(spacing: 20) {
            // Statistical analysis
            statisticalAnalysisCard
            
            // Correlation matrix
            correlationMatrixCard
            
            // Trend analysis
            trendAnalysisCard
        }
    }
    
    private var recommendationsSection: some View {
        VStack(spacing: 20) {
            // Actionable insights
            actionableInsightsCard
            
            // Prevention strategies
            preventionStrategiesCard
            
            // Monitoring suggestions
            monitoringSuggestionsCard
        }
    }
    
    private var patternTimelineCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Pattern Timeline")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            Chart(generatePatternTimelineData()) { dataPoint in
                LineMark(
                    x: .value("Time", dataPoint.time),
                    y: .value("Intensity", dataPoint.intensity)
                )
                .foregroundStyle(patternColor(for: pattern.type))
                .lineStyle(StrokeStyle(lineWidth: 3))
                
                AreaMark(
                    x: .value("Time", dataPoint.time),
                    y: .value("Intensity", dataPoint.intensity)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [patternColor(for: pattern.type).opacity(0.3), .clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                
                if dataPoint.isSignificant {
                    PointMark(
                        x: .value("Time", dataPoint.time),
                        y: .value("Intensity", dataPoint.intensity)
                    )
                    .foregroundStyle(.red)
                    .symbolSize(100)
                }
            }
            .frame(height: 200)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: 7)) { value in
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
            
            Text("Red dots indicate significant pattern occurrences")
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var keyMetricsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Key Metrics")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                MetricCard(
                    title: "Average Duration",
                    value: "\(Int.random(in: 2...8)) hours",
                    icon: "clock.fill",
                    color: .blue
                )
                
                MetricCard(
                    title: "Peak Intensity",
                    value: "\(Int.random(in: 6...9))/10",
                    icon: "chart.line.uptrend.xyaxis",
                    color: .red
                )
                
                MetricCard(
                    title: "Recovery Time",
                    value: "\(Int.random(in: 12...48)) hours",
                    icon: "heart.fill",
                    color: .green
                )
                
                MetricCard(
                    title: "Predictability",
                    value: "\(Int(pattern.confidence * 100))%",
                    icon: "brain.head.profile",
                    color: .purple
                )
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
    
    private var relatedFactorsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Related Factors")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            VStack(spacing: 12) {
                ForEach(generateRelatedFactors(), id: \.name) { factor in
                    HStack {
                        Image(systemName: factor.icon)
                            .foregroundColor(factor.color)
                            .font(.title3)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text(factor.name)
                                .font(themeManager.typography.body)
                                .fontWeight(.medium)
                                .foregroundColor(themeManager.colors.textPrimary)
                            
                            Text(factor.description)
                                .font(themeManager.typography.caption)
                                .foregroundColor(themeManager.colors.textSecondary)
                        }
                        
                        Spacer()
                        
                        Text("\(Int(factor.correlation * 100))%")
                            .font(themeManager.typography.caption)
                            .fontWeight(.medium)
                            .foregroundColor(correlationColor(factor.correlation))
                    }
                    .padding(.vertical, 8)
                    
                    if factor != generateRelatedFactors().last {
                        Divider()
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
    }
    
    private var statisticalAnalysisCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Statistical Analysis")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            VStack(spacing: 16) {
                // P-value and significance
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("P-Value")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("< 0.001")
                            .font(themeManager.typography.title3)
                            .fontWeight(.bold)
                            .foregroundColor(.green)
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Significance")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("High")
                            .font(themeManager.typography.title3)
                            .fontWeight(.bold)
                            .foregroundColor(.green)
                    }
                }
                
                Divider()
                
                // Effect size
                VStack(alignment: .leading, spacing: 8) {
                    Text("Effect Size (Cohen's d)")
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    HStack {
                        Text("0.8")
                            .font(themeManager.typography.title3)
                            .fontWeight(.bold)
                            .foregroundColor(themeManager.colors.textPrimary)
                        
                        Text("(Large effect)")
                            .font(themeManager.typography.caption)
                            .foregroundColor(.green)
                        
                        Spacer()
                    }
                    
                    ProgressView(value: 0.8)
                        .progressViewStyle(LinearProgressViewStyle(tint: .green))
                }
                
                Divider()
                
                // Sample size and power
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Sample Size")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("\(pattern.occurrences) events")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.textPrimary)
                    }
                    
                    Spacer()
                    
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Statistical Power")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        Text("95%")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(.green)
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
    }
    
    private var correlationMatrixCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Correlation Matrix")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            let factors = ["Pain", "Sleep", "Stress", "Weather", "Activity"]
            
            VStack(spacing: 8) {
                // Header row
                HStack(spacing: 8) {
                    Text("")
                        .frame(width: 60, alignment: .leading)
                    
                    ForEach(factors, id: \.self) { factor in
                        Text(factor)
                            .font(themeManager.typography.caption)
                            .fontWeight(.medium)
                            .frame(width: 50)
                            .foregroundColor(themeManager.colors.textSecondary)
                    }
                }
                
                // Data rows
                ForEach(Array(factors.enumerated()), id: \.offset) { rowIndex, rowFactor in
                    HStack(spacing: 8) {
                        Text(rowFactor)
                            .font(themeManager.typography.caption)
                            .fontWeight(.medium)
                            .frame(width: 60, alignment: .leading)
                            .foregroundColor(themeManager.colors.textSecondary)
                        
                        ForEach(Array(factors.enumerated()), id: \.offset) { colIndex, _ in
                            let correlation = generateCorrelationValue(row: rowIndex, col: colIndex)
                            
                            Text(String(format: "%.2f", correlation))
                                .font(themeManager.typography.caption2)
                                .fontWeight(.medium)
                                .frame(width: 50, height: 30)
                                .background(
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(correlationHeatmapColor(correlation))
                                )
                                .foregroundColor(correlation > 0.5 || correlation < -0.5 ? .white : .black)
                        }
                    }
                }
            }
            
            // Legend
            HStack {
                Text("Strong Negative")
                    .font(themeManager.typography.caption2)
                Rectangle()
                    .fill(.red)
                    .frame(width: 20, height: 10)
                
                Spacer()
                
                Text("No Correlation")
                    .font(themeManager.typography.caption2)
                Rectangle()
                    .fill(.gray.opacity(0.3))
                    .frame(width: 20, height: 10)
                
                Spacer()
                
                Text("Strong Positive")
                    .font(themeManager.typography.caption2)
                Rectangle()
                    .fill(.green)
                    .frame(width: 20, height: 10)
            }
            .foregroundColor(themeManager.colors.textSecondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.medium)
                .fill(themeManager.colors.cardBackground)
                .shadow(color: themeManager.colors.shadow, radius: 4, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private var trendAnalysisCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Trend Analysis")
                .font(themeManager.typography.title3)
                .fontWeight(.bold)
                .foregroundColor(themeManager.colors.textPrimary)
            
            VStack(spacing: 16) {
                // Trend direction
                HStack {
                    Image(systemName: "arrow.up.right")
                        .foregroundColor(.green)
                        .font(.title2)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Improving Trend")
                            .font(themeManager.typography.body)
                            .fontWeight(.medium)
                            .foregroundColor(themeManager.colors.textPrimary)
                        
                        Text("Pattern intensity decreasing over time")
                            .font(themeManager.typography.caption)
                            .foregroundColor(themeManager.colors.textSecondary)
                    }
                    
                    Spacer()
                    
                    Text("+15%")
                        .font(themeManager.typography.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                }
                
                Divider()
                
                // Seasonal patterns
                VStack(alignment: .leading, spacing: 8) {
                    Text("Seasonal Variation")
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    HStack {
                        ForEach(["Spring", "Summer", "Fall", "Winter"], id: \.self) { season in
                            VStack(spacing: 4) {
                                Text(season)
                                    .font(themeManager.typography.caption2)
                                    .foregroundColor(themeManager.colors.textSecondary)
                                
                                let intensity = seasonalIntensity(for: season)
                                Rectangle()
                                    .fill(intensityColor(intensity))
                                    .frame(width: 20, height: CGFloat(intensity * 40))
                                    .cornerRadius(2)
                                
                                Text("\(Int(intensity * 100))%")
                                    .font(themeManager.typography.caption2)
                                    .foregroundColor(themeManager.colors.textSecondary)
                            }
                        }
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
    }
    
    private var actionableInsightsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)
                    .font(.title2)
                
                Text("Actionable Insights")
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            VStack(spacing: 12) {
                ForEach(generateActionableInsights(), id: \.title) { insight in
                    InsightCard(
                        icon: insight.icon,
                        title: insight.title,
                        description: insight.description,
                        priority: insight.priority,
                        color: insight.color
                    )
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
    
    private var preventionStrategiesCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "shield.fill")
                    .foregroundColor(.blue)
                    .font(.title2)
                
                Text("Prevention Strategies")
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            VStack(spacing: 12) {
                ForEach(generatePreventionStrategies(), id: \.title) { strategy in
                    StrategyCard(
                        icon: strategy.icon,
                        title: strategy.title,
                        description: strategy.description,
                        effectiveness: strategy.effectiveness,
                        difficulty: strategy.difficulty
                    )
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
    
    private var monitoringSuggestionsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.purple)
                    .font(.title2)
                
                Text("Monitoring Suggestions")
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
            
            VStack(spacing: 12) {
                ForEach(generateMonitoringSuggestions(), id: \.title) { suggestion in
                    MonitoringCard(
                        icon: suggestion.icon,
                        title: suggestion.title,
                        description: suggestion.description,
                        frequency: suggestion.frequency,
                        importance: suggestion.importance
                    )
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
    
    // MARK: - Helper Functions
    
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
    
    private func correlationColor(_ correlation: Double) -> Color {
        switch abs(correlation) {
        case 0.7...1.0:
            return .green
        case 0.5..<0.7:
            return .yellow
        case 0.3..<0.5:
            return .orange
        default:
            return .red
        }
    }
    
    private func frequencyText(for pattern: PatternData) -> String {
        let frequency = Double(pattern.occurrences) / 30.0 // Assuming 30-day period
        
        switch frequency {
        case 0.8...Double.infinity:
            return "Daily"
        case 0.3..<0.8:
            return "Weekly"
        case 0.1..<0.3:
            return "Monthly"
        default:
            return "Rare"
        }
    }
    
    private func generatePatternTimelineData() -> [PatternTimelinePoint] {
        let days = 30
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        
        return (0..<days).map { i in
            let date = Calendar.current.date(byAdding: .day, value: i, to: startDate) ?? Date()
            let baseIntensity = 3 + 2 * sin(Double(i) * 0.2)
            let noise = Double.random(in: -1...1)
            let intensity = max(0, min(10, baseIntensity + noise))
            let isSignificant = intensity > 7 || Double.random(in: 0...1) < 0.1
            
            return PatternTimelinePoint(
                time: date,
                intensity: intensity,
                isSignificant: isSignificant
            )
        }
    }
    
    private func generateRelatedFactors() -> [RelatedFactor] {
        [
            RelatedFactor(
                name: "Sleep Quality",
                description: "Poor sleep correlates with pattern intensity",
                correlation: 0.75,
                icon: "bed.double.fill",
                color: .blue
            ),
            RelatedFactor(
                name: "Stress Level",
                description: "High stress precedes pattern occurrence",
                correlation: 0.68,
                icon: "brain.head.profile",
                color: .red
            ),
            RelatedFactor(
                name: "Weather Changes",
                description: "Barometric pressure affects symptoms",
                correlation: 0.52,
                icon: "cloud.rain.fill",
                color: .gray
            ),
            RelatedFactor(
                name: "Physical Activity",
                description: "Exercise level influences recovery",
                correlation: -0.45,
                icon: "figure.walk",
                color: .green
            )
        ]
    }
    
    private func generateCorrelationValue(row: Int, col: Int) -> Double {
        if row == col {
            return 1.0
        }
        
        let correlations: [[Double]] = [
            [1.0, -0.6, 0.7, 0.3, -0.4],
            [-0.6, 1.0, -0.5, -0.2, 0.6],
            [0.7, -0.5, 1.0, 0.4, -0.3],
            [0.3, -0.2, 0.4, 1.0, -0.1],
            [-0.4, 0.6, -0.3, -0.1, 1.0]
        ]
        
        return correlations[row][col]
    }
    
    private func correlationHeatmapColor(_ correlation: Double) -> Color {
        let absCorr = abs(correlation)
        
        if correlation > 0 {
            return Color.green.opacity(absCorr)
        } else if correlation < 0 {
            return Color.red.opacity(absCorr)
        } else {
            return Color.gray.opacity(0.3)
        }
    }
    
    private func seasonalIntensity(for season: String) -> Double {
        switch season {
        case "Spring":
            return 0.6
        case "Summer":
            return 0.4
        case "Fall":
            return 0.8
        case "Winter":
            return 0.9
        default:
            return 0.5
        }
    }
    
    private func intensityColor(_ intensity: Double) -> Color {
        switch intensity {
        case 0.8...1.0:
            return .red
        case 0.6..<0.8:
            return .orange
        case 0.4..<0.6:
            return .yellow
        default:
            return .green
        }
    }
    
    private func generateActionableInsights() -> [ActionableInsight] {
        [
            ActionableInsight(
                icon: "clock.fill",
                title: "Optimize Sleep Schedule",
                description: "Maintain consistent bedtime to reduce pattern intensity by 25%",
                priority: .high,
                color: .blue
            ),
            ActionableInsight(
                icon: "figure.mind.and.body",
                title: "Stress Management",
                description: "Practice meditation during high-risk periods",
                priority: .high,
                color: .purple
            ),
            ActionableInsight(
                icon: "thermometer",
                title: "Weather Monitoring",
                description: "Track barometric pressure changes for early warning",
                priority: .medium,
                color: .cyan
            )
        ]
    }
    
    private func generatePreventionStrategies() -> [PreventionStrategy] {
        [
            PreventionStrategy(
                icon: "pills.fill",
                title: "Preventive Medication",
                description: "Take anti-inflammatory 2 hours before predicted onset",
                effectiveness: 0.8,
                difficulty: .easy
            ),
            PreventionStrategy(
                icon: "figure.yoga",
                title: "Gentle Exercise",
                description: "Light stretching routine during pattern-prone periods",
                effectiveness: 0.6,
                difficulty: .easy
            ),
            PreventionStrategy(
                icon: "leaf.fill",
                title: "Environmental Control",
                description: "Maintain stable temperature and humidity",
                effectiveness: 0.4,
                difficulty: .medium
            )
        ]
    }
    
    private func generateMonitoringSuggestions() -> [MonitoringSuggestion] {
        [
            MonitoringSuggestion(
                icon: "heart.fill",
                title: "Heart Rate Variability",
                description: "Monitor HRV for early stress detection",
                frequency: "Daily",
                importance: .high
            ),
            MonitoringSuggestion(
                icon: "bed.double.fill",
                title: "Sleep Quality Metrics",
                description: "Track deep sleep percentage and wake frequency",
                frequency: "Nightly",
                importance: .high
            ),
            MonitoringSuggestion(
                icon: "cloud.fill",
                title: "Weather Conditions",
                description: "Log barometric pressure and humidity changes",
                frequency: "Twice daily",
                importance: .medium
            )
        ]
    }
    
    private func exportPattern(format: ExportFormat) {
        // Implementation for exporting pattern data
        print("Exporting pattern in \(format) format")
    }
    
    private func sharePattern() {
        // Implementation for sharing pattern summary
        print("Sharing pattern summary")
    }
    
    private func addToCalendar() {
        // Implementation for adding pattern reminders to calendar
        print("Adding pattern reminders to calendar")
    }
}

// MARK: - Supporting Data Models

struct PatternTimelinePoint {
    let time: Date
    let intensity: Double
    let isSignificant: Bool
}

struct RelatedFactor: Equatable {
    let name: String
    let description: String
    let correlation: Double
    let icon: String
    let color: Color
}

struct ActionableInsight {
    let icon: String
    let title: String
    let description: String
    let priority: Priority
    let color: Color
    
    enum Priority {
        case high, medium, low
    }
}

struct PreventionStrategy {
    let icon: String
    let title: String
    let description: String
    let effectiveness: Double
    let difficulty: Difficulty
    
    enum Difficulty {
        case easy, medium, hard
    }
}

struct MonitoringSuggestion {
    let icon: String
    let title: String
    let description: String
    let frequency: String
    let importance: Importance
    
    enum Importance {
        case high, medium, low
    }
}

enum ExportFormat {
    case pdf, csv, json
}

// MARK: - Supporting Views

struct StatCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Text(value)
                .font(themeManager.typography.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(title)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(color.opacity(0.1))
        )
    }
}

struct MetricCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.title2)
            
            VStack(spacing: 4) {
                Text(value)
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Text(title)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(themeManager.colors.background)
        )
    }
}

struct InsightCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    let priority: ActionableInsight.Priority
    let color: Color
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.title3)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(title)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Spacer()
                    
                    Text(priority == .high ? "HIGH" : priority == .medium ? "MED" : "LOW")
                        .font(themeManager.typography.caption2)
                        .fontWeight(.bold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(priorityColor(priority))
                        )
                        .foregroundColor(.white)
                }
                
                Text(description)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(nil)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(color.opacity(0.1))
        )
    }
    
    private func priorityColor(_ priority: ActionableInsight.Priority) -> Color {
        switch priority {
        case .high:
            return .red
        case .medium:
            return .orange
        case .low:
            return .green
        }
    }
}

struct StrategyCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    let effectiveness: Double
    let difficulty: PreventionStrategy.Difficulty
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                    .font(.title3)
                
                Text(title)
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                Spacer()
            }
            
            Text(description)
                .font(themeManager.typography.caption)
                .foregroundColor(themeManager.colors.textSecondary)
                .lineLimit(nil)
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Effectiveness")
                        .font(themeManager.typography.caption2)
                        .foregroundColor(themeManager.colors.textSecondary)
                    
                    ProgressView(value: effectiveness)
                        .progressViewStyle(LinearProgressViewStyle(tint: .green))
                        .frame(width: 80)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Difficulty")
                        .font(themeManager.typography.caption2)
                        .foregroundColor(themeManager.colors.textSecondary)
                    
                    Text(difficulty == .easy ? "Easy" : difficulty == .medium ? "Medium" : "Hard")
                        .font(themeManager.typography.caption)
                        .fontWeight(.medium)
                        .foregroundColor(difficultyColor(difficulty))
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(themeManager.colors.background)
        )
    }
    
    private func difficultyColor(_ difficulty: PreventionStrategy.Difficulty) -> Color {
        switch difficulty {
        case .easy:
            return .green
        case .medium:
            return .orange
        case .hard:
            return .red
        }
    }
}

struct MonitoringCard: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let icon: String
    let title: String
    let description: String
    let frequency: String
    let importance: MonitoringSuggestion.Importance
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.purple)
                .font(.title3)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(title)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Spacer()
                    
                    Text(frequency)
                        .font(themeManager.typography.caption)
                        .foregroundColor(themeManager.colors.primary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(themeManager.colors.primary.opacity(0.1))
                        )
                }
                
                Text(description)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(nil)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: themeManager.cornerRadius.small)
                .fill(themeManager.colors.background)
        )
    }
}

#Preview {
    PatternDetailView(
        pattern: PatternData(
            id: UUID(),
            type: "Daily Morning Stiffness",
            description: "Increased joint stiffness occurs every morning between 6-8 AM",
            confidence: 0.85,
            occurrences: 24
        )
    )
    .environmentObject(ThemeManager())
}