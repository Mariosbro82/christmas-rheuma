//
//  InsightDetailComponents.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import SwiftUI

// MARK: - Action Item View

struct ActionItem: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    @State private var isCompleted: Bool
    let action: InsightDetailView.ActionItem
    
    init(action: InsightDetailView.ActionItem) {
        self.action = action
        self._isCompleted = State(initialValue: action.completed)
    }
    
    var body: some View {
        HStack(spacing: 12) {
            // Completion checkbox
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isCompleted.toggle()
                }
            } label: {
                Image(systemName: isCompleted ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isCompleted ? .green : .secondary)
                    .font(.title3)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(action.title)
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundColor(isCompleted ? .secondary : .primary)
                        .strikethrough(isCompleted)
                    
                    Spacer()
                    
                    // Priority badge
                    Text(priorityText(action.priority))
                        .font(.caption2)
                        .fontWeight(.bold)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(priorityColor(action.priority))
                        )
                        .foregroundColor(.white)
                }
                
                Text(action.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                HStack {
                    Image(systemName: "clock")
                        .foregroundColor(.blue)
                        .font(.caption)
                    
                    Text(action.estimatedTime)
                        .font(.caption2)
                        .foregroundColor(.blue)
                    
                    Spacer()
                    
                    if !isCompleted {
                        Button("Start") {
                            startAction()
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isCompleted ? Color(.systemBackground).opacity(0.5) : Color(.systemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isCompleted ? .green.opacity(0.3) : Color.clear, lineWidth: 1)
                )
        )
    }
    
    private func priorityText(_ priority: AIInsight.Priority) -> String {
        switch priority {
        case .low: return "LOW"
        case .medium: return "MED"
        case .high: return "HIGH"
        }
    }
    
    private func priorityColor(_ priority: AIInsight.Priority) -> Color {
        switch priority {
        case .low: return .green
        case .medium: return .orange
        case .high: return .red
        }
    }
    
    private func startAction() {
        // Implementation for starting an action
        print("Starting action: \(action.title)")
    }
}

// MARK: - Strategy Item View

struct StrategyItem: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let strategy: InsightDetailView.StrategyItem
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(strategy.title)
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text(strategy.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(isExpanded ? nil : 2)
                }
                
                Spacer()
                
                Button {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        isExpanded.toggle()
                    }
                } label: {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                        .font(.caption)
                }
            }
            
            // Metrics row
            HStack(spacing: 16) {
                StrategyMetric(title: "Timeframe", value: strategy.timeframe, color: .blue)
                StrategyMetric(title: "Impact", value: strategy.expectedImpact, color: impactColor(strategy.expectedImpact))
                StrategyMetric(title: "Difficulty", value: strategy.difficulty, color: difficultyColor(strategy.difficulty))
            }
            
            if isExpanded {
                // Expanded content
                VStack(alignment: .leading, spacing: 8) {
                    Divider()
                    
                    Text("Implementation Steps:")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    ForEach(generateImplementationSteps(), id: \.self) { step in
                        HStack(alignment: .top, spacing: 8) {
                            Circle()
                                .fill(.blue)
                                .frame(width: 4, height: 4)
                                .padding(.top, 6)
                            
                            Text(step)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    HStack {
                        Button("Add to Plan") {
                            addToPlan()
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                        
                        Spacer()
                        
                        Button("Learn More") {
                            learnMore()
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemBackground))
        )
    }
    
    private func impactColor(_ impact: String) -> Color {
        switch impact.lowercased() {
        case "high": return .green
        case "medium": return .orange
        case "low": return .red
        default: return .gray
        }
    }
    
    private func difficultyColor(_ difficulty: String) -> Color {
        switch difficulty.lowercased() {
        case "easy", "low": return .green
        case "medium": return .orange
        case "hard", "high": return .red
        default: return .gray
        }
    }
    
    private func generateImplementationSteps() -> [String] {
        [
            "Set up tracking system for relevant metrics",
            "Establish baseline measurements over 1 week",
            "Implement gradual changes to avoid disruption",
            "Monitor progress and adjust as needed",
            "Evaluate effectiveness after timeframe completion"
        ]
    }
    
    private func addToPlan() {
        print("Adding strategy to plan: \(strategy.title)")
    }
    
    private func learnMore() {
        print("Learning more about: \(strategy.title)")
    }
}

// MARK: - Strategy Metric View

struct StrategyMetric: View {
    // @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Monitoring Item View

struct MonitoringItem: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let recommendation: InsightDetailView.MonitoringItem
    @State private var isEnabled = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with toggle
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(recommendation.title)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Text(recommendation.description)
                        .font(themeManager.typography.caption)
                        .foregroundColor(themeManager.colors.textSecondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                Toggle("", isOn: $isEnabled)
                    .labelsHidden()
            }
            
            // Monitoring details
            HStack(spacing: 16) {
                MonitoringDetail(title: "Frequency", value: recommendation.frequency, icon: "clock")
                MonitoringDetail(title: "Duration", value: recommendation.duration, icon: "calendar")
            }
            
            // Metrics to track
            VStack(alignment: .leading, spacing: 8) {
                Text("Metrics to Track:")
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                    ForEach(recommendation.metrics, id: \.self) { metric in
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption2)
                            
                            Text(metric)
                                .font(themeManager.typography.caption2)
                                .foregroundColor(themeManager.colors.textSecondary)
                            
                            Spacer()
                        }
                    }
                }
            }
            
            if isEnabled {
                // Action buttons
                HStack {
                    Button("Set Reminders") {
                        setReminders()
                    }
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.primary)
                    
                    Spacer()
                    
                    Button("Customize") {
                        customize()
                    }
                    .font(themeManager.typography.caption)
                    .foregroundColor(.blue)
                }
                .padding(.top, 8)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(themeManager.colors.background)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isEnabled ? themeManager.colors.primary.opacity(0.3) : Color.clear, lineWidth: 1)
                )
        )
    }
    
    private func setReminders() {
        print("Setting reminders for: \(recommendation.title)")
    }
    
    private func customize() {
        print("Customizing monitoring: \(recommendation.title)")
    }
}

// MARK: - Monitoring Detail View

struct MonitoringDetail: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let title: String
    let value: String
    let icon: String
    
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .font(.caption)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(themeManager.typography.caption2)
                    .foregroundColor(themeManager.colors.textSecondary)
                
                Text(value)
                    .font(themeManager.typography.caption)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Follow-up Item View

struct FollowUpItem: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let followUp: InsightDetailView.FollowUpItem
    @State private var isScheduled = false
    
    var body: some View {
        HStack(spacing: 12) {
            // Date indicator
            VStack(spacing: 2) {
                Text(followUp.date, format: .dateTime.month(.abbreviated))
                    .font(themeManager.typography.caption2)
                    .foregroundColor(themeManager.colors.textSecondary)
                
                Text(followUp.date, format: .dateTime.day())
                    .font(themeManager.typography.title3)
                    .fontWeight(.bold)
                    .foregroundColor(themeManager.colors.primary)
            }
            .frame(width: 40)
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(followUp.title)
                        .font(themeManager.typography.body)
                        .fontWeight(.medium)
                        .foregroundColor(themeManager.colors.textPrimary)
                    
                    Spacer()
                    
                    Text(followUp.type)
                        .font(themeManager.typography.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(.blue.opacity(0.2))
                        )
                        .foregroundColor(.blue)
                }
                
                Text(followUp.description)
                    .font(themeManager.typography.caption)
                    .foregroundColor(themeManager.colors.textSecondary)
                    .lineLimit(2)
                
                HStack {
                    Button(isScheduled ? "Scheduled" : "Schedule") {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isScheduled.toggle()
                        }
                    }
                    .font(themeManager.typography.caption)
                    .foregroundColor(isScheduled ? .green : themeManager.colors.primary)
                    
                    Spacer()
                    
                    if isScheduled {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.caption)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(themeManager.colors.background)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isScheduled ? .green.opacity(0.3) : Color.clear, lineWidth: 1)
                )
        )
    }
}

// MARK: - Similar Insight Item View

struct SimilarInsightItem: View {
    @EnvironmentObject private var themeManager: ThemeManager
    let insight: InsightDetailView.SimilarInsightItem
    
    var body: some View {
        HStack(spacing: 12) {
            // Similarity indicator
            VStack {
                Text("\(Int(insight.similarity * 100))%")
                    .font(themeManager.typography.caption)
                    .fontWeight(.bold)
                    .foregroundColor(similarityColor(insight.similarity))
                
                Text("similar")
                    .font(themeManager.typography.caption2)
                    .foregroundColor(themeManager.colors.textSecondary)
            }
            .frame(width: 50)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(insight.title)
                    .font(themeManager.typography.body)
                    .fontWeight(.medium)
                    .foregroundColor(themeManager.colors.textPrimary)
                    .lineLimit(1)
                
                HStack {
                    Text(insight.date, style: .date)
                        .font(themeManager.typography.caption2)
                        .foregroundColor(themeManager.colors.textSecondary)
                    
                    Spacer()
                    
                    Text(insight.outcome)
                        .font(themeManager.typography.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(outcomeColor(insight.outcome).opacity(0.2))
                        )
                        .foregroundColor(outcomeColor(insight.outcome))
                }
            }
            
            Image(systemName: "chevron.right")
                .foregroundColor(themeManager.colors.textSecondary)
                .font(.caption)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(themeManager.colors.background)
        )
    }
    
    private func similarityColor(_ similarity: Double) -> Color {
        switch similarity {
        case 0.8...1.0: return .green
        case 0.6..<0.8: return .orange
        default: return .red
        }
    }
    
    private func outcomeColor(_ outcome: String) -> Color {
        switch outcome.lowercased() {
        case "implemented successfully", "successful": return .green
        case "partially effective", "partial": return .orange
        case "not effective", "failed": return .red
        default: return .blue
        }
    }
}

#Preview {
    VStack(spacing: 20) {
        ActionItem(
            action: InsightDetailView.ActionItem(
                id: UUID(),
                title: "Take preventive medication",
                description: "Weather forecast shows pressure drop in 2 hours",
                priority: .high,
                estimatedTime: "5 minutes",
                completed: false
            )
        )
        
        StrategyItem(
            strategy: InsightDetailView.StrategyItem(
                id: UUID(),
                title: "Optimize sleep schedule",
                description: "Consistent bedtime could reduce morning stiffness by 30%",
                timeframe: "2-4 weeks",
                expectedImpact: "High",
                difficulty: "Medium"
            )
        )
        
        MonitoringItem(
            recommendation: InsightDetailView.MonitoringItem(
                id: UUID(),
                title: "Track sleep quality",
                description: "Monitor sleep duration and quality for correlation analysis",
                frequency: "Daily",
                duration: "2 weeks",
                metrics: ["Duration", "Quality", "Interruptions"]
            )
        )
    }
    .padding()
    .environmentObject(ThemeManager())
}