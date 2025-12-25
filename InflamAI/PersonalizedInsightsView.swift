//
//  PersonalizedInsightsView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData

struct PersonalizedInsightsView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var insightsEngine = HealthInsightsEngine.shared
    
    @State private var recommendations: [HealthRecommendation] = []
    @State private var nudges: [HealthNudge] = []
    @State private var isLoading = true
    @State private var selectedTimeframe: TimeFrame = .week
    @State private var showingDetailSheet = false
    @State private var selectedRecommendation: HealthRecommendation?
    
    enum TimeFrame: String, CaseIterable {
        case week = "This Week"
        case month = "This Month"
        case quarter = "Last 3 Months"
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with timeframe selector
                VStack(spacing: 16) {
                    HStack {
                        Text("Personalized for You")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Spacer()
                        
                        Picker("Timeframe", selection: $selectedTimeframe) {
                            ForEach(TimeFrame.allCases, id: \.self) { timeframe in
                                Text(timeframe.rawValue).tag(timeframe)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                    }
                    
                    Text("Based on your health patterns and data")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding()
                .background(Color(.systemGroupedBackground))
                
                if isLoading {
                    LoadingView()
                } else {
                    ScrollView {
                        LazyVStack(spacing: 20) {
                            // Priority Nudges Section
                            if !nudges.isEmpty {
                                NudgesSection(nudges: nudges)
                            }
                            
                            // Recommendations Section
                            if !recommendations.isEmpty {
                                RecommendationsSection(
                                    recommendations: recommendations,
                                    onRecommendationTap: { recommendation in
                                        selectedRecommendation = recommendation
                                        showingDetailSheet = true
                                    }
                                )
                            }
                            
                            // Empty state
                            if recommendations.isEmpty && nudges.isEmpty {
                                EmptyPersonalizedView()
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Your Health Guide")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Refresh") {
                        refreshRecommendations()
                    }
                }
            }
            .sheet(isPresented: $showingDetailSheet) {
                if let recommendation = selectedRecommendation {
                    RecommendationDetailView(recommendation: recommendation)
                }
            }
        }
        .onAppear {
            refreshRecommendations()
        }
        .onChange(of: selectedTimeframe) { _ in
            refreshRecommendations()
        }
    }
    
    private func refreshRecommendations() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let generator = PersonalizedRecommendationGenerator()
            let newRecommendations = generator.generateRecommendations(
                context: viewContext,
                timeframe: selectedTimeframe
            )
            let newNudges = generator.generateNudges(
                context: viewContext,
                timeframe: selectedTimeframe
            )
            
            DispatchQueue.main.async {
                self.recommendations = newRecommendations
                self.nudges = newNudges
                self.isLoading = false
            }
        }
    }
}

// MARK: - Data Models

struct HealthRecommendation {
    let id = UUID()
    let title: String
    let description: String
    let category: RecommendationCategory
    let priority: Priority
    let confidence: Double
    let actionSteps: [String]
    let expectedBenefit: String
    let timeToSeeResults: String
    let basedOnPattern: String
}

struct HealthNudge {
    let id = UUID()
    let title: String
    let message: String
    let type: NudgeType
    let urgency: Urgency
    let actionRequired: Bool
    let dismissible: Bool
}

enum RecommendationCategory: String, CaseIterable {
    case medication = "Medication"
    case activity = "Activity"
    case sleep = "Sleep"
    case stress = "Stress Management"
    case nutrition = "Nutrition"
    case lifestyle = "Lifestyle"
}

enum Priority: String, CaseIterable {
    case high = "High"
    case medium = "Medium"
    case low = "Low"
}

enum NudgeType: String, CaseIterable {
    case reminder = "Reminder"
    case warning = "Warning"
    case encouragement = "Encouragement"
    case suggestion = "Suggestion"
}

enum Urgency: String, CaseIterable {
    case immediate = "Immediate"
    case today = "Today"
    case thisWeek = "This Week"
    case general = "General"
}

// MARK: - Recommendation Generator

class PersonalizedRecommendationGenerator {
    
    func generateRecommendations(
        context: NSManagedObjectContext,
        timeframe: PersonalizedInsightsView.TimeFrame
    ) -> [HealthRecommendation] {
        let insights = HealthInsightsEngine.shared.analyzeHealthData(context: context)
        let correlations = HealthInsightsEngine.shared.calculateCorrelations(context: context)
        let medicationEffects = HealthInsightsEngine.shared.analyzeMedicationEffectiveness(context: context)
        
        var recommendations: [HealthRecommendation] = []
        
        // Generate medication recommendations
        recommendations.append(contentsOf: generateMedicationRecommendations(effects: medicationEffects))
        
        // Generate activity recommendations
        recommendations.append(contentsOf: generateActivityRecommendations(insights: insights))
        
        // Generate sleep recommendations
        recommendations.append(contentsOf: generateSleepRecommendations(correlations: correlations))
        
        // Generate stress management recommendations
        recommendations.append(contentsOf: generateStressRecommendations(insights: insights))
        
        return recommendations.sorted { $0.priority.rawValue < $1.priority.rawValue }
    }
    
    func generateNudges(
        context: NSManagedObjectContext,
        timeframe: PersonalizedInsightsView.TimeFrame
    ) -> [HealthNudge] {
        var nudges: [HealthNudge] = []
        
        // Check for recent patterns that need immediate attention
        nudges.append(contentsOf: generateImmediateNudges(context: context))
        
        // Generate encouragement nudges
        nudges.append(contentsOf: generateEncouragementNudges(context: context))
        
        // Generate reminder nudges
        nudges.append(contentsOf: generateReminderNudges(context: context))
        
        return nudges.sorted { $0.urgency.rawValue < $1.urgency.rawValue }
    }
    
    // MARK: - Private Generation Methods
    
    private func generateMedicationRecommendations(effects: [HealthInsightsEngine.MedicationEffect]) -> [HealthRecommendation] {
        var recommendations: [HealthRecommendation] = []
        
        for effect in effects.prefix(3) {
            if effect.effectOnPain > 1.5 && effect.confidence > 0.6 {
                recommendations.append(HealthRecommendation(
                    title: "Optimize \(effect.medicationName) Timing",
                    description: "Your data shows \(effect.medicationName) is highly effective for pain relief.",
                    category: .medication,
                    priority: .high,
                    confidence: effect.confidence,
                    actionSteps: [
                        "Take \(effect.medicationName) consistently at the same time daily",
                        "Track pain levels 2-4 hours after taking medication",
                        "Discuss optimal timing with your healthcare provider"
                    ],
                    expectedBenefit: "\(String(format: "%.1f", effect.effectOnPain)) point reduction in pain levels",
                    timeToSeeResults: "2-4 hours after taking medication",
                    basedOnPattern: "Analysis of \(effect.sampleSize) medication doses"
                ))
            }
            
            if effect.effectOnEnergy < -1.0 && effect.confidence > 0.5 {
                recommendations.append(HealthRecommendation(
                    title: "Address \(effect.medicationName) Fatigue",
                    description: "\(effect.medicationName) appears to be causing fatigue side effects.",
                    category: .medication,
                    priority: .medium,
                    confidence: effect.confidence,
                    actionSteps: [
                        "Take medication in the evening if possible",
                        "Ensure adequate rest after taking medication",
                        "Discuss alternative timing or dosage with your doctor"
                    ],
                    expectedBenefit: "Reduced fatigue and improved energy levels",
                    timeToSeeResults: "1-2 weeks with timing adjustments",
                    basedOnPattern: "Fatigue correlation from \(effect.sampleSize) doses"
                ))
            }
        }
        
        return recommendations
    }
    
    private func generateActivityRecommendations(insights: [HealthInsightsEngine.PatternInsight]) -> [HealthRecommendation] {
        var recommendations: [HealthRecommendation] = []
        
        let activityInsights = insights.filter { $0.type == .activityCorrelation }
        
        for insight in activityInsights.prefix(2) {
            if insight.confidence > 0.5 {
                let isPositive = insight.title.contains("Helps")
                
                recommendations.append(HealthRecommendation(
                    title: isPositive ? "Increase Beneficial Activity" : "Modify Challenging Activity",
                    description: insight.description,
                    category: .activity,
                    priority: isPositive ? .medium : .high,
                    confidence: insight.confidence,
                    actionSteps: isPositive ? [
                        "Schedule this activity 3-4 times per week",
                        "Start with shorter durations and gradually increase",
                        "Track how you feel before and after the activity"
                    ] : [
                        "Reduce intensity or duration of this activity",
                        "Consider alternative approaches or modifications",
                        "Monitor pain levels when engaging in this activity"
                    ],
                    expectedBenefit: isPositive ? "Improved pain management and wellbeing" : "Reduced pain flares and better symptom control",
                    timeToSeeResults: "2-3 weeks of consistent changes",
                    basedOnPattern: insight.description
                ))
            }
        }
        
        return recommendations
    }
    
    private func generateSleepRecommendations(correlations: [HealthInsightsEngine.CorrelationResult]) -> [HealthRecommendation] {
        var recommendations: [HealthRecommendation] = []
        
        let sleepCorrelations = correlations.filter { 
            $0.factor1.contains("Sleep") || $0.factor2.contains("Sleep")
        }
        
        for correlation in sleepCorrelations.prefix(1) {
            if correlation.confidence > 0.5 {
                recommendations.append(HealthRecommendation(
                    title: "Improve Sleep Quality",
                    description: "Better sleep quality is strongly linked to reduced pain and improved wellbeing.",
                    category: .sleep,
                    priority: .high,
                    confidence: correlation.confidence,
                    actionSteps: [
                        "Maintain a consistent sleep schedule",
                        "Create a relaxing bedtime routine",
                        "Limit screen time 1 hour before bed",
                        "Keep your bedroom cool and dark"
                    ],
                    expectedBenefit: "Better pain management and increased energy",
                    timeToSeeResults: "1-2 weeks of consistent sleep hygiene",
                    basedOnPattern: correlation.description
                ))
            }
        }
        
        return recommendations
    }
    
    private func generateStressRecommendations(insights: [HealthInsightsEngine.PatternInsight]) -> [HealthRecommendation] {
        var recommendations: [HealthRecommendation] = []
        
        let moodInsights = insights.filter { $0.type == .moodInfluencer }
        
        for insight in moodInsights.prefix(1) {
            if insight.confidence > 0.4 && insight.title.contains("Higher Pain") {
                recommendations.append(HealthRecommendation(
                    title: "Stress Management Strategy",
                    description: "Negative mood states are associated with increased pain levels.",
                    category: .stress,
                    priority: .medium,
                    confidence: insight.confidence,
                    actionSteps: [
                        "Practice daily mindfulness or meditation",
                        "Try deep breathing exercises during stressful moments",
                        "Consider gentle yoga or stretching",
                        "Maintain social connections and support networks"
                    ],
                    expectedBenefit: "Improved mood and reduced pain sensitivity",
                    timeToSeeResults: "2-4 weeks of regular practice",
                    basedOnPattern: insight.description
                ))
            }
        }
        
        return recommendations
    }
    
    private func generateImmediateNudges(context: NSManagedObjectContext) -> [HealthNudge] {
        var nudges: [HealthNudge] = []
        
        // Check for missed medication tracking
        let today = Calendar.current.startOfDay(for: Date())
        let medicationRequest: NSFetchRequest<MedicationIntake> = MedicationIntake.fetchRequest()
        medicationRequest.predicate = NSPredicate(format: "timestamp >= %@", today as CVarArg)
        
        do {
            let todayIntakes = try context.fetch(medicationRequest)
            if todayIntakes.isEmpty {
                nudges.append(HealthNudge(
                    title: "Medication Tracking",
                    message: "Don't forget to track your medications today for better insights.",
                    type: .reminder,
                    urgency: .today,
                    actionRequired: false,
                    dismissible: true
                ))
            }
        } catch {
            print("Error checking medication tracking: \(error)")
        }
        
        return nudges
    }
    
    private func generateEncouragementNudges(context: NSManagedObjectContext) -> [HealthNudge] {
        var nudges: [HealthNudge] = []
        
        // Check tracking streak
        let calendar = Calendar.current
        let last7Days = calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(format: "timestamp >= %@", last7Days as CVarArg)
        
        do {
            let recentEntries = try context.fetch(painRequest)
            let uniqueDays = Set(recentEntries.compactMap { entry in
                entry.timestamp.map { calendar.startOfDay(for: $0) }
            })
            
            if uniqueDays.count >= 5 {
                nudges.append(HealthNudge(
                    title: "Great Progress!",
                    message: "You've been consistently tracking your health. Keep it up!",
                    type: .encouragement,
                    urgency: .general,
                    actionRequired: false,
                    dismissible: true
                ))
            }
        } catch {
            print("Error checking tracking streak: \(error)")
        }
        
        return nudges
    }
    
    private func generateReminderNudges(context: NSManagedObjectContext) -> [HealthNudge] {
        var nudges: [HealthNudge] = []
        
        // Check for BASDAI assessment
        let lastWeek = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let bassdaiRequest: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
        bassdaiRequest.predicate = NSPredicate(format: "date >= %@", lastWeek as CVarArg)
        
        do {
            let recentAssessments = try context.fetch(bassdaiRequest)
            if recentAssessments.isEmpty {
                nudges.append(HealthNudge(
                    title: "Weekly Assessment",
                    message: "Consider completing a BASDAI assessment to track your progress.",
                    type: .suggestion,
                    urgency: .thisWeek,
                    actionRequired: false,
                    dismissible: true
                ))
            }
        } catch {
            print("Error checking BASDAI assessments: \(error)")
        }
        
        return nudges
    }
}

// MARK: - View Components

struct NudgesSection: View {
    let nudges: [HealthNudge]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Priority Nudges")
                .font(.headline)
                .foregroundColor(.primary)
            
            ForEach(nudges.prefix(3), id: \.id) { nudge in
                NudgeCard(nudge: nudge)
            }
        }
    }
}

struct NudgeCard: View {
    let nudge: HealthNudge
    @State private var isDismissed = false
    
    var body: some View {
        if !isDismissed {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(nudge.title)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Spacer()
                        
                        NudgeTypeIcon(type: nudge.type)
                    }
                    
                    Text(nudge.message)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                if nudge.dismissible {
                    Button {
                        withAnimation {
                            isDismissed = true
                        }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding()
            .background(nudgeBackgroundColor)
            .cornerRadius(8)
        }
    }
    
    private var nudgeBackgroundColor: Color {
        switch nudge.type {
        case .warning: return Color.red.opacity(0.1)
        case .encouragement: return Color.green.opacity(0.1)
        case .reminder: return Color.blue.opacity(0.1)
        case .suggestion: return Color.orange.opacity(0.1)
        }
    }
}

struct NudgeTypeIcon: View {
    let type: NudgeType
    
    var body: some View {
        Image(systemName: iconName)
            .foregroundColor(iconColor)
            .font(.caption)
    }
    
    private var iconName: String {
        switch type {
        case .warning: return "exclamationmark.triangle.fill"
        case .encouragement: return "hand.thumbsup.fill"
        case .reminder: return "bell.fill"
        case .suggestion: return "lightbulb.fill"
        }
    }
    
    private var iconColor: Color {
        switch type {
        case .warning: return .red
        case .encouragement: return .green
        case .reminder: return .blue
        case .suggestion: return .orange
        }
    }
}

struct RecommendationsSection: View {
    let recommendations: [HealthRecommendation]
    let onRecommendationTap: (HealthRecommendation) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Personalized Recommendations")
                .font(.headline)
                .foregroundColor(.primary)
            
            ForEach(recommendations.prefix(5), id: \.id) { recommendation in
                RecommendationCard(recommendation: recommendation) {
                    onRecommendationTap(recommendation)
                }
            }
        }
    }
}

struct RecommendationCard: View {
    let recommendation: HealthRecommendation
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(recommendation.title)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                        
                        Text(recommendation.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                    }
                    
                    Spacer()
                    
                    VStack {
                        PriorityBadge(priority: recommendation.priority)
                        CategoryIcon(category: recommendation.category)
                    }
                }
                
                HStack {
                    Text(recommendation.expectedBenefit)
                        .font(.caption2)
                        .foregroundColor(.blue)
                    
                    Spacer()
                    
                    Text("\(Int(recommendation.confidence * 100))% confidence")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(8)
            .shadow(color: .black.opacity(0.05), radius: 1, x: 0, y: 1)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct PriorityBadge: View {
    let priority: Priority
    
    var body: some View {
        Text(priority.rawValue)
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(priorityColor.opacity(0.2))
            .foregroundColor(priorityColor)
            .cornerRadius(4)
    }
    
    private var priorityColor: Color {
        switch priority {
        case .high: return .red
        case .medium: return .orange
        case .low: return .green
        }
    }
}

struct CategoryIcon: View {
    let category: RecommendationCategory
    
    var body: some View {
        Image(systemName: iconName)
            .foregroundColor(.blue)
            .font(.caption)
    }
    
    private var iconName: String {
        switch category {
        case .medication: return "pills.fill"
        case .activity: return "figure.walk"
        case .sleep: return "moon.fill"
        case .stress: return "brain.head.profile"
        case .nutrition: return "leaf.fill"
        case .lifestyle: return "heart.fill"
        }
    }
}

struct EmptyPersonalizedView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "person.crop.circle.badge.checkmark")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text("Building Your Profile")
                .font(.headline)
                .foregroundColor(.primary)
            
            Text("Keep tracking your health data to receive personalized recommendations and insights.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - Recommendation Detail View

struct RecommendationDetailView: View {
    let recommendation: HealthRecommendation
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        Text(recommendation.title)
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        Text(recommendation.description)
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    // Action Steps
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Action Steps")
                            .font(.headline)
                        
                        ForEach(Array(recommendation.actionSteps.enumerated()), id: \.offset) { index, step in
                            HStack(alignment: .top, spacing: 12) {
                                Text("\(index + 1)")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .foregroundColor(.white)
                                    .frame(width: 20, height: 20)
                                    .background(Color.blue)
                                    .clipShape(Circle())
                                
                                Text(step)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                                
                                Spacer()
                            }
                        }
                    }
                    
                    // Expected Benefits
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Expected Benefits")
                            .font(.headline)
                        
                        Text(recommendation.expectedBenefit)
                            .font(.body)
                            .padding()
                            .background(Color.green.opacity(0.1))
                            .cornerRadius(8)
                    }
                    
                    // Timeline
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Timeline")
                            .font(.headline)
                        
                        Text(recommendation.timeToSeeResults)
                            .font(.body)
                            .padding()
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                    }
                    
                    // Based On
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Based On")
                            .font(.headline)
                        
                        Text(recommendation.basedOnPattern)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                    }
                }
                .padding()
            }
            .navigationTitle("Recommendation")
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
    PersonalizedInsightsView()
        let container = NSPersistentContainer(name: "InflamAI")
        container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        container.loadPersistentStores { _, _ in }
        let context = container.viewContext
        
        return PersonalizedInsightsView()
            .environment(\.managedObjectContext, context)
}