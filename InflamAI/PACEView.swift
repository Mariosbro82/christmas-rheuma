//
//  PACEView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData

struct PACEView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var paceEngine = PACESuggestionEngine.shared
    @State private var currentSuggestion: PACESuggestion?
    @State private var isLoading = true
    @State private var showingActivityDetail = false
    @State private var selectedActivity = ""
    @State private var hasCompletedActivity = false
    @State private var showingFeedback = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    if isLoading {
                        LoadingView()
                    } else if let suggestion = currentSuggestion {
                        VStack(spacing: 20) {
                            // Header Card
                            HeaderCard(suggestion: suggestion)
                            
                            // Main Activity Card
                            MainActivityCard(
                                suggestion: suggestion,
                                selectedActivity: $selectedActivity,
                                showingActivityDetail: $showingActivityDetail
                            )
                            
                            // Alternative Activities
                            if !suggestion.alternativeActivities.isEmpty {
                                AlternativeActivitiesCard(
                                    activities: suggestion.alternativeActivities,
                                    selectedActivity: $selectedActivity,
                                    showingActivityDetail: $showingActivityDetail
                                )
                            }
                            
                            // Timing and Duration
                            TimingCard(suggestion: suggestion)
                            
                            // Precautions
                            if !suggestion.precautions.isEmpty {
                                PrecautionsCard(precautions: suggestion.precautions)
                            }
                            
                            // Motivation
                            MotivationCard(motivation: suggestion.motivation)
                            
                            // Action Buttons
                            ActionButtonsCard(
                                hasCompleted: $hasCompletedActivity,
                                showingFeedback: $showingFeedback
                            )
                        }
                        .padding(.horizontal)
                    } else {
                        ErrorView {
                            loadSuggestion()
                        }
                    }
                }
            }
            .navigationTitle("Today's PACE")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Refresh") {
                        loadSuggestion()
                    }
                }
            }
            .sheet(isPresented: $showingActivityDetail) {
                ActivityDetailSheet(
                    activity: selectedActivity,
                    suggestion: currentSuggestion
                )
            }
            .sheet(isPresented: $showingFeedback) {
                FeedbackSheet(suggestion: currentSuggestion)
            }
        }
        .onAppear {
            loadSuggestion()
        }
    }
    
    private func loadSuggestion() {
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let suggestion = paceEngine.generateDailySuggestion(context: viewContext)
            
            DispatchQueue.main.async {
                self.currentSuggestion = suggestion
                self.isLoading = false
            }
        }
    }
}

// MARK: - Header Card

struct HeaderCard: View {
    let suggestion: PACESuggestion
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Today's Recommendation")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text(suggestion.paceLevel.rawValue)
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(colorForPaceLevel(suggestion.paceLevel))
                }
                
                Spacer()
                
                VStack(spacing: 4) {
                    Image(systemName: iconForPaceLevel(suggestion.paceLevel))
                        .font(.system(size: 40))
                        .foregroundColor(colorForPaceLevel(suggestion.paceLevel))
                    
                    Text("\(Int(suggestion.confidence * 100))% confident")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Text(suggestion.paceLevel.description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func colorForPaceLevel(_ level: PACELevel) -> Color {
        switch level {
        case .rest: return .red
        case .gentle: return .orange
        case .moderate: return .yellow
        case .active: return .green
        }
    }
    
    private func iconForPaceLevel(_ level: PACELevel) -> String {
        switch level {
        case .rest: return "bed.double.fill"
        case .gentle: return "figure.walk"
        case .moderate: return "figure.run"
        case .active: return "figure.strengthtraining.traditional"
        }
    }
}

// MARK: - Main Activity Card

struct MainActivityCard: View {
    let suggestion: PACESuggestion
    @Binding var selectedActivity: String
    @Binding var showingActivityDetail: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recommended Activity")
                .font(.headline)
                .foregroundColor(.primary)
            
            Button(action: {
                selectedActivity = suggestion.primaryActivity
                showingActivityDetail = true
            }) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(suggestion.primaryActivity)
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(.primary)
                        
                        Text("Duration: \(suggestion.duration)")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Alternative Activities Card

struct AlternativeActivitiesCard: View {
    let activities: [String]
    @Binding var selectedActivity: String
    @Binding var showingActivityDetail: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Alternative Options")
                .font(.headline)
                .foregroundColor(.primary)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 8) {
                ForEach(activities, id: \.self) { activity in
                    Button(action: {
                        selectedActivity = activity
                        showingActivityDetail = true
                    }) {
                        Text(activity)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                            .padding(.vertical, 8)
                            .padding(.horizontal, 12)
                            .frame(maxWidth: .infinity)
                            .background(Color(.systemBackground))
                            .cornerRadius(8)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color(.systemGray4), lineWidth: 1)
                            )
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Timing Card

struct TimingCard: View {
    let suggestion: PACESuggestion
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Timing & Duration")
                .font(.headline)
                .foregroundColor(.primary)
            
            VStack(spacing: 8) {
                HStack {
                    Image(systemName: "clock")
                        .foregroundColor(.blue)
                    Text(suggestion.recommendedTiming)
                        .font(.subheadline)
                    Spacer()
                }
                
                HStack {
                    Image(systemName: "timer")
                        .foregroundColor(.green)
                    Text(suggestion.duration)
                        .font(.subheadline)
                    Spacer()
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Precautions Card

struct PrecautionsCard: View {
    let precautions: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundColor(.orange)
                Text("Important Reminders")
                    .font(.headline)
                    .foregroundColor(.primary)
            }
            
            VStack(alignment: .leading, spacing: 6) {
                ForEach(precautions, id: \.self) { precaution in
                    HStack(alignment: .top) {
                        Text("•")
                            .foregroundColor(.orange)
                        Text(precaution)
                            .font(.subheadline)
                            .foregroundColor(.primary)
                        Spacer()
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Motivation Card

struct MotivationCard: View {
    let motivation: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "heart.fill")
                    .foregroundColor(.pink)
                Text("Motivation")
                    .font(.headline)
                    .foregroundColor(.primary)
            }
            
            Text(motivation)
                .font(.subheadline)
                .foregroundColor(.primary)
                .italic()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Action Buttons Card

struct ActionButtonsCard: View {
    @Binding var hasCompleted: Bool
    @Binding var showingFeedback: Bool
    
    var body: some View {
        VStack(spacing: 12) {
            if !hasCompleted {
                Button(action: {
                    hasCompleted = true
                    showingFeedback = true
                }) {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                        Text("Mark as Completed")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .cornerRadius(12)
                }
            } else {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Activity Completed!")
                        .font(.headline)
                        .foregroundColor(.green)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.green.opacity(0.1))
                .cornerRadius(12)
            }
            
            Button(action: {
                showingFeedback = true
            }) {
                HStack {
                    Image(systemName: "message")
                    Text("Provide Feedback")
                }
                .font(.subheadline)
                .foregroundColor(.blue)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue.opacity(0.1))
                .cornerRadius(12)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Activity Detail Sheet

struct ActivityDetailSheet: View {
    let activity: String
    let suggestion: PACESuggestion?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(activity)
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    if let suggestion = suggestion {
                        VStack(alignment: .leading, spacing: 16) {
                            DetailSection(title: "Duration", content: suggestion.duration)
                            DetailSection(title: "Best Time", content: suggestion.recommendedTiming)
                            
                            if !suggestion.precautions.isEmpty {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Safety Tips")
                                        .font(.headline)
                                    
                                    ForEach(suggestion.precautions, id: \.self) { precaution in
                                        HStack(alignment: .top) {
                                            Text("•")
                                            Text(precaution)
                                            Spacer()
                                        }
                                    }
                                }
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(8)
                            }
                        }
                    }
                    
                    // Activity-specific tips
                    ActivityTips(activity: activity)
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle("Activity Details")
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

struct DetailSection: View {
    let title: String
    let content: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.headline)
            Text(content)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

struct ActivityTips: View {
    let activity: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Tips for \(activity)")
                .font(.headline)
            
            ForEach(tipsForActivity(activity), id: \.self) { tip in
                HStack(alignment: .top) {
                    Text("💡")
                    Text(tip)
                    Spacer()
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
    
    private func tipsForActivity(_ activity: String) -> [String] {
        let lowercased = activity.lowercased()
        
        if lowercased.contains("walk") {
            return [
                "Start slowly and gradually increase pace",
                "Wear comfortable, supportive shoes",
                "Choose flat, even surfaces when possible"
            ]
        } else if lowercased.contains("yoga") || lowercased.contains("stretch") {
            return [
                "Never force a stretch",
                "Breathe deeply and slowly",
                "Hold stretches for 15-30 seconds"
            ]
        } else if lowercased.contains("swim") {
            return [
                "Water temperature should be comfortable",
                "Start with gentle movements",
                "Pool walking is a great low-impact option"
            ]
        } else if lowercased.contains("meditat") || lowercased.contains("breath") {
            return [
                "Find a quiet, comfortable space",
                "Start with just 5 minutes",
                "Focus on your breath"
            ]
        } else {
            return [
                "Listen to your body",
                "Start gently and build up slowly",
                "Stop if you feel pain or discomfort"
            ]
        }
    }
}

// MARK: - Feedback Sheet

struct FeedbackSheet: View {
    let suggestion: PACESuggestion?
    @Environment(\.dismiss) private var dismiss
    @State private var rating = 3
    @State private var feedback = ""
    @State private var wasHelpful = true
    
    var body: some View {
        NavigationView {
            Form {
                Section("How was today's suggestion?") {
                    HStack {
                        Text("Rating:")
                        Spacer()
                        HStack {
                            ForEach(1...5, id: \.self) { star in
                                Button(action: {
                                    rating = star
                                }) {
                                    Image(systemName: star <= rating ? "star.fill" : "star")
                                        .foregroundColor(.yellow)
                                }
                            }
                        }
                    }
                    
                    Toggle("Was this helpful?", isOn: $wasHelpful)
                }
                
                Section("Additional Comments") {
                    TextField("How did the activity go? Any suggestions?", text: $feedback, axis: .vertical)
                        .lineLimit(3...6)
                }
            }
            .navigationTitle("Feedback")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Submit") {
                        submitFeedback()
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func submitFeedback() {
        // Here you would typically save the feedback to Core Data or send to analytics
        print("Feedback submitted: Rating \(rating), Helpful: \(wasHelpful), Comments: \(feedback)")
    }
}

// MARK: - Supporting Views

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Analyzing your health data...")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct ErrorView: View {
    let retry: () -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 50))
                .foregroundColor(.orange)
            
            Text("Unable to generate suggestion")
                .font(.headline)
            
            Text("Please try again or check your data.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            Button("Try Again") {
                retry()
            }
            .buttonStyle(.borderedProminent)
            .tint(Colors.Primary.p500)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

#Preview {
    PACEView()
        let container = NSPersistentContainer(name: "InflamAI")
        container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        container.loadPersistentStores { _, _ in }
        let context = container.viewContext
        
        return PACEView()
            .environment(\.managedObjectContext, context)
}