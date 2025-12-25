//
//  BASSDAIView.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import SwiftUI
import CoreData

struct BASSDAIView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: false)],
        animation: .default)
    private var assessments: FetchedResults<BASSDAIAssessment>
    
    @State private var showingNewAssessment = false
    @State private var showingPracticeQuestions = false
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(spacing: 20) {
                // Latest Score Card
                if let latestAssessment = assessments.first {
                    LatestScoreCard(assessment: latestAssessment)
                } else {
                    EmptyStateCard()
                }

                // Assessment History
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Assessment History")
                            .font(.headline)
                        Spacer()
                        Button("Practice") {
                            showingPracticeQuestions = true
                        }
                        .buttonStyle(.bordered)

                        Button("New Assessment") {
                            showingNewAssessment = true
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(Colors.Primary.p500)
                    }

                    if assessments.isEmpty {
                        Text("No assessments yet. Take your first BASDAI assessment to track your symptoms.")
                            .foregroundColor(.secondary)
                            .padding()
                    } else {
                        ForEach(assessments, id: \.objectID) { assessment in
                            AssessmentRow(assessment: assessment)
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
        .navigationTitle("BASDAI")
        .sheet(isPresented: $showingNewAssessment) {
            NewBASSDAIAssessmentView()
                .environment(\.managedObjectContext, viewContext)
        }
        .sheet(isPresented: $showingPracticeQuestions) {
            BASSDAIPracticeView()
        }
    }
}

struct LatestScoreCard: View {
    let assessment: BASSDAIAssessment
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Latest BASDAI Score")
                    .font(.headline)
                Spacer()
                Text(assessment.date ?? Date(), style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            HStack {
                VStack {
                    Text(String(format: "%.1f", assessment.totalScore))
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundColor(scoreColor(assessment.totalScore))
                    Text("out of 10")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text(scoreDescription(assessment.totalScore))
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(scoreColor(assessment.totalScore))
                    Text(scoreAdvice(assessment.totalScore))
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.trailing)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .padding(.horizontal)
    }
    
    private func scoreColor(_ score: Double) -> Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return .yellow
        case 4..<6: return .orange
        default: return .red
        }
    }
    
    private func scoreDescription(_ score: Double) -> String {
        switch score {
        case 0..<2: return "Low Activity"
        case 2..<4: return "Moderate Activity"
        case 4..<6: return "High Activity"
        default: return "Very High Activity"
        }
    }
    
    private func scoreAdvice(_ score: Double) -> String {
        switch score {
        case 0..<2: return "Symptoms are well controlled"
        case 2..<4: return "Consider discussing with your doctor"
        case 4..<6: return "Consult your healthcare provider"
        default: return "Seek immediate medical attention"
        }
    }
}

struct EmptyStateCard: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 48))
                .foregroundColor(.blue)
            
            Text("Track Your BASDAI Score")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("The Bath Ankylosing Spondylitis Disease Activity Index helps monitor your symptoms over time.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

struct AssessmentRow: View {
    let assessment: BASSDAIAssessment
    @State private var showingDetail = false
    
    var body: some View {
        Button {
            showingDetail = true
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(assessment.date ?? Date(), style: .date)
                        .font(.headline)
                    Text(assessment.date ?? Date(), style: .time)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(String(format: "%.1f", assessment.totalScore))
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(scoreColor(assessment.totalScore))
                    Text(scoreDescription(assessment.totalScore))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(8)
            .shadow(radius: 1)
        }
        .buttonStyle(PlainButtonStyle())
        .sheet(isPresented: $showingDetail) {
            AssessmentDetailView(assessment: assessment)
        }
    }
    
    private func scoreColor(_ score: Double) -> Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return .yellow
        case 4..<6: return .orange
        default: return .red
        }
    }
    
    private func scoreDescription(_ score: Double) -> String {
        switch score {
        case 0..<2: return "Low"
        case 2..<4: return "Moderate"
        case 4..<6: return "High"
        default: return "Very High"
        }
    }
}

struct NewBASSDAIAssessmentView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss
    
    @State private var currentQuestionIndex = 0
    @State private var responses: [Double] = Array(repeating: 0.0, count: 6)
    @State private var isLoading = false
    @State private var showingSuccessAlert = false
    
    private let questions = [
        "How would you describe the overall level of fatigue/tiredness you have experienced?",
        "How would you describe the overall level of AS neck, back or hip pain you have had?",
        "How would you describe the overall level of pain/swelling in joints other than neck, back or hips you have had?",
        "How would you describe the overall level of pain/swelling you have had?",
        "How would you describe the overall level of morning stiffness you have had from the time you wake up?",
        "How would you describe your overall wellbeing?"
    ]
    
    private var totalScore: Double {
        let sum = responses.prefix(4).reduce(0, +)
        let morningStiffnessScore = responses[4] // Duration converted to 0-10 scale
        let wellbeingScore = responses[5]
        return (sum + morningStiffnessScore + wellbeingScore) / 6.0
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Progress indicator
                ProgressView(value: Double(currentQuestionIndex + 1), total: 6)
                    .progressViewStyle(LinearProgressViewStyle())
                    .padding(.horizontal)
                
                Text("Question \(currentQuestionIndex + 1) of 6")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                // Question
                VStack(spacing: 20) {
                    Text(questions[currentQuestionIndex])
                        .font(.title3)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    
                    // Response slider
                    VStack(spacing: 12) {
                        if currentQuestionIndex == 4 {
                            // Special handling for morning stiffness duration (0-10 scale represents 0-120+ minutes)
                            let durationMinutes = Int(responses[currentQuestionIndex] * 12)
                            Text("Duration: \(durationMinutes) minutes")
                                .font(.system(size: 48, weight: .bold, design: .rounded))
                                .foregroundColor(.blue)
                        } else {
                            Text(String(format: "%.1f", responses[currentQuestionIndex]))
                                .font(.system(size: 48, weight: .bold, design: .rounded))
                                .foregroundColor(.blue)
                        }

                        Slider(value: $responses[currentQuestionIndex], in: 0...10, step: 0.1)
                            .padding(.horizontal)
                            .accentColor(.blue)

                        HStack {
                            Text(currentQuestionIndex == 4 ? "0 minutes" : "None")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(currentQuestionIndex == 4 ? "120+ minutes" : "Very severe")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
                    }
                }
                
                Spacer()
                
                // Navigation buttons
                HStack(spacing: 16) {
                    if currentQuestionIndex > 0 {
                        Button("Previous") {
                            currentQuestionIndex -= 1
                        }
                        .buttonStyle(.bordered)
                    }
                    
                    Spacer()
                    
                    if currentQuestionIndex < 5 {
                        Button("Next") {
                            currentQuestionIndex += 1
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(Colors.Primary.p500)
                    } else {
                        Button("Complete Assessment") {
                            saveAssessment()
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(Colors.Primary.p500)
                    }
                }
                .padding(.horizontal)
            }
            .padding()
            .navigationTitle("BASDAI Assessment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .coreDataErrorAlert()
            .alert("Success", isPresented: $showingSuccessAlert) {
                Button("OK") {
                    dismiss()
                }
            } message: {
                Text("BASDAI assessment has been saved successfully.")
            }
        }
    }
    
    private func saveAssessment() {
        isLoading = true
        
        // Create BASDAI assessment using safe operations
        let result: Result<BASSDAIAssessment, CoreDataError> = CoreDataOperations.createEntity(
            entityName: "BASSDAIAssessment",
            context: viewContext
        )
        
        switch result {
        case .success(let assessment):
            // Set assessment properties
            assessment.date = Date()
            assessment.fatigue = responses[0]
            assessment.spinalPain = responses[1]
            assessment.neckShoulderHipPain = responses[2]
            assessment.swellingPain = responses[3]
            assessment.morningStiffness = responses[4]
            assessment.overallWellbeing = responses[5]
            assessment.totalScore = totalScore
            
            // Save with proper error handling
            CoreDataOperations.safeSave(context: viewContext) { saveResult in
                DispatchQueue.main.async {
                    self.isLoading = false
                    
                    switch saveResult {
                    case .success:
                        self.showingSuccessAlert = true
                        
                    case .failure:
                        // Error already handled by CoreDataOperations
                        break
                    }
                }
            }
            
        case .failure:
            isLoading = false
            // Error already handled by CoreDataOperations
        }
    }
}

struct AssessmentDetailView: View {
    let assessment: BASSDAIAssessment
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Overall Score
                    VStack(spacing: 12) {
                        Text("Overall BASDAI Score")
                            .font(.headline)
                        
                        Text(String(format: "%.1f", assessment.totalScore))
                            .font(.system(size: 48, weight: .bold, design: .rounded))
                            .foregroundColor(scoreColor(assessment.totalScore))
                        
                        Text(scoreDescription(assessment.totalScore))
                            .font(.title3)
                            .foregroundColor(scoreColor(assessment.totalScore))
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Individual Scores
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Individual Scores")
                            .font(.headline)
                        
                        ScoreRow(title: "Fatigue", score: assessment.fatigue)
                        ScoreRow(title: "Spinal Pain", score: assessment.spinalPain)
                        ScoreRow(title: "Neck/Shoulder/Hip Pain", score: assessment.neckShoulderHipPain)
                        ScoreRow(title: "Swelling Pain", score: assessment.swellingPain)
                        ScoreRow(title: "Morning Stiffness", score: assessment.morningStiffness)
                        ScoreRow(title: "Overall Wellbeing", score: assessment.overallWellbeing)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
                .padding()
            }
            .navigationTitle("Assessment Details")
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
    
    private func scoreColor(_ score: Double) -> Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return .yellow
        case 4..<6: return .orange
        default: return .red
        }
    }
    
    private func scoreDescription(_ score: Double) -> String {
        switch score {
        case 0..<2: return "Low Activity"
        case 2..<4: return "Moderate Activity"
        case 4..<6: return "High Activity"
        default: return "Very High Activity"
        }
    }
}

struct ScoreRow: View {
    let title: String
    let score: Double
    let isDuration: Bool
    
    init(title: String, score: Double, isDuration: Bool = false) {
        self.title = title
        self.score = score
        self.isDuration = isDuration
    }
    
    var body: some View {
        HStack {
            Text(title)
                .font(.body)
            Spacer()
            if isDuration {
                Text("\(Int(score * 12)) min")
                    .font(.body)
                    .fontWeight(.semibold)
            } else {
                Text(String(format: "%.1f", score))
                    .font(.body)
                    .fontWeight(.semibold)
            }
        }
        .padding(.vertical, 4)
    }
}

struct BASSDAIView_Previews: PreviewProvider {
    static var previews: some View {
        BASSDAIView()
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}