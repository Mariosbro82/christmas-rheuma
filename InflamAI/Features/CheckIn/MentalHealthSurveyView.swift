//
//  MentalHealthSurveyView.swift
//  InflamAI
//
//  Mental Health Assessment Screen
//  Phase 4: Cognitive Function, Emotional Regulation, Depression Screening
//
//  ML Features enabled:
//  - cognitive_function (index 71)
//  - emotional_regulation (index 72)
//  - depression_risk (index 75)
//  - mental_wellbeing (index 74)
//

import SwiftUI
import CoreData

struct MentalHealthSurveyView: View {
    @StateObject private var viewModel: MentalHealthSurveyViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: MentalHealthSurveyViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Progress indicator
                ProgressView(value: viewModel.progress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .purple))
                    .padding()

                Text(viewModel.currentSectionTitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 8)

                // Question Content
                ScrollView {
                    VStack(spacing: 24) {
                        questionCard
                    }
                    .padding(.top, 12)
                    .padding(.bottom, 32)
                }

                // Navigation
                navigationButtons
            }
            .navigationTitle("Mental Health Check")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $viewModel.showingResults) {
                MentalHealthResultsView(
                    cognitiveScore: viewModel.cognitiveScore,
                    emotionalScore: viewModel.emotionalScore,
                    phq2Score: viewModel.phq2Score,
                    wellbeingScore: viewModel.wellbeingScore
                ) {
                    dismiss()
                }
            }
            .alert("Error", isPresented: $viewModel.showingError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage)
            }
        }
    }

    // MARK: - Question Card

    private var questionCard: some View {
        VStack(spacing: 20) {
            // Section icon
            ZStack {
                Circle()
                    .fill(viewModel.currentSectionColor.opacity(0.15))
                    .frame(width: 64, height: 64)

                Image(systemName: viewModel.currentQuestion.icon)
                    .font(.system(size: 28))
                    .foregroundColor(viewModel.currentSectionColor)
            }

            // Question text
            VStack(spacing: 8) {
                Text(viewModel.currentQuestion.text)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                if let subtitle = viewModel.currentQuestion.subtitle {
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }

            // Answer options
            if viewModel.currentQuestion.isFrequencyScale {
                frequencyOptions
            } else {
                sliderAnswer
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(20)
        .shadow(color: .black.opacity(0.08), radius: 12, x: 0, y: 4)
        .padding(.horizontal)
    }

    // MARK: - Frequency Options (PHQ-2 style)

    private var frequencyOptions: some View {
        VStack(spacing: 12) {
            ForEach(FrequencyOption.allCases) { option in
                Button {
                    viewModel.setAnswer(Double(option.rawValue))
                    if !reduceMotion {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    }
                } label: {
                    HStack {
                        Text(option.label)
                            .font(.body)
                            .fontWeight(viewModel.currentAnswer == Double(option.rawValue) ? .semibold : .regular)

                        Spacer()

                        if viewModel.currentAnswer == Double(option.rawValue) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(viewModel.currentSectionColor)
                        }
                    }
                    .padding()
                    .background(
                        viewModel.currentAnswer == Double(option.rawValue) ?
                        viewModel.currentSectionColor.opacity(0.1) :
                        Color(.systemGray6)
                    )
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Slider Answer

    private var sliderAnswer: some View {
        VStack(spacing: 16) {
            // Value display
            Text(String(format: "%.0f", viewModel.currentAnswer))
                .font(.system(size: 48, weight: .bold, design: .rounded))
                .foregroundColor(answerColor)

            Text(answerLabel)
                .font(.headline)
                .foregroundColor(answerColor)

            Slider(
                value: Binding(
                    get: { viewModel.currentAnswer },
                    set: { viewModel.setAnswer($0) }
                ),
                in: 0...10,
                step: 1
            ) { editing in
                if !editing && !reduceMotion {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                }
            }
            .tint(viewModel.currentSectionColor)
            .padding(.horizontal, 32)

            HStack {
                Text(viewModel.currentQuestion.minLabel ?? "Not at all")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text(viewModel.currentQuestion.maxLabel ?? "Extremely")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 32)
        }
    }

    // MARK: - Navigation Buttons

    private var navigationButtons: some View {
        HStack(spacing: 16) {
            if viewModel.currentIndex > 0 {
                Button {
                    withAnimation(reduceMotion ? .none : .easeInOut) {
                        viewModel.previousQuestion()
                    }
                } label: {
                    HStack {
                        Image(systemName: "chevron.left")
                        Text("Previous")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                .buttonStyle(.bordered)
            }

            Spacer()

            if viewModel.isLastQuestion {
                Button {
                    viewModel.completeSurvey()
                } label: {
                    HStack {
                        Text("Complete")
                        Image(systemName: "checkmark")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
            } else {
                Button {
                    withAnimation(reduceMotion ? .none : .easeInOut) {
                        viewModel.nextQuestion()
                    }
                } label: {
                    HStack {
                        Text("Next")
                        Image(systemName: "chevron.right")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
            }
        }
        .padding()
    }

    // MARK: - Helpers

    private var answerColor: Color {
        let value = viewModel.currentAnswer
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }

    private var answerLabel: String {
        let value = viewModel.currentAnswer
        switch value {
        case 0..<2: return "Very Good"
        case 2..<4: return "Good"
        case 4..<6: return "Moderate"
        case 6..<8: return "Difficulty"
        default: return "Significant Difficulty"
        }
    }
}

// MARK: - Frequency Option Enum (PHQ-2)

enum FrequencyOption: Int, CaseIterable, Identifiable {
    case notAtAll = 0
    case severalDays = 1
    case moreThanHalf = 2
    case nearlyEveryDay = 3

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .notAtAll: return "Not at all"
        case .severalDays: return "Several days"
        case .moreThanHalf: return "More than half the days"
        case .nearlyEveryDay: return "Nearly every day"
        }
    }
}

// MARK: - Results View

struct MentalHealthResultsView: View {
    let cognitiveScore: Double
    let emotionalScore: Double
    let phq2Score: Int
    let wellbeingScore: Double
    let onDismiss: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Success indicator
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 72))
                        .foregroundColor(.green)
                        .padding(.top, 32)

                    Text("Assessment Complete")
                        .font(.title)
                        .fontWeight(.bold)

                    // Score Cards
                    scoreCard(
                        title: "Cognitive Function",
                        value: String(format: "%.1f", cognitiveScore),
                        interpretation: cognitiveInterpretation,
                        color: cognitiveColor,
                        icon: "brain.head.profile"
                    )

                    scoreCard(
                        title: "Emotional Regulation",
                        value: String(format: "%.1f", emotionalScore),
                        interpretation: emotionalInterpretation,
                        color: emotionalColor,
                        icon: "heart.circle"
                    )

                    phq2Card

                    scoreCard(
                        title: "Mental Wellbeing",
                        value: String(format: "%.1f", wellbeingScore),
                        interpretation: wellbeingInterpretation,
                        color: wellbeingColor,
                        icon: "sparkles"
                    )

                    // Disclaimer
                    disclaimerCard

                    Spacer()
                }
            }
            .navigationTitle("Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        onDismiss()
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }

    private func scoreCard(title: String, value: String, interpretation: String, color: Color, icon: String) -> some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
                Spacer()
            }

            HStack {
                Text(value)
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(color)
                Text("/ 10")
                    .font(.title3)
                    .foregroundColor(.secondary)
                Spacer()
                Text(interpretation)
                    .font(.subheadline)
                    .foregroundColor(color)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
        .padding(.horizontal)
    }

    private var phq2Card: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "heart.text.square")
                    .foregroundColor(phq2Color)
                Text("Depression Screening (PHQ-2)")
                    .font(.headline)
                Spacer()
            }

            HStack {
                Text("\(phq2Score)")
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(phq2Color)
                Text("/ 6")
                    .font(.title3)
                    .foregroundColor(.secondary)
                Spacer()
                Text(phq2Interpretation)
                    .font(.subheadline)
                    .foregroundColor(phq2Color)
            }

            if phq2Score >= 3 {
                Text("A score of 3 or higher suggests further evaluation may be beneficial. Please discuss with your healthcare provider.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
        .padding(.horizontal)
    }

    private var disclaimerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                Text("Important Notice")
                    .font(.headline)
            }

            Text("This assessment is for self-monitoring only and is not a medical diagnosis. If you're experiencing mental health concerns, please consult a healthcare professional.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(16)
        .padding(.horizontal)
    }

    // Interpretations
    private var cognitiveInterpretation: String {
        switch cognitiveScore {
        case 0..<3: return "Good clarity"
        case 3..<5: return "Mild fog"
        case 5..<7: return "Moderate difficulty"
        default: return "Significant difficulty"
        }
    }

    private var cognitiveColor: Color {
        cognitiveScore < 3 ? .green : (cognitiveScore < 5 ? .yellow : (cognitiveScore < 7 ? .orange : .red))
    }

    private var emotionalInterpretation: String {
        switch emotionalScore {
        case 0..<3: return "Well regulated"
        case 3..<5: return "Mildly affected"
        case 5..<7: return "Moderately affected"
        default: return "Significantly affected"
        }
    }

    private var emotionalColor: Color {
        emotionalScore < 3 ? .green : (emotionalScore < 5 ? .yellow : (emotionalScore < 7 ? .orange : .red))
    }

    private var phq2Interpretation: String {
        phq2Score < 3 ? "Low risk" : "Consider further evaluation"
    }

    private var phq2Color: Color {
        phq2Score < 3 ? .green : .orange
    }

    private var wellbeingInterpretation: String {
        switch wellbeingScore {
        case 7...: return "Good wellbeing"
        case 5..<7: return "Moderate wellbeing"
        case 3..<5: return "Low wellbeing"
        default: return "Very low wellbeing"
        }
    }

    private var wellbeingColor: Color {
        wellbeingScore >= 7 ? .green : (wellbeingScore >= 5 ? .yellow : (wellbeingScore >= 3 ? .orange : .red))
    }
}

// MARK: - Preview

struct MentalHealthSurveyView_Previews: PreviewProvider {
    static var previews: some View {
        MentalHealthSurveyView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
