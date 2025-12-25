//
//  BASFIQuestionnaireView.swift
//  InflamAI
//
//  Bath Ankylosing Spondylitis Functional Index (BASFI)
//  10-question validated questionnaire measuring functional ability
//
//  ML Feature: basfi (index 8)
//  Reference: Calin A, et al. J Rheumatol. 1994;21(12):2281-5
//

import SwiftUI
import CoreData

struct BASFIQuestionnaireView: View {
    @StateObject private var viewModel: BASFIQuestionnaireViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: BASFIQuestionnaireViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Progress Bar
                ProgressView(value: viewModel.progress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .accentColor))
                    .padding()

                Text("Question \(viewModel.currentIndex + 1) of 10")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 8)

                // Question Content
                ScrollView {
                    VStack(spacing: 24) {
                        // Question Card
                        questionCard
                    }
                    .padding(.top, 16)
                    .padding(.bottom, 32)
                }

                // Navigation Buttons
                navigationButtons
            }
            .navigationTitle("BASFI Assessment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $viewModel.showingResults) {
                BASFIResultsView(score: viewModel.basfiScore) {
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
            // Question icon & number
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.1))
                    .frame(width: 64, height: 64)

                Text(viewModel.currentQuestion.icon)
                    .font(.system(size: 32))
            }

            // Question text
            VStack(spacing: 8) {
                Text(viewModel.currentQuestion.text)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                Text(viewModel.currentQuestion.subtitle)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            // Value Display
            VStack(spacing: 8) {
                Text(String(format: "%.0f", viewModel.answers[viewModel.currentIndex]))
                    .font(.system(size: 56, weight: .bold, design: .rounded))
                    .foregroundColor(difficultyColor(viewModel.answers[viewModel.currentIndex]))

                Text(difficultyLabel(viewModel.answers[viewModel.currentIndex]))
                    .font(.headline)
                    .foregroundColor(difficultyColor(viewModel.answers[viewModel.currentIndex]))
            }
            .padding(.vertical, 8)

            // Slider
            Slider(
                value: $viewModel.answers[viewModel.currentIndex],
                in: 0...10,
                step: 1
            ) { editing in
                if !editing && !reduceMotion {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                }
            }
            .tint(difficultyColor(viewModel.answers[viewModel.currentIndex]))
            .padding(.horizontal, 20)

            // Labels
            HStack {
                VStack(alignment: .leading) {
                    Text("0")
                        .font(.caption)
                        .fontWeight(.bold)
                    Text("Easy")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("10")
                        .font(.caption)
                        .fontWeight(.bold)
                    Text("Impossible")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 20)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(20)
        .shadow(color: .black.opacity(0.08), radius: 12, x: 0, y: 4)
        .padding(.horizontal)
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
                    viewModel.completeQuestionnaire()
                } label: {
                    HStack {
                        Text("Complete")
                        Image(systemName: "checkmark")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
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
                .tint(Colors.Primary.p500)
            }
        }
        .padding()
    }

    // MARK: - Helpers

    private func difficultyColor(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        case 7..<9: return .red
        default: return .red
        }
    }

    private func difficultyLabel(_ value: Double) -> String {
        switch value {
        case 0..<2: return "Very Easy"
        case 2..<4: return "Easy"
        case 4..<6: return "Moderate"
        case 6..<8: return "Difficult"
        case 8..<10: return "Very Difficult"
        default: return "Impossible"
        }
    }
}

// MARK: - BASFI Results View

struct BASFIResultsView: View {
    let score: Double
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

                    // Score Card
                    VStack(spacing: 16) {
                        Text("BASFI Score")
                            .font(.headline)
                            .foregroundColor(.secondary)

                        Text(String(format: "%.1f", score))
                            .font(.system(size: 72, weight: .bold, design: .rounded))
                            .foregroundColor(interpretationColor)

                        Text(interpretation)
                            .font(.title2)
                            .fontWeight(.semibold)
                            .foregroundColor(interpretationColor)

                        Text(advice)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(16)
                    .padding(.horizontal)

                    // Scale explanation
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Understanding BASFI")
                            .font(.headline)

                        Text("BASFI measures your functional ability to perform daily activities. The score ranges from 0 (no limitation) to 10 (severe limitation).")
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        Divider()

                        scaleRow(range: "0-2", label: "Minimal limitation", color: .green)
                        scaleRow(range: "2-4", label: "Mild limitation", color: .yellow)
                        scaleRow(range: "4-6", label: "Moderate limitation", color: .orange)
                        scaleRow(range: "6-8", label: "Significant limitation", color: .red)
                        scaleRow(range: "8-10", label: "Severe limitation", color: .red)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(16)
                    .padding(.horizontal)

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

    private var interpretation: String {
        switch score {
        case 0..<2: return "Minimal Limitation"
        case 2..<4: return "Mild Limitation"
        case 4..<6: return "Moderate Limitation"
        case 6..<8: return "Significant Limitation"
        default: return "Severe Limitation"
        }
    }

    private var interpretationColor: Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return .yellow
        case 4..<6: return .orange
        default: return .red
        }
    }

    private var advice: String {
        switch score {
        case 0..<4: return "Good functional capacity. Continue with regular exercise and stretching."
        case 4..<6: return "Moderate functional impact. Discuss mobility options with your rheumatologist."
        default: return "Significant functional impact. Please discuss treatment options with your healthcare provider."
        }
    }

    private func scaleRow(range: String, label: String, color: Color) -> some View {
        HStack {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(range)
                .font(.caption)
                .fontWeight(.medium)
                .frame(width: 40, alignment: .leading)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
        }
    }
}

// MARK: - Preview

struct BASFIQuestionnaireView_Previews: PreviewProvider {
    static var previews: some View {
        BASFIQuestionnaireView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
