//
//  DailyCheckInView.swift
//  InflamAI
//
//  Production-grade BASDAI daily check-in flow
//  6 medical questions with real-time calculation
//

import SwiftUI
import CoreData

struct DailyCheckInView: View {
    @StateObject private var viewModel: DailyCheckInViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: DailyCheckInViewModel(context: context))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Progress Header - Fixed at top
                progressHeader

                // Question Content - Scrollable
                ScrollView {
                    VStack(spacing: 32) {
                        questionContent
                    }
                    .padding(.vertical, 24)
                }
                .scrollIndicators(.hidden)

                // Navigation Buttons - Fixed at bottom
                navigationButtons
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Daily Check-In")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .foregroundColor(.secondary)
                }
            }
            .sheet(isPresented: $viewModel.showingResults) {
                CheckInResultsView(
                    basdaiScore: viewModel.basdaiScore,
                    asdasScore: viewModel.asdasScore
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

    // MARK: - Progress Header

    private var progressHeader: some View {
        VStack(spacing: 12) {
            // Progress Bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 8)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            LinearGradient(
                                colors: viewModel.isBASDAIQuestion ? [.blue, .cyan] : [.green, .mint],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * viewModel.progress, height: 8)
                        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: viewModel.progress)
                }
            }
            .frame(height: 8)
            .accessibilityLabel("Question progress")
            .accessibilityValue("\(viewModel.currentIndex + 1) of \(DailyCheckInViewModel.totalQuestions)")

            // Section indicator
            HStack(spacing: 12) {
                // Section badge
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.isBASDAIQuestion ? Color.blue : Color.green)
                        .frame(width: 8, height: 8)

                    Text(viewModel.isBASDAIQuestion ? "BASDAI" : "Wellness")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(viewModel.isBASDAIQuestion ? .blue : .green)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill(viewModel.isBASDAIQuestion ? Color.blue.opacity(0.12) : Color.green.opacity(0.12))
                )

                Spacer()

                Text("\(viewModel.currentIndex + 1) / \(DailyCheckInViewModel.totalQuestions)")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 8)
        .background(Color(.systemBackground))
    }

    // MARK: - Question Content

    @ViewBuilder
    private var questionContent: some View {
        if viewModel.isBASDAIQuestion {
            // BASDAI Questions (1-6)
            VStack(spacing: 20) {
                // Question text
                VStack(spacing: 8) {
                    Text(viewModel.currentQuestion.text)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)
                        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                        .fixedSize(horizontal: false, vertical: true)

                    if !viewModel.currentQuestion.subtitle.isEmpty {
                        Text(viewModel.currentQuestion.subtitle)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                }

                // Answer Input
                if viewModel.currentQuestion.isDuration {
                    DurationAnswerView(
                        value: $viewModel.answers[viewModel.currentIndex],
                        question: viewModel.currentQuestion
                    )
                } else {
                    SliderAnswerView(
                        value: $viewModel.answers[viewModel.currentIndex],
                        question: viewModel.currentQuestion
                    )
                }
            }
            .id(viewModel.currentIndex) // Force view refresh on question change
            .transition(.asymmetric(
                insertion: .move(edge: .trailing).combined(with: .opacity),
                removal: .move(edge: .leading).combined(with: .opacity)
            ))
        } else if let mlQuestion = viewModel.currentMLQuestion {
            // ML Questions (7-12)
            VStack(spacing: 20) {
                // Question text
                VStack(spacing: 8) {
                    Text(mlQuestion.text)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)
                        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                        .fixedSize(horizontal: false, vertical: true)

                    if !mlQuestion.subtitle.isEmpty {
                        Text(mlQuestion.subtitle)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                }

                // ML Slider Answer
                MLSliderAnswerView(
                    value: $viewModel.answers[viewModel.currentIndex],
                    question: mlQuestion
                )
            }
            .id(viewModel.currentIndex)
            .transition(.asymmetric(
                insertion: .move(edge: .trailing).combined(with: .opacity),
                removal: .move(edge: .leading).combined(with: .opacity)
            ))
        }
    }

    // MARK: - Navigation Buttons

    private var navigationButtons: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(spacing: 12) {
                // Previous button
                if viewModel.currentIndex > 0 {
                    Button {
                        withAnimation(reduceMotion ? .none : .spring(response: 0.35, dampingFraction: 0.8)) {
                            viewModel.previousQuestion()
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 14, weight: .semibold))
                            Text("Previous")
                                .fontWeight(.medium)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .foregroundColor(.primary)
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                    .accessibilityHint("Go to previous question")
                }

                // Next/Complete button
                PrimaryButton(title: viewModel.isLastQuestion ? "Complete" : "Next") {
                    if viewModel.isLastQuestion {
                        viewModel.completeCheckIn()
                    } else {
                        withAnimation(reduceMotion ? .none : .spring(response: 0.35, dampingFraction: 0.8)) {
                            viewModel.nextQuestion()
                        }
                    }
                }
                .accessibilityLabel(viewModel.isLastQuestion ? "Complete check-in" : "Go to next question")
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(Color(.systemBackground))
        }
    }
}

// MARK: - Slider Answer View

struct SliderAnswerView: View {
    @Binding var value: Double
    let question: BASDAIQuestion

    var body: some View {
        VStack(spacing: 24) {
            // Value Display Card
            VStack(spacing: 12) {
                // Emoji Indicator
                Text(emoji(for: value))
                    .font(.system(size: 56))
                    .scaleEffect(1.0 + (value / 100))
                    .animation(.spring(response: 0.3, dampingFraction: 0.6), value: value)

                // Value Display
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(String(format: "%.1f", value))
                        .font(.system(size: 64, weight: .bold, design: .rounded))
                        .foregroundColor(painColor(value))
                        .contentTransition(.numericText())
                        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: value)

                    Text("/ 10")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(.systemBackground))
                    .shadow(color: painColor(value).opacity(0.2), radius: 20, x: 0, y: 8)
            )
            .padding(.horizontal, 24)

            // Premium Slider
            VStack(spacing: 12) {
                // Custom slider track
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Background track
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color(.systemGray5))
                            .frame(height: 12)

                        // Filled track with gradient
                        RoundedRectangle(cornerRadius: 6)
                            .fill(
                                LinearGradient(
                                    colors: gradientColors(for: value),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: geometry.size.width * (value / 10), height: 12)
                            .animation(.spring(response: 0.3, dampingFraction: 0.8), value: value)
                    }
                }
                .frame(height: 12)
                .padding(.horizontal, 24)

                // Native slider (invisible but functional)
                Slider(value: $value, in: 0...10, step: 0.5) { editing in
                    if !editing {
                        if [0, 5, 10].contains(value) {
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        } else {
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        }
                    }
                }
                .tint(painColor(value))
                .padding(.horizontal, 24)
                .accessibilityLabel(question.text)
                .accessibilityValue("\(value, specifier: "%.1f") out of 10")

                // Labels
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("0")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundColor(.secondary)
                        Text(question.minLabel)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("10")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundColor(.secondary)
                        Text(question.maxLabel)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal, 24)
            }
        }
    }

    private func painColor(_ value: Double) -> Color {
        switch value {
        case 0..<3: return .green
        case 3..<5: return .yellow
        case 5..<7: return .orange
        default: return .red
        }
    }

    private func gradientColors(for value: Double) -> [Color] {
        switch value {
        case 0..<3: return [.green, .mint]
        case 3..<5: return [.yellow, .orange]
        case 5..<7: return [.orange, .red]
        default: return [.red, .pink]
        }
    }

    private func emoji(for value: Double) -> String {
        switch value {
        case 0..<2: return "😊"
        case 2..<4: return "🙂"
        case 4..<6: return "😐"
        case 6..<8: return "😟"
        case 8..<9: return "😣"
        default: return "😖"
        }
    }
}

// MARK: - Duration Answer View

struct DurationAnswerView: View {
    @Binding var value: Double
    let question: BASDAIQuestion

    private let durations: [(value: Double, label: String, icon: String)] = [
        (0, "None", "checkmark.circle"),
        (7.5, "<15 min", "clock"),
        (22.5, "15-30 min", "clock.fill"),
        (45, "30-60 min", "alarm"),
        (90, "1-2 hours", "alarm.fill"),
        (120, "2+ hours", "exclamationmark.circle")
    ]

    var body: some View {
        VStack(spacing: 24) {
            // Current Selection Card
            VStack(spacing: 8) {
                Image(systemName: selectedIcon)
                    .font(.system(size: 40))
                    .foregroundColor(durationColor)
                    .symbolEffect(.bounce, value: value)

                Text(selectedDurationLabel)
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(durationColor)

                if value > 0 {
                    Text("\(Int(value)) minutes of stiffness")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(.systemBackground))
                    .shadow(color: durationColor.opacity(0.2), radius: 20, x: 0, y: 8)
            )
            .padding(.horizontal, 24)

            // Duration Buttons - 2 column grid
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                ForEach(durations, id: \.value) { duration in
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            value = duration.value
                        }
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    } label: {
                        VStack(spacing: 8) {
                            Image(systemName: duration.icon)
                                .font(.system(size: 24))
                                .foregroundColor(value == duration.value ? .white : colorForDuration(duration.value))

                            Text(duration.label)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(value == duration.value ? .white : .primary)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(minHeight: 64)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(value == duration.value ?
                                    LinearGradient(
                                        colors: gradientForDuration(duration.value),
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ) :
                                    LinearGradient(
                                        colors: [Color(.systemGray6), Color(.systemGray6)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(value == duration.value ? Color.clear : Color(.systemGray4), lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(duration.label)
                    .accessibilityAddTraits(value == duration.value ? [.isSelected] : [])
                }
            }
            .padding(.horizontal, 24)
        }
    }

    private var selectedDurationLabel: String {
        durations.first(where: { $0.value == value })?.label ?? "Select"
    }

    private var selectedIcon: String {
        durations.first(where: { $0.value == value })?.icon ?? "clock"
    }

    private var durationColor: Color {
        colorForDuration(value)
    }

    private func colorForDuration(_ value: Double) -> Color {
        switch value {
        case 0: return .green
        case 7.5: return .mint
        case 22.5: return .yellow
        case 45: return .orange
        case 90: return .red
        default: return .red
        }
    }

    private func gradientForDuration(_ value: Double) -> [Color] {
        switch value {
        case 0: return [.green, .mint]
        case 7.5: return [.cyan, .blue]
        case 22.5: return [.yellow, .orange]
        case 45: return [.orange, .red]
        case 90: return [.red, .pink]
        default: return [.red, .purple]
        }
    }
}

// MARK: - ML Slider Answer View

/// Slider for ML/Wellness questions (uses MLQuestion instead of BASDAIQuestion)
struct MLSliderAnswerView: View {
    @Binding var value: Double
    let question: MLQuestion

    private var isPositiveQuestion: Bool {
        question.number == 7 || question.number == 8
    }

    var body: some View {
        VStack(spacing: 24) {
            // Value Display Card
            VStack(spacing: 12) {
                // Emoji Indicator
                Text(emoji(for: value))
                    .font(.system(size: 56))
                    .scaleEffect(1.0 + (value / 100))
                    .animation(.spring(response: 0.3, dampingFraction: 0.6), value: value)

                // Value Display
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(String(format: "%.1f", value))
                        .font(.system(size: 64, weight: .bold, design: .rounded))
                        .foregroundColor(colorForValue(value))
                        .contentTransition(.numericText())
                        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: value)

                    Text("/ 10")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(.systemBackground))
                    .shadow(color: colorForValue(value).opacity(0.2), radius: 20, x: 0, y: 8)
            )
            .padding(.horizontal, 24)

            // Premium Slider
            VStack(spacing: 12) {
                // Custom slider track
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Background track
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color(.systemGray5))
                            .frame(height: 12)

                        // Filled track with gradient
                        RoundedRectangle(cornerRadius: 6)
                            .fill(
                                LinearGradient(
                                    colors: gradientColors(for: value),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: geometry.size.width * (value / 10), height: 12)
                            .animation(.spring(response: 0.3, dampingFraction: 0.8), value: value)
                    }
                }
                .frame(height: 12)
                .padding(.horizontal, 24)

                // Native slider (functional)
                Slider(value: $value, in: 0...10, step: 0.5) { editing in
                    if !editing {
                        if [0, 5, 10].contains(value) {
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        } else {
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        }
                    }
                }
                .tint(colorForValue(value))
                .padding(.horizontal, 24)
                .accessibilityLabel(question.text)
                .accessibilityValue("\(value, specifier: "%.1f") out of 10")

                // Labels
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("0")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundColor(.secondary)
                        Text(question.minLabel)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("10")
                            .font(.system(size: 14, weight: .semibold, design: .rounded))
                            .foregroundColor(.secondary)
                        Text(question.maxLabel)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.horizontal, 24)
            }
        }
    }

    /// Color based on question type and value
    private func colorForValue(_ value: Double) -> Color {
        if isPositiveQuestion {
            // High is good
            switch value {
            case 0..<3: return .red
            case 3..<5: return .orange
            case 5..<7: return .yellow
            default: return .green
            }
        } else {
            // Low is good
            switch value {
            case 0..<3: return .green
            case 3..<5: return .yellow
            case 5..<7: return .orange
            default: return .red
            }
        }
    }

    private func gradientColors(for value: Double) -> [Color] {
        if isPositiveQuestion {
            switch value {
            case 0..<3: return [.red, .pink]
            case 3..<5: return [.orange, .red]
            case 5..<7: return [.yellow, .orange]
            default: return [.green, .mint]
            }
        } else {
            switch value {
            case 0..<3: return [.green, .mint]
            case 3..<5: return [.yellow, .orange]
            case 5..<7: return [.orange, .red]
            default: return [.red, .pink]
            }
        }
    }

    private func emoji(for value: Double) -> String {
        if isPositiveQuestion {
            switch value {
            case 0..<2: return "😫"
            case 2..<4: return "😔"
            case 4..<6: return "😐"
            case 6..<8: return "🙂"
            case 8..<9: return "😊"
            default: return "🤩"
            }
        } else {
            switch value {
            case 0..<2: return "😌"
            case 2..<4: return "🙂"
            case 4..<6: return "😐"
            case 6..<8: return "😟"
            case 8..<9: return "😣"
            default: return "😰"
            }
        }
    }
}

// MARK: - Results View

struct CheckInResultsView: View {
    let basdaiScore: Double
    let asdasScore: Double?
    let onDismiss: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showContent = false

    private var interpretation: BASDAIInterpretation {
        BASDAICalculator.interpretation(score: basdaiScore)
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 28) {
                    // Success Animation
                    ZStack {
                        Circle()
                            .fill(
                                RadialGradient(
                                    colors: [interpretation.color.opacity(0.3), interpretation.color.opacity(0.05)],
                                    center: .center,
                                    startRadius: 0,
                                    endRadius: 80
                                )
                            )
                            .frame(width: 160, height: 160)
                            .scaleEffect(showContent ? 1 : 0.5)
                            .opacity(showContent ? 1 : 0)

                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 80))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.green, .mint],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .scaleEffect(showContent ? 1 : 0.3)
                            .opacity(showContent ? 1 : 0)
                    }
                    .padding(.top, 24)

                    VStack(spacing: 8) {
                        Text("Check-In Complete")
                            .font(.title)
                            .fontWeight(.bold)

                        Text("Great job tracking your health today!")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .opacity(showContent ? 1 : 0)
                    .offset(y: showContent ? 0 : 20)

                    // BASDAI Score Card - Premium
                    VStack(spacing: 20) {
                        HStack {
                            Text("BASDAI Score")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Bath Ankylosing Spondylitis Disease Activity Index")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }

                        // Score Display
                        HStack(alignment: .firstTextBaseline, spacing: 4) {
                            Text(String(format: "%.1f", basdaiScore))
                                .font(.system(size: 80, weight: .bold, design: .rounded))
                                .foregroundColor(interpretation.color)

                            Text("/ 10")
                                .font(.system(size: 28, weight: .medium, design: .rounded))
                                .foregroundColor(.secondary)
                        }

                        // Category Badge
                        Text(interpretation.category)
                            .font(.headline)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 10)
                            .background(
                                Capsule()
                                    .fill(
                                        LinearGradient(
                                            colors: gradientForScore(basdaiScore),
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                            )

                        // Advice
                        Text(interpretation.advice)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 16)
                    }
                    .padding(24)
                    .background(
                        RoundedRectangle(cornerRadius: 24)
                            .fill(Color(.systemBackground))
                            .shadow(color: interpretation.color.opacity(0.15), radius: 20, x: 0, y: 10)
                    )
                    .padding(.horizontal, 20)
                    .opacity(showContent ? 1 : 0)
                    .offset(y: showContent ? 0 : 30)

                    // ASDAS Score (if available)
                    if let asdasScore = asdasScore {
                        let asdasInterpretation = ASDACalculator.interpretation(score: asdasScore)

                        VStack(spacing: 16) {
                            HStack {
                                Text("ASDAS-CRP Score")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                    .foregroundColor(.secondary)
                                Spacer()
                            }

                            HStack(alignment: .firstTextBaseline, spacing: 4) {
                                Text(String(format: "%.2f", asdasScore))
                                    .font(.system(size: 48, weight: .bold, design: .rounded))
                                    .foregroundColor(asdasInterpretation.color)
                            }

                            Text(asdasInterpretation.category)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(asdasInterpretation.color)
                        }
                        .padding(20)
                        .background(
                            RoundedRectangle(cornerRadius: 20)
                                .fill(Color(.systemBackground))
                                .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 4)
                        )
                        .padding(.horizontal, 20)
                        .opacity(showContent ? 1 : 0)
                        .offset(y: showContent ? 0 : 30)
                    }

                    // Recommendations Section
                    VStack(alignment: .leading, spacing: 16) {
                        Text("What's Next?")
                            .font(.headline)
                            .padding(.horizontal, 4)

                        VStack(spacing: 12) {
                            if basdaiScore >= 6 {
                                CheckInRecommendationRow(
                                    icon: "phone.fill",
                                    text: "Contact your rheumatologist",
                                    subtitle: "High activity detected",
                                    color: .red
                                )
                            }

                            CheckInRecommendationRow(
                                icon: "chart.line.uptrend.xyaxis",
                                text: "View your trends",
                                subtitle: "Track your progress over time",
                                color: .blue
                            )

                            CheckInRecommendationRow(
                                icon: "figure.walk",
                                text: "Try a mobility routine",
                                subtitle: "Movement can help reduce stiffness",
                                color: .green
                            )
                        }
                    }
                    .padding(20)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color(.systemBackground))
                            .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 4)
                    )
                    .padding(.horizontal, 20)
                    .opacity(showContent ? 1 : 0)
                    .offset(y: showContent ? 0 : 30)

                    Spacer(minLength: 40)
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        onDismiss()
                        dismiss()
                    } label: {
                        Text("Done")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(
                                Capsule()
                                    .fill(
                                        LinearGradient(
                                            colors: [.blue, .cyan],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                            )
                    }
                }
            }
            .onAppear {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.8).delay(0.1)) {
                    showContent = true
                }
            }
        }
    }

    private func gradientForScore(_ score: Double) -> [Color] {
        switch score {
        case 0..<2: return [.green, .mint]
        case 2..<4: return [.yellow, .orange]
        case 4..<6: return [.orange, .red]
        default: return [.red, .pink]
        }
    }
}

struct CheckInRecommendationRow: View {
    let icon: String
    let text: String
    var subtitle: String = ""
    let color: Color

    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .font(.system(size: 18))
                    .foregroundColor(color)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(text)
                    .font(.body)
                    .fontWeight(.medium)

                if !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Preview

struct DailyCheckInView_Previews: PreviewProvider {
    static var previews: some View {
        DailyCheckInView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
