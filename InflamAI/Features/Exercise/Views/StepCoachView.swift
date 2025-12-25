//
//  StepCoachView.swift
//  InflamAI-Swift
//
//  Step-by-step exercise coach with timer and repetition tracking
//

import SwiftUI
import AVFoundation

@MainActor
class StepCoachViewModel: ObservableObject {
    @Published var currentStepIndex: Int = 0
    @Published var timeRemaining: Int = 0
    @Published var repsCompleted: Int = 0
    @Published var isTimerRunning: Bool = false
    @Published var isCompleted: Bool = false

    private var timer: Timer?
    private var audioPlayer: AVAudioPlayer?

    let steps: [ExerciseStep]

    init(steps: [ExerciseStep]) {
        self.steps = steps
        if let firstStep = steps.first, firstStep.type == .timer, let duration = firstStep.duration {
            self.timeRemaining = duration
        }
    }

    var currentStep: ExerciseStep? {
        guard currentStepIndex < steps.count else { return nil }
        return steps[currentStepIndex]
    }

    var canGoNext: Bool {
        currentStepIndex < steps.count - 1
    }

    var canGoPrevious: Bool {
        currentStepIndex > 0
    }

    func startTimer() {
        guard let step = currentStep, step.type == .timer else { return }
        isTimerRunning = true

        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            Task { @MainActor in
                if self.timeRemaining > 0 {
                    self.timeRemaining -= 1
                } else {
                    self.stopTimer()
                    self.playCompletionSound()
                }
            }
        }
    }

    func pauseTimer() {
        isTimerRunning = false
        timer?.invalidate()
    }

    func stopTimer() {
        isTimerRunning = false
        timer?.invalidate()
    }

    func resetTimer() {
        guard let step = currentStep, let duration = step.duration else { return }
        timeRemaining = duration
        isTimerRunning = false
        timer?.invalidate()
    }

    func incrementRep() {
        guard let step = currentStep, step.type == .repetitions else { return }
        repsCompleted += 1

        if let targetReps = step.repetitions, repsCompleted >= targetReps {
            playCompletionSound()
        }
    }

    func decrementRep() {
        guard repsCompleted > 0 else { return }
        repsCompleted -= 1
    }

    func goToNextStep() {
        guard canGoNext else {
            isCompleted = true
            return
        }

        stopTimer()
        currentStepIndex += 1
        repsCompleted = 0

        if let nextStep = currentStep {
            if nextStep.type == .timer, let duration = nextStep.duration {
                timeRemaining = duration
            }
        }
    }

    func goToPreviousStep() {
        guard canGoPrevious else { return }

        stopTimer()
        currentStepIndex -= 1
        repsCompleted = 0

        if let prevStep = currentStep {
            if prevStep.type == .timer, let duration = prevStep.duration {
                timeRemaining = duration
            }
        }
    }

    private func playCompletionSound() {
        // Play system sound for completion
        AudioServicesPlaySystemSound(1054) // Tink sound

        // Haptic feedback
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
    }

    deinit {
        timer?.invalidate()
    }
}

struct StepCoachView: View {
    @StateObject private var viewModel: StepCoachViewModel
    let onComplete: () -> Void

    init(steps: [ExerciseStep], onComplete: @escaping () -> Void) {
        _viewModel = StateObject(wrappedValue: StepCoachViewModel(steps: steps))
        self.onComplete = onComplete
    }

    var body: some View {
        VStack(spacing: 0) {
            if viewModel.isCompleted {
                completedView
            } else {
                // Progress indicator
                progressBar

                // Main content
                ScrollView {
                    VStack(spacing: 24) {
                        if let step = viewModel.currentStep {
                            stepContent(step)
                        }
                    }
                    .padding()
                }

                // Navigation buttons
                navigationButtons
            }
        }
    }

    private var progressBar: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Step \(viewModel.currentStepIndex + 1) of \(viewModel.steps.count)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Spacer()

                if let step = viewModel.currentStep {
                    Text(stepTypeLabel(step.type))
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(stepTypeColor(step.type).opacity(0.2))
                        .foregroundColor(stepTypeColor(step.type))
                        .cornerRadius(8)
                }
            }
            .padding(.horizontal)
            .padding(.top, 8)

            ProgressView(value: Double(viewModel.currentStepIndex), total: Double(viewModel.steps.count - 1))
                .padding(.horizontal)
        }
        .padding(.bottom, 8)
        .background(Color(.systemGroupedBackground))
    }

    private func stepContent(_ step: ExerciseStep) -> some View {
        VStack(spacing: 20) {
            // Image hint
            if let imageHint = step.imageHint {
                Image(systemName: imageHint)
                    .font(.system(size: 60))
                    .foregroundColor(.blue)
                    .accessibilityHidden(true)
            }

            // Instruction
            Text(step.instruction)
                .font(.title2)
                .fontWeight(.semibold)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)

            // Step type specific content
            switch step.type {
            case .timer:
                timerView(step)
            case .repetitions:
                repetitionsView(step)
            case .info:
                EmptyView()
            }
        }
    }

    private func timerView(_ step: ExerciseStep) -> some View {
        VStack(spacing: 16) {
            // Timer display
            Text(formatTime(viewModel.timeRemaining))
                .font(.system(size: 72, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundColor(viewModel.timeRemaining <= 5 && viewModel.timeRemaining > 0 ? .orange : .primary)

            // Timer controls
            HStack(spacing: 20) {
                if !viewModel.isTimerRunning {
                    Button(action: viewModel.startTimer) {
                        Label("Start", systemImage: "play.fill")
                            .font(.headline)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                } else {
                    Button(action: viewModel.pauseTimer) {
                        Label("Pause", systemImage: "pause.fill")
                            .font(.headline)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                }

                Button(action: viewModel.resetTimer) {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                        .font(.headline)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                        .background(Color.gray.opacity(0.2))
                        .foregroundColor(.primary)
                        .cornerRadius(12)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func repetitionsView(_ step: ExerciseStep) -> some View {
        VStack(spacing: 16) {
            // Rep counter display
            HStack(spacing: 8) {
                Text("\(viewModel.repsCompleted)")
                    .font(.system(size: 72, weight: .bold, design: .rounded))
                    .monospacedDigit()

                if let targetReps = step.repetitions {
                    Text("/ \(targetReps)")
                        .font(.system(size: 36, weight: .semibold, design: .rounded))
                        .foregroundColor(.secondary)
                }
            }

            if let targetReps = step.repetitions {
                ProgressView(value: Double(viewModel.repsCompleted), total: Double(targetReps))
                    .tint(viewModel.repsCompleted >= targetReps ? .green : .blue)
            }

            // Rep controls
            HStack(spacing: 20) {
                Button(action: viewModel.decrementRep) {
                    Image(systemName: "minus.circle.fill")
                        .font(.system(size: 44))
                        .foregroundColor(.red)
                }
                .disabled(viewModel.repsCompleted == 0)

                Button(action: viewModel.incrementRep) {
                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 44))
                        .foregroundColor(.blue)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private var navigationButtons: some View {
        HStack(spacing: 16) {
            // Previous button
            Button(action: viewModel.goToPreviousStep) {
                Label("Previous", systemImage: "chevron.left")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.canGoPrevious ? Color.blue.opacity(0.1) : Color.gray.opacity(0.1))
                    .foregroundColor(viewModel.canGoPrevious ? .blue : .gray)
                    .cornerRadius(12)
            }
            .disabled(!viewModel.canGoPrevious)

            // Next/Finish button
            Button(action: {
                if viewModel.canGoNext {
                    viewModel.goToNextStep()
                } else {
                    viewModel.isCompleted = true
                }
            }) {
                Label(viewModel.canGoNext ? "Next" : "Finish", systemImage: viewModel.canGoNext ? "chevron.right" : "checkmark")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
        }
        .padding()
        .background(Color(.systemGroupedBackground))
    }

    private var completedView: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 80))
                .foregroundColor(.green)

            Text("Exercise Complete!")
                .font(.title)
                .fontWeight(.bold)

            Text("Great work! How did it feel?")
                .font(.body)
                .foregroundColor(.secondary)

            Spacer()

            Button(action: onComplete) {
                Text("Continue")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
            .padding()
        }
    }

    // MARK: - Helpers

    private func formatTime(_ seconds: Int) -> String {
        let minutes = seconds / 60
        let remainingSeconds = seconds % 60
        if minutes > 0 {
            return String(format: "%d:%02d", minutes, remainingSeconds)
        } else {
            return String(format: "%d", remainingSeconds)
        }
    }

    private func stepTypeLabel(_ type: StepType) -> String {
        switch type {
        case .info: return "Info"
        case .timer: return "Timer"
        case .repetitions: return "Reps"
        }
    }

    private func stepTypeColor(_ type: StepType) -> Color {
        switch type {
        case .info: return .gray
        case .timer: return .blue
        case .repetitions: return .green
        }
    }
}
