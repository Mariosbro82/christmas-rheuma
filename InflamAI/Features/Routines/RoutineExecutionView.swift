//
//  RoutineExecutionView.swift
//  InflamAI
//
//  Complete routine execution with step-by-step coaching, feedback, and emergency stop
//

import SwiftUI
import CoreData

struct RoutineExecutionView: View {
    @ObservedObject var routine: UserRoutine
    let exercises: [String]

    @StateObject private var viewModel: RoutineExecutionViewModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.managedObjectContext) private var viewContext

    init(routine: UserRoutine, exercises: [String]) {
        self.routine = routine
        self.exercises = exercises
        _viewModel = StateObject(wrappedValue: RoutineExecutionViewModel(routine: routine, exercises: exercises))
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if viewModel.showingEmergencyStop {
                    emergencyStopView
                } else if viewModel.showingFinalCompletion {
                    finalCompletionView
                } else if let currentExercise = viewModel.currentExercise {
                    // Exercise in progress
                    if viewModel.showingFeedback {
                        ExerciseFeedbackView(
                            exerciseName: currentExercise.name,
                            onSubmit: { feedback, notes in
                                viewModel.saveFeedback(feedback, notes: notes)
                            },
                            onSkip: {
                                viewModel.skipFeedback()
                            }
                        )
                    } else {
                        exerciseContent(currentExercise)
                    }
                } else {
                    emptyStateView
                }
            }
            .navigationTitle(routine.name ?? "Routine")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel", role: .destructive) {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        viewModel.showingEmergencyStop = true
                    }) {
                        Label("Emergency Stop", systemImage: "stop.circle.fill")
                            .foregroundColor(.red)
                    }
                }
            }
        }
    }

    // MARK: - Exercise Content

    @ViewBuilder
    private func exerciseContent(_ exercise: Exercise) -> some View {
        VStack(spacing: 0) {
            // Progress indicator
            progressHeader(exercise)

            // Main content area
            TabView(selection: $viewModel.currentTab) {
                // Tab 1: Video (if available)
                if let videoURL = exercise.videoURL {
                    ScrollView {
                        VStack(spacing: 16) {
                            Text("Tutorial Video")
                                .font(.headline)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal)

                            YouTubePlayerView(url: videoURL)
                                .frame(height: 220)
                                .padding(.horizontal)

                            Text("Swipe to start the step-by-step coach →")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .padding()
                        }
                        .padding(.vertical)
                    }
                    .tag(0)
                }

                // Tab 2: Step Coach
                if let steps = exercise.steps, !steps.isEmpty {
                    StepCoachView(steps: steps, onComplete: {
                        viewModel.showingFeedback = true
                    })
                    .tag(1)
                } else {
                    // Fallback if no steps defined
                    basicExerciseView(exercise)
                        .tag(1)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
        }
    }

    private func progressHeader(_ exercise: Exercise) -> some View {
        VStack(spacing: 12) {
            // Exercise counter
            HStack {
                Text("Exercise \(viewModel.currentExerciseIndex + 1) of \(viewModel.exerciseObjects.count)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Spacer()

                Text(exercise.name)
                    .font(.headline)
                    .lineLimit(1)
            }
            .padding(.horizontal)
            .padding(.top, 8)

            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.systemGray5))

                    Rectangle()
                        .fill(Color.green)
                        .frame(width: geometry.size.width * viewModel.progress)
                }
            }
            .frame(height: 4)
        }
        .padding(.bottom, 8)
        .background(Color(.systemGroupedBackground))
    }

    // MARK: - Basic Exercise View (Fallback)

    private func basicExerciseView(_ exercise: Exercise) -> some View {
        ScrollView {
            VStack(spacing: 24) {
                Text(exercise.name)
                    .font(.title2)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)

                // Duration Badge
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                    Text("\(exercise.duration) min")
                }
                .font(.subheadline)
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.blue)
                .cornerRadius(20)

                // Instructions
                VStack(alignment: .leading, spacing: 12) {
                    Text("Instructions")
                        .font(.headline)

                    ForEach(Array(exercise.instructions.enumerated()), id: \.offset) { index, instruction in
                        HStack(alignment: .top, spacing: 12) {
                            Text("\(index + 1)")
                                .font(.caption)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .frame(width: 24, height: 24)
                                .background(Color.blue)
                                .clipShape(Circle())

                            Text(instruction)
                                .font(.body)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .cornerRadius(16)

                Button(action: {
                    viewModel.showingFeedback = true
                }) {
                    Text("Mark as Complete")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            }
            .padding()
        }
    }

    // MARK: - Emergency Stop View

    private var emergencyStopView: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.orange)

            Text("Emergency Stop")
                .font(.title)
                .fontWeight(.bold)

            Text("Are you sure you want to stop this routine?")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Text("Your progress will be saved for completed exercises only.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()

            VStack(spacing: 12) {
                Button(action: {
                    viewModel.performEmergencyStop()
                    dismiss()
                }) {
                    Text("Yes, Stop Routine")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }

                Button(action: {
                    viewModel.showingEmergencyStop = false
                }) {
                    Text("Continue Routine")
                        .font(.subheadline)
                        .foregroundColor(.blue)
                }
            }
            .padding()
        }
        .padding()
    }

    // MARK: - Final Completion View

    private var finalCompletionView: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 80))
                .foregroundColor(.green)

            Text("Routine Complete!")
                .font(.title)
                .fontWeight(.bold)

            Text("Great work! You completed \(viewModel.completedExercises.count) of \(viewModel.exerciseObjects.count) exercises.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()

            Button(action: {
                dismiss()
            }) {
                Text("Done")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
            .padding()
        }
        .padding()
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "figure.mixed.cardio")
                .font(.system(size: 60))
                .foregroundColor(.secondary)

            Text("No exercises in this routine")
                .font(.headline)

            Button(action: { dismiss() }) {
                Text("Go Back")
                    .font(.subheadline)
                    .foregroundColor(.blue)
            }
        }
    }
}

// MARK: - View Model

@MainActor
class RoutineExecutionViewModel: ObservableObject {
    @Published var currentExerciseIndex = 0
    @Published var currentTab = 0
    @Published var showingFeedback = false
    @Published var showingEmergencyStop = false
    @Published var showingFinalCompletion = false
    @Published var completedExercises: [CompletedExercise] = []

    let routine: UserRoutine
    let exercises: [String]
    var exerciseObjects: [Exercise] = []
    private var startTime: Date = Date()

    struct CompletedExercise {
        let exercise: Exercise
        let feedback: ExerciseFeedback?
        let notes: String?
        let duration: TimeInterval
    }

    var currentExercise: Exercise? {
        guard currentExerciseIndex < exerciseObjects.count else { return nil }
        return exerciseObjects[currentExerciseIndex]
    }

    var progress: CGFloat {
        guard !exerciseObjects.isEmpty else { return 0 }
        return CGFloat(currentExerciseIndex + 1) / CGFloat(exerciseObjects.count)
    }

    init(routine: UserRoutine, exercises: [String]) {
        self.routine = routine
        self.exercises = exercises
        loadExerciseObjects()
    }

    private func loadExerciseObjects() {
        exerciseObjects = exercises.compactMap { exerciseId in
            // First try to find by UUID
            if let exercise = Exercise.allExercises.first(where: { $0.id.uuidString == exerciseId }) {
                return exercise
            }
            // Then try to find by name (demo data stores exercise names directly)
            if let exercise = Exercise.allExercises.first(where: { $0.name == exerciseId }) {
                return exercise
            }
            // Fallback: create a basic Exercise object for unknown exercises
            if !exerciseId.isEmpty {
                return Exercise(
                    id: UUID(),
                    name: exerciseId,
                    category: .stretching,
                    difficulty: .beginner,
                    duration: 5,
                    targetAreas: ["General"],
                    instructions: ["Follow the exercise instructions", "Take your time", "Listen to your body"],
                    benefits: ["Improves flexibility", "Reduces stiffness"],
                    safetyTips: ["Move slowly", "Stop if you feel pain"],
                    videoURL: nil,
                    steps: nil
                )
            }
            return nil
        }
    }

    func saveFeedback(_ feedback: ExerciseFeedback, notes: String?) {
        guard let exercise = currentExercise else { return }

        let duration = Date().timeIntervalSince(startTime)
        let completed = CompletedExercise(
            exercise: exercise,
            feedback: feedback,
            notes: notes,
            duration: duration
        )
        completedExercises.append(completed)

        // Save to Core Data
        saveExerciseCompletion(completed)

        // Move to next or finish
        moveToNextExercise()
    }

    func skipFeedback() {
        guard let exercise = currentExercise else { return }

        let duration = Date().timeIntervalSince(startTime)
        let completed = CompletedExercise(
            exercise: exercise,
            feedback: nil,
            notes: nil,
            duration: duration
        )
        completedExercises.append(completed)

        // Save to Core Data
        saveExerciseCompletion(completed)

        // Move to next or finish
        moveToNextExercise()
    }

    private func moveToNextExercise() {
        showingFeedback = false

        if currentExerciseIndex < exerciseObjects.count - 1 {
            currentExerciseIndex += 1
            currentTab = 0 // Reset to video tab
            startTime = Date() // Reset timer for next exercise
        } else {
            // Routine complete
            finalizeRoutine()
        }
    }

    func performEmergencyStop() {
        // Save all completed exercises
        saveRoutineSession(wasEmergencyStop: true)
    }

    private func finalizeRoutine() {
        saveRoutineSession(wasEmergencyStop: false)
        showingFinalCompletion = true
    }

    private func saveExerciseCompletion(_ completed: CompletedExercise) {
        guard let context = routine.managedObjectContext else { return }

        let completion = ExerciseCompletion(context: context)
        completion.id = UUID()
        completion.timestamp = Date()
        completion.exerciseName = completed.exercise.name
        completion.exerciseID = completed.exercise.id
        completion.durationSeconds = Int32(completed.duration)
        completion.feedback = completed.feedback?.rawValue
        completion.feedbackNotes = completed.notes
        completion.fromRoutine = true
        completion.routineName = routine.name
        completion.routineID = routine.id
        completion.stepsCompleted = Int16(completed.exercise.steps?.count ?? 0)
        completion.totalSteps = Int16(completed.exercise.steps?.count ?? 0)
        completion.wasCompleted = true

        // FIXED: Proper error handling instead of silent try?
        do {
            try context.save()
        } catch {
            print("❌ CRITICAL: Failed to save exercise completion: \(error)")
        }
    }

    private func saveRoutineSession(wasEmergencyStop: Bool) {
        guard let context = routine.managedObjectContext else { return }

        // Update routine stats
        routine.timesCompleted += 1
        routine.lastPerformed = Date()

        // Create session record
        let session = ExerciseSession(context: context)
        session.id = UUID()
        session.timestamp = Date()
        session.routineType = routine.name ?? "Custom Routine"
        session.durationMinutes = routine.totalDuration
        session.completedSuccessfully = !wasEmergencyStop
        session.stoppedEarly = wasEmergencyStop
        if wasEmergencyStop {
            session.stopReason = "Emergency stop by user"
        }

        // FIXED: Proper error handling instead of silent try?
        do {
            try context.save()
        } catch {
            print("❌ CRITICAL: Failed to save routine session: \(error)")
        }
    }
}

// MARK: - Preview

struct RoutineExecutionView_Previews: PreviewProvider {
    static var previews: some View {
        let context = InflamAIPersistenceController.preview.container.viewContext
        let routine = UserRoutine(context: context)
        routine.id = UUID()
        routine.name = "Morning Routine"
        routine.totalDuration = 15

        return RoutineExecutionView(routine: routine, exercises: [])
            .environment(\.managedObjectContext, context)
    }
}
