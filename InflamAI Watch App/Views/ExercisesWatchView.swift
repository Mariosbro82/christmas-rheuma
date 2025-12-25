//
//  ExercisesWatchView.swift
//  InflamAI-Swift Watch App
//
//  Created by Claude Code on 2025-11-11.
//  Beautiful exercise tracking for Apple Watch
//

import SwiftUI

// MARK: - ExerciseCategory
enum ExerciseCategory: String, Codable, CaseIterable {
    case stretching = "Stretching"
    case mobility = "Mobility"
    case strengthening = "Strengthening"
    case balance = "Balance"
    case breathing = "Breathing"
    case warmup = "Warm Up"
    case cooldown = "Cool Down"
}

struct ExercisesWatchView: View {
    @EnvironmentObject var connectivityManager: WatchConnectivityManager
    @State private var selectedExercise: WatchExercise?
    @State private var showingExerciseDetail = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) {
                    // Header
                    VStack(spacing: 4) {
                        Image(systemName: "figure.walk")
                            .font(.system(size: 32))
                            .foregroundColor(.green)

                        Text("Exercises")
                            .font(.headline)
                            .fontWeight(.bold)
                    }
                    .padding(.top, 8)

                    // Quick Start Button
                    Button(action: {
                        // Start quick routine
                    }) {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Quick Routine")
                            Spacer()
                            Text("5 min")
                                .font(.caption)
                        }
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(12)
                        .background(
                            LinearGradient(
                                colors: [.green, .mint],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .cornerRadius(12)
                    }
                    .buttonStyle(.plain)

                    // Exercise Categories
                    Text("Categories")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 4)
                        .padding(.top, 8)

                    ForEach(ExerciseCategory.watchCategories, id: \.title) { category in
                        NavigationLink(destination: ExerciseCategoryDetailView(category: category)) {
                            ExerciseCategoryCard(category: category)
                        }
                        .buttonStyle(.plain)
                    }

                    // Recent Exercises
                    Text("Recent Activity")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 4)
                        .padding(.top, 8)

                    RecentExercisesList()
                }
                .padding(.horizontal, 4)
                .padding(.bottom, 8)
            }
        }
    }
}

// MARK: - Exercise Category Card

struct ExerciseCategoryCard: View {
    let category: ExerciseCategoryInfo

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(category.color.opacity(0.2))
                    .frame(width: 45, height: 45)

                Image(systemName: category.icon)
                    .font(.system(size: 20))
                    .foregroundColor(category.color)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(category.title)
                    .font(.system(size: 14, weight: .semibold))

                Text("\(category.exerciseCount) exercises")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(white: 0.15))
        )
    }
}

// MARK: - Recent Exercises List

struct RecentExercisesList: View {
    @State private var recentExercises: [CompletedExercise] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @EnvironmentObject var connectivityManager: WatchConnectivityManager

    var body: some View {
        VStack(spacing: 8) {
            if isLoading {
                ProgressView()
                    .padding(.vertical, 20)
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(white: 0.15))
                    )
            } else if let error = errorMessage {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 40))
                        .foregroundColor(.orange)

                    Text(error)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)

                    Button("Retry") {
                        Task { await loadExerciseHistory() }
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                }
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(white: 0.15))
                )
            } else if recentExercises.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "figure.walk.circle")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)

                    Text("No recent exercises")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(white: 0.15))
                )
            } else {
                ForEach(recentExercises) { exercise in
                    RecentExerciseRow(exercise: exercise)
                }
            }
        }
        .task {
            await loadExerciseHistory()
        }
    }

    @MainActor
    private func loadExerciseHistory() async {
        isLoading = true
        errorMessage = nil

        let message: [String: Any] = [
            "type": "request_exercise_history",
            "timestamp": Date().timeIntervalSince1970
        ]

        do {
            if let response = await connectivityManager.sendMessage(message) {
                if let success = response["success"] as? Bool, success {
                    if let historyData = response["exercises"] as? [[String: Any]] {
                        recentExercises = parseExerciseHistory(historyData)
                    } else {
                        recentExercises = []
                    }
                } else if let error = response["error"] as? String {
                    errorMessage = error
                } else {
                    errorMessage = "Failed to load exercises"
                }
            } else {
                errorMessage = "iPhone not reachable"
            }
        } catch {
            errorMessage = "Connection error"
        }

        isLoading = false
    }

    private func parseExerciseHistory(_ historyData: [[String: Any]]) -> [CompletedExercise] {
        return historyData.compactMap { data in
            guard let name = data["name"] as? String,
                  let duration = data["duration"] as? Int,
                  let timestamp = data["timestamp"] as? TimeInterval else {
                return nil
            }
            return CompletedExercise(
                name: name,
                duration: duration,
                date: Date(timeIntervalSince1970: timestamp)
            )
        }
    }
}

struct RecentExerciseRow: View {
    let exercise: CompletedExercise

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
                .font(.system(size: 18))

            VStack(alignment: .leading, spacing: 2) {
                Text(exercise.name)
                    .font(.system(size: 13, weight: .medium))

                Text("\(exercise.duration) min")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(exercise.date, style: .relative)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(white: 0.15))
        )
    }
}

// MARK: - Exercise Category Detail View

struct ExerciseCategoryDetailView: View {
    let category: ExerciseCategoryInfo
    @State private var exercises: [WatchExercise] = []

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                // Category Header
                VStack(spacing: 8) {
                    ZStack {
                        Circle()
                            .fill(category.color.opacity(0.2))
                            .frame(width: 60, height: 60)

                        Image(systemName: category.icon)
                            .font(.system(size: 28))
                            .foregroundColor(category.color)
                    }

                    Text(category.title)
                        .font(.headline)

                    Text(category.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 8)

                // Exercise List
                ForEach(category.sampleExercises, id: \.name) { exercise in
                    NavigationLink(destination: ExerciseDetailWatchView(exercise: exercise, color: category.color)) {
                        ExerciseListCard(exercise: exercise, color: category.color)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .navigationTitle(category.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ExerciseListCard: View {
    let exercise: WatchExercise
    let color: Color

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "play.circle.fill")
                .font(.system(size: 24))
                .foregroundColor(color)

            VStack(alignment: .leading, spacing: 2) {
                Text(exercise.name)
                    .font(.system(size: 13, weight: .medium))

                HStack(spacing: 4) {
                    Image(systemName: "clock")
                        .font(.system(size: 10))
                    Text("\(exercise.duration) min")
                    Text("•")
                    Text(exercise.difficulty)
                }
                .font(.caption2)
                .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(white: 0.15))
        )
    }
}

// MARK: - Exercise Detail Watch View

struct ExerciseDetailWatchView: View {
    let exercise: WatchExercise
    let color: Color
    @State private var isPerforming = false
    @State private var showCompletion = false
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var isSaving = false
    @EnvironmentObject var connectivityManager: WatchConnectivityManager

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Exercise Icon
                ZStack {
                    Circle()
                        .fill(color.opacity(0.2))
                        .frame(width: 70, height: 70)

                    Image(systemName: "figure.flexibility")
                        .font(.system(size: 32))
                        .foregroundColor(color)
                }
                .padding(.top, 8)

                // Exercise Name
                Text(exercise.name)
                    .font(.headline)
                    .multilineTextAlignment(.center)

                // Metadata
                HStack(spacing: 16) {
                    MetadataPill(icon: "clock", text: "\(exercise.duration) min")
                    MetadataPill(icon: "gauge", text: exercise.difficulty)
                }

                // Instructions
                VStack(alignment: .leading, spacing: 8) {
                    Text("Instructions")
                        .font(.system(size: 13, weight: .semibold))

                    ForEach(Array(exercise.steps.enumerated()), id: \.offset) { index, step in
                        HStack(alignment: .top, spacing: 8) {
                            Text("\(index + 1)")
                                .font(.caption)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .frame(width: 20, height: 20)
                                .background(Circle().fill(color))

                            Text(step)
                                .font(.caption)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(white: 0.15))
                )

                // Start Button
                Button(action: {
                    startExercise()
                }) {
                    HStack {
                        Image(systemName: isPerforming ? "stop.fill" : "play.fill")
                        Text(isPerforming ? "Stop Exercise" : "Start Exercise")
                    }
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(
                        LinearGradient(
                            colors: isPerforming ? [.red, .orange] : [color, color.opacity(0.8)],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(12)
                }
                .buttonStyle(.plain)

                if isPerforming {
                    Button(action: {
                        Task { await completeExercise() }
                    }) {
                        HStack {
                            if isSaving {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "checkmark.circle.fill")
                            }
                            Text(isSaving ? "Saving..." : "Complete")
                        }
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color.green)
                        .cornerRadius(12)
                    }
                    .buttonStyle(.plain)
                    .disabled(isSaving)
                }
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .navigationTitle("Exercise")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Exercise Completed!", isPresented: $showCompletion) {
            Button("Done", role: .cancel) {}
        } message: {
            Text("Great job! Keep up the good work.")
        }
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage)
        }
    }

    private func startExercise() {
        isPerforming.toggle()
        if isPerforming {
            WKInterfaceDevice.current().play(.start)
        }
    }

    private func completeExercise() async {
        isSaving = true

        let message: [String: Any] = [
            "type": "exercise_completed",
            "name": exercise.name,
            "duration": exercise.duration,
            "timestamp": Date().timeIntervalSince1970
        ]

        do {
            if let response = await connectivityManager.sendMessage(message) {
                await MainActor.run {
                    isSaving = false

                    if let success = response["success"] as? Bool, success {
                        // Success - show completion feedback
                        isPerforming = false
                        showCompletion = true
                        WKInterfaceDevice.current().play(.success)
                    } else if let error = response["error"] as? String {
                        // Server-side error
                        errorMessage = error
                        showError = true
                        WKInterfaceDevice.current().play(.failure)
                    } else {
                        // Unknown response format
                        errorMessage = "Failed to save exercise"
                        showError = true
                        WKInterfaceDevice.current().play(.failure)
                    }
                }
            } else {
                // No response (iPhone not reachable)
                await MainActor.run {
                    isSaving = false
                    errorMessage = "iPhone not reachable. Exercise not saved."
                    showError = true
                    WKInterfaceDevice.current().play(.failure)
                }
            }
        } catch {
            // Connection error
            await MainActor.run {
                isSaving = false
                errorMessage = "Connection error: \(error.localizedDescription)"
                showError = true
                WKInterfaceDevice.current().play(.failure)
            }
        }
    }
}

struct MetadataPill: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
            Text(text)
        }
        .font(.caption)
        .foregroundColor(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(white: 0.15))
        )
    }
}

// MARK: - Data Models

struct ExerciseCategoryInfo {
    let title: String
    let icon: String
    let color: Color
    let description: String
    let exerciseCount: Int
    let sampleExercises: [WatchExercise]
}

extension ExerciseCategory {
    static let watchCategories: [ExerciseCategoryInfo] = [
        ExerciseCategoryInfo(
            title: "Stretching",
            icon: "figure.flexibility",
            color: .blue,
            description: "Gentle stretches for flexibility",
            exerciseCount: 12,
            sampleExercises: [
                WatchExercise(
                    name: "Neck Rotation",
                    duration: 5,
                    difficulty: "Beginner",
                    steps: [
                        "Sit comfortably",
                        "Slowly turn head right",
                        "Hold for 5 seconds",
                        "Return to center",
                        "Repeat on left side"
                    ]
                ),
                WatchExercise(
                    name: "Cat-Cow Stretch",
                    duration: 5,
                    difficulty: "Beginner",
                    steps: [
                        "Start on hands and knees",
                        "Arch back upward (cat)",
                        "Lower back downward (cow)",
                        "Repeat 10 times slowly"
                    ]
                )
            ]
        ),
        ExerciseCategoryInfo(
            title: "Mobility",
            icon: "figure.walk",
            color: .green,
            description: "Improve range of motion",
            exerciseCount: 15,
            sampleExercises: [
                WatchExercise(
                    name: "Hip Circles",
                    duration: 5,
                    difficulty: "Beginner",
                    steps: [
                        "Stand with feet hip-width",
                        "Place hands on hips",
                        "Make circular motions",
                        "Repeat 10 times each direction"
                    ]
                ),
                WatchExercise(
                    name: "Shoulder Rolls",
                    duration: 3,
                    difficulty: "Beginner",
                    steps: [
                        "Stand or sit comfortably",
                        "Roll shoulders forward",
                        "Then roll backward",
                        "Repeat 15 times"
                    ]
                )
            ]
        ),
        ExerciseCategoryInfo(
            title: "Breathing",
            icon: "wind",
            color: .purple,
            description: "Chest expansion exercises",
            exerciseCount: 6,
            sampleExercises: [
                WatchExercise(
                    name: "Deep Breathing",
                    duration: 5,
                    difficulty: "Beginner",
                    steps: [
                        "Sit or lie comfortably",
                        "Place hands on ribs",
                        "Breathe in deeply for 4 counts",
                        "Exhale slowly for 6 counts",
                        "Repeat for 5 minutes"
                    ]
                )
            ]
        ),
        ExerciseCategoryInfo(
            title: "Strengthening",
            icon: "figure.strengthtraining.traditional",
            color: .orange,
            description: "Build supporting muscles",
            exerciseCount: 10,
            sampleExercises: [
                WatchExercise(
                    name: "Wall Push-Ups",
                    duration: 5,
                    difficulty: "Intermediate",
                    steps: [
                        "Stand arm's length from wall",
                        "Place hands on wall",
                        "Lean forward slowly",
                        "Push back to start",
                        "Repeat 10 times"
                    ]
                ),
                WatchExercise(
                    name: "Pelvic Tilts",
                    duration: 5,
                    difficulty: "Beginner",
                    steps: [
                        "Lie on back, knees bent",
                        "Flatten back against floor",
                        "Tilt pelvis upward",
                        "Hold 5 seconds",
                        "Repeat 20 times"
                    ]
                )
            ]
        )
    ]
}

struct WatchExercise {
    let name: String
    let duration: Int
    let difficulty: String
    let steps: [String]
}

struct CompletedExercise: Identifiable {
    let id = UUID()
    let name: String
    let duration: Int
    let date: Date
}

// MARK: - Preview

#Preview {
    NavigationStack {
        ExercisesWatchView()
            .environmentObject(WatchConnectivityManager.shared)
    }
}
