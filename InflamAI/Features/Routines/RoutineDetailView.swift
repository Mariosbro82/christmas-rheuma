//
//  RoutineDetailView.swift
//  InflamAI
//
//  Detailed view of a single routine
//  Shows exercises, duration, notes, and allows starting the routine
//

import SwiftUI
import CoreData

/// Exercise item stored in routine JSON data
private struct RoutineExerciseItem: Codable {
    let id: UUID
    let exerciseId: String
    let duration: Int
    let order: Int
}

struct RoutineDetailView: View {
    @ObservedObject var routine: UserRoutine
    @State private var showingExecution = false
    @State private var showingAnalytics = false
    @State private var decodedExercises: [String] = []

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header Stats
                statsSection

                // Exercises List
                exercisesSection

                // Notes
                if let notes = routine.customNotes, !notes.isEmpty {
                    notesSection(notes)
                }

                // Reminder Settings
                reminderSection

                // Analytics Button
                analyticsButton

                // Start Button
                startButton
            }
            .padding()
        }
        .navigationTitle(routine.name ?? "Routine")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showingAnalytics = true
                } label: {
                    Image(systemName: "chart.bar.xaxis")
                }
                .accessibilityLabel("View Analytics")
            }
        }
        .onAppear {
            loadExercises()
        }
        .sheet(isPresented: $showingExecution) {
            RoutineExecutionView(routine: routine, exercises: decodedExercises)
        }
        .sheet(isPresented: $showingAnalytics) {
            RoutineAnalyticsView(routine: routine)
        }
    }

    // MARK: - Analytics Button

    private var analyticsButton: some View {
        Button {
            showingAnalytics = true
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.purple.opacity(0.15))
                        .frame(width: 44, height: 44)

                    Image(systemName: "chart.xyaxis.line")
                        .font(.system(size: 20))
                        .foregroundColor(.purple)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("View Analytics")
                        .font(.headline)
                        .foregroundColor(.primary)

                    Text("Track your progress and feelings")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color.purple.opacity(0.1))
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Stats Section

    private var statsSection: some View {
        HStack(spacing: 20) {
            RoutineStatCard(
                icon: "clock",
                label: "Duration",
                value: String(format: NSLocalizedString("routine.duration_format", comment: ""), routine.totalDuration),
                color: .blue
            )

            RoutineStatCard(
                icon: "figure.mixed.cardio",
                label: "Exercises",
                value: "\(decodedExercises.count)",
                color: .green
            )

            RoutineStatCard(
                icon: "checkmark.circle",
                label: "Completed",
                value: "\(routine.timesCompleted)",
                color: .orange
            )
        }
    }

    // MARK: - Exercises Section

    private var exercisesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Exercises")
                .font(.headline)

            if decodedExercises.isEmpty {
                Text("No exercises in this routine")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
            } else {
                ForEach(Array(decodedExercises.enumerated()), id: \.offset) { index, exerciseId in
                    ExerciseListItem(
                        number: index + 1,
                        exerciseId: exerciseId
                    )
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 3)
    }

    // MARK: - Notes Section

    private func notesSection(_ notes: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "note.text")
                    .foregroundColor(.blue)
                Text("Notes")
                    .font(.headline)
            }

            Text(notes)
                .font(.body)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(16)
    }

    // MARK: - Reminder Section

    private var reminderSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "bell")
                    .foregroundColor(.purple)
                Text(NSLocalizedString("routine.reminder_toggle", comment: ""))
                    .font(.headline)

                Spacer()

                if routine.reminderEnabled {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                }
            }

            if routine.reminderEnabled, let reminderTime = routine.reminderTime {
                HStack {
                    Text(NSLocalizedString("routine.reminder_time", comment: ""))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(reminderTime, style: .time)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
            }
        }
        .padding()
        .background(Color.purple.opacity(0.1))
        .cornerRadius(16)
    }

    // MARK: - Start Button

    private var startButton: some View {
        Button {
            showingExecution = true
        } label: {
            Label(NSLocalizedString("routine.start", comment: ""), systemImage: "play.fill")
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .cornerRadius(12)
        }
    }

    // MARK: - Helper Methods

    private func loadExercises() {
        guard let exerciseData = routine.exercises else {
            decodedExercises = []
            return
        }

        // Try decoding as [RoutineExerciseItem] first (DemoDataSeeder format)
        if let exerciseItems = try? JSONDecoder().decode([RoutineExerciseItem].self, from: exerciseData) {
            // Extract exercise IDs (which are exercise names in demo data)
            decodedExercises = exerciseItems.sorted { $0.order < $1.order }.map { $0.exerciseId }
            return
        }

        // Fallback: try decoding as simple [String] array
        if let exerciseIds = try? JSONDecoder().decode([String].self, from: exerciseData) {
            decodedExercises = exerciseIds
            return
        }

        decodedExercises = []
    }
}

// MARK: - Supporting Views

struct RoutineStatCard: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            Text(value)
                .font(.title3)
                .fontWeight(.bold)

            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct ExerciseListItem: View {
    let number: Int
    let exerciseId: String

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.blue.opacity(0.1))
                    .frame(width: 32, height: 32)

                Text("\(number)")
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(.blue)
            }

            Text(exerciseNameFromId(exerciseId))
                .font(.subheadline)

            Spacer()
        }
        .padding(.vertical, 8)
    }

    private func exerciseNameFromId(_ id: String) -> String {
        // First check if id is a UUID and try to find in Exercise.allExercises
        if let exercise = Exercise.allExercises.first(where: { $0.id.uuidString == id }) {
            return exercise.name
        }
        // Demo data stores exercise names directly as IDs
        // If not empty, use the ID itself as the name
        if !id.isEmpty {
            return id
        }
        return "Unknown Exercise"
    }
}

// MARK: - Preview

struct RoutineDetailView_Previews: PreviewProvider {
    static var previews: some View {
        let context = InflamAIPersistenceController.preview.container.viewContext
        let routine = UserRoutine(context: context)
        routine.id = UUID()
        routine.name = "Morning Mobility"
        routine.totalDuration = 15
        routine.timesCompleted = 5
        routine.customNotes = "Great for starting the day"
        routine.reminderEnabled = true
        routine.reminderTime = Date()

        return NavigationView {
            RoutineDetailView(routine: routine)
        }
    }
}
