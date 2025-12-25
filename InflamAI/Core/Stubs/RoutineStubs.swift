//
//  RoutineStubs.swift
//  InflamAI
//
//  Lightweight stubs for Routine-related views and Exercise type
//  to allow RoutineDetailView and RoutineManagementView to compile
//  Full implementations are excluded due to dependencies
//

import SwiftUI
import CoreData

// MARK: - Exercise Type (Stub)

struct Exercise: Identifiable, Codable, Equatable {
    let id: UUID
    let name: String
    let description: String
    let category: ExerciseCategory
    let duration: Int // in seconds
    let difficulty: ExerciseDifficulty
    let bodyParts: [String]
    let instructions: [String]
    let imageName: String?
    let videoURL: String?

    // Static list - empty stub, full list in ExerciseData.swift
    static var allExercises: [Exercise] = []

    init(
        id: UUID = UUID(),
        name: String = "",
        description: String = "",
        category: ExerciseCategory = .stretching,
        duration: Int = 30,
        difficulty: ExerciseDifficulty = .beginner,
        bodyParts: [String] = [],
        instructions: [String] = [],
        imageName: String? = nil,
        videoURL: String? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.duration = duration
        self.difficulty = difficulty
        self.bodyParts = bodyParts
        self.instructions = instructions
        self.imageName = imageName
        self.videoURL = videoURL
    }
}

enum ExerciseCategory: String, Codable, CaseIterable {
    case stretching = "Stretching"
    case mobility = "Mobility"
    case strengthening = "Strengthening"
    case balance = "Balance"
    case breathing = "Breathing"
    case warmup = "Warm Up"
    case cooldown = "Cool Down"
}

enum ExerciseDifficulty: String, Codable, CaseIterable {
    case beginner = "Beginner"
    case intermediate = "Intermediate"
    case advanced = "Advanced"
}

// MARK: - RoutineExecutionView (Stub)

struct RoutineExecutionView: View {
    let routine: UserRoutine
    let exercises: [String]  // Exercise IDs as strings

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image(systemName: "figure.mixed.cardio")
                    .font(.system(size: 60))
                    .foregroundColor(.green)

                Text("Routine Execution")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("This feature is under development")
                    .font(.body)
                    .foregroundColor(.secondary)

                if let name = routine.name {
                    Text("Routine: \(name)")
                        .font(.headline)
                }

                Text("\(exercises.count) exercises")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Button("Close") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Execute Routine")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - RoutineBuilderView (Stub)

struct RoutineBuilderView: View {
    let context: NSManagedObjectContext
    let routineToEdit: UserRoutine?

    @Environment(\.dismiss) private var dismiss

    init(context: NSManagedObjectContext, routineToEdit: UserRoutine? = nil) {
        self.context = context
        self.routineToEdit = routineToEdit
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image(systemName: "hammer.fill")
                    .font(.system(size: 60))
                    .foregroundColor(.purple)

                Text("Routine Builder")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("This feature is under development")
                    .font(.body)
                    .foregroundColor(.secondary)

                if let routine = routineToEdit, let name = routine.name {
                    Text("Editing: \(name)")
                        .font(.headline)
                }

                Button("Close") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .tint(Colors.Primary.p500)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(.systemGroupedBackground))
            .navigationTitle(routineToEdit == nil ? "New Routine" : "Edit Routine")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - RoutineExerciseItem (needed by RoutineDetailView)

struct RoutineExerciseItem: Identifiable, Codable {
    let id: UUID
    let exerciseId: String
    let duration: Int
    let order: Int

    init(id: UUID = UUID(), exerciseId: String = "", duration: Int = 30, order: Int = 0) {
        self.id = id
        self.exerciseId = exerciseId
        self.duration = duration
        self.order = order
    }
}
