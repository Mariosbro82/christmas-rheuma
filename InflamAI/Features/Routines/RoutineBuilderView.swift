//
//  RoutineBuilderView.swift
//  InflamAI
//
//  Build and edit custom exercise routines
//  Select exercises, set duration, reminders, and notes
//

import SwiftUI
import CoreData

struct RoutineBuilderView: View {
    @StateObject private var viewModel: RoutineBuilderViewModel
    @Environment(\.dismiss) private var dismiss

    init(context: NSManagedObjectContext, routineToEdit: UserRoutine? = nil) {
        _viewModel = StateObject(wrappedValue: RoutineBuilderViewModel(context: context, routineToEdit: routineToEdit))
    }

    var body: some View {
        NavigationView {
            Form {
                // Basic Info Section
                Section {
                    TextField(NSLocalizedString("routine.name_placeholder", comment: ""), text: $viewModel.routineName)

                    TextEditor(text: $viewModel.customNotes)
                        .frame(minHeight: 100)
                        .foregroundColor(viewModel.customNotes.isEmpty ? .secondary : .primary)
                        .overlay(alignment: .topLeading) {
                            if viewModel.customNotes.isEmpty {
                                Text(NSLocalizedString("routine.notes_placeholder", comment: ""))
                                    .foregroundColor(.secondary)
                                    .padding(.top, 8)
                                    .padding(.leading, 5)
                                    .allowsHitTesting(false)
                            }
                        }
                } header: {
                    Text("Details")
                }

                // Reminder Section
                Section {
                    Toggle(NSLocalizedString("routine.reminder_toggle", comment: ""), isOn: $viewModel.reminderEnabled)

                    if viewModel.reminderEnabled {
                        DatePicker(NSLocalizedString("routine.reminder_time", comment: ""),
                                 selection: $viewModel.reminderTime,
                                 displayedComponents: .hourAndMinute)
                    }
                } header: {
                    Text("Reminder")
                }

                // Exercise Selection Section
                Section {
                    ForEach(viewModel.availableExercises) { exercise in
                        ExerciseSelectionRow(
                            exercise: exercise,
                            isSelected: viewModel.selectedExercises.contains(where: { $0.id == exercise.id })
                        ) {
                            viewModel.toggleExercise(exercise)
                        }
                    }
                } header: {
                    HStack {
                        Text("Select Exercises")
                        Spacer()
                        Text("\(viewModel.selectedExercises.count) selected")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } footer: {
                    Text("Total duration: \(viewModel.totalDuration) min")
                }

                // Active Routine Toggle
                if viewModel.routineToEdit != nil {
                    Section {
                        Toggle(NSLocalizedString("routine.set_active", comment: ""), isOn: $viewModel.setAsActive)
                    }
                }
            }
            .navigationTitle(viewModel.isEditing ? NSLocalizedString("routine.edit", comment: "") : NSLocalizedString("routine.create_new", comment: ""))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(NSLocalizedString("action.cancel", comment: "")) {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(NSLocalizedString("action.ok", comment: "")) {
                        viewModel.saveRoutine()
                        dismiss()
                    }
                    .disabled(!viewModel.canSave)
                }
            }
        }
    }
}

// MARK: - Exercise Selection Row

struct ExerciseSelectionRow: View {
    let exercise: Exercise
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(exercise.name)
                        .font(.subheadline)
                        .foregroundColor(.primary)

                    HStack {
                        Label("\(exercise.duration) min", systemImage: "clock")
                        Label(exercise.difficulty.rawValue, systemImage: "gauge")
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.blue)
                        .font(.title3)
                }
            }
        }
    }
}

// MARK: - View Model

@MainActor
class RoutineBuilderViewModel: ObservableObject {
    @Published var routineName = ""
    @Published var customNotes = ""
    @Published var reminderEnabled = false
    @Published var reminderTime = Date()
    @Published var selectedExercises: [Exercise] = []
    @Published var setAsActive = false

    let context: NSManagedObjectContext
    let routineToEdit: UserRoutine?
    let availableExercises: [Exercise]

    var isEditing: Bool {
        routineToEdit != nil
    }

    var totalDuration: Int {
        selectedExercises.reduce(0) { $0 + $1.duration }
    }

    var canSave: Bool {
        !routineName.isEmpty && !selectedExercises.isEmpty
    }

    init(context: NSManagedObjectContext, routineToEdit: UserRoutine? = nil) {
        self.context = context
        self.routineToEdit = routineToEdit
        self.availableExercises = Exercise.allExercises

        // Load existing routine data if editing
        if let routine = routineToEdit {
            self.routineName = routine.name ?? ""
            self.customNotes = routine.customNotes ?? ""
            self.reminderEnabled = routine.reminderEnabled
            self.reminderTime = routine.reminderTime ?? Date()
            self.setAsActive = routine.isActive

            // Decode exercises
            if let exerciseData = routine.exercises,
               let exerciseIds = try? JSONDecoder().decode([String].self, from: exerciseData) {
                self.selectedExercises = exerciseIds.compactMap { id in
                    availableExercises.first(where: { $0.id.uuidString == id })
                }
            }
        }
    }

    func toggleExercise(_ exercise: Exercise) {
        if let index = selectedExercises.firstIndex(where: { $0.id == exercise.id }) {
            selectedExercises.remove(at: index)
        } else {
            selectedExercises.append(exercise)
        }
    }

    func saveRoutine() {
        let routine: UserRoutine
        if let existing = routineToEdit {
            routine = existing
        } else {
            routine = UserRoutine(context: context)
            routine.id = UUID()
            routine.createdAt = Date()
        }

        routine.name = routineName
        routine.customNotes = customNotes.isEmpty ? nil : customNotes
        routine.reminderEnabled = reminderEnabled
        routine.reminderTime = reminderEnabled ? reminderTime : nil
        routine.totalDuration = Int16(totalDuration)

        // If setting as active, deactivate others first
        if setAsActive {
            let request: NSFetchRequest<UserRoutine> = UserRoutine.fetchRequest()
            request.predicate = NSPredicate(format: "isActive == YES")

            if let activeRoutines = try? context.fetch(request) {
                activeRoutines.forEach { $0.isActive = false }
            }

            routine.isActive = true
        }

        // Encode exercise IDs
        let exerciseIds = selectedExercises.map { $0.id.uuidString }
        if let encodedData = try? JSONEncoder().encode(exerciseIds) {
            routine.exercises = encodedData
        }

        do {
            try context.save()
        } catch {
            print("Failed to save routine: \(error)")
        }
    }
}

// MARK: - Preview

struct RoutineBuilderView_Previews: PreviewProvider {
    static var previews: some View {
        RoutineBuilderView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
