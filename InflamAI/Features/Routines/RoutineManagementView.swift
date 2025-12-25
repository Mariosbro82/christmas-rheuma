//
//  RoutineManagementView.swift
//  InflamAI
//
//  Manage custom exercise routines
//  List, create, edit, and delete user routines
//

import SwiftUI
import CoreData

struct RoutineManagementView: View {
    @StateObject private var viewModel: RoutineManagementViewModel
    @State private var showingCreateRoutine = false
    @State private var routineToEdit: UserRoutine?
    @State private var showingDeleteConfirmation = false
    @State private var routineToDelete: UserRoutine?
    @Environment(\.dismiss) private var dismiss

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: RoutineManagementViewModel(context: context))
    }

    var body: some View {
        Group {
            if viewModel.routines.isEmpty {
                emptyStateView
            } else {
                routineListView
            }
        }
        .navigationTitle(NSLocalizedString("routine.title", comment: ""))
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showingCreateRoutine = true
                } label: {
                    Label(NSLocalizedString("routine.create_new", comment: ""), systemImage: "plus")
                }
            }
        }
        .sheet(isPresented: $showingCreateRoutine) {
            RoutineBuilderView(context: viewModel.context)
        }
        .sheet(item: $routineToEdit) { routine in
            RoutineBuilderView(context: viewModel.context, routineToEdit: routine)
        }
        .alert(NSLocalizedString("routine.delete_confirm_title", comment: ""), isPresented: $showingDeleteConfirmation) {
            Button(NSLocalizedString("action.cancel", comment: ""), role: .cancel) {}
            Button(NSLocalizedString("routine.delete_confirm_button", comment: ""), role: .destructive) {
                if let routine = routineToDelete {
                    viewModel.deleteRoutine(routine)
                }
            }
        } message: {
            Text(NSLocalizedString("routine.delete_confirm_message", comment: ""))
        }
        .onAppear {
            viewModel.loadRoutines()
        }
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 24) {
            Image(systemName: "figure.mixed.cardio")
                .font(.system(size: 80))
                .foregroundColor(.secondary)

            VStack(spacing: 8) {
                Text(NSLocalizedString("routine.no_routines", comment: ""))
                    .font(.title3)
                    .fontWeight(.semibold)

                Text(NSLocalizedString("routine.no_routines_desc", comment: ""))
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

            Button {
                showingCreateRoutine = true
            } label: {
                Label(NSLocalizedString("routine.create_new", comment: ""), systemImage: "plus.circle.fill")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(12)
            }
        }
        .padding()
    }

    // MARK: - Routine List

    private var routineListView: some View {
        List {
            ForEach(viewModel.routines) { routine in
                RoutineRow(
                    routine: routine,
                    isActive: routine.isActive,
                    onToggleActive: {
                        viewModel.toggleActiveRoutine(routine)
                    },
                    onEdit: {
                        routineToEdit = routine
                    },
                    onDelete: {
                        routineToDelete = routine
                        showingDeleteConfirmation = true
                    }
                )
            }
        }
        .listStyle(.insetGrouped)
    }
}

// MARK: - Routine Row

struct RoutineRow: View {
    let routine: UserRoutine
    let isActive: Bool
    let onToggleActive: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        NavigationLink {
            RoutineDetailView(routine: routine)
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(routine.name ?? "Unnamed Routine")
                        .font(.headline)

                    Spacer()

                    if isActive {
                        Text(NSLocalizedString("home.routine.active", comment: ""))
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.green)
                            .cornerRadius(8)
                    }
                }

                HStack(spacing: 12) {
                    Label(String(format: NSLocalizedString("routine.duration_format", comment: ""), routine.totalDuration), systemImage: "clock")
                    Label(String(format: NSLocalizedString("routine.completed_count", comment: ""), routine.timesCompleted), systemImage: "checkmark.circle")
                }
                .font(.caption)
                .foregroundColor(.secondary)

                if let notes = routine.customNotes, !notes.isEmpty {
                    Text(notes)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label(NSLocalizedString("routine.delete", comment: ""), systemImage: "trash")
            }

            Button {
                onEdit()
            } label: {
                Label(NSLocalizedString("routine.edit", comment: ""), systemImage: "pencil")
            }
            .tint(.blue)
        }
        .swipeActions(edge: .leading, allowsFullSwipe: true) {
            Button {
                onToggleActive()
            } label: {
                if isActive {
                    Label(NSLocalizedString("routine.remove_active", comment: ""), systemImage: "star.slash")
                } else {
                    Label(NSLocalizedString("routine.set_active", comment: ""), systemImage: "star.fill")
                }
            }
            .tint(isActive ? .orange : .green)
        }
    }
}

// MARK: - View Model

@MainActor
class RoutineManagementViewModel: ObservableObject {
    @Published var routines: [UserRoutine] = []
    let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func loadRoutines() {
        let request: NSFetchRequest<UserRoutine> = UserRoutine.fetchRequest()
        request.sortDescriptors = [
            NSSortDescriptor(keyPath: \UserRoutine.isActive, ascending: false),
            NSSortDescriptor(keyPath: \UserRoutine.createdAt, ascending: false)
        ]

        do {
            routines = try context.fetch(request)
        } catch {
            print("Failed to load routines: \(error)")
        }
    }

    func toggleActiveRoutine(_ routine: UserRoutine) {
        // If this routine is already active, deactivate it
        if routine.isActive {
            routine.isActive = false
        } else {
            // Deactivate all other routines first
            let request: NSFetchRequest<UserRoutine> = UserRoutine.fetchRequest()
            request.predicate = NSPredicate(format: "isActive == YES")

            if let activeRoutines = try? context.fetch(request) {
                activeRoutines.forEach { $0.isActive = false }
            }

            // Activate this routine
            routine.isActive = true
        }

        do {
            try context.save()
            loadRoutines()
        } catch {
            print("Failed to toggle active routine: \(error)")
        }
    }

    func deleteRoutine(_ routine: UserRoutine) {
        context.delete(routine)

        do {
            try context.save()
            loadRoutines()
        } catch {
            print("Failed to delete routine: \(error)")
        }
    }
}

// MARK: - Preview

struct RoutineManagementView_Previews: PreviewProvider {
    static var previews: some View {
        RoutineManagementView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
