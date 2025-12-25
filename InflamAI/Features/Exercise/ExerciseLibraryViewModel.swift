//
//  ExerciseLibraryViewModel.swift
//  InflamAI-Swift
//
//  Created by Claude Code on 2025-01-25.
//

import Foundation
import CoreData
import Combine

@MainActor
class ExerciseLibraryViewModel: ObservableObject {

    // MARK: - Published Properties
    @Published var exercises: [Exercise] = []
    @Published var filteredExercises: [Exercise] = []
    @Published var selectedCategory: ExerciseCategory = .all
    @Published var searchText: String = ""
    @Published var loadingState: LoadingState = .idle
    @Published var exerciseLogs: [ExerciseLog] = []
    @Published var errorMessage: String?

    // MARK: - Dependencies
    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization
    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
        setupSearchObserver()
    }

    // MARK: - Public Methods

    func loadExercises() async {
        loadingState = .loading

        do {
            // Load exercises from Core Data or use default exercises
            exercises = await loadSavedExercises()

            if exercises.isEmpty {
                exercises = getDefaultExercises()
            }

            applyFilters()
            loadingState = .loaded
        } catch {
            loadingState = .error(error)
            errorMessage = "Failed to load exercises: \(error.localizedDescription)"
        }
    }

    func logExercise(_ exercise: Exercise, duration: Int, difficulty: DifficultyRating) async {
        let context = persistenceController.viewContext
        let log = ExerciseLog(context: context)
        log.id = UUID()
        log.date = Date()
        log.exerciseName = exercise.name
        log.duration = Int16(duration)
        log.difficultyRating = Int16(difficulty.rawValue)

        do {
            try context.save()
            await loadExerciseLogs()
        } catch {
            errorMessage = "Failed to log exercise: \(error.localizedDescription)"
        }
    }

    func loadExerciseLogs() async {
        let context = persistenceController.viewContext
        let request = ExerciseLog.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \ExerciseLog.date, ascending: false)]
        request.fetchLimit = 50

        do {
            exerciseLogs = try context.fetch(request)
        } catch {
            errorMessage = "Failed to load exercise logs: \(error.localizedDescription)"
        }
    }

    func setCategory(_ category: ExerciseCategory) {
        selectedCategory = category
        applyFilters()
    }

    // MARK: - Private Methods

    private func setupSearchObserver() {
        $searchText
            .debounce(for: .milliseconds(300), scheduler: RunLoop.main)
            .sink { [weak self] _ in
                self?.applyFilters()
            }
            .store(in: &cancellables)
    }

    private func applyFilters() {
        var result = exercises

        // Filter by category
        if selectedCategory != .all {
            result = result.filter { $0.category == selectedCategory }
        }

        // Filter by search text
        if !searchText.isEmpty {
            result = result.filter { exercise in
                exercise.name.localizedCaseInsensitiveContains(searchText) ||
                exercise.description.localizedCaseInsensitiveContains(searchText)
            }
        }

        filteredExercises = result
    }

    private func loadSavedExercises() async -> [Exercise] {
        // Placeholder for loading custom exercises from Core Data
        // For now, we'll use the default exercises
        return []
    }

    private func getDefaultExercises() -> [Exercise] {
        return [
            Exercise(
                name: "Gentle Neck Rolls",
                category: .stretching,
                description: "Slowly roll your head in a circular motion to relieve neck tension",
                duration: 5,
                difficulty: .easy,
                instructions: [
                    "Sit or stand with good posture",
                    "Slowly tilt your head forward",
                    "Roll your head to the right shoulder",
                    "Continue rolling back and to the left",
                    "Repeat 5 times in each direction"
                ]
            ),
            Exercise(
                name: "Cat-Cow Stretch",
                category: .stretching,
                description: "Spinal flexibility exercise to reduce stiffness",
                duration: 5,
                difficulty: .easy,
                instructions: [
                    "Start on hands and knees",
                    "Arch back up like a cat (exhale)",
                    "Drop belly down like a cow (inhale)",
                    "Repeat slowly 10 times"
                ]
            ),
            Exercise(
                name: "Wall Angels",
                category: .strengthening,
                description: "Improve posture and shoulder mobility",
                duration: 10,
                difficulty: .moderate,
                instructions: [
                    "Stand with back against wall",
                    "Raise arms to shoulder height",
                    "Slowly slide arms up the wall",
                    "Keep elbows and wrists against wall",
                    "Repeat 10-15 times"
                ]
            ),
            Exercise(
                name: "Gentle Walking",
                category: .aerobic,
                description: "Low-impact cardiovascular exercise",
                duration: 20,
                difficulty: .easy,
                instructions: [
                    "Start with 5-minute warm-up",
                    "Walk at comfortable pace",
                    "Maintain good posture",
                    "Cool down for final 5 minutes"
                ]
            ),
            Exercise(
                name: "Deep Breathing",
                category: .breathing,
                description: "Chest expansion and relaxation exercise",
                duration: 10,
                difficulty: .easy,
                instructions: [
                    "Sit comfortably with straight spine",
                    "Breathe in deeply through nose (4 counts)",
                    "Hold for 4 counts",
                    "Exhale slowly through mouth (6 counts)",
                    "Repeat 10 times"
                ]
            )
        ]
    }
}

// MARK: - Supporting Types

struct Exercise: Identifiable {
    let id: UUID
    let name: String
    let category: ExerciseCategory
    let description: String
    let duration: Int // in minutes
    let difficulty: DifficultyLevel
    let instructions: [String]
    var videoURL: String?
    var steps: [ExerciseStep]? // NEW: Step-by-step guidance

    // Additional fields for detailed exercises
    var targetAreas: [String]?
    var benefits: [String]?
    var safetyTips: [String]?

    // Default initializer
    init(
        id: UUID = UUID(),
        name: String,
        category: ExerciseCategory,
        description: String = "",
        duration: Int,
        difficulty: DifficultyLevel,
        instructions: [String],
        videoURL: String? = nil,
        steps: [ExerciseStep]? = nil,
        targetAreas: [String]? = nil,
        benefits: [String]? = nil,
        safetyTips: [String]? = nil
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.description = description
        self.duration = duration
        self.difficulty = difficulty
        self.instructions = instructions
        self.videoURL = videoURL
        self.steps = steps
        self.targetAreas = targetAreas
        self.benefits = benefits
        self.safetyTips = safetyTips
    }
}

enum ExerciseCategory: String, CaseIterable {
    case all = "All"
    case stretching = "Stretching"
    case strengthening = "Strengthening"
    case aerobic = "Aerobic"
    case breathing = "Breathing"

    var icon: String {
        switch self {
        case .all: return "list.bullet"
        case .stretching: return "figure.flexibility"
        case .strengthening: return "dumbbell"
        case .aerobic: return "figure.walk"
        case .breathing: return "wind"
        }
    }
}

enum DifficultyLevel: String {
    case easy = "Easy"
    case moderate = "Moderate"
    case challenging = "Challenging"

    var color: String {
        switch self {
        case .easy: return "green"
        case .moderate: return "yellow"
        case .challenging: return "orange"
        }
    }
}

enum DifficultyRating: Int16 {
    case veryEasy = 1
    case easy = 2
    case moderate = 3
    case hard = 4
    case veryHard = 5
}
