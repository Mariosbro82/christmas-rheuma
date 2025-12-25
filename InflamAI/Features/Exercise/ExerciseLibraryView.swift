//
//  ExerciseLibraryView.swift
//  InflamAI
//
//  Complete exercise/mobility system with routine library for AS patients
//  Includes 50+ exercises, custom routines, progress tracking
//

import SwiftUI
import AVKit
import CoreData

struct ExerciseLibraryView: View {
    @StateObject private var viewModel = ExerciseViewModel()
    @State private var selectedCategory: ExerciseCategory = .all
    @State private var searchText = ""
    @State private var showingCustomRoutine = false

    var body: some View {
        VStack(spacing: 0) {
            // Category Filter
            categoryScrollView

            // Search Bar
            searchBar

            // Exercise List
            exerciseList
        }
        .navigationTitle("Exercise Library")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    Button {
                        showingCustomRoutine = true
                    } label: {
                        Label("Create Routine", systemImage: "plus.circle")
                    }

                    Button {
                        // Start quick session
                    } label: {
                        Label("Quick Session", systemImage: "play.circle")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showingCustomRoutine) {
            CustomRoutineBuilder(viewModel: viewModel)
        }
    }

    private var categoryScrollView: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(ExerciseCategory.allCases, id: \.self) { category in
                    ExerciseCategoryChip(
                        category: category,
                        isSelected: selectedCategory == category
                    ) {
                        selectedCategory = category
                    }
                }
            }
            .padding()
        }
        .background(Color(.systemGray6))
    }

    private var searchBar: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)

            TextField("Search exercises", text: $searchText)
                .textFieldStyle(.plain)

            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
    }

    private var exerciseList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                // CRIT-005: Add loading state
                if viewModel.isLoading {
                    ForEach(0..<5, id: \.self) { _ in
                        ExerciseCardSkeleton()
                    }
                }
                // CRIT-005: Add empty state
                else if filteredExercises.isEmpty {
                    EmptyStateView(
                        icon: "figure.walk",
                        title: searchText.isEmpty ? "No Exercises" : "No Results",
                        message: searchText.isEmpty
                            ? "Exercise library is loading..."
                            : "No exercises match '\(searchText)'. Try a different search.",
                        actionTitle: searchText.isEmpty ? nil : "Clear Search"
                    ) {
                        searchText = ""
                    }
                    .padding(.top, 40)
                }
                // Content
                else {
                    ForEach(filteredExercises) { exercise in
                        NavigationLink {
                            ExerciseDetailView(exercise: exercise)
                        } label: {
                            ExerciseCard(exercise: exercise)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding()
        }
    }

    private var filteredExercises: [Exercise] {
        viewModel.exercises
            .filter { exercise in
                (selectedCategory == .all || exercise.category == selectedCategory) &&
                (searchText.isEmpty || exercise.name.localizedCaseInsensitiveContains(searchText))
            }
    }
}

// MARK: - Category Chip

struct ExerciseCategoryChip: View {
    let category: ExerciseCategory
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: category.icon)
                Text(category.rawValue)
            }
            .font(.subheadline)
            .fontWeight(isSelected ? .semibold : .regular)
            .foregroundColor(isSelected ? .white : .primary)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(isSelected ? Color.blue : Color(.systemGray5))
            .cornerRadius(20)
        }
    }
}

// MARK: - Exercise Card

struct ExerciseCard: View {
    let exercise: Exercise

    var body: some View {
        HStack(spacing: 12) {
            // Thumbnail
            ZStack {
                Rectangle()
                    .fill(Color.blue.opacity(0.1))
                    .frame(width: 80, height: 80)
                    .cornerRadius(12)

                Image(systemName: exercise.category.icon)
                    .font(.system(size: 30))
                    .foregroundColor(.blue)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(exercise.name)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                HStack {
                    Label(exercise.difficulty.rawValue, systemImage: "gauge")
                        .font(.caption)
                        .foregroundColor(exercise.difficulty.color)

                    Text("•")
                        .foregroundColor(.secondary)

                    Text("\(exercise.duration) min")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if !exercise.targetAreas.isEmpty {
                    Text(exercise.targetAreas.prefix(2).joined(separator: ", "))
                        .font(.caption)
                        .foregroundColor(.blue)
                        .lineLimit(1)
                }
            }

            Spacer()
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 2)
    }
}

// MARK: - Exercise Detail

struct ExerciseDetailView: View {
    let exercise: Exercise
    @State private var isPlaying = false
    @State private var hasCompleted = false
    @State private var showMannequinCoach = false

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Video Player Placeholder
                videoPlayerSection

                // Exercise Info
                exerciseInfoSection

                // Instructions
                instructionsSection

                // Benefits
                benefitsSection

                // Safety Tips
                safetySection

                // Action Buttons
                actionButtons
            }
            .padding()
        }
        .navigationTitle(exercise.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink(destination: ExerciseStatsView(exercise: exercise)) {
                    Label("Stats", systemImage: "chart.bar.fill")
                }
            }
        }
    }

    private var videoPlayerSection: some View {
        VStack(spacing: 0) {
            if let videoURL = exercise.videoURL, !videoURL.isEmpty {
                YouTubePlayerView(url: videoURL)
                    .accessibilityLabel("Tutorial video for \(exercise.name)")
            } else {
                // Fallback for exercises without videos
                ZStack {
                    Rectangle()
                        .fill(Color.black.opacity(0.1))
                        .aspectRatio(16/9, contentMode: .fit)

                    VStack(spacing: 12) {
                        Image(systemName: "video.slash")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)

                        Text("No Video Available")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                }
                .cornerRadius(16)
            }
        }
    }

    private var exerciseInfoSection: some View {
        VStack(spacing: 12) {
            HStack {
                InfoBadge(icon: "clock", text: "\(exercise.duration) min")
                InfoBadge(icon: "flame", text: exercise.difficulty.rawValue)
                InfoBadge(icon: "figure.walk", text: exercise.category.rawValue)
            }

            if !exercise.targetAreas.isEmpty {
                HStack {
                    Text("Target Areas:")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    ForEach(exercise.targetAreas, id: \.self) { area in
                        Text(area)
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.1))
                            .foregroundColor(.blue)
                            .cornerRadius(8)
                    }
                }
            }
        }
    }

    private var instructionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Instructions")
                .font(.headline)

            ForEach(Array(exercise.instructions.enumerated()), id: \.offset) { index, instruction in
                HStack(alignment: .top, spacing: 12) {
                    Text("\(index + 1)")
                        .font(.subheadline)
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
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private var benefitsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Benefits")
                .font(.headline)

            ForEach(exercise.benefits, id: \.self) { benefit in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)

                    Text(benefit)
                        .font(.body)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .cornerRadius(16)
    }

    private var safetySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                Text("Safety Tips")
                    .font(.headline)
            }

            ForEach(exercise.safetyTips, id: \.self) { tip in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "hand.raised.fill")
                        .foregroundColor(.orange)
                        .font(.caption)

                    Text(tip)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(16)
    }

    private var actionButtons: some View {
        VStack(spacing: 12) {
            // Mannequin Coach Button (Primary)
            Button {
                showMannequinCoach = true
            } label: {
                HStack {
                    Image(systemName: "figure.walk")
                    Text("Start with Coach")
                    Image(systemName: "sparkles")
                        .font(.caption)
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    LinearGradient(
                        colors: [Color.blue, Color.blue.opacity(0.8)],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(12)
            }
            .accessibilityLabel("Start exercise with animated coach")
            .accessibilityHint("Opens the mannequin coach with visual guidance, pain monitoring, and voice instructions")

            // Traditional Start Button (Secondary)
            Button {
                // Start exercise without coach
                hasCompleted = true
            } label: {
                Label("Start Without Coach", systemImage: "play.fill")
                    .font(.subheadline)
                    .foregroundColor(.blue)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(12)
            }

            if hasCompleted {
                Button {
                    // Log completion
                } label: {
                    Label("Log Completion", systemImage: "checkmark.circle")
                        .font(.subheadline)
                        .foregroundColor(.green)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(12)
                }
            }
        }
    }
}

struct InfoBadge: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
            Text(text)
        }
        .font(.caption)
        .foregroundColor(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Custom Routine Builder

struct CustomRoutineBuilder: View {
    @ObservedObject var viewModel: ExerciseViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var routineName = ""
    @State private var selectedExercises: [Exercise] = []

    var body: some View {
        NavigationView {
            Form {
                Section {
                    TextField("Routine Name", text: $routineName)
                } header: {
                    Text("Details")
                }

                Section {
                    ForEach(viewModel.exercises) { exercise in
                        Button {
                            toggleExercise(exercise)
                        } label: {
                            HStack {
                                Text(exercise.name)
                                    .foregroundColor(.primary)

                                Spacer()

                                if selectedExercises.contains(where: { $0.id == exercise.id }) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                    }
                } header: {
                    Text("Select Exercises")
                } footer: {
                    Text("\(selectedExercises.count) exercises selected")
                }
            }
            .navigationTitle("New Routine")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveRoutine()
                    }
                    .disabled(routineName.isEmpty || selectedExercises.isEmpty)
                }
            }
        }
    }

    private func toggleExercise(_ exercise: Exercise) {
        if let index = selectedExercises.firstIndex(where: { $0.id == exercise.id }) {
            selectedExercises.remove(at: index)
        } else {
            selectedExercises.append(exercise)
        }
    }

    private func saveRoutine() {
        // Save routine
        dismiss()
    }
}

// MARK: - View Model

@MainActor
class ExerciseViewModel: ObservableObject {
    @Published var exercises: [Exercise] = []
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    init() {
        Task {
            await loadExercises()
        }
    }

    func loadExercises() async {
        isLoading = true
        errorMessage = nil

        // Simulate brief loading for better UX
        try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds

        // Load 52 AS-specific exercises
        exercises = Exercise.allExercises
        isLoading = false
    }

    func retry() {
        Task {
            await loadExercises()
        }
    }
}

// MARK: - Models

struct Exercise: Identifiable {
    let id: UUID
    let name: String
    let category: ExerciseCategory
    let difficulty: ExerciseDifficulty
    let duration: Int // minutes
    let targetAreas: [String]
    let instructions: [String]
    let benefits: [String]
    let safetyTips: [String]
    let videoURL: String?
    var steps: [ExerciseStep]? = nil
}

enum ExerciseCategory: String, CaseIterable {
    case all = "All"
    case stretching = "Stretching"
    case strengthening = "Strengthening"
    case mobility = "Mobility"
    case breathing = "Breathing"
    case posture = "Posture"
    case balance = "Balance"

    var icon: String {
        switch self {
        case .all: return "list.bullet"
        case .stretching: return "figure.flexibility"
        case .strengthening: return "figure.strengthtraining.traditional"
        case .mobility: return "figure.walk"
        case .breathing: return "wind"
        case .posture: return "figure.stand"
        case .balance: return "figure.mind.and.body"
        }
    }
}

enum ExerciseDifficulty: String {
    case beginner = "Beginner"
    case intermediate = "Intermediate"
    case advanced = "Advanced"

    var color: Color {
        switch self {
        case .beginner: return .green
        case .intermediate: return .orange
        case .advanced: return .red
        }
    }
}

// MARK: - Sample Data

extension Exercise {
    static let sampleExercises: [Exercise] = [
        Exercise(
            id: UUID(),
            name: "Cervical Spine Rotation",
            category: .mobility,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Cervical Spine", "Neck"],
            instructions: [
                "Sit or stand with good posture",
                "Slowly turn head to the right as far as comfortable",
                "Hold for 5 seconds",
                "Return to center",
                "Repeat on left side",
                "Perform 10 repetitions each side"
            ],
            benefits: [
                "Improves cervical spine mobility",
                "Reduces neck stiffness",
                "Maintains range of motion"
            ],
            safetyTips: [
                "Move slowly and gently",
                "Stop if you feel sharp pain",
                "Do not force the movement"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Thoracic Extension Over Foam Roller",
            category: .mobility,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Thoracic Spine", "Upper Back"],
            instructions: [
                "Lie on foam roller positioned at mid-back",
                "Support head with hands",
                "Slowly extend backwards over roller",
                "Hold for 3-5 seconds",
                "Return to starting position",
                "Repeat 10 times"
            ],
            benefits: [
                "Increases thoracic extension",
                "Reduces kyphosis",
                "Improves posture"
            ],
            safetyTips: [
                "Use firm foam roller",
                "Avoid if you have osteoporosis",
                "Stop if experiencing pain"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Cat-Cow Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Entire Spine", "Lower Back"],
            instructions: [
                "Start on hands and knees",
                "Arch back (cow position) - look up",
                "Hold for 3 seconds",
                "Round back (cat position) - tuck chin",
                "Hold for 3 seconds",
                "Repeat 10 cycles"
            ],
            benefits: [
                "Mobilizes entire spine",
                "Reduces stiffness",
                "Warms up spinal muscles"
            ],
            safetyTips: [
                "Move slowly between positions",
                "Keep movements gentle",
                "Breathe deeply throughout"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Hip Flexor Stretch",
            category: .stretching,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Hips", "Lower Back"],
            instructions: [
                "Kneel on right knee, left foot forward",
                "Keep back straight",
                "Push hips forward gently",
                "Hold for 30 seconds",
                "Switch sides",
                "Repeat 3 times each side"
            ],
            benefits: [
                "Reduces hip stiffness",
                "Improves posture",
                "Decreases lower back strain"
            ],
            safetyTips: [
                "Don't bounce",
                "Keep pelvis neutral",
                "Use cushion under knee"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Deep Breathing Exercise",
            category: .breathing,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Chest", "Rib Cage"],
            instructions: [
                "Sit or lie comfortably",
                "Place hands on ribs",
                "Breathe in deeply through nose for 4 counts",
                "Feel ribs expand",
                "Exhale slowly through mouth for 6 counts",
                "Repeat for 5 minutes"
            ],
            benefits: [
                "Maintains chest expansion",
                "Prevents fusion complications",
                "Reduces anxiety"
            ],
            safetyTips: [
                "Don't hyperventilate",
                "Stop if dizzy",
                "Practice regularly"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Wall Angels",
            category: .posture,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Shoulders", "Upper Back", "Thoracic Spine"],
            instructions: [
                "Stand with back against wall",
                "Press lower back to wall",
                "Raise arms to 90 degrees (goal post position)",
                "Keep elbows and hands touching wall",
                "Slowly raise arms overhead",
                "Return to start position",
                "Repeat 15 times"
            ],
            benefits: [
                "Improves shoulder mobility",
                "Strengthens upper back",
                "May help improve head posture"
            ],
            safetyTips: [
                "Don't arch lower back",
                "Move within pain-free range",
                "Stop if shoulder pain occurs"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Pelvic Tilts",
            category: .strengthening,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Lower Back", "Pelvis", "Core"],
            instructions: [
                "Lie on back, knees bent",
                "Flatten lower back against floor",
                "Tilt pelvis upward",
                "Hold for 5 seconds",
                "Relax",
                "Repeat 20 times"
            ],
            benefits: [
                "Strengthens core muscles",
                "Improves pelvic mobility",
                "Reduces lower back pain"
            ],
            safetyTips: [
                "Move slowly",
                "Don't hold breath",
                "Keep movements controlled"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Bridge Exercise",
            category: .strengthening,
            difficulty: .intermediate,
            duration: 10,
            targetAreas: ["Glutes", "Lower Back", "Core"],
            instructions: [
                "Lie on back, knees bent, feet flat",
                "Lift hips off ground",
                "Form straight line from knees to shoulders",
                "Hold for 10 seconds",
                "Lower slowly",
                "Repeat 15 times"
            ],
            benefits: [
                "Strengthens glutes and lower back",
                "Improves hip extension",
                "May support spinal mobility"
            ],
            safetyTips: [
                "Don't overarch back",
                "Keep core engaged",
                "Breathe throughout"
            ],
            videoURL: nil
        ),
        // Add more exercises to reach 50+
        Exercise(
            id: UUID(),
            name: "Spinal Twist (Supine)",
            category: .stretching,
            difficulty: .beginner,
            duration: 8,
            targetAreas: ["Thoracic Spine", "Lumbar Spine"],
            instructions: [
                "Lie on back",
                "Bring right knee to chest",
                "Gently guide knee across body to left",
                "Keep shoulders on ground",
                "Hold for 30 seconds",
                "Repeat on other side"
            ],
            benefits: [
                "Improves spinal rotation",
                "Stretches back muscles",
                "Relieves tension"
            ],
            safetyTips: [
                "Don't force the twist",
                "Keep movement gentle",
                "Stop if pain increases"
            ],
            videoURL: nil
        ),
        Exercise(
            id: UUID(),
            name: "Chin Tucks",
            category: .posture,
            difficulty: .beginner,
            duration: 5,
            targetAreas: ["Cervical Spine", "Neck"],
            instructions: [
                "Sit or stand with good posture",
                "Look straight ahead",
                "Gently tuck chin back (double chin)",
                "Hold for 5 seconds",
                "Relax",
                "Repeat 10 times"
            ],
            benefits: [
                "May help improve head posture",
                "Strengthens deep neck flexors",
                "Reduces neck pain"
            ],
            safetyTips: [
                "Don't tilt head down",
                "Keep eyes level",
                "Movement should be small"
            ],
            videoURL: nil
        )
    ]
}

// MARK: - Preview

struct ExerciseLibraryView_Previews: PreviewProvider {
    static var previews: some View {
        ExerciseLibraryView()
    }
}
