//
//  CoachCompositorView.swift
//  InflamAI
//
//  AI-powered exercise routine generator for AS patients
//  Creates personalized routines based on symptoms, mobility, and goals
//

import SwiftUI
import CoreData

struct CoachCompositorView: View {
    @StateObject private var viewModel: CoachCompositorViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var currentStep = 0

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: CoachCompositorViewModel(context: context))
    }

    var body: some View {
        VStack(spacing: 0) {
            // Progress Indicator
            progressBar

            // Content
            TabView(selection: $currentStep) {
                goalSelectionView.tag(0)
                symptomAssessmentView.tag(1)
                mobilityLevelView.tag(2)
                timePreferenceView.tag(3)
                generatedRoutineView.tag(4)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            // Navigation Buttons
            navigationButtons
        }
        .navigationTitle("Exercise Coach")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button("Cancel") {
                    dismiss()
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink {
                    RoutineManagementView()
                } label: {
                    Label(NSLocalizedString("routine.title", comment: ""), systemImage: "list.bullet.rectangle")
                }
            }
        }
    }

    // MARK: - Progress Bar

    private var progressBar: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Rectangle()
                    .fill(Colors.Gray.g200)
                    .frame(height: 4)

                Rectangle()
                    .fill(Colors.Primary.p500)
                    .frame(width: geometry.size.width * progressValue, height: 4)
                    .animation(Animations.easeOut, value: currentStep)
            }
        }
        .frame(height: 4)
    }

    private var progressValue: CGFloat {
        CGFloat(currentStep + 1) / 5.0
    }

    // MARK: - Step 1: Goal Selection

    private var goalSelectionView: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                VStack(spacing: Spacing.sm) {
                    Image(systemName: "target")
                        .font(.system(size: 60))
                        .foregroundColor(Colors.Primary.p500)

                    Text("What's your goal?")
                        .font(.system(size: Typography.xl, weight: .bold))
                        .foregroundColor(Colors.Gray.g900)

                    Text("Select your primary exercise goal")
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }
                .padding(.top, Spacing.xl)

                VStack(spacing: Spacing.sm) {
                    ForEach(ExerciseGoal.allCases, id: \.self) { goal in
                        GoalCard(
                            goal: goal,
                            isSelected: viewModel.selectedGoal == goal
                        ) {
                            viewModel.selectedGoal = goal
                        }
                    }
                }
            }
            .padding(Spacing.md)
        }
    }

    // MARK: - Step 2: Symptom Assessment

    private var symptomAssessmentView: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                VStack(spacing: Spacing.sm) {
                    Image(systemName: "stethoscope")
                        .font(.system(size: 60))
                        .foregroundColor(Colors.Primary.p500)

                    Text("Current Symptoms")
                        .font(.system(size: Typography.xl, weight: .bold))
                        .foregroundColor(Colors.Gray.g900)

                    Text("Select all that apply today")
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }
                .padding(.top, Spacing.xl)

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                    ForEach(CurrentSymptom.allCases, id: \.self) { symptom in
                        SymptomCard(
                            symptom: symptom,
                            isSelected: viewModel.selectedSymptoms.contains(symptom)
                        ) {
                            viewModel.toggleSymptom(symptom)
                        }
                    }
                }
            }
            .padding(Spacing.md)
        }
    }

    // MARK: - Step 3: Mobility Level

    private var mobilityLevelView: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                VStack(spacing: Spacing.sm) {
                    Image(systemName: "figure.walk")
                        .font(.system(size: 60))
                        .foregroundColor(Colors.Primary.p500)

                    Text("Mobility Level")
                        .font(.system(size: Typography.xl, weight: .bold))
                        .foregroundColor(Colors.Gray.g900)

                    Text("How mobile are you today?")
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }
                .padding(.top, Spacing.xl)

                VStack(spacing: Spacing.sm) {
                    ForEach(MobilityLevel.allCases, id: \.self) { level in
                        MobilityCard(
                            level: level,
                            isSelected: viewModel.mobilityLevel == level
                        ) {
                            viewModel.mobilityLevel = level
                        }
                    }
                }
            }
            .padding(Spacing.md)
        }
    }

    // MARK: - Step 4: Time Preference

    private var timePreferenceView: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                VStack(spacing: Spacing.sm) {
                    Image(systemName: "clock")
                        .font(.system(size: 60))
                        .foregroundColor(Colors.Primary.p500)

                    Text("Time Available")
                        .font(.system(size: Typography.xl, weight: .bold))
                        .foregroundColor(Colors.Gray.g900)

                    Text("How long can you exercise?")
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }
                .padding(.top, Spacing.xl)

                VStack(spacing: Spacing.sm) {
                    ForEach(TimePreference.allCases, id: \.self) { time in
                        TimeCard(
                            time: time,
                            isSelected: viewModel.timePreference == time
                        ) {
                            viewModel.timePreference = time
                        }
                    }
                }
            }
            .padding(Spacing.md)
        }
    }

    // MARK: - Step 5: Generated Routine

    private var generatedRoutineView: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                if viewModel.isGenerating {
                    generatingView
                } else if let routine = viewModel.generatedRoutine {
                    routineResultView(routine: routine)
                }
            }
            .padding(Spacing.md)
        }
        .onAppear {
            if viewModel.generatedRoutine == nil {
                viewModel.generateRoutine()
            }
        }
    }

    private var generatingView: some View {
        VStack(spacing: Spacing.lg) {
            ProgressView()
                .scaleEffect(1.5)

            Text("Creating your personalized routine...")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text("Analyzing your needs and selecting optimal exercises")
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.top, 100)
    }

    private func routineResultView(routine: GeneratedRoutine) -> some View {
        VStack(spacing: Spacing.lg) {
            // Success Header
            VStack(spacing: Spacing.sm) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 60))
                    .foregroundColor(Colors.Semantic.success)

                Text("Routine Ready!")
                    .font(.system(size: Typography.xl, weight: .bold))
                    .foregroundColor(Colors.Gray.g900)

                Text("\(routine.exercises.count) exercises • \(routine.totalDuration) min")
                    .font(.system(size: Typography.base))
                    .foregroundColor(Colors.Gray.g500)
            }
            .padding(.top, Spacing.lg)

            // Routine Details
            VStack(alignment: .leading, spacing: Spacing.md) {
                Text("Your Personalized Routine")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                ForEach(Array(routine.exercises.enumerated()), id: \.offset) { index, exercise in
                    HStack(spacing: Spacing.sm) {
                        ZStack {
                            Circle()
                                .fill(Colors.Primary.p50)
                                .frame(width: 40, height: 40)

                            Text("\(index + 1)")
                                .font(.system(size: Typography.base, weight: .bold))
                                .foregroundColor(Colors.Primary.p500)
                        }

                        VStack(alignment: .leading, spacing: Spacing.xxs) {
                            Text(exercise.name)
                                .font(.system(size: Typography.base, weight: .semibold))
                                .foregroundColor(Colors.Gray.g900)

                            HStack {
                                Label("\(exercise.duration) min", systemImage: "clock")
                                Text("•")
                                Label(exercise.difficulty.rawValue, systemImage: "gauge")
                            }
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                        }

                        Spacer()

                        Image(systemName: "chevron.right")
                            .foregroundColor(Colors.Gray.g400)
                    }
                    .padding(Spacing.md)
                    .background(Colors.Gray.g100)
                    .cornerRadius(Radii.lg)
                }
            }
            .padding(Spacing.md)
            .background(Color(.systemBackground))
            .cornerRadius(Radii.xl)
            .dshadow(Shadows.sm)

            // AI Insights
            VStack(alignment: .leading, spacing: Spacing.sm) {
                HStack {
                    Image(systemName: "sparkles")
                        .foregroundColor(Colors.Accent.purple)
                    Text("Coach Insights")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)
                }

                Text(routine.coachNotes)
                    .font(.system(size: Typography.base))
                    .foregroundColor(Colors.Gray.g600)
            }
            .padding(Spacing.md)
            .background(Colors.Accent.purpleLight)
            .cornerRadius(Radii.xl)

            // Action Buttons
            VStack(spacing: Spacing.sm) {
                Button {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    viewModel.saveRoutine()
                    dismiss()
                } label: {
                    Label("Save Routine", systemImage: "square.and.arrow.down")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 48)
                }
                .background(Colors.Primary.p500)
                .cornerRadius(Radii.lg)

                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    viewModel.startRoutineNow()
                    dismiss()
                } label: {
                    Label("Start Now", systemImage: "play.fill")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Primary.p500)
                        .frame(maxWidth: .infinity)
                        .frame(height: 48)
                }
                .background(Colors.Primary.p50)
                .cornerRadius(Radii.lg)

                Button {
                    viewModel.regenerateRoutine()
                } label: {
                    Label("Generate New Routine", systemImage: "arrow.clockwise")
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }
                .padding(.top, Spacing.xs)
            }
            .padding(.bottom, Spacing.lg)
        }
    }

    // MARK: - Navigation

    private var navigationButtons: some View {
        HStack(spacing: Spacing.md) {
            if currentStep > 0 && currentStep < 4 {
                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    withAnimation(Animations.easeOut) {
                        currentStep -= 1
                    }
                } label: {
                    HStack {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                    .font(.system(size: Typography.md, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .background(Colors.Gray.g200)
                    .foregroundColor(Colors.Gray.g900)
                    .cornerRadius(Radii.lg)
                }
            }

            if currentStep < 4 {
                Button {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    withAnimation(Animations.easeOut) {
                        currentStep += 1
                    }
                } label: {
                    HStack {
                        Text(currentStep == 3 ? "Generate" : "Next")
                        Image(systemName: currentStep == 3 ? "sparkles" : "chevron.right")
                    }
                    .font(.system(size: Typography.md, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .background(canProceed ? Colors.Primary.p500 : Colors.Gray.g300)
                    .foregroundColor(.white)
                    .cornerRadius(Radii.lg)
                }
                .disabled(!canProceed)
            }
        }
        .padding(Spacing.md)
    }

    private var canProceed: Bool {
        switch currentStep {
        case 0: return viewModel.selectedGoal != nil
        case 1: return !viewModel.selectedSymptoms.isEmpty
        case 2: return viewModel.mobilityLevel != nil
        case 3: return viewModel.timePreference != nil
        default: return true
        }
    }
}

// MARK: - Supporting Card Views

struct GoalCard: View {
    let goal: ExerciseGoal
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: {
            UISelectionFeedbackGenerator().selectionChanged()
            action()
        }) {
            HStack(spacing: Spacing.md) {
                ZStack {
                    Circle()
                        .fill(isSelected ? Colors.Primary.p500 : Colors.Gray.g200)
                        .frame(width: 56, height: 56)

                    Text(goal.icon)
                        .font(.system(size: 28))
                }

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(goal.rawValue)
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    Text(goal.description)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                        .lineLimit(2)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(Colors.Primary.p500)
                        .font(.title2)
                }
            }
            .padding(Spacing.md)
            .background(isSelected ? Colors.Primary.p50 : Colors.Gray.g100)
            .cornerRadius(Radii.lg)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .stroke(isSelected ? Colors.Primary.p500 : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

struct SymptomCard: View {
    let symptom: CurrentSymptom
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: {
            UISelectionFeedbackGenerator().selectionChanged()
            action()
        }) {
            VStack(spacing: Spacing.sm) {
                Text(symptom.icon)
                    .font(.system(size: 36))

                Text(symptom.rawValue)
                    .font(.system(size: Typography.xs, weight: .semibold))
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, Spacing.lg)
            .background(isSelected ? Colors.Semantic.warning : Colors.Gray.g100)
            .foregroundColor(isSelected ? .white : Colors.Gray.g900)
            .cornerRadius(Radii.lg)
        }
        .buttonStyle(.plain)
    }
}

struct MobilityCard: View {
    let level: MobilityLevel
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: {
            UISelectionFeedbackGenerator().selectionChanged()
            action()
        }) {
            HStack(spacing: Spacing.md) {
                Text(level.icon)
                    .font(.system(size: 36))

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(level.rawValue)
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    Text(level.description)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(Colors.Primary.p500)
                        .font(.title2)
                }
            }
            .padding(Spacing.md)
            .background(isSelected ? Colors.Primary.p50 : Colors.Gray.g100)
            .cornerRadius(Radii.lg)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .stroke(isSelected ? Colors.Primary.p500 : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

struct TimeCard: View {
    let time: TimePreference
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: {
            UISelectionFeedbackGenerator().selectionChanged()
            action()
        }) {
            HStack {
                Image(systemName: "clock")
                    .font(.title2)
                    .foregroundColor(isSelected ? Colors.Primary.p500 : Colors.Gray.g400)

                Text(time.rawValue)
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(Colors.Primary.p500)
                        .font(.title2)
                }
            }
            .padding(Spacing.md)
            .background(isSelected ? Colors.Primary.p50 : Colors.Gray.g100)
            .cornerRadius(Radii.lg)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .stroke(isSelected ? Colors.Primary.p500 : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - View Model

@MainActor
class CoachCompositorViewModel: ObservableObject {
    @Published var selectedGoal: ExerciseGoal?
    @Published var selectedSymptoms: Set<CurrentSymptom> = []
    @Published var mobilityLevel: MobilityLevel?
    @Published var timePreference: TimePreference?
    @Published var generatedRoutine: GeneratedRoutine?
    @Published var isGenerating = false

    private let context: NSManagedObjectContext
    private let allExercises = Exercise.allExercises

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func toggleSymptom(_ symptom: CurrentSymptom) {
        if selectedSymptoms.contains(symptom) {
            selectedSymptoms.remove(symptom)
        } else {
            selectedSymptoms.insert(symptom)
        }
    }

    func generateRoutine() {
        guard let goal = selectedGoal,
              let mobility = mobilityLevel,
              let time = timePreference else { return }

        isGenerating = true

        // Simulate AI processing
        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

            let routine = createPersonalizedRoutine(
                goal: goal,
                symptoms: Array(selectedSymptoms),
                mobility: mobility,
                time: time
            )

            await MainActor.run {
                self.generatedRoutine = routine
                self.isGenerating = false
            }
        }
    }

    func regenerateRoutine() {
        generatedRoutine = nil
        generateRoutine()
    }

    private func createPersonalizedRoutine(
        goal: ExerciseGoal,
        symptoms: [CurrentSymptom],
        mobility: MobilityLevel,
        time: TimePreference
    ) -> GeneratedRoutine {
        var selectedExercises: [Exercise] = []
        let targetDuration = time.minutes
        var currentDuration = 0

        // Filter exercises based on mobility level
        let suitableExercises = allExercises.filter { exercise in
            switch mobility {
            case .limited:
                return exercise.difficulty == .beginner && exercise.duration <= 10
            case .moderate:
                return exercise.difficulty != .advanced
            case .good:
                return true
            }
        }

        // Prioritize exercises based on goal
        let prioritizedExercises = suitableExercises.sorted { ex1, ex2 in
            let score1 = scoreExercise(ex1, for: goal, symptoms: symptoms)
            let score2 = scoreExercise(ex2, for: goal, symptoms: symptoms)
            return score1 > score2
        }

        // Select exercises to fit time preference
        for exercise in prioritizedExercises {
            if currentDuration + exercise.duration <= targetDuration {
                selectedExercises.append(exercise)
                currentDuration += exercise.duration
            }

            if currentDuration >= targetDuration - 5 {
                break
            }
        }

        // Generate coach notes
        let notes = generateCoachNotes(
            goal: goal,
            symptoms: symptoms,
            mobility: mobility,
            exerciseCount: selectedExercises.count
        )

        return GeneratedRoutine(
            exercises: selectedExercises,
            totalDuration: currentDuration,
            coachNotes: notes,
            createdDate: Date()
        )
    }

    private func scoreExercise(_ exercise: Exercise, for goal: ExerciseGoal, symptoms: [CurrentSymptom]) -> Int {
        var score = 0

        // Score based on goal alignment
        switch goal {
        case .flexibility:
            if exercise.category == .stretching || exercise.category == .mobility {
                score += 10
            }
        case .strength:
            if exercise.category == .strengthening {
                score += 10
            }
        case .pain:
            if exercise.category == .stretching || exercise.category == .mobility {
                score += 8
            }
            if exercise.category == .breathing {
                score += 5
            }
        case .posture:
            if exercise.category == .posture || exercise.category == .strengthening {
                score += 10
            }
        case .balance:
            if exercise.category == .balance {
                score += 10
            }
        case .breathing:
            if exercise.category == .breathing {
                score += 10
            }
        }

        // Adjust based on symptoms
        for symptom in symptoms {
            switch symptom {
            case .neckPain:
                if exercise.targetAreas.contains(where: { $0.contains("Cervical") || $0.contains("Neck") }) {
                    score += 5
                }
            case .backStiffness:
                if exercise.category == .mobility || exercise.category == .stretching {
                    score += 5
                }
            case .hipPain:
                if exercise.targetAreas.contains(where: { $0.contains("Hip") }) {
                    score += 5
                }
            case .chestTightness:
                if exercise.category == .breathing {
                    score += 8
                }
            case .fatigue:
                if exercise.difficulty == .beginner {
                    score += 3
                }
            case .morningStiffness:
                if exercise.category == .mobility {
                    score += 5
                }
            }
        }

        return score
    }

    private func generateCoachNotes(
        goal: ExerciseGoal,
        symptoms: [CurrentSymptom],
        mobility: MobilityLevel,
        exerciseCount: Int
    ) -> String {
        var notes = "This routine has been personalized for your "

        switch goal {
        case .flexibility:
            notes += "flexibility goals with focus on gentle stretching and mobility work."
        case .strength:
            notes += "strength goals with progressive exercises to build core and back strength."
        case .pain:
            notes += "pain management with gentle movements to reduce discomfort."
        case .posture:
            notes += "posture awareness with mobility exercises for the back."
        case .balance:
            notes += "balance enhancement with stability-focused movements."
        case .breathing:
            notes += "respiratory health with chest expansion and breathing techniques."
        }

        if !symptoms.isEmpty {
            notes += " The exercises address your current symptoms"
            if symptoms.contains(.morningStiffness) {
                notes += ", particularly focusing on relieving morning stiffness"
            }
            notes += "."
        }

        switch mobility {
        case .limited:
            notes += " All exercises are beginner-friendly and can be done with limited mobility."
        case .moderate:
            notes += " Exercises are tailored to your moderate mobility level."
        case .good:
            notes += " This routine includes some challenging exercises to match your good mobility."
        }

        notes += " Remember to move within your comfortable range and stop if you experience sharp pain."

        return notes
    }

    func saveRoutine() {
        guard let routine = generatedRoutine else { return }

        Task {
            await context.perform {
                // Save routine to UserRoutine so it appears in saved routines
                let savedRoutine = UserRoutine(context: self.context)
                savedRoutine.id = UUID()
                savedRoutine.name = self.generateRoutineName()
                savedRoutine.totalDuration = Int16(routine.totalDuration)
                savedRoutine.customNotes = routine.coachNotes
                savedRoutine.createdAt = Date()
                savedRoutine.isActive = false
                savedRoutine.timesCompleted = 0

                // Encode exercises in the format expected by RoutineDetailView
                struct RoutineExerciseItem: Codable {
                    let id: UUID
                    let exerciseId: String
                    let duration: Int
                    let order: Int
                }

                let exerciseItems = routine.exercises.enumerated().map { index, exercise in
                    RoutineExerciseItem(
                        id: UUID(),
                        exerciseId: exercise.name, // Store name for demo compatibility
                        duration: exercise.duration,
                        order: index
                    )
                }

                if let exerciseData = try? JSONEncoder().encode(exerciseItems) {
                    savedRoutine.exercises = exerciseData
                }

                do {
                    try self.context.save()
                    print("✅ Saved coach routine '\(savedRoutine.name ?? "")' to UserRoutine")
                } catch {
                    print("❌ CRITICAL: Failed to save coach routine: \(error)")
                }
            }
        }
    }

    private func generateRoutineName() -> String {
        let goalName: String
        switch selectedGoal {
        case .flexibility: goalName = "Flexibility"
        case .strength: goalName = "Strength"
        case .pain: goalName = "Pain Relief"
        case .posture: goalName = "Posture"
        case .balance: goalName = "Balance"
        case .breathing: goalName = "Breathing"
        case .none: goalName = "Custom"
        }

        let timeStr = timePreference?.rawValue.components(separatedBy: " ").first ?? ""
        return "\(goalName) Routine (\(timeStr) min)"
    }

    func startRoutineNow() {
        // This would trigger the exercise session view
        // Implementation would depend on exercise session tracking system
        print("Starting routine now...")
    }
}

// MARK: - Models

enum ExerciseGoal: String, CaseIterable {
    case flexibility = "Improve Flexibility"
    case strength = "Build Strength"
    case pain = "Reduce Pain"
    case posture = "Better Posture"
    case balance = "Improve Balance"
    case breathing = "Breathing Capacity"

    var icon: String {
        switch self {
        case .flexibility: return "🤸"
        case .strength: return "💪"
        case .pain: return "🩹"
        case .posture: return "🧍"
        case .balance: return "⚖️"
        case .breathing: return "🫁"
        }
    }

    var description: String {
        switch self {
        case .flexibility: return "Increase range of motion and reduce stiffness"
        case .strength: return "Build core and back strength for better support"
        case .pain: return "Gentle exercises to manage and reduce pain"
        case .posture: return "Correct alignment and prevent forward stooping"
        case .balance: return "Enhance stability and prevent falls"
        case .breathing: return "Maintain chest expansion and lung function"
        }
    }
}

enum CurrentSymptom: String, CaseIterable {
    case neckPain = "Neck Pain"
    case backStiffness = "Back Stiffness"
    case hipPain = "Hip Pain"
    case chestTightness = "Chest Tight"
    case fatigue = "Fatigue"
    case morningStiffness = "Morning Stiff"

    var icon: String {
        switch self {
        case .neckPain: return "😣"
        case .backStiffness: return "🔒"
        case .hipPain: return "🦴"
        case .chestTightness: return "🫁"
        case .fatigue: return "😴"
        case .morningStiffness: return "🌅"
        }
    }
}

enum MobilityLevel: String, CaseIterable {
    case limited = "Limited Mobility"
    case moderate = "Moderate Mobility"
    case good = "Good Mobility"

    var icon: String {
        switch self {
        case .limited: return "🐌"
        case .moderate: return "🚶"
        case .good: return "🏃"
        }
    }

    var description: String {
        switch self {
        case .limited: return "Difficulty with basic movements, need gentle exercises"
        case .moderate: return "Can do most activities with some limitations"
        case .good: return "Good range of motion, ready for challenges"
        }
    }
}

enum TimePreference: String, CaseIterable {
    case quick = "5-10 minutes"
    case short = "10-15 minutes"
    case medium = "15-20 minutes"
    case long = "20-30 minutes"

    var minutes: Int {
        switch self {
        case .quick: return 10
        case .short: return 15
        case .medium: return 20
        case .long: return 30
        }
    }
}

struct GeneratedRoutine {
    let exercises: [Exercise]
    let totalDuration: Int
    let coachNotes: String
    let createdDate: Date
}

// MARK: - Preview

struct CoachCompositorView_Previews: PreviewProvider {
    static var previews: some View {
        CoachCompositorView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
