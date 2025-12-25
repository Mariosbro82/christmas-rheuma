//
//  OnboardingRedesignFlow.swift
//  InflamAI
//
//  Complete Onboarding Redesign with Psychology-Driven Personalization
//
//  Flow Structure:
//  1. Animated Splash (Anky wakes up)
//  2. Meet Anky (companion introduction)
//  3. User Type (patient vs caregiver)
//  4. Personalization Deep-Dive:
//     - AS History & Diagnosis
//     - Current Medications
//     - Goals & Expectations
//     - Known Triggers
//  5. Guided Feature Experience:
//     - Quick Log Tutorial
//     - Body Map Introduction (IMPORTANT!)
//     - Flare Capture Introduction
//  6. Notification Setup (brief)
//  7. Ready to Go (celebration)
//
//  Psychology: Heavy personalization creates investment.
//  The more someone tells you about themselves, the more committed they become.
//  This is the Ikea Effect applied to app onboarding.
//

import SwiftUI
import Combine

// MARK: - Main Onboarding Flow

struct OnboardingRedesignFlow: View {
    @StateObject private var viewModel = OnboardingRedesignViewModel()
    @State private var showSplash = true

    var body: some View {
        ZStack {
            // Background - consistent throughout
            backgroundGradient

            if showSplash {
                AnimatedSplashScreen {
                    withAnimation(.easeInOut(duration: 0.5)) {
                        showSplash = false
                    }
                }
                .transition(.opacity)
            } else {
                // Main onboarding content
                onboardingContent
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.95)),
                        removal: .opacity
                    ))
            }
        }
        .ignoresSafeArea()
    }

    // MARK: - Background

    private var backgroundGradient: some View {
        LinearGradient(
            colors: [
                Color(hex: "#F8FAFC"),
                Color(hex: "#EFF6FF"),
                Color(hex: "#F0FDFA")
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    // MARK: - Main Content

    private var onboardingContent: some View {
        VStack(spacing: 0) {
            // Progress indicator (always visible)
            OnboardingProgressBar(
                currentStage: viewModel.currentStage,
                totalStages: OnboardingStage.allCases.count
            )
            .padding(.horizontal, Spacing.lg)
            .padding(.top, 60)

            // Stage content with Anky always present
            TabView(selection: $viewModel.currentStage) {
                ForEach(OnboardingStage.allCases, id: \.self) { stage in
                    stageView(for: stage)
                        .tag(stage)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .animation(.spring(response: 0.5, dampingFraction: 0.8), value: viewModel.currentStage)
        }
    }

    // MARK: - Stage Router

    @ViewBuilder
    private func stageView(for stage: OnboardingStage) -> some View {
        switch stage {
        case .meetAnky:
            Stage1MeetAnkyView(viewModel: viewModel)
        case .userType:
            Stage2UserTypeView(viewModel: viewModel)
        case .asHistory:
            Stage3ASHistoryView(viewModel: viewModel)
        case .medications:
            Stage4MedicationsView(viewModel: viewModel)
        case .goals:
            Stage5GoalsView(viewModel: viewModel)
        case .triggers:
            Stage6TriggersView(viewModel: viewModel)
        case .quickLogTutorial:
            Stage7QuickLogTutorialView(viewModel: viewModel)
        case .bodyMapTutorial:
            Stage8BodyMapTutorialView(viewModel: viewModel)
        case .flareTutorial:
            Stage9FlareTutorialView(viewModel: viewModel)
        case .notifications:
            Stage10NotificationsView(viewModel: viewModel)
        case .ready:
            Stage11ReadyView(viewModel: viewModel)
        }
    }
}

// MARK: - Onboarding Stages

enum OnboardingStage: Int, CaseIterable {
    case meetAnky = 0
    case userType
    case asHistory
    case medications
    case goals
    case triggers
    case quickLogTutorial
    case bodyMapTutorial
    case flareTutorial
    case notifications
    case ready

    var title: String {
        switch self {
        case .meetAnky: return "Welcome"
        case .userType: return "About You"
        case .asHistory: return "Your Journey"
        case .medications: return "Medications"
        case .goals: return "Your Goals"
        case .triggers: return "Triggers"
        case .quickLogTutorial: return "Quick Log"
        case .bodyMapTutorial: return "Body Map"
        case .flareTutorial: return "Flare Capture"
        case .notifications: return "Stay Updated"
        case .ready: return "All Set!"
        }
    }

    var isPersonalization: Bool {
        switch self {
        case .userType, .asHistory, .medications, .goals, .triggers:
            return true
        default:
            return false
        }
    }

    var isTutorial: Bool {
        switch self {
        case .quickLogTutorial, .bodyMapTutorial, .flareTutorial:
            return true
        default:
            return false
        }
    }
}

// MARK: - View Model

@MainActor
class OnboardingRedesignViewModel: ObservableObject {
    // Navigation
    @Published var currentStage: OnboardingStage = .meetAnky

    // User Type
    @Published var userType: UserType = .patient

    // AS History
    @Published var diagnosisYear: Int = Calendar.current.component(.year, from: Date())
    @Published var diagnosisStatus: DiagnosisStatus = .diagnosed
    @Published var yearsWithSymptoms: Int = 1

    // Medications
    @Published var selectedMedications: Set<ASMedication> = []
    @Published var customMedication: String = ""

    // Goals
    @Published var selectedGoals: Set<AppGoal> = []

    // Triggers
    @Published var knownTriggers: Set<KnownTrigger> = []

    // Notifications
    @Published var enableDailyReminder = true
    @Published var enableFlareAlerts = true
    @Published var reminderTime = Calendar.current.date(from: DateComponents(hour: 9, minute: 0)) ?? Date()

    // Tutorial Progress
    @Published var quickLogCompleted = false
    @Published var bodyMapExplored = false
    @Published var flareCaptureSeen = false

    // UI State
    @Published var ankyState: AnkyState = .happy
    @Published var isAnimating = false

    // MARK: - Navigation

    func nextStage() {
        guard let currentIndex = OnboardingStage.allCases.firstIndex(of: currentStage),
              currentIndex < OnboardingStage.allCases.count - 1 else {
            completeOnboarding()
            return
        }

        withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
            currentStage = OnboardingStage.allCases[currentIndex + 1]
        }

        // Update Anky state based on stage
        updateAnkyForStage()
        HapticFeedback.selection()
    }

    func previousStage() {
        guard let currentIndex = OnboardingStage.allCases.firstIndex(of: currentStage),
              currentIndex > 0 else { return }

        withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
            currentStage = OnboardingStage.allCases[currentIndex - 1]
        }
        updateAnkyForStage()
    }

    func skipToStage(_ stage: OnboardingStage) {
        withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
            currentStage = stage
        }
        updateAnkyForStage()
    }

    private func updateAnkyForStage() {
        switch currentStage {
        case .meetAnky:
            ankyState = .waving
        case .userType:
            ankyState = .curious
        case .asHistory, .medications:
            ankyState = .attentive
        case .goals:
            ankyState = .encouraging
        case .triggers:
            ankyState = .sympathetic
        case .quickLogTutorial, .bodyMapTutorial, .flareTutorial:
            ankyState = .explaining
        case .notifications:
            ankyState = .thinking
        case .ready:
            ankyState = .celebrating
        }
    }

    // MARK: - Completion

    func completeOnboarding() {
        // Save personalization data
        savePersonalizationData()

        // Mark onboarding as complete
        UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")

        // Haptic celebration
        HapticFeedback.success()

        // Post notification
        NotificationCenter.default.post(name: .onboardingCompleted, object: nil)
    }

    private func savePersonalizationData() {
        let defaults = UserDefaults.standard

        // User type
        defaults.set(userType.rawValue, forKey: "userType")

        // AS History
        defaults.set(diagnosisYear, forKey: "diagnosisYear")
        defaults.set(diagnosisStatus.rawValue, forKey: "diagnosisStatus")
        defaults.set(yearsWithSymptoms, forKey: "yearsWithSymptoms")

        // Medications (store IDs)
        let medIds = selectedMedications.map { $0.id }
        defaults.set(medIds, forKey: "selectedMedications")

        // Goals
        let goalIds = selectedGoals.map { $0.rawValue }
        defaults.set(goalIds, forKey: "selectedGoals")

        // Triggers
        let triggerIds = knownTriggers.map { $0.rawValue }
        defaults.set(triggerIds, forKey: "knownTriggers")

        // Notifications
        defaults.set(enableDailyReminder, forKey: "enableDailyReminder")
        defaults.set(enableFlareAlerts, forKey: "enableFlareAlerts")
    }
}

// MARK: - Data Models

enum UserType: String, CaseIterable {
    case patient = "patient"
    case caregiver = "caregiver"

    var title: String {
        switch self {
        case .patient: return "I have AS"
        case .caregiver: return "I'm supporting someone"
        }
    }

    var description: String {
        switch self {
        case .patient: return "Track your symptoms, medications, and patterns"
        case .caregiver: return "Help your loved one manage their condition"
        }
    }

    var icon: String {
        switch self {
        case .patient: return "person.fill"
        case .caregiver: return "heart.circle.fill"
        }
    }
}

enum DiagnosisStatus: String, CaseIterable {
    case diagnosed = "diagnosed"
    case suspected = "suspected"
    case newlyDiagnosed = "newly_diagnosed"

    var title: String {
        switch self {
        case .diagnosed: return "Diagnosed with AS"
        case .suspected: return "Suspected AS"
        case .newlyDiagnosed: return "Recently diagnosed"
        }
    }
}

struct ASMedication: Identifiable, Hashable {
    let id: String
    let name: String
    let category: MedicationCategory
    let isCommon: Bool

    enum MedicationCategory: String {
        case biologic = "Biologics"
        case nsaid = "NSAIDs"
        case dmard = "DMARDs"
        case painRelief = "Pain Relief"
        case other = "Other"
    }

    static let commonMedications: [ASMedication] = [
        // Biologics (most common for AS)
        ASMedication(id: "humira", name: "Humira (Adalimumab)", category: .biologic, isCommon: true),
        ASMedication(id: "enbrel", name: "Enbrel (Etanercept)", category: .biologic, isCommon: true),
        ASMedication(id: "cosentyx", name: "Cosentyx (Secukinumab)", category: .biologic, isCommon: true),
        ASMedication(id: "taltz", name: "Taltz (Ixekizumab)", category: .biologic, isCommon: true),
        ASMedication(id: "remicade", name: "Remicade (Infliximab)", category: .biologic, isCommon: true),
        ASMedication(id: "simponi", name: "Simponi (Golimumab)", category: .biologic, isCommon: true),

        // NSAIDs
        ASMedication(id: "naproxen", name: "Naproxen", category: .nsaid, isCommon: true),
        ASMedication(id: "ibuprofen", name: "Ibuprofen", category: .nsaid, isCommon: true),
        ASMedication(id: "diclofenac", name: "Diclofenac", category: .nsaid, isCommon: true),
        ASMedication(id: "celecoxib", name: "Celecoxib (Celebrex)", category: .nsaid, isCommon: true),
        ASMedication(id: "indomethacin", name: "Indomethacin", category: .nsaid, isCommon: true),

        // DMARDs
        ASMedication(id: "methotrexate", name: "Methotrexate", category: .dmard, isCommon: false),
        ASMedication(id: "sulfasalazine", name: "Sulfasalazine", category: .dmard, isCommon: false),

        // Pain Relief
        ASMedication(id: "paracetamol", name: "Paracetamol / Acetaminophen", category: .painRelief, isCommon: false),
        ASMedication(id: "tramadol", name: "Tramadol", category: .painRelief, isCommon: false),
    ]
}

enum AppGoal: String, CaseIterable {
    case trackSymptoms = "track_symptoms"
    case findPatterns = "find_patterns"
    case medicationReminders = "med_reminders"
    case doctorReports = "doctor_reports"
    case exerciseGuidance = "exercise"
    case understandTriggers = "triggers"
    case connectWithOthers = "community"
    case manageFlares = "flares"

    var title: String {
        switch self {
        case .trackSymptoms: return "Track daily symptoms"
        case .findPatterns: return "Find patterns in my data"
        case .medicationReminders: return "Medication reminders"
        case .doctorReports: return "Generate reports for my doctor"
        case .exerciseGuidance: return "Get exercise guidance"
        case .understandTriggers: return "Understand my triggers"
        case .connectWithOthers: return "Connect with others"
        case .manageFlares: return "Manage flare-ups better"
        }
    }

    var icon: String {
        switch self {
        case .trackSymptoms: return "chart.line.uptrend.xyaxis"
        case .findPatterns: return "brain.head.profile"
        case .medicationReminders: return "pills.fill"
        case .doctorReports: return "doc.text.fill"
        case .exerciseGuidance: return "figure.run"
        case .understandTriggers: return "exclamationmark.triangle.fill"
        case .connectWithOthers: return "person.2.fill"
        case .manageFlares: return "flame.fill"
        }
    }
}

enum KnownTrigger: String, CaseIterable {
    case weather = "weather"
    case stress = "stress"
    case lackOfSleep = "sleep"
    case physicalActivity = "activity"
    case diet = "diet"
    case alcohol = "alcohol"
    case infection = "infection"
    case unknown = "unknown"

    var title: String {
        switch self {
        case .weather: return "Weather changes"
        case .stress: return "Stress"
        case .lackOfSleep: return "Lack of sleep"
        case .physicalActivity: return "Physical activity"
        case .diet: return "Certain foods"
        case .alcohol: return "Alcohol"
        case .infection: return "Infections/illness"
        case .unknown: return "Not sure yet"
        }
    }

    var icon: String {
        switch self {
        case .weather: return "cloud.rain.fill"
        case .stress: return "brain.head.profile"
        case .lackOfSleep: return "moon.zzz.fill"
        case .physicalActivity: return "figure.run"
        case .diet: return "fork.knife"
        case .alcohol: return "wineglass.fill"
        case .infection: return "allergens"
        case .unknown: return "questionmark.circle.fill"
        }
    }
}

// MARK: - Progress Bar

struct OnboardingProgressBar: View {
    let currentStage: OnboardingStage
    let totalStages: Int

    private var progress: CGFloat {
        CGFloat(currentStage.rawValue + 1) / CGFloat(totalStages)
    }

    var body: some View {
        VStack(spacing: Spacing.xs) {
            // Section labels
            HStack {
                Text(currentStage.title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(Colors.Gray.g600)

                Spacer()

                Text("\(currentStage.rawValue + 1)/\(totalStages)")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Colors.Gray.g400)
            }

            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background track
                    Capsule()
                        .fill(Colors.Gray.g200)
                        .frame(height: 4)

                    // Progress fill
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [Colors.Accent.teal, Colors.Primary.p500],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * progress, height: 4)
                        .animation(.spring(response: 0.4, dampingFraction: 0.7), value: progress)
                }
            }
            .frame(height: 4)
        }
    }
}

// MARK: - Stage 1: Meet Anky

struct Stage1MeetAnkyView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            // Anky mascot - center stage (Rive animation with Canvas fallback)
            AnkyRiveView(size: 200, state: .waving, showShadow: true)
                .scaleEffect(showContent ? 1 : 0.5)
                .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.xl)

            // Speech bubble
            VStack(spacing: Spacing.md) {
                Text("Hi there! I'm Anky 👋")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text("I'll be your companion on this journey. Living with AS isn't easy, but together we'll make it more manageable.")
                    .font(.system(size: 17, weight: .regular))
                    .foregroundColor(Colors.Gray.g600)
                    .multilineTextAlignment(.center)
                    .lineSpacing(4)
            }
            .padding(.horizontal, Spacing.xl)
            .opacity(showContent ? 1 : 0)
            .offset(y: showContent ? 0 : 20)

            Spacer()

            // Continue button
            OnboardingPrimaryButton(title: "Let's get started") {
                viewModel.nextStage()
            }
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
            .opacity(showContent ? 1 : 0)
        }
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7).delay(0.3)) {
                showContent = true
            }
        }
    }
}

// MARK: - Stage 2: User Type

struct Stage2UserTypeView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false

    var body: some View {
        VStack(spacing: 0) {
            // Anky at top (smaller, but always present)
            AnkyRiveView(size: 100, state: .curious, showShadow: false)
                .padding(.top, Spacing.lg)

            // Question
            VStack(spacing: Spacing.sm) {
                Text("First, tell me about you")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text("This helps me personalize your experience")
                    .font(.system(size: 15))
                    .foregroundColor(Colors.Gray.g500)
            }
            .padding(.top, Spacing.md)
            .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.xl)

            // User type cards
            VStack(spacing: Spacing.md) {
                ForEach(UserType.allCases, id: \.self) { type in
                    UserTypeCard(
                        type: type,
                        isSelected: viewModel.userType == type
                    ) {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            viewModel.userType = type
                        }
                        HapticFeedback.selection()
                    }
                    .staggeredListItem(index: UserType.allCases.firstIndex(of: type) ?? 0, total: 2)
                }
            }
            .padding(.horizontal, Spacing.xl)
            .opacity(showContent ? 1 : 0)

            Spacer()

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

struct UserTypeCard: View {
    let type: UserType
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: Spacing.md) {
                // Icon
                ZStack {
                    Circle()
                        .fill(isSelected ? Colors.Accent.teal.opacity(0.15) : Colors.Gray.g100)
                        .frame(width: 56, height: 56)

                    Image(systemName: type.icon)
                        .font(.system(size: 24))
                        .foregroundColor(isSelected ? Colors.Accent.teal : Colors.Gray.g500)
                }

                // Text
                VStack(alignment: .leading, spacing: 4) {
                    Text(type.title)
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    Text(type.description)
                        .font(.system(size: 14))
                        .foregroundColor(Colors.Gray.g500)
                }

                Spacer()

                // Selection indicator
                ZStack {
                    Circle()
                        .stroke(isSelected ? Colors.Accent.teal : Colors.Gray.g300, lineWidth: 2)
                        .frame(width: 24, height: 24)

                    if isSelected {
                        Circle()
                            .fill(Colors.Accent.teal)
                            .frame(width: 14, height: 14)
                    }
                }
            }
            .padding(Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .fill(Color.white)
                    .shadow(color: isSelected ? Colors.Accent.teal.opacity(0.15) : Color.black.opacity(0.05), radius: isSelected ? 12 : 8)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .stroke(isSelected ? Colors.Accent.teal : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

// MARK: - Stage 3: AS History

struct Stage3ASHistoryView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false

    private let currentYear = Calendar.current.component(.year, from: Date())
    private var years: [Int] {
        Array((currentYear - 50)...currentYear).reversed()
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView(showsIndicators: false) {
                VStack(spacing: 0) {
                    // Anky
                    AnkyRiveView(size: 90, state: .attentive, showShadow: false)
                        .padding(.top, Spacing.md)

                    // Title
                    VStack(spacing: Spacing.sm) {
                        Text("Your AS Journey")
                            .font(.system(size: 24, weight: .bold, design: .rounded))
                            .foregroundColor(Colors.Gray.g900)

                        Text("Help me understand your history")
                            .font(.system(size: 15))
                            .foregroundColor(Colors.Gray.g500)
                    }
                    .padding(.top, Spacing.md)
                    .opacity(showContent ? 1 : 0)

                    Spacer().frame(height: Spacing.xl)

                    // Questions
                    VStack(spacing: Spacing.lg) {
                        // Diagnosis status
                        PersonalizationSection(title: "Where are you in your journey?") {
                            VStack(spacing: Spacing.sm) {
                                ForEach(DiagnosisStatus.allCases, id: \.self) { status in
                                    SelectableChip(
                                        title: status.title,
                                        isSelected: viewModel.diagnosisStatus == status
                                    ) {
                                        viewModel.diagnosisStatus = status
                                    }
                                }
                            }
                        }

                        // Diagnosis year
                        if viewModel.diagnosisStatus != .suspected {
                            PersonalizationSection(title: "When were you diagnosed?") {
                                YearPicker(selectedYear: $viewModel.diagnosisYear, years: years)
                            }
                        }

                        // Years with symptoms
                        PersonalizationSection(title: "How long have you had symptoms?") {
                            SymptomYearsPicker(years: $viewModel.yearsWithSymptoms)
                        }
                    }
                    .padding(.horizontal, Spacing.xl)
                    .opacity(showContent ? 1 : 0)

                    Spacer().frame(height: Spacing.xxl)
                }
            }

            // Navigation - pinned at bottom
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

// MARK: - Stage 4: Medications

struct Stage4MedicationsView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false
    @State private var searchText = ""

    private var filteredMedications: [ASMedication] {
        if searchText.isEmpty {
            return ASMedication.commonMedications
        }
        return ASMedication.commonMedications.filter {
            $0.name.localizedCaseInsensitiveContains(searchText)
        }
    }

    private var groupedMedications: [ASMedication.MedicationCategory: [ASMedication]] {
        Dictionary(grouping: filteredMedications, by: { $0.category })
    }

    var body: some View {
        VStack(spacing: 0) {
            // Anky
            AnkyRiveView(size: 80, state: .attentive, showShadow: false)
                .padding(.top, Spacing.sm)

            // Title
            VStack(spacing: Spacing.sm) {
                Text("Current Medications")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text("Select any medications you're taking")
                    .font(.system(size: 15))
                    .foregroundColor(Colors.Gray.g500)
            }
            .padding(.top, Spacing.sm)
            .opacity(showContent ? 1 : 0)

            // Search
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(Colors.Gray.g400)
                TextField("Search medications...", text: $searchText)
                    .font(.system(size: 16))
            }
            .padding(Spacing.sm)
            .background(Colors.Gray.g100)
            .cornerRadius(Radii.md)
            .padding(.horizontal, Spacing.xl)
            .padding(.top, Spacing.md)

            // Medications list
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: Spacing.lg) {
                    // Selected count
                    if !viewModel.selectedMedications.isEmpty {
                        HStack {
                            Text("\(viewModel.selectedMedications.count) selected")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(Colors.Accent.teal)

                            Spacer()

                            Button("Clear all") {
                                viewModel.selectedMedications.removeAll()
                            }
                            .font(.system(size: 14))
                            .foregroundColor(Colors.Gray.g500)
                        }
                        .padding(.horizontal, Spacing.xl)
                    }

                    // Grouped medications
                    ForEach([ASMedication.MedicationCategory.biologic, .nsaid, .dmard, .painRelief], id: \.self) { category in
                        if let meds = groupedMedications[category], !meds.isEmpty {
                            VStack(alignment: .leading, spacing: Spacing.sm) {
                                Text(category.rawValue)
                                    .font(.system(size: 13, weight: .semibold))
                                    .foregroundColor(Colors.Gray.g500)
                                    .padding(.horizontal, Spacing.xl)

                                OnboardingFlowLayout(spacing: Spacing.sm) {
                                    ForEach(meds) { med in
                                        MedicationChip(
                                            medication: med,
                                            isSelected: viewModel.selectedMedications.contains(med)
                                        ) {
                                            if viewModel.selectedMedications.contains(med) {
                                                viewModel.selectedMedications.remove(med)
                                            } else {
                                                viewModel.selectedMedications.insert(med)
                                            }
                                            HapticFeedback.selection()
                                        }
                                    }
                                }
                                .padding(.horizontal, Spacing.xl)
                            }
                        }
                    }

                    // Skip option
                    Button {
                        viewModel.nextStage()
                    } label: {
                        Text("I'm not currently on medication")
                            .font(.system(size: 15))
                            .foregroundColor(Colors.Gray.g500)
                            .underline()
                    }
                    .padding(.horizontal, Spacing.xl)
                    .padding(.top, Spacing.sm)
                }
                .padding(.top, Spacing.md)
                .padding(.bottom, Spacing.xxl)
            }
            .opacity(showContent ? 1 : 0)

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

struct MedicationChip: View {
    let medication: ASMedication
    let isSelected: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: Spacing.xs) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                }
                Text(medication.name)
                    .font(.system(size: 14, weight: isSelected ? .semibold : .regular))
            }
            .padding(.horizontal, Spacing.sm)
            .padding(.vertical, Spacing.xs)
            .background(isSelected ? Colors.Accent.teal.opacity(0.15) : Colors.Gray.g100)
            .foregroundColor(isSelected ? Colors.Accent.teal : Colors.Gray.g700)
            .cornerRadius(Radii.full)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.full)
                    .stroke(isSelected ? Colors.Accent.teal : Color.clear, lineWidth: 1.5)
            )
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

// MARK: - Stage 5: Goals

struct Stage5GoalsView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false

    var body: some View {
        VStack(spacing: 0) {
            // Anky
            AnkyRiveView(size: 90, state: .encouraging, showShadow: false)
                .padding(.top, Spacing.md)

            // Title
            VStack(spacing: Spacing.sm) {
                Text("What brings you here?")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text("Select all that apply")
                    .font(.system(size: 15))
                    .foregroundColor(Colors.Gray.g500)
            }
            .padding(.top, Spacing.md)
            .opacity(showContent ? 1 : 0)

            // Goals grid
            ScrollView(showsIndicators: false) {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.md) {
                    ForEach(AppGoal.allCases, id: \.self) { goal in
                        OnboardingGoalCard(
                            goal: goal,
                            isSelected: viewModel.selectedGoals.contains(goal)
                        ) {
                            if viewModel.selectedGoals.contains(goal) {
                                viewModel.selectedGoals.remove(goal)
                            } else {
                                viewModel.selectedGoals.insert(goal)
                            }
                            HapticFeedback.selection()
                        }
                    }
                }
                .padding(.horizontal, Spacing.xl)
                .padding(.top, Spacing.lg)
                .padding(.bottom, Spacing.xxl)
            }
            .opacity(showContent ? 1 : 0)

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true,
                nextEnabled: !viewModel.selectedGoals.isEmpty
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

struct OnboardingGoalCard: View {
    let goal: AppGoal
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            VStack(spacing: Spacing.sm) {
                ZStack {
                    Circle()
                        .fill(isSelected ? Colors.Primary.p100 : Colors.Gray.g100)
                        .frame(width: 48, height: 48)

                    Image(systemName: goal.icon)
                        .font(.system(size: 22))
                        .foregroundColor(isSelected ? Colors.Primary.p500 : Colors.Gray.g500)
                }

                Text(goal.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(Colors.Gray.g800)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .minimumScaleFactor(0.9)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, Spacing.md)
            .padding(.horizontal, Spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .fill(Color.white)
                    .shadow(color: Color.black.opacity(0.05), radius: 8)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .stroke(isSelected ? Colors.Primary.p500 : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

// MARK: - Stage 6: Triggers

struct Stage6TriggersView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false

    var body: some View {
        VStack(spacing: 0) {
            // Anky - sympathetic
            AnkyRiveView(size: 90, state: .sympathetic, showShadow: false)
                .padding(.top, Spacing.md)

            // Title
            VStack(spacing: Spacing.sm) {
                Text("Known Triggers")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text("What tends to worsen your symptoms?")
                    .font(.system(size: 15))
                    .foregroundColor(Colors.Gray.g500)
            }
            .padding(.top, Spacing.md)
            .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.xl)

            // Triggers
            ScrollView(showsIndicators: false) {
                VStack(spacing: Spacing.sm) {
                    ForEach(KnownTrigger.allCases, id: \.self) { trigger in
                        OnboardingTriggerRow(
                            trigger: trigger,
                            isSelected: viewModel.knownTriggers.contains(trigger)
                        ) {
                            if viewModel.knownTriggers.contains(trigger) {
                                viewModel.knownTriggers.remove(trigger)
                            } else {
                                viewModel.knownTriggers.insert(trigger)
                            }
                            HapticFeedback.selection()
                        }
                    }
                }
                .padding(.horizontal, Spacing.xl)
                .padding(.bottom, Spacing.xxl)
            }
            .opacity(showContent ? 1 : 0)

            // Info note
            HStack(spacing: Spacing.sm) {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(Colors.Semantic.warning)

                Text("Don't worry if you're not sure yet - we'll help you discover patterns over time")
                    .font(.system(size: 13))
                    .foregroundColor(Colors.Gray.g600)
            }
            .padding(Spacing.md)
            .background(Colors.Semantic.warningLight.opacity(0.5))
            .cornerRadius(Radii.md)
            .padding(.horizontal, Spacing.xl)

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

struct OnboardingTriggerRow: View {
    let trigger: KnownTrigger
    let isSelected: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: Spacing.md) {
                Image(systemName: trigger.icon)
                    .font(.system(size: 20))
                    .foregroundColor(isSelected ? Colors.Primary.p500 : Colors.Gray.g500)
                    .frame(width: 32)

                Text(trigger.title)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(Colors.Gray.g800)

                Spacer()

                ZStack {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(isSelected ? Colors.Primary.p500 : Colors.Gray.g300, lineWidth: 2)
                        .frame(width: 24, height: 24)

                    if isSelected {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Colors.Primary.p500)
                            .frame(width: 24, height: 24)

                        Image(systemName: "checkmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(.white)
                    }
                }
            }
            .padding(Spacing.md)
            .background(Color.white)
            .cornerRadius(Radii.md)
            .shadow(color: Color.black.opacity(0.03), radius: 4)
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

// MARK: - Stage 7: Quick Log Tutorial

struct Stage7QuickLogTutorialView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false
    @State private var demoSliderValue: Double = 5
    @State private var hasInteracted = false

    var body: some View {
        VStack(spacing: 0) {
            // Section header
            TutorialSectionHeader(
                icon: "bolt.fill",
                title: "Quick Log",
                subtitle: "Your daily check-in"
            )
            .padding(.top, Spacing.lg)

            // Anky explaining
            HStack {
                AnkyRiveView(size: 80, state: .explaining, showShadow: false)

                OnboardingSpeechBubble(text: "This is the fastest way to log how you're feeling. Try the slider!")
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, Spacing.lg)
            .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.xl)

            // Demo interface
            VStack(spacing: Spacing.lg) {
                // Pain level demo
                VStack(alignment: .leading, spacing: Spacing.sm) {
                    HStack {
                        Text("Overall Pain Level")
                            .font(.system(size: 15, weight: .medium))
                            .foregroundColor(Colors.Gray.g700)

                        Spacer()

                        Text("\(Int(demoSliderValue))/10")
                            .font(.system(size: 17, weight: .bold, design: .rounded))
                            .foregroundColor(painColor)
                    }

                    // Custom slider
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            // Track
                            Capsule()
                                .fill(Colors.Gray.g200)
                                .frame(height: 8)

                            // Fill
                            Capsule()
                                .fill(
                                    LinearGradient(
                                        colors: [Colors.Semantic.success, Colors.Semantic.warning, Colors.Semantic.error],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: geometry.size.width * (demoSliderValue / 10), height: 8)

                            // Thumb
                            Circle()
                                .fill(Color.white)
                                .frame(width: 28, height: 28)
                                .shadow(color: Color.black.opacity(0.15), radius: 4)
                                .offset(x: geometry.size.width * (demoSliderValue / 10) - 14)
                                .gesture(
                                    DragGesture()
                                        .onChanged { value in
                                            let newValue = min(max(value.location.x / geometry.size.width * 10, 0), 10)
                                            demoSliderValue = newValue
                                            if !hasInteracted {
                                                hasInteracted = true
                                                viewModel.quickLogCompleted = true
                                            }
                                            HapticFeedback.selection()
                                        }
                                )
                        }
                    }
                    .frame(height: 28)

                    // Scale labels
                    HStack {
                        Text("No pain")
                            .font(.system(size: 12))
                            .foregroundColor(Colors.Gray.g500)
                        Spacer()
                        Text("Severe")
                            .font(.system(size: 12))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }
                .padding(Spacing.lg)
                .background(Color.white)
                .cornerRadius(Radii.lg)
                .shadow(color: Color.black.opacity(0.05), radius: 8)

                // Success feedback
                if hasInteracted {
                    HStack(spacing: Spacing.sm) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(Colors.Semantic.success)
                        Text("Great! That's all it takes to log your daily check-in")
                            .font(.system(size: 14))
                            .foregroundColor(Colors.Gray.g600)
                    }
                    .padding(Spacing.md)
                    .background(Colors.Semantic.successLight.opacity(0.5))
                    .cornerRadius(Radii.md)
                    .transition(.scale.combined(with: .opacity))
                }
            }
            .padding(.horizontal, Spacing.xl)
            .opacity(showContent ? 1 : 0)

            Spacer()

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true,
                nextTitle: hasInteracted ? "Continue" : "Try it first"
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }

    private var painColor: Color {
        switch Int(demoSliderValue) {
        case 0...3: return Colors.Semantic.success
        case 4...6: return Colors.Semantic.warning
        default: return Colors.Semantic.error
        }
    }
}

// MARK: - Stage 8: Body Map Tutorial (THE MISSING ONE!)

struct Stage8BodyMapTutorialView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false
    @State private var selectedRegions: Set<String> = []
    @State private var hasInteracted = false

    // Demo regions for the simplified body map
    let bodyRegions = [
        BodyRegionDemo(id: "neck", name: "Neck", x: 0.5, y: 0.12),
        BodyRegionDemo(id: "upper_back", name: "Upper Back", x: 0.5, y: 0.22),
        BodyRegionDemo(id: "lower_back", name: "Lower Back", x: 0.5, y: 0.35),
        BodyRegionDemo(id: "si_joint", name: "SI Joint", x: 0.5, y: 0.42),
        BodyRegionDemo(id: "left_hip", name: "Left Hip", x: 0.35, y: 0.48),
        BodyRegionDemo(id: "right_hip", name: "Right Hip", x: 0.65, y: 0.48),
        BodyRegionDemo(id: "left_knee", name: "Left Knee", x: 0.38, y: 0.7),
        BodyRegionDemo(id: "right_knee", name: "Right Knee", x: 0.62, y: 0.7),
    ]

    var body: some View {
        VStack(spacing: 0) {
            // Section header
            TutorialSectionHeader(
                icon: "figure.stand",
                title: "Body Map",
                subtitle: "Tap to mark affected areas"
            )
            .padding(.top, Spacing.lg)

            // Anky explaining
            HStack {
                AnkyRiveView(size: 80, state: .explaining, showShadow: false)

                OnboardingSpeechBubble(text: "Tap on the body where you feel pain. This helps track exactly where your symptoms occur!")
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, Spacing.lg)
            .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.md)

            // Interactive body map demo
            ZStack {
                // Body outline
                BodyOutlineShape()
                    .stroke(Colors.Gray.g300, lineWidth: 2)
                    .fill(Colors.Gray.g100.opacity(0.5))
                    .frame(width: 160, height: 320)

                // Tappable regions
                ForEach(bodyRegions, id: \.id) { region in
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                            if selectedRegions.contains(region.id) {
                                selectedRegions.remove(region.id)
                            } else {
                                selectedRegions.insert(region.id)
                                if !hasInteracted {
                                    hasInteracted = true
                                    viewModel.bodyMapExplored = true
                                }
                            }
                        }
                        HapticFeedback.light()
                    } label: {
                        Circle()
                            .fill(selectedRegions.contains(region.id) ? Colors.Semantic.error.opacity(0.8) : Colors.Primary.p500.opacity(0.3))
                            .frame(width: selectedRegions.contains(region.id) ? 36 : 28, height: selectedRegions.contains(region.id) ? 36 : 28)
                            .overlay(
                                Circle()
                                    .stroke(selectedRegions.contains(region.id) ? Colors.Semantic.error : Colors.Primary.p500, lineWidth: 2)
                            )
                            .shadow(color: selectedRegions.contains(region.id) ? Colors.Semantic.error.opacity(0.3) : Color.clear, radius: 8)
                    }
                    .position(
                        x: 160 * region.x,
                        y: 320 * region.y
                    )
                }

                // Labels for selected regions
                ForEach(bodyRegions.filter { selectedRegions.contains($0.id) }, id: \.id) { region in
                    Text(region.name)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Colors.Semantic.error)
                        .cornerRadius(4)
                        .position(
                            x: 160 * region.x + (region.x > 0.5 ? 40 : -40),
                            y: 320 * region.y
                        )
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .frame(width: 200, height: 340)
            .opacity(showContent ? 1 : 0)

            // Feedback
            VStack(spacing: Spacing.sm) {
                if !selectedRegions.isEmpty {
                    Text("\(selectedRegions.count) region\(selectedRegions.count > 1 ? "s" : "") selected")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(Colors.Primary.p500)

                    if hasInteracted {
                        HStack(spacing: Spacing.xs) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(Colors.Semantic.success)
                            Text("Perfect! You've got the hang of it")
                                .font(.system(size: 13))
                                .foregroundColor(Colors.Gray.g600)
                        }
                        .transition(.scale.combined(with: .opacity))
                    }
                } else {
                    Text("Tap on any highlighted area")
                        .font(.system(size: 14))
                        .foregroundColor(Colors.Gray.g500)
                }
            }
            .padding(.top, Spacing.md)
            .animation(.spring(response: 0.3), value: selectedRegions)

            Spacer()

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true,
                nextTitle: hasInteracted ? "Continue" : "Try it first"
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

struct BodyRegionDemo {
    let id: String
    let name: String
    let x: CGFloat
    let y: CGFloat
}

// Simple body outline shape
struct BodyOutlineShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let w = rect.width
        let h = rect.height

        // Head
        path.addEllipse(in: CGRect(x: w * 0.35, y: 0, width: w * 0.3, height: h * 0.1))

        // Neck
        path.addRect(CGRect(x: w * 0.42, y: h * 0.08, width: w * 0.16, height: h * 0.04))

        // Torso
        path.move(to: CGPoint(x: w * 0.25, y: h * 0.12))
        path.addLine(to: CGPoint(x: w * 0.75, y: h * 0.12))
        path.addLine(to: CGPoint(x: w * 0.7, y: h * 0.45))
        path.addLine(to: CGPoint(x: w * 0.3, y: h * 0.45))
        path.closeSubpath()

        // Left arm
        path.move(to: CGPoint(x: w * 0.25, y: h * 0.12))
        path.addLine(to: CGPoint(x: w * 0.1, y: h * 0.35))
        path.addLine(to: CGPoint(x: w * 0.15, y: h * 0.36))
        path.addLine(to: CGPoint(x: w * 0.28, y: h * 0.15))

        // Right arm
        path.move(to: CGPoint(x: w * 0.75, y: h * 0.12))
        path.addLine(to: CGPoint(x: w * 0.9, y: h * 0.35))
        path.addLine(to: CGPoint(x: w * 0.85, y: h * 0.36))
        path.addLine(to: CGPoint(x: w * 0.72, y: h * 0.15))

        // Pelvis
        path.move(to: CGPoint(x: w * 0.3, y: h * 0.45))
        path.addLine(to: CGPoint(x: w * 0.7, y: h * 0.45))
        path.addLine(to: CGPoint(x: w * 0.65, y: h * 0.52))
        path.addLine(to: CGPoint(x: w * 0.35, y: h * 0.52))
        path.closeSubpath()

        // Left leg
        path.move(to: CGPoint(x: w * 0.35, y: h * 0.52))
        path.addLine(to: CGPoint(x: w * 0.3, y: h * 0.95))
        path.addLine(to: CGPoint(x: w * 0.42, y: h * 0.95))
        path.addLine(to: CGPoint(x: w * 0.45, y: h * 0.52))

        // Right leg
        path.move(to: CGPoint(x: w * 0.55, y: h * 0.52))
        path.addLine(to: CGPoint(x: w * 0.58, y: h * 0.95))
        path.addLine(to: CGPoint(x: w * 0.7, y: h * 0.95))
        path.addLine(to: CGPoint(x: w * 0.65, y: h * 0.52))

        return path
    }
}

// MARK: - Stage 9: Flare Tutorial

struct Stage9FlareTutorialView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false
    @State private var showSOSDemo = false
    @State private var selectedSeverity: Int = 0

    var body: some View {
        VStack(spacing: 0) {
            // Section header
            TutorialSectionHeader(
                icon: "flame.fill",
                title: "Flare Capture",
                subtitle: "Quick logging when it matters most"
            )
            .padding(.top, Spacing.lg)

            // Anky explaining
            HStack {
                AnkyRiveView(size: 80, state: .sympathetic, showShadow: false)

                OnboardingSpeechBubble(text: "When a flare hits, you need to log it fast. Tap the SOS button to see how quick it is!")
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, Spacing.lg)
            .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.xl)

            if !showSOSDemo {
                // SOS Button Demo
                VStack(spacing: Spacing.lg) {
                    // Big SOS button
                    Button {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            showSOSDemo = true
                            viewModel.flareCaptureSeen = true
                        }
                        HapticFeedback.heavy()
                    } label: {
                        ZStack {
                            Circle()
                                .fill(
                                    RadialGradient(
                                        colors: [Color.red.opacity(0.8), Color.red],
                                        center: .center,
                                        startRadius: 0,
                                        endRadius: 60
                                    )
                                )
                                .frame(width: 120, height: 120)
                                .shadow(color: Color.red.opacity(0.4), radius: 20)

                            VStack(spacing: 4) {
                                Image(systemName: "flame.fill")
                                    .font(.system(size: 32))
                                    .foregroundColor(.white)

                                Text("SOS")
                                    .font(.system(size: 16, weight: .bold))
                                    .foregroundColor(.white)
                            }
                        }
                    }
                    .buttonStyle(ScaleButtonStyle())
                    .pulsingGlow(color: .red, radius: 20)

                    Text("Tap to try the SOS capture")
                        .font(.system(size: 14))
                        .foregroundColor(Colors.Gray.g500)
                }
                .opacity(showContent ? 1 : 0)
            } else {
                // Demo flare capture interface
                VStack(spacing: Spacing.lg) {
                    Text("How bad is this flare?")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(Colors.Gray.g800)

                    // Severity selector
                    HStack(spacing: Spacing.md) {
                        ForEach(1...5, id: \.self) { level in
                            Button {
                                selectedSeverity = level
                                HapticFeedback.selection()
                            } label: {
                                VStack(spacing: 4) {
                                    ZStack {
                                        Circle()
                                            .fill(severityColor(level).opacity(selectedSeverity == level ? 0.2 : 0.1))
                                            .frame(width: 50, height: 50)

                                        Image(systemName: "flame.fill")
                                            .font(.system(size: 20))
                                            .foregroundColor(selectedSeverity == level ? severityColor(level) : Colors.Gray.g400)
                                    }

                                    Text(severityLabel(level))
                                        .font(.system(size: 10))
                                        .foregroundColor(selectedSeverity == level ? severityColor(level) : Colors.Gray.g500)
                                }
                            }
                            .scaleEffect(selectedSeverity == level ? 1.1 : 1)
                            .animation(.spring(response: 0.3), value: selectedSeverity)
                        }
                    }

                    if selectedSeverity > 0 {
                        HStack(spacing: Spacing.sm) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(Colors.Semantic.success)
                            Text("That's it! Logged in just 2 taps")
                                .font(.system(size: 14))
                                .foregroundColor(Colors.Gray.g600)
                        }
                        .padding(Spacing.md)
                        .background(Colors.Semantic.successLight.opacity(0.5))
                        .cornerRadius(Radii.md)
                        .transition(.scale.combined(with: .opacity))
                    }
                }
                .padding(Spacing.lg)
                .background(Color.white)
                .cornerRadius(Radii.xl)
                .shadow(color: Color.black.opacity(0.1), radius: 16)
                .padding(.horizontal, Spacing.xl)
                .transition(.scale.combined(with: .opacity))
            }

            Spacer()

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true,
                nextTitle: viewModel.flareCaptureSeen ? "Continue" : "Try SOS first"
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }

    private func severityColor(_ level: Int) -> Color {
        switch level {
        case 1: return Colors.Semantic.success
        case 2: return Color.yellow
        case 3: return Colors.Semantic.warning
        case 4: return Color.orange
        case 5: return Colors.Semantic.error
        default: return Colors.Gray.g400
        }
    }

    private func severityLabel(_ level: Int) -> String {
        switch level {
        case 1: return "Mild"
        case 2: return "Low"
        case 3: return "Mod"
        case 4: return "High"
        case 5: return "Severe"
        default: return ""
        }
    }
}

// MARK: - Stage 10: Notifications

struct Stage10NotificationsView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false

    var body: some View {
        VStack(spacing: 0) {
            // Anky
            AnkyRiveView(size: 100, state: .thinking, showShadow: false)
                .padding(.top, Spacing.lg)

            // Title
            VStack(spacing: Spacing.sm) {
                Text("Stay on Track")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text("I can send you gentle reminders")
                    .font(.system(size: 15))
                    .foregroundColor(Colors.Gray.g500)
            }
            .padding(.top, Spacing.md)
            .opacity(showContent ? 1 : 0)

            Spacer().frame(height: Spacing.xl)

            // Notification options
            VStack(spacing: Spacing.md) {
                // Daily reminder toggle
                NotificationToggleRow(
                    icon: "bell.fill",
                    title: "Daily Check-in Reminder",
                    subtitle: "A gentle nudge to log how you're feeling",
                    isEnabled: $viewModel.enableDailyReminder
                )

                // Time picker (if enabled)
                if viewModel.enableDailyReminder {
                    HStack {
                        Text("Reminder time")
                            .font(.system(size: 15))
                            .foregroundColor(Colors.Gray.g600)

                        Spacer()

                        DatePicker("", selection: $viewModel.reminderTime, displayedComponents: .hourAndMinute)
                            .labelsHidden()
                            .tint(Colors.Primary.p500)
                    }
                    .padding(Spacing.md)
                    .background(Colors.Gray.g50)
                    .cornerRadius(Radii.md)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // Flare alerts toggle
                NotificationToggleRow(
                    icon: "exclamationmark.triangle.fill",
                    title: "Flare Risk Alerts",
                    subtitle: "Get warned when conditions might trigger a flare",
                    isEnabled: $viewModel.enableFlareAlerts
                )
            }
            .padding(.horizontal, Spacing.xl)
            .opacity(showContent ? 1 : 0)
            .animation(.spring(response: 0.3), value: viewModel.enableDailyReminder)

            Spacer()

            // Skip option
            Button {
                viewModel.enableDailyReminder = false
                viewModel.enableFlareAlerts = false
                viewModel.nextStage()
            } label: {
                Text("Skip notifications for now")
                    .font(.system(size: 14))
                    .foregroundColor(Colors.Gray.g500)
                    .underline()
            }
            .padding(.bottom, Spacing.md)

            // Navigation
            OnboardingNavBar(
                onBack: { viewModel.previousStage() },
                onNext: { viewModel.nextStage() },
                canGoBack: true,
                nextTitle: "Almost done!"
            )
            .padding(.horizontal, Spacing.xl)
            .padding(.bottom, Spacing.xxl)
        }
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7).delay(0.2)) {
                showContent = true
            }
        }
    }
}

struct NotificationToggleRow: View {
    let icon: String
    let title: String
    let subtitle: String
    @Binding var isEnabled: Bool

    var body: some View {
        HStack(spacing: Spacing.md) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(isEnabled ? Colors.Primary.p500 : Colors.Gray.g400)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(Colors.Gray.g800)

                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()

            Toggle("", isOn: $isEnabled)
                .labelsHidden()
                .tint(Colors.Accent.teal)
        }
        .padding(Spacing.md)
        .background(Color.white)
        .cornerRadius(Radii.lg)
        .shadow(color: Color.black.opacity(0.05), radius: 8)
    }
}

// MARK: - Stage 11: Ready!

struct Stage11ReadyView: View {
    @ObservedObject var viewModel: OnboardingRedesignViewModel
    @State private var showContent = false
    @State private var showConfetti = false

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                Spacer()

                // Celebrating Anky
                ZStack {
                    if showConfetti {
                        CelebrationBurst()
                    }

                    AnkyRiveView(size: 180, state: .celebrating, showShadow: true)
                        .scaleEffect(showContent ? 1 : 0.5)
                }

                Spacer().frame(height: Spacing.xl)

                // Success message
                VStack(spacing: Spacing.md) {
                    Text("You're All Set! 🎉")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(Colors.Gray.g900)

                    Text("We've personalized InflamAI just for you.\nLet's start this journey together!")
                        .font(.system(size: 17))
                        .foregroundColor(Colors.Gray.g600)
                        .multilineTextAlignment(.center)
                        .lineSpacing(4)

                    // Personalization summary
                    if !viewModel.selectedGoals.isEmpty {
                        VStack(spacing: Spacing.xs) {
                            Text("Based on your goals, we'll help you:")
                                .font(.system(size: 14))
                                .foregroundColor(Colors.Gray.g500)

                            OnboardingFlowLayout(spacing: Spacing.xs) {
                                ForEach(Array(viewModel.selectedGoals.prefix(3)), id: \.self) { goal in
                                    Text(goal.title)
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundColor(Colors.Primary.p600)
                                        .padding(.horizontal, Spacing.sm)
                                        .padding(.vertical, 4)
                                        .background(Colors.Primary.p100)
                                        .cornerRadius(Radii.full)
                                }
                            }
                        }
                        .padding(.top, Spacing.md)
                    }
                }
                .padding(.horizontal, Spacing.xl)
                .opacity(showContent ? 1 : 0)

                Spacer()

                // Start button
                Button {
                    viewModel.completeOnboarding()
                } label: {
                    HStack(spacing: Spacing.sm) {
                        Text("Start Using InflamAI")
                            .font(.system(size: 17, weight: .semibold))

                        Image(systemName: "arrow.right")
                            .font(.system(size: 15, weight: .semibold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, Spacing.md)
                    .background(
                        LinearGradient(
                            colors: [Colors.Accent.teal, Colors.Primary.p500],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(Radii.lg)
                    .shadow(color: Colors.Primary.p500.opacity(0.3), radius: 12, y: 4)
                }
                .buttonStyle(ScaleButtonStyle())
                .padding(.horizontal, Spacing.xl)
                .padding(.bottom, Spacing.xxl)
                .opacity(showContent ? 1 : 0)
            }
        }
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7).delay(0.3)) {
                showContent = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                showConfetti = true
                HapticFeedback.success()
            }
        }
    }
}

// MARK: - Supporting Components

struct TutorialSectionHeader: View {
    let icon: String
    let title: String
    let subtitle: String

    var body: some View {
        VStack(spacing: Spacing.xs) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(Colors.Primary.p500)

                Text(title)
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)
            }

            Text(subtitle)
                .font(.system(size: 14))
                .foregroundColor(Colors.Gray.g500)
        }
    }
}

struct OnboardingPrimaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 17, weight: .semibold))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, Spacing.md)
                .background(
                    LinearGradient(
                        colors: [Colors.Accent.teal, Colors.Primary.p500],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(Radii.lg)
                .shadow(color: Colors.Primary.p500.opacity(0.3), radius: 8, y: 4)
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

struct OnboardingNavBar: View {
    let onBack: () -> Void
    let onNext: () -> Void
    var canGoBack: Bool = true
    var nextTitle: String = "Continue"
    var nextEnabled: Bool = true

    var body: some View {
        HStack(spacing: Spacing.md) {
            if canGoBack {
                Button(action: onBack) {
                    HStack(spacing: Spacing.xs) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .semibold))
                        Text("Back")
                            .font(.system(size: 15, weight: .medium))
                    }
                    .foregroundColor(Colors.Gray.g600)
                    .padding(.vertical, Spacing.sm)
                    .padding(.horizontal, Spacing.md)
                }
            }

            Spacer()

            Button(action: onNext) {
                HStack(spacing: Spacing.xs) {
                    Text(nextTitle)
                        .font(.system(size: 16, weight: .semibold))
                    Image(systemName: "chevron.right")
                        .font(.system(size: 14, weight: .semibold))
                }
                .foregroundColor(.white)
                .padding(.vertical, Spacing.sm)
                .padding(.horizontal, Spacing.lg)
                .background(nextEnabled ? Colors.Primary.p500 : Colors.Gray.g300)
                .cornerRadius(Radii.full)
            }
            .disabled(!nextEnabled)
        }
    }
}

struct PersonalizationSection<Content: View>: View {
    let title: String
    let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(Colors.Gray.g700)

            content
        }
    }
}

struct SelectableChip: View {
    let title: String
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack {
                Text(title)
                    .font(.system(size: 15, weight: isSelected ? .semibold : .regular))

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                }
            }
            .foregroundColor(isSelected ? Colors.Primary.p600 : Colors.Gray.g700)
            .padding(Spacing.md)
            .background(isSelected ? Colors.Primary.p100 : Color.white)
            .cornerRadius(Radii.md)
            .overlay(
                RoundedRectangle(cornerRadius: Radii.md)
                    .stroke(isSelected ? Colors.Primary.p500 : Colors.Gray.g200, lineWidth: isSelected ? 2 : 1)
            )
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

struct YearPicker: View {
    @Binding var selectedYear: Int
    let years: [Int]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: Spacing.sm) {
                ForEach(years.prefix(20), id: \.self) { year in
                    Button {
                        selectedYear = year
                        HapticFeedback.selection()
                    } label: {
                        Text(String(year))
                            .font(.system(size: 14, weight: selectedYear == year ? .bold : .regular))
                            .foregroundColor(selectedYear == year ? .white : Colors.Gray.g700)
                            .padding(.horizontal, Spacing.sm)
                            .padding(.vertical, Spacing.xs)
                            .background(selectedYear == year ? Colors.Primary.p500 : Colors.Gray.g100)
                            .cornerRadius(Radii.sm)
                    }
                }
            }
        }
    }
}

struct SymptomYearsPicker: View {
    @Binding var years: Int

    let options = [
        ("< 1", 0),
        ("1-2", 1),
        ("3-5", 3),
        ("5-10", 5),
        ("10+", 10)
    ]

    var body: some View {
        HStack(spacing: Spacing.sm) {
            ForEach(options, id: \.0) { label, value in
                Button {
                    years = value
                    HapticFeedback.selection()
                } label: {
                    Text(label)
                        .font(.system(size: 14, weight: years == value ? .semibold : .regular))
                        .foregroundColor(years == value ? .white : Colors.Gray.g700)
                        .padding(.horizontal, Spacing.sm)
                        .padding(.vertical, Spacing.xs)
                        .background(years == value ? Colors.Primary.p500 : Colors.Gray.g100)
                        .cornerRadius(Radii.sm)
                }
            }
        }
    }
}

struct ScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(response: 0.2, dampingFraction: 0.6), value: configuration.isPressed)
    }
}

// Simple flow layout for chips
struct OnboardingFlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrangeSubviews(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrangeSubviews(proposal: proposal, subviews: subviews)
        for (index, subview) in subviews.enumerated() {
            subview.place(at: CGPoint(x: bounds.minX + result.positions[index].x, y: bounds.minY + result.positions[index].y), proposal: .unspecified)
        }
    }

    private func arrangeSubviews(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        var positions: [CGPoint] = []
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxWidth: CGFloat = 0

        let maxX = proposal.width ?? .infinity

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)

            if currentX + size.width > maxX && currentX > 0 {
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }

            positions.append(CGPoint(x: currentX, y: currentY))

            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            maxWidth = max(maxWidth, currentX - spacing)
        }

        return (CGSize(width: maxWidth, height: currentY + lineHeight), positions)
    }
}

// MARK: - Speech Bubble

struct OnboardingSpeechBubble: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 14))
            .foregroundColor(Colors.Gray.g700)
            .padding(Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Radii.lg)
                    .fill(Color.white)
                    .shadow(color: Color.black.opacity(0.08), radius: 8)
            )
    }
}

// MARK: - Notification Extension

extension Notification.Name {
    static let onboardingCompleted = Notification.Name("onboardingCompleted")
}

// MARK: - Previews

#Preview("Full Flow") {
    OnboardingRedesignFlow()
}

#Preview("Body Map Tutorial") {
    Stage8BodyMapTutorialView(viewModel: OnboardingRedesignViewModel())
}

#Preview("Goals") {
    Stage5GoalsView(viewModel: OnboardingRedesignViewModel())
}
