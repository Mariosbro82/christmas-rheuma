//
//  HomeView.swift
//  InflamAI
//
//  Main dashboard bringing together all app features
//  Quick access to daily logging, trends, medications, exercises, and flare tracking
//

import SwiftUI
import CoreData
import HealthKit

struct HomeView: View {
    @StateObject private var viewModel: HomeViewModel

    // MARK: - Neural Engine (Primary ML Source)
    @ObservedObject private var neuralEngine = UnifiedNeuralEngine.shared

    @State private var showingQuickLog = false
    @State private var showingSOSFlare = false
    @State private var showingNeuralEngine = false
    @State private var selectedTab: DashboardTab = .overview
    @State private var isLoading = true
    @State private var animateContent = false

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: HomeViewModel(context: context))
    }

    var body: some View {
        ScrollView {
            if isLoading {
                // Loading State - Design System Applied
                VStack(spacing: Spacing.lg) {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Colors.Accent.purple.opacity(0.2), Colors.Primary.p500.opacity(0.1)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 80, height: 80)

                        ProgressView()
                            .scaleEffect(1.3)
                            .tint(Colors.Accent.purple)
                    }

                    VStack(spacing: Spacing.xs) {
                        Text("Loading your dashboard")
                            .font(.system(size: Typography.md, weight: .semibold))
                            .foregroundColor(Colors.Gray.g900)

                        Text("Preparing your health insights...")
                            .font(.system(size: Typography.base))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.top, 120)
            } else {
                VStack(spacing: Spacing.lg) {
                    // Greeting Header
                    greetingHeader
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Neural Engine AI Predictions (PRIMARY - most important)
                    neuralEnginePredictionCard
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Weather Flare Risk
                    WeatherFlareRiskCard()
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Quick Actions
                    quickActionsSection
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Today's Questionnaires
                    todaysQuestionnairesSection
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Today's Summary
                    todaysSummarySection
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Medication Reminders
                    if viewModel.hasPendingMedications {
                        medicationRemindersSection
                            .opacity(animateContent ? 1 : 0)
                            .offset(y: animateContent ? 0 : 20)
                    }

                    // Recent Trends
                    recentTrendsSection
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // My Routine
                    myRoutineSection
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Exercise Suggestion
                    exerciseSuggestionSection
                        .opacity(animateContent ? 1 : 0)
                        .offset(y: animateContent ? 0 : 20)

                    // Flare Alert
                    if viewModel.hasActiveFlare {
                        activeFlareAlert
                            .opacity(animateContent ? 1 : 0)
                            .offset(y: animateContent ? 0 : 20)
                    }
                }
                .padding(Spacing.md)
            }
        }
        .background(Colors.Gray.g50)
        .navigationTitle("InflamAI")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                NavigationLink {
                    SettingsView()
                } label: {
                    Image(systemName: "gearshape")
                        .font(.title3)
                }
            }
        }
        .sheet(isPresented: $showingQuickLog) {
            QuickLogView()
        }
        .sheet(isPresented: $showingSOSFlare) {
            SOSFlareView()
        }
        .task {
            // Load data asynchronously after view appears
            await viewModel.loadDashboardData()

            withAnimation(.easeOut(duration: 0.4)) {
                isLoading = false
            }

            // Staggered animation for content
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
            withAnimation(Animations.spring) {
                animateContent = true
            }
        }
    }

    // MARK: - Greeting Header

    private var greetingHeader: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(viewModel.greeting)
                    .font(.system(size: Typography.xxl, weight: .bold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                Text(Date().formatted(date: .complete, time: .omitted))
                    .font(.system(size: Typography.base))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()

            // Premium Streak indicator - Section 9.1: 16pt margin from edges
            if viewModel.currentStreak > 0 {
                HStack(spacing: Spacing.xs) {
                    Image(systemName: "flame.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [Colors.Semantic.warning, Colors.Semantic.error],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )

                    VStack(alignment: .leading, spacing: Spacing.xxxs) {
                        Text("\(viewModel.currentStreak)")
                            .font(.system(size: Typography.xl, weight: .bold, design: .rounded))
                            .foregroundColor(Colors.Gray.g900)

                        Text("day streak")
                            .font(.system(size: Typography.xs, weight: .medium))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }
                .padding(.horizontal, Spacing.md)  // 16pt horizontal padding per Section 9.1
                .padding(.vertical, Spacing.sm)    // 12pt vertical padding
                .background(
                    RoundedRectangle(cornerRadius: Radii.lg)
                        .fill(
                            LinearGradient(
                                colors: [Colors.Semantic.warning.opacity(0.15), Colors.Semantic.error.opacity(0.08)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: Radii.lg)
                        .stroke(Colors.Semantic.warning.opacity(0.3), lineWidth: 1)
                )
            }
        }
    }

    // MARK: - Neural Engine Prediction Card (PRIMARY ML)
    // Section 9.1: Uses Blue (positive), Orange (moderate), Red (high risk)

    private var neuralEnginePredictionCard: some View {
        Button {
            showingNeuralEngine = true
        } label: {
            VStack(spacing: Spacing.md) {
                // Header with status
                HStack {
                    ZStack {
                        Circle()
                            .fill(neuralEngineStatusColor.opacity(0.2))
                            .frame(width: 50, height: 50)

                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 24))
                            .foregroundColor(neuralEngineStatusColor)
                            .opacity(neuralEngine.engineStatus == .learning ? 0.7 : 1.0)
                    }

                    VStack(alignment: .leading, spacing: Spacing.xxs) {
                        HStack {
                            Text("AI Pattern Overview")
                                .font(.system(size: Typography.md, weight: .semibold))
                                .foregroundColor(Colors.Gray.g900)

                            if neuralEngine.isPersonalized {
                                Text("PERSONALIZED")
                                    .font(.system(size: Typography.xxs, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, Spacing.xs)
                                    .padding(.vertical, Spacing.xxxs)
                                    .background(Colors.Primary.p500)
                                    .cornerRadius(Radii.xs)
                            }
                        }

                        Text(neuralEngine.engineStatus.displayMessage)
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundColor(Colors.Gray.g400)
                }

                // Main prediction display
                if let prediction = neuralEngine.currentPrediction {
                    HStack(alignment: .center, spacing: Spacing.md) {
                        // Left: Risk info - Focus on STATUS not scary percentages
                        VStack(alignment: .leading, spacing: Spacing.xxs) {
                            Text("Your Data Patterns")
                                .font(.system(size: Typography.xs))
                                .foregroundColor(Colors.Gray.g500)

                            // Show friendly status with semantic colors per Section 9.1
                            // Blue = positive, Orange = moderate, Red = notable patterns
                            if prediction.willFlare {
                                let riskColor = predictionRiskColor(prediction)
                                Text(prediction.riskLevel == .high ? "NOTABLE PATTERNS" : "HEADS UP")
                                    .font(.system(size: Typography.xxl, weight: .bold, design: .rounded))
                                    .foregroundColor(riskColor)
                                Text(prediction.riskLevel == .high ? "Discuss with your doctor" : "Changes observed in your data")
                                    .font(.system(size: Typography.sm))
                                    .foregroundColor(riskColor)
                            } else {
                                Text("STABLE PATTERNS")
                                    .font(.system(size: Typography.xxl, weight: .bold, design: .rounded))
                                    .foregroundColor(Colors.Primary.p500)
                                Text("No notable changes in your data")
                                    .font(.system(size: Typography.sm))
                                    .foregroundColor(Colors.Primary.p500)
                            }

                            // Show pattern level badge with semantic colors
                            Text(prediction.riskLevel.rawValue + " Pattern")
                                .font(.system(size: Typography.xs, weight: .medium))
                                .foregroundColor(.white)
                                .padding(.horizontal, Spacing.xs)
                                .padding(.vertical, Spacing.xxxs)
                                .background(predictionRiskColor(prediction))
                                .cornerRadius(Radii.xs)
                        }

                        Spacer()

                        // Right: Visual icon with semantic colors
                        ZStack {
                            Circle()
                                .fill(predictionRiskColor(prediction).opacity(0.15))
                                .frame(width: 80, height: 80)

                            Image(systemName: prediction.willFlare ? "exclamationmark.triangle.fill" : "checkmark.shield.fill")
                                .font(.system(size: 36))
                                .foregroundColor(predictionRiskColor(prediction))
                        }
                    }
                    .padding(.vertical, Spacing.xs)

                    // Data quality bar
                    VStack(alignment: .leading, spacing: Spacing.xxs) {
                        HStack {
                            Text("Data Quality: \(prediction.confidence.rawValue)")
                                .font(.system(size: Typography.xs))
                                .foregroundColor(Colors.Gray.g500)
                            Spacer()
                            Text(prediction.timestamp, style: .relative)
                                .font(.system(size: Typography.xxs))
                                .foregroundColor(Colors.Gray.g500)
                        }

                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: Radii.xs)
                                    .fill(Colors.Gray.g200)

                                RoundedRectangle(cornerRadius: Radii.xs)
                                    .fill(Colors.Primary.p500)
                                    .frame(width: geometry.size.width * confidenceValue(prediction.confidence))
                            }
                        }
                        .frame(height: 6)
                    }

                    // Recommendation
                    if prediction.willFlare {
                        let riskColor = predictionRiskColor(prediction)
                        HStack(spacing: Spacing.xs) {
                            Image(systemName: "lightbulb.fill")
                                .foregroundColor(riskColor)
                                .font(.caption)

                            Text(prediction.recommendedAction.rawValue)
                                .font(.system(size: Typography.xs))
                                .foregroundColor(Colors.Gray.g900)

                            Spacer()
                        }
                        .padding(Spacing.sm)
                        .background(riskColor.opacity(0.1))
                        .cornerRadius(Radii.md)
                    }
                } else {
                    // No prediction yet - show setup card
                    VStack(spacing: Spacing.sm) {
                        Image(systemName: "waveform.path.ecg")
                            .font(.title)
                            .foregroundColor(Colors.Gray.g400)

                        if neuralEngine.daysOfUserData < 7 {
                            Text("Need \(7 - neuralEngine.daysOfUserData) more days of data")
                                .font(.system(size: Typography.sm))
                                .foregroundColor(Colors.Gray.g500)

                            ProgressView(value: Float(neuralEngine.daysOfUserData), total: 7)
                                .tint(Colors.Primary.p500)
                                .frame(width: 150)

                            Text("\(neuralEngine.daysOfUserData)/7 days logged")
                                .font(.system(size: Typography.xs))
                                .foregroundColor(Colors.Gray.g500)
                        } else {
                            Text("Tap to view your data patterns")
                                .font(.system(size: Typography.sm))
                                .foregroundColor(Colors.Gray.g500)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, Spacing.lg)
                }

                // Personalization progress bar
                HStack(spacing: Spacing.xs) {
                    Image(systemName: "brain")
                        .font(.caption)
                        .foregroundColor(Colors.Primary.p500)

                    Text("Learning: \(neuralEngine.personalizationPhase.rawValue)")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)

                    Spacer()

                    Text("\(Int(neuralEngine.learningProgress * 100))%")
                        .font(.system(size: Typography.xs, weight: .medium))
                        .foregroundColor(Colors.Primary.p500)
                }

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: Radii.xs)
                            .fill(Colors.Gray.g200)

                        RoundedRectangle(cornerRadius: Radii.xs)
                            .fill(Colors.Primary.p500)
                            .frame(width: geometry.size.width * CGFloat(neuralEngine.learningProgress))
                    }
                }
                .frame(height: 4)
            }
            .padding(Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Radii.xl)
                    .fill(Color(.systemBackground))
            )
            .dshadow(Shadows.sm)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showingNeuralEngine) {
            NavigationStack {
                UnifiedNeuralEngineView()
            }
        }
    }

    // MARK: - Neural Engine Helpers

    /// Section 9.1: Blue for positive, Orange for moderate, Red for high risk
    private func predictionRiskColor(_ prediction: FlareRiskPrediction) -> Color {
        if !prediction.willFlare {
            return Colors.Primary.p500  // Blue for positive
        }
        switch prediction.riskLevel {
        case .critical, .high:
            return Colors.Semantic.error  // Red for high/critical risk
        case .moderate, .low:
            return Colors.Semantic.warning  // Orange for moderate/low risk
        }
    }

    private var neuralEngineStatusColor: Color {
        switch neuralEngine.engineStatus {
        case .ready:
            if let prediction = neuralEngine.currentPrediction {
                return predictionRiskColor(prediction)
            }
            return Colors.Primary.p500  // Blue for positive
        case .initializing:
            return Colors.Semantic.warning
        case .learning:
            return Colors.Primary.p500
        case .error:
            return Colors.Semantic.error
        }
    }

    private func confidenceValue(_ level: ConfidenceLevel) -> CGFloat {
        switch level {
        case .low: return 0.25
        case .moderate: return 0.50
        case .high: return 0.75
        case .veryHigh: return 1.0
        }
    }

    // MARK: - Quick Actions

    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Text("Quick Actions")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                QuickActionCard(
                    icon: "pencil.circle.fill",
                    title: "Log Symptoms",
                    color: Colors.Primary.p500
                ) {
                    showingQuickLog = true
                }

                QuickActionCard(
                    icon: "flame.fill",
                    title: "SOS Flare",
                    color: Colors.Semantic.error
                ) {
                    showingSOSFlare = true
                }

                NavigationLink {
                    CoachCompositorView()
                } label: {
                    QuickActionCardView(
                        icon: "sparkles",
                        title: "Exercise Coach",
                        color: Colors.Accent.purple
                    )
                }

                NavigationLink {
                    TrendsView(context: viewModel.viewContext)
                } label: {
                    QuickActionCardView(
                        icon: "chart.xyaxis.line",
                        title: "View Trends",
                        color: Colors.Semantic.success
                    )
                }
            }
        }
    }

    // MARK: - Today's Questionnaires

    @ViewBuilder
    private var todaysQuestionnairesSection: some View {
        // Use viewModel's cached data instead of computing on every render
        if viewModel.shouldShowQuestionnaires {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                HStack {
                    Image(systemName: "list.clipboard.fill")
                        .font(.title3)
                        .foregroundColor(Colors.Primary.p500)

                    Text("Today's Questionnaires")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    Spacer()

                    if viewModel.dueQuestionnaires.isEmpty {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(Colors.Semantic.success)
                    } else {
                        Text("\(viewModel.dueQuestionnaires.count)")
                            .font(.system(size: Typography.xs, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.horizontal, Spacing.xs)
                            .padding(.vertical, Spacing.xxs)
                            .background(Colors.Primary.p500)
                            .clipShape(Capsule())
                    }
                }

                if viewModel.dueQuestionnaires.isEmpty {
                    // All done for today
                    VStack(spacing: Spacing.sm) {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.title)
                                .foregroundColor(Colors.Semantic.success)

                            VStack(alignment: .leading, spacing: Spacing.xxs) {
                                Text("All caught up!")
                                    .font(.system(size: Typography.base, weight: .semibold))
                                    .foregroundColor(Colors.Gray.g900)

                                Text("No questionnaires due today")
                                    .font(.system(size: Typography.xs))
                                    .foregroundColor(Colors.Gray.g500)
                            }

                            Spacer()
                        }

                        // Link to manage questionnaires
                        NavigationLink {
                            QuestionnaireSettingsView()
                        } label: {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .font(.caption)
                                Text("Manage Questionnaires")
                                    .font(.system(size: Typography.xs, weight: .medium))
                            }
                            .foregroundColor(Colors.Primary.p500)
                            .padding(.horizontal, Spacing.sm)
                            .padding(.vertical, Spacing.xs)
                            .background(Colors.Primary.p50)
                            .cornerRadius(Radii.md)
                        }
                    }
                    .padding(Spacing.md)
                    .background(Colors.Gray.g100)
                    .cornerRadius(Radii.lg)
                } else {
                    // Show due questionnaires
                    ForEach(Array(viewModel.dueQuestionnaires.prefix(3)), id: \.self) { questionnaireID in
                        NavigationLink {
                            QuestionnaireFormView(questionnaireID: questionnaireID)
                        } label: {
                            HStack(spacing: Spacing.sm) {
                                Circle()
                                    .fill(Colors.Primary.p100)
                                    .frame(width: 40, height: 40)
                                    .overlay(
                                        Image(systemName: "doc.text.fill")
                                            .font(.system(size: 18))
                                            .foregroundColor(Colors.Primary.p500)
                                    )

                                VStack(alignment: .leading, spacing: Spacing.xxs) {
                                    Text(NSLocalizedString(questionnaireID.titleKey, comment: ""))
                                        .font(.system(size: Typography.base, weight: .medium))
                                        .foregroundColor(Colors.Gray.g900)

                                    HStack(spacing: Spacing.xxs) {
                                        Image(systemName: "clock")
                                            .font(.caption2)
                                        Text(viewModel.scheduleDescription(for: questionnaireID))
                                            .font(.system(size: Typography.xs))
                                    }
                                    .foregroundColor(Colors.Gray.g500)
                                }

                                Spacer()
                            }
                            .padding(Spacing.md)
                            .background(Color(.systemBackground))
                            .cornerRadius(Radii.lg)
                            .dshadow(Shadows.xs)
                        }
                    }

                    if viewModel.dueQuestionnaires.count > 3 {
                        NavigationLink {
                            QuestionnaireSettingsView()
                        } label: {
                            HStack {
                                Text("View all (\(viewModel.dueQuestionnaires.count))")
                                    .font(.system(size: Typography.xs))
                                    .foregroundColor(Colors.Primary.p500)
                                Spacer()
                                Image(systemName: "arrow.right")
                                    .font(.caption2)
                                    .foregroundColor(Colors.Primary.p500)
                            }
                            .padding(.horizontal, Spacing.sm)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Today's Summary

    private var todaysSummarySection: some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            // Header
            Text("Today's Summary")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            // Metrics grid - premium 2x2 layout
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                SummaryMetric(
                    icon: "bolt.circle.fill",
                    label: "Pain Entries",
                    value: "\(viewModel.todayPainEntries)",
                    color: Colors.Semantic.error
                )

                SummaryMetric(
                    icon: "chart.bar.fill",
                    label: "Assessments",
                    value: "\(viewModel.todayAssessments)",
                    color: Colors.Primary.p500
                )

                SummaryMetric(
                    icon: "book.fill",
                    label: "Journal",
                    value: "\(viewModel.todayJournalEntries)",
                    color: Colors.Semantic.success
                )

                SummaryMetric(
                    icon: "pills.fill",
                    label: "Medications",
                    value: "\(viewModel.activeMedicationsCount)",
                    color: Colors.Accent.purple
                )
            }

            // Log prompt - premium card
            if !viewModel.hasLoggedToday {
                Button {
                    showingQuickLog = true
                } label: {
                    HStack(spacing: Spacing.sm) {
                        ZStack {
                            Circle()
                                .fill(Colors.Semantic.warning.opacity(0.15))
                                .frame(width: 40, height: 40)

                            Image(systemName: "exclamationmark.circle.fill")
                                .font(.system(size: 18))
                                .foregroundColor(Colors.Semantic.warning)
                        }

                        VStack(alignment: .leading, spacing: Spacing.xxxs) {
                            Text("No logs yet today")
                                .font(.system(size: Typography.base, weight: .semibold))
                                .foregroundColor(Colors.Gray.g900)

                            Text("Tap to log your symptoms")
                                .font(.system(size: Typography.xs))
                                .foregroundColor(Colors.Gray.g500)
                        }

                        Spacer()

                        Image(systemName: "chevron.right")
                            .font(.caption)
                            .foregroundColor(Colors.Gray.g400)
                    }
                    .padding(Spacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: Radii.lg)
                            .fill(Colors.Semantic.warning.opacity(0.08))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: Radii.lg)
                            .stroke(Colors.Semantic.warning.opacity(0.2), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Radii.xxl)
                .fill(Color(.systemBackground))
        )
        .dshadow(Shadows.sm)
    }

    // MARK: - Medication Reminders

    private var medicationRemindersSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            HStack {
                HStack(spacing: Spacing.xs) {
                    Image(systemName: "pills.fill")
                        .font(.system(size: 16))
                        .foregroundColor(Colors.Accent.purple)

                    Text("Medications Due")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)
                }

                Spacer()

                NavigationLink {
                    MedicationManagementView()
                } label: {
                    Text("Manage")
                        .font(.system(size: Typography.base, weight: .medium))
                        .foregroundColor(Colors.Accent.purple)
                }
            }

            VStack(spacing: Spacing.sm) {
                ForEach(viewModel.pendingMedications.prefix(3)) { med in
                    MedicationReminderRow(
                        medication: med,
                        onTake: { viewModel.markMedicationTaken(med) },
                        onSkip: { viewModel.skipMedication(med) },
                        onRemindLater: { viewModel.remindMedicationLater(med) }
                    )
                }
            }
        }
        .padding(Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Radii.xxl)
                .fill(Color(.systemBackground))
        )
        .dshadow(Shadows.sm)
        .overlay(
            RoundedRectangle(cornerRadius: Radii.xxl)
                .stroke(Colors.Accent.purple.opacity(0.1), lineWidth: 1)
        )
    }

    // MARK: - Recent Trends

    private var recentTrendsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            HStack {
                HStack(spacing: Spacing.xs) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 16))
                        .foregroundColor(Colors.Primary.p500)

                    Text("7-Day Trends")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)
                }

                Spacer()

                NavigationLink {
                    TrendsView(context: viewModel.viewContext)
                } label: {
                    Text("See All")
                        .font(.system(size: Typography.base, weight: .medium))
                        .foregroundColor(Colors.Primary.p500)
                }
            }

            if let trendSummary = viewModel.weeklyTrendSummary {
                VStack(spacing: Spacing.sm) {
                    TrendIndicator(
                        label: "Pain Level",
                        trend: trendSummary.painTrend,
                        value: trendSummary.avgPain
                    )

                    Divider()

                    TrendIndicator(
                        label: "Stiffness",
                        trend: trendSummary.stiffnessTrend,
                        value: trendSummary.avgStiffness
                    )

                    Divider()

                    TrendIndicator(
                        label: "Fatigue",
                        trend: trendSummary.fatigueTrend,
                        value: trendSummary.avgFatigue
                    )
                }
            } else {
                HStack(spacing: Spacing.sm) {
                    Image(systemName: "chart.line.uptrend.xyaxis.circle")
                        .font(.title2)
                        .foregroundColor(Colors.Gray.g400)

                    VStack(alignment: .leading, spacing: Spacing.xxxs) {
                        Text("Not enough data")
                            .font(.system(size: Typography.base, weight: .medium))
                            .foregroundColor(Colors.Gray.g700)

                        Text("Log more symptoms to see trends")
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, Spacing.xs)
            }
        }
        .padding(Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Radii.xxl)
                .fill(Color(.systemBackground))
        )
        .dshadow(Shadows.sm)
    }

    // MARK: - My Routine

    private var myRoutineSection: some View {
        NavigationLink {
            RoutineManagementView()
        } label: {
            HStack(spacing: Spacing.md) {
                ZStack {
                    Circle()
                        .fill(Colors.Semantic.success.opacity(0.15))
                        .frame(width: 56, height: 56)

                    Image(systemName: "figure.mixed.cardio")
                        .font(.title2)
                        .foregroundColor(Colors.Semantic.success)
                }

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("My Routines")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    if let routine = viewModel.activeRoutine {
                        HStack(spacing: Spacing.xxs) {
                            Text("Active:")
                                .foregroundColor(Colors.Gray.g500)
                            Text(routine.name ?? "Routine")
                                .fontWeight(.medium)
                                .foregroundColor(Colors.Semantic.success)
                        }
                        .font(.system(size: Typography.base))

                        HStack(spacing: Spacing.xs) {
                            Label("\(routine.totalDuration) min", systemImage: "clock")
                            if routine.timesCompleted > 0 {
                                Text("•")
                                Text("\(routine.timesCompleted)x completed")
                            }
                        }
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                    } else {
                        Text("Tap to view all routines")
                            .font(.system(size: Typography.base))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }

                Spacer()
            }
            .padding(Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Radii.xl)
                    .fill(Color(.systemBackground))
            )
            .dshadow(Shadows.sm)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Exercise Suggestion

    private var exerciseSuggestionSection: some View {
        NavigationLink {
            ExerciseLibraryView()
        } label: {
            HStack(spacing: Spacing.md) {
                ZStack {
                    Circle()
                        .fill(Colors.Accent.purple.opacity(0.15))
                        .frame(width: 56, height: 56)

                    Image(systemName: "figure.flexibility")
                        .font(.title2)
                        .foregroundColor(Colors.Accent.purple)
                }

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Daily Exercise")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    Text(viewModel.exerciseSuggestion)
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }

                Spacer()
            }
            .padding(Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Radii.xl)
                    .fill(Color(.systemBackground))
            )
            .dshadow(Shadows.sm)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Active Flare Alert

    private var activeFlareAlert: some View {
        NavigationLink {
            FlareTimelineView(context: viewModel.viewContext)
        } label: {
            HStack(spacing: Spacing.sm) {
                ZStack {
                    Circle()
                        .fill(Colors.Semantic.error.opacity(0.15))
                        .frame(width: 44, height: 44)

                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 20))
                        .foregroundColor(Colors.Semantic.error)
                }

                VStack(alignment: .leading, spacing: Spacing.xxxs) {
                    Text("Active Flare")
                        .font(.system(size: Typography.md, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    Text("Tap to manage your flare")
                        .font(.system(size: Typography.base))
                        .foregroundColor(Colors.Gray.g500)
                }

                Spacer()
            }
            .padding(Spacing.md)
            .background(
                RoundedRectangle(cornerRadius: Radii.xl)
                    .fill(Colors.Semantic.error.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: Radii.xl)
                    .stroke(Colors.Semantic.error.opacity(0.3), lineWidth: 1.5)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Supporting Views

/// Banner shown when user profile is incomplete - critical for ML accuracy (hidden in demo mode)
struct ProfileCompletionBanner: View {
    @StateObject private var profileChecker = ProfileCompletenessChecker()

    var body: some View {
        if !profileChecker.isComplete {
            NavigationLink {
                SettingsView()
            } label: {
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.headline)
                            .foregroundColor(Colors.Semantic.warning)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Profile Incomplete")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.primary)

                            Text("Complete your profile for better insights")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Text("\(profileChecker.completionPercentage)%")
                            .font(.headline)
                            .foregroundColor(Colors.Semantic.warning)
                    }

                    // Progress bar
                    ProgressView(value: Double(profileChecker.completionPercentage) / 100.0)
                        .progressViewStyle(LinearProgressViewStyle(tint: Colors.Semantic.warning))
                }
                .padding()
                .background(Colors.Semantic.warning.opacity(0.1))
                .cornerRadius(12)
            }
            .buttonStyle(.plain)
        }
    }
}

/// Checks profile completeness for ML feature extraction
@MainActor
class ProfileCompletenessChecker: ObservableObject {
    @Published var isComplete: Bool = false
    @Published var completionPercentage: Int = 0

    init() {
        checkProfile()
    }

    func checkProfile() {
        let context = InflamAIPersistenceController.shared.container.viewContext
        let request: NSFetchRequest<UserProfile> = UserProfile.fetchRequest()
        request.fetchLimit = 1

        guard let profile = try? context.fetch(request).first else {
            isComplete = false
            completionPercentage = 0
            return
        }

        var complete = 0
        let total = 6

        // Gender
        let gender = profile.gender ?? ""
        if !gender.isEmpty && gender.lowercased() != "unknown" {
            complete += 1
        }

        // Height
        if profile.heightCm > 0 {
            complete += 1
        }

        // Weight
        if profile.weightKg > 0 {
            complete += 1
        }

        // Smoking status
        if profile.smokingStatus != nil && !profile.smokingStatus!.isEmpty {
            complete += 1
        }

        // Date of birth
        if profile.dateOfBirth != nil {
            complete += 1
        }

        // Diagnosis date
        if profile.diagnosisDate != nil && !Calendar.current.isDateInToday(profile.diagnosisDate!) {
            complete += 1
        }

        completionPercentage = Int((Double(complete) / Double(total)) * 100)
        isComplete = complete >= 3 // At minimum need gender, height, weight
    }
}

/// Section 9.1: Quick Action Cards - white background with semantic-colored icons only
struct QuickActionCard: View {
    let icon: String
    let title: String
    let color: Color
    let action: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button(action: {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            action()
        }) {
            VStack(spacing: Spacing.sm) {
                // Icon without colored background per Section 9.1
                Image(systemName: icon)
                    .font(.system(size: 28))
                    .foregroundColor(color)
                    .frame(width: 52, height: 52)

                Text(title)
                    .font(.system(size: Typography.base, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, Spacing.lg)
            .background(
                RoundedRectangle(cornerRadius: Radii.xl)
                    .fill(Color.white)
            )
            .dshadow(isPressed ? Shadows.xs : Shadows.sm)
            .scaleEffect(isPressed ? 0.98 : 1.0)
        }
        .buttonStyle(.plain)
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    withAnimation(Animations.easeOut) { isPressed = true }
                }
                .onEnded { _ in
                    withAnimation(Animations.easeOut) { isPressed = false }
                }
        )
    }
}

/// Section 9.1: Quick Action Cards - white background with semantic-colored icons only
struct QuickActionCardView: View {
    let icon: String
    let title: String
    let color: Color

    var body: some View {
        VStack(spacing: Spacing.sm) {
            // Icon without colored background per Section 9.1
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundColor(color)
                .frame(width: 52, height: 52)

            Text(title)
                .font(.system(size: Typography.base, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Spacing.lg)
        .background(
            RoundedRectangle(cornerRadius: Radii.xl)
                .fill(Color.white)
        )
        .dshadow(Shadows.sm)
    }
}

struct SummaryMetric: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: Spacing.sm) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.12))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .font(.system(size: 18))
                    .foregroundColor(color)
            }

            Text(value)
                .font(.system(size: Typography.xxl, weight: .bold, design: .rounded))
                .foregroundColor(Colors.Gray.g900)

            Text(label)
                .font(.system(size: Typography.xs, weight: .medium))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: Radii.lg)
                .fill(Color(.systemBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: Radii.lg)
                .stroke(Colors.Gray.g200, lineWidth: 1)
        )
    }
}

struct MedicationReminderRow: View {
    let medication: PendingMedication
    let onTake: () -> Void
    var onSkip: (() -> Void)? = nil
    var onRemindLater: (() -> Void)? = nil

    @State private var showActions = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: Spacing.sm) {
                // Medication icon - shows checkmark when taken
                ZStack {
                    Circle()
                        .fill(medication.isTaken ? Colors.Semantic.success.opacity(0.15) : Colors.Accent.purple.opacity(0.12))
                        .frame(width: 44, height: 44)

                    Image(systemName: medication.isTaken ? "checkmark.circle.fill" : "pill.fill")
                        .font(.system(size: medication.isTaken ? 22 : 18))
                        .foregroundColor(medication.isTaken ? Colors.Semantic.success : Colors.Accent.purple)
                }

                VStack(alignment: .leading, spacing: Spacing.xxxs) {
                    HStack(spacing: Spacing.xs) {
                        Text(medication.name)
                            .font(.system(size: Typography.base, weight: .semibold))
                            .foregroundColor(Colors.Gray.g900)

                        if medication.isTaken {
                            Text("Taken")
                                .font(.system(size: Typography.xxs, weight: .semibold))
                                .foregroundColor(Colors.Semantic.success)
                                .padding(.horizontal, Spacing.xs)
                                .padding(.vertical, Spacing.xxxs)
                                .background(Colors.Semantic.success.opacity(0.12))
                                .cornerRadius(Radii.xs)
                        }
                    }

                    HStack(spacing: Spacing.xxs) {
                        Image(systemName: "clock")
                            .font(.caption2)
                        Text(medication.time, style: .time)
                            .font(.system(size: Typography.xs))
                    }
                    .foregroundColor(Colors.Gray.g500)
                }

                Spacer()

                if !medication.isTaken {
                    // Quick Take Button
                    Button {
                        HapticFeedback.success()
                        withAnimation(MotionSpring.snappy) {
                            onTake()
                        }
                    } label: {
                        Text("Take")
                            .font(.system(size: Typography.sm, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(.horizontal, Spacing.md)
                            .padding(.vertical, Spacing.xs)
                            .background(
                                Capsule()
                                    .fill(
                                        LinearGradient(
                                            colors: [Colors.Accent.purple, Colors.Primary.p500],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                            )
                    }
                    .buttonStyle(PressableButtonStyle())

                    // More options button
                    Button {
                        HapticFeedback.light()
                        withAnimation(MotionSpring.snappy) {
                            showActions.toggle()
                        }
                    } label: {
                        Image(systemName: "ellipsis")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundColor(Colors.Gray.g500)
                            .frame(width: 32, height: 32)
                            .background(Circle().fill(Colors.Gray.g100))
                    }
                } else {
                    // Show completion indicator
                    Image(systemName: "checkmark")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(Colors.Semantic.success)
                        .padding(10)
                        .background(Circle().fill(Colors.Semantic.success.opacity(0.12)))
                }
            }
            .padding(Spacing.md)

            // Expandable action buttons
            if showActions && !medication.isTaken {
                HStack(spacing: Spacing.sm) {
                    // Skip button
                    Button {
                        HapticFeedback.warning()
                        withAnimation(MotionSpring.snappy) {
                            showActions = false
                            onSkip?()
                        }
                    } label: {
                        HStack(spacing: Spacing.xxs) {
                            Image(systemName: "forward.fill")
                                .font(.system(size: 12))
                            Text("Skip")
                                .font(.system(size: Typography.sm, weight: .medium))
                        }
                        .foregroundColor(Colors.Semantic.warning)
                        .padding(.horizontal, Spacing.md)
                        .padding(.vertical, Spacing.xs)
                        .background(
                            Capsule()
                                .stroke(Colors.Semantic.warning, lineWidth: 1.5)
                        )
                    }
                    .buttonStyle(PressableButtonStyle())

                    // Remind Later button
                    Button {
                        HapticFeedback.light()
                        withAnimation(MotionSpring.snappy) {
                            showActions = false
                            onRemindLater?()
                        }
                    } label: {
                        HStack(spacing: Spacing.xxs) {
                            Image(systemName: "bell.badge")
                                .font(.system(size: 12))
                            Text("Remind Later")
                                .font(.system(size: Typography.sm, weight: .medium))
                        }
                        .foregroundColor(Colors.Primary.p500)
                        .padding(.horizontal, Spacing.md)
                        .padding(.vertical, Spacing.xs)
                        .background(
                            Capsule()
                                .stroke(Colors.Primary.p500, lineWidth: 1.5)
                        )
                    }
                    .buttonStyle(PressableButtonStyle())

                    Spacer()
                }
                .padding(.horizontal, Spacing.md)
                .padding(.bottom, Spacing.md)
                .transition(MotionDirection.riseUp)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: Radii.lg)
                .fill(medication.isTaken ? Colors.Semantic.success.opacity(0.05) : Color.white)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Radii.lg)
                .stroke(medication.isTaken ? Colors.Semantic.success.opacity(0.2) : Colors.Gray.g200, lineWidth: 1)
        )
        .shadow(color: Colors.Accent.purple.opacity(0.08), radius: 6, y: 3)
        .animation(Animations.easeOut, value: medication.isTaken)
    }
}

struct TrendIndicator: View {
    let label: String
    let trend: HomeTrendDirection
    let value: Double

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)

            Spacer()

            HStack(spacing: Spacing.xs) {
                // Value display
                Text(String(format: "%.1f", value))
                    .font(.system(size: Typography.md, weight: .semibold, design: .rounded))
                    .foregroundColor(Colors.Gray.g900)

                // Trend indicator badge
                HStack(spacing: Spacing.xxs) {
                    Image(systemName: trend.icon)
                        .font(.system(size: 10, weight: .bold))

                    Text(trend.label)
                        .font(.system(size: Typography.xxs, weight: .semibold))
                }
                .foregroundColor(trend.color)
                .padding(.horizontal, Spacing.xs)
                .padding(.vertical, Spacing.xxs)
                .background(
                    Capsule()
                        .fill(trend.color.opacity(0.12))
                )
            }
        }
        .padding(.vertical, Spacing.xxxs)
    }
}

// MARK: - View Model

@MainActor
class HomeViewModel: ObservableObject {
    @Published var currentStreak = 0
    @Published var hasLoggedToday = false
    @Published var todayPainEntries = 0
    @Published var todayAssessments = 0
    @Published var todayJournalEntries = 0
    @Published var activeMedicationsCount = 0
    @Published var todayBASDAI = "--"
    @Published var todayPain = "--"
    @Published var todayMobility = "--"
    @Published var hasPendingMedications = false
    @Published var pendingMedications: [PendingMedication] = []
    @Published var hasActiveFlare = false
    @Published var weeklyTrendSummary: WeeklyTrendSummary?
    @Published var exerciseSuggestion = "Start your daily routine"
    @Published var activeRoutine: UserRoutine?
    @Published var dueQuestionnaires: [QuestionnaireID] = []
    @Published var shouldShowQuestionnaires = false

    private let context: NSManagedObjectContext
    private let preferences = QuestionnaireUserPreferences.shared

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    var viewContext: NSManagedObjectContext {
        return context
    }

    func scheduleDescription(for questionnaireID: QuestionnaireID) -> String {
        let schedule = preferences.getSchedule(for: questionnaireID)
        switch schedule.frequency {
        case .daily: return "Daily"
        case .weekly: return "Weekly"
        case .monthly: return "Monthly"
        case .onDemand: return "On-Demand"
        }
    }

    var greeting: String {
        let hour = Calendar.current.component(.hour, from: Date())
        switch hour {
        case 0..<12: return "Good Morning"
        case 12..<17: return "Good Afternoon"
        default: return "Good Evening"
        }
    }

    func loadDashboardData() async {
        // Load only the most critical data first
        await loadTodaysData()
        await loadQuestionnaires()

        // Load less critical data afterwards
        Task.detached(priority: .background) { [weak self] in
            guard let self = self else { return }
            await self.loadStreak()
            await self.loadMedications()
            await self.loadActiveFlare()
            await self.loadWeeklyTrends()
            await self.loadActiveRoutine()
            await self.generateExerciseSuggestion()
        }
    }

    private func loadQuestionnaires() async {
        let manager = QuestionnaireManager(viewContext: context)
        let due = preferences.getDueQuestionnaires(using: manager)

        dueQuestionnaires = due
        shouldShowQuestionnaires = !due.isEmpty || preferences.enabledCount > 0
    }

    private func loadTodaysData() async {
        let today = Calendar.current.startOfDay(for: Date())
        let tomorrow = Calendar.current.date(byAdding: .day, value: 1, to: today)!

        let (painCount, assessmentCount, journalCount, medicationCount, hasData) = await context.perform {
            // Count pain entries
            let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
            painRequest.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@",
                                               today as NSDate, tomorrow as NSDate)
            let painEntries = (try? self.context.count(for: painRequest)) ?? 0

            // Count BASDAI assessments
            let basdaiRequest: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
            basdaiRequest.predicate = NSPredicate(format: "date >= %@ AND date < %@",
                                                  today as NSDate, tomorrow as NSDate)
            let assessments = (try? self.context.count(for: basdaiRequest)) ?? 0

            // Count journal entries
            let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
            journalRequest.predicate = NSPredicate(format: "date >= %@ AND date < %@",
                                                   today as NSDate, tomorrow as NSDate)
            let journals = (try? self.context.count(for: journalRequest)) ?? 0

            // Count active medications
            let medicationRequest: NSFetchRequest<Medication> = Medication.fetchRequest()
            medicationRequest.predicate = NSPredicate(format: "isActive == YES")
            let medications = (try? self.context.count(for: medicationRequest)) ?? 0

            let hasAnyData = painEntries > 0 || assessments > 0 || journals > 0

            return (painEntries, assessments, journals, medications, hasAnyData)
        }

        // Update on main actor
        self.todayPainEntries = painCount
        self.todayAssessments = assessmentCount
        self.todayJournalEntries = journalCount
        self.activeMedicationsCount = medicationCount
        self.hasLoggedToday = hasData
    }

    private func loadStreak() async {
        // Calculate logging streak
        var streak = 0
        var currentDate = Calendar.current.startOfDay(for: Date())

        await context.perform {
            for _ in 0..<365 {
                let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
                let nextDay = Calendar.current.date(byAdding: .day, value: 1, to: currentDate)!
                request.predicate = NSPredicate(format: "timestamp >= %@ AND timestamp < %@",
                                                currentDate as NSDate, nextDay as NSDate)

                if let count = try? self.context.count(for: request), count > 0 {
                    streak += 1
                    currentDate = Calendar.current.date(byAdding: .day, value: -1, to: currentDate)!
                } else {
                    break
                }
            }
        }

        currentStreak = streak
    }

    private func loadMedications() async {
        await context.perform {
            // Fetch active medications
            let request: NSFetchRequest<Medication> = Medication.fetchRequest()
            request.predicate = NSPredicate(format: "isActive == YES")
            request.sortDescriptors = [NSSortDescriptor(keyPath: \Medication.name, ascending: true)]

            guard let medications = try? self.context.fetch(request) else {
                Task { @MainActor in
                    self.pendingMedications = []
                    self.hasPendingMedications = false
                }
                return
            }

            // Filter medications that are due today and map to PendingMedication
            var pending: [PendingMedication] = []
            let calendar = Calendar.current
            let now = Date()

            for med in medications {
                // If medication has reminder times, check if any are upcoming today
                if let reminderData = med.reminderTimes,
                   let reminderTimes = try? JSONDecoder().decode([Date].self, from: reminderData) {

                    // Find the next reminder time
                    for reminderTime in reminderTimes {
                        // Check if this reminder is today and in the future or recent past (within last hour)
                        if calendar.isDateInToday(reminderTime) {
                            let hourAgo = calendar.date(byAdding: .hour, value: -1, to: now)!
                            if reminderTime >= hourAgo {
                                pending.append(PendingMedication(
                                    name: med.name ?? "Unnamed Medication",
                                    time: reminderTime
                                ))
                                break // Only add once per medication
                            }
                        }
                    }
                } else if med.reminderEnabled {
                    // If reminders are enabled but no specific times, show the medication
                    pending.append(PendingMedication(
                        name: med.name ?? "Unnamed Medication",
                        time: now
                    ))
                }
            }

            // Sort by time
            pending.sort { $0.time < $1.time }

            Task { @MainActor in
                self.pendingMedications = pending
                self.hasPendingMedications = !pending.isEmpty
            }
        }
    }

    func markMedicationTaken(_ medication: PendingMedication) {
        // Immediate visual feedback - update local state
        if let index = pendingMedications.firstIndex(where: { $0.id == medication.id }) {
            pendingMedications[index].isTaken = true
        }

        // Save to Core Data
        Task {
            await context.perform {
                // Create a dose log
                let doseLog = DoseLog(context: self.context)
                doseLog.id = UUID()
                doseLog.timestamp = Date()
                doseLog.scheduledTime = medication.time
                doseLog.wasSkipped = false

                do {
                    try self.context.save()
                    print("✅ Medication '\(medication.name)' marked as taken")
                } catch {
                    print("❌ Failed to save medication dose: \(error)")
                }
            }
        }
    }

    func skipMedication(_ medication: PendingMedication) {
        // Remove from pending list
        pendingMedications.removeAll { $0.id == medication.id }
        hasPendingMedications = !pendingMedications.isEmpty

        // Save skip to Core Data
        Task {
            await context.perform {
                let doseLog = DoseLog(context: self.context)
                doseLog.id = UUID()
                doseLog.timestamp = Date()
                doseLog.scheduledTime = medication.time
                doseLog.wasSkipped = true

                do {
                    try self.context.save()
                    print("⏭️ Medication '\(medication.name)' skipped")
                } catch {
                    print("❌ Failed to save skipped dose: \(error)")
                }
            }
        }
    }

    func remindMedicationLater(_ medication: PendingMedication) {
        // Update the time to 30 minutes from now
        if let index = pendingMedications.firstIndex(where: { $0.id == medication.id }) {
            let newTime = Calendar.current.date(byAdding: .minute, value: 30, to: Date()) ?? Date()
            pendingMedications[index] = PendingMedication(
                id: medication.id,
                name: medication.name,
                time: newTime,
                isTaken: false
            )
            // Re-sort by time
            pendingMedications.sort { $0.time < $1.time }
            print("🔔 Medication '\(medication.name)' reminder set for 30 min later")
        }
    }

    private func loadActiveFlare() async {
        await context.perform {
            let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
            request.predicate = NSPredicate(format: "endDate == nil")

            if let count = try? self.context.count(for: request), count > 0 {
                Task { @MainActor in
                    self.hasActiveFlare = true
                }
            }
        }
    }

    private func loadWeeklyTrends() async {
        let weekAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date())!

        await context.perform {
            let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
            request.predicate = NSPredicate(format: "timestamp >= %@", weekAgo as NSDate)

            if let logs = try? self.context.fetch(request), logs.count >= 3 {
                let avgPain = logs.map { $0.basdaiScore }.reduce(0.0, +) / Double(max(logs.count, 1))
                let avgStiffness = Double(logs.map { $0.morningStiffnessMinutes }.reduce(0, +)) / Double(max(logs.count, 1))
                let avgFatigue = Double(logs.map { $0.fatigueLevel }.reduce(0, +)) / Double(max(logs.count, 1))

                Task { @MainActor in
                    self.weeklyTrendSummary = WeeklyTrendSummary(
                        painTrend: .stable,
                        avgPain: avgPain,
                        stiffnessTrend: .stable,
                        avgStiffness: avgStiffness,
                        fatigueTrend: .stable,
                        avgFatigue: avgFatigue
                    )
                }
            }
        }
    }

    private func loadActiveRoutine() async {
        await context.perform {
            let request: NSFetchRequest<UserRoutine> = UserRoutine.fetchRequest()
            request.predicate = NSPredicate(format: "isActive == YES")
            request.fetchLimit = 1

            if let routines = try? self.context.fetch(request), let active = routines.first {
                Task { @MainActor in
                    self.activeRoutine = active
                }
            }
        }
    }

    private func generateExerciseSuggestion() async {
        let suggestions = [
            "Try 5 minutes of stretching",
            "Start with gentle mobility exercises",
            "Focus on breathing exercises today",
            "Work on posture improvement",
            "Balance exercises for stability"
        ]

        exerciseSuggestion = suggestions.randomElement() ?? "Browse exercise library"
    }
}

// MARK: - Data Models

struct PendingMedication: Identifiable {
    let id: UUID
    let name: String
    let time: Date
    var isTaken: Bool = false

    init(id: UUID = UUID(), name: String, time: Date, isTaken: Bool = false) {
        self.id = id
        self.name = name
        self.time = time
        self.isTaken = isTaken
    }
}

struct WeeklyTrendSummary {
    let painTrend: HomeTrendDirection
    let avgPain: Double
    let stiffnessTrend: HomeTrendDirection
    let avgStiffness: Double
    let fatigueTrend: HomeTrendDirection
    let avgFatigue: Double
}

enum HomeTrendDirection {
    case improving
    case stable
    case worsening

    var icon: String {
        switch self {
        case .improving: return "arrow.down"
        case .stable: return "minus"
        case .worsening: return "arrow.up"
        }
    }

    var label: String {
        switch self {
        case .improving: return "Better"
        case .stable: return "Stable"
        case .worsening: return "Worse"
        }
    }

    var color: Color {
        switch self {
        case .improving: return Colors.Semantic.success
        case .stable: return Colors.Primary.p500
        case .worsening: return Colors.Semantic.error
        }
    }
}

enum DashboardTab {
    case overview
    case trends
    case exercises
    case medications
}

// MARK: - Preview

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
