//
//  TriggerInsightsView.swift
//  InflamAI
//
//  Main view for the Hybrid Trigger Detection System
//  Shows personalized trigger analysis, recommendations, and predictions
//

import SwiftUI

// NOTE: String.displayName extension moved to
// InflamAI/Extensions/StringExtensions.swift for project-wide use

struct TriggerInsightsView: View {
    @StateObject private var viewModel = TriggerInsightsViewModel()
    @State private var selectedCategory: TriggerCategory? = nil
    @State private var showingTriggerDetail: UnifiedTriggerResult? = nil
    @State private var showingNeuralOptIn: Bool = false

    var body: some View {
        // CRIT-001 FIX: Removed NavigationStack wrapper.
        // This view is presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        // Nested navigation containers cause duplicate back arrows.
        ScrollView {
            VStack(spacing: Spacing.lg) {
                // Phase Progress Header
                phaseProgressSection

                // Prediction Card (if available)
                if viewModel.tomorrowPrediction?.hasPrediction == true {
                    predictionCard
                }

                // Recommendations Section
                if !viewModel.recommendations.isEmpty {
                    recommendationsSection
                }

                // Top Triggers Section
                topTriggersSection

                // Category Filter
                categoryFilterSection

                // All Triggers List
                triggersListSection

                // Neural Opt-In Banner
                if viewModel.currentPhase == .knn && viewModel.daysOfData >= 90 {
                    neuralOptInBanner
                }

                // Compliance Disclaimer
                complianceDisclaimer
            }
            .padding()
        }
        .navigationTitle("Patterns in Your Data")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task {
                        await viewModel.refresh()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(viewModel.isLoading)
            }
        }
        .refreshable {
            await viewModel.refresh()
        }
        .task {
            await viewModel.loadData()
        }
        .sheet(item: $showingTriggerDetail) { trigger in
            TriggerDetailView(trigger: trigger)
        }
        .sheet(isPresented: $showingNeuralOptIn) {
            NeuralOptInView(viewModel: viewModel)
        }
    }

    // MARK: - Phase Progress Section

    private var phaseProgressSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundStyle(Colors.Primary.p500)

                VStack(alignment: .leading) {
                    Text(viewModel.currentPhase.displayName)
                        .font(.system(size: Typography.md, weight: .semibold))
                    Text("\(viewModel.daysOfData) days of data")
                        .font(.system(size: Typography.xs))
                        .foregroundStyle(Colors.Gray.g500)
                }

                Spacer()

                if let progress = viewModel.progressToNextPhase {
                    CircularProgressView(progress: progress)
                        .frame(width: 44, height: 44)
                }
            }

            if let daysUntil = viewModel.daysUntilNextPhase, daysUntil > 0 {
                Text("\(daysUntil) more days until \(viewModel.nextPhaseName)")
                    .font(.system(size: Typography.xs))
                    .foregroundStyle(Colors.Gray.g500)
            }

            // Active engines indicator
            HStack(spacing: Spacing.xs) {
                ForEach(viewModel.activeEngines, id: \.self) { engine in
                    EngineChip(engine: engine, isActive: true)
                }
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }

    // MARK: - Prediction Card

    private var predictionCard: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            HStack {
                Image(systemName: "sparkles")
                    .foregroundStyle(Colors.Accent.purple)
                Text("Tomorrow's Outlook")
                    .font(.system(size: Typography.md, weight: .semibold))
            }

            if let prediction = viewModel.tomorrowPrediction {
                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: Spacing.xs) {
                        Text(prediction.predictionLevel ?? "Unknown")
                            .font(.system(size: Typography.xxxl, weight: .bold))

                        Text(prediction.explanation)
                            .font(.system(size: Typography.xs))
                            .foregroundStyle(Colors.Gray.g500)
                    }

                    Spacer()

                    if let pred = prediction.ensemblePrediction {
                        PainLevelIndicator(level: pred)
                    }
                }
            }
        }
        .padding(Spacing.md)
        .background(
            LinearGradient(
                colors: [Colors.Accent.purple.opacity(0.1), Colors.Primary.p500.opacity(0.1)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }

    // MARK: - Recommendations Section

    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Recommendations")
                .font(.system(size: Typography.md, weight: .semibold))

            ForEach(viewModel.recommendations.prefix(3)) { rec in
                RecommendationRow(recommendation: rec)
            }
        }
    }

    // MARK: - Top Triggers Section

    private var topTriggersSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Frequently Observed Correlations")
                .font(.system(size: Typography.md, weight: .semibold))

            if viewModel.topTriggers.isEmpty {
                EmptyTriggersView()
            } else {
                ForEach(viewModel.topTriggers.prefix(5)) { trigger in
                    TriggerRow(trigger: trigger)
                        .onTapGesture {
                            UISelectionFeedbackGenerator().selectionChanged()
                            showingTriggerDetail = trigger
                        }
                }
            }
        }
    }

    // MARK: - Category Filter

    private var categoryFilterSection: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: Spacing.sm) {
                CategoryChip(
                    category: nil,
                    isSelected: selectedCategory == nil,
                    action: { selectedCategory = nil }
                )

                ForEach(TriggerCategory.allCases) { category in
                    CategoryChip(
                        category: category,
                        isSelected: selectedCategory == category,
                        action: { selectedCategory = category }
                    )
                }
            }
        }
    }

    // MARK: - Triggers List

    private var triggersListSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("All Observed Patterns")
                .font(.system(size: Typography.md, weight: .semibold))

            let filteredTriggers = selectedCategory == nil
                ? viewModel.allTriggers
                : viewModel.allTriggers.filter { $0.triggerCategory == selectedCategory }

            if filteredTriggers.isEmpty {
                Text("No patterns in this category")
                    .font(.system(size: Typography.sm))
                    .foregroundStyle(Colors.Gray.g500)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(Spacing.md)
            } else {
                ForEach(filteredTriggers) { trigger in
                    TriggerRow(trigger: trigger)
                        .onTapGesture {
                            UISelectionFeedbackGenerator().selectionChanged()
                            showingTriggerDetail = trigger
                        }
                }
            }
        }
    }

    // MARK: - Neural Opt-In Banner

    private var neuralOptInBanner: some View {
        Button {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            showingNeuralOptIn = true
        } label: {
            HStack(spacing: Spacing.md) {
                Image(systemName: "brain")
                    .font(.system(size: Typography.xl))
                    .foregroundStyle(Colors.Accent.purple)

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Neural Network Available")
                        .font(.system(size: Typography.md, weight: .semibold))
                    Text("Enable advanced pattern detection")
                        .font(.system(size: Typography.xs))
                        .foregroundStyle(Colors.Gray.g500)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: Typography.sm))
                    .foregroundStyle(Colors.Gray.g400)
            }
            .padding(Spacing.md)
            .background(Colors.Accent.purple.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Compliance Disclaimer

    private var complianceDisclaimer: some View {
        VStack(spacing: Spacing.xs) {
            HStack(spacing: Spacing.xs) {
                Image(systemName: "info.circle.fill")
                    .foregroundStyle(Colors.Gray.g400)
                    .font(.caption)
                Text("Important Information")
                    .font(.system(size: Typography.xs, weight: .semibold))
                    .foregroundStyle(Colors.Gray.g600)
                Spacer()
            }

            Text("Correlation ≠ Causation. These patterns are statistical observations from your logged data, not medical diagnoses. Discuss any findings with your healthcare provider before making changes to your routine or treatment.")
                .font(.system(size: Typography.xxs))
                .foregroundStyle(Colors.Gray.g500)
                .multilineTextAlignment(.leading)
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.md))
        .padding(.top, Spacing.md)
    }
}

// MARK: - Supporting Views

struct CircularProgressView: View {
    let progress: Double

    var body: some View {
        ZStack {
            Circle()
                .stroke(Colors.Gray.g200, lineWidth: 4)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(Colors.Primary.p500, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))

            Text("\(Int(progress * 100))%")
                .font(.system(size: Typography.xxs, weight: .bold))
                .foregroundStyle(Colors.Gray.g700)
        }
    }
}

struct EngineChip: View {
    let engine: EngineType
    let isActive: Bool

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            Image(systemName: engine.icon)
                .font(.system(size: Typography.xxs))

            Text(engine.displayName)
                .font(.system(size: Typography.xxs))
        }
        .padding(.horizontal, Spacing.sm)
        .padding(.vertical, Spacing.xxs)
        .background(isActive ? Colors.Primary.p500.opacity(0.1) : Colors.Gray.g100)
        .foregroundStyle(isActive ? Colors.Primary.p500 : Colors.Gray.g500)
        .clipShape(Capsule())
    }
}

struct CategoryChip: View {
    let category: TriggerCategory?
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: {
            UISelectionFeedbackGenerator().selectionChanged()
            action()
        }) {
            HStack(spacing: Spacing.xxs) {
                if let category = category {
                    Image(systemName: category.icon)
                }
                Text(category?.displayName ?? "All")
            }
            .font(.system(size: Typography.xs))
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.sm)
            .background(isSelected ? Colors.Primary.p500 : Colors.Gray.g100)
            .foregroundStyle(isSelected ? .white : Colors.Gray.g700)
            .clipShape(Capsule())
        }
    }
}

struct TriggerRow: View {
    let trigger: UnifiedTriggerResult

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: trigger.icon)
                .font(.system(size: Typography.lg))
                .foregroundStyle(trigger.triggerCategory.color)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                Text(trigger.triggerName.displayName)
                    .font(.system(size: Typography.sm, weight: .medium))

                Text(trigger.effectDescription)
                    .font(.system(size: Typography.xs))
                    .foregroundStyle(Colors.Gray.g500)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: Spacing.xxs) {
                ConfidenceBadge(confidence: trigger.ensembleConfidence)

                if trigger.isSignificant {
                    Text("Significant")
                        .font(.system(size: Typography.xxs))
                        .foregroundStyle(Colors.Semantic.success)
                }
            }

            Image(systemName: "chevron.right")
                .font(.system(size: Typography.xs))
                .foregroundStyle(Colors.Gray.g400)
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }
}

struct ConfidenceBadge: View {
    let confidence: TriggerConfidence

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            Image(systemName: confidence.icon)
            Text(confidence.displayName)
        }
        .font(.system(size: Typography.xxs))
        .foregroundStyle(confidence.color)
    }
}

struct RecommendationRow: View {
    let recommendation: TriggerRecommendation

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: recommendation.type.icon)
                .font(.system(size: Typography.lg))
                .foregroundStyle(recommendation.type.color)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(recommendation.title)
                    .font(.system(size: Typography.sm, weight: .medium))

                Text(recommendation.description)
                    .font(.system(size: Typography.xs))
                    .foregroundStyle(Colors.Gray.g500)
                    .lineLimit(2)
            }

            Spacer()
        }
        .padding(Spacing.md)
        .background(recommendation.type.color.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }
}

struct PainLevelIndicator: View {
    let level: Double

    var color: Color {
        switch level {
        case 0..<2: return Colors.Semantic.success
        case 2..<4: return Colors.Semantic.warning
        case 4..<6: return .orange
        case 6..<8: return Colors.Semantic.error
        default: return Colors.Accent.purple
        }
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(color.opacity(0.2))
                .frame(width: 60, height: 60)

            Text(String(format: "%.1f", level))
                .font(.system(size: Typography.xl, weight: .bold))
                .foregroundStyle(color)
        }
    }
}

struct EmptyTriggersView: View {
    var body: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "chart.bar.doc.horizontal")
                .font(.system(size: Typography.xxxl))
                .foregroundStyle(Colors.Gray.g400)

            Text("No pattern analysis yet")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundStyle(Colors.Gray.g700)

            Text("Log your daily data and symptoms to discover patterns. Correlation ≠ causation.")
                .font(.system(size: Typography.xs))
                .foregroundStyle(Colors.Gray.g500)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Spacing.xxl)
    }
}

// MARK: - Preview

#Preview {
    TriggerInsightsView()
}
