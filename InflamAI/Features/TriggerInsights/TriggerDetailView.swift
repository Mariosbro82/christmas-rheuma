//
//  TriggerDetailView.swift
//  InflamAI
//
//  Detailed view for a specific trigger analysis result
//  Shows statistical details, lag analysis, and recommendations
//

import SwiftUI
import Charts

// NOTE: String.displayName extension moved to
// InflamAI/Extensions/StringExtensions.swift for project-wide use

struct TriggerDetailView: View {
    let trigger: UnifiedTriggerResult
    @State private var explanation: TriggerExplanation?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: Spacing.xl) {
                    // Header
                    headerSection

                    // Effect Summary
                    effectSummarySection

                    // Statistical Details
                    if let stat = trigger.statisticalResult {
                        statisticalDetailsSection(stat)
                    }

                    // Lag Analysis Chart
                    if let stat = trigger.statisticalResult, !stat.laggedResults.isEmpty {
                        lagAnalysisSection(stat.laggedResults)
                    }

                    // k-NN Insights
                    if let knn = trigger.knnResult {
                        knnInsightsSection(knn)
                    }

                    // Explanation
                    if let explanation = explanation {
                        explanationSection(explanation)
                    }

                    // Medical Disclaimer
                    disclaimerSection
                }
                .padding()
            }
            // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
            .navigationTitle(trigger.triggerName.displayName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .task {
                explanation = await TriggerAnalysisService.shared.getExplanation(for: trigger.triggerName)
            }
        }
    }

    // MARK: - Header Section

    private var headerSection: some View {
        HStack(spacing: Spacing.md) {
            Image(systemName: trigger.icon)
                .font(.system(size: Typography.xxxl))
                .foregroundStyle(trigger.triggerCategory.color)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                Text(trigger.triggerName.displayName)
                    .font(.system(size: Typography.xl, weight: .bold))
                    .foregroundColor(Colors.Gray.g900)

                Text(trigger.triggerCategory.displayName)
                    .font(.system(size: Typography.sm))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()

            ConfidenceBadge(confidence: trigger.ensembleConfidence)
        }
        .padding(Spacing.md)
        .background(trigger.triggerCategory.color.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: Radii.xl))
    }

    // MARK: - Effect Summary Section

    private var effectSummarySection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Effect on Symptoms")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            HStack {
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(trigger.effectDescription)
                        .font(.system(size: Typography.lg, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    if let stat = trigger.statisticalResult {
                        Text(stat.impactDescription)
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g500)
                    }
                }

                Spacer()

                if trigger.isSignificant {
                    Label("Significant", systemImage: "checkmark.seal.fill")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Semantic.success)
                        .padding(.horizontal, Spacing.sm)
                        .padding(.vertical, Spacing.xxs)
                        .background(Colors.Semantic.success.opacity(0.1))
                        .clipShape(Capsule())
                }
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }

    // MARK: - Statistical Details

    private func statisticalDetailsSection(_ stat: StatisticalTriggerResult) -> some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            Text("Statistical Analysis")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.md) {
                TriggerStatCard(title: "Days Analyzed", value: "\(stat.totalDays)")
                TriggerStatCard(title: "Trigger Present", value: "\(stat.triggerDays) days")
                TriggerStatCard(title: "Avg Pain With", value: String(format: "%.1f", stat.effectSize.meanWithTrigger))
                TriggerStatCard(title: "Avg Pain Without", value: String(format: "%.1f", stat.effectSize.meanWithoutTrigger))
            }

            HStack {
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Effect Size (Cohen's d)")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                    Text(String(format: "%.2f", stat.effectSize.cohenD))
                        .font(.system(size: Typography.lg, weight: .bold))
                        .foregroundColor(Colors.Gray.g900)
                    Text(stat.effectSize.cohenDInterpretation)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: Spacing.xxs) {
                    Text("P-Value")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                    Text(stat.correctedPValue < 0.001 ? "< 0.001" : String(format: "%.3f", stat.correctedPValue))
                        .font(.system(size: Typography.lg, weight: .bold))
                        .foregroundColor(Colors.Gray.g900)
                    Text(stat.isSignificant ? "Significant" : "Not Significant")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(stat.isSignificant ? Colors.Semantic.success : Colors.Semantic.warning)
                }
            }
            .padding(Spacing.md)
            .background(Colors.Gray.g100)
            .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
        }
    }

    // MARK: - Lag Analysis Chart

    private func lagAnalysisSection(_ lagResults: [LaggedCorrelationResult]) -> some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Timing Analysis")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text("How long after exposure do symptoms change?")
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)

            Chart(lagResults) { result in
                BarMark(
                    x: .value("Lag", result.lagDescription),
                    y: .value("Correlation", abs(result.correlation))
                )
                .foregroundStyle(result.isSignificant ? Colors.Primary.p500 : Colors.Gray.g300)
            }
            .frame(height: 150)
            .chartYAxis {
                AxisMarks(position: .leading)
            }

            if let best = lagResults.min(by: { $0.pValue < $1.pValue }), best.isSignificant {
                HStack(spacing: Spacing.sm) {
                    Image(systemName: "clock.fill")
                        .foregroundStyle(Colors.Primary.p500)
                    Text("Strongest effect: \(best.lagDescription)")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g700)
                }
                .padding(Spacing.sm)
                .background(Colors.Primary.p500.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: Radii.sm))
            }
        }
    }

    // MARK: - k-NN Insights

    private func knnInsightsSection(_ knn: KNNTriggerResult) -> some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Similar Day Analysis")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text("Days with similar conditions")
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)

            ForEach(knn.similarDays.prefix(3)) { day in
                SimilarDayRow(day: day)
            }
        }
    }

    // MARK: - Explanation Section

    private func explanationSection(_ explanation: TriggerExplanation) -> some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Summary")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text(explanation.summary)
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g700)

            Divider()

            Text("Recommendation")
                .font(.system(size: Typography.sm, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text(explanation.recommendation)
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g600)

            if !explanation.caveats.isEmpty {
                Divider()

                Text("Notes")
                    .font(.system(size: Typography.sm, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                ForEach(explanation.caveats, id: \.self) { caveat in
                    HStack(alignment: .top, spacing: Spacing.sm) {
                        Image(systemName: "info.circle")
                            .foregroundColor(Colors.Gray.g500)
                        Text(caveat)
                            .font(.system(size: Typography.xs))
                            .foregroundColor(Colors.Gray.g600)
                    }
                }
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }

    // MARK: - Disclaimer

    private var disclaimerSection: some View {
        HStack(alignment: .top, spacing: Spacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Colors.Semantic.warning)

            Text("This analysis is for informational purposes only and does not constitute medical advice. Always discuss changes with your rheumatologist.")
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g600)
        }
        .padding(Spacing.md)
        .background(Colors.Semantic.warning.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: Radii.lg))
    }
}

// MARK: - Supporting Views

struct TriggerStatCard: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xxs) {
            Text(title)
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)
            Text(value)
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.sm))
    }
}

struct SimilarDayRow: View {
    let day: SimilarDay

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(day.date.formatted(date: .abbreviated, time: .omitted))
                    .font(.system(size: Typography.sm, weight: .medium))
                    .foregroundColor(Colors.Gray.g900)
                Text(day.similarityDescription)
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()

            PainLevelIndicator(level: day.painLevel)
                .scaleEffect(0.7)
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .clipShape(RoundedRectangle(cornerRadius: Radii.sm))
    }
}

#Preview {
    TriggerDetailView(trigger: UnifiedTriggerResult(
        triggerName: "Coffee",
        triggerCategory: .food,
        icon: "cup.and.saucer.fill",
        statisticalResult: nil,
        knnResult: nil,
        neuralResult: nil,
        ensembleScore: 0.5,
        ensembleConfidence: .medium,
        primaryEngine: .statistical,
        activeEngines: [.statistical]
    ))
}
