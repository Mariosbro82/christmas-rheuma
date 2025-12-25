//
//  QuestionnaireHomeSection.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-06-09.
//

import SwiftUI
import CoreData
import Charts

struct QuestionnaireHomeSection: View {
    @Environment(\.managedObjectContext) private var viewContext
    @State private var basdaiState: QuestionnaireDueState?
    @State private var basfiState: QuestionnaireDueState?
    @State private var basgState: QuestionnaireDueState?
    @State private var basdaiSamples: [QuestionnaireTrendSample] = []
    @State private var basfiSamples: [QuestionnaireTrendSample] = []
    @State private var basgSamples: [QuestionnaireTrendSample] = []
    @State private var isLoading = false
    @State private var loadError: String?
    @State private var showHistoryFor: QuestionnaireID?

    let openForm: (QuestionnaireID) -> Void
    
    private var context: NSManagedObjectContext {
        if viewContext.persistentStoreCoordinator != nil {
            return viewContext
        }
        return InflamAIPersistenceController.shared.container.viewContext
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            sectionHeader
            if isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity, alignment: .center)
            } else if let error = loadError {
                Text(error)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                basdaiCard
                weeklyCard
            }
        }
        .task {
            await loadData()
        }
        .refreshable {
            await loadData()
        }
        .sheet(item: $showHistoryFor) { questionnaireID in
            QuestionnaireHistoryView(questionnaireID: questionnaireID)
        }
    }
    
    private var sectionHeader: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.xs) {
            Text("questionnaire.section.title")
                .font(TraeTypography.title3)
                .foregroundStyle(TraePalette.graphite)
            Text("questionnaire.section.subtitle")
                .font(TraeTypography.footnote)
                .foregroundStyle(.secondary)
        }
    }
    
    private var basdaiCard: some View {
        Group {
            if let basdaiState = basdaiState {
                AssessmentCard(
                    state: basdaiState,
                    samples: basdaiSamples,
                    latestScore: basdaiSamples.last?.value,
                    title: Text(LocalizedStringResource("questionnaire.basdai.title")),
                    actionTitle: basdaiState.isDue ? Text("questionnaire.card.start_now") : Text("questionnaire.card.view_history"),
                    openForm: { openForm(.basdai) },
                    openHistory: { showHistoryFor = .basdai }
                )
            } else {
                EmptyAssessmentCard(title: Text(LocalizedStringResource("questionnaire.basdai.title"))) {
                    openForm(.basdai)
                }
            }
        }
    }
    
    private var weeklyCard: some View {
        Group {
            if let basfiState = basfiState, let basgState = basgState {
                WeeklyAssessmentCard(
                    basfiState: basfiState,
                    basgState: basgState,
                    basfiSamples: basfiSamples,
                    basgSamples: basgSamples,
                    openBasfi: { openForm(.basfi) },
                    openBasg: { openForm(.basg) },
                    openBasfiHistory: { showHistoryFor = .basfi },
                    openBasgHistory: { showHistoryFor = .basg }
                )
            } else {
                EmptyAssessmentCard(title: Text(LocalizedStringResource("questionnaire.basfi.title"))) {
                    openForm(.basfi)
                }
            }
        }
    }
    
    private func loadData() async {
        guard !isLoading else { return }
        await MainActor.run {
            isLoading = true
            loadError = nil
        }
        let manager = QuestionnaireManager(viewContext: context)
        await MainActor.run {
            basdaiState = manager.state(for: .basdai)
            basfiState = manager.state(for: .basfi)
            basgState = manager.state(for: .basg)
            
            basdaiSamples = manager.fetchRecentResponses(for: .basdai, limit: 30)
                .compactMap(QuestionnaireTrendSample.init)
            basfiSamples = manager.fetchRecentResponses(for: .basfi, limit: 24)
                .compactMap(QuestionnaireTrendSample.init)
            basgSamples = manager.fetchRecentResponses(for: .basg, limit: 24)
                .compactMap(QuestionnaireTrendSample.init)
            
            isLoading = false
        }
    }
}

private struct QuestionnaireTrendSample: Identifiable {
    let id = UUID()
    let date: Date
    let localDate: String
    let value: Double
    
    init?(response: QuestionnaireResponse) {
        guard let createdAt = response.createdAt as Date? else { return nil }
        self.date = createdAt
        self.localDate = response.localDate ?? ""
        self.value = response.score
    }
}

private struct AssessmentCard: View {
    let state: QuestionnaireDueState
    let samples: [QuestionnaireTrendSample]
    let latestScore: Double?
    let title: Text
    let actionTitle: Text
    let openForm: () -> Void
    let openHistory: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            HStack {
                VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                    HStack {
                        title
                            .font(TraeTypography.headline)
                        Spacer()
                        // History button - always visible when there are samples
                        if !samples.isEmpty {
                            Button(action: openHistory) {
                                Image(systemName: "chart.line.uptrend.xyaxis")
                                    .font(.system(size: 16))
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    statusLabel
                }
                Spacer()
                if let latestScore = latestScore {
                    Text(String(format: "%.1f", latestScore))
                        .font(.system(size: 40, weight: .bold, design: .rounded))
                        .foregroundStyle(scoreColor(for: latestScore))
                        .accessibilityLabel(
                            Text(String(format: NSLocalizedString("questionnaire.latest_score_accessibility", comment: ""), latestScore))
                        )
                } else {
                    Text("–")
                        .font(.system(size: 40, weight: .bold, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }
            
            if !samples.isEmpty {
                Chart(samples) { sample in
                    LineMark(
                        x: .value("Date", sample.date),
                        y: .value("Score", sample.value)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                    .foregroundStyle(scoreColor(for: sample.value))
                    
                    AreaMark(
                        x: .value("Date", sample.date),
                        yStart: .value("Baseline", 0),
                        yEnd: .value("Score", sample.value)
                    )
                    .foregroundStyle(scoreColor(for: sample.value).opacity(0.12))
                }
                .frame(height: 120)
                .chartXAxis {
                    AxisMarks(values: .automatic) { value in
                        AxisGridLine()
                        AxisValueLabel(formatter: .dateTime.day().month())
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .stride(by: 2)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let doubleValue = value.as(Double.self) {
                                Text(String(format: "%.0f", doubleValue))
                            }
                        }
                    }
                }
            } else {
                Text("questionnaire.no_history")
                    .foregroundStyle(.secondary)
            }
            
            PrimaryButton(title: actionTitle) {
                openForm()
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.05), radius: 8, x: 0, y: 4)
        )
        .accessibilityElement(children: .combine)
    }
    
    private var statusLabel: some View {
        Group {
            if state.isDue {
                Text("questionnaire.status.due")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(TraePalette.saffron)
            } else if let last = state.lastSubmission {
                Text(String(format: NSLocalizedString("questionnaire.status.completed_on", comment: ""),
                            DateFormatter.localizedString(from: last.createdAt ?? Date(), dateStyle: .medium, timeStyle: .short)))
                .font(TraeTypography.footnote)
                .foregroundStyle(.secondary)
            } else {
                Text("questionnaire.status.not_started")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
    
    private func scoreColor(for score: Double) -> Color {
        switch score {
        case ..<4: return Color.hex("#3FB16B")
        case 4..<6: return Color.hex("#FFC857")
        case 6..<8: return Color.hex("#FF8C42")
        default: return Color.hex("#D9534F")
        }
    }
}

private struct WeeklyAssessmentCard: View {
    let basfiState: QuestionnaireDueState
    let basgState: QuestionnaireDueState
    let basfiSamples: [QuestionnaireTrendSample]
    let basgSamples: [QuestionnaireTrendSample]
    let openBasfi: () -> Void
    let openBasg: () -> Void
    let openBasfiHistory: () -> Void
    let openBasgHistory: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            HStack {
                VStack(alignment: .leading, spacing: TraeSpacing.xs) {
                    Text(LocalizedStringResource("questionnaire.weekly.title"))
                        .font(TraeTypography.headline)
                    Text(LocalizedStringResource("questionnaire.weekly.subtitle"))
                        .font(TraeTypography.footnote)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            WeeklyMetricRow(
                title: Text(LocalizedStringResource("questionnaire.basfi.title")),
                state: basfiState,
                samples: basfiSamples,
                action: openBasfi,
                historyAction: openBasfiHistory
            )

            WeeklyMetricRow(
                title: Text(LocalizedStringResource("questionnaire.basg.title")),
                state: basgState,
                samples: basgSamples,
                action: openBasg,
                historyAction: openBasgHistory
            )
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.05), radius: 8, x: 0, y: 4)
        )
    }
}

private struct WeeklyMetricRow: View {
    let title: Text
    let state: QuestionnaireDueState
    let samples: [QuestionnaireTrendSample]
    let action: () -> Void
    let historyAction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.sm) {
            HStack {
                title
                    .font(TraeTypography.subheadline)
                Spacer()
                // History button when there are samples
                if !samples.isEmpty {
                    Button(action: historyAction) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                statusLabel
            }

            if !samples.isEmpty {
                Chart(samples) { sample in
                    LineMark(
                        x: .value("Date", sample.date),
                        y: .value("Score", sample.value)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
                    .foregroundStyle(TraePalette.graphite)
                }
                .frame(height: 80)
            } else {
                Text("questionnaire.no_history")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                PrimaryButton(title: Text(state.isDue ? "questionnaire.card.start_now" : "questionnaire.card.add_new")) {
                    action()
                }

                if !samples.isEmpty {
                    Button(action: historyAction) {
                        Text("History")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    private var statusLabel: some View {
        Group {
            if state.isDue {
                Text("questionnaire.status.due")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(TraePalette.saffron)
            } else if let last = state.lastSubmission {
                Text(last.createdAt ?? Date(), style: .date)
                    .font(TraeTypography.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("questionnaire.status.not_started")
                    .font(TraeTypography.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct EmptyAssessmentCard: View {
    let title: Text
    let action: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: TraeSpacing.md) {
            title
                .font(TraeTypography.headline)
            Text("questionnaire.no_history")
                .foregroundStyle(.secondary)
            PrimaryButton(title: "questionnaire.card.start_now") {
                action()
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.05), radius: 8, x: 0, y: 4)
        )
    }
}

private extension Color {
    static func hex(_ value: String) -> Color {
        var hex = value
        if hex.hasPrefix("#") {
            hex.removeFirst()
        }
        guard let int = UInt64(hex, radix: 16) else {
            return Color.accentColor
        }
        let r = Double((int >> 16) & 0xFF) / 255.0
        let g = Double((int >> 8) & 0xFF) / 255.0
        let b = Double(int & 0xFF) / 255.0
        return Color(red: r, green: g, blue: b)
    }
}
