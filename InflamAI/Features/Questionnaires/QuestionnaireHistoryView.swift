//
//  QuestionnaireHistoryView.swift
//  InflamAI
//
//  Displays history and trends for a specific questionnaire with statistics,
//  chart visualization, and list of past responses.
//

import SwiftUI
import CoreData
import Charts

struct QuestionnaireHistoryView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @Environment(\.dismiss) private var dismiss

    let questionnaireID: QuestionnaireID

    @State private var responses: [QuestionnaireResponse] = []
    @State private var isLoading = true
    @State private var selectedTimeRange: TimeRange = .thirtyDays
    @State private var selectedResponse: QuestionnaireResponse?

    private var context: NSManagedObjectContext {
        if viewContext.persistentStoreCoordinator != nil {
            return viewContext
        }
        return InflamAIPersistenceController.shared.container.viewContext
    }

    enum TimeRange: String, CaseIterable {
        case sevenDays = "7 Days"
        case thirtyDays = "30 Days"
        case ninetyDays = "90 Days"
        case all = "All Time"

        var days: Int? {
            switch self {
            case .sevenDays: return 7
            case .thirtyDays: return 30
            case .ninetyDays: return 90
            case .all: return nil
            }
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    if isLoading {
                        ProgressView()
                            .frame(maxWidth: .infinity, minHeight: 200)
                    } else if responses.isEmpty {
                        emptyStateView
                    } else {
                        // Time range picker
                        timeRangePicker

                        // Statistics card
                        statisticsCard

                        // Trend chart
                        trendChartCard

                        // Interpretation guide
                        interpretationGuide

                        // Response list
                        responseListCard
                    }
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle(questionnaireTitle)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .task {
                await loadResponses()
            }
            .sheet(item: $selectedResponse) { response in
                ResponseDetailSheet(response: response)
            }
        }
    }

    // MARK: - Title

    private var questionnaireTitle: String {
        switch questionnaireID {
        case .basdai: return "BASDAI History"
        case .basfi: return "BASFI History"
        case .basg: return "BAS-G History"
        case .asqol: return "ASQoL History"
        default: return NSLocalizedString(questionnaireID.titleKey, comment: "")
        }
    }

    // MARK: - Time Range Picker

    private var timeRangePicker: some View {
        Picker("Time Range", selection: $selectedTimeRange) {
            ForEach(TimeRange.allCases, id: \.self) { range in
                Text(range.rawValue).tag(range)
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: selectedTimeRange) { _ in
            Task { await loadResponses() }
        }
    }

    // MARK: - Statistics Card

    private var statisticsCard: some View {
        let filteredResponses = filteredByTimeRange
        let scores = filteredResponses.map { $0.score }

        let average = scores.isEmpty ? 0 : scores.reduce(0, +) / Double(scores.count)
        let trend = calculateTrend(from: filteredResponses)
        let completionRate = calculateCompletionRate()

        return VStack(spacing: 16) {
            Text("Statistics")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 16) {
                // Average score
                StatisticItem(
                    icon: "chart.bar.fill",
                    value: String(format: "%.1f", average),
                    label: "Average",
                    color: scoreColor(for: average)
                )

                // Trend
                StatisticItem(
                    icon: trend >= 0 ? "arrow.up.right" : "arrow.down.right",
                    value: String(format: "%+.1f", trend),
                    label: "Trend",
                    color: trend <= 0 ? .green : .orange
                )

                // Total responses
                StatisticItem(
                    icon: "checkmark.circle.fill",
                    value: "\(filteredResponses.count)",
                    label: "Responses",
                    color: .blue
                )
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    // MARK: - Trend Chart

    private var trendChartCard: some View {
        let filteredResponses = filteredByTimeRange.sorted { ($0.createdAt ?? Date.distantPast) < ($1.createdAt ?? Date.distantPast) }

        return VStack(spacing: 16) {
            HStack {
                Text("Score Trend")
                    .font(.headline)
                Spacer()
                Text(maxScoreLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if filteredResponses.isEmpty {
                Text("No data for this period")
                    .foregroundStyle(.secondary)
                    .frame(height: 200)
            } else {
                Chart(filteredResponses, id: \.objectID) { response in
                    LineMark(
                        x: .value("Date", response.createdAt ?? Date()),
                        y: .value("Score", response.score)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [scoreColor(for: response.score), scoreColor(for: response.score).opacity(0.7)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )

                    PointMark(
                        x: .value("Date", response.createdAt ?? Date()),
                        y: .value("Score", response.score)
                    )
                    .foregroundStyle(scoreColor(for: response.score))
                    .symbolSize(60)

                    AreaMark(
                        x: .value("Date", response.createdAt ?? Date()),
                        yStart: .value("Baseline", 0),
                        yEnd: .value("Score", response.score)
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [scoreColor(for: response.score).opacity(0.3), scoreColor(for: response.score).opacity(0.05)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
                .frame(height: 200)
                .chartYScale(domain: 0...maxScore)
                .chartXAxis {
                    AxisMarks(values: .automatic) { value in
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.day().month())
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .stride(by: yAxisStride)) { value in
                        AxisGridLine()
                        AxisValueLabel()
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    // MARK: - Interpretation Guide

    private var interpretationGuide: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Score Interpretation")
                .font(.headline)

            ForEach(interpretationItems, id: \.range) { item in
                HStack(spacing: 12) {
                    Circle()
                        .fill(item.color)
                        .frame(width: 12, height: 12)

                    Text(item.range)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .frame(width: 50, alignment: .leading)

                    Text(item.label)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private var interpretationItems: [(range: String, label: String, color: Color)] {
        switch questionnaireID {
        case .basdai, .basfi, .basg:
            return [
                ("0-2", "Low / Remission", .green),
                ("2-4", "Mild Activity", .yellow),
                ("4-6", "Moderate Activity", .orange),
                ("6+", "High Activity", .red)
            ]
        case .asqol:
            return [
                ("0-4", "Good Quality of Life", .green),
                ("5-9", "Moderate Impact", .yellow),
                ("10-13", "Significant Impact", .orange),
                ("14+", "Severe Impact", .red)
            ]
        default:
            return [
                ("0-3", "Low", .green),
                ("4-6", "Moderate", .yellow),
                ("7+", "High", .red)
            ]
        }
    }

    // MARK: - Response List

    private var responseListCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("All Responses")
                .font(.headline)

            LazyVStack(spacing: 12) {
                ForEach(filteredByTimeRange.sorted { ($0.createdAt ?? Date.distantPast) > ($1.createdAt ?? Date.distantPast) }, id: \.objectID) { response in
                    ResponseRow(response: response, questionnaireID: questionnaireID)
                        .onTapGesture {
                            selectedResponse = response
                        }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)

            Text("No History Yet")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Complete your first \(questionnaireShortName) assessment to start tracking your progress over time.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, minHeight: 300)
        .padding()
    }

    private var questionnaireShortName: String {
        switch questionnaireID {
        case .basdai: return "BASDAI"
        case .basfi: return "BASFI"
        case .basg: return "BAS-G"
        case .asqol: return "ASQoL"
        default: return "questionnaire"
        }
    }

    // MARK: - Helpers

    private var filteredByTimeRange: [QuestionnaireResponse] {
        guard let days = selectedTimeRange.days else { return responses }
        let cutoff = Calendar.current.date(byAdding: .day, value: -days, to: Date()) ?? Date()
        return responses.filter { ($0.createdAt ?? Date.distantPast) >= cutoff }
    }

    private var maxScore: Double {
        switch questionnaireID {
        case .asqol: return 18
        default: return 10
        }
    }

    private var maxScoreLabel: String {
        "0-\(Int(maxScore)) scale"
    }

    private var yAxisStride: Double {
        switch questionnaireID {
        case .asqol: return 3
        default: return 2
        }
    }

    private func scoreColor(for score: Double) -> Color {
        let normalizedScore = questionnaireID == .asqol ? score / 18.0 * 10.0 : score
        switch normalizedScore {
        case ..<4: return .green
        case 4..<6: return .yellow
        case 6..<8: return .orange
        default: return .red
        }
    }

    private func calculateTrend(from responses: [QuestionnaireResponse]) -> Double {
        let sorted = responses.sorted { ($0.createdAt ?? Date.distantPast) < ($1.createdAt ?? Date.distantPast) }
        guard sorted.count >= 2 else { return 0 }

        let halfIndex = sorted.count / 2
        let firstHalf = sorted.prefix(halfIndex)
        let secondHalf = sorted.suffix(sorted.count - halfIndex)

        let firstAvg = firstHalf.map(\.score).reduce(0, +) / Double(firstHalf.count)
        let secondAvg = secondHalf.map(\.score).reduce(0, +) / Double(secondHalf.count)

        return secondAvg - firstAvg
    }

    private func calculateCompletionRate() -> Double {
        // Simplified - could be enhanced with schedule tracking
        return 0.85
    }

    private func loadResponses() async {
        await MainActor.run { isLoading = true }

        let manager = QuestionnaireManager(viewContext: context)
        let fetched = manager.fetchRecentResponses(for: questionnaireID, limit: 365)

        await MainActor.run {
            responses = fetched
            isLoading = false
        }
    }
}

// MARK: - Supporting Views

private struct StatisticItem: View {
    let icon: String
    let value: String
    let label: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            Text(value)
                .font(.title2)
                .fontWeight(.bold)

            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct ResponseRow: View {
    let response: QuestionnaireResponse
    let questionnaireID: QuestionnaireID

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(response.createdAt ?? Date(), style: .date)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(response.createdAt ?? Date(), style: .time)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if let note = response.note, !note.isEmpty {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            Text(String(format: "%.1f", response.score))
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(scoreColor)

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding()
        .background(Color(.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }

    private var scoreColor: Color {
        let maxScore: Double = questionnaireID == .asqol ? 18 : 10
        let normalizedScore = response.score / maxScore * 10
        switch normalizedScore {
        case ..<4: return .green
        case 4..<6: return .yellow
        case 6..<8: return .orange
        default: return .red
        }
    }
}

private struct ResponseDetailSheet: View {
    let response: QuestionnaireResponse
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Response Details") {
                    LabeledContent("Date", value: response.createdAt ?? Date(), format: .dateTime)
                    LabeledContent("Score", value: String(format: "%.1f", response.score))

                    if response.durationMs > 0 {
                        LabeledContent("Duration", value: "\(Int(response.durationMs / 1000)) seconds")
                    }
                }

                if let note = response.note, !note.isEmpty {
                    Section("Notes") {
                        Text(note)
                    }
                }

                Section("Individual Answers") {
                    let answers = response.answers.values.sorted { $0.key < $1.key }
                    ForEach(Array(answers), id: \.key) { key, value in
                        LabeledContent(key, value: String(format: "%.0f", value))
                    }
                }
            }
            .navigationTitle("Response Detail")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

// MARK: - Preview

struct QuestionnaireHistoryView_Previews: PreviewProvider {
    static var previews: some View {
        QuestionnaireHistoryView(questionnaireID: .basdai)
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
    }
}
