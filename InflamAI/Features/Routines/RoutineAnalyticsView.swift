//
//  RoutineAnalyticsView.swift
//  InflamAI
//
//  Analytics dashboard for exercise routines
//  Shows completion history, feedback trends, pain levels, and activeness
//

import SwiftUI
import Charts
import CoreData

struct RoutineAnalyticsView: View {
    let routine: UserRoutine
    @StateObject private var viewModel: RoutineAnalyticsViewModel
    @Environment(\.dismiss) private var dismiss

    init(routine: UserRoutine) {
        self.routine = routine
        _viewModel = StateObject(wrappedValue: RoutineAnalyticsViewModel(routine: routine))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Summary Stats
                    summarySection

                    // Completion Chart
                    if !viewModel.completionData.isEmpty {
                        completionChartSection
                    }

                    // Feedback Distribution
                    if !viewModel.feedbackData.isEmpty {
                        feedbackDistributionSection
                    }

                    // Session History
                    sessionHistorySection
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Routine Analytics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                viewModel.loadAnalytics()
            }
        }
    }

    // MARK: - Summary Section

    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(routine.name ?? "Routine")
                .font(.title2)
                .fontWeight(.bold)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                AnalyticsStatCard(
                    icon: "checkmark.circle.fill",
                    label: "Times Completed",
                    value: "\(routine.timesCompleted)",
                    color: .green
                )

                AnalyticsStatCard(
                    icon: "clock.fill",
                    label: "Total Duration",
                    value: "\(routine.totalDuration) min",
                    color: .blue
                )

                AnalyticsStatCard(
                    icon: "flame.fill",
                    label: "Current Streak",
                    value: "\(viewModel.currentStreak) days",
                    color: .orange
                )

                AnalyticsStatCard(
                    icon: "star.fill",
                    label: "Avg Feeling",
                    value: viewModel.avgFeelingText,
                    color: .purple
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
    }

    // MARK: - Completion Chart

    private var completionChartSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(.blue)
                Text("Completion History")
                    .font(.headline)
            }

            Chart(viewModel.completionData) { dataPoint in
                BarMark(
                    x: .value("Date", dataPoint.date, unit: .day),
                    y: .value("Completed", dataPoint.completed ? 1 : 0)
                )
                .foregroundStyle(dataPoint.completed ? Color.green : Color(.systemGray4))
                .cornerRadius(4)
            }
            .frame(height: 120)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: 7)) { value in
                    AxisValueLabel(format: .dateTime.weekday(.abbreviated))
                }
            }
            .chartYAxis(.hidden)

            Text("Last 30 days")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
    }

    // MARK: - Feedback Distribution

    private var feedbackDistributionSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "face.smiling")
                    .foregroundColor(.purple)
                Text("How You Felt After Routines")
                    .font(.headline)
            }

            // Pie chart representation using bars
            VStack(spacing: 12) {
                ForEach(viewModel.feedbackData, id: \.feedback) { item in
                    AnalyticsFeedbackRow(
                        feedback: item.feedback,
                        count: item.count,
                        percentage: item.percentage
                    )
                }
            }

            if !viewModel.recentNotes.isEmpty {
                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    Text("Recent Notes")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    ForEach(viewModel.recentNotes.prefix(3), id: \.self) { note in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "quote.opening")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(note)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
    }

    // MARK: - Session History

    private var sessionHistorySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .foregroundColor(.green)
                Text("Recent Sessions")
                    .font(.headline)
                Spacer()
            }

            if viewModel.recentSessions.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "figure.mixed.cardio")
                        .font(.largeTitle)
                        .foregroundColor(.secondary)
                    Text("No sessions yet")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Text("Complete this routine to see your history")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else {
                ForEach(viewModel.recentSessions) { session in
                    AnalyticsSessionRow(session: session)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
    }
}

// MARK: - Supporting Views

private struct AnalyticsStatCard: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .font(.system(size: 20))
                    .foregroundColor(color)
            }

            Text(value)
                .font(.title3)
                .fontWeight(.bold)

            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

private struct AnalyticsFeedbackRow: View {
    let feedback: String
    let count: Int
    let percentage: Double

    var body: some View {
        HStack(spacing: 12) {
            Text(feedbackEmoji)
                .font(.title2)

            VStack(alignment: .leading, spacing: 4) {
                Text(feedback)
                    .font(.subheadline)
                    .fontWeight(.medium)

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(.systemGray5))

                        RoundedRectangle(cornerRadius: 4)
                            .fill(feedbackColor)
                            .frame(width: geometry.size.width * percentage)
                    }
                }
                .frame(height: 8)
            }

            Text("\(count)")
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundColor(.secondary)
        }
    }

    private var feedbackEmoji: String {
        switch feedback.lowercased() {
        case "great", "excellent": return "😊"
        case "good": return "🙂"
        case "okay", "ok": return "😐"
        case "tired": return "😴"
        case "hard", "difficult": return "😓"
        case "painful": return "😣"
        default: return "📝"
        }
    }

    private var feedbackColor: Color {
        switch feedback.lowercased() {
        case "great", "excellent": return .green
        case "good": return .blue
        case "okay", "ok": return .yellow
        case "tired": return .orange
        case "hard", "difficult": return .orange
        case "painful": return .red
        default: return .gray
        }
    }
}

private struct AnalyticsSessionRow: View {
    let session: RoutineSessionData

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(session.wasCompleted ? Color.green.opacity(0.15) : Color.orange.opacity(0.15))
                    .frame(width: 40, height: 40)

                Image(systemName: session.wasCompleted ? "checkmark" : "xmark")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(session.wasCompleted ? .green : .orange)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(session.date.formatted(date: .abbreviated, time: .shortened))
                    .font(.subheadline)
                    .fontWeight(.medium)

                HStack(spacing: 8) {
                    if let duration = session.duration {
                        Label("\(duration) min", systemImage: "clock")
                    }
                    if let feedback = session.feedback {
                        Text("• \(feedback)")
                    }
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 8)
    }
}

// MARK: - View Model

@MainActor
class RoutineAnalyticsViewModel: ObservableObject {
    @Published var completionData: [CompletionDataPoint] = []
    @Published var feedbackData: [FeedbackDataPoint] = []
    @Published var recentSessions: [RoutineSessionData] = []
    @Published var recentNotes: [String] = []
    @Published var currentStreak: Int = 0
    @Published var avgFeelingText: String = "--"

    private let routine: UserRoutine
    private var context: NSManagedObjectContext {
        routine.managedObjectContext ?? InflamAIPersistenceController.shared.container.viewContext
    }

    init(routine: UserRoutine) {
        self.routine = routine
    }

    func loadAnalytics() {
        loadCompletionData()
        loadFeedbackData()
        loadRecentSessions()
        calculateStreak()
    }

    private func loadCompletionData() {
        let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date())!
        var data: [CompletionDataPoint] = []

        // Create data points for last 30 days
        for dayOffset in 0..<30 {
            let date = Calendar.current.date(byAdding: .day, value: -dayOffset, to: Date())!
            let startOfDay = Calendar.current.startOfDay(for: date)
            let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: startOfDay)!

            let request: NSFetchRequest<ExerciseSession> = ExerciseSession.fetchRequest()
            request.predicate = NSPredicate(
                format: "routineType CONTAINS %@ AND timestamp >= %@ AND timestamp < %@",
                routine.name ?? "",
                startOfDay as NSDate,
                endOfDay as NSDate
            )

            let count = (try? context.count(for: request)) ?? 0
            data.append(CompletionDataPoint(date: startOfDay, completed: count > 0))
        }

        completionData = data.reversed()
    }

    private func loadFeedbackData() {
        guard let routineID = routine.id else { return }

        let request: NSFetchRequest<ExerciseCompletion> = ExerciseCompletion.fetchRequest()
        request.predicate = NSPredicate(format: "routineID == %@ AND feedback != nil", routineID as CVarArg)

        guard let completions = try? context.fetch(request) else { return }

        // Group by feedback
        var feedbackCounts: [String: Int] = [:]
        var notes: [String] = []

        for completion in completions {
            if let feedback = completion.feedback {
                feedbackCounts[feedback, default: 0] += 1
            }
            if let note = completion.feedbackNotes, !note.isEmpty {
                notes.append(note)
            }
        }

        let total = max(feedbackCounts.values.reduce(0, +), 1)
        feedbackData = feedbackCounts.map { key, value in
            FeedbackDataPoint(feedback: key, count: value, percentage: Double(value) / Double(total))
        }.sorted { $0.count > $1.count }

        recentNotes = notes.suffix(5).reversed()

        // Calculate average feeling
        if !feedbackData.isEmpty {
            let topFeedback = feedbackData.first?.feedback ?? "--"
            avgFeelingText = topFeedback
        }
    }

    private func loadRecentSessions() {
        guard let routineID = routine.id else { return }

        let request: NSFetchRequest<ExerciseSession> = ExerciseSession.fetchRequest()
        request.predicate = NSPredicate(format: "routineType CONTAINS %@", routine.name ?? "")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \ExerciseSession.timestamp, ascending: false)]
        request.fetchLimit = 10

        guard let sessions = try? context.fetch(request) else { return }

        recentSessions = sessions.compactMap { session in
            guard let timestamp = session.timestamp else { return nil }
            return RoutineSessionData(
                id: session.id ?? UUID(),
                date: timestamp,
                duration: Int(session.durationMinutes),
                wasCompleted: session.completedSuccessfully,
                feedback: nil // Would need to join with ExerciseCompletion
            )
        }
    }

    private func calculateStreak() {
        var streak = 0
        var currentDate = Calendar.current.startOfDay(for: Date())

        for _ in 0..<365 {
            let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: currentDate)!

            let request: NSFetchRequest<ExerciseSession> = ExerciseSession.fetchRequest()
            request.predicate = NSPredicate(
                format: "routineType CONTAINS %@ AND timestamp >= %@ AND timestamp < %@",
                routine.name ?? "",
                currentDate as NSDate,
                endOfDay as NSDate
            )

            if let count = try? context.count(for: request), count > 0 {
                streak += 1
                currentDate = Calendar.current.date(byAdding: .day, value: -1, to: currentDate)!
            } else {
                break
            }
        }

        currentStreak = streak
    }
}

// MARK: - Data Models

struct CompletionDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let completed: Bool
}

struct FeedbackDataPoint {
    let feedback: String
    let count: Int
    let percentage: Double
}

struct RoutineSessionData: Identifiable {
    let id: UUID
    let date: Date
    let duration: Int?
    let wasCompleted: Bool
    let feedback: String?
}

// MARK: - Preview

struct RoutineAnalyticsView_Previews: PreviewProvider {
    static var previews: some View {
        let context = InflamAIPersistenceController.preview.container.viewContext
        let routine = UserRoutine(context: context)
        routine.id = UUID()
        routine.name = "Morning Mobility"
        routine.totalDuration = 15
        routine.timesCompleted = 12

        return RoutineAnalyticsView(routine: routine)
    }
}
