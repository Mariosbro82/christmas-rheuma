//
//  ExerciseStatsView.swift
//  InflamAI-Swift
//
//  Detailed statistics for individual exercises
//

import SwiftUI
import CoreData
import Charts

struct ExerciseStatsView: View {
    let exercise: Exercise
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var viewModel: ExerciseStatsViewModel

    init(exercise: Exercise, context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        self.exercise = exercise
        _viewModel = StateObject(wrappedValue: ExerciseStatsViewModel(exerciseName: exercise.name, context: context))
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Summary stats
                summaryCard

                // Trend chart
                trendChart

                // Feedback distribution
                feedbackDistribution

                // Recent completions
                recentCompletions
            }
            .padding()
        }
        .navigationTitle(exercise.name)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            viewModel.loadStats()
        }
    }

    // MARK: - Summary Card

    private var summaryCard: some View {
        VStack(spacing: 20) {
            Text("Last 30 Days")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                statBox(
                    icon: "checkmark.circle.fill",
                    value: "\(viewModel.completions30Days)",
                    label: "Completions",
                    color: .green
                )

                statBox(
                    icon: "calendar",
                    value: viewModel.lastCompletedDate,
                    label: "Last Done",
                    color: .blue
                )

                statBox(
                    icon: "face.smiling",
                    value: viewModel.averageFeedback,
                    label: "Avg Feedback",
                    color: .orange
                )

                statBox(
                    icon: "clock",
                    value: viewModel.totalTimeSpent,
                    label: "Total Time",
                    color: .purple
                )
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func statBox(icon: String, value: String, label: String, color: Color) -> some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .lineLimit(1)
                .minimumScaleFactor(0.8)

            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }

    // MARK: - Trend Chart

    private var trendChart: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Weekly Activity")
                .font(.headline)

            if #available(iOS 16.0, *) {
                Chart(viewModel.weeklyData) { item in
                    BarMark(
                        x: .value("Week", item.weekLabel),
                        y: .value("Count", item.count)
                    )
                    .foregroundStyle(Color.blue.gradient)
                }
                .frame(height: 200)
                .chartYAxis {
                    AxisMarks(position: .leading)
                }
            } else {
                // Fallback for iOS 15
                HStack(alignment: .bottom, spacing: 12) {
                    ForEach(viewModel.weeklyData) { item in
                        VStack {
                            Spacer()

                            Rectangle()
                                .fill(Color.blue)
                                .frame(width: 40, height: CGFloat(item.count) * 20)

                            Text(item.weekLabel)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .frame(height: 200)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    // MARK: - Feedback Distribution

    private var feedbackDistribution: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("How It Feels")
                .font(.headline)

            VStack(spacing: 12) {
                feedbackRow("😊 Easy", count: viewModel.feedbackCounts[.easy] ?? 0, total: viewModel.completions30Days, color: .green)
                feedbackRow("😐 Manageable", count: viewModel.feedbackCounts[.manageable] ?? 0, total: viewModel.completions30Days, color: .blue)
                feedbackRow("😰 Difficult", count: viewModel.feedbackCounts[.difficult] ?? 0, total: viewModel.completions30Days, color: .orange)
                feedbackRow("⚠️ Unbearable", count: viewModel.feedbackCounts[.unbearable] ?? 0, total: viewModel.completions30Days, color: .red)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func feedbackRow(_ label: String, count: Int, total: Int, color: Color) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .frame(width: 120, alignment: .leading)

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color(.tertiarySystemGroupedBackground))

                    Rectangle()
                        .fill(color)
                        .frame(width: total > 0 ? geometry.size.width * CGFloat(count) / CGFloat(total) : 0)
                }
            }
            .frame(height: 24)
            .cornerRadius(4)

            Text("\(count)")
                .font(.subheadline)
                .fontWeight(.semibold)
                .frame(width: 30, alignment: .trailing)
        }
    }

    // MARK: - Recent Completions

    private var recentCompletions: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recent Activity")
                .font(.headline)

            if viewModel.recentCompletions.isEmpty {
                Text("No recent activity")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(viewModel.recentCompletions, id: \.id) { completion in
                    completionRow(completion)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func completionRow(_ completion: ExerciseCompletion) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                if let timestamp = completion.timestamp {
                    Text(timestamp, style: .date)
                        .font(.subheadline)
                        .fontWeight(.medium)
                }

                if let feedback = completion.feedback,
                   let feedbackType = ExerciseFeedback(rawValue: feedback) {
                    Text(feedbackType.emoji + " " + feedbackType.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            if completion.fromRoutine {
                Image(systemName: "list.bullet.circle.fill")
                    .foregroundColor(.green)
                    .font(.caption)
            }

            Text("\(completion.durationSeconds / 60) min")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }
}

// MARK: - View Model

@MainActor
class ExerciseStatsViewModel: ObservableObject {
    @Published var completions30Days: Int = 0
    @Published var lastCompletedDate: String = "Never"
    @Published var averageFeedback: String = "-"
    @Published var totalTimeSpent: String = "0 min"
    @Published var weeklyData: [WeeklyDataPoint] = []
    @Published var feedbackCounts: [ExerciseFeedback: Int] = [:]
    @Published var recentCompletions: [ExerciseCompletion] = []

    private let exerciseName: String
    private let context: NSManagedObjectContext

    init(exerciseName: String, context: NSManagedObjectContext) {
        self.exerciseName = exerciseName
        self.context = context
    }

    func loadStats() {
        fetchCompletions()
        calculateStats()
    }

    private func fetchCompletions() {
        let request: NSFetchRequest<ExerciseCompletion> = ExerciseCompletion.fetchRequest()
        request.predicate = NSPredicate(format: "exerciseName == %@", exerciseName)
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        request.fetchLimit = 50

        do {
            recentCompletions = try context.fetch(request)
        } catch {
            print("Failed to fetch exercise completions: \(error)")
        }
    }

    private func calculateStats() {
        let calendar = Calendar.current
        let thirtyDaysAgo = calendar.date(byAdding: .day, value: -30, to: Date())!

        // Filter to last 30 days
        let recent = recentCompletions.filter { completion in
            guard let timestamp = completion.timestamp else { return false }
            return timestamp >= thirtyDaysAgo
        }

        // Count
        completions30Days = recent.count

        // Last completed
        if let lastDate = recentCompletions.first?.timestamp {
            let formatter = RelativeDateTimeFormatter()
            formatter.unitsStyle = .short
            lastCompletedDate = formatter.localizedString(for: lastDate, relativeTo: Date())
        }

        // Average feedback
        let feedbackValues = recent.compactMap { completion -> Int? in
            guard let feedbackString = completion.feedback,
                  let feedback = ExerciseFeedback(rawValue: feedbackString) else { return nil }

            switch feedback {
            case .easy: return 4
            case .manageable: return 3
            case .difficult: return 2
            case .unbearable: return 1
            }
        }

        if !feedbackValues.isEmpty {
            let avg = Double(feedbackValues.reduce(0, +)) / Double(feedbackValues.count)
            if avg >= 3.5 {
                averageFeedback = "😊"
            } else if avg >= 2.5 {
                averageFeedback = "😐"
            } else if avg >= 1.5 {
                averageFeedback = "😰"
            } else {
                averageFeedback = "⚠️"
            }
        }

        // Total time
        let totalSeconds = recent.reduce(0) { $0 + Int($1.durationSeconds) }
        let minutes = totalSeconds / 60
        totalTimeSpent = "\(minutes) min"

        // Weekly data
        generateWeeklyData(from: recent)

        // Feedback counts
        calculateFeedbackCounts(from: recent)
    }

    private func generateWeeklyData(from completions: [ExerciseCompletion]) {
        let calendar = Calendar.current
        var weekCounts: [Int: Int] = [:]

        for completion in completions {
            guard let timestamp = completion.timestamp else { continue }
            let weekOfYear = calendar.component(.weekOfYear, from: timestamp)
            weekCounts[weekOfYear, default: 0] += 1
        }

        let sortedWeeks = weekCounts.keys.sorted().suffix(4)
        weeklyData = sortedWeeks.map { week in
            WeeklyDataPoint(weekLabel: "W\(week)", count: weekCounts[week] ?? 0)
        }
    }

    private func calculateFeedbackCounts(from completions: [ExerciseCompletion]) {
        var counts: [ExerciseFeedback: Int] = [
            .easy: 0,
            .manageable: 0,
            .difficult: 0,
            .unbearable: 0
        ]

        for completion in completions {
            guard let feedbackString = completion.feedback,
                  let feedback = ExerciseFeedback(rawValue: feedbackString) else { continue }
            counts[feedback, default: 0] += 1
        }

        feedbackCounts = counts
    }
}

// MARK: - Supporting Types

struct WeeklyDataPoint: Identifiable {
    let id = UUID()
    let weekLabel: String
    let count: Int
}

// MARK: - Preview

struct ExerciseStatsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            ExerciseStatsView(
                exercise: Exercise(
                    id: UUID(),
                    name: "Cat-Cow Stretch",
                    category: .stretching,
                    difficulty: .beginner,
                    duration: 5,
                    targetAreas: [],
                    instructions: [],
                    benefits: [],
                    safetyTips: [],
                    videoURL: nil
                )
            )
            .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
        }
    }
}
