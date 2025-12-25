//
//  ExerciseHistoryView.swift
//  InflamAI-Swift
//
//  Exercise history calendar with colored dots for routines and standalone exercises
//

import SwiftUI
import CoreData

struct ExerciseHistoryView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @StateObject private var viewModel: ExerciseHistoryViewModel
    @State private var selectedDate: Date?

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: ExerciseHistoryViewModel(context: context))
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Statistics card
                statisticsCard

                // Calendar
                calendarView

                // Selected date details
                if let date = selectedDate {
                    dateDetailsView(for: date)
                }
            }
            .padding()
        }
        .navigationTitle("Exercise History")
        .navigationBarTitleDisplayMode(.large)
        .onAppear {
            viewModel.loadHistory()
        }
    }

    // MARK: - Statistics Card

    private var statisticsCard: some View {
        VStack(spacing: 16) {
            Text("Last 30 Days")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 16) {
                statItem(
                    icon: "figure.walk",
                    value: "\(viewModel.totalExercises30Days)",
                    label: "Total Exercises",
                    color: .blue
                )

                statItem(
                    icon: "list.bullet.circle",
                    value: "\(viewModel.totalRoutines30Days)",
                    label: "Routines",
                    color: .green
                )

                statItem(
                    icon: "flame.fill",
                    value: "\(viewModel.currentStreak)",
                    label: "Day Streak",
                    color: .orange
                )
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func statItem(icon: String, value: String, label: String, color: Color) -> some View {
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
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Calendar View

    private var calendarView: some View {
        VStack(spacing: 16) {
            Text("Activity Calendar")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Month navigation
            HStack {
                Button(action: { viewModel.previousMonth() }) {
                    Image(systemName: "chevron.left")
                        .font(.title3)
                }

                Spacer()

                Text(viewModel.currentMonthYear)
                    .font(.headline)

                Spacer()

                Button(action: { viewModel.nextMonth() }) {
                    Image(systemName: "chevron.right")
                        .font(.title3)
                }
            }
            .padding(.horizontal)

            // Weekday headers
            HStack(spacing: 0) {
                ForEach(["S", "M", "T", "W", "T", "F", "S"], id: \.self) { day in
                    Text(day)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity)
                }
            }

            // Calendar grid
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4), count: 7), spacing: 4) {
                ForEach(viewModel.calendarDays, id: \.date) { day in
                    calendarDayView(day: day)
                }
            }

            // Legend
            HStack(spacing: 16) {
                legendItem(color: .blue, label: "Exercise")
                legendItem(color: .green, label: "Routine")
                legendItem(color: .purple, label: "Both")
            }
            .font(.caption)
            .padding(.top, 8)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func calendarDayView(day: CalendarDay) -> some View {
        Button(action: {
            if day.hasActivity {
                selectedDate = day.date
            }
        }) {
            VStack(spacing: 4) {
                Text("\(Calendar.current.component(.day, from: day.date))")
                    .font(.system(size: 14))
                    .fontWeight(day.isToday ? .bold : .regular)
                    .foregroundColor(day.isCurrentMonth ? .primary : .secondary)

                // Activity dots
                if day.hasActivity {
                    Circle()
                        .fill(day.activityColor)
                        .frame(width: 6, height: 6)
                }
            }
            .frame(height: 44)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(day.isToday ? Color.blue.opacity(0.1) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(selectedDate == day.date ? Color.blue : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }

    private func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Date Details View

    private func dateDetailsView(for date: Date) -> some View {
        VStack(spacing: 16) {
            HStack {
                Text(date, style: .date)
                    .font(.headline)

                Spacer()

                Button(action: { selectedDate = nil }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
            }

            if let activities = viewModel.activities(for: date) {
                ForEach(activities, id: \.id) { activity in
                    activityRow(activity)
                }
            } else {
                Text("No activities on this date")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .cornerRadius(16)
    }

    private func activityRow(_ activity: ExerciseActivity) -> some View {
        HStack(spacing: 12) {
            Image(systemName: activity.fromRoutine ? "list.bullet.circle.fill" : "figure.walk")
                .font(.title3)
                .foregroundColor(activity.fromRoutine ? .green : .blue)

            VStack(alignment: .leading, spacing: 4) {
                Text(activity.exerciseName)
                    .font(.body)
                    .fontWeight(.medium)

                if let routineName = activity.routineName {
                    Text(routineName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if let feedback = activity.feedback {
                    Text(feedback.emoji)
                        .font(.caption)
                }
            }

            Spacer()

            Text(formatDuration(activity.durationSeconds))
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }

    private func formatDuration(_ seconds: Int32) -> String {
        let minutes = seconds / 60
        if minutes > 0 {
            return "\(minutes) min"
        } else {
            return "\(seconds) sec"
        }
    }
}

// MARK: - View Model

@MainActor
class ExerciseHistoryViewModel: ObservableObject {
    @Published var calendarDays: [CalendarDay] = []
    @Published var currentMonth: Date = Date()
    @Published var totalExercises30Days: Int = 0
    @Published var totalRoutines30Days: Int = 0
    @Published var currentStreak: Int = 0

    private let context: NSManagedObjectContext
    private var exerciseCompletions: [ExerciseCompletion] = []

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    var currentMonthYear: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMMM yyyy"
        return formatter.string(from: currentMonth)
    }

    func loadHistory() {
        fetchExerciseCompletions()
        calculateStatistics()
        generateCalendarDays()
    }

    private func fetchExerciseCompletions() {
        let request: NSFetchRequest<ExerciseCompletion> = ExerciseCompletion.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]

        do {
            exerciseCompletions = try context.fetch(request)
        } catch {
            print("Failed to fetch exercise completions: \(error)")
        }
    }

    private func calculateStatistics() {
        let calendar = Calendar.current
        let thirtyDaysAgo = calendar.date(byAdding: .day, value: -30, to: Date())!

        // Count exercises in last 30 days
        let recent = exerciseCompletions.filter { completion in
            guard let timestamp = completion.timestamp else { return false }
            return timestamp >= thirtyDaysAgo
        }

        totalExercises30Days = recent.count
        totalRoutines30Days = recent.filter { $0.fromRoutine }.count

        // Calculate streak
        currentStreak = calculateCurrentStreak()
    }

    private func calculateCurrentStreak() -> Int {
        let calendar = Calendar.current
        var streak = 0
        var checkDate = calendar.startOfDay(for: Date())

        while true {
            let hasActivity = exerciseCompletions.contains { completion in
                guard let timestamp = completion.timestamp else { return false }
                return calendar.isDate(timestamp, inSameDayAs: checkDate)
            }

            if hasActivity {
                streak += 1
                guard let previousDay = calendar.date(byAdding: .day, value: -1, to: checkDate) else { break }
                checkDate = previousDay
            } else {
                break
            }
        }

        return streak
    }

    func generateCalendarDays() {
        let calendar = Calendar.current
        let month = calendar.component(.month, from: currentMonth)
        let year = calendar.component(.year, from: currentMonth)

        guard let monthStart = calendar.date(from: DateComponents(year: year, month: month, day: 1)) else { return }
        guard let monthRange = calendar.range(of: .day, in: .month, for: monthStart) else { return }

        let firstWeekday = calendar.component(.weekday, from: monthStart)
        let daysInMonth = monthRange.count

        var days: [CalendarDay] = []

        // Previous month padding
        for _ in 1..<firstWeekday {
            guard let date = calendar.date(byAdding: .day, value: days.count - firstWeekday + 1, to: monthStart) else { continue }
            days.append(CalendarDay(date: date, isCurrentMonth: false, hasActivity: false, activityColor: .clear))
        }

        // Current month days
        for day in 1...daysInMonth {
            guard let date = calendar.date(from: DateComponents(year: year, month: month, day: day)) else { continue }

            let hasExercise = exerciseCompletions.contains { completion in
                guard let timestamp = completion.timestamp else { return false }
                return calendar.isDate(timestamp, inSameDayAs: date) && !completion.fromRoutine
            }

            let hasRoutine = exerciseCompletions.contains { completion in
                guard let timestamp = completion.timestamp else { return false }
                return calendar.isDate(timestamp, inSameDayAs: date) && completion.fromRoutine
            }

            let color: Color
            if hasExercise && hasRoutine {
                color = .purple
            } else if hasRoutine {
                color = .green
            } else if hasExercise {
                color = .blue
            } else {
                color = .clear
            }

            days.append(CalendarDay(
                date: date,
                isCurrentMonth: true,
                hasActivity: hasExercise || hasRoutine,
                activityColor: color,
                isToday: calendar.isDateInToday(date)
            ))
        }

        calendarDays = days
    }

    func previousMonth() {
        guard let newMonth = Calendar.current.date(byAdding: .month, value: -1, to: currentMonth) else { return }
        currentMonth = newMonth
        generateCalendarDays()
    }

    func nextMonth() {
        guard let newMonth = Calendar.current.date(byAdding: .month, value: 1, to: currentMonth) else { return }
        currentMonth = newMonth
        generateCalendarDays()
    }

    func activities(for date: Date) -> [ExerciseActivity]? {
        let calendar = Calendar.current

        let activities = exerciseCompletions.compactMap { completion -> ExerciseActivity? in
            guard let timestamp = completion.timestamp,
                  calendar.isDate(timestamp, inSameDayAs: date),
                  let exerciseName = completion.exerciseName else {
                return nil
            }

            return ExerciseActivity(
                id: completion.id ?? UUID(),
                exerciseName: exerciseName,
                fromRoutine: completion.fromRoutine,
                routineName: completion.routineName,
                durationSeconds: completion.durationSeconds,
                feedback: completion.feedback.flatMap { ExerciseFeedback(rawValue: $0) }
            )
        }

        return activities.isEmpty ? nil : activities
    }
}

// MARK: - Supporting Types

struct CalendarDay {
    let date: Date
    let isCurrentMonth: Bool
    let hasActivity: Bool
    let activityColor: Color
    var isToday: Bool = false
}

struct ExerciseActivity {
    let id: UUID
    let exerciseName: String
    let fromRoutine: Bool
    let routineName: String?
    let durationSeconds: Int32
    let feedback: ExerciseFeedback?
}

// MARK: - Preview

struct ExerciseHistoryView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            ExerciseHistoryView()
                .environment(\.managedObjectContext, InflamAIPersistenceController.preview.container.viewContext)
        }
    }
}
