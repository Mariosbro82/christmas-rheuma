//
//  MedicationManagementView.swift
//  InflamAI
//
//  Complete medication tracking system with reminders and adherence analytics
//

import SwiftUI
import CoreData
import UserNotifications

struct MedicationManagementView: View {
    @StateObject private var viewModel: MedicationViewModel
    @State private var showingAddMedication = false
    @State private var showingAdherenceReport = false

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: MedicationViewModel(context: context))
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is accessed via tab bar which already provides navigation.
        ScrollView {
            VStack(spacing: Spacing.lg) {
                // Today's Medications
                todaysMedicationsSection

                // Adherence Stats
                adherenceStatsSection

                // All Medications List
                allMedicationsSection

                // Adherence Calendar
                adherenceCalendarSection
            }
            .padding(Spacing.md)
        }
        .navigationTitle("Medications")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showingAddMedication = true
                } label: {
                    Image(systemName: "plus")
                }
            }

            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    showingAdherenceReport = true
                } label: {
                    Image(systemName: "chart.bar.fill")
                }
            }
        }
        .sheet(isPresented: $showingAddMedication) {
            AddMedicationView(viewModel: viewModel)
        }
        .sheet(isPresented: $showingAdherenceReport) {
            AdherenceReportView(viewModel: viewModel)
        }
        .onAppear {
            viewModel.loadMedications()
            viewModel.loadTodaysDoses()
        }
    }

    // MARK: - Today's Medications

    private var todaysMedicationsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                Image(systemName: "pills.fill")
                    .foregroundColor(Colors.Primary.p500)
                Text("Today's Medications")
                    .font(.system(size: Typography.md, weight: .semibold))

                Spacer()

                if !viewModel.todaysDoses.isEmpty {
                    Text("\(viewModel.takenCount)/\(viewModel.todaysDoses.count)")
                        .font(.system(size: Typography.sm))
                        .foregroundColor(Colors.Gray.g500)
                }
            }

            if viewModel.todaysDoses.isEmpty {
                emptyTodayView
            } else {
                ForEach(viewModel.todaysDoses) { dose in
                    TodaysDoseCard(dose: dose, viewModel: viewModel)
                }
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private var emptyTodayView: some View {
        VStack(spacing: Spacing.sm) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 40))
                .foregroundColor(Colors.Semantic.success)

            Text("All done for today!")
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }

    // MARK: - Adherence Stats

    private var adherenceStatsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Text("Adherence Overview")
                .font(.system(size: Typography.md, weight: .semibold))

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                AdherenceStatCard(
                    title: "This Week",
                    percentage: viewModel.weeklyAdherence,
                    color: adherenceColor(viewModel.weeklyAdherence)
                )

                AdherenceStatCard(
                    title: "This Month",
                    percentage: viewModel.monthlyAdherence,
                    color: adherenceColor(viewModel.monthlyAdherence)
                )
            }

            if viewModel.weeklyAdherence < 80 {
                adherenceWarning
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private var adherenceWarning: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(Colors.Semantic.warning)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Low Adherence Detected")
                    .font(.system(size: Typography.base, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                Text("Missing doses may reduce treatment effectiveness")
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)
            }

            Spacer()
        }
        .padding(Spacing.md)
        .background(Colors.Semantic.warning.opacity(0.1))
        .cornerRadius(Radii.md)
    }

    // MARK: - All Medications

    private var allMedicationsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Text("All Medications")
                .font(.system(size: Typography.md, weight: .semibold))

            // CRIT-005: Add loading and error states
            if viewModel.isLoading {
                ForEach(0..<3, id: \.self) { _ in
                    MedicationCardSkeleton()
                }
            } else if let error = viewModel.errorMessage {
                ErrorStateView(
                    title: "Couldn't load medications",
                    message: error,
                    retryAction: { viewModel.loadMedications() }
                )
            } else if viewModel.activeMedications.isEmpty {
                emptyMedicationsView
            } else {
                ForEach(viewModel.activeMedications) { medication in
                    NavigationLink {
                        MedicationDetailView(medication: medication, viewModel: viewModel)
                    } label: {
                        MedicationRow(
                            medication: medication,
                            takenToday: viewModel.todaysDoses.contains { $0.medicationName == medication.name && $0.isTaken }
                        )
                    }
                    .buttonStyle(.plain)
                }
            }

            if !viewModel.inactiveMedications.isEmpty {
                DisclosureGroup("Inactive Medications (\(viewModel.inactiveMedications.count))") {
                    ForEach(viewModel.inactiveMedications) { medication in
                        MedicationRow(medication: medication)
                            .opacity(0.6)
                    }
                }
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private var emptyMedicationsView: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "cross.vial")
                .font(.system(size: 60))
                .foregroundColor(Colors.Gray.g300)

            Text("No Medications Yet")
                .font(.system(size: Typography.lg, weight: .semibold))
                .foregroundColor(Colors.Gray.g700)

            Text("Tap + to add your first medication")
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(Spacing.xl)
    }

    // MARK: - Adherence Calendar

    private var adherenceCalendarSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Text("30-Day Adherence")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            AdherenceCalendarView(doses: viewModel.last30DaysDoses)
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private func adherenceColor(_ percentage: Double) -> Color {
        switch percentage {
        case 90...100: return Colors.Semantic.success
        case 70..<90: return Colors.Semantic.warning
        default: return Colors.Semantic.error
        }
    }
}

// MARK: - Supporting Views

struct TodaysDoseCard: View {
    let dose: ScheduledDose
    @ObservedObject var viewModel: MedicationViewModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                HStack(spacing: Spacing.xs) {
                    Text(dose.medicationName)
                        .font(.system(size: Typography.base, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    if dose.isTaken {
                        Text("Taken")
                            .font(.system(size: Typography.xxs, weight: .bold))
                            .foregroundColor(Colors.Semantic.success)
                            .padding(.horizontal, Spacing.xs)
                            .padding(.vertical, Spacing.xxxs)
                            .background(Colors.Semantic.success.opacity(0.15))
                            .cornerRadius(Radii.xs)
                    }
                }

                HStack {
                    Text("\(dose.dosage, specifier: "%.1f") \(dose.unit)")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)

                    Text("•")
                        .foregroundColor(Colors.Gray.g400)

                    Text(dose.scheduledTime, style: .time)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Gray.g500)
                }

                if dose.isBiologic {
                    Label("Biologic", systemImage: "syringe.fill")
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Accent.purple)
                }
            }

            Spacer()

            if dose.isTaken {
                // Completed state - show checkmark
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(Colors.Semantic.success)
            } else {
                // Quick action buttons - no dialog needed
                HStack(spacing: Spacing.xs) {
                    // Skip button (small)
                    Button {
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        viewModel.markDoseSkipped(dose)
                    } label: {
                        Image(systemName: "forward.fill")
                            .font(.caption)
                            .foregroundColor(Colors.Semantic.warning)
                            .padding(Spacing.xs)
                            .background(Colors.Semantic.warning.opacity(0.12))
                            .clipShape(Circle())
                    }
                    .accessibilityLabel("Skip dose")

                    // Take button (prominent) - ONE TAP, NO DIALOG
                    Button {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        viewModel.markDoseTaken(dose)
                    } label: {
                        Text("Take")
                            .font(.system(size: Typography.xs, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.horizontal, Spacing.md)
                            .padding(.vertical, Spacing.xs)
                            .background(Colors.Primary.p500)
                            .cornerRadius(Radii.md)
                    }
                    .accessibilityLabel("Mark as taken")
                }
            }
        }
        .padding(Spacing.md)
        .background(dose.isTaken ? Colors.Semantic.success.opacity(0.08) : Colors.Gray.g100)
        .cornerRadius(Radii.lg)
        .animation(Animations.easeOut, value: dose.isTaken)
    }
}

struct AdherenceStatCard: View {
    let title: String
    let percentage: Double
    let color: Color

    var body: some View {
        VStack(spacing: Spacing.xs) {
            Text(title)
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)

            Text("\(Int(percentage))%")
                .font(.system(size: Typography.xxxl, weight: .bold, design: .rounded))
                .foregroundColor(color)

            ProgressView(value: percentage, total: 100)
                .tint(color)
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }
}

struct MedicationRow: View {
    let medication: MedicationData
    var takenToday: Bool = false

    var body: some View {
        HStack(spacing: Spacing.sm) {
            // Icon - shows checkmark overlay when taken today
            ZStack {
                Image(systemName: medication.isBiologic ? "syringe.fill" : "pills.fill")
                    .font(.title2)
                    .foregroundColor(takenToday ? Colors.Semantic.success : (medication.isBiologic ? Colors.Accent.purple : Colors.Primary.p500))
                    .frame(width: 40)

                if takenToday {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundColor(Colors.Semantic.success)
                        .background(Circle().fill(Color.white).frame(width: 12, height: 12))
                        .offset(x: 12, y: 10)
                }
            }

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                HStack(spacing: Spacing.xs) {
                    Text(medication.name)
                        .font(.system(size: Typography.base, weight: .semibold))
                        .foregroundColor(Colors.Gray.g900)

                    // "Taken" badge when taken today
                    if takenToday {
                        Text("Taken")
                            .font(.system(size: Typography.xxs, weight: .bold))
                            .foregroundColor(Colors.Semantic.success)
                            .padding(.horizontal, Spacing.xs)
                            .padding(.vertical, Spacing.xxxs)
                            .background(Colors.Semantic.success.opacity(0.15))
                            .cornerRadius(Radii.xs)
                    }
                }

                Text("\(medication.dosage, specifier: "%.1f") \(medication.unit) • \(medication.frequency)")
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g500)

                if let category = medication.category {
                    Text(category)
                        .font(.system(size: Typography.xs))
                        .foregroundColor(Colors.Primary.p500)
                        .padding(.horizontal, Spacing.xs)
                        .padding(.vertical, Spacing.xxxs)
                        .background(Colors.Primary.p50)
                        .cornerRadius(Radii.xs)
                }
            }

            Spacer()

            if medication.reminderEnabled {
                Image(systemName: "bell.fill")
                    .font(.caption)
                    .foregroundColor(Colors.Semantic.warning)
            }
        }
        .padding(Spacing.md)
        .background(takenToday ? Colors.Semantic.success.opacity(0.06) : Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }
}

struct AdherenceCalendarView: View {
    let doses: [DoseLog]

    private let columns = Array(repeating: GridItem(.flexible()), count: 7)
    private let calendar = Calendar.current

    var body: some View {
        LazyVGrid(columns: columns, spacing: 8) {
            ForEach(last30Days(), id: \.self) { date in
                DayCell(date: date, adherence: adherenceFor(date: date))
            }
        }
    }

    private func last30Days() -> [Date] {
        let today = calendar.startOfDay(for: Date())
        return (0..<30).compactMap { offset in
            calendar.date(byAdding: .day, value: -offset, to: today)
        }.reversed()
    }

    private func adherenceFor(date: Date) -> Double {
        let dayDoses = doses.filter { log in
            calendar.isDate(log.timestamp ?? Date(), inSameDayAs: date)
        }

        guard !dayDoses.isEmpty else { return 0 }

        let takenCount = dayDoses.filter { !$0.wasSkipped }.count
        return Double(takenCount) / Double(dayDoses.count) * 100
    }
}

struct DayCell: View {
    let date: Date
    let adherence: Double

    var body: some View {
        VStack(spacing: 4) {
            Text(date, format: .dateTime.day())
                .font(.caption2)
                .foregroundColor(.secondary)

            Circle()
                .fill(adherenceColor)
                .frame(width: 30, height: 30)
                .overlay(
                    Text("\(Int(adherence))")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.white)
                )
        }
    }

    private var adherenceColor: Color {
        switch adherence {
        case 100: return Colors.Semantic.success
        case 70..<100: return Colors.Semantic.warning
        case 0..<70 where adherence > 0: return Colors.Semantic.error
        default: return Color(.systemGray4)
        }
    }
}

// MARK: - Add Medication

struct AddMedicationView: View {
    @ObservedObject var viewModel: MedicationViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var category = "NSAID"
    @State private var dosage = 0.0
    @State private var unit = "mg"
    @State private var frequency = "Daily"
    @State private var route = "Oral"
    @State private var isBiologic = false
    @State private var reminderEnabled = false
    @State private var reminderTimes: [Date] = [Calendar.current.date(bySettingHour: 8, minute: 0, second: 0, of: Date())!]

    private let categories = ["NSAID", "DMARD", "Biologic", "Corticosteroid", "Pain Reliever", "Supplement"]
    private let units = ["mg", "g", "mL", "IU"]
    private let frequencies = ["As Needed", "Daily", "Twice Daily", "Three Times Daily", "Weekly", "Biweekly", "Monthly"]
    private let routes = ["Oral", "Injection", "IV Infusion", "Topical"]

    var body: some View {
        NavigationView {
            Form {
                Section {
                    TextField("Medication Name", text: $name)
                        .accessibilityLabel("Medication name")

                    Picker("Category", selection: $category) {
                        ForEach(categories, id: \.self) { category in
                            Text(category).tag(category)
                        }
                    }

                    Toggle("Biologic Medication", isOn: $isBiologic)
                } header: {
                    Text("Basic Information")
                }

                Section {
                    HStack {
                        Text("Dosage")
                        Spacer()
                        TextField("Amount", value: $dosage, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 100)
                    }

                    Picker("Unit", selection: $unit) {
                        ForEach(units, id: \.self) { unit in
                            Text(unit).tag(unit)
                        }
                    }

                    Picker("Frequency", selection: $frequency) {
                        ForEach(frequencies, id: \.self) { freq in
                            Text(freq).tag(freq)
                        }
                    }

                    Picker("Route", selection: $route) {
                        ForEach(routes, id: \.self) { route in
                            Text(route).tag(route)
                        }
                    }
                } header: {
                    Text("Dosage Information")
                }

                Section {
                    Toggle("Enable Reminders", isOn: $reminderEnabled)

                    if reminderEnabled {
                        ForEach(reminderTimes.indices, id: \.self) { index in
                            DatePicker("Reminder \(index + 1)", selection: $reminderTimes[index], displayedComponents: .hourAndMinute)
                        }

                        Button {
                            reminderTimes.append(Date())
                        } label: {
                            Label("Add Reminder Time", systemImage: "plus.circle")
                        }
                    }
                } header: {
                    Text("Reminders")
                }
            }
            .navigationTitle("Add Medication")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveMedication()
                    }
                    .disabled(name.isEmpty || dosage <= 0)
                }
            }
        }
    }

    private func saveMedication() {
        Task {
            await viewModel.addMedication(
                name: name,
                category: category,
                dosage: dosage,
                unit: unit,
                frequency: frequency,
                route: route,
                isBiologic: isBiologic,
                reminderEnabled: reminderEnabled,
                reminderTimes: reminderTimes
            )
            dismiss()
        }
    }
}

// MARK: - Medication Detail

struct MedicationDetailView: View {
    let medication: MedicationData
    @ObservedObject var viewModel: MedicationViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showingEditSheet = false
    @State private var showingDeleteAlert = false
    @State private var recentDoses: [DoseLog] = []

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                medicationHeader

                // Dosage History
                dosageHistorySection

                // Actions
                actionsSection
            }
            .padding()
        }
        .navigationTitle(medication.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Menu {
                    Button {
                        showingEditSheet = true
                    } label: {
                        Label("Edit", systemImage: "pencil")
                    }

                    Button(role: .destructive) {
                        showingDeleteAlert = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .alert("Delete Medication?", isPresented: $showingDeleteAlert) {
            Button("Delete", role: .destructive) {
                viewModel.deleteMedication(medication)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        }
        .onAppear {
            loadRecentDoses()
        }
    }

    private var medicationHeader: some View {
        VStack(spacing: 12) {
            Image(systemName: medication.isBiologic ? "syringe.fill" : "pills.fill")
                .font(.system(size: 60))
                .foregroundColor(medication.isBiologic ? Colors.Accent.purple : Colors.Primary.p500)

            Text("\(medication.dosage, specifier: "%.1f") \(medication.unit)")
                .font(.title)
                .fontWeight(.bold)

            Text(medication.frequency)
                .font(.subheadline)
                .foregroundColor(.secondary)

            if let category = medication.category {
                Text(category)
                    .font(.caption)
                    .foregroundColor(Colors.Primary.p500)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(Colors.Primary.p500.opacity(0.1))
                    .cornerRadius(8)
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private var dosageHistorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Doses (Last 7 Days)")
                .font(.headline)

            if recentDoses.isEmpty {
                Text("No doses logged yet")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(recentDoses.prefix(7), id: \.id) { dose in
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(dose.timestamp ?? Date(), style: .date)
                                .font(.subheadline)
                            Text(dose.timestamp ?? Date(), style: .time)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        if dose.wasSkipped {
                            Label("Skipped", systemImage: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundColor(Colors.Semantic.warning)
                        } else {
                            Label("Taken", systemImage: "checkmark.circle.fill")
                                .font(.caption)
                                .foregroundColor(Colors.Semantic.success)
                        }
                    }
                    .padding(.vertical, 8)

                    if dose.id != recentDoses.prefix(7).last?.id {
                        Divider()
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
    }

    private func loadRecentDoses() {
        Task {
            let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date())!
            let logs = viewModel.last30DaysDoses.filter { log in
                guard let timestamp = log.timestamp else { return false }
                return timestamp >= sevenDaysAgo
            }
            recentDoses = logs.sorted { ($0.timestamp ?? Date.distantPast) > ($1.timestamp ?? Date.distantPast) }
        }
    }

    private var actionsSection: some View {
        VStack(spacing: 12) {
            Button {
                logDoseNow()
            } label: {
                Label("Log Dose Now", systemImage: "checkmark.circle.fill")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Colors.Primary.p500)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }

            Button {
                toggleMedicationActive()
            } label: {
                Label(medication.isActive ? "Mark Inactive" : "Mark Active", systemImage: "pause.circle")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color(.systemGray5))
                    .foregroundColor(.primary)
                    .cornerRadius(12)
            }
        }
    }

    private func logDoseNow() {
        let dose = ScheduledDose(
            id: UUID(),
            medicationName: medication.name,
            dosage: medication.dosage,
            unit: medication.unit,
            scheduledTime: Date(),
            isTaken: false,
            isBiologic: medication.isBiologic
        )
        viewModel.markDoseTaken(dose)
        loadRecentDoses()
    }

    private func toggleMedicationActive() {
        viewModel.toggleMedicationActive(medication)
        dismiss()
    }
}

// MARK: - Adherence Report

import Charts

struct AdherenceReportView: View {
    @ObservedObject var viewModel: MedicationViewModel
    @Environment(\.dismiss) private var dismiss

    private var weeklyAdherenceData: [(String, Double)] {
        let calendar = Calendar.current
        var data: [(String, Double)] = []

        for weekOffset in (0..<4).reversed() {
            let weekStart = calendar.date(byAdding: .weekOfYear, value: -weekOffset, to: Date())!
            let weekEnd = calendar.date(byAdding: .day, value: 7, to: weekStart)!

            let weekDoses = viewModel.last30DaysDoses.filter { dose in
                guard let timestamp = dose.timestamp else { return false }
                return timestamp >= weekStart && timestamp < weekEnd
            }

            let adherence: Double
            if weekDoses.isEmpty {
                adherence = 0
            } else {
                let takenCount = weekDoses.filter { !$0.wasSkipped }.count
                adherence = Double(takenCount) / Double(weekDoses.count) * 100
            }

            let formatter = DateFormatter()
            formatter.dateFormat = "MMM d"
            data.append((formatter.string(from: weekStart), adherence))
        }

        return data
    }

    private var dailyAdherenceData: [(Date, Double)] {
        let calendar = Calendar.current
        var data: [(Date, Double)] = []

        for dayOffset in (0..<14).reversed() {
            let day = calendar.date(byAdding: .day, value: -dayOffset, to: calendar.startOfDay(for: Date()))!

            let dayDoses = viewModel.last30DaysDoses.filter { dose in
                guard let timestamp = dose.timestamp else { return false }
                return calendar.isDate(timestamp, inSameDayAs: day)
            }

            let adherence: Double
            if dayDoses.isEmpty {
                adherence = 0
            } else {
                let takenCount = dayDoses.filter { !$0.wasSkipped }.count
                adherence = Double(takenCount) / Double(dayDoses.count) * 100
            }

            data.append((day, adherence))
        }

        return data
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Overall Stats
                    overallStatsSection

                    // Weekly Trend Chart
                    weeklyTrendSection

                    // Daily Adherence Chart
                    dailyAdherenceSection

                    // Medication Breakdown
                    medicationBreakdownSection

                    // Insights
                    insightsSection
                }
                .padding()
            }
            .navigationTitle("Adherence Report")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }

    private var overallStatsSection: some View {
        VStack(spacing: 16) {
            Text("30-Day Overview")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 16) {
                StatBox(
                    title: "Weekly",
                    value: "\(Int(viewModel.weeklyAdherence))%",
                    color: adherenceColor(viewModel.weeklyAdherence)
                )

                StatBox(
                    title: "Monthly",
                    value: "\(Int(viewModel.monthlyAdherence))%",
                    color: adherenceColor(viewModel.monthlyAdherence)
                )

                StatBox(
                    title: "Doses Taken",
                    value: "\(viewModel.last30DaysDoses.filter { !$0.wasSkipped }.count)",
                    color: Colors.Primary.p500
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    private var weeklyTrendSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Weekly Trend")
                .font(.headline)

            Chart {
                ForEach(weeklyAdherenceData, id: \.0) { week, adherence in
                    BarMark(
                        x: .value("Week", week),
                        y: .value("Adherence %", adherence)
                    )
                    .foregroundStyle(adherenceGradient(adherence))
                    .cornerRadius(8)
                }
            }
            .frame(height: 200)
            .chartYScale(domain: 0...100)
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisValueLabel {
                        if let intValue = value.as(Int.self) {
                            Text("\(intValue)%")
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    private var dailyAdherenceSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("14-Day Adherence")
                .font(.headline)

            Chart {
                ForEach(dailyAdherenceData, id: \.0) { date, adherence in
                    LineMark(
                        x: .value("Date", date),
                        y: .value("Adherence %", adherence)
                    )
                    .foregroundStyle(Colors.Primary.p500)
                    .interpolationMethod(.catmullRom)

                    AreaMark(
                        x: .value("Date", date),
                        y: .value("Adherence %", adherence)
                    )
                    .foregroundStyle(Colors.Primary.p500.opacity(0.1))
                    .interpolationMethod(.catmullRom)
                }

                RuleMark(y: .value("Target", 80))
                    .foregroundStyle(Colors.Semantic.success.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
            }
            .frame(height: 200)
            .chartYScale(domain: 0...100)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: 2)) { value in
                    AxisValueLabel(format: .dateTime.month(.abbreviated).day())
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    private var medicationBreakdownSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("By Medication")
                .font(.headline)

            ForEach(viewModel.activeMedications) { medication in
                HStack {
                    Text(medication.name)
                        .font(.subheadline)

                    Spacer()

                    // Calculate adherence for this specific medication
                    Text("--% ")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 8)

                if medication.id != viewModel.activeMedications.last?.id {
                    Divider()
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    private var insightsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Insights")
                .font(.headline)

            if viewModel.monthlyAdherence >= 90 {
                InsightCard(
                    icon: "star.fill",
                    color: Colors.Semantic.success,
                    title: "Excellent Adherence!",
                    description: "You're maintaining excellent medication adherence. Keep up the great work!"
                )
            } else if viewModel.monthlyAdherence >= 70 {
                InsightCard(
                    icon: "exclamationmark.triangle.fill",
                    color: Colors.Semantic.warning,
                    title: "Room for Improvement",
                    description: "Try setting more reminders to help remember your medications."
                )
            } else {
                InsightCard(
                    icon: "exclamationmark.circle.fill",
                    color: Colors.Semantic.error,
                    title: "Adherence Needs Attention",
                    description: "Low adherence may affect treatment effectiveness. Consider discussing with your doctor."
                )
            }

            if !viewModel.todaysDoses.isEmpty {
                let todayAdherence = Double(viewModel.takenCount) / Double(viewModel.todaysDoses.count) * 100
                if todayAdherence < 100 {
                    InsightCard(
                        icon: "clock.fill",
                        color: Colors.Primary.p500,
                        title: "Today's Reminder",
                        description: "You have \(viewModel.todaysDoses.count - viewModel.takenCount) dose(s) remaining today."
                    )
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5)
    }

    private func adherenceColor(_ percentage: Double) -> Color {
        switch percentage {
        case 90...100: return Colors.Semantic.success
        case 70..<90: return Colors.Semantic.warning
        default: return Colors.Semantic.error
        }
    }

    private func adherenceGradient(_ percentage: Double) -> LinearGradient {
        let color = adherenceColor(percentage)
        return LinearGradient(
            colors: [color.opacity(0.8), color],
            startPoint: .bottom,
            endPoint: .top
        )
    }
}

struct StatBox: View {
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct InsightCard: View {
    let icon: String
    let color: Color
    let title: String
    let description: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Preview

struct MedicationManagementView_Previews: PreviewProvider {
    static var previews: some View {
        MedicationManagementView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
