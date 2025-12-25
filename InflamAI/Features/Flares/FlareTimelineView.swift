//
//  FlareTimelineView.swift
//  InflamAI
//
//  Comprehensive flare event tracking with timeline, patterns, and analytics
//

import SwiftUI
import CoreData
import Charts

struct FlareTimelineView: View {
    @StateObject private var viewModel: FlareTimelineViewModel
    @State private var selectedPeriod: TimePeriod = .month
    @State private var showingAddFlare = false
    @State private var selectedFlare: FlareEventData?

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: FlareTimelineViewModel(context: context))
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        ScrollView {
            VStack(spacing: Spacing.lg) {
                // Stats Overview
                statsOverviewSection

                // Frequency Chart
                frequencyChartSection

                // Period Selector
                periodSelector

                // Timeline
                timelineSection

                // Pattern Insights
                patternInsightsSection
            }
            .padding()
        }
        .navigationTitle("Flare Timeline")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showingAddFlare = true
                } label: {
                    Image(systemName: "plus")
                }
            }
        }
        .sheet(isPresented: $showingAddFlare) {
            JointTapSOSView()
        }
        .sheet(item: $selectedFlare) { flare in
            FlareDetailView(flare: flare, viewModel: viewModel)
        }
        .onAppear {
            viewModel.loadFlares()
        }
    }

    // MARK: - Stats Overview

    private var statsOverviewSection: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.md) {
            FlareStatCard(
                icon: "flame.fill",
                title: "This Month",
                value: "\(viewModel.flaresThisMonth)",
                color: Colors.Semantic.error
            )

            FlareStatCard(
                icon: "calendar",
                title: "Last Flare",
                value: viewModel.daysSinceLastFlare,
                color: Colors.Semantic.warning
            )

            FlareStatCard(
                icon: "chart.line.uptrend.xyaxis",
                title: "Avg Duration",
                value: viewModel.averageDuration,
                color: Colors.Accent.purple
            )

            FlareStatCard(
                icon: "exclamationmark.triangle",
                title: "Severe Flares",
                value: "\(viewModel.severeFlareCount)",
                color: Colors.Semantic.error
            )
        }
    }

    // MARK: - Frequency Chart

    private var frequencyChartSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Flare Frequency")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Chart {
                ForEach(viewModel.monthlyFlareData, id: \.month) { data in
                    BarMark(
                        x: .value("Month", data.month, unit: .month),
                        y: .value("Flares", data.count)
                    )
                    .foregroundStyle(flareGradient)
                    .cornerRadius(Radii.sm)
                }
            }
            .frame(height: 200)
            .chartYAxis {
                AxisMarks(position: .leading)
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .month)) { value in
                    AxisValueLabel(format: .dateTime.month(.abbreviated))
                }
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private var flareGradient: LinearGradient {
        LinearGradient(
            colors: [Colors.Semantic.error.opacity(0.8), Colors.Semantic.error],
            startPoint: .bottom,
            endPoint: .top
        )
    }

    // MARK: - Period Selector

    private var periodSelector: some View {
        Picker("Period", selection: $selectedPeriod) {
            ForEach(TimePeriod.allCases, id: \.self) { period in
                Text(period.rawValue).tag(period)
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: selectedPeriod) { newPeriod in
            viewModel.updatePeriod(newPeriod)
        }
    }

    // MARK: - Timeline

    private var timelineSection: some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            Text("Flare History")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            // CRIT-005: Add loading state
            if viewModel.isLoading {
                FlareTimelineSkeleton()
            } else if viewModel.filteredFlares.isEmpty {
                emptyStateView
            } else {
                ForEach(viewModel.filteredFlares) { flare in
                    FlareTimelineCard(flare: flare) {
                        UISelectionFeedbackGenerator().selectionChanged()
                        selectedFlare = flare
                    }
                }
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private var emptyStateView: some View {
        VStack(spacing: Spacing.lg) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(Colors.Semantic.success)

            Text("No Flares Recorded")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text("Great news! No flares in this period.")
                .font(.system(size: Typography.sm))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(Spacing.xxl)
    }

    // MARK: - Pattern Insights

    private var patternInsightsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(Colors.Accent.purple)
                Text("Pattern Insights")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)
            }

            if let insights = viewModel.patternInsights {
                ForEach(insights, id: \.self) { insight in
                    InsightRow(insight: insight)
                }
            } else {
                Text("Log more flares to see pattern insights")
                    .font(.system(size: Typography.sm))
                    .foregroundColor(Colors.Gray.g500)
            }
        }
        .padding(Spacing.md)
        .background(Colors.Accent.purple.opacity(0.1))
        .cornerRadius(Radii.xl)
    }
}

// MARK: - Supporting Views

struct FlareStatCard: View {
    let icon: String
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: Spacing.sm) {
            Image(systemName: icon)
                .font(.system(size: Typography.xl))
                .foregroundColor(color)

            Text(value)
                .font(.system(size: Typography.xl, weight: .bold))
                .foregroundColor(Colors.Gray.g900)

            Text(title)
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }
}

struct FlareTimelineCard: View {
    let flare: FlareEventData
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.md) {
                // Timeline marker
                VStack {
                    Circle()
                        .fill(severityColor(flare.severity))
                        .frame(width: 12, height: 12)

                    Rectangle()
                        .fill(Colors.Gray.g300)
                        .frame(width: 2)
                }

                VStack(alignment: .leading, spacing: Spacing.sm) {
                    HStack {
                        Text(flare.startDate, style: .date)
                            .font(.system(size: Typography.sm, weight: .semibold))
                            .foregroundColor(Colors.Gray.g900)

                        Spacer()

                        severityBadge(flare.severity)
                    }

                    if !flare.affectedRegions.isEmpty {
                        HStack(spacing: Spacing.xxs) {
                            Image(systemName: "figure.stand")
                                .font(.system(size: Typography.xs))
                            Text(flare.affectedRegions.prefix(3).joined(separator: ", "))
                                .font(.system(size: Typography.xs))
                        }
                        .foregroundColor(Colors.Gray.g500)
                    }

                    if let duration = flare.duration {
                        HStack(spacing: Spacing.xxs) {
                            Image(systemName: "clock")
                                .font(.system(size: Typography.xs))
                            Text("\(duration) days")
                                .font(.system(size: Typography.xs))
                        }
                        .foregroundColor(Colors.Gray.g500)
                    }

                    if !flare.triggers.isEmpty {
                        HStack(spacing: Spacing.xxs) {
                            Image(systemName: "exclamationmark.triangle")
                                .font(.system(size: Typography.xs))
                            // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                            Text("Triggers: \(flare.triggers.map { $0.displayName }.joined(separator: ", "))")
                                .font(.system(size: Typography.xs))
                                .lineLimit(1)
                        }
                        .foregroundColor(Colors.Semantic.warning)
                    }
                }

                Image(systemName: "chevron.right")
                    .font(.system(size: Typography.xs))
                    .foregroundColor(Colors.Gray.g400)
            }
            .padding(Spacing.md)
            .background(Colors.Gray.g100)
            .cornerRadius(Radii.lg)
        }
        .buttonStyle(.plain)
    }

    private func severityColor(_ severity: Int16) -> Color {
        switch severity {
        case 1: return Colors.Semantic.success
        case 2: return Colors.Semantic.warning
        case 3: return Colors.Semantic.warning
        case 4: return Colors.Semantic.error
        default: return Colors.Gray.g400
        }
    }

    private func severityBadge(_ severity: Int16) -> some View {
        let text: String
        let color: Color

        switch severity {
        case 1:
            text = "Mild"
            color = Colors.Semantic.success
        case 2:
            text = "Moderate"
            color = Colors.Semantic.warning
        case 3:
            text = "Severe"
            color = Colors.Semantic.warning
        case 4:
            text = "Extreme"
            color = Colors.Semantic.error
        default:
            text = "Unknown"
            color = Colors.Gray.g400
        }

        return Text(text)
            .font(.system(size: Typography.xs, weight: .semibold))
            .foregroundColor(.white)
            .padding(.horizontal, Spacing.sm)
            .padding(.vertical, Spacing.xxs)
            .background(color)
            .cornerRadius(Radii.sm)
    }
}

struct InsightRow: View {
    let insight: String

    var body: some View {
        HStack(spacing: Spacing.md) {
            Image(systemName: "lightbulb.fill")
                .foregroundColor(Colors.Semantic.warning)

            Text(insight)
                .font(.system(size: Typography.sm))
                .foregroundColor(Colors.Gray.g700)

            Spacer()
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.md)
    }
}

// MARK: - Flare Detail View

struct FlareDetailView: View {
    let flare: FlareEventData
    @ObservedObject var viewModel: FlareTimelineViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showingEndFlare = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: Spacing.xl) {
                    // Header
                    headerSection

                    // Severity & Duration
                    metricsSection

                    // Affected Regions
                    if !flare.affectedRegions.isEmpty {
                        affectedRegionsSection
                    }

                    // Triggers
                    if !flare.triggers.isEmpty {
                        triggersSection
                    }

                    // Notes
                    if let notes = flare.notes, !notes.isEmpty {
                        notesSection(notes)
                    }

                    // Actions
                    if flare.endDate == nil {
                        activeFlareActions
                    }
                }
                .padding()
            }
            .navigationTitle("Flare Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .alert("End Flare?", isPresented: $showingEndFlare) {
                Button("End Now") {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    viewModel.endFlare(flare)
                    dismiss()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Mark this flare as ended?")
            }
        }
    }

    private var headerSection: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "flame.fill")
                .font(.system(size: 60))
                .foregroundColor(Colors.Semantic.error)

            Text(flare.startDate, style: .date)
                .font(.system(size: Typography.xl, weight: .bold))
                .foregroundColor(Colors.Gray.g900)

            if flare.endDate == nil {
                Label("Active Flare", systemImage: "exclamationmark.circle.fill")
                    .font(.system(size: Typography.sm))
                    .foregroundColor(Colors.Semantic.error)
                    .padding(.horizontal, Spacing.md)
                    .padding(.vertical, Spacing.xs)
                    .background(Colors.Semantic.error.opacity(0.1))
                    .cornerRadius(Radii.md)
            }
        }
    }

    private var metricsSection: some View {
        HStack(spacing: Spacing.lg) {
            MetricBox(
                icon: "exclamationmark.triangle.fill",
                label: "Severity",
                value: severityText(flare.severity),
                color: severityColor(flare.severity)
            )

            if let duration = flare.duration {
                MetricBox(
                    icon: "clock.fill",
                    label: "Duration",
                    value: "\(duration) days",
                    color: Colors.Primary.p500
                )
            } else {
                MetricBox(
                    icon: "clock.fill",
                    label: "Duration",
                    value: "Ongoing",
                    color: Colors.Semantic.warning
                )
            }
        }
    }

    private var affectedRegionsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Affected Regions")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                ForEach(flare.affectedRegions, id: \.self) { region in
                    Text(region)
                        .font(.system(size: Typography.sm))
                        .padding(.horizontal, Spacing.md)
                        .padding(.vertical, Spacing.sm)
                        .frame(maxWidth: .infinity)
                        .background(Colors.Primary.p500.opacity(0.1))
                        .foregroundColor(Colors.Primary.p500)
                        .cornerRadius(Radii.md)
                }
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }

    private var triggersSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Suspected Triggers")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            ForEach(flare.triggers, id: \.self) { trigger in
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(Colors.Semantic.warning)
                    // CRIT-003 FIX: Apply displayName to convert snake_case to Title Case
                    Text(trigger.displayName)
                        .font(.system(size: Typography.sm))
                        .foregroundColor(Colors.Gray.g700)
                    Spacer()
                }
                .padding(Spacing.md)
                .background(Colors.Semantic.warning.opacity(0.1))
                .cornerRadius(Radii.md)
            }
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }

    private func notesSection(_ notes: String) -> some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Text("Notes")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            Text(notes)
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
        }
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }

    private var activeFlareActions: some View {
        VStack(spacing: Spacing.md) {
            Button {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                showingEndFlare = true
            } label: {
                Label("End Flare", systemImage: "checkmark.circle.fill")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(Spacing.md)
                    .background(Colors.Semantic.success)
                    .cornerRadius(Radii.lg)
            }

            Text("Feeling better? Mark this flare as ended.")
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)
        }
    }

    private func severityText(_ severity: Int16) -> String {
        switch severity {
        case 1: return "Mild"
        case 2: return "Moderate"
        case 3: return "Severe"
        case 4: return "Extreme"
        default: return "Unknown"
        }
    }

    private func severityColor(_ severity: Int16) -> Color {
        switch severity {
        case 1: return Colors.Semantic.success
        case 2: return Colors.Semantic.warning
        case 3: return Colors.Semantic.warning
        case 4: return Colors.Semantic.error
        default: return Colors.Gray.g400
        }
    }
}

struct MetricBox: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: Spacing.sm) {
            Image(systemName: icon)
                .font(.system(size: Typography.xl))
                .foregroundColor(color)

            Text(value)
                .font(.system(size: Typography.lg, weight: .bold))
                .foregroundColor(Colors.Gray.g900)

            Text(label)
                .font(.system(size: Typography.xs))
                .foregroundColor(Colors.Gray.g500)
        }
        .frame(maxWidth: .infinity)
        .padding(Spacing.md)
        .background(Colors.Gray.g100)
        .cornerRadius(Radii.lg)
    }
}

// MARK: - View Model

@MainActor
class FlareTimelineViewModel: ObservableObject {
    @Published var allFlares: [FlareEventData] = []
    @Published var filteredFlares: [FlareEventData] = []
    @Published var selectedPeriod: TimePeriod = .month
    @Published var monthlyFlareData: [MonthlyFlareData] = []
    @Published var patternInsights: [String]?
    // CRIT-005: Add loading and error states
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    var flaresThisMonth: Int {
        let calendar = Calendar.current
        let now = Date()
        let startOfMonth = calendar.date(from: calendar.dateComponents([.year, .month], from: now))!

        return allFlares.filter { $0.startDate >= startOfMonth }.count
    }

    var daysSinceLastFlare: String {
        guard let lastFlare = allFlares.first else { return "N/A" }

        let days = Calendar.current.dateComponents([.day], from: lastFlare.startDate, to: Date()).day ?? 0
        return "\(days)d ago"
    }

    var averageDuration: String {
        let flaresWithDuration = allFlares.compactMap { $0.duration }
        guard !flaresWithDuration.isEmpty else { return "N/A" }

        let avg = flaresWithDuration.reduce(0, +) / flaresWithDuration.count
        return "\(avg)d"
    }

    var severeFlareCount: Int {
        allFlares.filter { $0.severity >= 3 }.count
    }

    func loadFlares() {
        isLoading = true
        errorMessage = nil

        Task {
            let flares: [FlareEventData] = await context.perform {
                let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
                request.sortDescriptors = [NSSortDescriptor(keyPath: \FlareEvent.startDate, ascending: false)]

                guard let results = try? self.context.fetch(request) else { return [] }

                return results.compactMap { flareEvent -> FlareEventData? in
                    guard let id = flareEvent.id,
                          let startDate = flareEvent.startDate else { return nil }

                    let affectedRegions: [String]
                    if let regionsData = flareEvent.primaryRegions,
                       let decodedRegions = try? JSONDecoder().decode([String].self, from: regionsData) {
                        affectedRegions = decodedRegions
                    } else {
                        affectedRegions = []
                    }

                    let triggers: [String]
                    if let triggersData = flareEvent.suspectedTriggers,
                       let decodedTriggers = try? JSONDecoder().decode([String].self, from: triggersData) {
                        triggers = decodedTriggers
                    } else {
                        triggers = []
                    }

                    let duration: Int?
                    if let endDate = flareEvent.endDate {
                        duration = Calendar.current.dateComponents([.day], from: startDate, to: endDate).day
                    } else {
                        duration = nil
                    }

                    return FlareEventData(
                        id: id,
                        startDate: startDate,
                        endDate: flareEvent.endDate,
                        severity: flareEvent.severity,
                        affectedRegions: affectedRegions,
                        triggers: triggers,
                        notes: flareEvent.notes,
                        duration: duration
                    )
                }
            }

            self.allFlares = flares
            updatePeriod(selectedPeriod)
            generateMonthlyData()
            analyzePatterns()
            isLoading = false
        }
    }

    func updatePeriod(_ period: TimePeriod) {
        selectedPeriod = period
        let cutoffDate = Calendar.current.date(byAdding: period.dateComponent, value: -period.value, to: Date())!
        filteredFlares = allFlares.filter { $0.startDate >= cutoffDate }
    }

    private func generateMonthlyData() {
        let calendar = Calendar.current
        var data: [MonthlyFlareData] = []

        // Last 6 months
        for monthOffset in (0..<6).reversed() {
            let date = calendar.date(byAdding: .month, value: -monthOffset, to: Date())!
            let startOfMonth = calendar.date(from: calendar.dateComponents([.year, .month], from: date))!
            let endOfMonth = calendar.date(byAdding: .month, value: 1, to: startOfMonth)!

            let count = allFlares.filter { flare in
                flare.startDate >= startOfMonth && flare.startDate < endOfMonth
            }.count

            data.append(MonthlyFlareData(month: startOfMonth, count: count))
        }

        monthlyFlareData = data
    }

    private func analyzePatterns() {
        guard allFlares.count >= 3 else {
            patternInsights = nil
            return
        }

        var insights: [String] = []

        // Frequency insight
        if flaresThisMonth >= 3 {
            insights.append("High flare frequency this month. Consider discussing with your doctor.")
        }

        // Common triggers
        let allTriggers = allFlares.flatMap { $0.triggers }
        let triggerCounts = Dictionary(grouping: allTriggers, by: { $0 }).mapValues { $0.count }
        if let mostCommon = triggerCounts.max(by: { $0.value < $1.value }), mostCommon.value >= 2 {
            insights.append("'\(mostCommon.key)' appears as a trigger in multiple flares.")
        }

        // Severity trend
        let recentFlares = allFlares.prefix(5)
        let recentSeverity = recentFlares.map { Double($0.severity) }.reduce(0, +) / Double(recentFlares.count)
        if recentSeverity >= 3.0 {
            insights.append("Recent flares have been more severe than usual.")
        }

        patternInsights = insights.isEmpty ? nil : insights
    }

    func endFlare(_ flare: FlareEventData) {
        Task {
            await context.perform {
                let request: NSFetchRequest<FlareEvent> = FlareEvent.fetchRequest()
                request.predicate = NSPredicate(format: "id == %@", flare.id as CVarArg)

                if let flareEvent = try? self.context.fetch(request).first {
                    flareEvent.endDate = Date()
                    // FIXED: Proper error handling instead of silent try?
                    do {
                        try self.context.save()
                    } catch {
                        print("❌ CRITICAL: Failed to save flare end date: \(error)")
                    }
                }
            }

            loadFlares()
        }
    }
}

// MARK: - Data Models

struct FlareEventData: Identifiable {
    let id: UUID
    let startDate: Date
    let endDate: Date?
    let severity: Int16
    let affectedRegions: [String]
    let triggers: [String]
    let notes: String?
    let duration: Int?
}

struct MonthlyFlareData {
    let month: Date
    let count: Int
}

enum TimePeriod: String, CaseIterable {
    case week = "Week"
    case month = "Month"
    case quarter = "Quarter"
    case year = "Year"

    var dateComponent: Calendar.Component {
        switch self {
        case .week: return .weekOfYear
        case .month: return .month
        case .quarter: return .month
        case .year: return .year
        }
    }

    var value: Int {
        switch self {
        case .week: return 1
        case .month: return 1
        case .quarter: return 3
        case .year: return 1
        }
    }
}

// MARK: - Preview

struct FlareTimelineView_Previews: PreviewProvider {
    static var previews: some View {
        FlareTimelineView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
