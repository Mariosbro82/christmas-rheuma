//
//  TrendsView.swift
//  InflamAI
//
//  Complete trends visualization with Swift Charts
//  BASDAI timeline, correlation insights, pattern detection
//

import SwiftUI
import Charts
import CoreData

struct TrendsView: View {
    @StateObject private var viewModel: TrendsViewModel
    @State private var selectedTimeRange: TimeRange = .month
    @State private var selectedMetric: TrendMetric = .basdai
    @State private var showingExportSheet = false
    @State private var animateContent = false

    init(context: NSManagedObjectContext = InflamAIPersistenceController.shared.container.viewContext) {
        _viewModel = StateObject(wrappedValue: TrendsViewModel(context: context))
    }

    var body: some View {
        // CRIT-001 FIX: Removed NavigationView wrapper.
        // This view is presented via NavigationLink from MoreView,
        // which is already wrapped in NavigationView in MainTabView.
        // Nested navigation containers cause duplicate back arrows.
        ScrollView {
            VStack(spacing: Spacing.lg) {
                // Time Range Selector
                timeRangePicker
                    .opacity(animateContent ? 1 : 0)
                    .offset(y: animateContent ? 0 : -10)

                // Main Chart
                mainChartSection
                    .opacity(animateContent ? 1 : 0)
                    .offset(y: animateContent ? 0 : 20)

                // Statistics Cards
                statisticsSection
                    .opacity(animateContent ? 1 : 0)
                    .offset(y: animateContent ? 0 : 20)

                // Trigger Insights
                triggerInsightsSection
                    .opacity(animateContent ? 1 : 0)
                    .offset(y: animateContent ? 0 : 20)

                // Pattern Detection
                patternDetectionSection
                    .opacity(animateContent ? 1 : 0)
                    .offset(y: animateContent ? 0 : 20)

                // Flare History
                flareHistorySection
                    .opacity(animateContent ? 1 : 0)
                    .offset(y: animateContent ? 0 : 20)
            }
            .padding(Spacing.md)
        }
        .navigationTitle("Trends & Insights")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showingExportSheet = true
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .accessibilityLabel("Export data")
            }
        }
        .sheet(isPresented: $showingExportSheet) {
            ExportOptionsView(viewModel: viewModel)
        }
        .onAppear {
            viewModel.loadData(timeRange: selectedTimeRange)
            withAnimation(Animations.spring.delay(0.1)) {
                animateContent = true
            }
        }
        .onChange(of: selectedTimeRange) { newRange in
            withAnimation(Animations.easeOut) {
                animateContent = false
            }
            viewModel.loadData(timeRange: newRange)
            withAnimation(Animations.spring.delay(0.2)) {
                animateContent = true
            }
        }
    }

    // MARK: - Time Range Picker

    private var timeRangePicker: some View {
        Picker("Time Range", selection: $selectedTimeRange) {
            Text("Week").tag(TimeRange.week)
            Text("Month").tag(TimeRange.month)
            Text("3 Months").tag(TimeRange.threeMonths)
            Text("Year").tag(TimeRange.year)
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("Select time range for trends")
    }

    // MARK: - Main Chart Section

    private var mainChartSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                Text("BASDAI Trend")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)

                Spacer()

                Picker("Metric", selection: $selectedMetric) {
                    Text("BASDAI").tag(TrendMetric.basdai)
                    Text("Pain").tag(TrendMetric.pain)
                    Text("Stiffness").tag(TrendMetric.stiffness)
                    Text("Fatigue").tag(TrendMetric.fatigue)
                    Text("Assessments").tag(TrendMetric.assessments)
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }

            if viewModel.isLoading {
                skeletonLoadingView
            } else if viewModel.dataPoints.isEmpty && selectedMetric != .assessments {
                emptyStateView
            } else if selectedMetric == .assessments && viewModel.assessmentDataByType.isEmpty {
                emptyStateView
            } else {
                chartView

                // Show legend for assessments
                if selectedMetric == .assessments && !viewModel.assessmentDataByType.isEmpty {
                    assessmentLegend
                }
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    private var assessmentLegend: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Assessment Legend")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                ForEach(viewModel.assessmentDataByType.keys.sorted(), id: \.self) { assessmentType in
                    HStack(spacing: 6) {
                        Circle()
                            .fill(viewModel.colorForAssessment(assessmentType))
                            .frame(width: 10, height: 10)

                        Text(assessmentType)
                            .font(.caption)
                            .foregroundColor(.primary)

                        Spacer()
                    }
                }
            }
        }
        .padding(.top, 8)
    }

    private var chartView: some View {
        Chart {
            if selectedMetric == .assessments {
                // Multi-line chart for all assessments
                ForEach(viewModel.assessmentDataByType.keys.sorted(), id: \.self) { assessmentType in
                    if let dataPoints = viewModel.assessmentDataByType[assessmentType] {
                        ForEach(dataPoints) { point in
                            LineMark(
                                x: .value("Date", point.date),
                                y: .value("Score", point.score)
                            )
                            .foregroundStyle(viewModel.colorForAssessment(assessmentType))
                            .lineStyle(StrokeStyle(lineWidth: 2))

                            PointMark(
                                x: .value("Date", point.date),
                                y: .value("Score", point.score)
                            )
                            .foregroundStyle(viewModel.colorForAssessment(assessmentType))
                            .symbolSize(30)
                        }
                    }
                }
            } else {
                // Single metric chart
                ForEach(viewModel.dataPoints) { point in
                    LineMark(
                        x: .value("Date", point.date),
                        y: .value(selectedMetric.rawValue, point.value(for: selectedMetric))
                    )
                    .foregroundStyle(selectedMetric.color)
                    .lineStyle(StrokeStyle(lineWidth: 3))

                    AreaMark(
                        x: .value("Date", point.date),
                        y: .value(selectedMetric.rawValue, point.value(for: selectedMetric))
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [selectedMetric.color.opacity(0.3), selectedMetric.color.opacity(0.05)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )

                    // Mark flare events
                    if point.isFlare {
                        PointMark(
                            x: .value("Date", point.date),
                            y: .value(selectedMetric.rawValue, point.value(for: selectedMetric))
                        )
                        .foregroundStyle(Colors.Semantic.error)
                        .symbolSize(100)
                        .annotation(position: .top) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(Colors.Semantic.error)
                                .font(.caption)
                        }
                    }
                }

                // Average line
                if let average = viewModel.averageValue(for: selectedMetric) {
                    RuleMark(y: .value("Average", average))
                        .foregroundStyle(Colors.Gray.g500.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
                        .annotation(position: .trailing, alignment: .center) {
                            Text("Avg: \(average, specifier: "%.1f")")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                }
            }
        }
        .frame(height: 300)
        .chartXAxis {
            AxisMarks(values: .stride(by: viewModel.xAxisStride)) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.day().month(.abbreviated))
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel()
            }
        }
        .chartYScale(domain: 0...10)
        .accessibilityLabel("\(selectedMetric.rawValue) trend chart")
        .accessibilityValue(selectedMetric == .assessments ? "Multiple assessment scores over time" : "Average \(selectedMetric.rawValue): \(viewModel.averageValue(for: selectedMetric) ?? 0, specifier: "%.1f")")
    }

    private var emptyStateView: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "chart.xyaxis.line")
                .font(.system(size: 60))
                .foregroundColor(Colors.Gray.g300)

            Text("No Data Yet")
                .font(.system(size: Typography.lg, weight: .semibold))
                .foregroundColor(Colors.Gray.g700)

            Text("Complete daily check-ins to see your trends")
                .font(.system(size: Typography.base))
                .foregroundColor(Colors.Gray.g500)
                .multilineTextAlignment(.center)
        }
        .frame(height: 300)
        .frame(maxWidth: .infinity)
    }

    private var skeletonLoadingView: some View {
        VStack(spacing: Spacing.sm) {
            // Skeleton chart bars
            HStack(alignment: .bottom, spacing: Spacing.xs) {
                ForEach(0..<12, id: \.self) { index in
                    RoundedRectangle(cornerRadius: Radii.xs)
                        .fill(Colors.Gray.g200)
                        .frame(width: 20, height: CGFloat.random(in: 80...250))
                        .opacity(0.6)
                }
            }
            .frame(height: 280)

            // Skeleton x-axis labels
            HStack {
                ForEach(0..<4, id: \.self) { _ in
                    RoundedRectangle(cornerRadius: Radii.xs)
                        .fill(Colors.Gray.g200)
                        .frame(width: 50, height: 12)
                    Spacer()
                }
            }
        }
        .shimmer()
    }

    // MARK: - Statistics Section

    private var statisticsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Text("Statistics")
                .font(.system(size: Typography.md, weight: .semibold))
                .foregroundColor(Colors.Gray.g900)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                StatCard(
                    title: "Average BASDAI",
                    value: String(format: "%.1f", viewModel.averageValue(for: .basdai) ?? 0),
                    trend: viewModel.trend(for: .basdai),
                    color: Colors.Primary.p500
                )

                StatCard(
                    title: "Flare Days",
                    value: "\(viewModel.flareDayCount)",
                    subtitle: "in \(selectedTimeRange.displayName)",
                    color: Colors.Semantic.error
                )

                StatCard(
                    title: "Best Score",
                    value: String(format: "%.1f", viewModel.minValue(for: .basdai) ?? 0),
                    subtitle: viewModel.bestScoreDate,
                    color: Colors.Semantic.success
                )

                StatCard(
                    title: "Check-In Streak",
                    value: "\(viewModel.currentStreak)",
                    subtitle: "days",
                    color: Colors.Semantic.warning
                )
            }
        }
    }

    // MARK: - Trigger Insights Section

    private var triggerInsightsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(Colors.Semantic.warning)
                Text("Your Top Triggers")
                    .font(.system(size: Typography.md, weight: .semibold))
                    .foregroundColor(Colors.Gray.g900)
            }

            if viewModel.topTriggers.isEmpty {
                Text("Not enough data to identify triggers. Continue logging to discover your patterns.")
                    .font(.system(size: Typography.base))
                    .foregroundColor(Colors.Gray.g500)
                    .padding(Spacing.md)
            } else {
                ForEach(viewModel.topTriggers) { trigger in
                    TriggerInsightCard(trigger: trigger)
                }
            }
        }
        .padding(Spacing.md)
        .background(Color(.systemBackground))
        .cornerRadius(Radii.xl)
        .dshadow(Shadows.sm)
    }

    // MARK: - Pattern Detection Section

    private var patternDetectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "sparkles")
                    .foregroundColor(Colors.Accent.purple)
                Text("Detected Patterns")
                    .font(.headline)
            }

            if let patterns = viewModel.detectedPatterns, !patterns.isEmpty {
                ForEach(patterns, id: \.id) { pattern in
                    PatternCard(pattern: pattern)
                }
            } else {
                Text("Continue tracking to detect patterns in your symptoms")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding()
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
    }

    // MARK: - Flare History Section

    private var flareHistorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Flare History")
                .font(.headline)

            if viewModel.flareEvents.isEmpty {
                Text("No flare events recorded in this period")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding()
            } else {
                ForEach(viewModel.flareEvents) { flare in
                    FlareEventRow(flare: flare)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
}

// MARK: - Supporting Views

struct StatCard: View {
    let title: String
    let value: String
    var trend: TrendDirection? = nil
    var subtitle: String? = nil
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                if let trend = trend {
                    trendIndicator(trend)
                }
            }

            Text(value)
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(color)

            if let subtitle = subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    @ViewBuilder
    private func trendIndicator(_ trend: TrendDirection) -> some View {
        HStack(spacing: 4) {
            Image(systemName: trend.iconName)
            Text(trend.percentageText)
        }
        .font(.caption)
        .foregroundColor(trend.color)
    }
}

struct TriggerInsightCard: View {
    let trigger: Trigger

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: trigger.icon)
                    .foregroundColor(Colors.Primary.p500)
                    .frame(width: 24)

                VStack(alignment: .leading, spacing: 4) {
                    Text(trigger.name)
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    Text(trigger.strengthIcon + " " + String(describing: trigger.strength).capitalized + " Correlation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Text(String(format: "r=%.2f", trigger.correlation))
                    .font(.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Colors.Primary.p500.opacity(0.1))
                    .cornerRadius(8)
            }

            Text(trigger.explanation)
                .font(.caption)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct PatternCard: View {
    let pattern: DetectedPattern

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: pattern.iconName)
                .font(.title2)
                .foregroundColor(pattern.color)
                .frame(width: 40)

            VStack(alignment: .leading, spacing: 4) {
                Text(pattern.title)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Text(pattern.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct FlareEventRow: View {
    let flare: TrendsFlareData

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(flare.date, style: .date)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                if let triggers = flare.suspectedTriggers {
                    Text(triggers)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text("Severity: \(flare.severity)/10")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(severityColor(flare.severity))

                if let duration = flare.durationDays {
                    Text("\(duration) days")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private func severityColor(_ severity: Int) -> Color {
        switch severity {
        case 0..<4: return Colors.Semantic.success
        case 4..<7: return Colors.Semantic.warning
        default: return Colors.Semantic.error
        }
    }
}

// MARK: - Export Options

struct ExportOptionsView: View {
    @ObservedObject var viewModel: TrendsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var exportFormat: ExportFormat = .pdf

    var body: some View {
        NavigationView {
            Form {
                Section {
                    Picker("Format", selection: $exportFormat) {
                        Text("PDF Report").tag(ExportFormat.pdf)
                        Text("CSV Data").tag(ExportFormat.csv)
                        Text("JSON Data").tag(ExportFormat.json)
                    }
                } header: {
                    Text("Export Format")
                }

                Section {
                    Button {
                        exportData()
                    } label: {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            Text("Export Data")
                        }
                    }
                }
            }
            .navigationTitle("Export Data")
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

    private func exportData() {
        // Export implementation
        Task {
            await viewModel.exportData(format: exportFormat)
            dismiss()
        }
    }
}

// MARK: - Supporting Enums

enum TrendMetric: String {
    case basdai = "BASDAI"
    case pain = "Pain"
    case stiffness = "Stiffness"
    case fatigue = "Fatigue"
    case assessments = "Assessments"

    var color: Color {
        switch self {
        case .basdai: return Colors.Primary.p500
        case .pain: return Colors.Semantic.error
        case .stiffness: return Colors.Semantic.warning
        case .fatigue: return Colors.Accent.purple
        case .assessments: return Colors.Semantic.success
        }
    }
}

enum TrendDirection {
    case up(Double)
    case down(Double)
    case stable

    var iconName: String {
        switch self {
        case .up: return "arrow.up.right"
        case .down: return "arrow.down.right"
        case .stable: return "arrow.right"
        }
    }

    var color: Color {
        switch self {
        case .up: return Colors.Semantic.error
        case .down: return Colors.Semantic.success
        case .stable: return Colors.Gray.g500
        }
    }

    var percentageText: String {
        switch self {
        case .up(let percent): return "+\(Int(percent))%"
        case .down(let percent): return "-\(Int(percent))%"
        case .stable: return "0%"
        }
    }
}

enum ExportFormat {
    case pdf
    case csv
    case json
}

// MARK: - Preview

struct TrendsView_Previews: PreviewProvider {
    static var previews: some View {
        TrendsView(context: InflamAIPersistenceController.preview.container.viewContext)
    }
}
