//
//  DailyDashboardWidget.swift
//  InflamAIWidgetExtension
//
//  Large daily dashboard widget - comprehensive health overview
//

import WidgetKit
import SwiftUI

// MARK: - Dashboard Entry

struct DashboardEntry: TimelineEntry {
    let date: Date
    let flareData: WidgetFlareData
    let basdaiData: WidgetBASDAIData
    let streakData: WidgetStreakData
    let todaySummary: WidgetTodaySummary
}

// MARK: - Dashboard Provider

struct DashboardProvider: TimelineProvider {
    typealias Entry = DashboardEntry

    func placeholder(in context: Context) -> DashboardEntry {
        DashboardEntry(
            date: Date(),
            flareData: .placeholder,
            basdaiData: .placeholder,
            streakData: .placeholder,
            todaySummary: .placeholder
        )
    }

    func getSnapshot(in context: Context, completion: @escaping (DashboardEntry) -> Void) {
        let provider = WidgetDataProvider.shared
        let entry = DashboardEntry(
            date: Date(),
            flareData: provider.getFlareRiskData(),
            basdaiData: provider.getBASDAIData(),
            streakData: provider.getStreakData(),
            todaySummary: provider.getTodaySummaryData()
        )
        completion(entry)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<DashboardEntry>) -> Void) {
        let provider = WidgetDataProvider.shared
        let entry = DashboardEntry(
            date: Date(),
            flareData: provider.getFlareRiskData(),
            basdaiData: provider.getBASDAIData(),
            streakData: provider.getStreakData(),
            todaySummary: provider.getTodaySummaryData()
        )

        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Widget Definition

struct DailyDashboardWidget: Widget {
    let kind: String = "DailyDashboardWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: DashboardProvider()) { entry in
            DashboardWidgetView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Daily Dashboard")
        .description("Your complete daily health overview")
        .supportedFamilies([.systemLarge, .systemMedium])
    }
}

// MARK: - Dashboard Views

struct DashboardWidgetView: View {
    @Environment(\.widgetFamily) var family
    let entry: DashboardEntry

    var body: some View {
        switch family {
        case .systemLarge:
            DashboardLargeView(entry: entry)
        case .systemMedium:
            DashboardMediumView(entry: entry)
        default:
            DashboardMediumView(entry: entry)
        }
    }
}

struct DashboardMediumView: View {
    let entry: DashboardEntry

    var body: some View {
        HStack(spacing: 16) {
            // Left: Flare Risk Gauge
            VStack(spacing: 4) {
                FlareRiskGaugeView(
                    percentage: entry.flareData.riskPercentage,
                    riskLevel: entry.flareData.riskLevel,
                    size: 60
                )

                Text("Pattern")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            // Center: BASDAI & Today's Summary
            VStack(alignment: .leading, spacing: 8) {
                // BASDAI row
                HStack {
                    Text("BASDAI")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    Text(String(format: "%.1f", entry.basdaiData.score))
                        .font(.headline)
                        .foregroundColor(entry.basdaiData.severityColor)

                    Image(systemName: entry.basdaiData.trend.icon)
                        .font(.caption)
                        .foregroundColor(entry.basdaiData.trend.color)
                }

                Divider()

                // Today's activity
                HStack(spacing: 12) {
                    Label("\(entry.todaySummary.painEntries)", systemImage: "bolt.fill")
                        .font(.caption)

                    Label("\(entry.todaySummary.assessments)", systemImage: "chart.bar.fill")
                        .font(.caption)

                    Spacer()

                    if entry.todaySummary.hasActiveFlare {
                        Label("Flare", systemImage: "flame.fill")
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
            }

            // Right: Streak
            VStack(spacing: 4) {
                Image(systemName: "flame.fill")
                    .font(.title2)
                    .foregroundColor(.orange)

                Text("\(entry.streakData.streakDays)")
                    .font(.title3)
                    .fontWeight(.bold)

                Text("days")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .widgetURL(URL(string: "spinalytics://widget/dashboard"))
    }
}

struct DashboardLargeView: View {
    let entry: DashboardEntry

    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                VStack(alignment: .leading) {
                    Text("Daily Health")
                        .font(.headline)
                    Text(entry.date, style: .date)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Streak badge
                HStack(spacing: 4) {
                    Image(systemName: "flame.fill")
                        .foregroundColor(.orange)
                    Text("\(entry.streakData.streakDays)")
                        .fontWeight(.bold)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(Color.orange.opacity(0.2))
                .cornerRadius(8)
            }

            // Active Flare Alert
            if entry.todaySummary.hasActiveFlare {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                    Text("Active Flare")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(10)
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }

            // Main metrics row
            HStack(spacing: 20) {
                // Flare Risk
                VStack(spacing: 8) {
                    FlareRiskGaugeView(
                        percentage: entry.flareData.riskPercentage,
                        riskLevel: entry.flareData.riskLevel,
                        size: 80
                    )

                    Text(entry.flareData.riskLevel.displayName)
                        .font(.caption)
                        .foregroundColor(entry.flareData.riskLevel.color)
                }

                // BASDAI
                VStack(spacing: 8) {
                    Text(String(format: "%.1f", entry.basdaiData.score))
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(entry.basdaiData.severityColor)

                    HStack(spacing: 4) {
                        Text("BASDAI")
                            .font(.caption)

                        Image(systemName: entry.basdaiData.trend.icon)
                            .font(.caption)
                            .foregroundColor(entry.basdaiData.trend.color)
                    }
                    .foregroundColor(.secondary)
                }
            }

            Divider()

            // Today's Summary
            VStack(alignment: .leading, spacing: 8) {
                Text("Today's Activity")
                    .font(.subheadline)
                    .fontWeight(.medium)

                HStack(spacing: 16) {
                    SummaryItem(
                        icon: "bolt.circle.fill",
                        color: .red,
                        value: "\(entry.todaySummary.painEntries)",
                        label: "Pain"
                    )

                    SummaryItem(
                        icon: "chart.bar.fill",
                        color: .blue,
                        value: "\(entry.todaySummary.assessments)",
                        label: "Assess"
                    )

                    SummaryItem(
                        icon: "checkmark.circle.fill",
                        color: entry.todaySummary.hasLoggedToday ? .green : .gray,
                        value: entry.todaySummary.hasLoggedToday ? "Yes" : "No",
                        label: "Logged"
                    )
                }
            }

            // Risk Factors
            if !entry.flareData.topFactors.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Risk Factors")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        ForEach(entry.flareData.topFactors.prefix(3), id: \.self) { factor in
                            Text(factor)
                                .font(.caption2)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.orange.opacity(0.2))
                                .cornerRadius(4)
                        }
                    }
                }
            }
        }
        .padding()
        .widgetURL(URL(string: "spinalytics://widget/dashboard"))
    }
}

struct SummaryItem: View {
    let icon: String
    let color: Color
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)

            Text(value)
                .font(.headline)

            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview("Dashboard Large", as: .systemLarge) {
    DailyDashboardWidget()
} timeline: {
    DashboardEntry(
        date: Date(),
        flareData: .placeholder,
        basdaiData: .placeholder,
        streakData: .placeholder,
        todaySummary: .placeholder
    )
}

#Preview("Dashboard Medium", as: .systemMedium) {
    DailyDashboardWidget()
} timeline: {
    DashboardEntry(
        date: Date(),
        flareData: .placeholder,
        basdaiData: .placeholder,
        streakData: .placeholder,
        todaySummary: .placeholder
    )
}
