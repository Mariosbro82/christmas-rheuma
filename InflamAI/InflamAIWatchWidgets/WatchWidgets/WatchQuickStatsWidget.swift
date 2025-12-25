//
//  WatchQuickStatsWidget.swift
//  InflamAIWatchWidgets
//
//  Quick health stats widget for Apple Watch Smart Stack
//

import WidgetKit
import SwiftUI

// MARK: - Watch Quick Stats Widget

struct WatchQuickStatsWidget: Widget {
    let kind: String = "WatchQuickStatsWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: WatchQuickStatsProvider()) { entry in
            WatchQuickStatsWidgetView(entry: entry)
        }
        .configurationDisplayName("Quick Stats")
        .description("Your key health metrics at a glance")
        .supportedFamilies([.accessoryRectangular])
    }
}

// MARK: - Provider

struct WatchQuickStatsProvider: TimelineProvider {
    typealias Entry = WatchQuickStatsEntry

    func placeholder(in context: Context) -> WatchQuickStatsEntry {
        WatchQuickStatsEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (WatchQuickStatsEntry) -> Void) {
        let data = WatchDataProvider.shared.getQuickStatsData()
        completion(WatchQuickStatsEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<WatchQuickStatsEntry>) -> Void) {
        let data = WatchDataProvider.shared.getQuickStatsData()
        let entry = WatchQuickStatsEntry(date: Date(), data: data)

        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Entry

struct WatchQuickStatsEntry: TimelineEntry {
    let date: Date
    let data: WatchQuickStatsData
}

// MARK: - Data Model

struct WatchQuickStatsData {
    let steps: Int
    let hrv: Double
    let painLevel: Int

    static var placeholder: WatchQuickStatsData {
        WatchQuickStatsData(steps: 4500, hrv: 45.0, painLevel: 3)
    }
}

// MARK: - Views

struct WatchQuickStatsWidgetView: View {
    let entry: WatchQuickStatsEntry

    var body: some View {
        if #available(iOS 17.0, watchOS 10.0, *) {
            content
                .containerBackground(.fill.tertiary, for: .widget)
        } else {
            content
        }
    }

    private var content: some View {
        HStack(spacing: 8) {
            // Steps
            VStack(spacing: 2) {
                Image(systemName: "figure.walk")
                    .font(.caption2)
                Text("\(entry.data.steps)")
                    .font(.system(.caption, design: .rounded).weight(.bold))
            }

            Divider()
                .frame(height: 24)

            // HRV
            VStack(spacing: 2) {
                Image(systemName: "heart.text.square")
                    .font(.caption2)
                Text("\(Int(entry.data.hrv))")
                    .font(.system(.caption, design: .rounded).weight(.bold))
            }

            Divider()
                .frame(height: 24)

            // Pain Level
            VStack(spacing: 2) {
                Image(systemName: "bolt.fill")
                    .font(.caption2)
                    .foregroundColor(painColor)
                Text("\(entry.data.painLevel)")
                    .font(.system(.caption, design: .rounded).weight(.bold))
                    .foregroundColor(painColor)
            }
        }
    }

    private var painColor: Color {
        switch entry.data.painLevel {
        case 0...3: return .green
        case 4...6: return .orange
        default: return .red
        }
    }
}
