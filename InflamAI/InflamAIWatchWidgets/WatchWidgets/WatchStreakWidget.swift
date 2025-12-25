//
//  WatchStreakWidget.swift
//  InflamAIWatchWidgets
//
//  Logging streak widget for Apple Watch
//

import WidgetKit
import SwiftUI

// MARK: - Watch Streak Widget

struct WatchStreakWidget: Widget {
    let kind: String = "WatchStreakWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: WatchStreakProvider()) { entry in
            WatchStreakWidgetView(entry: entry)
        }
        .configurationDisplayName("Streak")
        .description("Your logging streak")
        #if os(watchOS)
        .supportedFamilies([
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryCorner,
            .accessoryInline
        ])
        #else
        .supportedFamilies([
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
        #endif
    }
}

// MARK: - Provider

struct WatchStreakProvider: TimelineProvider {
    typealias Entry = WatchStreakEntry

    func placeholder(in context: Context) -> WatchStreakEntry {
        WatchStreakEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (WatchStreakEntry) -> Void) {
        let data = WatchDataProvider.shared.getStreakData()
        completion(WatchStreakEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<WatchStreakEntry>) -> Void) {
        let data = WatchDataProvider.shared.getStreakData()
        let entry = WatchStreakEntry(date: Date(), data: data)

        // Update at midnight
        let tomorrow = Calendar.current.startOfDay(for: Date().addingTimeInterval(86400))
        let timeline = Timeline(entries: [entry], policy: .after(tomorrow))
        completion(timeline)
    }
}

// MARK: - Entry

struct WatchStreakEntry: TimelineEntry {
    let date: Date
    let data: WatchStreakData
}

// MARK: - Views

struct WatchStreakWidgetView: View {
    @Environment(\.widgetFamily) var family
    let entry: WatchStreakEntry

    var body: some View {
        switch family {
        case .accessoryCircular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchStreakCircularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchStreakCircularView(data: entry.data)
            }

        case .accessoryRectangular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchStreakRectangularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchStreakRectangularView(data: entry.data)
            }

        #if os(watchOS)
        case .accessoryCorner:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchStreakCornerView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchStreakCornerView(data: entry.data)
            }
        #endif

        case .accessoryInline:
            Label("\(entry.data.streakDays) day streak", systemImage: "flame.fill")

        default:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchStreakCircularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchStreakCircularView(data: entry.data)
            }
        }
    }
}

struct WatchStreakCircularView: View {
    let data: WatchStreakData

    var body: some View {
        ZStack {
            AccessoryWidgetBackground()

            VStack(spacing: 0) {
                Image(systemName: "flame.fill")
                    .font(.system(size: 16))
                    .foregroundColor(.orange)

                Text("\(data.streakDays)")
                    .font(.system(.title3, design: .rounded).weight(.bold))
            }
        }
    }
}

struct WatchStreakRectangularView: View {
    let data: WatchStreakData

    var body: some View {
        HStack {
            Image(systemName: "flame.fill")
                .font(.title2)
                .foregroundColor(.orange)

            VStack(alignment: .leading, spacing: 2) {
                Text("Streak")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text("\(data.streakDays) \(data.streakDays == 1 ? "day" : "days")")
                    .font(.headline)
            }

            Spacer()
        }
    }
}

#if os(watchOS)
struct WatchStreakCornerView: View {
    let data: WatchStreakData

    var body: some View {
        Text("\(data.streakDays)")
            .font(.system(.title3, design: .rounded).weight(.bold))
            .foregroundColor(.orange)
            .widgetLabel {
                Label("Day Streak", systemImage: "flame.fill")
            }
    }
}
#endif
