//
//  StreakWidget.swift
//  InflamAIWidgetExtension
//
//  Logging streak widget - displays consecutive days of symptom logging
//

import WidgetKit
import SwiftUI

struct StreakWidget: Widget {
    let kind: String = "StreakWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: StreakProvider()) { entry in
            StreakWidgetEntryView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Logging Streak")
        .description("Track your consecutive days of symptom logging")
        .supportedFamilies([
            .systemSmall,
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
    }
}

struct StreakWidgetEntryView: View {
    @Environment(\.widgetFamily) var family
    let entry: StreakEntry

    var body: some View {
        switch family {
        case .systemSmall:
            StreakSmallView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/quicklog"))

        case .accessoryCircular:
            StreakCircularView(entry: entry)

        case .accessoryRectangular:
            StreakRectangularView(entry: entry)

        case .accessoryInline:
            StreakInlineView(entry: entry)

        default:
            StreakSmallView(entry: entry)
        }
    }
}

#Preview("Streak Small", as: .systemSmall) {
    StreakWidget()
} timeline: {
    StreakEntry(date: Date(), data: WidgetStreakData(streakDays: 12, lastUpdated: Date()))
}

#Preview("Streak Circular", as: .accessoryCircular) {
    StreakWidget()
} timeline: {
    StreakEntry(date: Date(), data: WidgetStreakData(streakDays: 7, lastUpdated: Date()))
}

#Preview("Streak Zero", as: .systemSmall) {
    StreakWidget()
} timeline: {
    StreakEntry(date: Date(), data: WidgetStreakData(streakDays: 0, lastUpdated: Date()))
}
