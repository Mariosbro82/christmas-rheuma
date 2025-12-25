//
//  BASDAIWidget.swift
//  InflamAIWidgetExtension
//
//  BASDAI score widget - displays current disease activity index
//

import WidgetKit
import SwiftUI

struct BASDAIWidget: Widget {
    let kind: String = "BASDAIWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: BASDAIProvider()) { entry in
            BASDAIWidgetEntryView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("BASDAI Score")
        .description("Track your Bath Ankylosing Spondylitis Disease Activity Index")
        .supportedFamilies([
            .systemSmall,
            .systemMedium,
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
    }
}

struct BASDAIWidgetEntryView: View {
    @Environment(\.widgetFamily) var family
    let entry: BASDAIEntry

    var body: some View {
        switch family {
        case .systemSmall:
            BASDAISmallView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/basdai"))

        case .systemMedium:
            BASDAIMediumView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/basdai"))

        case .accessoryCircular:
            BASDAICircularView(entry: entry)

        case .accessoryRectangular:
            BASDAIRectangularView(entry: entry)

        case .accessoryInline:
            BASDAIInlineView(entry: entry)

        default:
            BASDAISmallView(entry: entry)
        }
    }
}

#Preview("BASDAI Small", as: .systemSmall) {
    BASDAIWidget()
} timeline: {
    BASDAIEntry(date: Date(), data: WidgetBASDAIData(
        score: 4.2,
        category: "Moderate",
        trend: .stable,
        lastAssessed: Date()
    ))
}

#Preview("BASDAI Medium", as: .systemMedium) {
    BASDAIWidget()
} timeline: {
    BASDAIEntry(date: Date(), data: WidgetBASDAIData(
        score: 5.8,
        category: "Moderate-High",
        trend: .worsening,
        lastAssessed: Date().addingTimeInterval(-3600)
    ))
}

#Preview("BASDAI Circular", as: .accessoryCircular) {
    BASDAIWidget()
} timeline: {
    BASDAIEntry(date: Date(), data: .placeholder)
}
