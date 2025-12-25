//
//  FlareRiskWidget.swift
//  InflamAIWidgetExtension
//
//  Flare Risk widget - displays current AI-predicted flare risk
//

import WidgetKit
import SwiftUI

struct FlareRiskWidget: Widget {
    let kind: String = "FlareRiskWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: SimpleFlareRiskProvider()) { entry in
            FlareRiskWidgetEntryView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Pattern Monitor")
        .description("View your daily symptom patterns at a glance")
        .supportedFamilies([
            .systemSmall,
            .systemMedium,
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
    }
}

struct FlareRiskWidgetEntryView: View {
    @Environment(\.widgetFamily) var family
    let entry: FlareRiskEntry

    var body: some View {
        switch family {
        case .systemSmall:
            FlareRiskSmallView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/flare"))

        case .systemMedium:
            FlareRiskMediumView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/flare"))

        case .accessoryCircular:
            LockScreenFlareGaugeView(
                percentage: entry.data.riskPercentage,
                riskLevel: entry.data.riskLevel
            )

        case .accessoryRectangular:
            FlareRiskRectangularView(entry: entry)

        case .accessoryInline:
            FlareRiskInlineView(entry: entry)

        default:
            FlareRiskSmallView(entry: entry)
        }
    }
}

#Preview("Flare Risk Small", as: .systemSmall) {
    FlareRiskWidget()
} timeline: {
    FlareRiskEntry(date: Date(), data: WidgetFlareData(
        riskPercentage: 42,
        riskLevel: .moderate,
        topFactors: ["Weather", "Sleep"],
        lastUpdated: Date()
    ))
}

#Preview("Flare Risk Medium", as: .systemMedium) {
    FlareRiskWidget()
} timeline: {
    FlareRiskEntry(date: Date(), data: WidgetFlareData(
        riskPercentage: 67,
        riskLevel: .high,
        topFactors: ["Pressure Drop", "Poor Sleep"],
        lastUpdated: Date()
    ))
}

#Preview("Flare Risk Circular", as: .accessoryCircular) {
    FlareRiskWidget()
} timeline: {
    FlareRiskEntry(date: Date(), data: .placeholder)
}

#Preview("Flare Risk Rectangular", as: .accessoryRectangular) {
    FlareRiskWidget()
} timeline: {
    FlareRiskEntry(date: Date(), data: .placeholder)
}
