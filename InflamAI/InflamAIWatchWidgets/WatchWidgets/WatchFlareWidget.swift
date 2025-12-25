//
//  WatchFlareWidget.swift
//  InflamAIWatchWidgets
//
//  Flare risk widget for Apple Watch Smart Stack and complications
//

import WidgetKit
import SwiftUI

// MARK: - Watch Flare Risk Widget

struct WatchFlareRiskWidget: Widget {
    let kind: String = "WatchFlareRiskWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: WatchFlareProvider()) { entry in
            WatchFlareWidgetView(entry: entry)
        }
        .configurationDisplayName("Pattern Status")
        .description("View your symptom patterns")
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

struct WatchFlareProvider: TimelineProvider {
    typealias Entry = WatchFlareEntry

    func placeholder(in context: Context) -> WatchFlareEntry {
        WatchFlareEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (WatchFlareEntry) -> Void) {
        let data = WatchDataProvider.shared.getFlareRiskData()
        completion(WatchFlareEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<WatchFlareEntry>) -> Void) {
        let data = WatchDataProvider.shared.getFlareRiskData()
        let entry = WatchFlareEntry(date: Date(), data: data)

        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Entry

struct WatchFlareEntry: TimelineEntry {
    let date: Date
    let data: WatchFlareData
}

// MARK: - Views

struct WatchFlareWidgetView: View {
    @Environment(\.widgetFamily) var family
    let entry: WatchFlareEntry

    var body: some View {
        switch family {
        case .accessoryCircular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchFlareCircularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchFlareCircularView(data: entry.data)
            }

        case .accessoryRectangular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchFlareRectangularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchFlareRectangularView(data: entry.data)
            }

        #if os(watchOS)
        case .accessoryCorner:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchFlareCornerView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchFlareCornerView(data: entry.data)
            }
        #endif

        case .accessoryInline:
            Text("Flare: \(entry.data.riskPercentage)%")

        default:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchFlareCircularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchFlareCircularView(data: entry.data)
            }
        }
    }
}

struct WatchFlareCircularView: View {
    let data: WatchFlareData

    var body: some View {
        Gauge(value: Double(data.riskPercentage), in: 0...100) {
            Image(systemName: data.riskLevel.icon)
        } currentValueLabel: {
            Text("\(data.riskPercentage)")
                .font(.system(.title3, design: .rounded).weight(.bold))
        }
        .gaugeStyle(.accessoryCircular)
        .tint(data.riskLevel.color)
    }
}

struct WatchFlareRectangularView: View {
    let data: WatchFlareData

    var body: some View {
        HStack {
            // Gauge
            Gauge(value: Double(data.riskPercentage), in: 0...100) {
                EmptyView()
            } currentValueLabel: {
                Text("\(data.riskPercentage)%")
                    .font(.system(.caption, design: .rounded).weight(.bold))
            }
            .gaugeStyle(.accessoryCircular)
            .tint(data.riskLevel.color)
            .scaleEffect(0.8)

            VStack(alignment: .leading, spacing: 2) {
                Text("Pattern")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text(data.riskLevel.displayName)
                    .font(.headline)
                    .foregroundColor(data.riskLevel.color)
            }

            Spacer()
        }
    }
}

#if os(watchOS)
struct WatchFlareCornerView: View {
    let data: WatchFlareData

    var body: some View {
        Text("\(data.riskPercentage)%")
            .font(.system(.title3, design: .rounded).weight(.bold))
            .foregroundColor(data.riskLevel.color)
            .widgetLabel {
                Gauge(value: Double(data.riskPercentage), in: 0...100) {
                    Text("Risk")
                }
                .gaugeStyle(.accessoryLinear)
                .tint(data.riskLevel.color)
            }
    }
}
#endif
