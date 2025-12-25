//
//  WatchBASDAIWidget.swift
//  InflamAIWatchWidgets
//
//  BASDAI score widget for Apple Watch Smart Stack and complications
//

import WidgetKit
import SwiftUI

// MARK: - Watch BASDAI Widget

struct WatchBASDAIWidget: Widget {
    let kind: String = "WatchBASDAIWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: WatchBASDAIProvider()) { entry in
            WatchBASDAIWidgetView(entry: entry)
        }
        .configurationDisplayName("BASDAI Score")
        .description("Track your disease activity")
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

struct WatchBASDAIProvider: TimelineProvider {
    typealias Entry = WatchBASDAIEntry

    func placeholder(in context: Context) -> WatchBASDAIEntry {
        WatchBASDAIEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (WatchBASDAIEntry) -> Void) {
        let data = WatchDataProvider.shared.getBASDAIData()
        completion(WatchBASDAIEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<WatchBASDAIEntry>) -> Void) {
        let data = WatchDataProvider.shared.getBASDAIData()
        let entry = WatchBASDAIEntry(date: Date(), data: data)

        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 30, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Entry

struct WatchBASDAIEntry: TimelineEntry {
    let date: Date
    let data: WatchBASDAIData
}

// MARK: - Views

struct WatchBASDAIWidgetView: View {
    @Environment(\.widgetFamily) var family
    let entry: WatchBASDAIEntry

    var body: some View {
        switch family {
        case .accessoryCircular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchBASDAICircularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchBASDAICircularView(data: entry.data)
            }

        case .accessoryRectangular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchBASDAIRectangularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchBASDAIRectangularView(data: entry.data)
            }

        #if os(watchOS)
        case .accessoryCorner:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchBASDAICornerView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchBASDAICornerView(data: entry.data)
            }
        #endif

        case .accessoryInline:
            Text("BASDAI: \(String(format: "%.1f", entry.data.score))")

        default:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchBASDAICircularView(data: entry.data)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchBASDAICircularView(data: entry.data)
            }
        }
    }
}

struct WatchBASDAICircularView: View {
    let data: WatchBASDAIData

    var body: some View {
        Gauge(value: data.score, in: 0...10) {
            Text("BASDAI")
                .font(.system(size: 8))
        } currentValueLabel: {
            Text(String(format: "%.1f", data.score))
                .font(.system(.title3, design: .rounded).weight(.bold))
        }
        .gaugeStyle(.accessoryCircular)
        .tint(data.severityColor)
    }
}

struct WatchBASDAIRectangularView: View {
    let data: WatchBASDAIData

    var body: some View {
        HStack {
            // Score display
            ZStack {
                Circle()
                    .fill(data.severityColor.opacity(0.2))
                    .frame(width: 36, height: 36)

                Text(String(format: "%.1f", data.score))
                    .font(.system(.body, design: .rounded).weight(.bold))
                    .foregroundColor(data.severityColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("BASDAI")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                HStack(spacing: 4) {
                    Text(data.category)
                        .font(.headline)

                    Image(systemName: data.trend.icon)
                        .font(.caption)
                        .foregroundColor(data.trend.color)
                }
            }

            Spacer()
        }
    }
}

#if os(watchOS)
struct WatchBASDAICornerView: View {
    let data: WatchBASDAIData

    var body: some View {
        Text(String(format: "%.1f", data.score))
            .font(.system(.title3, design: .rounded).weight(.bold))
            .foregroundColor(data.severityColor)
            .widgetLabel {
                Gauge(value: data.score, in: 0...10) {
                    Text("BASDAI")
                }
                .gaugeStyle(.accessoryLinear)
                .tint(data.severityColor)
            }
    }
}
#endif
