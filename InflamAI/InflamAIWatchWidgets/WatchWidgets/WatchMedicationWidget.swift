//
//  WatchMedicationWidget.swift
//  InflamAIWatchWidgets
//
//  Medication reminder widget for Apple Watch Smart Stack
//

import WidgetKit
import SwiftUI

// MARK: - Watch Medication Widget

struct WatchMedicationWidget: Widget {
    let kind: String = "WatchMedicationWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: WatchMedicationProvider()) { entry in
            WatchMedicationWidgetView(entry: entry)
        }
        .configurationDisplayName("Medications")
        .description("Your upcoming medications")
        .supportedFamilies([
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
    }
}

// MARK: - Provider

struct WatchMedicationProvider: TimelineProvider {
    typealias Entry = WatchMedicationEntry

    func placeholder(in context: Context) -> WatchMedicationEntry {
        WatchMedicationEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (WatchMedicationEntry) -> Void) {
        let data = WatchDataProvider.shared.getMedicationData()
        completion(WatchMedicationEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<WatchMedicationEntry>) -> Void) {
        let data = WatchDataProvider.shared.getMedicationData()

        var entries: [WatchMedicationEntry] = []
        let now = Date()

        entries.append(WatchMedicationEntry(date: now, data: data))

        // Update at each medication time
        for med in data.medications where med.time > now {
            entries.append(WatchMedicationEntry(date: med.time, data: data))
        }

        let nextUpdate = data.medications.first { $0.time > now }?.time
            ?? Calendar.current.date(byAdding: .minute, value: 5, to: now)!

        let timeline = Timeline(entries: entries, policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Entry

struct WatchMedicationEntry: TimelineEntry {
    let date: Date
    let data: WatchMedicationData

    var nextMedication: WatchMedicationData.MedicationReminder? {
        data.medications.first { $0.time > Date() }
    }
}

// MARK: - Views

struct WatchMedicationWidgetView: View {
    @Environment(\.widgetFamily) var family
    let entry: WatchMedicationEntry

    var body: some View {
        switch family {
        case .accessoryCircular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchMedicationCircularView(entry: entry)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchMedicationCircularView(entry: entry)
            }

        case .accessoryRectangular:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchMedicationRectangularView(entry: entry)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchMedicationRectangularView(entry: entry)
            }

        case .accessoryInline:
            if let nextMed = entry.nextMedication {
                Text("\(nextMed.name) \(nextMed.relativeTimeString)")
            } else {
                Text("No medications due")
            }

        default:
            if #available(iOS 17.0, watchOS 10.0, *) {
                WatchMedicationCircularView(entry: entry)
                    .containerBackground(.fill.tertiary, for: .widget)
            } else {
                WatchMedicationCircularView(entry: entry)
            }
        }
    }
}

struct WatchMedicationCircularView: View {
    let entry: WatchMedicationEntry

    var body: some View {
        ZStack {
            AccessoryWidgetBackground()

            if let nextMed = entry.nextMedication {
                VStack(spacing: 2) {
                    Image(systemName: "pills.fill")
                        .font(.system(size: 14))

                    Text(nextMed.timeString)
                        .font(.system(.caption2, design: .rounded).weight(.bold))
                }
            } else {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.green)
            }
        }
    }
}

struct WatchMedicationRectangularView: View {
    let entry: WatchMedicationEntry

    var body: some View {
        HStack {
            Image(systemName: "pills.fill")
                .font(.title3)
                .foregroundColor(.blue)

            if let nextMed = entry.nextMedication {
                VStack(alignment: .leading, spacing: 2) {
                    Text(nextMed.name)
                        .font(.headline)
                        .lineLimit(1)

                    HStack(spacing: 4) {
                        Text(nextMed.relativeTimeString)
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("•")
                            .foregroundColor(.secondary)

                        Text(nextMed.dosage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    Text("All done!")
                        .font(.headline)

                    Text("No meds due")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()
        }
    }
}
