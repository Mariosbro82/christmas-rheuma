//
//  MedicationWidget.swift
//  InflamAIWidgetExtension
//
//  Medication widget - displays upcoming medication reminders
//

import WidgetKit
import SwiftUI

struct MedicationWidget: Widget {
    let kind: String = "MedicationWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: MedicationProvider()) { entry in
            MedicationWidgetEntryView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Medications")
        .description("View your upcoming medication schedule")
        .supportedFamilies([
            .systemSmall,
            .systemMedium,
            .accessoryCircular,
            .accessoryRectangular,
            .accessoryInline
        ])
    }
}

struct MedicationWidgetEntryView: View {
    @Environment(\.widgetFamily) var family
    let entry: MedicationEntry

    var body: some View {
        switch family {
        case .systemSmall:
            MedicationSmallView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/medication"))

        case .systemMedium:
            MedicationMediumView(entry: entry)
                .widgetURL(URL(string: "spinalytics://widget/medication"))

        case .accessoryCircular:
            MedicationCircularView(entry: entry)

        case .accessoryRectangular:
            MedicationRectangularView(entry: entry)

        case .accessoryInline:
            MedicationInlineView(entry: entry)

        default:
            MedicationSmallView(entry: entry)
        }
    }
}

#Preview("Medication Small", as: .systemSmall) {
    MedicationWidget()
} timeline: {
    MedicationEntry(date: Date(), data: .placeholder)
}

#Preview("Medication Medium", as: .systemMedium) {
    MedicationWidget()
} timeline: {
    MedicationEntry(date: Date(), data: WidgetMedicationData(
        medications: [
            WidgetMedicationData.MedicationReminder(name: "Humira", dosage: "40mg", nextDoseTime: Date().addingTimeInterval(1800)),
            WidgetMedicationData.MedicationReminder(name: "Naproxen", dosage: "500mg", nextDoseTime: Date().addingTimeInterval(3600)),
            WidgetMedicationData.MedicationReminder(name: "Folic Acid", dosage: "5mg", nextDoseTime: Date().addingTimeInterval(7200))
        ],
        lastUpdated: Date()
    ))
}

#Preview("Medication Empty", as: .systemSmall) {
    MedicationWidget()
} timeline: {
    MedicationEntry(date: Date(), data: WidgetMedicationData(medications: [], lastUpdated: Date()))
}
