//
//  MedicationProvider.swift
//  InflamAIWidgetExtension
//
//  Timeline provider for medication reminder widgets
//

import WidgetKit
import SwiftUI

struct MedicationEntry: TimelineEntry {
    let date: Date
    let data: WidgetMedicationData

    init(date: Date, data: WidgetMedicationData) {
        self.date = date
        self.data = data
    }

    var nextMedication: WidgetMedicationData.MedicationReminder? {
        data.medications.first { $0.nextDoseTime > Date() }
    }
}

struct MedicationProvider: TimelineProvider {
    typealias Entry = MedicationEntry

    func placeholder(in context: Context) -> MedicationEntry {
        MedicationEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (MedicationEntry) -> Void) {
        let data = WidgetDataProvider.shared.getMedicationData()
        completion(MedicationEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<MedicationEntry>) -> Void) {
        let data = WidgetDataProvider.shared.getMedicationData()

        // Create entries for each upcoming medication time
        var entries: [MedicationEntry] = []
        let now = Date()

        // Add current entry
        entries.append(MedicationEntry(date: now, data: data))

        // Add entries for when each medication is due (to update countdown)
        for med in data.medications where med.nextDoseTime > now {
            entries.append(MedicationEntry(date: med.nextDoseTime, data: data))
        }

        // Update in 5 minutes if no upcoming medications
        let nextUpdate = data.medications.first { $0.nextDoseTime > now }?.nextDoseTime
            ?? Calendar.current.date(byAdding: .minute, value: 5, to: now)!

        let timeline = Timeline(entries: entries, policy: .after(nextUpdate))
        completion(timeline)
    }
}
