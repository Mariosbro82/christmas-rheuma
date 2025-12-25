//
//  ClinicianReportPreview.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct ClinicianReportPreview: View {
    @ObservedObject var environment: TraeAppEnvironment
    private let composer = ClinicianReportComposer()
    
    var body: some View {
        List {
            Section(header: Text(String(localized: "clinician.section.summary"))) {
                if environment.symptomEntries.isEmpty {
                    Text(String(localized: "clinician.no_entries"))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else {
                    Text(String(format: NSLocalizedString("clinician.entries_count", comment: ""), environment.symptomEntries.count))
                    Text(String(format: NSLocalizedString("clinician.average_pain", comment: ""), environment.symptomEntries.averagePainFormatted()))
                    Text(String(format: NSLocalizedString("clinician.average_stiffness", comment: ""), environment.symptomEntries.averageStiffnessFormatted()))
                }
            }
            Section(header: Text(String(localized: "clinician.section.export"))) {
                Button(String(localized: "clinician.button.generate")) {
                    exportReport()
                }
            }
            Section {
                Text(String(localized: "disclaimer.general_info"))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Clinician report")
    }
    
    private func exportReport() {
        let entries = environment.symptomEntries
        let completed = entries.filter { $0.mobilityCompleted }.count
        let rate = entries.isEmpty ? 0 : Int(Double(completed) / Double(entries.count) * 100.0)
        let medicationSummaries = environment.notifications.map { note in
            String(format: NSLocalizedString("clinician.medication.entry", comment: ""), note.title, note.deliveryDate.formatted(date: .abbreviated, time: .shortened))
        }
        let data = composer.makePDF(entries: entries, mobilityCompletionRate: rate, medicationSummaries: medicationSummaries)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ClinicianReport.pdf")
        try? data.write(to: url)
        let controller = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        UIApplication.presentTopViewController(controller)
    }
}

private extension Array where Element == SymptomEntry {
    func averagePainFormatted() -> String {
        guard !isEmpty else { return "--" }
        let value = reduce(0) { $0 + $1.pain } / Double(count)
        return String(format: "%.1f", value)
    }
    
    func averageStiffnessFormatted() -> Int {
        guard !isEmpty else { return 0 }
        let value = Double(reduce(0) { $0 + $1.stiffnessMinutes }) / Double(count)
        return Int(value.rounded())
    }
}
