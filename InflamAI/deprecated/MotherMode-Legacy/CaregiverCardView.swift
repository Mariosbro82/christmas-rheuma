//
//  CaregiverCardView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct CaregiverCardView: View {
    @State private var whatHelps: String = ""
    @State private var avoid: String = ""
    @State private var medsHandled: String = ""
    @State private var emergencyInfo: String = ""
    @State private var contacts: String = ""
    
    private let composer = CaregiverCardComposer()
    
    var body: some View {
        Form {
            Section(String(localized: "mother.caregiver.what_helps")) {
                TextEditor(text: $whatHelps).frame(minHeight: 80)
            }
            Section(String(localized: "mother.caregiver.avoid")) {
                TextEditor(text: $avoid).frame(minHeight: 80)
            }
            Section(String(localized: "mother.caregiver.medications")) {
                TextEditor(text: $medsHandled).frame(minHeight: 60)
            }
            Section(String(localized: "mother.caregiver.emergency")) {
                TextEditor(text: $emergencyInfo).frame(minHeight: 60)
            }
            Section(String(localized: "mother.caregiver.contacts")) {
                TextEditor(text: $contacts).frame(minHeight: 60)
            }
            Section {
                Button(String(localized: "mother.caregiver.share"), action: share)
                    .buttonStyle(.borderedProminent)
            }
        }
        .navigationTitle(String(localized: "mother.caregiver.title"))
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text(String(localized: "mother.caregiver.subtitle"))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
    
    private func share() {
        let data = CaregiverCardData(
            whatHelps: whatHelps,
            avoid: avoid,
            medsHandled: medsHandled,
            emergencyInfo: emergencyInfo,
            contacts: contacts
        )
        let pdf = composer.makePDF(data: data)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("CaregiverCard.pdf")
        do {
            try pdf.write(to: url)
        } catch {
            return
        }
        let controller = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        UIApplication.presentTopViewController(controller)
    }
}
