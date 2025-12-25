//
//  MotherModeSettingsView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct MotherModeSettingsView: View {
    @ObservedObject var viewModel: MotherModeViewModel
    
    var body: some View {
        Form {
            Section(String(localized: "mother.toggle.title")) {
                Toggle(String(localized: "mother.toggle.title"), isOn: Binding(
                    get: { viewModel.settings.isEnabled },
                    set: { viewModel.toggleMotherMode($0) }
                ))
                Toggle(String(localized: "mother.settings.pregnant"), isOn: Binding(
                    get: { viewModel.settings.isPregnantOrPostpartum },
                    set: { viewModel.updatePregnancyStatus($0) }
                ))
                Text(String(localized: "mother.pregnancy.caution"))
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
            
            Section(String(localized: "mother.settings.onehand")) {
                Toggle(String(localized: "mother.settings.onehand"), isOn: Binding(
                    get: { viewModel.settings.oneHandMode },
                    set: { viewModel.toggleOneHandMode($0) }
                ))
            }
            
            quietWindowSection(
                title: "mother.settings.nap_window",
                windows: viewModel.settings.napWindows,
                addAction: viewModel.addNapWindow,
                deleteAction: viewModel.removeNapWindow)
            
            quietWindowSection(
                title: "mother.settings.feeding_window",
                windows: viewModel.settings.feedingWindows,
                addAction: viewModel.addFeedingWindow,
                deleteAction: viewModel.removeFeedingWindow)
            
            Section(String(localized: "mother.settings.voice_consent")) {
                Toggle(String(localized: "mother.settings.voice_consent"), isOn: Binding(
                    get: { viewModel.settings.voiceNotesAllowed },
                    set: { viewModel.toggleVoiceNotes($0) }
                ))
            }
        }
        .navigationTitle(String(localized: "mother.toggle.subtitle"))
    }
    
    private func quietWindowSection(
        title: LocalizedStringKey,
        windows: [MotherQuietWindow],
        addAction: @escaping () -> Void,
        deleteAction: @escaping (IndexSet) -> Void
    ) -> some View {
        Section(title) {
            ForEach(windows) { window in
                HStack {
                    Text(window.start.formattedHourMinute + " – " + window.end.formattedHourMinute)
                    Spacer()
                }
                .accessibilityLabel(String(format: NSLocalizedString("mother.settings.window_accessibility", comment: ""), window.start.formattedHourMinute, window.end.formattedHourMinute))
            }
            .onDelete(perform: deleteAction)
            
            Button(String(localized: "mother.settings.add_window"), action: addAction)
        }
    }
}

private extension DateComponents {
    var formattedHourMinute: String {
        guard let hour = hour, let minute = minute else { return "--:--" }
        return String(format: "%02d:%02d", hour, minute)
    }
}
