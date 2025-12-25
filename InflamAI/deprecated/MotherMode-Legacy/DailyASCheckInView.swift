//
//  DailyASCheckInView.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import SwiftUI

struct DailyASCheckInView: View {
    @ObservedObject var viewModel: DailyCheckInViewModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(String(localized: "disclaimer.general_info"))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(nil)
                }
                Section(header: Text(String(localized: "daily.section.symptom"))) {
                    sliderRow(label: LocalizedStringResource("daily.field.pain"), value: $viewModel.pain)
                    Stepper(value: $viewModel.stiffnessMinutes, in: 0...240, step: 5) {
                        Text(String(format: NSLocalizedString("daily.field.stiffness", comment: ""), viewModel.stiffnessMinutes))
                    }
                    sliderRow(label: LocalizedStringResource("daily.field.fatigue"), value: $viewModel.fatigue)
                    sliderRow(label: LocalizedStringResource("daily.field.sleep"), value: $viewModel.sleepQuality)
                }
                Section(header: Text(String(localized: "daily.section.mobility"))) {
                    Toggle(String(localized: "daily.field.mobility_toggle"), isOn: $viewModel.mobilityCompleted)
                        .toggleStyle(SwitchToggleStyle(tint: .accentColor))
                    Text(String(localized: "mother.micro.stop"))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Section(header: Text(String(localized: "daily.section.notes"))) {
                    TextEditor(text: $viewModel.notes)
                        .frame(minHeight: 120)
                        .accessibilityLabel("Notes")
                }
            }
            .navigationTitle(String(localized: "daily.title"))
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: { dismiss() })
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        viewModel.saveEntry()
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func sliderRow(label: LocalizedStringResource, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
            Slider(value: value, in: 0...10, step: 1)
                .accessibilityLabel(label)
        }
    }
}
