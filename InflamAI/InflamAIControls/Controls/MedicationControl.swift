//
//  MedicationControl.swift
//  InflamAIControlExtension
//
//  Medication control button for lock screen
//  Opens the medication logging screen
//

import WidgetKit
import SwiftUI
import AppIntents

@available(iOS 18.0, *)
struct MedicationControl: ControlWidget {
    static let kind: String = "com.spinalytics.control.medication"

    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: Self.kind) {
            ControlWidgetButton(action: MedicationControlIntent()) {
                Label("Log Meds", systemImage: "pills.fill")
            }
        }
        .displayName("Log Medication")
        .description("Log your medication intake")
    }
}

@available(iOS 18.0, *)
struct MedicationControlIntent: ControlConfigurationIntent {
    static var title: LocalizedStringResource = "Log Medication"
    static var description = IntentDescription("Log your medication intake")
    static var isDiscoverable: Bool = true
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & OpensIntent {
        return .result(opensIntent: OpenURLIntent(URL(string: "spinalytics://widget/medication")!))
    }
}
