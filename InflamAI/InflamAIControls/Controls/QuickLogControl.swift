//
//  QuickLogControl.swift
//  InflamAIControlExtension
//
//  Quick Log control button for lock screen
//  Opens the quick symptom logging screen
//

import WidgetKit
import SwiftUI
import AppIntents

@available(iOS 18.0, *)
struct QuickLogControl: ControlWidget {
    static let kind: String = "com.spinalytics.control.quicklog"

    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: Self.kind) {
            ControlWidgetButton(action: QuickLogControlIntent()) {
                Label("Quick Log", systemImage: "pencil.circle.fill")
            }
        }
        .displayName("Quick Log")
        .description("Quickly log your symptoms")
    }
}

@available(iOS 18.0, *)
struct QuickLogControlIntent: ControlConfigurationIntent {
    static var title: LocalizedStringResource = "Quick Log"
    static var description = IntentDescription("Open the quick symptom logging screen")
    static var isDiscoverable: Bool = true
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & OpensIntent {
        // Opens the app with deep link to quick log
        return .result(opensIntent: OpenURLIntent(URL(string: "spinalytics://widget/quicklog")!))
    }
}
