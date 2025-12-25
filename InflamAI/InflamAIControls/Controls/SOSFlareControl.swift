//
//  SOSFlareControl.swift
//  InflamAIControlExtension
//
//  SOS Flare control button for lock screen
//  Opens the emergency flare logging screen
//

import WidgetKit
import SwiftUI
import AppIntents

@available(iOS 18.0, *)
struct SOSFlareControl: ControlWidget {
    static let kind: String = "com.spinalytics.control.sosflare"

    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: Self.kind) {
            ControlWidgetButton(action: SOSFlareControlIntent()) {
                Label("SOS Flare", systemImage: "flame.fill")
            }
        }
        .displayName("SOS Flare")
        .description("Quickly log a flare event")
    }
}

@available(iOS 18.0, *)
struct SOSFlareControlIntent: ControlConfigurationIntent {
    static var title: LocalizedStringResource = "SOS Flare"
    static var description = IntentDescription("Quickly log a flare event")
    static var isDiscoverable: Bool = true
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & OpensIntent {
        return .result(opensIntent: OpenURLIntent(URL(string: "spinalytics://widget/sosflare")!))
    }
}
