//
//  ExerciseControl.swift
//  InflamAIControlExtension
//
//  Exercise control button for lock screen
//  Opens the exercise routine screen
//

import WidgetKit
import SwiftUI
import AppIntents

@available(iOS 18.0, *)
struct ExerciseControl: ControlWidget {
    static let kind: String = "com.spinalytics.control.exercise"

    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: Self.kind) {
            ControlWidgetButton(action: ExerciseControlIntent()) {
                Label("Exercise", systemImage: "figure.walk")
            }
        }
        .displayName("Start Exercise")
        .description("Start your exercise routine")
    }
}

@available(iOS 18.0, *)
struct ExerciseControlIntent: ControlConfigurationIntent {
    static var title: LocalizedStringResource = "Start Exercise"
    static var description = IntentDescription("Start your exercise routine")
    static var isDiscoverable: Bool = true
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & OpensIntent {
        return .result(opensIntent: OpenURLIntent(URL(string: "spinalytics://widget/exercise")!))
    }
}
