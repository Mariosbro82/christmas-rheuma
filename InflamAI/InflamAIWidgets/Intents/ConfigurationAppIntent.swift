//
//  ConfigurationAppIntent.swift
//  InflamAIWidgetExtension
//
//  App intents for widget configuration
//

import AppIntents
import WidgetKit

struct ConfigurationAppIntent: WidgetConfigurationIntent {
    static var title: LocalizedStringResource = "Configure Widget"
    static var description = IntentDescription("Customize your InflamAI widget")

    // Placeholder for future configuration options
    // Examples could include:
    // - Show/hide trend indicators
    // - Color theme selection
    // - Data source selection
}

// MARK: - Deep Link Intents

struct OpenQuickLogIntent: AppIntent {
    static var title: LocalizedStringResource = "Open Quick Log"
    static var description = IntentDescription("Open the quick symptom logging screen")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        // The app will handle the deep link
        return .result()
    }
}

struct OpenSOSFlareIntent: AppIntent {
    static var title: LocalizedStringResource = "SOS Flare"
    static var description = IntentDescription("Quickly log a flare event")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        return .result()
    }
}

struct OpenMedicationIntent: AppIntent {
    static var title: LocalizedStringResource = "Log Medication"
    static var description = IntentDescription("Log medication intake")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        return .result()
    }
}

struct OpenExerciseIntent: AppIntent {
    static var title: LocalizedStringResource = "Start Exercise"
    static var description = IntentDescription("Start an exercise routine")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        return .result()
    }
}
