//
//  FlareRiskProvider.swift
//  InflamAIWidgetExtension
//
//  Timeline provider for Flare Risk widgets
//

import WidgetKit
import SwiftUI

struct FlareRiskEntry: TimelineEntry {
    let date: Date
    let data: WidgetFlareData
    let configuration: ConfigurationAppIntent?

    init(date: Date, data: WidgetFlareData, configuration: ConfigurationAppIntent? = nil) {
        self.date = date
        self.data = data
        self.configuration = configuration
    }
}

struct FlareRiskProvider: AppIntentTimelineProvider {
    typealias Entry = FlareRiskEntry
    typealias Intent = ConfigurationAppIntent

    func placeholder(in context: Context) -> FlareRiskEntry {
        FlareRiskEntry(date: Date(), data: .placeholder)
    }

    func snapshot(for configuration: ConfigurationAppIntent, in context: Context) async -> FlareRiskEntry {
        let data = WidgetDataProvider.shared.getFlareRiskData()
        return FlareRiskEntry(date: Date(), data: data, configuration: configuration)
    }

    func timeline(for configuration: ConfigurationAppIntent, in context: Context) async -> Timeline<FlareRiskEntry> {
        let data = WidgetDataProvider.shared.getFlareRiskData()
        let entry = FlareRiskEntry(date: Date(), data: data, configuration: configuration)

        // Update every 15 minutes
        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        return Timeline(entries: [entry], policy: .after(nextUpdate))
    }
}

// MARK: - Simple Provider (no configuration)

struct SimpleFlareRiskProvider: TimelineProvider {
    typealias Entry = FlareRiskEntry

    func placeholder(in context: Context) -> FlareRiskEntry {
        FlareRiskEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (FlareRiskEntry) -> Void) {
        let data = WidgetDataProvider.shared.getFlareRiskData()
        completion(FlareRiskEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<FlareRiskEntry>) -> Void) {
        let data = WidgetDataProvider.shared.getFlareRiskData()
        let entry = FlareRiskEntry(date: Date(), data: data)

        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}
