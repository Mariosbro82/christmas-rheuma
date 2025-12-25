//
//  StreakProvider.swift
//  InflamAIWidgetExtension
//
//  Timeline provider for logging streak widgets
//

import WidgetKit
import SwiftUI

struct StreakEntry: TimelineEntry {
    let date: Date
    let data: WidgetStreakData

    init(date: Date, data: WidgetStreakData) {
        self.date = date
        self.data = data
    }
}

struct StreakProvider: TimelineProvider {
    typealias Entry = StreakEntry

    func placeholder(in context: Context) -> StreakEntry {
        StreakEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (StreakEntry) -> Void) {
        let data = WidgetDataProvider.shared.getStreakData()
        completion(StreakEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<StreakEntry>) -> Void) {
        let data = WidgetDataProvider.shared.getStreakData()
        let entry = StreakEntry(date: Date(), data: data)

        // Update at midnight (when streak might change)
        let tomorrow = Calendar.current.startOfDay(for: Date().addingTimeInterval(86400))
        let timeline = Timeline(entries: [entry], policy: .after(tomorrow))
        completion(timeline)
    }
}
