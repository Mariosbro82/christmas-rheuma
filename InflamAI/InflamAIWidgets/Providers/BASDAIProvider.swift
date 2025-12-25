//
//  BASDAIProvider.swift
//  InflamAIWidgetExtension
//
//  Timeline provider for BASDAI score widgets
//

import WidgetKit
import SwiftUI

struct BASDAIEntry: TimelineEntry {
    let date: Date
    let data: WidgetBASDAIData

    init(date: Date, data: WidgetBASDAIData) {
        self.date = date
        self.data = data
    }
}

struct BASDAIProvider: TimelineProvider {
    typealias Entry = BASDAIEntry

    func placeholder(in context: Context) -> BASDAIEntry {
        BASDAIEntry(date: Date(), data: .placeholder)
    }

    func getSnapshot(in context: Context, completion: @escaping (BASDAIEntry) -> Void) {
        let data = WidgetDataProvider.shared.getBASDAIData()
        completion(BASDAIEntry(date: Date(), data: data))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<BASDAIEntry>) -> Void) {
        let data = WidgetDataProvider.shared.getBASDAIData()
        let entry = BASDAIEntry(date: Date(), data: data)

        // Update every 30 minutes (BASDAI doesn't change frequently)
        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 30, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}
