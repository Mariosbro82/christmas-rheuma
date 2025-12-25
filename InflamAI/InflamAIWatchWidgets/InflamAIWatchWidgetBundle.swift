//
//  InflamAIWatchWidgetBundle.swift
//  InflamAIWatchWidgets
//
//  Watch widgets and complications for watchOS 10+
//  Displays in Smart Stack and as watch face complications
//

import WidgetKit
import SwiftUI

#if os(watchOS)
@main
struct InflamAIWatchWidgetBundle: WidgetBundle {
    var body: some Widget {
        WatchFlareRiskWidget()
        WatchBASDAIWidget()
        WatchMedicationWidget()
        WatchStreakWidget()
        WatchQuickStatsWidget()
    }
}
#endif
