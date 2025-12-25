//
//  InflamAIWidgetBundle.swift
//  InflamAIWidgets
//
//  Main widget bundle entry point - registers all InflamAI widgets
//

import WidgetKit
import SwiftUI

@main
struct InflamAIWidgetBundle: WidgetBundle {
    var body: some Widget {
        // Core health widgets
        FlareRiskWidget()
        BASDAIWidget()
        StreakWidget()
        MedicationWidget()

        // Dashboard widgets
        DailyDashboardWidget()
    }
}
