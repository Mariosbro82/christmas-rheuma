//
//  InflamAIControlBundle.swift
//  InflamAIControls
//
//  Control widgets for iOS 18+ lock screen action buttons
//  These replace the camera/flashlight buttons on the lock screen
//

import WidgetKit
import SwiftUI

@available(iOS 18.0, *)
@main
struct InflamAIControlBundle: WidgetBundle {
    var body: some Widget {
        QuickLogControl()
        SOSFlareControl()
        MedicationControl()
        ExerciseControl()
    }
}
