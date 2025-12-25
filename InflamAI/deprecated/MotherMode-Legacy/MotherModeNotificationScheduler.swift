//
//  MotherModeNotificationScheduler.swift
//  InflamAI-Swift
//
//  Created by Codex on 2024-05-29.
//

import Foundation
import UserNotifications

final class MotherModeNotificationScheduler {
    private let center = UNUserNotificationCenter.current()
    private let settingsProvider: () -> MotherModeSettings
    
    init(settingsProvider: @escaping () -> MotherModeSettings) {
        self.settingsProvider = settingsProvider
        registerNotificationCategoryIfNeeded()
    }
    
    func scheduleGentleNudge(identifier: String, body: String, at date: Date) {
        let settings = settingsProvider()
        guard shouldSendNotification(on: date, settings: settings) else {
            return
        }
        
        var components = Calendar.current.dateComponents([.hour, .minute], from: date)
        components.second = 0
        
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)
        
        let content = UNMutableNotificationContent()
        content.title = "Bechterew Flow"
        content.body = body
        content.categoryIdentifier = "MOTHER_MODE_GENTLE"
        
        let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)
        center.add(request)
    }
    
    private func shouldSendNotification(on date: Date, settings: MotherModeSettings) -> Bool {
        let components = Calendar.current.dateComponents([.hour, .minute], from: date)
        let windows = settings.napWindows + settings.feedingWindows
        return !windows.contains { $0.contains(components) }
    }
    
    private func registerNotificationCategoryIfNeeded() {
        let snoozeAction = UNNotificationAction(
            identifier: "MOTHER_SNOOZE_90",
            title: NSLocalizedString("mother.notifications.snooze_90", comment: ""),
            options: [])
        let doneAction = UNNotificationAction(
            identifier: "MOTHER_MARK_DONE",
            title: NSLocalizedString("Done", comment: ""),
            options: [.foreground])
        
        let category = UNNotificationCategory(
            identifier: "MOTHER_MODE_GENTLE",
            actions: [snoozeAction, doneAction],
            intentIdentifiers: [],
            options: [])
        UNUserNotificationCenter.current().setNotificationCategories([category])
    }
}
