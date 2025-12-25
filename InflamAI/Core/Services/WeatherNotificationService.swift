//
//  WeatherNotificationService.swift
//  InflamAI
//
//  Background weather monitoring and notification system
//  Alerts users about upcoming pressure drops that may trigger flares
//  Now uses Open-Meteo API (FREE, no API key required)
//

import Foundation
import UserNotifications
import CoreLocation

@MainActor
class WeatherNotificationService {
    static let shared = WeatherNotificationService()

    private let weatherService: OpenMeteoService
    private let notificationCenter = UNUserNotificationCenter.current()

    // User preferences
    private var notificationsEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "weatherNotificationsEnabled") }
        set { UserDefaults.standard.set(newValue, forKey: "weatherNotificationsEnabled") }
    }

    private var sensitivityThreshold: Double {
        get {
            let threshold = UserDefaults.standard.double(forKey: "weatherNotificationThreshold")
            return threshold > 0 ? threshold : 10.0 // Default 10 mmHg
        }
        set { UserDefaults.standard.set(newValue, forKey: "weatherNotificationThreshold") }
    }

    private var advanceWarningHours: Int {
        get {
            let hours = UserDefaults.standard.integer(forKey: "weatherNotificationAdvanceHours")
            return hours > 0 ? hours : 6 // Default 6 hours
        }
        set { UserDefaults.standard.set(newValue, forKey: "weatherNotificationAdvanceHours") }
    }

    private init() {
        self.weatherService = OpenMeteoService.shared
    }

    // MARK: - Public Methods

    func requestPermissions() async throws {
        let granted = try await notificationCenter.requestAuthorization(options: [.alert, .sound, .badge])
        guard granted else {
            throw NotificationError.permissionDenied
        }

        // Enable notifications after permission granted
        notificationsEnabled = true
    }

    /// Schedule weather monitoring (call this once per day or when app launches)
    func scheduleWeatherMonitoring() async {
        guard notificationsEnabled else {
            print("ℹ️ Weather notifications disabled")
            return
        }

        do {
            // Fetch 48-hour forecast using WeatherKit
            try await weatherService.fetchAllWeatherData()

            // Get pressure forecast data
            let pressureData = weatherService.getPressureForecast()

            // Find significant pressure drops
            let drops = findSignificantPressureDrops(in: pressureData)

            // Clear any pending weather notifications
            await cancelAllWeatherNotifications()

            // Schedule notifications for each significant drop
            for drop in drops {
                let hoursUntil = Calendar.current.dateComponents([.hour], from: Date(), to: drop.startTime).hour ?? 0

                // Only notify if within advance warning window
                if hoursUntil >= 0 && hoursUntil <= advanceWarningHours {
                    await scheduleNotification(for: drop, hoursUntil: hoursUntil)
                }
            }

            print("✅ Weather monitoring scheduled. Found \(drops.count) pressure drops in next 48h")

        } catch {
            print("❌ Weather monitoring error: \(error)")
        }
    }

    /// Cancel all weather notifications
    func cancelAllWeatherNotifications() async {
        let pendingRequests = await notificationCenter.pendingNotificationRequests()
        let weatherNotificationIDs = pendingRequests
            .filter { $0.identifier.starts(with: "weather-alert-") }
            .map { $0.identifier }

        notificationCenter.removePendingNotificationRequests(withIdentifiers: weatherNotificationIDs)
        print("🗑️ Cancelled \(weatherNotificationIDs.count) weather notifications")
    }

    /// Send a test notification
    func sendTestNotification() async {
        let content = UNMutableNotificationContent()
        content.title = "Weather Flare Alert (Test)"
        content.body = "This is a test notification. You'll receive alerts like this when pressure drops significantly."
        content.sound = .default
        content.categoryIdentifier = "WEATHER_FLARE_ALERT"

        let request = UNNotificationRequest(
            identifier: "test-weather-alert-\(Date().timeIntervalSince1970)",
            content: content,
            trigger: nil // Immediate
        )

        do {
            try await notificationCenter.add(request)
            print("📱 Test notification sent")
        } catch {
            print("❌ Failed to send test notification: \(error)")
        }
    }

    // MARK: - Settings

    func updateSettings(enabled: Bool, threshold: Double, advanceHours: Int) {
        notificationsEnabled = enabled
        sensitivityThreshold = threshold
        advanceWarningHours = advanceHours

        if enabled {
            Task {
                await scheduleWeatherMonitoring()
            }
        } else {
            Task {
                await cancelAllWeatherNotifications()
            }
        }
    }

    func getSettings() -> (enabled: Bool, threshold: Double, advanceHours: Int) {
        return (notificationsEnabled, sensitivityThreshold, advanceWarningHours)
    }

    // MARK: - Private Methods

    private func findSignificantPressureDrops(in data: [PressureDataPoint]) -> [PressureDrop] {
        var drops: [PressureDrop] = []
        let threshold = sensitivityThreshold

        guard data.count > 12 else { return drops }

        // Look for 12-hour drops
        for i in 0..<(data.count - 12) {
            let startPoint = data[i]
            let endPoint = data[i + 12]
            let drop = endPoint.pressure - startPoint.pressure

            if drop < -threshold {
                drops.append(PressureDrop(
                    startTime: startPoint.timestamp,
                    endTime: endPoint.timestamp,
                    magnitude: abs(drop)
                ))
            }
        }

        return drops
    }

    private func scheduleNotification(for drop: PressureDrop, hoursUntil: Int) async {
        let content = UNMutableNotificationContent()
        content.title = "Weather Flare Alert"
        content.body = String(format: "Pressure will drop %.1f mmHg in the next 12 hours. This may trigger symptoms.", drop.magnitude)
        content.sound = .default
        content.categoryIdentifier = "WEATHER_FLARE_ALERT"
        content.userInfo = [
            "type": "weather_alert",
            "pressureDrop": drop.magnitude,
            "startTime": drop.startTime.timeIntervalSince1970
        ]

        // Schedule notification for 2 hours before pressure starts dropping
        // Or immediately if drop is imminent
        let notificationTime: Date
        if hoursUntil > 2 {
            notificationTime = Calendar.current.date(byAdding: .hour, value: -2, to: drop.startTime) ?? drop.startTime
        } else {
            notificationTime = Date().addingTimeInterval(60) // 1 minute from now
        }

        let dateComponents = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: notificationTime)
        let trigger = UNCalendarNotificationTrigger(dateMatching: dateComponents, repeats: false)

        let request = UNNotificationRequest(
            identifier: "weather-alert-\(drop.startTime.timeIntervalSince1970)",
            content: content,
            trigger: trigger
        )

        do {
            try await notificationCenter.add(request)
            print("📱 Scheduled weather alert for \(notificationTime.formatted())")
        } catch {
            print("❌ Failed to schedule notification: \(error)")
        }
    }
}

// MARK: - Models

struct PressureDrop {
    let startTime: Date
    let endTime: Date
    let magnitude: Double
}

enum NotificationError: Error {
    case permissionDenied
}
