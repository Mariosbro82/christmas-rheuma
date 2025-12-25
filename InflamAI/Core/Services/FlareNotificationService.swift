//
//  FlareNotificationService.swift
//  InflamAI
//
//  Notification service for ML-based flare predictions
//  Sends alerts when flare risk is elevated
//

import Foundation
import UserNotifications
import CoreData

// MARK: - Navigation Event

/// Published when user taps a notification action
enum FlareNotificationEvent {
    case logSymptoms
    case checkIn
    case quickFlareLog(riskLevel: String, probability: Double)
}

@MainActor
class FlareNotificationService {
    static let shared = FlareNotificationService()

    private let notificationCenter = UNUserNotificationCenter.current()

    /// Publisher for notification action events
    @Published var lastEvent: FlareNotificationEvent?

    // MARK: - User Preferences

    var notificationsEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "flareNotificationsEnabled") }
        set { UserDefaults.standard.set(newValue, forKey: "flareNotificationsEnabled") }
    }

    var riskThreshold: Double {
        get {
            let threshold = UserDefaults.standard.double(forKey: "flareNotificationThreshold")
            return threshold > 0 ? threshold : 0.6 // Default 60% risk
        }
        set { UserDefaults.standard.set(newValue, forKey: "flareNotificationThreshold") }
    }

    private var lastNotificationDate: Date? {
        get { UserDefaults.standard.object(forKey: "flareLastNotificationDate") as? Date }
        set { UserDefaults.standard.set(newValue, forKey: "flareLastNotificationDate") }
    }

    // Minimum hours between notifications to avoid spam
    private let minimumNotificationInterval: TimeInterval = 12 * 60 * 60 // 12 hours

    private init() {}

    // MARK: - Public Methods

    func requestPermissions() async throws {
        let granted = try await notificationCenter.requestAuthorization(options: [.alert, .sound, .badge])
        guard granted else {
            throw FlareNotificationError.permissionDenied
        }
        notificationsEnabled = true
    }

    /// Check prediction and send notification if risk is high
    func checkAndNotify() async {
        guard notificationsEnabled else {
            print("ℹ️ [FlareNotification] Notifications disabled")
            return
        }

        // Check if we've notified recently
        if let lastDate = lastNotificationDate,
           Date().timeIntervalSince(lastDate) < minimumNotificationInterval {
            print("ℹ️ [FlareNotification] Skipping - notified recently")
            return
        }

        // Get current prediction from Neural Engine
        let prediction = UnifiedNeuralEngine.shared.currentPrediction

        guard let prediction = prediction else {
            print("ℹ️ [FlareNotification] No prediction available")
            return
        }

        // Check if risk exceeds threshold
        guard Double(prediction.probability) >= riskThreshold else {
            print("ℹ️ [FlareNotification] Risk below threshold (\(Int(prediction.probability * 100))% < \(Int(riskThreshold * 100))%)")
            return
        }

        // Send notification
        await sendFlareWarningNotification(prediction: prediction)
    }

    /// Schedule daily check (call from app launch)
    func scheduleDailyCheck() {
        // Remove any existing scheduled checks
        notificationCenter.removePendingNotificationRequests(withIdentifiers: ["flare_daily_check"])

        guard notificationsEnabled else { return }

        // Schedule for 8 AM daily
        var dateComponents = DateComponents()
        dateComponents.hour = 8
        dateComponents.minute = 0

        let trigger = UNCalendarNotificationTrigger(dateMatching: dateComponents, repeats: true)

        let content = UNMutableNotificationContent()
        content.title = "Daily Check-In Reminder"
        content.body = "Log your symptoms to keep your flare predictions accurate."
        content.sound = .default
        content.categoryIdentifier = "FLARE_REMINDER"

        let request = UNNotificationRequest(
            identifier: "flare_daily_check",
            content: content,
            trigger: trigger
        )

        notificationCenter.add(request) { error in
            if let error = error {
                print("❌ [FlareNotification] Failed to schedule daily check: \(error)")
            } else {
                print("✅ [FlareNotification] Daily check scheduled for 8 AM")
            }
        }
    }

    // MARK: - Private Methods

    private func sendFlareWarningNotification(prediction: NeuralPrediction) async {
        let content = UNMutableNotificationContent()

        let riskPercent = Int(prediction.probability * 100)

        content.title = "⚠️ Elevated Flare Risk"
        content.body = "Your flare risk is \(riskPercent)%. \(prediction.recommendedAction.rawValue)"
        content.sound = .default
        content.categoryIdentifier = "FLARE_WARNING"

        // Add action buttons
        content.userInfo = [
            "riskLevel": prediction.riskLevel.rawValue,
            "probability": prediction.probability,
            "timestamp": prediction.timestamp.timeIntervalSince1970
        ]

        let request = UNNotificationRequest(
            identifier: "flare_warning_\(UUID().uuidString)",
            content: content,
            trigger: nil // Send immediately
        )

        do {
            try await notificationCenter.add(request)
            lastNotificationDate = Date()
            print("✅ [FlareNotification] Sent warning notification (risk: \(riskPercent)%)")
        } catch {
            print("❌ [FlareNotification] Failed to send: \(error)")
        }
    }

    /// Register notification categories and actions
    func registerNotificationCategories() {
        // Flare warning actions
        let logSymptomsAction = UNNotificationAction(
            identifier: "LOG_SYMPTOMS",
            title: "Log Symptoms",
            options: .foreground
        )

        let dismissAction = UNNotificationAction(
            identifier: "DISMISS",
            title: "Dismiss",
            options: .destructive
        )

        let flareWarningCategory = UNNotificationCategory(
            identifier: "FLARE_WARNING",
            actions: [logSymptomsAction, dismissAction],
            intentIdentifiers: [],
            options: .customDismissAction
        )

        // Daily reminder actions
        let checkInAction = UNNotificationAction(
            identifier: "CHECK_IN",
            title: "Check In Now",
            options: .foreground
        )

        let reminderCategory = UNNotificationCategory(
            identifier: "FLARE_REMINDER",
            actions: [checkInAction, dismissAction],
            intentIdentifiers: [],
            options: .customDismissAction
        )

        notificationCenter.setNotificationCategories([flareWarningCategory, reminderCategory])
    }

    // MARK: - Error Types

    enum FlareNotificationError: LocalizedError {
        case permissionDenied

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Notification permission was denied. Enable in Settings."
            }
        }
    }

    // MARK: - Notification Response Handler

    /// Handle notification action responses
    func handleNotificationResponse(_ response: UNNotificationResponse) {
        let userInfo = response.notification.request.content.userInfo
        let actionIdentifier = response.actionIdentifier

        switch actionIdentifier {
        case "LOG_SYMPTOMS", UNNotificationDefaultActionIdentifier:
            // User tapped "Log Symptoms" or tapped the notification itself
            if let riskLevel = userInfo["riskLevel"] as? String,
               let probability = userInfo["probability"] as? Double {
                // Quick flare log with prediction context
                lastEvent = .quickFlareLog(riskLevel: riskLevel, probability: probability)
                logQuickFlare(riskLevel: riskLevel, probability: probability)
            } else {
                lastEvent = .logSymptoms
            }
            print("✅ [FlareNotification] User tapped Log Symptoms action")

        case "CHECK_IN":
            lastEvent = .checkIn
            print("✅ [FlareNotification] User tapped Check In action")

        case "DISMISS":
            print("ℹ️ [FlareNotification] User dismissed notification")

        default:
            print("ℹ️ [FlareNotification] Unknown action: \(actionIdentifier)")
        }
    }

    // MARK: - Quick Flare Logging

    /// Log a quick flare event when user responds to notification
    private func logQuickFlare(riskLevel: String, probability: Double) {
        let context = InflamAIPersistenceController.shared.container.viewContext

        // Create a new FlareEvent
        let flareEvent = FlareEvent(context: context)
        flareEvent.id = UUID()
        flareEvent.startDate = Date()
        flareEvent.severity = mapRiskToSeverity(riskLevel)
        flareEvent.notes = "Quick logged from AI prediction alert (Risk: \(Int(probability * 100))%)"

        // Encode triggers as JSON Data (Core Data stores as Binary Data)
        if let triggersArray = predictedTriggersArray() {
            flareEvent.suspectedTriggers = try? JSONEncoder().encode(triggersArray)
        }

        // Save to Core Data
        do {
            try context.save()
            print("✅ [FlareNotification] Quick flare logged successfully")

            // Also record to ML feedback loop for prediction accuracy tracking
            recordFlareToMLFeedback(severity: flareEvent.severity)
        } catch {
            print("❌ [FlareNotification] Failed to save flare: \(error)")
        }
    }

    private func recordFlareToMLFeedback(severity: Int16) {
        // Update the Neural Engine with flare feedback
        // This helps improve prediction accuracy over time
        UnifiedNeuralEngine.shared.incrementDaysOfUserData()
        print("📊 [FlareNotification] Recorded flare to ML feedback (severity: \(severity))")
    }

    private func mapRiskToSeverity(_ riskLevel: String) -> Int16 {
        switch riskLevel {
        case "Very High": return 4
        case "High": return 3
        case "Moderate": return 2
        case "Low": return 1
        default: return 2
        }
    }

    private func predictedTriggersArray() -> [String]? {
        // Get contributing factors from current prediction
        guard let prediction = UnifiedNeuralEngine.shared.currentPrediction else {
            return nil
        }
        // Map ContributingFactor objects to their string names
        return prediction.topFactors.map { $0.name }
    }
}

// MARK: - Notification Delegate

/// Notification center delegate to handle responses
class FlareNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    static let shared = FlareNotificationDelegate()

    // Handle notification when app is in foreground
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        // Show notification even when app is in foreground
        completionHandler([.banner, .sound, .badge])
    }

    // Handle notification action response
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        Task { @MainActor in
            FlareNotificationService.shared.handleNotificationResponse(response)
        }
        completionHandler()
    }
}
