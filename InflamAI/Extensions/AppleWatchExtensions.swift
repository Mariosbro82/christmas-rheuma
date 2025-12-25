//
//  AppleWatchExtensions.swift
//  InflamAI-Swift
//
//  Created by AI Assistant on 2024
//

import SwiftUI
import Foundation

// MARK: - WatchHealthData Extensions

extension WatchHealthData {
    var displayTitle: String {
        switch dataType {
        case .heartRate:
            return "Heart Rate"
        case .steps:
            return "Steps"
        case .activeCalories:
            return "Active Calories"
        case .restingHeartRate:
            return "Resting Heart Rate"
        case .heartRateVariability:
            return "HRV"
        case .oxygenSaturation:
            return "Blood Oxygen"
        case .bodyTemperature:
            return "Body Temperature"
        case .bloodPressure:
            return "Blood Pressure"
        }
    }
    
    var displayValue: String {
        switch dataType {
        case .heartRate, .restingHeartRate:
            return "\(Int(value)) BPM"
        case .steps:
            return NumberFormatter.localizedString(from: NSNumber(value: value), number: .decimal)
        case .activeCalories:
            return "\(Int(value)) cal"
        case .heartRateVariability:
            return String(format: "%.1f ms", value)
        case .oxygenSaturation:
            return String(format: "%.1f%%", value)
        case .bodyTemperature:
            return String(format: "%.1f°F", value)
        case .bloodPressure:
            return "\(Int(value))/\(Int(metadata?["diastolic"] as? Double ?? 0))"
        }
    }
    
    var systemImage: String {
        switch dataType {
        case .heartRate, .restingHeartRate:
            return "heart.fill"
        case .steps:
            return "figure.walk"
        case .activeCalories:
            return "flame.fill"
        case .heartRateVariability:
            return "waveform.path.ecg"
        case .oxygenSaturation:
            return "lungs.fill"
        case .bodyTemperature:
            return "thermometer"
        case .bloodPressure:
            return "heart.circle"
        }
    }
    
    var color: Color {
        switch dataType {
        case .heartRate, .restingHeartRate:
            return .red
        case .steps:
            return .green
        case .activeCalories:
            return .orange
        case .heartRateVariability:
            return .purple
        case .oxygenSaturation:
            return .blue
        case .bodyTemperature:
            return .yellow
        case .bloodPressure:
            return .pink
        }
    }
    
    var isNormalRange: Bool {
        switch dataType {
        case .heartRate:
            return value >= 60 && value <= 100
        case .restingHeartRate:
            return value >= 50 && value <= 90
        case .oxygenSaturation:
            return value >= 95
        case .bodyTemperature:
            return value >= 97.0 && value <= 99.5
        case .bloodPressure:
            let systolic = value
            let diastolic = metadata?["diastolic"] as? Double ?? 0
            return systolic < 140 && diastolic < 90
        default:
            return true // No specific range for steps, calories, HRV
        }
    }
}

// MARK: - WatchSymptomEntry Extensions

extension WatchSymptomEntry {
    var severityColor: Color {
        switch severity {
        case 1...2:
            return .green
        case 3...4:
            return .yellow
        case 5...6:
            return .orange
        case 7...8:
            return .red
        case 9...10:
            return .purple
        default:
            return .gray
        }
    }
    
    var severityDescription: String {
        switch severity {
        case 1...2:
            return "Mild"
        case 3...4:
            return "Mild-Moderate"
        case 5...6:
            return "Moderate"
        case 7...8:
            return "Severe"
        case 9...10:
            return "Very Severe"
        default:
            return "Unknown"
        }
    }
    
    var displayText: String {
        if let notes = notes, !notes.isEmpty {
            return "\(symptomType): \(notes)"
        } else {
            return symptomType
        }
    }
}

// MARK: - WatchMedicationReminder Extensions

extension WatchMedicationReminder {
    var isOverdue: Bool {
        return !isTaken && scheduledTime < Date()
    }
    
    var timeUntilDue: TimeInterval {
        return scheduledTime.timeIntervalSinceNow
    }
    
    var displayTime: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: scheduledTime)
    }
    
    var statusColor: Color {
        if isTaken {
            return .green
        } else if isOverdue {
            return .red
        } else {
            return .orange
        }
    }
    
    var statusText: String {
        if isTaken {
            return "Taken"
        } else if isOverdue {
            return "Overdue"
        } else {
            return "Pending"
        }
    }
    
    var reminderTypeIcon: String {
        switch reminderType {
        case .daily:
            return "calendar"
        case .weekly:
            return "calendar.badge.clock"
        case .asNeeded:
            return "pills.fill"
        case .beforeMeals:
            return "fork.knife"
        case .afterMeals:
            return "fork.knife.circle"
        }
    }
}

// MARK: - WatchWorkoutData Extensions

extension WatchWorkoutData {
    var workoutTypeDisplayName: String {
        switch workoutType {
        case .walking:
            return "Walking"
        case .running:
            return "Running"
        case .cycling:
            return "Cycling"
        case .swimming:
            return "Swimming"
        case .yoga:
            return "Yoga"
        case .strength:
            return "Strength Training"
        case .other:
            return "Other"
        }
    }
    
    var workoutTypeIcon: String {
        switch workoutType {
        case .walking:
            return "figure.walk"
        case .running:
            return "figure.run"
        case .cycling:
            return "bicycle"
        case .swimming:
            return "figure.pool.swim"
        case .yoga:
            return "figure.mind.and.body"
        case .strength:
            return "dumbbell.fill"
        case .other:
            return "figure.mixed.cardio"
        }
    }
    
    var workoutTypeColor: Color {
        switch workoutType {
        case .walking:
            return .green
        case .running:
            return .red
        case .cycling:
            return .blue
        case .swimming:
            return .cyan
        case .yoga:
            return .purple
        case .strength:
            return .orange
        case .other:
            return .gray
        }
    }
    
    var formattedDuration: String {
        let hours = Int(duration) / 3600
        let minutes = Int(duration) % 3600 / 60
        let seconds = Int(duration) % 60
        
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        } else {
            return String(format: "%d:%02d", minutes, seconds)
        }
    }
    
    var formattedDistance: String? {
        guard let distance = distance else { return nil }
        
        if distance >= 1000 {
            return String(format: "%.2f km", distance / 1000)
        } else {
            return String(format: "%.0f m", distance)
        }
    }
    
    var formattedCalories: String {
        return "\(Int(activeCalories)) cal"
    }
    
    var averageHeartRateText: String? {
        guard let avgHR = averageHeartRate else { return nil }
        return "\(Int(avgHR)) BPM"
    }
}

// MARK: - WatchEnvironmentalData Extensions

extension WatchEnvironmentalData {
    var temperatureDisplayText: String? {
        guard let temp = temperature else { return nil }
        return String(format: "%.1f°F", temp)
    }
    
    var humidityDisplayText: String? {
        guard let humidity = humidity else { return nil }
        return String(format: "%.0f%%", humidity)
    }
    
    var pressureDisplayText: String? {
        guard let pressure = pressure else { return nil }
        return String(format: "%.1f hPa", pressure)
    }
    
    var uvIndexDisplayText: String? {
        guard let uvIndex = uvIndex else { return nil }
        return String(format: "%.1f", uvIndex)
    }
    
    var uvIndexColor: Color {
        guard let uvIndex = uvIndex else { return .gray }
        
        switch uvIndex {
        case 0...2:
            return .green
        case 3...5:
            return .yellow
        case 6...7:
            return .orange
        case 8...10:
            return .red
        default:
            return .purple
        }
    }
    
    var uvIndexDescription: String {
        guard let uvIndex = uvIndex else { return "Unknown" }
        
        switch uvIndex {
        case 0...2:
            return "Low"
        case 3...5:
            return "Moderate"
        case 6...7:
            return "High"
        case 8...10:
            return "Very High"
        default:
            return "Extreme"
        }
    }
    
    var airQualityColor: Color {
        guard let aqi = airQualityIndex else { return .gray }
        
        switch aqi {
        case 0...50:
            return .green
        case 51...100:
            return .yellow
        case 101...150:
            return .orange
        case 151...200:
            return .red
        case 201...300:
            return .purple
        default:
            return .brown
        }
    }
    
    var airQualityDescription: String {
        guard let aqi = airQualityIndex else { return "Unknown" }
        
        switch aqi {
        case 0...50:
            return "Good"
        case 51...100:
            return "Moderate"
        case 101...150:
            return "Unhealthy for Sensitive Groups"
        case 151...200:
            return "Unhealthy"
        case 201...300:
            return "Very Unhealthy"
        default:
            return "Hazardous"
        }
    }
}

// MARK: - WatchNotification Extensions

extension WatchNotification {
    var typeDisplayName: String {
        switch type {
        case .medicationReminder:
            return "Medication Reminder"
        case .symptomCheck:
            return "Symptom Check"
        case .appointmentReminder:
            return "Appointment Reminder"
        case .emergencyAlert:
            return "Emergency Alert"
        case .healthAlert:
            return "Health Alert"
        case .dataSync:
            return "Data Sync"
        }
    }
    
    var typeIcon: String {
        switch type {
        case .medicationReminder:
            return "pills.fill"
        case .symptomCheck:
            return "heart.text.square"
        case .appointmentReminder:
            return "calendar.badge.clock"
        case .emergencyAlert:
            return "exclamationmark.triangle.fill"
        case .healthAlert:
            return "heart.circle.fill"
        case .dataSync:
            return "arrow.triangle.2.circlepath"
        }
    }
    
    var typeColor: Color {
        switch type {
        case .medicationReminder:
            return .blue
        case .symptomCheck:
            return .orange
        case .appointmentReminder:
            return .green
        case .emergencyAlert:
            return .red
        case .healthAlert:
            return .purple
        case .dataSync:
            return .gray
        }
    }
    
    var priorityColor: Color {
        switch priority {
        case .low:
            return .gray
        case .normal:
            return .blue
        case .high:
            return .orange
        case .critical:
            return .red
        }
    }
    
    var priorityText: String {
        switch priority {
        case .low:
            return "Low"
        case .normal:
            return "Normal"
        case .high:
            return "High"
        case .critical:
            return "Critical"
        }
    }
    
    var isExpired: Bool {
        guard let expirationDate = expirationDate else { return false }
        return Date() > expirationDate
    }
    
    var timeUntilExpiration: TimeInterval? {
        guard let expirationDate = expirationDate else { return nil }
        return expirationDate.timeIntervalSinceNow
    }
}

// MARK: - WatchComplication Extensions

extension WatchComplication {
    var typeDisplayName: String {
        switch type {
        case .heartRate:
            return "Heart Rate"
        case .steps:
            return "Steps"
        case .medicationReminder:
            return "Medication"
        case .nextAppointment:
            return "Next Appointment"
        case .symptomSeverity:
            return "Symptoms"
        case .painLevel:
            return "Pain Level"
        }
    }
    
    var typeIcon: String {
        switch type {
        case .heartRate:
            return "heart.fill"
        case .steps:
            return "figure.walk"
        case .medicationReminder:
            return "pills.fill"
        case .nextAppointment:
            return "calendar"
        case .symptomSeverity:
            return "heart.text.square"
        case .painLevel:
            return "exclamationmark.circle"
        }
    }
    
    var formattedValue: String? {
        guard let value = value else { return nil }
        
        switch type {
        case .heartRate:
            return "\(Int(value)) BPM"
        case .steps:
            return NumberFormatter.localizedString(from: NSNumber(value: value), number: .decimal)
        case .medicationReminder:
            return "\(Int(value)) due"
        case .symptomSeverity, .painLevel:
            return "\(Int(value))/10"
        case .nextAppointment:
            return nil // Use displayText for appointments
        }
    }
    
    var needsUpdate: Bool {
        let updateInterval: TimeInterval
        
        switch type {
        case .heartRate:
            updateInterval = 300 // 5 minutes
        case .steps:
            updateInterval = 600 // 10 minutes
        case .medicationReminder:
            updateInterval = 3600 // 1 hour
        case .nextAppointment:
            updateInterval = 1800 // 30 minutes
        case .symptomSeverity, .painLevel:
            updateInterval = 7200 // 2 hours
        }
        
        return Date().timeIntervalSince(lastUpdated) > updateInterval
    }
}

// MARK: - AppleWatchManager Extensions

extension AppleWatchManager {
    var connectionStatusText: String {
        if isWatchConnected {
            return "Connected"
        } else if WCSession.default.isPaired {
            return "Paired but not reachable"
        } else {
            return "Not paired"
        }
    }
    
    var connectionStatusColor: Color {
        if isWatchConnected {
            return .green
        } else if WCSession.default.isPaired {
            return .orange
        } else {
            return .red
        }
    }
    
    var batteryStatusText: String {
        let percentage = Int(watchBatteryLevel * 100)
        
        if watchBatteryLevel > 0.5 {
            return "\(percentage)% - Good"
        } else if watchBatteryLevel > 0.2 {
            return "\(percentage)% - Low"
        } else {
            return "\(percentage)% - Critical"
        }
    }
    
    var batteryStatusColor: Color {
        if watchBatteryLevel > 0.5 {
            return .green
        } else if watchBatteryLevel > 0.2 {
            return .orange
        } else {
            return .red
        }
    }
    
    func canSendData() -> Bool {
        return isWatchConnected && isWatchAppInstalled
    }
    
    func shouldShowLowBatteryWarning() -> Bool {
        return isWatchConnected && watchBatteryLevel < 0.2
    }
}

// MARK: - WatchDataSyncManager Extensions

extension WatchDataSyncManager {
    var syncStatusDisplayText: String {
        switch syncStatus {
        case .idle:
            return "Ready to sync"
        case .syncing:
            return "Syncing data..."
        case .completed:
            return "Sync completed"
        case .failed:
            return "Sync failed"
        }
    }
    
    var syncStatusIcon: String {
        switch syncStatus {
        case .idle:
            return "arrow.triangle.2.circlepath"
        case .syncing:
            return "arrow.triangle.2.circlepath.circle"
        case .completed:
            return "checkmark.circle.fill"
        case .failed:
            return "xmark.circle.fill"
        }
    }
    
    var syncStatusColor: Color {
        switch syncStatus {
        case .idle:
            return .blue
        case .syncing:
            return .orange
        case .completed:
            return .green
        case .failed:
            return .red
        }
    }
    
    var lastSyncDisplayText: String? {
        guard let lastSync = lastSyncDate else { return nil }
        
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: lastSync, relativeTo: Date())
    }
    
    var pendingUploadsText: String {
        if pendingUploads == 0 {
            return "No pending uploads"
        } else if pendingUploads == 1 {
            return "1 item pending"
        } else {
            return "\(pendingUploads) items pending"
        }
    }
    
    func shouldShowSyncReminder() -> Bool {
        guard let lastSync = lastSyncDate else { return true }
        return Date().timeIntervalSince(lastSync) > 3600 // 1 hour
    }
}

// MARK: - WatchSettingsManager Extensions

extension WatchSettingsManager {
    func resetToDefaults() {
        settings = WatchSettings()
        updateSettings(settings)
    }
    
    func exportSettings() -> [String: Any] {
        return [
            "enableHeartRateMonitoring": settings.enableHeartRateMonitoring,
            "enableStepTracking": settings.enableStepTracking,
            "enableWorkoutTracking": settings.enableWorkoutTracking,
            "medicationReminderEnabled": settings.medicationReminderEnabled,
            "symptomReminderEnabled": settings.symptomReminderEnabled,
            "emergencyContactsEnabled": settings.emergencyContactsEnabled,
            "hapticFeedbackEnabled": settings.hapticFeedbackEnabled,
            "autoSyncEnabled": settings.autoSyncEnabled,
            "batteryOptimizationEnabled": settings.batteryOptimizationEnabled,
            "dataRetentionDays": settings.dataRetentionDays
        ]
    }
    
    func importSettings(from data: [String: Any]) {
        var newSettings = settings
        
        if let value = data["enableHeartRateMonitoring"] as? Bool {
            newSettings.enableHeartRateMonitoring = value
        }
        if let value = data["enableStepTracking"] as? Bool {
            newSettings.enableStepTracking = value
        }
        if let value = data["enableWorkoutTracking"] as? Bool {
            newSettings.enableWorkoutTracking = value
        }
        if let value = data["medicationReminderEnabled"] as? Bool {
            newSettings.medicationReminderEnabled = value
        }
        if let value = data["symptomReminderEnabled"] as? Bool {
            newSettings.symptomReminderEnabled = value
        }
        if let value = data["emergencyContactsEnabled"] as? Bool {
            newSettings.emergencyContactsEnabled = value
        }
        if let value = data["hapticFeedbackEnabled"] as? Bool {
            newSettings.hapticFeedbackEnabled = value
        }
        if let value = data["autoSyncEnabled"] as? Bool {
            newSettings.autoSyncEnabled = value
        }
        if let value = data["batteryOptimizationEnabled"] as? Bool {
            newSettings.batteryOptimizationEnabled = value
        }
        if let value = data["dataRetentionDays"] as? Int {
            newSettings.dataRetentionDays = value
        }
        
        settings = newSettings
        updateSettings(settings)
    }
}

// MARK: - Date Extensions for Watch

extension Date {
    var isToday: Bool {
        Calendar.current.isDateInToday(self)
    }
    
    var isYesterday: Bool {
        Calendar.current.isDateInYesterday(self)
    }
    
    var isTomorrow: Bool {
        Calendar.current.isDateInTomorrow(self)
    }
    
    var timeAgoText: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: self, relativeTo: Date())
    }
    
    var watchDisplayTime: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: self)
    }
    
    var watchDisplayDate: String {
        let formatter = DateFormatter()
        
        if isToday {
            return "Today"
        } else if isYesterday {
            return "Yesterday"
        } else if isTomorrow {
            return "Tomorrow"
        } else {
            formatter.dateStyle = .short
            return formatter.string(from: self)
        }
    }
}

// MARK: - Color Extensions for Watch

extension Color {
    static let watchPrimary = Color.blue
    static let watchSecondary = Color.gray
    static let watchAccent = Color.orange
    static let watchSuccess = Color.green
    static let watchWarning = Color.yellow
    static let watchError = Color.red
    
    // Watch-specific colors for different data types
    static let watchHeartRate = Color.red
    static let watchSteps = Color.green
    static let watchCalories = Color.orange
    static let watchSleep = Color.purple
    static let watchWorkout = Color.blue
    
    // Battery level colors
    static func batteryColor(for level: Double) -> Color {
        if level > 0.5 {
            return .green
        } else if level > 0.2 {
            return .orange
        } else {
            return .red
        }
    }
    
    // Health status colors
    static func healthStatusColor(isNormal: Bool) -> Color {
        return isNormal ? .green : .red
    }
}

// MARK: - String Extensions for Watch

extension String {
    var watchDisplayText: String {
        // Truncate long text for watch display
        if count > 30 {
            return String(prefix(27)) + "..."
        }
        return self
    }
    
    var watchTitleText: String {
        // Truncate titles for watch display
        if count > 20 {
            return String(prefix(17)) + "..."
        }
        return self
    }
}

// MARK: - NumberFormatter Extensions

extension NumberFormatter {
    static let watchSteps: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 0
        return formatter
    }()
    
    static let watchHeartRate: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 0
        return formatter
    }()
    
    static let watchCalories: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 0
        return formatter
    }()
    
    static let watchDistance: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 2
        return formatter
    }()
}