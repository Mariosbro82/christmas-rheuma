//
//  AppGroupConfig.swift
//  InflamAI
//
//  App Group configuration for sharing data between main app and widget extensions
//

import Foundation
import CoreData

/// App Group configuration for widget data sharing
public enum AppGroupConfig {
    /// App Group identifier - must match in all targets' entitlements
    /// Using unified identifier across iOS app, Watch app, and all widget extensions
    public static let appGroupIdentifier = "group.com.inflamai.InflamAI"

    /// Shared UserDefaults for lightweight data
    public static var sharedDefaults: UserDefaults? {
        UserDefaults(suiteName: appGroupIdentifier)
    }

    /// Shared container URL for Core Data
    public static var sharedContainerURL: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupIdentifier)
    }

    /// Core Data store URL in shared container
    public static var sharedStoreURL: URL? {
        sharedContainerURL?.appendingPathComponent("InflamAI.sqlite")
    }
}

// MARK: - Shared UserDefaults Keys

public enum WidgetDataKeys {
    public static let flareRiskPercentage = "widget.flareRisk.percentage"
    public static let flareRiskLevel = "widget.flareRisk.level"
    public static let flareRiskFactors = "widget.flareRisk.factors"
    public static let flareRiskUpdated = "widget.flareRisk.updated"

    public static let basdaiScore = "widget.basdai.score"
    public static let basdaiCategory = "widget.basdai.category"
    public static let basdaiTrend = "widget.basdai.trend"
    public static let basdaiUpdated = "widget.basdai.updated"

    public static let loggingStreak = "widget.streak.days"
    public static let streakUpdated = "widget.streak.updated"

    public static let hasActiveFlare = "widget.flare.active"
    public static let activeFlareStartDate = "widget.flare.startDate"

    public static let nextMedications = "widget.medications.next"
    public static let medicationsUpdated = "widget.medications.updated"

    public static let todayPainEntries = "widget.today.painEntries"
    public static let todayAssessments = "widget.today.assessments"
    public static let hasLoggedToday = "widget.today.hasLogged"

    // Health/biometric data keys (from HealthKit via Watch)
    public static let healthSteps = "widget.health.steps"
    public static let healthHRV = "widget.health.hrv"
    public static let healthPainLevel = "widget.health.painLevel"
    public static let healthRestingHR = "widget.health.restingHR"
    public static let healthSleepHours = "widget.health.sleepHours"
    public static let healthUpdated = "widget.health.updated"

    // Watch-specific sync keys
    public static let lastWatchSync = "widget.watch.lastSync"
    public static let watchConnected = "widget.watch.connected"
}
