//
//  WidgetDataProvider.swift
//  InflamAI
//
//  Centralized data provider for widgets - fetches and caches data for widget display
//

import Foundation
import CoreData
import WidgetKit

/// Provides data to widgets from shared storage
public class WidgetDataProvider {

    public static let shared = WidgetDataProvider()

    private let defaults: UserDefaults?

    private init() {
        self.defaults = AppGroupConfig.sharedDefaults
    }

    // MARK: - Flare Risk Data

    public func getFlareRiskData() -> WidgetFlareData {
        guard let defaults = defaults else {
            return WidgetFlareData.placeholder
        }

        let percentage = defaults.integer(forKey: WidgetDataKeys.flareRiskPercentage)
        let levelRaw = defaults.string(forKey: WidgetDataKeys.flareRiskLevel) ?? "low"
        let factorsData = defaults.data(forKey: WidgetDataKeys.flareRiskFactors)
        let updated = defaults.object(forKey: WidgetDataKeys.flareRiskUpdated) as? Date ?? Date()

        let factors: [String] = {
            guard let data = factorsData else { return [] }
            return (try? JSONDecoder().decode([String].self, from: data)) ?? []
        }()

        return WidgetFlareData(
            riskPercentage: percentage,
            riskLevel: WidgetFlareData.RiskLevel(rawValue: levelRaw) ?? .low,
            topFactors: factors,
            lastUpdated: updated
        )
    }

    // MARK: - BASDAI Data

    public func getBASDAIData() -> WidgetBASDAIData {
        guard let defaults = defaults else {
            return WidgetBASDAIData.placeholder
        }

        let score = defaults.double(forKey: WidgetDataKeys.basdaiScore)
        let category = defaults.string(forKey: WidgetDataKeys.basdaiCategory) ?? "Unknown"
        let trendRaw = defaults.string(forKey: WidgetDataKeys.basdaiTrend) ?? "stable"
        let updated = defaults.object(forKey: WidgetDataKeys.basdaiUpdated) as? Date ?? Date()

        return WidgetBASDAIData(
            score: score,
            category: category,
            trend: WidgetBASDAIData.TrendDirection(rawValue: trendRaw) ?? .stable,
            lastAssessed: updated
        )
    }

    // MARK: - Streak Data

    public func getStreakData() -> WidgetStreakData {
        guard let defaults = defaults else {
            return WidgetStreakData.placeholder
        }

        let streak = defaults.integer(forKey: WidgetDataKeys.loggingStreak)
        let updated = defaults.object(forKey: WidgetDataKeys.streakUpdated) as? Date ?? Date()

        return WidgetStreakData(
            streakDays: streak,
            lastUpdated: updated
        )
    }

    // MARK: - Medication Data

    public func getMedicationData() -> WidgetMedicationData {
        guard let defaults = defaults,
              let data = defaults.data(forKey: WidgetDataKeys.nextMedications) else {
            return WidgetMedicationData.placeholder
        }

        let medications = (try? JSONDecoder().decode([WidgetMedicationData.MedicationReminder].self, from: data)) ?? []
        let updated = defaults.object(forKey: WidgetDataKeys.medicationsUpdated) as? Date ?? Date()

        return WidgetMedicationData(
            medications: medications,
            lastUpdated: updated
        )
    }

    // MARK: - Today's Summary Data

    public func getTodaySummaryData() -> WidgetTodaySummary {
        guard let defaults = defaults else {
            return WidgetTodaySummary.placeholder
        }

        return WidgetTodaySummary(
            painEntries: defaults.integer(forKey: WidgetDataKeys.todayPainEntries),
            assessments: defaults.integer(forKey: WidgetDataKeys.todayAssessments),
            hasLoggedToday: defaults.bool(forKey: WidgetDataKeys.hasLoggedToday),
            hasActiveFlare: defaults.bool(forKey: WidgetDataKeys.hasActiveFlare)
        )
    }

    // MARK: - Active Flare Data

    public func hasActiveFlare() -> Bool {
        defaults?.bool(forKey: WidgetDataKeys.hasActiveFlare) ?? false
    }
}

// MARK: - Main App Data Writer

/// Used by main app to write data for widgets to read
public class WidgetDataWriter {

    public static let shared = WidgetDataWriter()

    private let defaults: UserDefaults?

    private init() {
        self.defaults = AppGroupConfig.sharedDefaults
    }

    /// Update flare risk data for widgets
    public func updateFlareRisk(percentage: Int, level: String, factors: [String]) {
        defaults?.set(percentage, forKey: WidgetDataKeys.flareRiskPercentage)
        defaults?.set(level, forKey: WidgetDataKeys.flareRiskLevel)
        defaults?.set(Date(), forKey: WidgetDataKeys.flareRiskUpdated)

        if let factorsData = try? JSONEncoder().encode(factors) {
            defaults?.set(factorsData, forKey: WidgetDataKeys.flareRiskFactors)
        }

        reloadWidgets()
    }

    /// Update BASDAI data for widgets
    public func updateBASDAI(score: Double, category: String, trend: String) {
        defaults?.set(score, forKey: WidgetDataKeys.basdaiScore)
        defaults?.set(category, forKey: WidgetDataKeys.basdaiCategory)
        defaults?.set(trend, forKey: WidgetDataKeys.basdaiTrend)
        defaults?.set(Date(), forKey: WidgetDataKeys.basdaiUpdated)

        reloadWidgets()
    }

    /// Update streak data for widgets
    public func updateStreak(days: Int) {
        defaults?.set(days, forKey: WidgetDataKeys.loggingStreak)
        defaults?.set(Date(), forKey: WidgetDataKeys.streakUpdated)

        reloadWidgets()
    }

    /// Update medication data for widgets
    public func updateMedications(_ medications: [WidgetMedicationData.MedicationReminder]) {
        if let data = try? JSONEncoder().encode(medications) {
            defaults?.set(data, forKey: WidgetDataKeys.nextMedications)
        }
        defaults?.set(Date(), forKey: WidgetDataKeys.medicationsUpdated)

        reloadWidgets()
    }

    /// Update today's summary for widgets
    public func updateTodaySummary(painEntries: Int, assessments: Int, hasLogged: Bool, hasActiveFlare: Bool) {
        defaults?.set(painEntries, forKey: WidgetDataKeys.todayPainEntries)
        defaults?.set(assessments, forKey: WidgetDataKeys.todayAssessments)
        defaults?.set(hasLogged, forKey: WidgetDataKeys.hasLoggedToday)
        defaults?.set(hasActiveFlare, forKey: WidgetDataKeys.hasActiveFlare)

        reloadWidgets()
    }

    /// Trigger widget refresh
    private func reloadWidgets() {
        WidgetCenter.shared.reloadAllTimelines()
    }
}
