//
//  WatchDataProvider.swift
//  InflamAIWatchWidgets
//
//  Data provider for Apple Watch widgets
//  Reads from shared App Group storage using unified AppGroupConfig
//

import Foundation
import SwiftUI

// MARK: - Watch Data Provider

class WatchDataProvider {
    static let shared = WatchDataProvider()

    #if os(watchOS)
    private let defaults: UserDefaults?

    private init() {
        // Use unified App Group config - matches iOS app, Watch app, and all widgets
        self.defaults = AppGroupConfig.sharedDefaults
    }
    #else
    private init() {
        // iOS stub - widgets run on watchOS only
    }
    #endif

    // MARK: - Flare Risk Data

    func getFlareRiskData() -> WatchFlareData {
        #if os(watchOS)
        guard let defaults = defaults else {
            return .placeholder
        }

        return WatchFlareData(
            riskPercentage: defaults.integer(forKey: WidgetDataKeys.flareRiskPercentage),
            riskLevel: WatchFlareData.RiskLevel(
                rawValue: defaults.string(forKey: WidgetDataKeys.flareRiskLevel) ?? "low"
            ) ?? .low
        )
        #else
        return .placeholder
        #endif
    }

    // MARK: - BASDAI Data

    func getBASDAIData() -> WatchBASDAIData {
        #if os(watchOS)
        guard let defaults = defaults else {
            return .placeholder
        }

        return WatchBASDAIData(
            score: defaults.double(forKey: WidgetDataKeys.basdaiScore),
            category: defaults.string(forKey: WidgetDataKeys.basdaiCategory) ?? "Unknown",
            trend: WatchBASDAIData.TrendDirection(
                rawValue: defaults.string(forKey: WidgetDataKeys.basdaiTrend) ?? "stable"
            ) ?? .stable
        )
        #else
        return .placeholder
        #endif
    }

    // MARK: - Medication Data

    func getMedicationData() -> WatchMedicationData {
        #if os(watchOS)
        guard let defaults = defaults,
              let data = defaults.data(forKey: WidgetDataKeys.nextMedications),
              let medications = try? JSONDecoder().decode([WatchMedicationData.MedicationReminder].self, from: data)
        else {
            return .placeholder
        }

        return WatchMedicationData(medications: medications)
        #else
        return .placeholder
        #endif
    }

    // MARK: - Streak Data

    func getStreakData() -> WatchStreakData {
        #if os(watchOS)
        guard let defaults = defaults else {
            return .placeholder
        }

        return WatchStreakData(
            streakDays: defaults.integer(forKey: WidgetDataKeys.loggingStreak)
        )
        #else
        return .placeholder
        #endif
    }

    // MARK: - Quick Stats Data

    func getQuickStatsData() -> WatchQuickStatsData {
        #if os(watchOS)
        guard let defaults = defaults else {
            return .placeholder
        }

        return WatchQuickStatsData(
            steps: defaults.integer(forKey: WidgetDataKeys.healthSteps),
            hrv: defaults.double(forKey: WidgetDataKeys.healthHRV),
            painLevel: defaults.integer(forKey: WidgetDataKeys.healthPainLevel)
        )
        #else
        return .placeholder
        #endif
    }
}

// MARK: - Watch Data Models

struct WatchFlareData {
    let riskPercentage: Int
    let riskLevel: RiskLevel

    enum RiskLevel: String {
        case low, moderate, high, veryHigh

        var displayName: String {
            switch self {
            case .low: return "Low"
            case .moderate: return "Moderate"
            case .high: return "High"
            case .veryHigh: return "Very High"
            }
        }

        var color: Color {
            switch self {
            case .low: return .green
            case .moderate: return .orange
            case .high: return Color(red: 0.9, green: 0.3, blue: 0.1)
            case .veryHigh: return .red
            }
        }

        var icon: String {
            switch self {
            case .low: return "checkmark.shield.fill"
            case .moderate: return "exclamationmark.triangle.fill"
            case .high: return "flame.fill"
            case .veryHigh: return "exclamationmark.octagon.fill"
            }
        }
    }

    static var placeholder: WatchFlareData {
        WatchFlareData(riskPercentage: 35, riskLevel: .moderate)
    }
}

struct WatchBASDAIData {
    let score: Double
    let category: String
    let trend: TrendDirection

    enum TrendDirection: String {
        case improving, stable, worsening

        var icon: String {
            switch self {
            case .improving: return "arrow.down.right"
            case .stable: return "arrow.right"
            case .worsening: return "arrow.up.right"
            }
        }

        var color: Color {
            switch self {
            case .improving: return .green
            case .stable: return .blue
            case .worsening: return .red
            }
        }
    }

    var severityColor: Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return Color(red: 0.6, green: 0.8, blue: 0.2)
        case 4..<6: return .orange
        case 6..<8: return Color(red: 0.9, green: 0.3, blue: 0.1)
        default: return .red
        }
    }

    static var placeholder: WatchBASDAIData {
        WatchBASDAIData(score: 4.2, category: "Moderate", trend: .stable)
    }
}

struct WatchMedicationData {
    let medications: [MedicationReminder]

    struct MedicationReminder: Codable, Identifiable {
        let id: UUID
        let name: String
        let time: Date
        let dosage: String

        var timeString: String {
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            return formatter.string(from: time)
        }

        var relativeTimeString: String {
            let formatter = RelativeDateTimeFormatter()
            formatter.unitsStyle = .abbreviated
            return formatter.localizedString(for: time, relativeTo: Date())
        }
    }

    static var placeholder: WatchMedicationData {
        WatchMedicationData(medications: [
            MedicationReminder(
                id: UUID(),
                name: "Humira",
                time: Date().addingTimeInterval(3600),
                dosage: "40mg"
            )
        ])
    }
}

struct WatchStreakData {
    let streakDays: Int

    static var placeholder: WatchStreakData {
        WatchStreakData(streakDays: 7)
    }
}
