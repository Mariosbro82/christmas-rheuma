//
//  WidgetDataModels.swift
//  InflamAIWidgetExtension
//
//  Data models for widget entries and shared data
//

import Foundation
import SwiftUI

// MARK: - Flare Risk Data

public struct WidgetFlareData: Codable {
    public let riskPercentage: Int
    public let riskLevel: RiskLevel
    public let topFactors: [String]
    public let lastUpdated: Date

    public enum RiskLevel: String, Codable {
        case low
        case moderate
        case high
        case veryHigh

        public var displayName: String {
            switch self {
            case .low: return "Low"
            case .moderate: return "Moderate"
            case .high: return "High"
            case .veryHigh: return "Very High"
            }
        }

        public var color: Color {
            switch self {
            case .low: return .green
            case .moderate: return .orange
            case .high: return Color(red: 0.9, green: 0.3, blue: 0.1)
            case .veryHigh: return .red
            }
        }

        public var icon: String {
            switch self {
            case .low: return "checkmark.shield.fill"
            case .moderate: return "exclamationmark.triangle.fill"
            case .high: return "flame.fill"
            case .veryHigh: return "exclamationmark.octagon.fill"
            }
        }
    }

    public static var placeholder: WidgetFlareData {
        WidgetFlareData(
            riskPercentage: 35,
            riskLevel: .moderate,
            topFactors: ["Weather", "Sleep"],
            lastUpdated: Date()
        )
    }
}

// MARK: - BASDAI Data

public struct WidgetBASDAIData: Codable {
    public let score: Double
    public let category: String
    public let trend: TrendDirection
    public let lastAssessed: Date

    public enum TrendDirection: String, Codable {
        case improving
        case stable
        case worsening

        public var icon: String {
            switch self {
            case .improving: return "arrow.down.right"
            case .stable: return "arrow.right"
            case .worsening: return "arrow.up.right"
            }
        }

        public var color: Color {
            switch self {
            case .improving: return .green
            case .stable: return .blue
            case .worsening: return .red
            }
        }
    }

    public var severityColor: Color {
        switch score {
        case 0..<2: return .green
        case 2..<4: return Color(red: 0.6, green: 0.8, blue: 0.2)
        case 4..<6: return .orange
        case 6..<8: return Color(red: 0.9, green: 0.3, blue: 0.1)
        default: return .red
        }
    }

    public static var placeholder: WidgetBASDAIData {
        WidgetBASDAIData(
            score: 4.2,
            category: "Moderate",
            trend: .stable,
            lastAssessed: Date()
        )
    }
}

// MARK: - Streak Data

public struct WidgetStreakData: Codable {
    public let streakDays: Int
    public let lastUpdated: Date

    public static var placeholder: WidgetStreakData {
        WidgetStreakData(streakDays: 7, lastUpdated: Date())
    }
}

// MARK: - Medication Data

public struct WidgetMedicationData: Codable {
    public let medications: [MedicationReminder]
    public let lastUpdated: Date

    public struct MedicationReminder: Codable, Identifiable {
        public let id: UUID
        public let name: String
        public let dosage: String
        public let nextDoseTime: Date
        public let frequency: String

        public init(id: UUID = UUID(), name: String, dosage: String, nextDoseTime: Date, frequency: String = "Daily") {
            self.id = id
            self.name = name
            self.dosage = dosage
            self.nextDoseTime = nextDoseTime
            self.frequency = frequency
        }

        public var timeString: String {
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            return formatter.string(from: nextDoseTime)
        }

        public var relativeTimeString: String {
            let formatter = RelativeDateTimeFormatter()
            formatter.unitsStyle = .abbreviated
            return formatter.localizedString(for: nextDoseTime, relativeTo: Date())
        }

        public var isOverdue: Bool {
            nextDoseTime < Date()
        }

        public var isDueSoon: Bool {
            let oneHour = Date().addingTimeInterval(3600)
            return nextDoseTime <= oneHour && !isOverdue
        }
    }

    public static var placeholder: WidgetMedicationData {
        WidgetMedicationData(
            medications: [
                MedicationReminder(name: "Humira", dosage: "40mg", nextDoseTime: Date().addingTimeInterval(3600), frequency: "Weekly"),
                MedicationReminder(name: "Naproxen", dosage: "500mg", nextDoseTime: Date().addingTimeInterval(7200), frequency: "Twice Daily")
            ],
            lastUpdated: Date()
        )
    }
}

// MARK: - Today's Summary Data

public struct WidgetTodaySummary: Codable {
    public let painEntries: Int
    public let assessments: Int
    public let hasLoggedToday: Bool
    public let hasActiveFlare: Bool

    public static var placeholder: WidgetTodaySummary {
        WidgetTodaySummary(
            painEntries: 2,
            assessments: 1,
            hasLoggedToday: true,
            hasActiveFlare: false
        )
    }
}
