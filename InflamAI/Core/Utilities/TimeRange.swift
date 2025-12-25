//
//  TimeRange.swift
//  InflamAI
//
//  Standard time range utilities for trend analysis and data filtering
//  Used across analytics, trends, and reporting features
//

import Foundation

/// Standard time ranges for data analysis and trend visualization
enum TimeRange: String, CaseIterable, Codable {
    case week = "Week"
    case month = "Month"
    case threeMonths = "3 Months"
    case year = "Year"

    // MARK: - Date Calculation

    /// Calculate start date for this time range
    var startDate: Date {
        let calendar = Calendar.current
        let now = Date()

        switch self {
        case .week:
            return calendar.date(byAdding: .day, value: -7, to: now) ?? now
        case .month:
            return calendar.date(byAdding: .day, value: -30, to: now) ?? now
        case .threeMonths:
            return calendar.date(byAdding: .day, value: -90, to: now) ?? now
        case .year:
            return calendar.date(byAdding: .day, value: -365, to: now) ?? now
        }
    }

    /// Calculate end date for this time range (always now)
    var endDate: Date {
        return Date()
    }

    // MARK: - Display Properties

    /// Human-readable display name
    var displayName: String {
        switch self {
        case .week:
            return "7 days"
        case .month:
            return "30 days"
        case .threeMonths:
            return "90 days"
        case .year:
            return "365 days"
        }
    }

    /// Short display name for compact UIs
    var shortName: String {
        switch self {
        case .week:
            return "1W"
        case .month:
            return "1M"
        case .threeMonths:
            return "3M"
        case .year:
            return "1Y"
        }
    }

    // MARK: - Chart Configuration

    /// Number of days in this range
    var daysCount: Int {
        switch self {
        case .week:
            return 7
        case .month:
            return 30
        case .threeMonths:
            return 90
        case .year:
            return 365
        }
    }

    /// Recommended x-axis stride for charts
    var chartStride: Calendar.Component {
        switch self {
        case .week:
            return .day
        case .month:
            return .day
        case .threeMonths:
            return .weekOfYear
        case .year:
            return .month
        }
    }

    /// Recommended number of data points to show
    var recommendedDataPoints: Int {
        switch self {
        case .week:
            return 7
        case .month:
            return 30
        case .threeMonths:
            return 12  // Weekly averages
        case .year:
            return 12  // Monthly averages
        }
    }
}
