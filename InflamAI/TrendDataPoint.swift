//
//  TrendDataPoint.swift
//  InflamAI-Swift
//
//  Created by AI Assistant
//

import Foundation
import CoreData

/// A data structure representing a single point in a trend chart
struct TrendDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let value: Double
    
    init(date: Date, value: Double) {
        self.date = date
        self.value = value
    }
}

/// Extension for creating trend data points from different data types
extension TrendDataPoint {
    /// Create a trend data point from a pain entry
    static func from(painEntry: PainEntry) -> TrendDataPoint? {
        guard let timestamp = painEntry.timestamp else { return nil }
        return TrendDataPoint(date: timestamp, value: painEntry.painLevel)
    }
    
    /// Create a trend data point from a journal entry (energy level)
    static func from(journalEntry: JournalEntry, using keyPath: KeyPath<JournalEntry, Double>) -> TrendDataPoint? {
        guard let date = journalEntry.date else { return nil }
        return TrendDataPoint(date: date, value: journalEntry[keyPath: keyPath])
    }
    
    /// Create a trend data point from a BASDAI assessment
    static func from(assessment: BASSDAIAssessment, using keyPath: KeyPath<BASSDAIAssessment, Double>) -> TrendDataPoint? {
        guard let date = assessment.date else { return nil }
        return TrendDataPoint(date: date, value: assessment[keyPath: keyPath])
    }
}

/// Helper for aggregating trend data by time periods
extension Array where Element == TrendDataPoint {
    /// Group trend data points by day and calculate average values
    func groupedByDay() -> [TrendDataPoint] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: self) { dataPoint in
            calendar.startOfDay(for: dataPoint.date)
        }
        
        return grouped.compactMap { (date, points) in
            let averageValue = points.reduce(0) { $0 + $1.value } / Double(points.count)
            return TrendDataPoint(date: date, value: averageValue)
        }.sorted { $0.date < $1.date }
    }
    
    /// Group trend data points by week and calculate average values
    func groupedByWeek() -> [TrendDataPoint] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: self) { dataPoint in
            calendar.dateInterval(of: .weekOfYear, for: dataPoint.date)?.start ?? dataPoint.date
        }
        
        return grouped.compactMap { (date, points) in
            let averageValue = points.reduce(0) { $0 + $1.value } / Double(points.count)
            return TrendDataPoint(date: date, value: averageValue)
        }.sorted { $0.date < $1.date }
    }
    
    /// Group trend data points by month and calculate average values
    func groupedByMonth() -> [TrendDataPoint] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: self) { dataPoint in
            calendar.dateInterval(of: .month, for: dataPoint.date)?.start ?? dataPoint.date
        }
        
        return grouped.compactMap { (date, points) in
            let averageValue = points.reduce(0) { $0 + $1.value } / Double(points.count)
            return TrendDataPoint(date: date, value: averageValue)
        }.sorted { $0.date < $1.date }
    }
}