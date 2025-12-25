//
//  TriggerDataService.swift
//  InflamAI
//
//  Service layer for trigger data operations
//  Provides clean async interface for trigger logging and querying
//

import Foundation
import CoreData
import Combine

// MARK: - TriggerDataService

@MainActor
public final class TriggerDataService: ObservableObject {

    // MARK: - Singleton

    public static let shared = TriggerDataService()

    // MARK: - Published State

    @Published public private(set) var todaysTriggers: [TriggerLog] = []
    @Published public private(set) var recentTriggers: [TriggerLog] = []
    @Published public private(set) var triggerStatistics: [String: TriggerStatistics] = [:]
    @Published public private(set) var isLoading: Bool = false

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    private init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController

        // Load initial data
        Task {
            await refreshTodaysTriggers()
            await refreshRecentTriggers()
        }
    }

    // MARK: - Context

    private var viewContext: NSManagedObjectContext {
        persistenceController.container.viewContext
    }

    // MARK: - Logging Methods

    /// Log a trigger from a definition
    @discardableResult
    public func logTrigger(
        definition: TriggerDefinition,
        value: Double,
        timestamp: Date = Date(),
        notes: String? = nil,
        symptomLog: SymptomLog? = nil
    ) async throws -> TriggerLog {
        let context = viewContext

        let log = TriggerLog.create(
            from: definition,
            value: value,
            timestamp: timestamp,
            notes: notes,
            symptomLog: symptomLog,
            in: context
        )

        try context.save()

        // Refresh today's triggers if relevant
        if Calendar.current.isDateInToday(timestamp) {
            await refreshTodaysTriggers()
        }

        // Invalidate cached analysis for this trigger
        await invalidateAnalysisCache(for: definition.id)

        return log
    }

    /// Log multiple triggers at once
    public func logTriggers(
        _ triggerValues: [(definition: TriggerDefinition, value: Double)],
        timestamp: Date = Date(),
        symptomLog: SymptomLog? = nil
    ) async throws {
        let context = viewContext

        for (definition, value) in triggerValues {
            TriggerLog.create(
                from: definition,
                value: value,
                timestamp: timestamp,
                symptomLog: symptomLog,
                in: context
            )
        }

        try context.save()

        // Refresh if today
        if Calendar.current.isDateInToday(timestamp) {
            await refreshTodaysTriggers()
        }

        // Invalidate caches
        for (definition, _) in triggerValues {
            await invalidateAnalysisCache(for: definition.id)
        }
    }

    /// Update an existing trigger log
    public func updateTrigger(
        _ log: TriggerLog,
        newValue: Double,
        notes: String? = nil
    ) async throws {
        log.triggerValue = newValue
        if let notes = notes {
            log.notes = notes
        }

        try viewContext.save()

        await refreshTodaysTriggers()
        await invalidateAnalysisCache(for: log.triggerName ?? "")
    }

    /// Delete a trigger log
    public func deleteTrigger(_ log: TriggerLog) async throws {
        let triggerName = log.triggerName ?? ""
        viewContext.delete(log)
        try viewContext.save()

        await refreshTodaysTriggers()
        await invalidateAnalysisCache(for: triggerName)
    }

    // MARK: - Query Methods

    /// Refresh today's triggers
    public func refreshTodaysTriggers() async {
        todaysTriggers = TriggerLog.fetch(forDate: Date(), in: viewContext)
    }

    /// Refresh recent triggers (last 7 days)
    public func refreshRecentTriggers() async {
        let startDate = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        recentTriggers = TriggerLog.fetch(from: startDate, to: Date(), in: viewContext)
    }

    /// Get all triggers for a date
    public func getTriggers(for date: Date) -> [TriggerLog] {
        TriggerLog.fetch(forDate: date, in: viewContext)
    }

    /// Get triggers for date range
    public func getTriggers(from startDate: Date, to endDate: Date) -> [TriggerLog] {
        TriggerLog.fetch(from: startDate, to: endDate, in: viewContext)
    }

    /// Get triggers by name
    public func getTriggers(
        named triggerName: String,
        from startDate: Date? = nil,
        to endDate: Date? = nil
    ) -> [TriggerLog] {
        TriggerLog.fetch(
            triggerName: triggerName,
            from: startDate,
            to: endDate,
            in: viewContext
        )
    }

    /// Get triggers by category
    public func getTriggers(
        category: TriggerCategory,
        from startDate: Date? = nil,
        to endDate: Date? = nil
    ) -> [TriggerLog] {
        TriggerLog.fetch(
            category: category,
            from: startDate,
            to: endDate,
            in: viewContext
        )
    }

    /// Check if a trigger has been logged today
    public func hasTriggerToday(_ triggerName: String) -> Bool {
        todaysTriggers.contains { $0.triggerName == triggerName }
    }

    /// Get today's value for a trigger
    public func todaysValue(for triggerName: String) -> Double? {
        todaysTriggers.first { $0.triggerName == triggerName }?.triggerValue
    }

    /// Get all unique trigger names that have been logged
    public func getLoggedTriggerNames() -> [String] {
        TriggerLog.uniqueTriggerNames(in: viewContext)
    }

    // MARK: - Statistics

    /// Get statistics for a trigger
    public func getStatistics(
        for triggerName: String,
        days: Int = 90
    ) -> TriggerStatistics {
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: endDate) ?? endDate

        return TriggerLog.statistics(
            for: triggerName,
            from: startDate,
            to: endDate,
            in: viewContext
        )
    }

    /// Refresh statistics for all logged triggers
    public func refreshAllStatistics(days: Int = 90) async {
        isLoading = true
        defer { isLoading = false }

        let triggerNames = getLoggedTriggerNames()
        var newStats: [String: TriggerStatistics] = [:]

        for name in triggerNames {
            newStats[name] = getStatistics(for: name, days: days)
        }

        triggerStatistics = newStats
    }

    // MARK: - Days Count

    /// Get total days of trigger data
    public func totalDaysWithTriggerData() -> Int {
        let logs = TriggerLog.fetchAll(in: viewContext)
        let calendar = Calendar.current
        let uniqueDays = Set(logs.compactMap { $0.timestamp.map { calendar.startOfDay(for: $0) } })
        return uniqueDays.count
    }

    /// Get number of days a specific trigger has been logged
    public func daysWithTrigger(_ triggerName: String) -> Int {
        let logs = getTriggers(named: triggerName)
        let calendar = Calendar.current
        let uniqueDays = Set(logs.compactMap { $0.timestamp.map { calendar.startOfDay(for: $0) } })
        return uniqueDays.count
    }

    // MARK: - Data Preparation for Analysis

    /// Get trigger values aligned with symptom logs for correlation analysis
    public func prepareForAnalysis(
        triggerName: String,
        from startDate: Date,
        to endDate: Date
    ) -> (triggerValues: [Double], dates: [Date]) {
        let logs = getTriggers(named: triggerName, from: startDate, to: endDate)

        // Group by date and take the max value for each day
        let calendar = Calendar.current
        var dailyValues: [Date: Double] = [:]

        for log in logs {
            guard let timestamp = log.timestamp else { continue }
            let day = calendar.startOfDay(for: timestamp)
            dailyValues[day] = max(dailyValues[day] ?? 0, log.triggerValue)
        }

        // Sort by date
        let sortedDays = dailyValues.keys.sorted()
        let values = sortedDays.map { dailyValues[$0] ?? 0 }

        return (values, sortedDays)
    }

    /// Get all triggers for a day as a dictionary
    public func getTriggersAsDict(for date: Date) -> [String: Double] {
        let logs = getTriggers(for: date)
        var result: [String: Double] = [:]

        for log in logs {
            if let name = log.triggerName {
                result[name] = log.triggerValue
            }
        }

        return result
    }

    // MARK: - Cache Invalidation

    /// Invalidate analysis cache for a trigger
    private func invalidateAnalysisCache(for triggerName: String) async {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(format: "triggerName == %@", triggerName)

        if let caches = try? viewContext.fetch(request) {
            for cache in caches {
                cache.isValid = false
            }
            try? viewContext.save()
        }
    }

    /// Invalidate all analysis caches
    public func invalidateAllAnalysisCaches() async {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()

        if let caches = try? viewContext.fetch(request) {
            for cache in caches {
                cache.isValid = false
            }
            try? viewContext.save()
        }
    }

    // MARK: - Cleanup

    /// Delete old trigger data
    public func cleanupOldData(olderThan days: Int) async throws {
        guard let cutoffDate = Calendar.current.date(byAdding: .day, value: -days, to: Date()) else {
            return
        }

        try TriggerLog.deleteOlderThan(date: cutoffDate, in: viewContext)
        try viewContext.save()
    }
}

// MARK: - Quick Trigger Logging

extension TriggerDataService {

    /// Quick log coffee
    public func logCoffee(cups: Int) async throws {
        guard let definition = getTriggerDefinition(id: "coffee") else { return }
        try await logTrigger(definition: definition, value: Double(cups))
    }

    /// Quick log alcohol
    public func logAlcohol(drinks: Int) async throws {
        guard let definition = getTriggerDefinition(id: "alcohol") else { return }
        try await logTrigger(definition: definition, value: Double(drinks))
    }

    /// Quick log stress level
    public func logStress(level: Int) async throws {
        guard let definition = getTriggerDefinition(id: "stress") else { return }
        try await logTrigger(definition: definition, value: Double(level))
    }

    /// Quick log sleep quality
    public func logSleepQuality(rating: Int) async throws {
        guard let definition = getTriggerDefinition(id: "sleep_quality") else { return }
        try await logTrigger(definition: definition, value: Double(rating))
    }

    /// Quick log sleep duration
    public func logSleepDuration(hours: Double) async throws {
        guard let definition = getTriggerDefinition(id: "sleep_duration") else { return }
        try await logTrigger(definition: definition, value: hours)
    }

    /// Quick log binary trigger (yes/no)
    public func logBinaryTrigger(_ triggerId: String, present: Bool) async throws {
        guard let definition = getTriggerDefinition(id: triggerId) else { return }
        try await logTrigger(definition: definition, value: present ? 1.0 : 0.0)
    }
}
