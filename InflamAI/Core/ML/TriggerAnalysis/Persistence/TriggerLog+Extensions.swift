//
//  TriggerLog+Extensions.swift
//  InflamAI
//
//  Convenience methods for TriggerLog Core Data entity
//  Used by the Hybrid Trigger Detection System
//

import Foundation
import CoreData

// MARK: - TriggerLog Convenience Extensions

extension TriggerLog {

    // MARK: - Category Accessor

    /// Get the trigger category as enum
    var category: TriggerCategory {
        TriggerCategory(rawValue: triggerCategory ?? "other") ?? .other
    }

    /// Set the trigger category from enum
    func setCategory(_ category: TriggerCategory) {
        self.triggerCategory = category.rawValue
    }

    // MARK: - Source Accessor

    /// Get the data source as enum
    var dataSource: TriggerDataSource {
        TriggerDataSource(rawValue: source ?? "manual") ?? .manual
    }

    /// Set the data source from enum
    func setDataSource(_ source: TriggerDataSource) {
        self.source = source.rawValue
    }

    // MARK: - Computed Properties

    /// Whether this trigger is considered "present" (value > 0)
    var isPresent: Bool {
        triggerValue > 0
    }

    /// Display-friendly value string
    var displayValue: String {
        if isBinary {
            return triggerValue > 0 ? "Yes" : "No"
        } else if let unit = triggerUnit {
            if unit == "steps" || unit == "minutes" {
                return "\(Int(triggerValue)) \(unit)"
            }
            return "\(String(format: "%.1f", triggerValue)) \(unit)"
        } else {
            return String(format: "%.1f", triggerValue)
        }
    }

    /// Get the trigger definition for this log
    var definition: TriggerDefinition? {
        getTriggerDefinition(id: triggerName ?? "")
    }

    /// Icon for this trigger (from definition or category)
    var icon: String {
        definition?.icon ?? category.icon
    }

    // MARK: - Factory Methods

    /// Create a new TriggerLog from a TriggerDefinition
    @discardableResult
    static func create(
        from definition: TriggerDefinition,
        value: Double,
        timestamp: Date = Date(),
        notes: String? = nil,
        symptomLog: SymptomLog? = nil,
        in context: NSManagedObjectContext
    ) -> TriggerLog {
        let log = TriggerLog(context: context)
        log.id = UUID()
        log.timestamp = timestamp
        log.triggerCategory = definition.category.rawValue
        log.triggerName = definition.id
        log.triggerValue = value
        log.triggerUnit = definition.unit
        log.isBinary = definition.isBinary
        log.source = definition.dataSource.rawValue
        log.confidence = 1.0
        log.notes = notes
        log.symptomLog = symptomLog
        return log
    }

    /// Create a new TriggerLog with explicit parameters
    @discardableResult
    static func create(
        name: String,
        category: TriggerCategory,
        value: Double,
        unit: String? = nil,
        isBinary: Bool = false,
        source: TriggerDataSource = .manual,
        timestamp: Date = Date(),
        notes: String? = nil,
        symptomLog: SymptomLog? = nil,
        in context: NSManagedObjectContext
    ) -> TriggerLog {
        let log = TriggerLog(context: context)
        log.id = UUID()
        log.timestamp = timestamp
        log.triggerCategory = category.rawValue
        log.triggerName = name
        log.triggerValue = value
        log.triggerUnit = unit
        log.isBinary = isBinary
        log.source = source.rawValue
        log.confidence = 1.0
        log.notes = notes
        log.symptomLog = symptomLog
        return log
    }

    // MARK: - Conversion

    /// Convert to TriggerValue struct
    func toTriggerValue() -> TriggerValue {
        TriggerValue(
            id: id ?? UUID(),
            name: triggerName ?? "unknown",
            category: category,
            value: triggerValue,
            unit: triggerUnit,
            timestamp: timestamp ?? Date()
        )
    }
}

// MARK: - TriggerLog Fetch Requests

extension TriggerLog {

    /// Fetch all trigger logs
    static func fetchAll(in context: NSManagedObjectContext) -> [TriggerLog] {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerLog.timestamp, ascending: false)]
        return (try? context.fetch(request)) ?? []
    }

    /// Fetch trigger logs for a date range
    static func fetch(
        from startDate: Date,
        to endDate: Date,
        in context: NSManagedObjectContext
    ) -> [TriggerLog] {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp <= %@",
            startDate as NSDate,
            endDate as NSDate
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerLog.timestamp, ascending: true)]
        return (try? context.fetch(request)) ?? []
    }

    /// Fetch trigger logs for a specific trigger name
    static func fetch(
        triggerName: String,
        from startDate: Date? = nil,
        to endDate: Date? = nil,
        in context: NSManagedObjectContext
    ) -> [TriggerLog] {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()

        var predicates = [NSPredicate(format: "triggerName == %@", triggerName)]

        if let start = startDate {
            predicates.append(NSPredicate(format: "timestamp >= %@", start as NSDate))
        }
        if let end = endDate {
            predicates.append(NSPredicate(format: "timestamp <= %@", end as NSDate))
        }

        request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerLog.timestamp, ascending: true)]

        return (try? context.fetch(request)) ?? []
    }

    /// Fetch trigger logs for a category
    static func fetch(
        category: TriggerCategory,
        from startDate: Date? = nil,
        to endDate: Date? = nil,
        in context: NSManagedObjectContext
    ) -> [TriggerLog] {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()

        var predicates = [NSPredicate(format: "triggerCategory == %@", category.rawValue)]

        if let start = startDate {
            predicates.append(NSPredicate(format: "timestamp >= %@", start as NSDate))
        }
        if let end = endDate {
            predicates.append(NSPredicate(format: "timestamp <= %@", end as NSDate))
        }

        request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerLog.timestamp, ascending: true)]

        return (try? context.fetch(request)) ?? []
    }

    /// Fetch trigger logs for a specific date
    static func fetch(
        forDate date: Date,
        in context: NSManagedObjectContext
    ) -> [TriggerLog] {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay) ?? date

        return fetch(from: startOfDay, to: endOfDay, in: context)
    }

    /// Fetch the latest N trigger logs
    static func fetchLatest(
        limit: Int,
        in context: NSManagedObjectContext
    ) -> [TriggerLog] {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerLog.timestamp, ascending: false)]
        request.fetchLimit = limit
        return (try? context.fetch(request)) ?? []
    }

    /// Count trigger logs for a trigger name
    static func count(
        triggerName: String,
        where valueCondition: NSPredicate? = nil,
        in context: NSManagedObjectContext
    ) -> Int {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()

        var predicates = [NSPredicate(format: "triggerName == %@", triggerName)]
        if let condition = valueCondition {
            predicates.append(condition)
        }

        request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        return (try? context.count(for: request)) ?? 0
    }

    /// Count days with this trigger present
    static func countDaysWithTrigger(
        triggerName: String,
        from startDate: Date,
        to endDate: Date,
        in context: NSManagedObjectContext
    ) -> Int {
        let logs = fetch(triggerName: triggerName, from: startDate, to: endDate, in: context)
            .filter { $0.isPresent }

        // Group by date and count unique days
        let calendar = Calendar.current
        let uniqueDays = Set(logs.compactMap { log in
            log.timestamp.map { calendar.startOfDay(for: $0) }
        })

        return uniqueDays.count
    }

    /// Get all unique trigger names that have been logged
    static func uniqueTriggerNames(in context: NSManagedObjectContext) -> [String] {
        let request: NSFetchRequest<NSDictionary> = NSFetchRequest(entityName: "TriggerLog")
        request.resultType = .dictionaryResultType
        request.propertiesToFetch = ["triggerName"]
        request.returnsDistinctResults = true

        guard let results = try? context.fetch(request) else { return [] }

        return results.compactMap { $0["triggerName"] as? String }
    }
}

// MARK: - TriggerLog Statistics

extension TriggerLog {

    /// Get aggregate statistics for a trigger
    static func statistics(
        for triggerName: String,
        from startDate: Date,
        to endDate: Date,
        in context: NSManagedObjectContext
    ) -> TriggerStatistics {
        let logs = fetch(triggerName: triggerName, from: startDate, to: endDate, in: context)

        let values = logs.map { $0.triggerValue }
        let presentLogs = logs.filter { $0.isPresent }

        let calendar = Calendar.current
        let totalDays = calendar.dateComponents([.day], from: startDate, to: endDate).day ?? 0
        let daysWithData = Set(logs.compactMap { $0.timestamp.map { calendar.startOfDay(for: $0) } }).count
        let daysPresent = Set(presentLogs.compactMap { $0.timestamp.map { calendar.startOfDay(for: $0) } }).count

        return TriggerStatistics(
            triggerName: triggerName,
            totalLogs: logs.count,
            totalDays: totalDays,
            daysWithData: daysWithData,
            daysPresent: daysPresent,
            mean: values.mean(),
            standardDeviation: values.standardDeviation(),
            min: values.min() ?? 0,
            max: values.max() ?? 0,
            frequency: totalDays > 0 ? Double(daysPresent) / Double(totalDays) : 0
        )
    }
}

/// Statistics summary for a trigger
public struct TriggerStatistics {
    public let triggerName: String
    public let totalLogs: Int
    public let totalDays: Int
    public let daysWithData: Int
    public let daysPresent: Int
    public let mean: Double
    public let standardDeviation: Double
    public let min: Double
    public let max: Double
    public let frequency: Double  // daysPresent / totalDays

    public var frequencyDescription: String {
        "\(Int(frequency * 100))% of days"
    }

    public var dataCompleteness: Double {
        totalDays > 0 ? Double(daysWithData) / Double(totalDays) : 0
    }
}

// MARK: - TriggerLog Cleanup

extension TriggerLog {

    /// Delete all trigger logs for a specific trigger name
    static func deleteAll(
        triggerName: String,
        in context: NSManagedObjectContext
    ) throws {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()
        request.predicate = NSPredicate(format: "triggerName == %@", triggerName)

        let logs = try context.fetch(request)
        for log in logs {
            context.delete(log)
        }
    }

    /// Delete trigger logs older than a date
    static func deleteOlderThan(
        date: Date,
        in context: NSManagedObjectContext
    ) throws {
        let request: NSFetchRequest<TriggerLog> = TriggerLog.fetchRequest()
        request.predicate = NSPredicate(format: "timestamp < %@", date as NSDate)

        let logs = try context.fetch(request)
        for log in logs {
            context.delete(log)
        }
    }
}
