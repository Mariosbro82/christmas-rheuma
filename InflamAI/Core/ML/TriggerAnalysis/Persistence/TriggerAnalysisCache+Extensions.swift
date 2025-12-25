//
//  TriggerAnalysisCache+Extensions.swift
//  InflamAI
//
//  Convenience methods for TriggerAnalysisCache Core Data entity
//  Caches expensive statistical analysis results
//

import Foundation
import CoreData

// MARK: - TriggerAnalysisCache Extensions

extension TriggerAnalysisCache {

    // MARK: - Confidence Accessor

    /// Get confidence as enum
    var confidenceLevel: TriggerConfidence {
        TriggerConfidence(rawValue: confidence ?? "insufficient") ?? .insufficient
    }

    /// Set confidence from enum
    func setConfidence(_ level: TriggerConfidence) {
        self.confidence = level.rawValue
    }

    // MARK: - Category Accessor

    /// Get category as enum
    var category: TriggerCategory {
        TriggerCategory(rawValue: triggerCategory ?? "other") ?? .other
    }

    /// Set category from enum
    func setCategory(_ category: TriggerCategory) {
        self.triggerCategory = category.rawValue
    }

    // MARK: - Computed Properties

    /// Whether this cache is still valid and not expired
    var isStillValid: Bool {
        guard isValid else { return false }

        if let expires = expiresAt, Date() > expires {
            return false
        }

        return true
    }

    /// Get decoded lagged correlation results
    var laggedResults: [LaggedCorrelationResult]? {
        guard let data = correlationData else { return nil }
        return try? JSONDecoder().decode([LaggedCorrelationResult].self, from: data)
    }

    /// Get decoded effect size
    var effectSizeResult: EffectSize? {
        guard let data = effectSizeData else { return nil }
        return try? JSONDecoder().decode(EffectSize.self, from: data)
    }

    // MARK: - Factory Methods

    /// Create or update cache for a trigger analysis
    @discardableResult
    static func createOrUpdate(
        triggerName: String,
        category: TriggerCategory,
        laggedResults: [LaggedCorrelationResult],
        effectSize: EffectSize,
        confidence: TriggerConfidence,
        daysAnalyzed: Int,
        triggerDaysCount: Int,
        in context: NSManagedObjectContext
    ) -> TriggerAnalysisCache {
        // Try to find existing cache
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(format: "triggerName == %@", triggerName)
        request.fetchLimit = 1

        let cache: TriggerAnalysisCache
        if let existing = try? context.fetch(request).first {
            cache = existing
        } else {
            cache = TriggerAnalysisCache(context: context)
            cache.id = UUID()
            cache.triggerName = triggerName
        }

        // Update fields
        cache.analysisDate = Date()
        cache.triggerCategory = category.rawValue
        cache.confidence = confidence.rawValue
        cache.daysAnalyzed = Int32(daysAnalyzed)
        cache.triggerDaysCount = Int32(triggerDaysCount)
        cache.isValid = true

        // Set expiration (cache valid for 24 hours or until new data logged)
        cache.expiresAt = Calendar.current.date(byAdding: .hour, value: 24, to: Date())

        // Best lag result
        if let bestLag = laggedResults.min(by: { $0.pValue < $1.pValue }) {
            cache.bestLagDays = Int16(bestLag.lag)
            cache.bestCorrelation = bestLag.correlation
            cache.bestPValue = bestLag.pValue
        }

        // Effect size
        cache.meanDifference = effectSize.meanDifference
        cache.cohenD = effectSize.cohenD

        // Encode complex data
        cache.correlationData = try? JSONEncoder().encode(laggedResults)
        cache.effectSizeData = try? JSONEncoder().encode(effectSize)

        return cache
    }

    // MARK: - Fetch Methods

    /// Get valid cache for a trigger
    static func getValidCache(
        for triggerName: String,
        in context: NSManagedObjectContext
    ) -> TriggerAnalysisCache? {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(
            format: "triggerName == %@ AND isValid == YES",
            triggerName
        )
        request.fetchLimit = 1
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerAnalysisCache.analysisDate, ascending: false)]

        guard let cache = try? context.fetch(request).first,
              cache.isStillValid else {
            return nil
        }

        return cache
    }

    /// Get all valid caches
    static func getAllValidCaches(in context: NSManagedObjectContext) -> [TriggerAnalysisCache] {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(format: "isValid == YES")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerAnalysisCache.analysisDate, ascending: false)]

        let results = (try? context.fetch(request)) ?? []
        return results.filter { $0.isStillValid }
    }

    /// Get caches for a category
    static func getCaches(
        for category: TriggerCategory,
        in context: NSManagedObjectContext
    ) -> [TriggerAnalysisCache] {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(
            format: "triggerCategory == %@ AND isValid == YES",
            category.rawValue
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \TriggerAnalysisCache.analysisDate, ascending: false)]

        let results = (try? context.fetch(request)) ?? []
        return results.filter { $0.isStillValid }
    }

    // MARK: - Invalidation

    /// Invalidate cache for a trigger
    static func invalidate(
        triggerName: String,
        in context: NSManagedObjectContext
    ) {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(format: "triggerName == %@", triggerName)

        if let caches = try? context.fetch(request) {
            for cache in caches {
                cache.isValid = false
            }
        }
    }

    /// Invalidate all caches
    static func invalidateAll(in context: NSManagedObjectContext) {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()

        if let caches = try? context.fetch(request) {
            for cache in caches {
                cache.isValid = false
            }
        }
    }

    // MARK: - Cleanup

    /// Delete expired caches
    static func deleteExpired(in context: NSManagedObjectContext) throws {
        let request: NSFetchRequest<TriggerAnalysisCache> = TriggerAnalysisCache.fetchRequest()
        request.predicate = NSPredicate(
            format: "expiresAt < %@ OR isValid == NO",
            Date() as NSDate
        )

        let caches = try context.fetch(request)
        for cache in caches {
            context.delete(cache)
        }
    }

    // MARK: - Conversion

    /// Convert to StatisticalTriggerResult
    func toStatisticalResult() -> StatisticalTriggerResult? {
        guard let lagResults = laggedResults,
              let effect = effectSizeResult else {
            return nil
        }

        let bestLag = lagResults.min(by: { $0.pValue < $1.pValue })

        return StatisticalTriggerResult(
            triggerName: triggerName ?? "unknown",
            triggerCategory: category,
            icon: getTriggerDefinition(id: triggerName ?? "")?.icon ?? category.icon,
            totalDays: Int(daysAnalyzed),
            triggerDays: Int(triggerDaysCount),
            nonTriggerDays: Int(daysAnalyzed) - Int(triggerDaysCount),
            laggedResults: lagResults,
            bestLag: bestLag,
            effectSize: effect,
            rawPValue: bestPValue,
            correctedPValue: bestPValue, // Will be corrected by analysis engine
            isSignificant: bestPValue < 0.05,
            confidence: confidenceLevel,
            analysisDate: analysisDate ?? Date()
        )
    }
}
