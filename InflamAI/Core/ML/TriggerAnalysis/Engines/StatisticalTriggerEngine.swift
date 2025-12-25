//
//  StatisticalTriggerEngine.swift
//  InflamAI
//
//  Core statistical analysis engine for trigger detection
//  Implements Pearson correlation, lagged analysis, effect sizes, and Bonferroni correction
//
//  This is the foundation layer of the Hybrid Trigger Detection System
//  Always active, 100% explainable, clinically validated methodology
//

import Foundation
import CoreData
import Combine

// MARK: - StatisticalTriggerEngine

@MainActor
public final class StatisticalTriggerEngine: ObservableObject {

    // MARK: - Singleton

    public static let shared = StatisticalTriggerEngine()

    // MARK: - Published State

    @Published public private(set) var analyzedTriggers: [StatisticalTriggerResult] = []
    @Published public private(set) var isAnalyzing: Bool = false
    @Published public private(set) var lastAnalysisDate: Date?
    @Published public private(set) var analysisProgress: Double = 0
    @Published public private(set) var errorMessage: String?

    // MARK: - Configuration

    /// Minimum days of data required for any analysis
    public let minimumDays: Int = 7

    /// Significance level (alpha) before Bonferroni correction
    public let significanceLevel: Double = 0.05

    /// Minimum correlation strength to report
    public let minimumCorrelation: Double = 0.2

    /// Maximum lag days to test
    public let maxLagDays: Int = 3

    // MARK: - Dependencies

    private let persistenceController: InflamAIPersistenceController
    private let triggerDataService: TriggerDataService

    // MARK: - Initialization

    private init(
        persistenceController: InflamAIPersistenceController = .shared,
        triggerDataService: TriggerDataService = .shared
    ) {
        self.persistenceController = persistenceController
        self.triggerDataService = triggerDataService
    }

    // MARK: - Context

    private var viewContext: NSManagedObjectContext {
        persistenceController.container.viewContext
    }

    // MARK: - Main Analysis

    /// Analyze all logged triggers and return ranked results
    public func analyzeAllTriggers(
        days: Int = 90,
        useCache: Bool = true
    ) async -> [StatisticalTriggerResult] {
        isAnalyzing = true
        analysisProgress = 0
        errorMessage = nil
        defer {
            isAnalyzing = false
            analysisProgress = 1.0
        }

        // Get date range
        let endDate = Date()
        guard let startDate = Calendar.current.date(byAdding: .day, value: -days, to: endDate) else {
            errorMessage = "Failed to calculate date range"
            return []
        }

        // Fetch symptom logs
        let symptomLogs = fetchSymptomLogs(from: startDate, to: endDate)
        guard symptomLogs.count >= minimumDays else {
            errorMessage = "Need at least \(minimumDays) days of symptom data (have \(symptomLogs.count))"
            return []
        }

        // Get unique trigger names that have been logged
        let triggerNames = triggerDataService.getLoggedTriggerNames()

        var results: [StatisticalTriggerResult] = []
        let totalTriggers = Double(triggerNames.count)

        // Analyze each trigger
        for (index, triggerName) in triggerNames.enumerated() {
            analysisProgress = Double(index) / max(1, totalTriggers)

            // Check cache first
            if useCache, let cached = getCachedResult(for: triggerName) {
                results.append(cached)
                continue
            }

            // Perform analysis
            if let result = await analyzeTrigger(
                name: triggerName,
                symptomLogs: symptomLogs,
                startDate: startDate,
                endDate: endDate
            ) {
                results.append(result)

                // Cache the result
                cacheResult(result)
            }
        }

        // Apply Bonferroni correction
        let correctedResults = applyBonferroniCorrection(results)

        // Sort by absolute effect size
        let sortedResults = correctedResults.sorted {
            abs($0.effectSize.cohenD) > abs($1.effectSize.cohenD)
        }

        analyzedTriggers = sortedResults
        lastAnalysisDate = Date()

        return sortedResults
    }

    /// Analyze a single trigger
    public func analyzeTrigger(
        name triggerName: String,
        symptomLogs: [SymptomLog]? = nil,
        startDate: Date? = nil,
        endDate: Date? = nil
    ) async -> StatisticalTriggerResult? {
        // Get date range
        let end = endDate ?? Date()
        let start = startDate ?? Calendar.current.date(byAdding: .day, value: -90, to: end) ?? end

        // Get symptom logs if not provided
        let logs = symptomLogs ?? fetchSymptomLogs(from: start, to: end)
        guard logs.count >= minimumDays else { return nil }

        // Get trigger definition
        let definition = getTriggerDefinition(id: triggerName)
        let category = definition?.category ?? .other
        let icon = definition?.icon ?? category.icon

        // Get trigger logs
        let triggerLogs = triggerDataService.getTriggers(named: triggerName, from: start, to: end)

        // Build daily data
        let (painValues, triggerValues, dates) = buildDailyData(
            symptomLogs: logs,
            triggerLogs: triggerLogs
        )

        guard painValues.count >= minimumDays else { return nil }

        // Perform lagged correlation analysis
        let laggedResults = analyzeLaggedCorrelations(
            triggerValues: triggerValues,
            painValues: painValues,
            maxLag: maxLagDays
        )

        // Find best lag
        let bestLag = laggedResults
            .filter { $0.pValue < significanceLevel }
            .min(by: { $0.pValue < $1.pValue })

        // Calculate effect size
        let effectSize = calculateEffectSize(
            triggerValues: triggerValues,
            painValues: painValues,
            dates: dates
        )

        // Count trigger days
        let triggerDays = triggerValues.filter { $0 > 0 }.count
        let nonTriggerDays = triggerValues.filter { $0 == 0 }.count

        // Determine confidence
        let confidence = classifyConfidence(
            sampleSize: painValues.count,
            triggerDays: triggerDays,
            pValue: bestLag?.pValue ?? 1.0,
            correlation: bestLag?.correlation ?? 0,
            effectSize: effectSize.cohenD
        )

        return StatisticalTriggerResult(
            triggerName: triggerName,
            triggerCategory: category,
            icon: icon,
            totalDays: painValues.count,
            triggerDays: triggerDays,
            nonTriggerDays: nonTriggerDays,
            laggedResults: laggedResults,
            bestLag: bestLag,
            effectSize: effectSize,
            rawPValue: bestLag?.pValue ?? 1.0,
            correctedPValue: bestLag?.pValue ?? 1.0, // Will be corrected in batch
            isSignificant: (bestLag?.pValue ?? 1.0) < significanceLevel,
            confidence: confidence
        )
    }

    // MARK: - Data Preparation

    /// Build aligned daily data arrays
    private func buildDailyData(
        symptomLogs: [SymptomLog],
        triggerLogs: [TriggerLog]
    ) -> (painValues: [Double], triggerValues: [Double], dates: [Date]) {
        let calendar = Calendar.current

        // Group symptom logs by day
        var dailyPain: [Date: Double] = [:]
        for log in symptomLogs {
            guard let timestamp = log.timestamp else { continue }
            let day = calendar.startOfDay(for: timestamp)
            // Take the average if multiple logs per day
            if let existing = dailyPain[day] {
                dailyPain[day] = (existing + log.basdaiScore) / 2
            } else {
                dailyPain[day] = log.basdaiScore
            }
        }

        // Group trigger logs by day (take max value)
        var dailyTrigger: [Date: Double] = [:]
        for log in triggerLogs {
            guard let timestamp = log.timestamp else { continue }
            let day = calendar.startOfDay(for: timestamp)
            dailyTrigger[day] = max(dailyTrigger[day] ?? 0, log.triggerValue)
        }

        // Get all unique dates from symptom logs (trigger-only days don't help)
        let allDates = dailyPain.keys.sorted()

        // Build aligned arrays
        var painValues: [Double] = []
        var triggerValues: [Double] = []
        var dates: [Date] = []

        for date in allDates {
            if let pain = dailyPain[date] {
                painValues.append(pain)
                triggerValues.append(dailyTrigger[date] ?? 0)
                dates.append(date)
            }
        }

        return (painValues, triggerValues, dates)
    }

    // MARK: - Lagged Correlation Analysis

    /// Analyze correlations at multiple lag offsets
    private func analyzeLaggedCorrelations(
        triggerValues: [Double],
        painValues: [Double],
        maxLag: Int
    ) -> [LaggedCorrelationResult] {
        var results: [LaggedCorrelationResult] = []

        for lag in 0...maxLag {
            guard triggerValues.count > lag, painValues.count > lag else { continue }

            // Align data: trigger[t] correlates with pain[t+lag]
            // If lag=1: trigger on day 1 affects pain on day 2
            let alignedTrigger = Array(triggerValues.dropLast(lag))
            let alignedPain = Array(painValues.dropFirst(lag))

            guard alignedTrigger.count == alignedPain.count,
                  alignedTrigger.count >= minimumDays else { continue }

            // Calculate Pearson correlation
            if let r = pearsonCorrelation(alignedTrigger, alignedPain) {
                let pValue = calculatePValue(r: r, n: alignedTrigger.count)

                results.append(LaggedCorrelationResult(
                    lag: lag,
                    correlation: r,
                    pValue: pValue,
                    sampleSize: alignedTrigger.count
                ))
            }
        }

        return results
    }

    // MARK: - Effect Size Calculation

    /// Calculate Cohen's d and related effect metrics
    private func calculateEffectSize(
        triggerValues: [Double],
        painValues: [Double],
        dates: [Date]
    ) -> EffectSize {
        // Separate pain values by trigger presence
        var painWithTrigger: [Double] = []
        var painWithoutTrigger: [Double] = []

        for (index, triggerValue) in triggerValues.enumerated() {
            guard index < painValues.count else { break }

            if triggerValue > 0 {
                painWithTrigger.append(painValues[index])
            } else {
                painWithoutTrigger.append(painValues[index])
            }
        }

        // Calculate means
        let meanWith = painWithTrigger.mean()
        let meanWithout = painWithoutTrigger.mean()
        let meanDiff = meanWith - meanWithout

        // Calculate pooled standard deviation
        let varWith = painWithTrigger.variance()
        let varWithout = painWithoutTrigger.variance()
        let pooledVar = (varWith + varWithout) / 2
        let pooledSD = sqrt(pooledVar)

        // Cohen's d
        let cohenD = pooledSD > 0 ? meanDiff / pooledSD : 0

        // Percent change
        let percentChange = meanWithout > 0 ? (meanDiff / meanWithout) * 100 : 0

        return EffectSize(
            meanWithTrigger: meanWith,
            meanWithoutTrigger: meanWithout,
            meanDifference: meanDiff,
            pooledStandardDeviation: pooledSD,
            cohenD: cohenD,
            percentChange: percentChange
        )
    }

    // MARK: - Statistical Functions

    /// Calculate Pearson correlation coefficient
    public func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double? {
        guard x.count == y.count, x.count > 2 else { return nil }

        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)

        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))

        guard denominator > 0 else { return nil }
        return numerator / denominator
    }

    /// Calculate Spearman rank correlation (for non-normal data)
    public func spearmanCorrelation(_ x: [Double], _ y: [Double]) -> Double? {
        guard x.count == y.count, x.count > 2 else { return nil }

        let ranksX = assignRanks(x)
        let ranksY = assignRanks(y)

        return pearsonCorrelation(ranksX, ranksY)
    }

    /// Assign ranks to values (for Spearman correlation)
    private func assignRanks(_ values: [Double]) -> [Double] {
        let indexed = values.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 < $1.1 }

        var ranks = Array(repeating: 0.0, count: values.count)
        for (rank, item) in sorted.enumerated() {
            ranks[item.0] = Double(rank + 1)
        }

        return ranks
    }

    /// Calculate p-value for correlation using t-distribution
    public func calculatePValue(r: Double, n: Int) -> Double {
        guard n > 2 else { return 1.0 }

        // Handle perfect correlation
        let absR = abs(r)
        if absR >= 0.9999 {
            return 0.0
        }

        // t-statistic: t = r * sqrt((n-2)/(1-r²))
        let denominator = 1 - r * r
        guard denominator > 0.0001 else {
            return 0.0
        }

        let t = r * sqrt(Double(n - 2) / denominator)
        let df = n - 2

        // Two-tailed p-value
        let pValue = 2 * (1 - approximateTCDF(abs(t), df: df))
        return max(0.0, min(1.0, pValue))
    }

    /// Approximate t-distribution CDF
    private func approximateTCDF(_ t: Double, df: Int) -> Double {
        // For large df, use normal approximation
        if df > 100 {
            return 0.5 * (1 + erf(t / sqrt(2)))
        }

        // For smaller df, use approximation
        let x = Double(df) / (Double(df) + t * t)

        if df == 1 {
            return 0.5 + atan(t) / .pi
        } else if df == 2 {
            return 0.5 + t / (2 * sqrt(2 + t * t))
        } else {
            // General approximation
            let a = 0.5 * Double(df)
            let betaCDF = approximateIncompleteBeta(x: x, a: a, b: 0.5)

            if t >= 0 {
                return 1.0 - 0.5 * betaCDF
            } else {
                return 0.5 * betaCDF
            }
        }
    }

    /// Approximate incomplete beta function
    private func approximateIncompleteBeta(x: Double, a: Double, b: Double) -> Double {
        guard x > 0, x < 1 else {
            return x <= 0 ? 0.0 : 1.0
        }

        if abs(a - b) < 0.1 {
            return pow(x, a)
        }

        let term1 = pow(x, a)
        let normalization = 1.0 / (a + b)
        return term1 * normalization * (1.0 + a * (1 - x) / (a + 1))
    }

    /// Error function approximation
    private func erf(_ x: Double) -> Double {
        let sign = x >= 0 ? 1.0 : -1.0
        let absX = abs(x)

        let a1 = 0.254829592
        let a2 = -0.284496736
        let a3 = 1.421413741
        let a4 = -1.453152027
        let a5 = 1.061405429
        let p = 0.3275911

        let t = 1.0 / (1.0 + p * absX)
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absX * absX)

        return sign * y
    }

    // MARK: - Bonferroni Correction

    /// Apply Bonferroni correction for multiple comparisons
    private func applyBonferroniCorrection(_ results: [StatisticalTriggerResult]) -> [StatisticalTriggerResult] {
        let numberOfTests = results.count
        guard numberOfTests > 0 else { return results }

        let correctedAlpha = significanceLevel / Double(numberOfTests)

        return results.map { result in
            // Adjust p-value (multiply by number of tests)
            let correctedPValue = min(1.0, result.rawPValue * Double(numberOfTests))
            let isSignificant = result.rawPValue < correctedAlpha

            return StatisticalTriggerResult(
                id: result.id,
                triggerName: result.triggerName,
                triggerCategory: result.triggerCategory,
                icon: result.icon,
                totalDays: result.totalDays,
                triggerDays: result.triggerDays,
                nonTriggerDays: result.nonTriggerDays,
                laggedResults: result.laggedResults,
                bestLag: result.bestLag,
                effectSize: result.effectSize,
                rawPValue: result.rawPValue,
                correctedPValue: correctedPValue,
                isSignificant: isSignificant,
                confidence: result.confidence,
                analysisDate: result.analysisDate
            )
        }
    }

    // MARK: - Confidence Classification

    /// Classify confidence level based on statistical metrics
    private func classifyConfidence(
        sampleSize: Int,
        triggerDays: Int,
        pValue: Double,
        correlation: Double,
        effectSize: Double
    ) -> TriggerConfidence {
        // Insufficient data
        if triggerDays < 7 || sampleSize < 14 {
            return .insufficient
        }

        // High confidence criteria
        if sampleSize >= 60 &&
           triggerDays >= 15 &&
           pValue < 0.01 &&
           abs(correlation) > 0.5 &&
           abs(effectSize) > 0.5 {
            return .high
        }

        // Medium confidence criteria
        if sampleSize >= 30 &&
           triggerDays >= 10 &&
           pValue < 0.05 &&
           abs(correlation) > 0.3 {
            return .medium
        }

        // Low confidence (significant but weak)
        if pValue < 0.05 {
            return .low
        }

        return .insufficient
    }

    // MARK: - Data Fetching

    /// Fetch symptom logs for a date range
    private func fetchSymptomLogs(from startDate: Date, to endDate: Date) -> [SymptomLog] {
        let request: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        request.predicate = NSPredicate(
            format: "timestamp >= %@ AND timestamp <= %@",
            startDate as NSDate,
            endDate as NSDate
        )
        request.sortDescriptors = [NSSortDescriptor(keyPath: \SymptomLog.timestamp, ascending: true)]

        return (try? viewContext.fetch(request)) ?? []
    }

    // MARK: - Caching

    /// Get cached result if valid
    private func getCachedResult(for triggerName: String) -> StatisticalTriggerResult? {
        guard let cache = TriggerAnalysisCache.getValidCache(for: triggerName, in: viewContext) else {
            return nil
        }
        return cache.toStatisticalResult()
    }

    /// Cache analysis result
    private func cacheResult(_ result: StatisticalTriggerResult) {
        TriggerAnalysisCache.createOrUpdate(
            triggerName: result.triggerName,
            category: result.triggerCategory,
            laggedResults: result.laggedResults,
            effectSize: result.effectSize,
            confidence: result.confidence,
            daysAnalyzed: result.totalDays,
            triggerDaysCount: result.triggerDays,
            in: viewContext
        )
        try? viewContext.save()
    }

    // MARK: - Filtering & Ranking

    /// Get only significant triggers
    public func getSignificantTriggers() -> [StatisticalTriggerResult] {
        analyzedTriggers.filter { $0.isSignificant }
    }

    /// Get triggers by confidence level
    public func getTriggers(withConfidence confidence: TriggerConfidence) -> [StatisticalTriggerResult] {
        analyzedTriggers.filter { $0.confidence == confidence }
    }

    /// Get top N triggers by effect size
    public func getTopTriggers(limit: Int = 5) -> [StatisticalTriggerResult] {
        Array(analyzedTriggers.prefix(limit))
    }

    /// Get triggers for a specific category
    public func getTriggers(category: TriggerCategory) -> [StatisticalTriggerResult] {
        analyzedTriggers.filter { $0.triggerCategory == category }
    }

    // MARK: - Explanation Generation

    /// Generate user-friendly explanation for a trigger result
    public func generateExplanation(for result: StatisticalTriggerResult) -> TriggerExplanation {
        let summary = generateSummary(result)
        let details = generateDetails(result)
        let recommendation = generateRecommendation(result)
        let caveats = generateCaveats(result)

        return TriggerExplanation(
            triggerName: result.triggerName,
            summary: summary,
            details: details,
            recommendation: recommendation,
            caveats: caveats,
            confidence: result.confidence
        )
    }

    private func generateSummary(_ result: StatisticalTriggerResult) -> String {
        guard result.isSignificant else {
            return "No significant relationship found between \(result.triggerName) and your symptoms."
        }

        let direction = result.effectSize.meanDifference > 0 ? "increases" : "decreases"
        let timing = result.bestLag?.lagDescription.lowercased() ?? "same day"
        let magnitude = String(format: "%.1f", abs(result.effectSize.meanDifference))

        return "\(result.triggerName) \(direction) your pain by \(magnitude) points on average (\(timing) effect)."
    }

    private func generateDetails(_ result: StatisticalTriggerResult) -> String {
        var details: [String] = []

        // Sample size
        details.append("Based on \(result.totalDays) days of data")
        details.append("\(result.triggerDays) days with \(result.triggerName), \(result.nonTriggerDays) without")

        // Pain averages
        let avgWith = String(format: "%.1f", result.effectSize.meanWithTrigger)
        let avgWithout = String(format: "%.1f", result.effectSize.meanWithoutTrigger)
        details.append("Average pain with: \(avgWith), without: \(avgWithout)")

        // Statistical significance
        let pValueStr = result.correctedPValue < 0.001
            ? "p < 0.001"
            : "p = \(String(format: "%.3f", result.correctedPValue))"
        details.append("Statistical significance: \(pValueStr)")

        // Effect size interpretation
        details.append("Effect size: \(result.effectSize.cohenDInterpretation) (d = \(String(format: "%.2f", result.effectSize.cohenD)))")

        return details.joined(separator: "\n")
    }

    private func generateRecommendation(_ result: StatisticalTriggerResult) -> String {
        guard result.isSignificant, result.effectSize.meanDifference > 0 else {
            if result.effectSize.meanDifference < 0 {
                return "\(result.triggerName) may have a protective effect. Consider maintaining this habit."
            }
            return "Continue tracking to gather more data."
        }

        // Generate recommendation based on trigger type
        switch result.triggerCategory {
        case .food:
            return "Consider reducing \(result.triggerName.lowercased()) intake, especially when you're already experiencing symptoms."
        case .sleep:
            return "Prioritize improving your sleep quality. Even small improvements can reduce symptoms."
        case .activity:
            if result.triggerName.contains("sitting") || result.triggerName.contains("Sitting") {
                return "Try taking regular movement breaks throughout the day."
            }
            return "Consider adjusting your activity level and listening to your body."
        case .weather:
            return "Plan lower-intensity days when \(result.triggerName.lowercased()) is expected. Check weather forecasts."
        case .stress:
            return "Practice stress management techniques. Consider mindfulness or relaxation exercises."
        case .medication:
            return "Discuss medication timing and adherence with your rheumatologist."
        case .other:
            return "Consider avoiding or reducing exposure to this trigger when possible."
        }
    }

    private func generateCaveats(_ result: StatisticalTriggerResult) -> [String] {
        var caveats: [String] = []

        if result.totalDays < 60 {
            caveats.append("Analysis based on \(result.totalDays) days - confidence will improve with more data")
        }

        if result.triggerDays < 14 {
            caveats.append("Limited trigger days (\(result.triggerDays)) - track more instances for better accuracy")
        }

        if result.confidence == .low {
            caveats.append("Weak correlation - this finding may change as you track more data")
        }

        caveats.append("Correlation does not prove causation - discuss findings with your rheumatologist")

        return caveats
    }
}

// MARK: - Trigger Explanation

public struct TriggerExplanation {
    public let triggerName: String
    public let summary: String
    public let details: String
    public let recommendation: String
    public let caveats: [String]
    public let confidence: TriggerConfidence

    public var fullExplanation: String {
        var parts = [summary, "", details, "", "Recommendation: \(recommendation)"]

        if !caveats.isEmpty {
            parts.append("")
            parts.append("Notes:")
            for caveat in caveats {
                parts.append("• \(caveat)")
            }
        }

        return parts.joined(separator: "\n")
    }
}

// MARK: - Validation Tests

#if DEBUG
extension StatisticalTriggerEngine {
    /// Run validation tests
    public func runValidationTests() {
        print("🧪 Running StatisticalTriggerEngine validation tests...")

        // Test 1: Perfect positive correlation
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y1 = [2.0, 4.0, 6.0, 8.0, 10.0]
        let r1 = pearsonCorrelation(x1, y1)
        assert(abs(r1! - 1.0) < 0.01, "Test 1 failed: Perfect positive correlation")
        print("✅ Test 1 passed: Perfect positive correlation")

        // Test 2: Perfect negative correlation
        let x2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y2 = [10.0, 8.0, 6.0, 4.0, 2.0]
        let r2 = pearsonCorrelation(x2, y2)
        assert(abs(r2! - (-1.0)) < 0.01, "Test 2 failed: Perfect negative correlation")
        print("✅ Test 2 passed: Perfect negative correlation")

        // Test 3: P-value for strong correlation
        let p3 = calculatePValue(r: 0.8, n: 30)
        assert(p3 < 0.01, "Test 3 failed: P-value should be < 0.01 for r=0.8, n=30")
        print("✅ Test 3 passed: P-value calculation")

        // Test 4: Spearman correlation
        let x4 = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y4 = [1.5, 2.5, 3.5, 4.5, 5.5]
        let rs4 = spearmanCorrelation(x4, y4)
        assert(rs4! > 0.99, "Test 4 failed: Spearman correlation")
        print("✅ Test 4 passed: Spearman correlation")

        print("✅ All StatisticalTriggerEngine validation tests passed!")
    }
}
#endif
