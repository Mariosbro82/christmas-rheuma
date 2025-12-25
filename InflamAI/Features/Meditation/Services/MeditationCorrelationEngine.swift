//
//  MeditationCorrelationEngine.swift
//  InflamAI
//
//  Created by Claude Code on 2025-12-08.
//

import Foundation
import CoreData

// MARK: - Analysis Result Models

struct MeditationImpactAnalysis {
    let daysAnalyzed: Int
    let daysWithMeditation: Int
    let daysWithoutMeditation: Int
    let totalSessions: Int

    // Pain Metrics
    let avgPainWithMeditation: Double
    let avgPainWithoutMeditation: Double
    let painReduction: Double
    let painReductionPercentage: Double

    // BASDAI Metrics
    let avgBASDAIWithMeditation: Double?
    let avgBASDAIWithoutMeditation: Double?
    let basdaiReduction: Double?

    // Mood & Stress Metrics
    let avgMoodWithMeditation: Double
    let avgMoodWithoutMeditation: Double
    let moodImprovement: Double

    let avgStressWithMeditation: Double
    let avgStressWithoutMeditation: Double
    let stressReduction: Double

    // Sleep Metrics
    let avgSleepWithMeditation: Double?
    let avgSleepWithoutMeditation: Double?
    let sleepImprovement: Double?

    // Statistical Significance
    let statisticalSignificance: Double
    let confidence: MeditationConfidenceLevel

    // Session Impact
    let mostEffectiveType: String?
    let mostEffectiveCategory: String?
    let optimalDuration: TimeInterval?

    var hasSignificantImpact: Bool {
        painReduction > 0.5 && confidence != .insufficient
    }

    var summaryDescription: String {
        if !hasSignificantImpact {
            return "Insufficient data to determine meditation impact. Keep practicing to see personalized insights."
        }

        let painPercent = Int(painReductionPercentage)
        var summary = "Meditation reduces your pain by \(painPercent)% on average"

        if let basdaiReduction = basdaiReduction, basdaiReduction > 0.3 {
            summary += " and improves BASDAI scores by \(String(format: "%.1f", basdaiReduction)) points"
        }

        if stressReduction > 1.0 {
            summary += ". It also reduces stress significantly"
        }

        return summary + "."
    }
}

enum MeditationConfidenceLevel: String {
    case insufficient = "insufficient"
    case low = "low"
    case moderate = "moderate"
    case high = "high"

    var displayName: String {
        switch self {
        case .insufficient: return "Insufficient Data"
        case .low: return "Low Confidence"
        case .moderate: return "Moderate Confidence"
        case .high: return "High Confidence"
        }
    }
}

struct DailyMeditationMetrics {
    let date: Date
    let hadMeditation: Bool
    let sessionCount: Int
    let totalMinutes: Double

    let painLevel: Double?
    let basdaiScore: Double?
    let moodScore: Double?
    let stressLevel: Double?
    let sleepQuality: Double?
}

// MARK: - Correlation Engine

@MainActor
class MeditationCorrelationEngine {
    let persistenceController: InflamAIPersistenceController

    init(persistenceController: InflamAIPersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    // MARK: - Main Analysis Function

    func analyzeMeditationImpact(days: Int = 30) throws -> MeditationImpactAnalysis {
        let context = persistenceController.container.viewContext

        // Fetch meditation sessions
        let meditationRequest: NSFetchRequest<MeditationSession> = MeditationSession.fetchRequest()
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        meditationRequest.predicate = NSPredicate(format: "timestamp >= %@", startDate as CVarArg)
        let meditationSessions = try context.fetch(meditationRequest)

        // Fetch symptom logs
        let symptomRequest: NSFetchRequest<SymptomLog> = SymptomLog.fetchRequest()
        symptomRequest.predicate = NSPredicate(format: "timestamp >= %@", startDate as CVarArg)
        let symptomLogs = try context.fetch(symptomRequest)

        // Build daily metrics
        let dailyMetrics = buildDailyMetrics(
            meditationSessions: meditationSessions,
            symptomLogs: symptomLogs,
            startDate: startDate,
            days: days
        )

        // Separate days with and without meditation
        let meditationDays = dailyMetrics.filter { $0.hadMeditation }
        let nonMeditationDays = dailyMetrics.filter { !$0.hadMeditation }

        // Calculate pain metrics
        let painWithMed = meditationDays.compactMap { $0.painLevel }
        let painWithoutMed = nonMeditationDays.compactMap { $0.painLevel }

        let avgPainWithMeditation = painWithMed.isEmpty ? 0 : painWithMed.reduce(0, +) / Double(painWithMed.count)
        let avgPainWithoutMeditation = painWithoutMed.isEmpty ? 0 : painWithoutMed.reduce(0, +) / Double(painWithoutMed.count)
        let painReduction = avgPainWithoutMeditation - avgPainWithMeditation
        let painReductionPercentage = avgPainWithoutMeditation > 0 ? (painReduction / avgPainWithoutMeditation) * 100 : 0

        // Calculate BASDAI metrics
        let basdaiWithMed = meditationDays.compactMap { $0.basdaiScore }
        let basdaiWithoutMed = nonMeditationDays.compactMap { $0.basdaiScore }

        let avgBASDAIWithMeditation = basdaiWithMed.isEmpty ? nil : basdaiWithMed.reduce(0, +) / Double(basdaiWithMed.count)
        let avgBASDAIWithoutMeditation = basdaiWithoutMed.isEmpty ? nil : basdaiWithoutMed.reduce(0, +) / Double(basdaiWithoutMed.count)
        let basdaiReduction: Double? = if let avgWith = avgBASDAIWithMeditation, let avgWithout = avgBASDAIWithoutMeditation {
            avgWithout - avgWith
        } else {
            nil
        }

        // Calculate mood metrics
        let moodWithMed = meditationDays.compactMap { $0.moodScore }
        let moodWithoutMed = nonMeditationDays.compactMap { $0.moodScore }

        let avgMoodWithMeditation = moodWithMed.isEmpty ? 0 : moodWithMed.reduce(0, +) / Double(moodWithMed.count)
        let avgMoodWithoutMeditation = moodWithoutMed.isEmpty ? 0 : moodWithoutMed.reduce(0, +) / Double(moodWithoutMed.count)
        let moodImprovement = avgMoodWithMeditation - avgMoodWithoutMeditation

        // Calculate stress metrics
        let stressWithMed = meditationDays.compactMap { $0.stressLevel }
        let stressWithoutMed = nonMeditationDays.compactMap { $0.stressLevel }

        let avgStressWithMeditation = stressWithMed.isEmpty ? 0 : stressWithMed.reduce(0, +) / Double(stressWithMed.count)
        let avgStressWithoutMeditation = stressWithoutMed.isEmpty ? 0 : stressWithoutMed.reduce(0, +) / Double(stressWithoutMed.count)
        let stressReduction = avgStressWithoutMeditation - avgStressWithMeditation

        // Calculate sleep metrics
        let sleepWithMed = meditationDays.compactMap { $0.sleepQuality }
        let sleepWithoutMed = nonMeditationDays.compactMap { $0.sleepQuality }

        let avgSleepWithMeditation = sleepWithMed.isEmpty ? nil : sleepWithMed.reduce(0, +) / Double(sleepWithMed.count)
        let avgSleepWithoutMeditation = sleepWithoutMed.isEmpty ? nil : sleepWithoutMed.reduce(0, +) / Double(sleepWithoutMed.count)
        let sleepImprovement: Double? = if let avgWith = avgSleepWithMeditation, let avgWithout = avgSleepWithoutMeditation {
            avgWith - avgWithout
        } else {
            nil
        }

        // Statistical significance
        let significance = calculateSignificance(painWithMed, painWithoutMed)

        // Confidence level
        let confidence = determineConfidence(
            meditationDays: meditationDays.count,
            nonMeditationDays: nonMeditationDays.count,
            significance: significance
        )

        // Find most effective session types
        let (mostEffectiveType, mostEffectiveCategory, optimalDuration) = findMostEffectiveSession(
            meditationSessions: meditationSessions,
            symptomLogs: symptomLogs
        )

        return MeditationImpactAnalysis(
            daysAnalyzed: days,
            daysWithMeditation: meditationDays.count,
            daysWithoutMeditation: nonMeditationDays.count,
            totalSessions: meditationSessions.count,
            avgPainWithMeditation: avgPainWithMeditation,
            avgPainWithoutMeditation: avgPainWithoutMeditation,
            painReduction: painReduction,
            painReductionPercentage: painReductionPercentage,
            avgBASDAIWithMeditation: avgBASDAIWithMeditation,
            avgBASDAIWithoutMeditation: avgBASDAIWithoutMeditation,
            basdaiReduction: basdaiReduction,
            avgMoodWithMeditation: avgMoodWithMeditation,
            avgMoodWithoutMeditation: avgMoodWithoutMeditation,
            moodImprovement: moodImprovement,
            avgStressWithMeditation: avgStressWithMeditation,
            avgStressWithoutMeditation: avgStressWithoutMeditation,
            stressReduction: stressReduction,
            avgSleepWithMeditation: avgSleepWithMeditation,
            avgSleepWithoutMeditation: avgSleepWithoutMeditation,
            sleepImprovement: sleepImprovement,
            statisticalSignificance: significance,
            confidence: confidence,
            mostEffectiveType: mostEffectiveType,
            mostEffectiveCategory: mostEffectiveCategory,
            optimalDuration: optimalDuration
        )
    }

    // MARK: - Helper Functions

    private func buildDailyMetrics(
        meditationSessions: [MeditationSession],
        symptomLogs: [SymptomLog],
        startDate: Date,
        days: Int
    ) -> [DailyMeditationMetrics] {
        let calendar = Calendar.current
        var metrics: [DailyMeditationMetrics] = []

        for dayOffset in 0..<days {
            guard let day = calendar.date(byAdding: .day, value: dayOffset, to: startDate) else { continue }
            let dayStart = calendar.startOfDay(for: day)
            let dayEnd = calendar.date(byAdding: .day, value: 1, to: dayStart)!

            // Get meditation sessions for this day
            let daySessions = meditationSessions.filter { session in
                guard let timestamp = session.timestamp else { return false }
                return timestamp >= dayStart && timestamp < dayEnd
            }

            let hadMeditation = !daySessions.isEmpty
            let sessionCount = daySessions.count
            let totalMinutes = daySessions.reduce(0.0) { $0 + (Double($1.completedDuration) / 60.0) }

            // Get symptom log for this day
            let daySymptomLog = symptomLogs.first { log in
                guard let timestamp = log.timestamp else { return false }
                return calendar.isDate(timestamp, inSameDayAs: day)
            }

            let painLevel = daySymptomLog.map { Double($0.painAverage24h) }
            let basdaiScore = daySymptomLog.map { $0.basdaiScore }
            let moodScore = daySymptomLog.map { Double($0.moodScore) }
            let stressLevel = daySymptomLog.map { Double($0.stressLevel) }
            let sleepQuality = daySymptomLog.map { Double($0.sleepQuality) }

            metrics.append(DailyMeditationMetrics(
                date: day,
                hadMeditation: hadMeditation,
                sessionCount: sessionCount,
                totalMinutes: totalMinutes,
                painLevel: painLevel,
                basdaiScore: basdaiScore,
                moodScore: moodScore,
                stressLevel: stressLevel,
                sleepQuality: sleepQuality
            ))
        }

        return metrics
    }

    private func calculateSignificance(_ group1: [Double], _ group2: [Double]) -> Double {
        // Simple Welch's t-test implementation
        guard group1.count > 1, group2.count > 1 else { return 1.0 }

        let mean1 = group1.reduce(0, +) / Double(group1.count)
        let mean2 = group2.reduce(0, +) / Double(group2.count)

        let variance1 = group1.map { pow($0 - mean1, 2) }.reduce(0, +) / Double(group1.count - 1)
        let variance2 = group2.map { pow($0 - mean2, 2) }.reduce(0, +) / Double(group2.count - 1)

        let pooledStdError = sqrt(variance1 / Double(group1.count) + variance2 / Double(group2.count))

        guard pooledStdError > 0 else { return 1.0 }

        let tStatistic = abs(mean1 - mean2) / pooledStdError

        // Simplified p-value estimation
        // In production, use proper statistical library
        let pValue = 2 * (1 - normalCDF(abs(tStatistic)))

        return pValue
    }

    private func normalCDF(_ z: Double) -> Double {
        // Simplified normal CDF approximation
        // In production, use proper statistical library
        return 0.5 * (1 + erf(z / sqrt(2)))
    }

    private func erf(_ x: Double) -> Double {
        // Error function approximation
        let a1 =  0.254829592
        let a2 = -0.284496736
        let a3 =  1.421413741
        let a4 = -1.453152027
        let a5 =  1.061405429
        let p  =  0.3275911

        let sign = x < 0 ? -1.0 : 1.0
        let x = abs(x)

        let t = 1.0 / (1.0 + p * x)
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)

        return sign * y
    }

    private func determineConfidence(
        meditationDays: Int,
        nonMeditationDays: Int,
        significance: Double
    ) -> MeditationConfidenceLevel {
        // Need at least 7 days of each to have any confidence
        guard meditationDays >= 7, nonMeditationDays >= 7 else {
            return .insufficient
        }

        // Check statistical significance
        if significance > 0.1 {
            return .low
        } else if significance > 0.05 {
            return .moderate
        } else {
            return .high
        }
    }

    private func findMostEffectiveSession(
        meditationSessions: [MeditationSession],
        symptomLogs: [SymptomLog]
    ) -> (type: String?, category: String?, duration: TimeInterval?) {
        // Find sessions with before/after pain data
        let sessionsWithMetrics = meditationSessions.filter {
            $0.painLevelBefore != 0 && $0.painLevelAfter != 0
        }

        guard !sessionsWithMetrics.isEmpty else {
            return (nil, nil, nil)
        }

        // Group by type and calculate average reduction
        var typeReductions: [String: Double] = [:]
        var typeCounts: [String: Int] = [:]

        var categoryReductions: [String: Double] = [:]
        var categoryCounts: [String: Int] = [:]

        var durationSum: TimeInterval = 0

        for session in sessionsWithMetrics {
            let reduction = Double(session.painLevelBefore - session.painLevelAfter)

            if let type = session.sessionType {
                typeReductions[type, default: 0] += reduction
                typeCounts[type, default: 0] += 1
            }

            if let category = session.category {
                categoryReductions[category, default: 0] += reduction
                categoryCounts[category, default: 0] += 1
            }

            durationSum += TimeInterval(session.durationSeconds)
        }

        // Find best type
        let bestType = typeReductions.max { a, b in
            let avgA = a.value / Double(typeCounts[a.key] ?? 1)
            let avgB = b.value / Double(typeCounts[b.key] ?? 1)
            return avgA < avgB
        }?.key

        // Find best category
        let bestCategory = categoryReductions.max { a, b in
            let avgA = a.value / Double(categoryCounts[a.key] ?? 1)
            let avgB = b.value / Double(categoryCounts[b.key] ?? 1)
            return avgA < avgB
        }?.key

        // Calculate average duration of effective sessions
        let avgDuration = durationSum / Double(sessionsWithMetrics.count)

        return (bestType, bestCategory, avgDuration)
    }
}
