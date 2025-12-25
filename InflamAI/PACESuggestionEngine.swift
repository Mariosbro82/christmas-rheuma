//
//  PACESuggestionEngine.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import Foundation
import CoreData

// MARK: - PACE Suggestion Engine

class PACESuggestionEngine: ObservableObject {
    static let shared = PACESuggestionEngine()
    
    private init() {}
    
    func generateDailySuggestion(context: NSManagedObjectContext) -> PACESuggestion {
        let currentState = analyzeCurrentState(context: context)
        let recentTrends = analyzeRecentTrends(context: context)
        let personalPatterns = analyzePersonalPatterns(context: context)
        
        return createPACESuggestion(
            currentState: currentState,
            trends: recentTrends,
            patterns: personalPatterns
        )
    }
    
    // MARK: - Analysis Methods
    
    private func analyzeCurrentState(context: NSManagedObjectContext) -> CurrentHealthState {
        let today = Calendar.current.startOfDay(for: Date())
        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: today) ?? today
        
        // Get today's pain entries
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(format: "timestamp >= %@", today as CVarArg)
        painRequest.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: false)]
        
        // Get yesterday's journal entry for sleep and energy
        let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        journalRequest.predicate = NSPredicate(
            format: "date >= %@ AND date < %@",
            yesterday as CVarArg,
            today as CVarArg
        )
        journalRequest.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: false)]
        
        // Get recent BASDAI assessment
        let bassdaiRequest: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
        bassdaiRequest.predicate = NSPredicate(
            format: "date >= %@",
            Calendar.current.date(byAdding: .day, value: -7, to: today) ?? today as CVarArg
        )
        bassdaiRequest.sortDescriptors = [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: false)]
        bassdaiRequest.fetchLimit = 1
        
        do {
            let todayPain = try context.fetch(painRequest)
            let lastJournal = try context.fetch(journalRequest).first
            let recentBASSDAI = try context.fetch(bassdaiRequest).first
            
            // FIXED: Use 0 for missing data, not fake "moderate" values
            let currentPainLevel = todayPain.first?.painLevel ?? 0
            let sleepQuality = lastJournal?.sleepQuality ?? 0  // 0 = not tracked
            let energyLevel = lastJournal?.energyLevel ?? 0    // 0 = not tracked
            let overallWellbeing = recentBASSDAI?.overallWellbeing ?? 0  // 0 = not assessed
            
            return CurrentHealthState(
                painLevel: Int(currentPainLevel),
                sleepQuality: Int(sleepQuality),
                energyLevel: Int(energyLevel),
                overallWellbeing: Int(overallWellbeing),
                hasTrackedToday: !todayPain.isEmpty
            )
        } catch {
            print("Error analyzing current state: \(error)")
            // FIXED: Return 0 for all values = "no data available"
            return CurrentHealthState(
                painLevel: 0,
                sleepQuality: 0,
                energyLevel: 0,
                overallWellbeing: 0,
                hasTrackedToday: false
            )
        }
    }
    
    private func analyzeRecentTrends(context: NSManagedObjectContext) -> HealthTrends {
        let last7Days = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let last14Days = Calendar.current.date(byAdding: .day, value: -14, to: Date()) ?? Date()
        
        // Analyze pain trend
        let painTrend = analyzePainTrend(context: context, since: last7Days)
        
        // Analyze energy trend
        let energyTrend = analyzeEnergyTrend(context: context, since: last7Days)
        
        // Analyze medication adherence
        let medicationAdherence = analyzeMedicationAdherence(context: context, since: last7Days)
        
        // Analyze activity level
        let activityLevel = analyzeActivityLevel(context: context, since: last7Days)
        
        return HealthTrends(
            painTrend: painTrend,
            energyTrend: energyTrend,
            medicationAdherence: medicationAdherence,
            activityLevel: activityLevel
        )
    }
    
    private func analyzePersonalPatterns(context: NSManagedObjectContext) -> PersonalPatterns {
        let last30Days = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
        
        // Find optimal activity times
        let optimalTimes = findOptimalActivityTimes(context: context, since: last30Days)
        
        // Find effective activities
        let effectiveActivities = findEffectiveActivities(context: context, since: last30Days)
        
        // Find pain triggers
        let painTriggers = findPainTriggers(context: context, since: last30Days)
        
        // Calculate average recovery time
        let recoveryTime = calculateAverageRecoveryTime(context: context, since: last30Days)
        
        return PersonalPatterns(
            optimalActivityTimes: optimalTimes,
            effectiveActivities: effectiveActivities,
            painTriggers: painTriggers,
            averageRecoveryTime: recoveryTime
        )
    }
    
    // MARK: - Trend Analysis Helpers
    
    private func analyzePainTrend(context: NSManagedObjectContext, since date: Date) -> TrendDirection {
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(format: "timestamp >= %@", date as CVarArg)
        painRequest.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)]
        
        do {
            let entries = try context.fetch(painRequest)
            guard entries.count >= 3 else { return .stable }
            
            let firstHalf = entries.prefix(entries.count / 2)
            let secondHalf = entries.suffix(entries.count / 2)
            
            let firstAvg = firstHalf.reduce(0) { $0 + $1.painLevel } / Double(firstHalf.count)
            let secondAvg = secondHalf.reduce(0) { $0 + $1.painLevel } / Double(secondHalf.count)
            
            let difference = secondAvg - firstAvg
            
            if difference > 0.5 {
                return .increasing
            } else if difference < -0.5 {
                return .decreasing
            } else {
                return .stable
            }
        } catch {
            return .stable
        }
    }
    
    private func analyzeEnergyTrend(context: NSManagedObjectContext, since date: Date) -> TrendDirection {
        let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        journalRequest.predicate = NSPredicate(format: "date >= %@", date as CVarArg)
        journalRequest.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        do {
            let entries = try context.fetch(journalRequest)
            guard entries.count >= 3 else { return .stable }
            
            let firstHalf = entries.prefix(entries.count / 2)
            let secondHalf = entries.suffix(entries.count / 2)
            
            let firstAvg = firstHalf.reduce(0) { $0 + $1.energyLevel } / Double(firstHalf.count)
            let secondAvg = secondHalf.reduce(0) { $0 + $1.energyLevel } / Double(secondHalf.count)
            
            let difference = secondAvg - firstAvg
            
            if difference > 0.5 {
                return .increasing
            } else if difference < -0.5 {
                return .decreasing
            } else {
                return .stable
            }
        } catch {
            return .stable
        }
    }
    
    private func analyzeMedicationAdherence(context: NSManagedObjectContext, since date: Date) -> Double {
        let medicationRequest: NSFetchRequest<MedicationIntake> = MedicationIntake.fetchRequest()
        medicationRequest.predicate = NSPredicate(format: "timestamp >= %@", date as CVarArg)
        
        let medicationsRequest: NSFetchRequest<Medication> = Medication.fetchRequest()
        
        do {
            let intakes = try context.fetch(medicationRequest)
            let medications = try context.fetch(medicationsRequest)
            
            guard !medications.isEmpty else { return 1.0 }
            
            let daysSince = Calendar.current.dateComponents([.day], from: date, to: Date()).day ?? 1
            let expectedIntakes = medications.count * daysSince
            
            return expectedIntakes > 0 ? Double(intakes.count) / Double(expectedIntakes) : 1.0
        } catch {
            return 0.5
        }
    }
    
    private func analyzeActivityLevel(context: NSManagedObjectContext, since date: Date) -> ActivityLevel {
        let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        journalRequest.predicate = NSPredicate(format: "date >= %@ AND activities != nil", date as CVarArg)
        
        do {
            let entries = try context.fetch(journalRequest)
            let totalActivities = entries.reduce(0) { total, entry in
                return total + (entry.activities?.components(separatedBy: ",").count ?? 0)
            }
            
            let averagePerDay = Double(totalActivities) / Double(max(entries.count, 1))
            
            if averagePerDay >= 3 {
                return .high
            } else if averagePerDay >= 1.5 {
                return .moderate
            } else {
                return .low
            }
        } catch {
            return .moderate
        }
    }
    
    // MARK: - Pattern Analysis Helpers
    
    private func findOptimalActivityTimes(context: NSManagedObjectContext, since date: Date) -> [Int] {
        let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        journalRequest.predicate = NSPredicate(
            format: "date >= %@ AND energyLevel >= 7",
            date as CVarArg
        )
        
        do {
            let highEnergyEntries = try context.fetch(journalRequest)
            let hours = highEnergyEntries.compactMap { entry in
                entry.date?.hour
            }
            
            // Find most common hours
            let hourCounts = Dictionary(grouping: hours, by: { $0 })
                .mapValues { $0.count }
                .sorted { $0.value > $1.value }
            
            return Array(hourCounts.prefix(3).map { $0.key })
        } catch {
            return [9, 14, 16] // Default optimal times
        }
    }
    
    private func findEffectiveActivities(context: NSManagedObjectContext, since date: Date) -> [String] {
        let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        journalRequest.predicate = NSPredicate(
            format: "date >= %@ AND activities != nil AND mood >= 7",
            date as CVarArg
        )
        
        do {
            let positiveEntries = try context.fetch(journalRequest)
            let allActivities = positiveEntries.compactMap { $0.activities }
                .flatMap { $0.components(separatedBy: ",") }
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            
            let activityCounts = Dictionary(grouping: allActivities, by: { $0 })
                .mapValues { $0.count }
                .sorted { $0.value > $1.value }
            
            return Array(activityCounts.prefix(5).map { $0.key })
        } catch {
            return ["Walking", "Stretching", "Reading", "Meditation", "Light Exercise"]
        }
    }
    
    private func findPainTriggers(context: NSManagedObjectContext, since date: Date) -> [String] {
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(
            format: "timestamp >= %@ AND painLevel >= 7 AND triggers != nil",
            date as CVarArg
        )
        
        do {
            let highPainEntries = try context.fetch(painRequest)
            let allTriggers = highPainEntries.compactMap { $0.triggers }
                .flatMap { $0.components(separatedBy: ",") }
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            
            let triggerCounts = Dictionary(grouping: allTriggers, by: { $0 })
                .mapValues { $0.count }
                .sorted { $0.value > $1.value }
            
            return Array(triggerCounts.prefix(3).map { $0.key })
        } catch {
            return []
        }
    }
    
    private func calculateAverageRecoveryTime(context: NSManagedObjectContext, since date: Date) -> TimeInterval {
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(format: "timestamp >= %@", date as CVarArg)
        painRequest.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)]
        
        do {
            let entries = try context.fetch(painRequest)
            var recoveryTimes: [TimeInterval] = []
            
            for i in 0..<(entries.count - 1) {
                let current = entries[i]
                let next = entries[i + 1]
                
                if current.painLevel >= 7 && next.painLevel <= 4,
                   let currentTime = current.timestamp,
                   let nextTime = next.timestamp {
                    recoveryTimes.append(nextTime.timeIntervalSince(currentTime))
                }
            }
            
            guard !recoveryTimes.isEmpty else { return 24 * 3600 } // Default 24 hours
            
            return recoveryTimes.reduce(0, +) / Double(recoveryTimes.count)
        } catch {
            return 24 * 3600 // Default 24 hours
        }
    }
    
    // MARK: - PACE Suggestion Creation
    
    private func createPACESuggestion(
        currentState: CurrentHealthState,
        trends: HealthTrends,
        patterns: PersonalPatterns
    ) -> PACESuggestion {
        
        let paceLevel = determinePACELevel(
            currentState: currentState,
            trends: trends
        )
        
        let activities = selectActivities(
            paceLevel: paceLevel,
            patterns: patterns,
            currentState: currentState
        )
        
        let timing = suggestTiming(
            paceLevel: paceLevel,
            patterns: patterns,
            currentState: currentState
        )
        
        let precautions = generatePrecautions(
            currentState: currentState,
            patterns: patterns
        )
        
        let motivation = generateMotivation(
            currentState: currentState,
            trends: trends
        )
        
        return PACESuggestion(
            date: Date(),
            paceLevel: paceLevel,
            primaryActivity: activities.first ?? "Gentle movement",
            alternativeActivities: Array(activities.dropFirst()),
            recommendedTiming: timing,
            duration: suggestDuration(paceLevel: paceLevel, currentState: currentState),
            precautions: precautions,
            motivation: motivation,
            confidence: calculateConfidence(currentState: currentState, trends: trends)
        )
    }
    
    private func determinePACELevel(
        currentState: CurrentHealthState,
        trends: HealthTrends
    ) -> PACELevel {
        
        // Base level on current pain and energy
        var score = 0
        
        // Pain level influence (lower pain = higher activity)
        if currentState.painLevel <= 3 {
            score += 3
        } else if currentState.painLevel <= 6 {
            score += 2
        } else {
            score += 1
        }
        
        // Energy level influence
        if currentState.energyLevel >= 7 {
            score += 2
        } else if currentState.energyLevel >= 5 {
            score += 1
        }
        
        // Sleep quality influence
        if currentState.sleepQuality >= 7 {
            score += 1
        } else if currentState.sleepQuality <= 4 {
            score -= 1
        }
        
        // Trend adjustments
        if trends.painTrend == .increasing {
            score -= 1
        } else if trends.painTrend == .decreasing {
            score += 1
        }
        
        if trends.energyTrend == .increasing {
            score += 1
        } else if trends.energyTrend == .decreasing {
            score -= 1
        }
        
        // Convert score to PACE level
        if score >= 6 {
            return .active
        } else if score >= 4 {
            return .moderate
        } else if score >= 2 {
            return .gentle
        } else {
            return .rest
        }
    }
    
    private func selectActivities(
        paceLevel: PACELevel,
        patterns: PersonalPatterns,
        currentState: CurrentHealthState
    ) -> [String] {
        
        var activities: [String] = []
        
        switch paceLevel {
        case .rest:
            activities = ["Deep breathing", "Meditation", "Gentle stretching", "Reading"]
        case .gentle:
            activities = ["Light walking", "Yoga", "Stretching", "Tai chi", "Swimming"]
        case .moderate:
            activities = ["Walking", "Cycling", "Swimming", "Yoga", "Strength training"]
        case .active:
            activities = ["Jogging", "Cycling", "Swimming laps", "Strength training", "Sports"]
        }
        
        // Prioritize effective activities from patterns
        let effectiveActivities = patterns.effectiveActivities
        let prioritized = activities.sorted { activity1, activity2 in
            let effective1 = effectiveActivities.contains { $0.lowercased().contains(activity1.lowercased()) }
            let effective2 = effectiveActivities.contains { $0.lowercased().contains(activity2.lowercased()) }
            return effective1 && !effective2
        }
        
        return Array(prioritized.prefix(4))
    }
    
    private func suggestTiming(
        paceLevel: PACELevel,
        patterns: PersonalPatterns,
        currentState: CurrentHealthState
    ) -> String {
        
        let currentHour = Calendar.current.component(.hour, from: Date())
        let optimalTimes = patterns.optimalActivityTimes
        
        // Find the next optimal time
        let nextOptimalTime = optimalTimes.first { $0 > currentHour } ?? optimalTimes.first ?? 10
        
        if currentState.energyLevel >= 6 && currentState.painLevel <= 5 {
            return "Now is a good time for activity"
        } else if nextOptimalTime > currentHour {
            let formatter = DateFormatter()
            formatter.dateFormat = "h a"
            let suggestedTime = Calendar.current.date(bySettingHour: nextOptimalTime, minute: 0, second: 0, of: Date()) ?? Date()
            return "Consider activity around \(formatter.string(from: suggestedTime))"
        } else {
            return "Listen to your body and choose a time when you feel ready"
        }
    }
    
    private func suggestDuration(paceLevel: PACELevel, currentState: CurrentHealthState) -> String {
        let baseDuration: String
        
        switch paceLevel {
        case .rest:
            baseDuration = "5-10 minutes"
        case .gentle:
            baseDuration = "10-20 minutes"
        case .moderate:
            baseDuration = "20-30 minutes"
        case .active:
            baseDuration = "30-45 minutes"
        }
        
        // Adjust based on current state
        if currentState.painLevel >= 7 {
            return "5-10 minutes (shorter if needed)"
        } else if currentState.energyLevel <= 4 {
            return "\(baseDuration) (start shorter if tired)"
        } else {
            return baseDuration
        }
    }
    
    private func generatePrecautions(
        currentState: CurrentHealthState,
        patterns: PersonalPatterns
    ) -> [String] {
        
        var precautions: [String] = []
        
        if currentState.painLevel >= 6 {
            precautions.append("Stop if pain increases")
            precautions.append("Start very gently")
        }
        
        if currentState.energyLevel <= 4 {
            precautions.append("Listen to your body")
            precautions.append("Rest if you feel more tired")
        }
        
        if currentState.sleepQuality <= 4 {
            precautions.append("Consider lighter activity due to poor sleep")
        }
        
        // Add trigger-based precautions
        for trigger in patterns.painTriggers.prefix(2) {
            precautions.append("Avoid \(trigger.lowercased()) if possible")
        }
        
        if precautions.isEmpty {
            precautions.append("Warm up before activity")
            precautions.append("Stay hydrated")
        }
        
        return precautions
    }
    
    private func generateMotivation(
        currentState: CurrentHealthState,
        trends: HealthTrends
    ) -> String {
        
        if trends.painTrend == .decreasing {
            return "Your pain levels are improving! Keep up the great work with consistent activity."
        } else if trends.energyTrend == .increasing {
            return "Your energy is on the rise! This is a great time to stay active."
        } else if currentState.painLevel <= 4 {
            return "You're having a good day! Take advantage of lower pain levels."
        } else if currentState.energyLevel >= 6 {
            return "You have good energy today. Gentle movement can help maintain it."
        } else if trends.medicationAdherence >= 0.8 {
            return "Your medication consistency is paying off. Stay active to maximize benefits."
        } else {
            return "Every small step counts. Be gentle with yourself today."
        }
    }
    
    private func calculateConfidence(
        currentState: CurrentHealthState,
        trends: HealthTrends
    ) -> Double {
        
        var confidence = 0.5
        
        // Higher confidence with more data
        if currentState.hasTrackedToday {
            confidence += 0.2
        }
        
        // Stable trends increase confidence
        if trends.painTrend == .stable {
            confidence += 0.1
        }
        
        if trends.energyTrend == .stable {
            confidence += 0.1
        }
        
        // Good medication adherence increases confidence
        if trends.medicationAdherence >= 0.8 {
            confidence += 0.1
        }
        
        return min(confidence, 1.0)
    }
}

// MARK: - Data Models

struct PACESuggestion {
    let date: Date
    let paceLevel: PACELevel
    let primaryActivity: String
    let alternativeActivities: [String]
    let recommendedTiming: String
    let duration: String
    let precautions: [String]
    let motivation: String
    let confidence: Double
}

enum PACELevel: String, CaseIterable {
    case rest = "Rest"
    case gentle = "Gentle"
    case moderate = "Moderate"
    case active = "Active"
    
    var description: String {
        switch self {
        case .rest:
            return "Focus on rest and recovery"
        case .gentle:
            return "Light, gentle activities"
        case .moderate:
            return "Moderate activity level"
        case .active:
            return "Higher activity level"
        }
    }
    
    var color: String {
        switch self {
        case .rest: return "red"
        case .gentle: return "orange"
        case .moderate: return "yellow"
        case .active: return "green"
        }
    }
}

struct CurrentHealthState {
    let painLevel: Int
    let sleepQuality: Int
    let energyLevel: Int
    let overallWellbeing: Int
    let hasTrackedToday: Bool
}

struct HealthTrends {
    let painTrend: TrendDirection
    let energyTrend: TrendDirection
    let medicationAdherence: Double
    let activityLevel: ActivityLevel
}

struct PersonalPatterns {
    let optimalActivityTimes: [Int]
    let effectiveActivities: [String]
    let painTriggers: [String]
    let averageRecoveryTime: TimeInterval
}

enum TrendDirection: String, CaseIterable {
    case increasing = "Increasing"
    case decreasing = "Decreasing"
    case stable = "Stable"
}

enum ActivityLevel: String, CaseIterable {
    case low = "Low"
    case moderate = "Moderate"
    case high = "High"
}

// MARK: - Extensions

extension Date {
    var hour: Int {
        return Calendar.current.component(.hour, from: self)
    }
}