//
//  HealthInsightsEngine.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import Foundation
import CoreData

class HealthInsightsEngine: ObservableObject {
    static let shared = HealthInsightsEngine()
    
    private init() {}
    
    // MARK: - Data Models
    
    struct HealthDataPoint {
        let date: Date
        let painLevel: Double
        let energyLevel: Double
        let sleepQuality: Double
        let bassdaiScore: Double?
        let medications: [String]
        let activities: [String]
        let mood: String?
    }
    
    struct CorrelationResult {
        let id = UUID()
        let factor1: String
        let factor2: String
        let correlation: Double
        let strength: CorrelationStrength
        let description: String
        let confidence: Double
        let sampleSize: Int
    }
    
    enum CorrelationStrength: String, CaseIterable {
        case veryWeak = "Very Weak"
        case weak = "Weak"
        case moderate = "Moderate"
        case strong = "Strong"
        case veryStrong = "Very Strong"
        
        static func from(correlation: Double) -> CorrelationStrength {
            let absCorr = abs(correlation)
            switch absCorr {
            case 0.0..<0.2: return .veryWeak
            case 0.2..<0.4: return .weak
            case 0.4..<0.6: return .moderate
            case 0.6..<0.8: return .strong
            default: return .veryStrong
            }
        }
    }
    
    struct MedicationEffect {
        let id = UUID()
        let medicationName: String
        let effectOnPain: Double
        let effectOnEnergy: Double
        let effectOnSleep: Double
        let timeToEffect: TimeInterval // in hours
        let confidence: Double
        let sampleSize: Int
    }
    
    struct PatternInsight {
        let id = UUID()
        let title: String
        let description: String
        let type: InsightType
        let confidence: Double
        let actionable: Bool
        let recommendation: String?
    }
    
    enum InsightType {
        case medicationEffect
        case activityCorrelation
        case sleepPattern
        case painTrigger
        case energyBooster
        case moodInfluencer
    }
    
    // MARK: - Core Analysis Functions
    
    func analyzeHealthData(context: NSManagedObjectContext) -> [PatternInsight] {
        let healthData = aggregateHealthData(context: context)
        var insights: [PatternInsight] = []
        
        // Analyze medication effects
        insights.append(contentsOf: analyzeMedicationEffects(data: healthData))
        
        // Analyze activity correlations
        insights.append(contentsOf: analyzeActivityCorrelations(data: healthData))
        
        // Analyze sleep patterns
        insights.append(contentsOf: analyzeSleepPatterns(data: healthData))
        
        // Analyze pain triggers
        insights.append(contentsOf: analyzePainTriggers(data: healthData))
        
        return insights.sorted { $0.confidence > $1.confidence }
    }
    
    func calculateCorrelations(context: NSManagedObjectContext) -> [CorrelationResult] {
        let healthData = aggregateHealthData(context: context)
        var correlations: [CorrelationResult] = []
        
        // Pain vs Sleep correlation
        let painSleepCorr = calculatePearsonCorrelation(
            x: healthData.map { $0.painLevel },
            y: healthData.map { $0.sleepQuality }
        )
        
        if let correlation = painSleepCorr, healthData.count >= 10 {
            correlations.append(CorrelationResult(
                factor1: "Pain Level",
                factor2: "Sleep Quality",
                correlation: correlation,
                strength: CorrelationStrength.from(correlation: correlation),
                description: generateCorrelationDescription("Pain Level", "Sleep Quality", correlation),
                confidence: calculateConfidence(correlation: correlation, sampleSize: healthData.count),
                sampleSize: healthData.count
            ))
        }
        
        // Energy vs Sleep correlation
        let energySleepCorr = calculatePearsonCorrelation(
            x: healthData.map { $0.energyLevel },
            y: healthData.map { $0.sleepQuality }
        )
        
        if let correlation = energySleepCorr, healthData.count >= 10 {
            correlations.append(CorrelationResult(
                factor1: "Energy Level",
                factor2: "Sleep Quality",
                correlation: correlation,
                strength: CorrelationStrength.from(correlation: correlation),
                description: generateCorrelationDescription("Energy Level", "Sleep Quality", correlation),
                confidence: calculateConfidence(correlation: correlation, sampleSize: healthData.count),
                sampleSize: healthData.count
            ))
        }
        
        // Pain vs Energy correlation
        let painEnergyCorr = calculatePearsonCorrelation(
            x: healthData.map { $0.painLevel },
            y: healthData.map { $0.energyLevel }
        )
        
        if let correlation = painEnergyCorr, healthData.count >= 10 {
            correlations.append(CorrelationResult(
                factor1: "Pain Level",
                factor2: "Energy Level",
                correlation: correlation,
                strength: CorrelationStrength.from(correlation: correlation),
                description: generateCorrelationDescription("Pain Level", "Energy Level", correlation),
                confidence: calculateConfidence(correlation: correlation, sampleSize: healthData.count),
                sampleSize: healthData.count
            ))
        }
        
        return correlations.filter { $0.confidence > 0.3 }
    }
    
    func analyzeMedicationEffectiveness(context: NSManagedObjectContext) -> [MedicationEffect] {
        let healthData = aggregateHealthData(context: context)
        var effects: [MedicationEffect] = []
        
        // Get unique medications
        let allMedications = Set(healthData.flatMap { $0.medications })
        
        for medication in allMedications {
            let effect = calculateMedicationEffect(medication: medication, data: healthData)
            if effect.confidence > 0.4 && effect.sampleSize >= 5 {
                effects.append(effect)
            }
        }
        
        return effects.sorted { $0.confidence > $1.confidence }
    }
    
    // MARK: - Private Analysis Methods
    
    private func aggregateHealthData(context: NSManagedObjectContext) -> [HealthDataPoint] {
        let calendar = Calendar.current
        let cutoffDate = calendar.date(byAdding: .month, value: -3, to: Date()) ?? Date()
        
        // Fetch all relevant data
        let painRequest: NSFetchRequest<PainEntry> = PainEntry.fetchRequest()
        painRequest.predicate = NSPredicate(format: "timestamp >= %@", cutoffDate as CVarArg)
        painRequest.sortDescriptors = [NSSortDescriptor(keyPath: \PainEntry.timestamp, ascending: true)]
        
        let journalRequest: NSFetchRequest<JournalEntry> = JournalEntry.fetchRequest()
        journalRequest.predicate = NSPredicate(format: "date >= %@", cutoffDate as CVarArg)
        journalRequest.sortDescriptors = [NSSortDescriptor(keyPath: \JournalEntry.date, ascending: true)]
        
        let bassdaiRequest: NSFetchRequest<BASSDAIAssessment> = BASSDAIAssessment.fetchRequest()
        bassdaiRequest.predicate = NSPredicate(format: "date >= %@", cutoffDate as CVarArg)
        bassdaiRequest.sortDescriptors = [NSSortDescriptor(keyPath: \BASSDAIAssessment.date, ascending: true)]
        
        let medicationRequest: NSFetchRequest<MedicationIntake> = MedicationIntake.fetchRequest()
        medicationRequest.predicate = NSPredicate(format: "timestamp >= %@", cutoffDate as CVarArg)
        medicationRequest.sortDescriptors = [NSSortDescriptor(keyPath: \MedicationIntake.timestamp, ascending: true)]
        
        do {
            let painEntries = try context.fetch(painRequest)
            let journalEntries = try context.fetch(journalRequest)
            let bassdaiAssessments = try context.fetch(bassdaiRequest)
            let medicationIntakes = try context.fetch(medicationRequest)
            
            return aggregateDataByDay(
                painEntries: painEntries,
                journalEntries: journalEntries,
                bassdaiAssessments: bassdaiAssessments,
                medicationIntakes: medicationIntakes
            )
        } catch {
            print("Error fetching health data: \(error)")
            return []
        }
    }
    
    private func aggregateDataByDay(
        painEntries: [PainEntry],
        journalEntries: [JournalEntry],
        bassdaiAssessments: [BASSDAIAssessment],
        medicationIntakes: [MedicationIntake]
    ) -> [HealthDataPoint] {
        let calendar = Calendar.current
        var dataPoints: [HealthDataPoint] = []
        
        // Group data by day
        let painByDay = Dictionary(grouping: painEntries) { entry in
            calendar.startOfDay(for: entry.timestamp ?? Date())
        }
        
        let journalByDay = Dictionary(grouping: journalEntries) { entry in
            calendar.startOfDay(for: entry.date ?? Date())
        }
        
        let bassdaiByDay = Dictionary(grouping: bassdaiAssessments) { assessment in
            calendar.startOfDay(for: assessment.date ?? Date())
        }
        
        let medicationByDay = Dictionary(grouping: medicationIntakes) { intake in
            calendar.startOfDay(for: intake.timestamp ?? Date())
        }
        
        // Get all unique days
        let allDays = Set(painByDay.keys)
            .union(Set(journalByDay.keys))
            .union(Set(bassdaiByDay.keys))
            .union(Set(medicationByDay.keys))
        
        for day in allDays.sorted() {
            let dayPainEntries = painByDay[day] ?? []
            let dayJournalEntries = journalByDay[day] ?? []
            let dayBassdaiAssessments = bassdaiByDay[day] ?? []
            let dayMedicationIntakes = medicationByDay[day] ?? []
            
            let avgPainLevel = dayPainEntries.isEmpty ? 0 : dayPainEntries.map { Double($0.painLevel) }.reduce(0, +) / Double(dayPainEntries.count)
            let avgEnergyLevel = dayJournalEntries.isEmpty ? 0 : dayJournalEntries.map { $0.energyLevel }.reduce(0, +) / Double(dayJournalEntries.count)
            let avgSleepQuality = dayJournalEntries.isEmpty ? 0 : dayJournalEntries.map { $0.sleepQuality }.reduce(0, +) / Double(dayJournalEntries.count)
            let avgBassdaiScore = dayBassdaiAssessments.isEmpty ? nil : dayBassdaiAssessments.map { $0.totalScore }.reduce(0, +) / Double(dayBassdaiAssessments.count)
            
            let medications = dayMedicationIntakes.compactMap { $0.medicationName }
            let activities = dayJournalEntries.compactMap { $0.activities?.components(separatedBy: ",") }.flatMap { $0 }
            let mood = dayJournalEntries.first?.mood
            
            dataPoints.append(HealthDataPoint(
                date: day,
                painLevel: avgPainLevel,
                energyLevel: avgEnergyLevel,
                sleepQuality: avgSleepQuality,
                bassdaiScore: avgBassdaiScore,
                medications: medications,
                activities: activities,
                mood: mood
            ))
        }
        
        return dataPoints.sorted { $0.date < $1.date }
    }
    
    private func calculatePearsonCorrelation(x: [Double], y: [Double]) -> Double? {
        guard x.count == y.count && x.count > 1 else { return nil }
        
        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        guard denominator != 0 else { return nil }
        
        return numerator / denominator
    }
    
    private func calculateMedicationEffect(medication: String, data: [HealthDataPoint]) -> MedicationEffect {
        let withMedication = data.filter { $0.medications.contains(medication) }
        let withoutMedication = data.filter { !$0.medications.contains(medication) }
        
        guard withMedication.count >= 3 && withoutMedication.count >= 3 else {
            return MedicationEffect(
                medicationName: medication,
                effectOnPain: 0,
                effectOnEnergy: 0,
                effectOnSleep: 0,
                timeToEffect: 0,
                confidence: 0,
                sampleSize: withMedication.count
            )
        }
        
        let avgPainWith = withMedication.map { $0.painLevel }.reduce(0, +) / Double(withMedication.count)
        let avgPainWithout = withoutMedication.map { $0.painLevel }.reduce(0, +) / Double(withoutMedication.count)
        
        let avgEnergyWith = withMedication.map { $0.energyLevel }.reduce(0, +) / Double(withMedication.count)
        let avgEnergyWithout = withoutMedication.map { $0.energyLevel }.reduce(0, +) / Double(withoutMedication.count)
        
        let avgSleepWith = withMedication.map { $0.sleepQuality }.reduce(0, +) / Double(withMedication.count)
        let avgSleepWithout = withoutMedication.map { $0.sleepQuality }.reduce(0, +) / Double(withoutMedication.count)
        
        let painEffect = avgPainWithout - avgPainWith // Positive means medication reduces pain
        let energyEffect = avgEnergyWith - avgEnergyWithout // Positive means medication increases energy
        let sleepEffect = avgSleepWith - avgSleepWithout // Positive means medication improves sleep
        
        let confidence = calculateEffectConfidence(
            withCount: withMedication.count,
            withoutCount: withoutMedication.count,
            effectSize: abs(painEffect)
        )
        
        return MedicationEffect(
            medicationName: medication,
            effectOnPain: painEffect,
            effectOnEnergy: energyEffect,
            effectOnSleep: sleepEffect,
            timeToEffect: 2 * 3600, // Assume 2 hours average
            confidence: confidence,
            sampleSize: withMedication.count
        )
    }
    
    private func analyzeMedicationEffects(data: [HealthDataPoint]) -> [PatternInsight] {
        var insights: [PatternInsight] = []
        
        let allMedications = Set(data.flatMap { $0.medications })
        
        for medication in allMedications {
            let effect = calculateMedicationEffect(medication: medication, data: data)
            
            if effect.confidence > 0.5 {
                if effect.effectOnPain > 1.0 {
                    insights.append(PatternInsight(
                        title: "\(medication) Reduces Pain",
                        description: "\(medication) appears to reduce your pain levels by an average of \(String(format: "%.1f", effect.effectOnPain)) points.",
                        type: .medicationEffect,
                        confidence: effect.confidence,
                        actionable: true,
                        recommendation: "Consider discussing optimal timing with your healthcare provider."
                    ))
                }
                
                if effect.effectOnEnergy > 1.0 {
                    insights.append(PatternInsight(
                        title: "\(medication) Boosts Energy",
                        description: "\(medication) seems to improve your energy levels by \(String(format: "%.1f", effect.effectOnEnergy)) points.",
                        type: .medicationEffect,
                        confidence: effect.confidence,
                        actionable: true,
                        recommendation: "Take this medication when you need an energy boost."
                    ))
                }
            }
        }
        
        return insights
    }
    
    private func analyzeActivityCorrelations(data: [HealthDataPoint]) -> [PatternInsight] {
        var insights: [PatternInsight] = []
        
        let allActivities = Set(data.flatMap { $0.activities })
        
        for activity in allActivities {
            let withActivity = data.filter { $0.activities.contains(activity) }
            let withoutActivity = data.filter { !$0.activities.contains(activity) }
            
            guard withActivity.count >= 3 && withoutActivity.count >= 3 else { continue }
            
            let avgPainWith = withActivity.map { $0.painLevel }.reduce(0, +) / Double(withActivity.count)
            let avgPainWithout = withoutActivity.map { $0.painLevel }.reduce(0, +) / Double(withoutActivity.count)
            
            let painDifference = avgPainWithout - avgPainWith
            
            if abs(painDifference) > 1.0 {
                let confidence = calculateEffectConfidence(
                    withCount: withActivity.count,
                    withoutCount: withoutActivity.count,
                    effectSize: abs(painDifference)
                )
                
                if confidence > 0.4 {
                    if painDifference > 0 {
                        insights.append(PatternInsight(
                            title: "\(activity) Helps with Pain",
                            description: "\(activity) appears to reduce your pain levels by \(String(format: "%.1f", painDifference)) points on average.",
                            type: .activityCorrelation,
                            confidence: confidence,
                            actionable: true,
                            recommendation: "Consider incorporating more \(activity.lowercased()) into your routine."
                        ))
                    } else {
                        insights.append(PatternInsight(
                            title: "\(activity) May Increase Pain",
                            description: "\(activity) seems to be associated with higher pain levels (\(String(format: "+%.1f", abs(painDifference))) points).",
                            type: .painTrigger,
                            confidence: confidence,
                            actionable: true,
                            recommendation: "Consider modifying or avoiding \(activity.lowercased()) when pain levels are high."
                        ))
                    }
                }
            }
        }
        
        return insights
    }
    
    private func analyzeSleepPatterns(data: [HealthDataPoint]) -> [PatternInsight] {
        var insights: [PatternInsight] = []
        
        let goodSleepDays = data.filter { $0.sleepQuality >= 7 }
        let poorSleepDays = data.filter { $0.sleepQuality <= 4 }
        
        guard goodSleepDays.count >= 3 && poorSleepDays.count >= 3 else { return insights }
        
        let avgPainGoodSleep = goodSleepDays.map { $0.painLevel }.reduce(0, +) / Double(goodSleepDays.count)
        let avgPainPoorSleep = poorSleepDays.map { $0.painLevel }.reduce(0, +) / Double(poorSleepDays.count)
        
        let painDifference = avgPainPoorSleep - avgPainGoodSleep
        
        if painDifference > 1.0 {
            let confidence = calculateEffectConfidence(
                withCount: goodSleepDays.count,
                withoutCount: poorSleepDays.count,
                effectSize: painDifference
            )
            
            insights.append(PatternInsight(
                title: "Sleep Quality Affects Pain",
                description: "Poor sleep is associated with \(String(format: "%.1f", painDifference)) points higher pain levels.",
                type: .sleepPattern,
                confidence: confidence,
                actionable: true,
                recommendation: "Focus on improving sleep hygiene and maintaining consistent sleep schedules."
            ))
        }
        
        return insights
    }
    
    private func analyzePainTriggers(data: [HealthDataPoint]) -> [PatternInsight] {
        var insights: [PatternInsight] = []
        
        // Analyze mood correlation with pain
        let moodGroups = Dictionary(grouping: data.compactMap { point in
            guard let mood = point.mood else { return nil }
            return (mood: mood, pain: point.painLevel)
        }) { $0.mood }
        
        for (mood, entries) in moodGroups {
            guard entries.count >= 3 else { continue }
            
            let avgPain = entries.map { $0.pain }.reduce(0, +) / Double(entries.count)
            let overallAvgPain = data.map { $0.painLevel }.reduce(0, +) / Double(data.count)
            
            let painDifference = avgPain - overallAvgPain
            
            if abs(painDifference) > 1.0 {
                let confidence = min(0.8, Double(entries.count) / 10.0)
                
                if painDifference > 0 {
                    insights.append(PatternInsight(
                        title: "\(mood) Mood Linked to Higher Pain",
                        description: "When feeling \(mood.lowercased()), your pain levels are \(String(format: "%.1f", painDifference)) points higher than average.",
                        type: .moodInfluencer,
                        confidence: confidence,
                        actionable: true,
                        recommendation: "Consider stress management techniques or mood-boosting activities."
                    ))
                } else {
                    insights.append(PatternInsight(
                        title: "\(mood) Mood Helps with Pain",
                        description: "When feeling \(mood.lowercased()), your pain levels are \(String(format: "%.1f", abs(painDifference))) points lower than average.",
                        type: .moodInfluencer,
                        confidence: confidence,
                        actionable: true,
                        recommendation: "Try to cultivate activities that promote this positive mood state."
                    ))
                }
            }
        }
        
        return insights
    }
    
    // MARK: - Helper Functions
    
    private func generateCorrelationDescription(_ factor1: String, _ factor2: String, _ correlation: Double) -> String {
        let strength = CorrelationStrength.from(correlation: correlation)
        let direction = correlation > 0 ? "positive" : "negative"
        
        return "There is a \(strength.rawValue.lowercased()) \(direction) correlation between \(factor1.lowercased()) and \(factor2.lowercased())."
    }
    
    private func calculateConfidence(correlation: Double, sampleSize: Int) -> Double {
        let absCorr = abs(correlation)
        let sizeBonus = min(1.0, Double(sampleSize) / 30.0)
        return absCorr * sizeBonus
    }
    
    private func calculateEffectConfidence(withCount: Int, withoutCount: Int, effectSize: Double) -> Double {
        let minSampleSize = min(withCount, withoutCount)
        let sampleBonus = min(1.0, Double(minSampleSize) / 10.0)
        let effectBonus = min(1.0, effectSize / 3.0)
        return (sampleBonus + effectBonus) / 2.0
    }
}